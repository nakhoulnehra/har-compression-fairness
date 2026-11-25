"""Ultra-small gap test (see README.md, "Ultra-small gap test"): the 2-bit gap test (run_qat.py) extended below
20.7 KB, with the same recipe, scoring and rule. After standard quantization-aware KD fine-tuning, does F1 recover
while explanation consistency stays low?

    python -m bitred.run_qat_small --fold 0 --device cuda --machine-name <m>      (from bit-reduction-test/)
    --smoke: tiny CPU check written to results-smoke/

Fold 0 first, HARTH only. Settings (sizes include the 1-bit-per-position bitmask of every pruned tensor):

    tensor-2bit                 every quantizable tensor at 2 bits, one scale per tensor                  17.5 KB
    lamp_recon90-channel-4bit   lamp_recon at 90 % unstructured sparsity, then 4 bits per channel         16.9 KB
    lamp_recon95-channel-4bit   lamp_recon at 95 %, then 4 bits per channel                               15.6 KB

lamp_recon is round 4's baseline, reused from har/xgp4 and har/xgp3 unchanged: LAMP keep counts (Lee et al. 2021)
through bounded_budget (floor 2 %, uniform counts for the tensors the reconstruction cannot refit, total = the
uniform total), K = 5 nested magnitude steps (iterative_prune_budget, beta 0), and after each step the ridge
layer-wise reconstruction (layerwise_reconstruct, damp 0.01) on round 4's calibration training windows (<= 10 per
(subject, class), seed 46). The head is not pruned. No gradient training before QAT.

QAT: bitred.qat.qat_kd_finetune (STE, the shared KD recipe, Adam, batch 256), 20 epochs, seed 0, lr 1e-4 and 1e-3
(fixed in advance), last-epoch model; the pruning masks are fixed during QAT. Each setting also reports its
no-QAT references (the PTQ model, and the fp32 pruned start), which the rule does not use.

Rule (per setting, fixed before the run, as the 2-bit gap test): a QAT run recovers if F1_val >= F1_val(original)
- 0.03; gap if the best LCI_val among recovering runs is <= 0.85; no gap if > 0.85; inconclusive if none recovers.

Writes results/bitred-qat-ultrasmall-tinyhar-fold<k>-<machine>-{models, per-subject-lci, finetune-history,
lamp-keep-counts}.csv, -decision.json, -config.json and a log.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
import sys
import time

import pandas as pd
import torch

from . import RESULTS_DIR, RESULTS_SMOKE_DIR
from har import runlog as R
from har.compression import head_unpruned, predict_logits, prunable_tensors
from har.xgp3 import layerwise_reconstruct
from har.xgp4 import ROUND4, bounded_budget, iterative_prune_budget, uniform_keep_counts, unreconstructed_tensors

from .context import FoldContext
from .qat import qat_kd_finetune
from .quant import mean_bits, pruned_size_record, quantize_model, uniform_bits
from .run import bitred_sha256
from .run_qat import GAP, GAP_RULE, decide

P = "bitred-qat-ultrasmall-tinyhar"
SETTINGS = (
    {"setting": "tensor-2bit", "prune_level": None, "bits": 2, "granularity": "tensor"},
    {"setting": "lamp_recon90-channel-4bit", "prune_level": 0.9, "bits": 4, "granularity": "channel"},
    {"setting": "lamp_recon95-channel-4bit", "prune_level": 0.95, "bits": 4, "granularity": "channel"},
)
QAT = {"epochs": 20, "lrs": (1e-4, 1e-3), "seed": 0, "batch_size": 256}
LAMP = {"K": ROUND4["K"], "beta": 0.0, "damp": ROUND4["damp"], "floor_frac": ROUND4["floor_frac"], "budget": "lamp"}
SMOKE_QAT = {"epochs": 1}
RULE = "Applied per setting: " + GAP_RULE


def lamp_recon_start(net0: torch.nn.Module, level: float, X_cal: torch.Tensor,
                     log=print) -> tuple[torch.nn.Module, dict[str, int], list[str], float]:
    """Round 4's lamp_recon start: (pruned + reconstructed model with prune_masks, keep counts, the
    budget-protected tensors, seconds). net0 is not modified."""
    t0 = time.perf_counter()
    unrecon = unreconstructed_tensors(net0, X_cal[:4])
    keep = bounded_budget(net0, level, unrecon, LAMP["budget"])
    net, _ = iterative_prune_budget(net0, keep, K=LAMP["K"], beta=LAMP["beta"], importance_fn=None,
                                    recon_fn=lambda n: layerwise_reconstruct(n, net0, X_cal, damp=LAMP["damp"]),
                                    log=log)
    if not head_unpruned(net):
        raise RuntimeError("lamp_recon pruned the head")
    return net.eval(), keep, unrecon, time.perf_counter() - t0


def check_masks(net: torch.nn.Module, masks: dict[str, torch.Tensor], keep: dict[str, int]) -> None:
    params = dict(net.named_parameters())
    for n, m in masks.items():
        m = m.to(params[n].device)
        if torch.any(params[n].detach()[~m] != 0):
            raise RuntimeError(f"{n}: a pruned weight is non-zero")
        if int(m.sum()) != keep[n]:
            raise RuntimeError(f"{n}: mask keeps {int(m.sum())}, budget {keep[n]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--machine-name", default=R.machine_name())
    ap.add_argument("--noise-floor-machine", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = RESULTS_SMOKE_DIR if args.smoke else RESULTS_DIR
    base = f"{P}-fold{args.fold}-{args.machine_name}"
    if (out / f"{base}-models.csv").exists() and not args.smoke:
        raise FileExistsError(f"{base} exists; delete it to rerun")
    qcfg = {**QAT, **(SMOKE_QAT if args.smoke else {})}
    with R.tee_to(out / "logs" / f"{base}-{R.stamp()}.log"):
        t_all = time.perf_counter()
        print(f"=== ultra-small gap test (QAT), fold {args.fold}, {args.machine_name}, {args.device}"
              f"{' (smoke)' if args.smoke else ''} ===")
        ctx = FoldContext(args.fold, args.device, args.smoke, args.noise_floor_machine)
        net0 = ctx.net0
        t0 = time.perf_counter()
        zt = torch.from_numpy(predict_logits(net0, ctx.tr.X, args.device)).to(args.device)
        t_logits = time.perf_counter() - t0
        common = {"fold": args.fold, "machine_name": args.machine_name, "device": args.device, "tau_k": ctx.tau,
                  "ceiling": ctx.ceiling}
        rows, prow, hist, krows, seconds = [], [], [], [], {"teacher_logits": t_logits, "lamp_recon": {}}

        def record(mid, net, s, role, lr, bits, masks, ft):
            met, sub = ctx.assess(net, mid)
            size = pruned_size_record(net0, bits, masks, s["granularity"])
            rows.append({**common, "model_id": mid, "setting": s["setting"], "role": role, "lr": lr,
                         "prune_level": s["prune_level"], "bits": None if not bits else s["bits"],
                         "granularity": s["granularity"] if bits else None, **size,
                         "mean_bits": mean_bits(net0, bits) if bits else 32.0, "finetune_seconds": ft,
                         "head_pruned": any(n.endswith("head.weight") for n in masks), **met})
            prow.extend({**common, "model_id": mid, "setting": s["setting"], **r} for r in sub)
            print(f"  {mid}: {size['size_kb']:.2f} KB, LCI {met['lci']:.4f} (worst {met['worst_subject_lci']:.4f}), "
                  f"F1 {met['f1']:.4f}" + (f"; fine-tuning {ft:.1f} s" if ft else ""))
            R.write_csv(out / f"{base}-models.csv", pd.DataFrame(rows))
            R.write_csv(out / f"{base}-per-subject-lci.csv", pd.DataFrame(prow))
            return met

        orig = {"setting": "original", "prune_level": None, "bits": None, "granularity": "channel"}
        pred0 = record("original", net0, orig, "original", None, {}, {}, 0.0)
        for s in SETTINGS:
            print(f"-- {s['setting']}")
            bits = uniform_bits(net0, s["bits"])
            masks, keep = {}, None
            if s["prune_level"] is None:
                start = net0
            else:
                start, keep, unrecon, sec = lamp_recon_start(net0, s["prune_level"], ctx.X_cal)
                seconds["lamp_recon"][s["setting"]] = sec
                masks = {n: m.detach().cpu() for n, m in start.prune_masks.items()}
                check_masks(start, masks, keep)
                uni = uniform_keep_counts(net0, s["prune_level"])
                krows += [{**common, "setting": s["setting"], "prune_level": s["prune_level"], "tensor": n,
                           "n": int(getattr(m, p).numel()), "uniform_keep": uni[n], "lamp_recon_keep": keep[n],
                           "budget_protected": n in unrecon} for n, m, p in prunable_tensors(net0)]
                R.write_csv(out / f"{base}-lamp-keep-counts.csv", pd.DataFrame(krows))
                record(f"{s['setting'].split('-')[0]}-fp32", start, s, "reference (no QAT)", None, {}, masks, 0.0)
            ptq_id = s["setting"].replace("-channel", "-ptq-channel") if masks else f"ptq-{s['setting']}"
            ptq = quantize_model(start, bits, s["granularity"])
            if masks:
                check_masks(ptq, masks, keep)
            record(ptq_id, ptq, s, "reference (no QAT)", None, bits, masks, 0.0)
            for lr in qcfg["lrs"]:
                mid = f"qat-{s['setting']}-lr{lr:g}"
                print(f"-- {mid}")
                net = copy.deepcopy(start)
                t0 = time.perf_counter()
                qnet, h = qat_kd_finetune(net, ctx.tr, ctx.va, zt, bits, granularity=s["granularity"],
                                          epochs=qcfg["epochs"], lr=lr, batch_size=qcfg["batch_size"],
                                          seed=qcfg["seed"], device=args.device)
                ft = time.perf_counter() - t0
                if masks:
                    check_masks(qnet, masks, keep)
                hist += [{**common, "model_id": mid, "setting": s["setting"], "lr": lr, **r} for r in h]
                R.write_csv(out / f"{base}-finetune-history.csv", pd.DataFrame(hist))
                record(mid, qnet, s, "qat", lr, bits, masks, ft)

        df = pd.DataFrame(rows)
        per = {}
        for s in SETTINGS:
            q = df[(df.setting == s["setting"]) & (df.role == "qat")]
            d = decide(q.to_dict("records"), pred0["f1"])
            d["verdict"] = d["verdict"].replace("2 bits is an accuracy problem first",
                                                f"{s['setting']} is an accuracy problem first")
            d["size_kb"] = float(q.size_kb.iloc[0])
            per[s["setting"]] = {k: v for k, v in d.items() if k not in ("rule", *GAP)}
        verdicts = {k: v["verdict"].split(":")[0] for k, v in per.items()}
        dec = {"per_setting": per, "verdict_by_setting": verdicts,
               "settings_with_gap": [k for k, v in verdicts.items() if v == "gap"],
               "f1_original": pred0["f1"], "rule": RULE, **GAP}
        R.write_json(out / f"{base}-decision.json", dec)
        seconds["wall"] = time.perf_counter() - t_all
        R.write_json(out / f"{base}-config.json",
                     {"command": " ".join(sys.argv), **R.device_info(args.device), **R.git_info(),
                      "bitred_code_sha256": bitred_sha256(), "har_code_sha256": R.code_sha256(),
                      "settings": list(SETTINGS), "qat": qcfg, "lamp_recon": LAMP, "gap": GAP, "rule": RULE,
                      "window_hashes": ctx.hashes, "calibration_matches_round4": ctx.cal_matches_round4,
                      "checkpoint": str(ctx.ckpt), "tau_k": ctx.tau, "ceiling": ctx.ceiling, "seconds": seconds,
                      "har70plus_loaded": False, "finished_at": R.now()})
        cols = ["model_id", "size_kb", "f1", "lci", "worst_subject_lci"]
        print("\n" + df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"\nverdicts: {verdicts}; wall {seconds['wall']:.1f} s")


if __name__ == "__main__":
    main()
