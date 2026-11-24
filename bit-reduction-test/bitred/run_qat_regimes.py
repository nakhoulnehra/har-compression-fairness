"""Data-regime gap test (see README.md, "Data-regime gap test"): the ultra-small gap test (run_qat_small.py) and the
2-bit gap test (run_qat.py) repeated with less QAT data. Same recipe, scoring and rule; only the QAT training set
changes.

    python -m bitred.run_qat_regimes --fold 0 --device cuda --machine-name <m>      (from bit-reduction-test/)
    --smoke: tiny CPU check written to results-smoke/

Fold 0, HARTH only. Regimes (QAT training windows, all from the fold's HARTH training subjects; teacher = the
original model's logits on those windows):

    kd5          round 4's 5 % subset: per (training subject, class) max(1, round(0.05 n)) windows, seed 48
                 (har.run_xgp_pilot4.kd5_table; checked against round 4's committed id table), ~2,580 windows
    calibration  round 4's calibration set: <= 10 windows per (subject, class), seed 46 (the same windows that
                 lamp_recon's reconstruction uses; checked against round 4's committed table), ~980 windows

Settings (sizes include the bitmask of every pruned tensor): channel-2bit 20.7 KB, tensor-2bit 17.5 KB,
lamp_recon90-channel-4bit 16.9 KB, lamp_recon95-channel-4bit 15.6 KB. lamp_recon starts are built exactly as in
run_qat_small (round 4's functions and values); masks fixed during QAT. The no-QAT references of every setting are
assessed once (regime-independent).

QAT: equal optimisation, not equal epochs. Every regime gets the full-data run's number of Adam steps,
20 epochs x steps_per_epoch(all training windows, batch 256) (4,020 on fold 0), i.e. many more epochs on the small
sets (kd5 ≈ 365, calibration 1,005); batch 256, seed 0, lr 1e-4 and 1e-3, the model after the last step. Validation
F1 is logged every steps_per_epoch(all training windows) steps (20 points, as in the full-data runs).

Rule (fixed before the run): the 2-bit gap test's rule, applied per setting x regime over its two lrs.

Writes results/bitred-qat-regimes-tinyhar-fold<k>-<machine>-{models, per-subject-lci, finetune-history,
lamp-keep-counts, regime-train-window-ids}.csv, -decision.json, -config.json and a log.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
import sys
import time

import numpy as np
import pandas as pd
import torch

from . import RESULTS_DIR, RESULTS_SMOKE_DIR
from har import runlog as R
from har.compression import head_unpruned, predict_logits, prunable_tensors
from har.data import find_windows, require_harth, window_ids
from har.run_xgp_pilot4 import kd5_table
from har.xgp4 import ROUND4, uniform_keep_counts

from .context import ROUND4_DIR, FoldContext
from .qat import qat_kd_finetune, steps_per_epoch
from .quant import mean_bits, pruned_size_record, quantize_model, uniform_bits
from .run import bitred_sha256
from .run_qat import GAP, GAP_RULE, decide
from .run_qat_small import LAMP, QAT, SMOKE_QAT, check_masks, lamp_recon_start

P = "bitred-qat-regimes-tinyhar"
SETTINGS = (
    {"setting": "channel-2bit", "prune_level": None, "bits": 2, "granularity": "channel"},
    {"setting": "tensor-2bit", "prune_level": None, "bits": 2, "granularity": "tensor"},
    {"setting": "lamp_recon90-channel-4bit", "prune_level": 0.9, "bits": 4, "granularity": "channel"},
    {"setting": "lamp_recon95-channel-4bit", "prune_level": 0.95, "bits": 4, "granularity": "channel"},
)
REGIMES = ("kd5", "calibration")
REGIME_SOURCE = {
    "kd5": f"round 4's kd5 subset (fraction {ROUND4['kd5_fraction']}, seed {ROUND4['kd5_seed']}, >= 1 per "
           f"(subject, class))",
    "calibration": f"round 4's calibration set (<= {ROUND4['cal_per_cell']} per (subject, class), seed "
                   f"{ROUND4['cal_seed']})",
}
RULE = "Applied per setting x regime (over the two lrs): " + GAP_RULE


def regime_indices(ctx: FoldContext, regime: str) -> np.ndarray:
    """Indices into ctx.tr of the regime's QAT training windows, checked against round 4's committed tables
    (outside smoke)."""
    if regime == "calibration":
        return np.asarray(ctx.windows["cal_idx"], dtype=np.int64)  # FoldContext already checked it vs round 4
    if regime != "kd5":
        raise ValueError(regime)
    ids = kd5_table(ctx.tr).window_id.tolist()
    committed = ROUND4_DIR / f"har-xgp-pilot4-tinyhar-fold{ctx.k}-kd5pct-train-window-ids.csv"
    if not ctx.smoke and committed.exists():              # round 4 committed folds 0-1; others are rebuilt
        saved = pd.read_csv(committed, dtype={"subject": str}).window_id.tolist()
        if saved != ids:
            raise RuntimeError(f"fold {ctx.k}: kd5 windows differ from round 4's committed {committed.name}")
    return find_windows(ctx.tr, "harth_train", ids)


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
        print(f"=== data-regime gap test (QAT), fold {args.fold}, {args.machine_name}, {args.device}"
              f"{' (smoke)' if args.smoke else ''} ===")
        ctx = FoldContext(args.fold, args.device, args.smoke, args.noise_floor_machine)
        net0 = ctx.net0
        common = {"fold": args.fold, "machine_name": args.machine_name, "device": args.device, "tau_k": ctx.tau,
                  "ceiling": ctx.ceiling}

        # ---- the regimes' training windows and teacher logits
        data, idrows = {}, []
        all_ids = window_ids(ctx.tr, "harth_train")
        for regime in REGIMES:
            idx = regime_indices(ctx, regime)
            tr = ctx.tr.select(idx)
            require_harth(tr)
            zt = torch.from_numpy(predict_logits(net0, tr.X, args.device)).to(args.device)
            data[regime] = (tr, zt)
            idrows += [{"fold": args.fold, "regime": regime, "window_id": w, "subject": s, "label": int(y)}
                       for w, s, y in zip(all_ids[idx], tr.subject, tr.y)]
            print(f"regime {regime}: {len(tr)} training windows ({len(tr) / len(ctx.tr):.2%} of {len(ctx.tr)}), "
                  f"{len(set(tr.subject.tolist()))} subjects; {REGIME_SOURCE[regime]}")
        R.write_csv(out / f"{base}-regime-train-window-ids.csv", pd.DataFrame(idrows))
        n_train = {r: len(d[0]) for r, d in data.items()}
        spe_full = steps_per_epoch(len(ctx.tr), qcfg["batch_size"])
        steps = qcfg["epochs"] * spe_full                       # the full-data run's step count
        epochs_eq = {r: steps / steps_per_epoch(n_train[r], qcfg["batch_size"]) for r in REGIMES}
        print(f"QAT: {steps} Adam steps per run (= {qcfg['epochs']} epochs x {spe_full} steps on all {len(ctx.tr)} "
              f"training windows); epochs: " + ", ".join(f"{r} {e:.1f}" for r, e in epochs_eq.items()))

        rows, prow, hist, krows, seconds = [], [], [], [], {"lamp_recon": {}}

        def record(mid, net, s, role, regime, lr, bits, masks, ft):
            met, sub = ctx.assess(net, mid)
            size = pruned_size_record(net0, bits, masks, s["granularity"])
            rows.append({**common, "model_id": mid, "setting": s["setting"], "role": role, "regime": regime,
                         "n_train_windows": n_train.get(regime), "lr": lr,
                         "adam_steps": steps if role == "qat" else None,
                         "epochs_equivalent": epochs_eq.get(regime) if role == "qat" else None, "prune_level": s["prune_level"],
                         "bits": s["bits"] if bits else None, "granularity": s["granularity"] if bits else None,
                         **size, "mean_bits": mean_bits(net0, bits) if bits else 32.0, "finetune_seconds": ft,
                         "head_pruned": any(n.endswith("head.weight") for n in masks), **met})
            prow.extend({**common, "model_id": mid, "setting": s["setting"], "regime": regime, **r} for r in sub)
            print(f"  {mid}: {size['size_kb']:.2f} KB, LCI {met['lci']:.4f} (worst {met['worst_subject_lci']:.4f}), "
                  f"F1 {met['f1']:.4f}" + (f"; fine-tuning {ft:.1f} s" if ft else ""))
            R.write_csv(out / f"{base}-models.csv", pd.DataFrame(rows))
            R.write_csv(out / f"{base}-per-subject-lci.csv", pd.DataFrame(prow))
            return met

        orig = {"setting": "original", "prune_level": None, "bits": None, "granularity": "channel"}
        pred0 = record("original", net0, orig, "original", None, None, {}, {}, 0.0)
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
                if not head_unpruned(start):
                    raise RuntimeError("lamp_recon pruned the head")
                record(f"{s['setting'].split('-')[0]}-fp32", start, s, "reference (no QAT)", None, None, {}, masks, 0.0)
            ptq_id = s["setting"].replace("-channel", "-ptq-channel") if masks else f"ptq-{s['setting']}"
            ptq = quantize_model(start, bits, s["granularity"])
            if masks:
                check_masks(ptq, masks, keep)
            record(ptq_id, ptq, s, "reference (no QAT)", None, None, bits, masks, 0.0)
            for regime in REGIMES:
                tr, zt = data[regime]
                for lr in qcfg["lrs"]:
                    mid = f"qat-{s['setting']}-{regime}-lr{lr:g}"
                    print(f"-- {mid} ({len(tr)} windows)")
                    net = copy.deepcopy(start)
                    t0 = time.perf_counter()
                    qnet, h = qat_kd_finetune(net, tr, ctx.va, zt, bits, granularity=s["granularity"],
                                              lr=lr, batch_size=qcfg["batch_size"], seed=qcfg["seed"],
                                              device=args.device, steps=steps, eval_every_steps=spe_full)
                    if h[-1]["step"] != steps:
                        raise RuntimeError(f"{mid}: {h[-1]['step']} steps, expected {steps}")
                    ft = time.perf_counter() - t0
                    if masks:
                        check_masks(qnet, masks, keep)
                    hist += [{**common, "model_id": mid, "setting": s["setting"], "regime": regime, "lr": lr, **r}
                             for r in h]
                    R.write_csv(out / f"{base}-finetune-history.csv", pd.DataFrame(hist))
                    record(mid, qnet, s, "qat", regime, lr, bits, masks, ft)

        df = pd.DataFrame(rows)
        per, verdicts = {}, {}
        for s in SETTINGS:
            for regime in REGIMES:
                q = df[(df.setting == s["setting"]) & (df.regime == regime) & (df.role == "qat")]
                d = decide(q.to_dict("records"), pred0["f1"])
                d["verdict"] = d["verdict"].replace("2 bits is an accuracy problem first",
                                                    f"{s['setting']} with {regime} data is an accuracy problem first")
                key = f"{s['setting']}|{regime}"
                per[key] = {"setting": s["setting"], "regime": regime, "n_train_windows": n_train[regime],
                            "size_kb": float(q.size_kb.iloc[0]),
                            **{k: v for k, v in d.items() if k not in ("rule", *GAP)}}
                verdicts[key] = d["verdict"].split(":")[0]
        dec = {"per_setting_regime": per, "verdict_by_setting_regime": verdicts,
               "with_gap": [k for k, v in verdicts.items() if v == "gap"],
               "f1_original": pred0["f1"], "rule": RULE, **GAP}
        R.write_json(out / f"{base}-decision.json", dec)
        seconds["wall"] = time.perf_counter() - t_all
        R.write_json(out / f"{base}-config.json",
                     {"command": " ".join(sys.argv), **R.device_info(args.device), **R.git_info(),
                      "bitred_code_sha256": bitred_sha256(), "har_code_sha256": R.code_sha256(),
                      "settings": list(SETTINGS), "regimes": {r: {"source": REGIME_SOURCE[r],
                                                                   "n_train_windows": n_train[r]} for r in REGIMES},
                      "n_train_full": len(ctx.tr), "qat": qcfg,
                      "optimisation": {"adam_steps_per_run": steps, "steps_per_epoch_full": spe_full,
                                       "eval_every_steps": spe_full, "epochs_equivalent": epochs_eq,
                                       "note": "equal Adam steps across regimes (changed 2026-11-25 before any real "
                                               "run, from equal epochs)"}, "lamp_recon": LAMP, "gap": GAP, "rule": RULE,
                      "window_hashes": ctx.hashes, "calibration_matches_round4": ctx.cal_matches_round4,
                      "checkpoint": str(ctx.ckpt), "tau_k": ctx.tau, "ceiling": ctx.ceiling, "seconds": seconds,
                      "har70plus_loaded": False, "finished_at": R.now()})
        cols = ["model_id", "size_kb", "f1", "lci", "worst_subject_lci"]
        print("\n" + df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"\nverdicts: {verdicts}; wall {seconds['wall']:.1f} s")


if __name__ == "__main__":
    main()
