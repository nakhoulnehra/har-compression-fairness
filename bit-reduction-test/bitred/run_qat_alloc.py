"""Like-for-like test (see README.md): every mixed-precision allocation gets the same quantization-aware KD fine-tuning
as the uniform 2-bit model, so XGQ + KD is compared with ZeroQ-KL + KD, HAWQ + KD and uniform 2-bit + KD.

    python -m bitred.run_qat_alloc --fold 0 --device cuda --machine-name <m>      (from bit-reduction-test/)
    --smoke: tiny CPU check written to results-smoke/ (needs the smoke allocations)

Allocations: read from the main test's bitred-tinyhar-fold<k>-<machine>-allocations.csv (budgets 2.5bit = 23.9 KB and
2.25bit = 22.3 KB; the 3bit budget is left out, every model there is already at the SHAP-noise ceiling without training).
Recipe = run_qat.py's lr 1e-4 run: QAT with the shared KD recipe, 20 epochs, seed 0, per-channel scales. Reference:
uniform 2-bit + the same recipe (20.7 KB), rerun here so every model comes from the same code and session. Writes
results/bitred-qatalloc-tinyhar-fold<k>-<machine>-{models, per-subject-lci, finetune-history}.csv, -decision.json,
-config.json and a log.
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
from har.compression import predict_logits

from .context import FoldContext
from .qat import qat_kd_finetune
from .quant import mean_bits, size_bits, tensor_info, uniform_bits
from .run import bitred_sha256

P = "bitred-qatalloc-tinyhar"
QAT = {"granularity": "channel", "epochs": 20, "lr": 1e-4, "seed": 0, "batch_size": 256}
SMOKE_QAT = {"epochs": 1}
BUDGETS = ("2.5bit", "2.25bit")
METHODS = ("xgq", "zeroq_kl", "hawq")
MARGIN = {"lci": 0.03, "f1_tol": 0.01}
RULE = ("Fold-0 screen, fixed before the run. At a budget B, XGQ + KD 'wins like-for-like' iff its LCI_val >= "
        "max(ZeroQ-KL + KD, HAWQ + KD, uniform 2-bit + KD) + 0.03 and its F1_val >= each of theirs - 0.01. The 0.03 "
        "margin = about one top-10 Jaccard step (~0.09 for one subject) averaged over the 3 inner-val subjects; smaller "
        "differences are within the metric's resolution. Uniform 2-bit + KD is smaller (20.7 KB), so XGQ + KD must "
        "beat it to show that mixed precision is worth its extra size. One fold and one seed: a win is a reason to run "
        "folds 1-4, not evidence; no win = no method contribution here.")


def decide(rows: list[dict]) -> dict:
    by = {r["model_id"]: r for r in rows}
    ref = by["uniform-2bit-kd"]
    out = {}
    for B in BUDGETS:
        ours = by[f"xgq-{B}-kd"]
        comps = [by[f"zeroq_kl-{B}-kd"], by[f"hawq-{B}-kd"], ref]
        best = max(comps, key=lambda r: r["lci"])
        win = (ours["lci"] >= best["lci"] + MARGIN["lci"] and all(ours["f1"] >= c["f1"] - MARGIN["f1_tol"] for c in comps))
        out[B] = {"xgq_kd": {k: ours[k] for k in ("lci", "f1", "worst_subject_lci", "size_kb")},
                  "best_comparator": best["model_id"], "best_comparator_lci": best["lci"],
                  "d_lci_vs_best": ours["lci"] - best["lci"], "wins": bool(win)}
    return {"by_budget": out, "any_win": any(v["wins"] for v in out.values()), "rule": RULE, **MARGIN}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--machine-name", default=R.machine_name())
    ap.add_argument("--alloc-machine", default=None, help="machine whose allocations file to use (default: --machine-name)")
    ap.add_argument("--noise-floor-machine", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = RESULTS_SMOKE_DIR if args.smoke else RESULTS_DIR
    base = f"{P}-fold{args.fold}-{args.machine_name}"
    if (out / f"{base}-models.csv").exists() and not args.smoke:
        raise FileExistsError(f"{base} exists; delete it to rerun")
    am = args.alloc_machine or args.machine_name
    alloc_p = out / f"bitred-tinyhar-fold{args.fold}-{am}-allocations.csv"
    alloc = pd.read_csv(alloc_p)
    cfg = {**QAT, **(SMOKE_QAT if args.smoke else {})}
    with R.tee_to(out / "logs" / f"{base}-{R.stamp()}.log"):
        t_all = time.perf_counter()
        print(f"=== like-for-like QAT test, fold {args.fold}, {args.machine_name}, {args.device}"
              f"{' (smoke)' if args.smoke else ''}; allocations from {alloc_p.name} ===")
        ctx = FoldContext(args.fold, args.device, args.smoke, args.noise_floor_machine)
        net0, info = ctx.net0, tensor_info(ctx.net0)
        zt = torch.from_numpy(predict_logits(net0, ctx.tr.X, args.device)).to(args.device)
        plan = [("uniform-2bit-kd", "uniform", None, uniform_bits(net0, 2))]
        for B in BUDGETS:
            for m in METHODS:
                a = alloc[(alloc.method == m) & (alloc.budget == B)]
                bits = {r.tensor: int(r.bits) for r in a.itertuples()}
                if set(bits) != set(info):
                    raise RuntimeError(f"allocation {m} @ {B} does not cover every tensor")
                plan.append((f"{m}-{B}-kd", m, B, bits))
        common = {"fold": args.fold, "machine_name": args.machine_name, "device": args.device, "tau_k": ctx.tau,
                  "ceiling": ctx.ceiling}
        rows, prow, hist = [], [], []
        for mid, method, B, bits in plan:
            print(f"-- {mid}: {size_bits(net0, bits) / 8 / 1024:.2f} KB, mean {mean_bits(net0, bits):.2f} bits")
            t0 = time.perf_counter()
            qnet, h = qat_kd_finetune(copy.deepcopy(net0), ctx.tr, ctx.va, zt, bits, granularity=cfg["granularity"],
                                      epochs=cfg["epochs"], lr=cfg["lr"], batch_size=cfg["batch_size"],
                                      seed=cfg["seed"], device=args.device)
            ft = time.perf_counter() - t0
            met, sub = ctx.assess(qnet, f"qatalloc-{mid}-seed{cfg['seed']}")
            rows.append({**common, "model_id": mid, "method": method, "budget": B,
                         "size_kb": size_bits(net0, bits) / 8 / 1024, "mean_bits": mean_bits(net0, bits),
                         "finetune_seconds": ft, **met})
            prow += [{**common, "model_id": mid, **r} for r in sub]
            hist += [{**common, "model_id": mid, **r} for r in h]
            print(f"  {mid}: LCI {met['lci']:.4f} (worst {met['worst_subject_lci']:.4f}), F1 {met['f1']:.4f}")
            R.write_csv(out / f"{base}-models.csv", pd.DataFrame(rows))
            R.write_csv(out / f"{base}-per-subject-lci.csv", pd.DataFrame(prow))
            R.write_csv(out / f"{base}-finetune-history.csv", pd.DataFrame(hist))
        dec = decide(rows)
        R.write_json(out / f"{base}-decision.json", dec)
        R.write_json(out / f"{base}-config.json",
                     {"command": " ".join(sys.argv), **R.device_info(args.device), **R.git_info(),
                      "bitred_code_sha256": bitred_sha256(), "har_code_sha256": R.code_sha256(), "qat": cfg,
                      "allocations_file": alloc_p.name, "rule": RULE, "window_hashes": ctx.hashes,
                      "seconds": {"wall": time.perf_counter() - t_all}, "har70plus_loaded": False,
                      "finished_at": R.now()})
        for B, v in dec["by_budget"].items():
            print(f"{B}: XGQ+KD LCI {v['xgq_kd']['lci']:.4f} F1 {v['xgq_kd']['f1']:.4f}; best comparator "
                  f"{v['best_comparator']} {v['best_comparator_lci']:.4f}; wins {v['wins']}")
        print(f"any win: {dec['any_win']}; wall {time.perf_counter() - t_all:.1f} s")


if __name__ == "__main__":
    main()
