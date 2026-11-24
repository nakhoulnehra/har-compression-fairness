"""2-bit gap test (see README.md, "2-bit gap test"): after standard quantization-aware KD fine-tuning at 2 bits, does
accuracy recover while explanation consistency stays low? Only then is there an explanation-specific gap for a new
method to fill.

    python -m bitred.run_qat --fold 0 --device cuda --machine-name <m>      (from bit-reduction-test/)
    --smoke: tiny CPU check written to results-smoke/

Fold 0, HARTH only. Start = the frozen TinyHAR of the fold; every quantizable tensor at 2 bits per-channel (20.7 KB);
QAT with the shared KD recipe, 20 epochs, seed 0, at two learning rates fixed in advance: 1e-4 (the project's
fine-tuning lr) and 1e-3 (TinyHAR's training lr). Assessed like every bit-reduction model (inner-val F1, LCI on the
300 explain windows, same SHAP). Writes results/bitred-qat2bit-tinyhar-fold<k>-<machine>-{models, per-subject-lci,
finetune-history}.csv, -decision.json, -config.json and a log.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
import json
import sys
import time

import numpy as np
import pandas as pd
import torch

from . import RESULTS_DIR, RESULTS_SMOKE_DIR
from har import runlog as R
from har.compression import predict_logits

from .context import FoldContext
from .qat import qat_kd_finetune
from .quant import mean_bits, size_bits, tensor_info, uniform_bits
from .run import bitred_sha256

P = "bitred-qat2bit-tinyhar"
QAT = {"bits": 2, "granularity": "channel", "epochs": 20, "lrs": (1e-4, 1e-3), "seed": 0, "batch_size": 256}
SMOKE_QAT = {"epochs": 1}
GAP = {"f1_recovered_tol": 0.03, "lci_gap_threshold": 0.85}
GAP_RULE = ("Recovered = F1_val >= F1_val(original) - 0.03. Among the QAT runs that recover F1, take the highest "
            "LCI_val. Gap (explanation-specific, room for a new method) iff that LCI_val <= 0.85 (uniform per-channel "
            "4-bit PTQ is 0.90 on fold 0, so <= 0.85 = at least 0.05 below it). No gap iff it is > 0.85 (a standard "
            "recipe fixes 2 bits). If no run recovers F1: inconclusive for explanations, 2 bits is an accuracy problem "
            "first. Fixed before the run; nothing is retuned.")


def decide(rows: list[dict], f1_original: float) -> dict:
    rec = [r for r in rows if r["f1"] >= f1_original - GAP["f1_recovered_tol"]]
    if not rec:
        verdict = "inconclusive: no QAT run recovers F1 (2 bits is an accuracy problem first)"
        best = None
    else:
        best = max(rec, key=lambda r: r["lci"])
        verdict = ("gap: F1 recovers but LCI stays <= 0.85" if best["lci"] <= GAP["lci_gap_threshold"]
                   else "no gap: a standard QAT recipe recovers both F1 and LCI")
    return {"verdict": verdict, "f1_original": f1_original, "recovered_runs": [r["model_id"] for r in rec],
            "best_recovered": None if best is None else {k: best[k] for k in ("model_id", "f1", "lci", "worst_subject_lci")},
            "rule": GAP_RULE, **GAP}


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
    cfg = {**QAT, **(SMOKE_QAT if args.smoke else {})}
    with R.tee_to(out / "logs" / f"{base}-{R.stamp()}.log"):
        t_all = time.perf_counter()
        print(f"=== 2-bit gap test (QAT), fold {args.fold}, {args.machine_name}, {args.device}"
              f"{' (smoke)' if args.smoke else ''} ===")
        ctx = FoldContext(args.fold, args.device, args.smoke, args.noise_floor_machine)
        net0, info = ctx.net0, tensor_info(ctx.net0)
        bits = uniform_bits(net0, cfg["bits"])
        t0 = time.perf_counter()
        zt = torch.from_numpy(predict_logits(net0, ctx.tr.X, args.device)).to(args.device)
        t_logits = time.perf_counter() - t0
        pred0 = ctx.assess(net0, "original")[0]
        rows, prow, hist = [], [], []
        common = {"fold": args.fold, "machine_name": args.machine_name, "device": args.device, "tau_k": ctx.tau,
                  "ceiling": ctx.ceiling}
        rows.append({**common, "model_id": "original", "lr": None, "size_kb": size_bits(net0, {}) / 8 / 1024,
                     "mean_bits": 32.0, "finetune_seconds": 0.0, **pred0})
        for lr in cfg["lrs"]:
            mid = f"qat-channel-{cfg['bits']}bit-lr{lr:g}"
            print(f"-- {mid}")
            net = copy.deepcopy(net0)
            t0 = time.perf_counter()
            qnet, h = qat_kd_finetune(net, ctx.tr, ctx.va, zt, bits, granularity=cfg["granularity"],
                                      epochs=cfg["epochs"], lr=lr, batch_size=cfg["batch_size"], seed=cfg["seed"],
                                      device=args.device)
            ft = time.perf_counter() - t0
            met, sub = ctx.assess(qnet, f"{mid}-seed{cfg['seed']}")
            rows.append({**common, "model_id": mid, "lr": lr, "size_kb": size_bits(net0, bits) / 8 / 1024,
                         "mean_bits": mean_bits(net0, bits), "finetune_seconds": ft, **met})
            prow += [{**common, "model_id": mid, **r} for r in sub]
            hist += [{**common, "model_id": mid, "lr": lr, **r} for r in h]
            print(f"  {mid}: LCI {met['lci']:.4f} (worst {met['worst_subject_lci']:.4f}), F1 {met['f1']:.4f}; "
                  f"fine-tuning {ft:.1f} s")
            R.write_csv(out / f"{base}-models.csv", pd.DataFrame(rows))
            R.write_csv(out / f"{base}-per-subject-lci.csv", pd.DataFrame(prow))
            R.write_csv(out / f"{base}-finetune-history.csv", pd.DataFrame(hist))
        dec = decide([r for r in rows if r["model_id"] != "original"], pred0["f1"])
        R.write_json(out / f"{base}-decision.json", dec)
        R.write_json(out / f"{base}-config.json",
                     {"command": " ".join(sys.argv), **R.device_info(args.device), **R.git_info(),
                      "bitred_code_sha256": bitred_sha256(), "har_code_sha256": R.code_sha256(), "qat": cfg,
                      "gap": GAP, "rule": GAP_RULE, "window_hashes": ctx.hashes, "checkpoint": str(ctx.ckpt),
                      "tau_k": ctx.tau, "ceiling": ctx.ceiling, "seconds": {"teacher_logits": t_logits,
                                                                           "wall": time.perf_counter() - t_all},
                      "har70plus_loaded": False, "finished_at": R.now()})
        print(f"\nverdict: {dec['verdict']}; best recovered: {dec['best_recovered']}; "
              f"wall {time.perf_counter() - t_all:.1f} s")


if __name__ == "__main__":
    main()
