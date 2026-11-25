"""Bit-reduction test runner (see README.md). Run from bit-reduction-test/, inside the repo's .venv:

    python -m bitred.run --prepare                                   # pre-registration (CPU, seconds)
    python -m bitred.run --folds 0 1 2 3 4 --device cuda --machine-name <m>
    python -m bitred.run --analyze
    --smoke: tiny CPU check written to results-smoke/; --resume continues a fold.

Per fold k (nothing is trained or fine-tuned; every model is the frozen TinyHAR with quantized weights):
  1. original, and the uniform probe: per-tensor 4/3/2-bit, per-channel 8/4/3/2-bit
  2. sensitivities on the calibration windows (training only): per tensor x bits {2, 3, 4, 8}, one tensor quantized
     at a time: LCI drop (ours, SHAP), output KL (ZeroQ-style); Hutchinson Hessian traces -> HAWQ-style
  3. allocation at the budgets 3bit / 2.5bit / 2.25bit (size of uniform per-channel 2-bit + 1, 1/2, 1/4 of the gap to
     uniform 3-bit) with the same exact knapsack for every method: xgq (ours), zeroq_kl, hawq
  Every quantized model is assessed on the inner-val windows (F1, LCI, worst-subject LCI).

Writes results/ (prefix bitred-tinyhar-fold<k>-<machine>): -models.csv, -per-subject-lci.csv, -sensitivity.csv,
-hessian-traces.csv, -allocations.csv, -config.json, logs/; --analyze writes the bitred-tinyhar-* summary tables
and bitred-tinyhar-decision.json; --prepare writes bitred-tinyhar-preregistration.json.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import BITRED_ROOT, RESULTS_DIR, RESULTS_SMOKE_DIR
from har import runlog as R

from .allocate import allocate, objective
from .context import DESIGN, NF_MACHINE, FoldContext, cfg, select_windows, window_hashes
from .quant import mean_bits, quant_tensors, quantize_model, size_bits, size_kb, tensor_info, uniform_bits
from .rules import (BIT_OPTIONS, BUDGET_FRACTION, BUDGETS, FOLDS, HUTCHINSON, METHODS, PROBE_BITS, RULES, SENS_BITS,
                    SMOKE_HUTCHINSON)
from .sensitivity import (ce_loss, check_omega, hawq_sensitivity, hutchinson_traces, kl_sensitivity, lci_pooled)

P = "bitred-tinyhar"
TABLES = ("models", "per-subject-lci", "sensitivity", "hessian-traces", "allocations")
QUESTION = ("Below 4 bits, does weight-only quantization of the frozen TinyHAR lose explanation consistency (LCI) "
            "faster than accuracy, and does a mixed-precision bit allocation driven by measured explanation "
            "sensitivity (XGQ) keep LCI higher than accuracy-driven allocations (ZeroQ-style KL, HAWQ-style Hessian) "
            "at the same storage size, without losing F1?")


def out_dir(smoke: bool) -> Path:
    return RESULTS_SMOKE_DIR if smoke else RESULTS_DIR


def prereg_path(smoke: bool) -> Path:
    return out_dir(smoke) / f"{P}-preregistration.json"


def bitred_sha256() -> str:
    h = hashlib.sha256()
    for p in sorted((BITRED_ROOT / "bitred").glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def design_record(smoke: bool) -> dict:
    return {"design": DESIGN, "smoke_overrides": {} if not smoke else {"windows": "context.SMOKE",
                                                                         "hutchinson": SMOKE_HUTCHINSON},
            "bit_options": BIT_OPTIONS, "sens_bits": SENS_BITS, "probe_bits": PROBE_BITS, "methods": METHODS,
            "budgets": BUDGETS, "budget_fraction_of_2to3bit_gap": BUDGET_FRACTION, "hutchinson": SMOKE_HUTCHINSON if smoke else HUTCHINSON, "folds": FOLDS,
            "granularity_mixed": "channel", "noise_floor_machine": NF_MACHINE, "rules": RULES}


# ------------------------------------------------------------------------------ prepare

def prepare(args) -> None:
    from har.data import load_harth_windows
    out = out_dir(args.smoke)
    p = prereg_path(args.smoke)
    if p.exists() and not args.smoke:
        raise FileExistsError(f"{p} exists: the pre-registration is written once (delete it only if no real run used it)")
    harth = load_harth_windows()
    folds = (0,) if args.smoke else FOLDS
    hashes = {str(k): window_hashes(select_windows(k, args.smoke, harth)) for k in folds}
    rec = {"question": QUESTION, **design_record(args.smoke), "window_hashes": hashes, "written_at": R.now(),
           **R.git_info(), "bitred_code_sha256": bitred_sha256(), "har_code_sha256": R.code_sha256(),
           "har70plus_loaded": False}
    R.write_json(p, rec)
    for k, h in hashes.items():
        print(f"fold {k}: explain {h['n_explain']}, F1 {h['n_f1']}, calibration {h['n_calibration']}, "
              f"LCI-calibration {h['n_lci_calibration']}; inner val {h['inner_val_subjects']}")
    print(f"saved {p}")


def check_prereg(smoke: bool, k: int, hashes: dict) -> dict:
    p = prereg_path(smoke)
    if not p.exists():
        if smoke:
            return {}
        raise FileNotFoundError(f"{p} missing: run `python -m bitred.run --prepare` first")
    pre = json.loads(p.read_text(encoding="utf-8"))
    if pre["rules"] != RULES:
        raise RuntimeError("the rules changed after the pre-registration")
    want = pre["window_hashes"].get(str(k))
    if want is None:
        raise RuntimeError(f"fold {k} is not in the pre-registration")
    if want != json.loads(json.dumps(hashes)):
        raise RuntimeError(f"fold {k}: window choices differ from the pre-registration")
    return pre


# ------------------------------------------------------------------------------ store

class Store:
    def __init__(self, out: Path, k: int, machine: str, resume: bool):
        self.base = f"{P}-fold{k}-{machine}"
        self.paths = {t: out / f"{self.base}-{t}.csv" for t in TABLES}
        self.config = out / f"{self.base}-config.json"
        self.tables = {t: [] for t in TABLES}
        if resume:
            for t, p in self.paths.items():
                if p.exists() and p.stat().st_size > 2:
                    self.tables[t] = pd.read_csv(p, dtype={"subject": str}).to_dict("records")
        elif any(p.exists() for p in self.paths.values()):
            raise FileExistsError(f"{self.base} files exist; pass --resume to continue, or delete them")
        self.done = {r["model_id"] for r in self.tables["models"]}
        self.sens_done = {(r["tensor"], int(r["bits"])) for r in self.tables["sensitivity"]
                          if pd.notna(r.get("lci"))}

    def add(self, table: str, rows) -> None:
        self.tables[table] += list(rows)
        R.write_csv(self.paths[table], pd.DataFrame(self.tables[table]))

    def replace(self, table: str, rows) -> None:
        self.tables[table] = list(rows)
        R.write_csv(self.paths[table], pd.DataFrame(self.tables[table]))


# ------------------------------------------------------------------------------ one fold

def run_fold(args, k: int) -> None:
    out = out_dir(args.smoke)
    machine = args.machine_name
    store = Store(out, k, machine, args.resume)
    with R.tee_to(out / "logs" / f"{store.base}-{R.stamp()}.log"):
        t_fold = time.perf_counter()
        print(f"=== bit-reduction test, fold {k}, {machine}, {args.device}{' (smoke)' if args.smoke else ''} ===")
        ctx = FoldContext(k, args.device, args.smoke, args.noise_floor_machine)
        pre = check_prereg(args.smoke, k, ctx.hashes)
        net0 = ctx.net0
        info = tensor_info(net0)
        base = {"fold": k, "machine_name": machine, "device": args.device, "tau_k": ctx.tau, "ceiling": ctx.ceiling}

        def assess(model_id: str, family: str, bits: dict, granularity: str, extra: dict | None = None):
            if model_id in store.done:
                print(f"  {model_id}: done (resume)")
                return
            net = net0 if family == "original" else quantize_model(net0, bits, granularity)
            met, rows = ctx.assess(net, model_id)
            sb = size_bits(net0, bits, granularity)
            rec = {**base, "model_id": model_id, "family": family, "granularity": granularity,
                   "size_bits": sb, "size_kb": sb / 8 / 1024, "mean_bits": mean_bits(net0, bits),
                   "bits_json": json.dumps({n: int(bits.get(n, 32)) for n in info}), **(extra or {}), **met}
            store.add("per-subject-lci", [{**base, "model_id": model_id, **r} for r in rows])
            store.add("models", [rec])
            store.done.add(model_id)
            print(f"  {model_id}: {rec['size_kb']:.2f} KB, mean {rec['mean_bits']:.2f} bits; LCI {met['lci']:.4f} "
                  f"(worst {met['worst_subject_lci']:.4f}), F1 {met['f1']:.4f}, SHAP {met['shap_seconds']:.1f} s")

        # 1. original and uniform probe
        print("-- 1. original and uniform probe")
        assess("original", "original", {}, "channel")
        for gran in ("tensor", "channel"):
            for b in PROBE_BITS:
                if b == 8 and gran == "tensor":
                    continue
                assess(f"uniform-{gran}-{b}bit", f"uniform_{gran}", uniform_bits(net0, b), gran, {"bits_uniform": b})

        # 2. sensitivities (calibration windows = training windows only)
        print("-- 2. sensitivities on the calibration windows")
        t0 = time.perf_counter()
        kl = kl_sensitivity(net0, ctx.X_cal, SENS_BITS, "channel")
        t_kl = time.perf_counter() - t0
        hcfg = SMOKE_HUTCHINSON if args.smoke else HUTCHINSON
        names = list(info)
        if store.tables["hessian-traces"]:
            traces = {r["tensor"]: r for r in store.tables["hessian-traces"]}
            t_tr = float(store.tables["hessian-traces"][0].get("seconds", np.nan))
            print("  Hessian traces: done (resume)")
        else:
            t0 = time.perf_counter()
            torch.manual_seed(hcfg["seed"])
            traces = hutchinson_traces(net0, ce_loss, ctx.X_cal, ctx.y_cal, names, hcfg["n_iter"], hcfg["seed"])
            t_tr = time.perf_counter() - t0
            store.replace("hessian-traces", [{"fold": k, "tensor": n, "numel": info[n]["numel"], **traces[n],
                                              "trace_per_weight": traces[n]["trace"] / info[n]["numel"],
                                              "seconds": t_tr} for n in names])
            print(f"  Hessian traces ({hcfg['n_iter']} Hutchinson iterations): {t_tr:.1f} s")
        hawq = hawq_sensitivity(net0, traces, SENS_BITS, "channel")
        sens_rows = {(r["tensor"], int(r["bits"])): r for r in store.tables["sensitivity"]}
        t0 = time.perf_counter()
        for n in names:
            for b in SENS_BITS:
                if (n, b) in store.sens_done:
                    continue
                net = quantize_model(net0, {n: b}, "channel")
                A = ctx.shap_cal(net, f"single-{n}-{b}bit")
                r = lci_pooled(ctx.A_ref_lcal, A, k=DESIGN["k"], omega=DESIGN["omega"])
                sens_rows[(n, b)] = {"fold": k, "tensor": n, "bits": b, "numel": info[n]["numel"],
                                     "rows": info[n]["rows"], "lci": r["lci"], "lci_rho": r["shap_corr"],
                                     "lci_jaccard": r["jaccard"], "omega_xgq": 1.0 - r["lci"],
                                     "omega_zeroq_kl": kl[n][b], "omega_hawq": hawq[n][b],
                                     "trace": traces[n]["trace"]}
                store.sens_done.add((n, b))
                store.replace("sensitivity", list(sens_rows.values()))
                print(f"  {n} @ {b} bit: LCI_cal {r['lci']:.4f}, KL {kl[n][b]:.5f}, HAWQ {hawq[n][b]:.3e}")
        # KL and HAWQ are recomputed each run (deterministic, seconds); refresh them in resumed rows
        for (n, b), r in sens_rows.items():
            r["omega_zeroq_kl"], r["omega_hawq"], r["trace"] = kl[n][b], hawq[n][b], traces[n]["trace"]
        store.replace("sensitivity", list(sens_rows.values()))
        t_lci = time.perf_counter() - t0
        omegas = {m: {n: {b: float(sens_rows[(n, b)][f"omega_{m}"]) for b in SENS_BITS} for n in names}
                  for m in METHODS}
        for m, om in omegas.items():
            check_omega(om, names, BIT_OPTIONS)

        # 3. allocations at the budgets, same knapsack for every method
        print("-- 3. allocations")
        b3, b2 = size_bits(net0, uniform_bits(net0, 3)), size_bits(net0, uniform_bits(net0, 2))
        budgets = {B: int(b2 + BUDGET_FRACTION[B] * (b3 - b2)) for B in BUDGETS}
        alloc_rows = [r for r in store.tables["allocations"]]
        have = {(r["method"], r["budget"]) for r in alloc_rows}
        for B in BUDGETS:
            for m in METHODS:
                bits = allocate(net0, omegas[m], budgets[B], "channel", BIT_OPTIONS)
                if bits is None:
                    raise RuntimeError(f"budget {B} infeasible for {m}")
                assert size_bits(net0, bits, "channel") <= budgets[B]
                if (m, B) not in have:
                    alloc_rows += [{"fold": k, "method": m, "budget": B, "budget_bits": budgets[B], "tensor": n,
                                    "numel": info[n]["numel"], "bits": bits[n]} for n in names]
                    store.replace("allocations", alloc_rows)
                pred = {f"objective_{o}": objective(omegas[o], bits) for o in METHODS}
                assess(f"alloc-{m}-{B}", "allocation", bits, "channel",
                       {"method": m, "budget": B, "budget_bits": budgets[B], **pred})

        cfgrec = {"command": " ".join(sys.argv), "fold": k, "machine_name": machine, **R.device_info(args.device),
                  **R.git_info(), "bitred_code_sha256": bitred_sha256(), "har_code_sha256": R.code_sha256(),
                  "preregistration_written_at": pre.get("written_at"), **design_record(args.smoke),
                  "window_hashes": ctx.hashes, "calibration_matches_round4_table": ctx.cal_matches_round4,
                  "checkpoint": str(ctx.ckpt), "tau_k": ctx.tau, "tau_source": ctx.tau_src, "ceiling": ctx.ceiling,
                  "ceiling_source": ctx.ceiling_src, "budgets_bits": budgets,
                  "budgets_kb": {B: v / 8 / 1024 for B, v in budgets.items()},
                  "tensors": info, "seconds": {"shap_original": ctx.shap_original_seconds, "kl_sensitivity": t_kl,
                                               "hessian_traces": t_tr, "lci_sensitivity_loop": t_lci,
                                               "fold_wall": time.perf_counter() - t_fold},
                  "har70plus_loaded": False, "finished_at": R.now()}
        R.write_json(store.config, cfgrec)
        print(f"fold {k} done in {time.perf_counter() - t_fold:.1f} s; saved {store.base}-*")


# ------------------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--folds", type=int, nargs="+")
    mode.add_argument("--analyze", action="store_true")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--machine-name", default=R.machine_name())
    ap.add_argument("--noise-floor-machine", default=None,
                    help="whose noise-floor files give tau_k (default: folds 0-1 nakhoul-3050, 2-4 uni-2080)")
    ap.add_argument("--use-machine", nargs="*", default=None, help="--analyze: k=machine for a fold run on two machines")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.prepare:
        return prepare(args)
    if args.analyze:
        from .analyze import analyze
        return analyze(args)
    if args.smoke and args.folds != [0]:
        raise SystemExit("--smoke runs fold 0 only")
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = False
    for k in args.folds:
        run_fold(args, k)


if __name__ == "__main__":
    main()
