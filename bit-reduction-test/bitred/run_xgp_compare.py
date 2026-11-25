"""Final comparison (see README.md, "Final comparison: XGP vs the best existing at ultra-small size"): in the
ultra-small pipeline (unstructured pruning, then 4-bit per-channel weights, then QAT + shared KD), does round 4's
XGP (xgp_xb: XGP budgets + XGP selection) beat the best existing baseline (lamp_recon) at the same size?

    python -m bitred.run_xgp_compare --folds 0 1 --device cuda --machine-name <m> [--resume]
    python -m bitred.run_xgp_compare --analyze [--use-machine k=<m> ...]
    --smoke: tiny CPU check written to results-smoke/

Folds 0-4, seed 0, HARTH only (inner-val scoring, as every bit-reduction test).

Starts (per fold, from the frozen TinyHAR; round 4's functions and values, har/xgp4.py and har/xgp3.py unchanged):
    lamp_recon  U_CELLS["lamp_recon"] = (lamp budget, l1 selection): LAMP keep counts through bounded_budget,
                K = 5 nested steps with beta 0 (magnitude), ridge reconstruction (damp 0.01) after every step
    xgp_xb      U_CELLS["xgp_xb"] = (xb budget, xgp selection): budget shares = worst_group_shares of the original
                model, through the same bounded_budget; K = 5 steps with beta = 0.1 and the worst-group EG
                importance (top-k 3, m = 8, EG seeds (0, 1) = round 4's seed 0) recomputed every step, the same
                reconstruction after every step
    Both use round 4's calibration set (<= 10 training windows per (subject, class), seed 46; groups = subjects)
    and the fold's 100 SHAP background windows as EG baselines. Levels 90 % and 95 %; same kept-weight total, so
    the same size. The head is never pruned.
Then every start is quantized to 4 bits per channel and trained with QAT + shared KD (bitred.qat, STE; masks fixed),
4,020 Adam steps at batch 256 (= 20 epochs of fold 0's full training set) in every regime, lr 1e-4 and 1e-3,
seed 0, the model after the last step. Reference: per-tensor 2-bit QAT of the unpruned model (17.5 KB), the same
regimes and lrs. No-QAT references (fp32 starts, 4-bit PTQ starts) are assessed once per start.

Regimes (QAT training windows): full (all training windows of the fold), kd5 (round 4's 5 % subset, seed 48),
calibration (round 4's calibration set).

Rule (R2, fixed before any run; README): per level x regime, at the primary lr, XGP beats LAMP iff over the 5 folds
(a) mean LCI_val gain >= 0.02, (b) LCI_val strictly higher in >= 4 of 5 folds, (c) mean F1_val >= LAMP's - 0.01,
(d) mean worst-subject LCI_val >= LAMP's - 0.01 (bitred.rules.compare). Primary lr per regime = the lr with the higher
mean F1_val of lamp_recon over the 5 folds and both levels (tie -> 1e-4); both lrs are reported.

Per fold: results/bitred-xgp-compare-tinyhar-fold<k>-<machine>-{models, per-subject-lci, finetune-history,
budgets}.csv, -config.json and a log. --analyze: -summary-mean-sd-over-folds.csv, -per-fold.csv,
-comparisons.csv, -primary-lr.csv and -decision.json.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
import glob
import sys
import time
from math import ceil

import numpy as np
import pandas as pd
import torch

from . import RESULTS_DIR, RESULTS_SMOKE_DIR
from har import runlog as R
from har.compression import head_unpruned, predict_logits, prunable_tensors
from har.data import require_harth
from har.paths import CHECKPOINT_DIR, RESULTS_DIR as REPO_RESULTS_DIR
from har.train import predict_proba
from har.xgp3 import layerwise_reconstruct
from har.xgp4 import (ROUND4, U_CELLS, bounded_budget, iterative_prune_budget, uniform_keep_counts,
                      unreconstructed_tensors, worst_group_importance, worst_group_shares)

from .context import FoldContext
from .qat import qat_kd_finetune, steps_per_epoch
from .quant import mean_bits, pruned_size_record, quantize_model, uniform_bits
from .rules import WIN, compare
from .run import bitred_sha256
from .run_qat_regimes import REGIME_SOURCE, regime_indices
from .run_qat_small import check_masks

P = "bitred-xgp-compare-tinyhar"
FOLDS = (0, 1, 2, 3, 4)
METHODS = ("lamp_recon", "xgp_xb")          # comparator, ours
LEVELS = (0.9, 0.95)
REGIMES = ("full", "kd5", "calibration")
QUANT = {"bits": 4, "granularity": "channel"}
REFERENCE = {"setting": "tensor-2bit", "bits": 2, "granularity": "tensor"}
QAT = {"steps": 4020, "eval_every_steps": 201, "lrs": (1e-4, 1e-3), "seed": 0, "batch_size": 256}
SMOKE_QAT = {"steps": 3, "eval_every_steps": 3}
XGP = {"seed": 0, "K": ROUND4["K"], "beta": ROUND4["beta_u"], "top_k": ROUND4["top_k"], "m": ROUND4["m_score"],
       "score_batch": ROUND4["score_batch"], "damp": ROUND4["damp"], "floor_frac": ROUND4["floor_frac"]}
SMOKE_XGP = {"m": 2, "K": 2}
PRIMARY_LR_RULE = ("Primary lr per regime = the lr with the higher mean F1_val of lamp_recon (QAT, 4-bit) over the "
                   "5 folds and both levels; tie -> 1e-4. Chosen on the comparator only, never on XGP or on LCI.")
R2_RULE = ("Per level x regime at the primary lr, XGP (xgp_xb) beats LAMP (lamp_recon) iff over the 5 folds: "
           "(a) mean LCI_val gain >= 0.02; (b) LCI_val strictly higher in >= 4 of 5 folds; (c) mean F1_val >= "
           "LAMP's - 0.01; (d) mean worst-subject LCI_val >= LAMP's - 0.01. Fewer than 5 folds -> incomplete, no "
           "verdict. Fixed before any run; nothing is retuned.")
ROUND4_DIR = REPO_RESULTS_DIR / "har-xgp-pilot" / "round4"
MODEL_COLS = ("model_id", "fold", "method", "level", "regime", "lr", "role", "f1", "lci", "worst_subject_lci")


def tag(method: str, level: float) -> str:
    return f"{method}{int(round(level * 100))}"


def xgp_cfg(smoke: bool) -> dict:
    return {**XGP, **(SMOKE_XGP if smoke else {})}


class Calibration:
    """Round 4's calibration inputs of the XGP importance: windows, original predictions, labels, groups."""

    def __init__(self, ctx: FoldContext):
        cidx = np.asarray(ctx.windows["cal_idx"])
        self.X = ctx.X_cal
        self.t = torch.from_numpy(predict_proba(ctx.net0, ctx.tr.X[cidx], ctx.dev).argmax(1)).to(ctx.dev)
        self.y = ctx.tr.y[cidx].copy()
        self.g = ctx.tr.subject[cidx].astype(str)            # round 4: group = subject
        self.unrecon = unreconstructed_tensors(ctx.net0, self.X[:4])


def build_start(net0, method: str, level: float, cal: Calibration, bg: torch.Tensor, xc: dict, shares_fn,
                log=print) -> tuple[torch.nn.Module, dict, dict | None, float]:
    """Round 4's start for method at level: (pruned + reconstructed model, keep counts, shares or None, seconds)."""
    budget, sel = U_CELLS[method]
    t0 = time.perf_counter()
    shares = shares_fn() if budget == "xb" else None
    keep = bounded_budget(net0, level, cal.unrecon, budget, shares)
    beta = xc["beta"] if sel == "xgp" else 0.0
    eg = (2 * xc["seed"], 2 * xc["seed"] + 1)
    imp = (lambda net: worst_group_importance(net, net0, cal.X, cal.t, cal.y, cal.g, bg, k=xc["top_k"], m=xc["m"],
                                              eg_seeds=eg, batch_size=xc["score_batch"])) if sel == "xgp" else None
    net, _ = iterative_prune_budget(net0, keep, K=xc["K"], beta=beta, importance_fn=imp,
                                    recon_fn=lambda n: layerwise_reconstruct(n, net0, cal.X, damp=xc["damp"]), log=log)
    if not head_unpruned(net):
        raise RuntimeError(f"{method}: the head was pruned")
    return net.eval(), keep, shares, time.perf_counter() - t0


def round4_reference(k: int, method: str, level: float) -> dict:
    """Round 4's own no-fine-tune score of the same start (seed 0), if committed (folds 0-1): a reproduction check."""
    for f in sorted(glob.glob(str(ROUND4_DIR / f"har-xgp-pilot4-tinyhar-fold{k}-unstructured-*-runs.csv"))):
        r = pd.read_csv(f)
        s = r[(r.method == method) & (r.scenario == "no_finetune") & (r.seed == 0) & (abs(r.level - level) < 1e-9)
              & r.ptq_bits.isna()]
        if len(s):
            return {"round4_f1": float(s.f1.iloc[0]), "round4_lci": float(s.lci.iloc[0])}
    return {"round4_f1": None, "round4_lci": None}


class Store:
    TABLES = ("models", "per-subject-lci", "finetune-history", "budgets")

    def __init__(self, out, base, resume: bool, smoke: bool):
        self.paths = {t: out / f"{base}-{t}.csv" for t in self.TABLES}
        self.rows = {t: [] for t in self.TABLES}
        if self.paths["models"].exists() and not (resume or smoke):
            raise FileExistsError(f"{self.paths['models'].name} exists; --resume to continue, or delete it")
        if resume:
            for t, p in self.paths.items():
                if p.exists():
                    self.rows[t] = pd.read_csv(p, dtype={"subject": str, "worst_subject": str}).to_dict("records")
        self.done = {r["model_id"] for r in self.rows["models"]}

    def save(self):
        for t, p in self.paths.items():
            if self.rows[t]:
                R.write_csv(p, pd.DataFrame(self.rows[t]))


def run_fold(k: int, args, out) -> None:
    S = args.smoke
    base = f"{P}-fold{k}-{args.machine_name}"
    store = Store(out, base, args.resume, S)
    qc, xc = {**QAT, **(SMOKE_QAT if S else {})}, xgp_cfg(S)
    ckdir = CHECKPOINT_DIR / "bit-reduction-test" / ("xgp-compare-smoke" if S else "xgp-compare")
    with R.tee_to(out / "logs" / f"{base}-{R.stamp()}.log"):
        t_all = time.perf_counter()
        print(f"=== XGP vs LAMP (ultra-small, QAT), fold {k}, {args.machine_name}, {args.device}"
              f"{' (smoke)' if S else ''}; {len(store.done)} models already done ===")
        ctx = FoldContext(k, args.device, S, args.noise_floor_machine)
        net0, cal = ctx.net0, Calibration(ctx)
        common = {"fold": k, "machine_name": args.machine_name, "device": args.device, "tau_k": ctx.tau,
                  "ceiling": ctx.ceiling}
        data = {}
        for regime in REGIMES:
            idx = np.arange(len(ctx.tr)) if regime == "full" else regime_indices(ctx, regime)
            tr = ctx.tr.select(idx)
            require_harth(tr)
            data[regime] = (tr, torch.from_numpy(predict_logits(net0, tr.X, args.device)).to(args.device))
            print(f"regime {regime}: {len(tr)} training windows, {qc['steps']} Adam steps = "
                  f"{qc['steps'] / steps_per_epoch(len(tr), qc['batch_size']):.1f} epochs")
        print(f"calibration: {len(cal.y)} windows, {len(set(cal.g.tolist()))} groups (subjects); not reconstructed: "
              f"{cal.unrecon}")
        seconds = {"starts": {}, "shares": None}
        shares_cache = {}

        def shares_fn():
            if "s" not in shares_cache:
                t0 = time.perf_counter()
                eg = (2 * xc["seed"], 2 * xc["seed"] + 1)
                shares_cache["s"] = worst_group_shares(net0, net0, cal.X, cal.t, cal.y, cal.g, ctx.bg, k=xc["top_k"],
                                                       m=xc["m"], eg_seeds=eg, batch_size=xc["score_batch"])
                seconds["shares"] = time.perf_counter() - t0
            return shares_cache["s"]

        def record(mid, net, method, level, role, regime, lr, bits, gran, masks, ft, extra=None):
            if mid in store.done:
                return
            met, sub = ctx.assess(net, mid)
            size = pruned_size_record(net0, bits, masks, gran)
            store.rows["models"].append({**common, "model_id": mid, "method": method, "level": level, "role": role,
                                         "regime": regime, "n_train_windows": len(data[regime][0]) if regime else None,
                                         "lr": lr, "adam_steps": qc["steps"] if role == "qat" else None,
                                         "bits": max(bits.values()) if bits else None,"granularity": gran if bits else None,
                                         **size, "mean_bits": mean_bits(net0, bits) if bits else 32.0,
                                         "finetune_seconds": ft, **(extra or {}), **met})
            store.rows["per-subject-lci"].extend({**common, "model_id": mid, "method": method, "level": level,
                                                  "regime": regime, "lr": lr, **r} for r in sub)
            store.done.add(mid)
            store.save()
            print(f"  {mid}: {size['size_kb']:.2f} KB, F1 {met['f1']:.4f}, LCI {met['lci']:.4f} "
                  f"(worst {met['worst_subject_lci']:.4f})" + (f"; QAT {ft:.1f} s" if ft else ""))

        def qat_all(start, method, level, prefix, bits, gran, masks, keep):
            for regime in REGIMES:
                tr, zt = data[regime]
                for lr in qc["lrs"]:
                    mid = f"qat-{prefix}-{regime}-lr{lr:g}"
                    if mid in store.done:
                        continue
                    print(f"-- {mid} ({len(tr)} windows)")
                    t0 = time.perf_counter()
                    qnet, h = qat_kd_finetune(copy.deepcopy(start), tr, ctx.va, zt, bits, granularity=gran, lr=lr,
                                              batch_size=qc["batch_size"], seed=qc["seed"], device=args.device,
                                              steps=qc["steps"], eval_every_steps=qc["eval_every_steps"])
                    ft = time.perf_counter() - t0
                    if h[-1]["step"] != qc["steps"]:
                        raise RuntimeError(f"{mid}: {h[-1]['step']} steps, expected {qc['steps']}")
                    if masks:
                        check_masks(qnet, masks, keep)
                    store.rows["finetune-history"].extend({**common, "model_id": mid, "method": method,
                                                           "level": level, "regime": regime, "lr": lr, **r} for r in h)
                    record(mid, qnet, method, level, "qat", regime, lr, bits, gran, masks, ft)

        record("original", net0, "original", None, "original", None, None, {}, "channel", {}, 0.0)
        # ---- quantization-only reference: per-tensor 2-bit
        rb = uniform_bits(net0, REFERENCE["bits"])
        record("ptq-tensor-2bit", quantize_model(net0, rb, "tensor"), "tensor-2bit", None, "reference (no QAT)",
               None, None, rb, "tensor", {}, 0.0)
        qat_all(net0, "tensor-2bit", None, "tensor-2bit", rb, "tensor", {}, None)
        # ---- pruned starts, 4-bit per channel
        qb = uniform_bits(net0, QUANT["bits"])
        sizes = {}
        for level in LEVELS:
            for method in METHODS:
                t = tag(method, level)
                ids = [f"{t}-fp32", f"{t}-ptq-channel-4bit"] + [f"qat-{t}-channel-4bit-{r}-lr{lr:g}"
                                                                for r in REGIMES for lr in qc["lrs"]]
                if all(i in store.done for i in ids):
                    continue
                print(f"-- start {t}")
                ck = ckdir / f"fold{k}-{t}.pt"
                if args.resume and ck.exists():
                    saved = torch.load(ck, map_location=args.device, weights_only=False)
                    start = copy.deepcopy(net0)
                    start.load_state_dict(saved["state_dict"])
                    start.prune_masks, keep, shares, sec = saved["masks"], saved["keep"], saved["shares"], 0.0
                    print(f"  reused {ck.name}")
                else:
                    start, keep, shares, sec = build_start(net0, method, level, cal, ctx.bg, xc, shares_fn)
                    ck.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"state_dict": start.state_dict(), "masks": start.prune_masks, "keep": keep,
                                "shares": shares}, ck)
                seconds["starts"][t] = sec
                masks = {n: m.detach().cpu() for n, m in start.prune_masks.items()}
                check_masks(start, masks, keep)
                if not any(r.get("start") == t for r in store.rows["budgets"]):
                    uni = uniform_keep_counts(net0, level)
                    store.rows["budgets"].extend(
                        {**common, "start": t, "method": method, "level": level, "tensor": n,
                         "n": int(getattr(m, p).numel()), "uniform_keep": uni[n], "keep": keep[n],
                         "share": (shares or {}).get(n), "budget_protected": n in cal.unrecon}
                        for n, m, p in prunable_tensors(net0))
                size = pruned_size_record(net0, qb, masks, "channel")["size_kb"]
                if sizes.setdefault(level, size) != size:
                    raise RuntimeError(f"{t}: {size} KB differs from the other method's {sizes[level]} KB")
                record(f"{t}-fp32", start, method, level, "reference (no QAT)", None, None, {}, "channel", masks, 0.0,
                       {"start_seconds": sec, **round4_reference(k, method, level)})
                ptq = quantize_model(start, qb, "channel")
                check_masks(ptq, masks, keep)
                record(f"{t}-ptq-channel-4bit", ptq, method, level, "reference (no QAT)", None, None, qb, "channel",
                       masks, 0.0)
                qat_all(start, method, level, f"{t}-channel-4bit", qb, "channel", masks, keep)
        seconds["wall"] = time.perf_counter() - t_all
        R.write_json(out / f"{base}-config.json",
                     {"command": " ".join(sys.argv), **R.device_info(args.device), **R.git_info(),
                      "bitred_code_sha256": bitred_sha256(), "har_code_sha256": R.code_sha256(),
                      "methods": {m: U_CELLS[m] for m in METHODS}, "levels": LEVELS, "quant": QUANT,
                      "reference": REFERENCE, "qat": qc, "xgp": xc, "regimes": {r: REGIME_SOURCE.get(r, "all training "
                      "windows of the fold") for r in REGIMES}, "n_train_windows": {r: len(d[0]) for r, d in data.items()},
                      "rule": R2_RULE, "primary_lr_rule": PRIMARY_LR_RULE, "window_hashes": ctx.hashes,
                      "calibration_matches_round4": ctx.cal_matches_round4, "checkpoint": str(ctx.ckpt),
                      "tau_k": ctx.tau, "ceiling": ctx.ceiling, "seconds": seconds, "har70plus_loaded": False,
                      "finished_at": R.now()})
        print(f"fold {k} done: {len(store.done)} models, wall {seconds['wall'] / 60:.1f} min")


# ------------------------------------------------------------------------------ analysis

def load_models(out, use_machine: dict[int, str]) -> pd.DataFrame:
    frames = []
    for k in FOLDS:
        files = sorted(glob.glob(str(out / f"{P}-fold{k}-*-models.csv")))
        if not files:
            continue
        machines = {f.split(f"{P}-fold{k}-")[1][:-len("-models.csv")]: f for f in files}
        m = use_machine.get(k) or (next(iter(machines)) if len(machines) == 1 else None)
        if m is None:
            raise RuntimeError(f"fold {k} ran on {sorted(machines)}: pass --use-machine {k}=<machine>")
        frames.append(pd.read_csv(machines[m], dtype={"worst_subject": str}))
    if not frames:
        raise FileNotFoundError(f"no {P}-fold*-models.csv in {out}")
    return pd.concat(frames, ignore_index=True)


def primary_lrs(m: pd.DataFrame) -> pd.DataFrame:
    q = m[(m.role == "qat") & (m.method == "lamp_recon")]
    rows = []
    for regime in REGIMES:
        f1 = {lr: float(q[(q.regime == regime) & np.isclose(q.lr, lr)].f1.mean()) for lr in QAT["lrs"]}
        a, b = QAT["lrs"]
        best = a if not (f1[b] > f1[a]) else b                   # tie (or NaN) -> 1e-4
        rows.append({"regime": regime, **{f"lamp_recon_mean_f1_lr{lr:g}": v for lr, v in f1.items()},
                     "n_runs": int(len(q[q.regime == regime])), "primary_lr": best})
    return pd.DataFrame(rows)


def comparisons(m: pd.DataFrame, prim: pd.DataFrame) -> pd.DataFrame:
    q = m[(m.role == "qat") & m.method.isin(METHODS)]
    pl = dict(zip(prim.regime, prim.primary_lr))
    rows = []
    for level in LEVELS:
        for regime in REGIMES:
            for lr in QAT["lrs"]:
                sel = q[np.isclose(q.level, level) & (q.regime == regime) & np.isclose(q.lr, lr)]
                pf = sel.assign(model_id=sel.method)[["model_id", "fold", "lci", "f1", "worst_subject_lci"]]
                c = compare(pf, "xgp_xb", "lamp_recon")
                n = c["n_folds"]
                rows.append({"level": level, "regime": regime, "lr": lr, "primary": bool(np.isclose(lr, pl[regime])),
                             "size_kb": float(sel.size_kb.iloc[0]) if len(sel) else None, **c,
                             "complete": n == len(FOLDS),
                             "verdict": ("incomplete" if n != len(FOLDS) else
                                         "XGP beats LAMP" if c["beats"] else "XGP does not beat LAMP")})
    return pd.DataFrame(rows)


def analyze(args, out) -> None:
    use = {int(a.split("=")[0]): a.split("=")[1] for a in (args.use_machine or [])}
    m = load_models(out, use)
    folds = sorted(m.fold.unique().tolist())
    keys = ["method", "level", "role", "regime", "lr"]
    g = m.assign(level=m.level.fillna(-1), regime=m.regime.fillna("-"), lr=m.lr.fillna(-1))
    summ = (g.groupby(keys).agg(n_folds=("fold", "nunique"), size_kb=("size_kb", "mean"),
                                f1_mean=("f1", "mean"), f1_sd=("f1", "std"), lci_mean=("lci", "mean"),
                                lci_sd=("lci", "std"), worst_subject_lci_mean=("worst_subject_lci", "mean"),
                                worst_subject_lci_sd=("worst_subject_lci", "std"))
            .reset_index().replace({"level": {-1: None}, "regime": {"-": None}, "lr": {-1: None}}))
    prim = primary_lrs(m)
    comp = comparisons(m, prim)
    R.write_csv(out / f"{P}-summary-mean-sd-over-folds.csv", summ)
    R.write_csv(out / f"{P}-per-fold.csv", m[[c for c in MODEL_COLS if c in m] + ["size_kb", "accuracy"]])
    R.write_csv(out / f"{P}-primary-lr.csv", prim)
    R.write_csv(out / f"{P}-comparisons.csv", comp)
    pc = comp[comp.primary]
    by = {f"{int(round(r.level * 100))}%|{r.regime}": {
        "verdict": r.verdict, "primary_lr": r.lr, "size_kb": r.size_kb, "n_folds": int(r.n_folds),
        "d_lci_mean": r.d_lci_mean if r.n_folds else None, "n_folds_higher": r.n_folds_higher if r.n_folds else None,
        "d_f1_mean": r.d_f1_mean if r.n_folds else None, "d_worst_mean": r.d_worst_mean if r.n_folds else None,
        **({c: bool(getattr(r, c)) for c in ("a_gain", "b_folds", "c_f1", "d_worst")} if r.n_folds else {})}
        for r in pc.itertuples()}
    dec = {"verdict_by_level_regime": {k: v["verdict"] for k, v in by.items()}, "by_level_regime": by,
           "xgp_beats_lamp_anywhere": any(v["verdict"] == "XGP beats LAMP" for v in by.values()),
           "complete": bool(pc.complete.all()), "folds": folds, "primary_lr": dict(zip(prim.regime, prim.primary_lr)),
           "rule": R2_RULE, "primary_lr_rule": PRIMARY_LR_RULE, "thresholds": WIN,
           "n_folds_needed": ceil(WIN["fold_share"] * len(FOLDS))}
    R.write_json(out / f"{P}-decision.json", dec)
    print(pc[["level", "regime", "lr", "n_folds", "lci_ours", "lci_comparator", "d_lci_mean", "n_folds_higher",
              "d_f1_mean", "d_worst_mean", "verdict"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--folds", type=int, nargs="+", choices=FOLDS)
    g.add_argument("--analyze", action="store_true")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--machine-name", default=R.machine_name())
    ap.add_argument("--noise-floor-machine", default=None)
    ap.add_argument("--use-machine", nargs="*", help="k=<machine> when a fold ran on more than one machine")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = RESULTS_SMOKE_DIR if args.smoke else RESULTS_DIR
    if args.analyze:
        analyze(args, out)
        return
    for k in args.folds:
        run_fold(k, args, out)


if __name__ == "__main__":
    main()
