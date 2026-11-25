"""Final test on HAR70+ (README.md, "Final test on HAR70+", pre-registered 2026-11-25 before any code or run; run once).

The final comparison's winning cell (xgp_xb vs lamp_recon at 90 %|kd5, lr 1e-3) on held-out test subjects with the 9A
test protocol. Nothing in the methods, baselines, QAT + KD recipe, metrics or 9A protocol changes: the models are
rebuilt with run_xgp_compare's own functions and values, and scored with har.run_compression_grid's Context / assess.

Order (enforced):
    python -m bitred.run_final_test --phase validation --folds 0 1 --device cuda --machine-name <m> [--resume]
    python -m bitred.run_final_test --analyze-validation          # needs all 5 folds; writes validation-complete.json
    python -m bitred.run_final_test --phase test --folds 0 1 --device cuda --machine-name <m> [--resume]
    python -m bitred.run_final_test --analyze                     # H1-H3 + secondary analyses
    --smoke: tiny CPU check on folds 0-1 (results/har-final-test-ultrasmall-smoke/)

validation  trains every model of the fold (fold_plan: 62 QAT models), saves its checkpoint, scores it on inner val
            (F1_val, LCI_val, worst-subject LCI_val, as run_xgp_compare); HAR70+ and the fold's HARTH test subjects are
            never loaded.
test        refuses until validation-complete.json lists the fold with the same sha256 of its validation file; loads
            the saved checkpoints (no training), checks inner-val F1 against validation, and runs 9A's assess (SHAP on the
            9A explain windows, LCI, AOPC, accuracy, non-stair F1, per class) on every model and the original.

Seed s (round 4's convention): QAT batch-order seed s; EG seeds (2s, 2s + 1) for the XGP importance and shares. Starts
without either (lamp_recon: LAMP budget + magnitude) are deterministic and built once (seed 0).
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import contextlib
import copy
import glob
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from . import RESULTS_DIR as BITRED_RESULTS_DIR, RESULTS_SMOKE_DIR as BITRED_RESULTS_SMOKE_DIR
from har import run_compression_grid as G
from har import runlog as R
from har.compression import predict_logits, prunable_tensors
from har.data import WALKING_AID_INFERRED, load_harth_windows, load_test_windows, require_harth
from har.folds import load_folds
from har.paths import CHECKPOINT_DIR, RESULTS_DIR as REPO_RESULTS_DIR
from har.stats import (BOOT_SEED, N_BOOT, bootstrap_fa_lci_ci, bootstrap_group_gap_ci, holm, paired_wilcoxon,
                       pool_over_folds)
from har.xgp4 import U_CELLS, uniform_keep_counts, worst_group_shares

from .context import NF_MACHINE, FoldContext
from .qat import qat_kd_finetune, steps_per_epoch
from .quant import mean_bits, pruned_size_record, uniform_bits
from .rules import WIN, compare
from .run import bitred_sha256
from .run_qat_regimes import REGIME_SOURCE, regime_indices
from .run_qat_small import check_masks
from .run_xgp_compare import (LEVELS, P as COMPARE_P, QAT, QUANT, REFERENCE, REGIMES, SMOKE_QAT, Calibration, Store,
                              build_start, tag, xgp_cfg)

P = "har-final-test-ultrasmall-tinyhar"
OUT = REPO_RESULTS_DIR / "har-final-test-ultrasmall"
OUT_SMOKE = REPO_RESULTS_DIR / "har-final-test-ultrasmall-smoke"
FOLDS, SMOKE_FOLDS = (0, 1, 2, 3, 4), (0, 1)
MAIN = ("lamp_recon", "xgp_xb")                     # comparator, ours
ABLATIONS = ("l1_xb_recon", "xgp_lamp")             # XGP budgets + magnitude; LAMP budgets + XGP selection
LRS, PRIMARY_LR = QAT["lrs"], 1e-3
H_LEVEL, H_REGIME = 0.9, "kd5"
SEEDS, EXTRA_SEEDS = (0, 1, 2), (1, 2)
ALPHA = 0.05
SENS_K, SENS_OMEGA = (5, 10, 20), (0.25, 0.5, 0.75)
COMPARE_CKPT = CHECKPOINT_DIR / "bit-reduction-test" / "xgp-compare"
TEST_TABLES = ("per-subject-metrics", "per-subject-class-lci", "per-group-class-lci", "models")
TEST_METRICS = ["lci", "shap_corr", "jaccard", "lci768", "lci_k16", "retention", "accuracy", "non_stair_macro_f1",
                "aopc", "drop_k5", "random_aopc", "random_drop_k5", "d_accuracy"]
KEYS = [*G.SETTING, "model_id", "regime", "lr", "seed", "role"]
PREREG = {
    "H1": ("Single primary, confirmatory, no multiplicity correction: at 90%|kd5, lr 1e-3, per-subject LCI (9A "
           "protocol) averaged over seeds 0, 1, 2 (each seed pooled over folds first); paired one-sided Wilcoxon "
           "signed-rank xgp_xb > lamp_recon over the 40 test subjects; supported iff p < 0.05."),
    "H2": ("At 90%|kd5, lr 1e-3, seed mean: D2 = |gap_lamp_recon| - |gap_xgp_xb|, gap = mean_older(AOPC) - "
           "mean_general(AOPC); paired stratified bootstrap (10,000, seed 42); supported iff the 95% CI of D2 > 0."),
    "H3": ("Exploratory (5 vs 13 subjects): at 90%|kd5, lr 1e-3, seed mean: D3 = wa_gap_xgp_xb - wa_gap_lamp_recon, "
           "wa_gap = mean_walking_aid(d_accuracy) - mean_other_older(d_accuracy); paired bootstrap; supported "
           "(exploratory) iff the 95% CI of D3 > 0."),
    "secondary_cells": "the other 5 level x regime cells, lr 1e-3, seed 0: one-sided Wilcoxon on LCI, Holm over the 5",
    "secondary_analyses": "ablations (validation first, then test, Holm over 12), seeds 1-2 on validation, both lrs, "
                          "LCI sensitivity k in {5,10,20} x omega in {0.25,0.5,0.75}; none changes H1-H3",
}


# ------------------------------------------------------------------------------ plan (pure)

def model_id(method: str, level: float | None, regime: str, lr: float, seed: int) -> str:
    if method == "tensor-2bit":
        return f"qat-tensor-2bit-{regime}-lr{lr:g}-seed{seed}"
    return f"qat-{tag(method, level)}-channel-4bit-{regime}-lr{lr:g}-seed{seed}"


def start_seed(method: str, seed: int) -> int:
    """The seed a start depends on: none (0) unless it uses XGP budgets or XGP selection (EG seeds 2s, 2s + 1)."""
    budget, sel = U_CELLS[method]
    return seed if (budget == "xb" or sel == "xgp") else 0


def fold_plan() -> list[dict]:
    """The fold's 62 QAT models, in run order, grouped by start."""
    rows = []

    def add(method, level, regime, lr, seed, role):
        rows.append({"model_id": model_id(method, level, regime, lr, seed), "method": method, "level": level,
                     "regime": regime, "lr": lr, "seed": seed, "role": role})

    for regime in REGIMES:
        for lr in LRS:
            add("tensor-2bit", None, regime, lr, 0, "reference")
    for level in LEVELS:
        for method in MAIN + ABLATIONS:
            for regime in REGIMES:
                for lr in LRS:
                    add(method, level, regime, lr, 0, "main" if method in MAIN else "ablation")
    for s in EXTRA_SEEDS:
        for method in MAIN:
            for lr in LRS:
                add(method, H_LEVEL, H_REGIME, lr, s, "seed")
    return rows


def level_str(level) -> str:
    return "none" if level is None or (isinstance(level, float) and np.isnan(level)) else f"{int(round(level * 100))}%"


def scenario(regime, lr, seed) -> str:
    return f"{regime}|lr{lr:g}|seed{seed}"


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------ validation phase

def run_validation_fold(k: int, args, out: Path) -> None:
    S = args.smoke
    vout = out / "validation"
    base = f"{P}-fold{k}-{args.machine_name}"
    store = Store(vout, base, args.resume, S)
    qc, xc0 = {**QAT, **(SMOKE_QAT if S else {})}, xgp_cfg(S)
    ckdir = CHECKPOINT_DIR / "bit-reduction-test" / ("final-test-smoke" if S else "final-test")
    ckdir.mkdir(parents=True, exist_ok=True)
    plan = fold_plan()
    with R.tee_to(vout / "logs" / f"{base}-validation-{R.stamp()}.log"):
        t_all = time.perf_counter()
        print(f"=== final test, VALIDATION phase, fold {k}, {args.machine_name}, {args.device}{' (smoke)' if S else ''}; "
              f"{len(store.done)} of {len(plan) + 1} models already done ===")
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
            print(f"regime {regime}: {len(tr)} windows, {qc['steps']} Adam steps = "
                  f"{qc['steps'] / steps_per_epoch(len(tr), qc['batch_size']):.1f} epochs")
        seconds = {"starts": {}, "shares": {}}
        shares_cache = {}

        def shares_fn(seed):
            if seed not in shares_cache:
                t0 = time.perf_counter()
                shares_cache[seed] = worst_group_shares(net0, net0, cal.X, cal.t, cal.y, cal.g, ctx.bg,
                                                        k=xc0["top_k"], m=xc0["m"], eg_seeds=(2 * seed, 2 * seed + 1),
                                                        batch_size=xc0["score_batch"])
                seconds["shares"][seed] = time.perf_counter() - t0
            return shares_cache[seed]

        def record(e, net, bits, gran, masks, ft, extra):
            met, sub = ctx.assess(net, e["model_id"])
            size = pruned_size_record(net0, bits, masks, gran)
            store.rows["models"].append({**common, **e, "n_train_windows": len(data[e["regime"]][0]) if e["regime"] else None,
                                         "adam_steps": qc["steps"] if e["role"] != "original" else None,
                                         "bits": max(bits.values()) if bits else None,
                                         "granularity": gran if bits else None, **size,
                                         "mean_bits": mean_bits(net0, bits) if bits else 32.0, "finetune_seconds": ft,
                                         **extra, **met})
            store.rows["per-subject-lci"].extend({**common, "model_id": e["model_id"], "method": e["method"],
                                                  "level": e["level"], "regime": e["regime"], "lr": e["lr"],
                                                  "seed": e["seed"], **r} for r in sub)
            store.done.add(e["model_id"])
            store.save()
            print(f"  {e['model_id']}: {size['size_kb']:.2f} KB, F1 {met['f1']:.4f}, LCI {met['lci']:.4f} "
                  f"(worst {met['worst_subject_lci']:.4f})" + (f"; QAT {ft:.1f} s" if ft else ""))

        if "original" not in store.done:
            record({"model_id": "original", "method": "original", "level": None, "regime": None, "lr": None,
                    "seed": None, "role": "original"}, net0, {}, "channel", {}, 0.0, {"checkpoint": None})

        def get_start(method, level, seed):
            ss = start_seed(method, seed)
            t = tag(method, level)
            own = ckdir / f"fold{k}-{t}-seed{ss}-start.pt"
            cmp_ck = COMPARE_CKPT / f"fold{k}-{t}.pt"
            if own.exists() and (args.resume or S):
                sv, source, sec = torch.load(own, map_location=args.device, weights_only=False), None, 0.0
                source = sv["source"]
            elif ss == 0 and method in MAIN and cmp_ck.exists() and not S:
                sv, source, sec = torch.load(cmp_ck, map_location=args.device, weights_only=False), \
                    f"run_xgp_compare checkpoint {cmp_ck.name}", 0.0
            else:
                net, keep, sh, sec = build_start(net0, method, level, cal, ctx.bg, {**xc0, "seed": ss},
                                                 lambda: shares_fn(ss))
                sv, source = {"state_dict": net.state_dict(), "masks": net.prune_masks, "keep": keep, "shares": sh}, \
                    "rebuilt (run_xgp_compare.build_start)"
            if not own.exists() or not (args.resume or S):
                torch.save({**sv, "source": source}, own)
            start = copy.deepcopy(net0)
            start.load_state_dict(sv["state_dict"])
            start.prune_masks = sv["masks"]
            return start.eval(), sv["keep"], sv["shares"], sec, source, ss

        qb, rb = uniform_bits(net0, QUANT["bits"]), uniform_bits(net0, REFERENCE["bits"])
        sizes, groups = {}, {}
        for e in plan:
            groups.setdefault((e["method"], e["level"], start_seed(e["method"], e["seed"])
                               if e["method"] != "tensor-2bit" else 0), []).append(e)
        for (method, level, ss), entries in groups.items():
            if all(e["model_id"] in store.done for e in entries):
                continue
            if method == "tensor-2bit":
                start, keep, masks, bits, gran = net0, None, {}, rb, REFERENCE["granularity"]
                extra = {"start_source": "original (unpruned)", "start_seconds": 0.0}
            else:
                print(f"-- start {tag(method, level)} seed {ss}")
                start, keep, shares, sec, source, _ = get_start(method, level, entries[0]["seed"])
                seconds["starts"][f"{tag(method, level)}-seed{ss}"] = sec
                masks = {n: m.detach().cpu() for n, m in start.prune_masks.items()}
                check_masks(start, masks, keep)
                bits, gran = qb, QUANT["granularity"]
                size = pruned_size_record(net0, bits, masks, gran)["size_kb"]
                if sizes.setdefault(level, size) != size:
                    raise RuntimeError(f"{method} {level}: {size} KB differs from {sizes[level]} KB at the same level")
                st = f"{tag(method, level)}-seed{ss}"
                if not any(r.get("start") == st for r in store.rows["budgets"]):
                    uni = uniform_keep_counts(net0, level)
                    store.rows["budgets"].extend(
                        {**common, "start": st, "method": method, "level": level, "start_seed": ss, "tensor": n,
                         "n": int(getattr(m, p).numel()), "uniform_keep": uni[n], "keep": keep[n],
                         "share": (shares or {}).get(n), "budget_protected": n in cal.unrecon}
                        for n, m, p in prunable_tensors(net0))
                extra = {"start_source": source, "start_seconds": sec}
            for e in entries:
                if e["model_id"] in store.done:
                    continue
                tr, zt = data[e["regime"]]
                print(f"-- {e['model_id']} ({len(tr)} windows)")
                t0 = time.perf_counter()
                qnet, h = qat_kd_finetune(copy.deepcopy(start), tr, ctx.va, zt, bits, granularity=gran, lr=e["lr"],
                                          batch_size=qc["batch_size"], seed=e["seed"], device=args.device,
                                          steps=qc["steps"], eval_every_steps=qc["eval_every_steps"])
                ft = time.perf_counter() - t0
                if h[-1]["step"] != qc["steps"]:
                    raise RuntimeError(f"{e['model_id']}: {h[-1]['step']} steps, expected {qc['steps']}")
                if masks:
                    check_masks(qnet, masks, keep)
                cp = ckdir / f"fold{k}-{e['model_id']}.pt"
                torch.save({"state_dict": {n: v.detach().cpu() for n, v in qnet.state_dict().items()}, "masks": masks,
                            "bits": bits, "granularity": gran, "entry": e}, cp)
                store.rows["finetune-history"].extend({**common, **e, **r} for r in h)
                record(e, qnet, bits, gran, masks, ft, {**extra, "checkpoint": str(cp)})
        seconds["wall"] = time.perf_counter() - t_all
        R.write_json(vout / f"{base}-config.json",
                     {"phase": "validation", "command": " ".join(sys.argv), **R.device_info(args.device), **R.git_info(),
                      "bitred_code_sha256": bitred_sha256(), "har_code_sha256": R.code_sha256(),
                      "methods": {m: U_CELLS[m] for m in MAIN + ABLATIONS}, "levels": LEVELS, "quant": QUANT,
                      "reference": REFERENCE, "qat": qc, "xgp": xc0, "seeds": {"main_cell": SEEDS, "other": (0,)},
                      "regimes": {r: REGIME_SOURCE.get(r, "all training windows of the fold") for r in REGIMES},
                      "n_train_windows": {r: len(d[0]) for r, d in data.items()}, "preregistration": PREREG,
                      "window_hashes": ctx.hashes, "calibration_matches_round4": ctx.cal_matches_round4,
                      "har70plus_loaded": False, "seconds": seconds, "finished_at": R.now()})
        print(f"fold {k} validation done: {len(store.done)} models, wall {seconds['wall'] / 60:.1f} min")


# ------------------------------------------------------------------------------ validation analysis

def load_fold_files(d: Path, pattern_suffix: str, folds, use_machine: dict) -> tuple[pd.DataFrame, dict]:
    frames, chosen = [], {}
    for k in folds:
        files = sorted(glob.glob(str(d / f"{P}-fold{k}-*-{pattern_suffix}")))
        machines = {Path(f).name[len(f"{P}-fold{k}-"):-len(f"-{pattern_suffix}")]: f for f in files}
        if not machines:
            continue
        m = use_machine.get(k) or (next(iter(machines)) if len(machines) == 1 else None)
        if m is None:
            raise RuntimeError(f"fold {k} has files from {sorted(machines)}: pass --use-machine {k}=<machine>")
        frames.append(pd.read_csv(machines[m], dtype={"subject": str, "worst_subject": str}))
        chosen[k] = m
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), chosen


def val_compare_rows(q: pd.DataFrame, ours: str, other: str, keys: dict) -> dict:
    pf = q[q.method.isin([ours, other])].assign(model_id=lambda d: d.method)[["model_id", "fold", "lci", "f1",
                                                                             "worst_subject_lci"]]
    c = compare(pf, ours, other)
    return {**keys, **c, "r2_verdict": ("incomplete" if c["n_folds"] < len(FOLDS) else
                                        f"{ours} beats {other}" if c["beats"] else f"{ours} does not beat {other}")}


def analyze_validation(args, out: Path) -> None:
    vdir = out / "validation"
    need = SMOKE_FOLDS if args.smoke else FOLDS
    use = {int(a.split("=")[0]): a.split("=", 1)[1] for a in (args.use_machine or [])}
    m, chosen = load_fold_files(vdir, "models.csv", need, use)
    missing = [k for k in need if k not in chosen]
    if missing:
        raise SystemExit(f"validation incomplete: folds {missing} missing in {vdir} (the test phase needs all of {need})")
    expected = {e["model_id"] for e in fold_plan()} | {"original"}
    for k in need:
        got = set(m[m.fold == k].model_id)
        if got != expected:
            raise SystemExit(f"fold {k}: validation has {len(got)} of {len(expected)} models; finish it with --resume")
    q = m[m.role != "original"]
    summ = (q.groupby(["method", "level", "regime", "lr", "seed", "role"], dropna=False)
            .agg(n_folds=("fold", "nunique"), size_kb=("size_kb", "mean"), f1_mean=("f1", "mean"), f1_sd=("f1", "std"),
                 lci_mean=("lci", "mean"), lci_sd=("lci", "std"), worst_subject_lci_mean=("worst_subject_lci", "mean"),
                 worst_subject_lci_sd=("worst_subject_lci", "std")).reset_index())
    comps = []
    for lr in LRS:
        for level in LEVELS:
            for regime in REGIMES:
                sel = q[np.isclose(q.level.astype(float), level) & (q.regime == regime) & np.isclose(q.lr, lr)]
                s0 = sel[sel.seed == 0]
                for ours in ("xgp_xb",) + ABLATIONS:
                    comps.append(val_compare_rows(s0, ours, "lamp_recon", {"level": level, "regime": regime, "lr": lr,
                                                                           "seed": "0", "primary_lr": lr == PRIMARY_LR}))
                if level == H_LEVEL and regime == H_REGIME:
                    for s in EXTRA_SEEDS:
                        comps.append(val_compare_rows(sel[sel.seed == s], "xgp_xb", "lamp_recon",
                                                      {"level": level, "regime": regime, "lr": lr, "seed": str(s),
                                                       "primary_lr": lr == PRIMARY_LR}))
                    sm = (sel[sel.method.isin(MAIN)].groupby(["method", "fold"])[["lci", "f1", "worst_subject_lci"]]
                          .mean().reset_index())
                    comps.append(val_compare_rows(sm, "xgp_xb", "lamp_recon", {"level": level, "regime": regime,
                                                                               "lr": lr, "seed": "mean 0-2",
                                                                               "primary_lr": lr == PRIMARY_LR}))
    comps = pd.DataFrame(comps)
    hc = q[np.isclose(q.level.astype(float), H_LEVEL) & (q.regime == H_REGIME) & q.method.isin(MAIN)]
    seeds = (hc.groupby(["method", "lr", "seed"]).agg(n_folds=("fold", "nunique"), f1_mean=("f1", "mean"),
                                                       lci_mean=("lci", "mean"),
                                                       worst_subject_lci_mean=("worst_subject_lci", "mean"))
             .reset_index())
    # reproduction check against the final comparison (seed 0, same models retrained)
    cdir = BITRED_RESULTS_SMOKE_DIR if args.smoke else BITRED_RESULTS_DIR
    cm = [pd.read_csv(f, dtype={"worst_subject": str}) for f in sorted(glob.glob(str(cdir / f"{COMPARE_P}-fold*-models.csv")))]
    rep = pd.DataFrame()
    if cm:
        cm = pd.concat(cm, ignore_index=True)
        cm = cm[(cm.role == "qat") & cm.method.isin(MAIN + ("tensor-2bit",))]
        ours = q[(q.seed == 0) & q.method.isin(MAIN + ("tensor-2bit",))]
        on = ["fold", "method", "regime", "lr"]
        cm = cm.assign(level=cm.level.fillna(-1.0))
        # one machine per fold (the real comparison ran every fold once, on uni-2080; smoke folders can hold several)
        first = cm.groupby("fold").machine_name.first()
        cm = cm[cm.machine_name == cm.fold.map(first)]
        ours = ours.assign(level=ours.level.astype(float).fillna(-1.0))
        rep = ours[on + ["level", "f1", "lci", "worst_subject_lci"]].merge(
            cm[on + ["level", "f1", "lci", "worst_subject_lci", "machine_name"]], on=on + ["level"],
            suffixes=("_final_test", "_xgp_compare"))
        for c in ("f1", "lci", "worst_subject_lci"):
            rep[f"d_{c}"] = rep[f"{c}_final_test"] - rep[f"{c}_xgp_compare"]
        rep["level"] = rep.level.replace(-1.0, np.nan)
    pre = f"{P}-validation"
    R.write_csv(vdir / f"{pre}-summary-mean-sd-over-folds.csv", summ)
    R.write_csv(vdir / f"{pre}-comparisons-vs-lamp-recon.csv", comps)
    R.write_csv(vdir / f"{pre}-seeds-90pct-kd5.csv", seeds)
    R.write_csv(vdir / f"{pre}-reproduction-vs-xgp-compare.csv", rep)
    files = {str(k): {"machine": mch, "file": f"{P}-fold{k}-{mch}-models.csv",
                      "sha256": sha256_file(vdir / f"{P}-fold{k}-{mch}-models.csv")} for k, mch in chosen.items()}
    R.write_json(vdir / f"{pre}-complete.json",
                 {"complete": True, "folds": files, "n_models_per_fold": len(expected), "written_at": R.now(),
                  "note": "the test phase of fold k refuses unless its validation models file still has this sha256",
                  **R.git_info()})
    pc = comps[(comps.primary_lr) & (comps.seed == "0")]
    print(pc[["level", "regime", "ours", "lci_ours", "lci_comparator", "d_lci_mean", "n_folds_higher", "d_f1_mean",
              "r2_verdict"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if len(rep):
        print(f"reproduction vs run_xgp_compare: mean |d F1| {rep.d_f1.abs().mean():.4f}, mean |d LCI| "
              f"{rep.d_lci.abs().mean():.4f} over {len(rep)} models")
    print(f"saved {vdir / f'{pre}-complete.json'}")


def check_gate(out: Path, k: int, machine: str) -> dict:
    """The pre-registered order: no test assessment before every fold's validation is analysed and unchanged."""
    vdir = out / "validation"
    p = vdir / f"{P}-validation-complete.json"
    if not p.exists():
        raise SystemExit(f"{p.name} missing: finish the validation phase on every fold and run --analyze-validation "
                         "first (pre-registered order: validation folds 0-4, then test)")
    g = json.loads(p.read_text(encoding="utf-8"))
    f = g.get("folds", {}).get(str(k))
    if not g.get("complete") or f is None:
        raise SystemExit(f"{p.name}: fold {k} is not in the completed validation")
    if f["machine"] != machine:
        raise SystemExit(f"fold {k}'s validation ran on {f['machine']}: run its test phase there (the checkpoints are "
                         f"on that machine), with --machine-name {f['machine']}")
    if sha256_file(vdir / f["file"]) != f["sha256"]:
        raise SystemExit(f"{f['file']} changed after --analyze-validation; rerun --analyze-validation")
    return g


# ------------------------------------------------------------------------------ test phase

class TestStore(G.FoldStore):
    """9A's FoldStore (same tables, same npz of 48-segment SHAP), under this test's file names."""

    TABLES = TEST_TABLES

    def __init__(self, out: Path, k: int, machine: str, resume: bool, smoke: bool):
        self.base = f"{P}-fold{k}-{machine}"
        self.paths = {t: out / f"{self.base}-{t}.csv" for t in self.TABLES}
        self.npz = out / f"{self.base}-shap-48-segments.npz"
        self.config = out / f"{self.base}-config.json"
        exists = self.paths["models"].exists()
        if exists and not (resume or smoke):
            raise FileExistsError(f"{self.paths['models'].name} exists: the test runs once; --resume only continues it")
        keep = resume and exists
        self.rows = {t: (self._read(p) if keep else []) for t, p in self.paths.items()}
        self.seg = {}
        if keep and self.npz.exists():
            with np.load(self.npz, allow_pickle=False) as z:
                self.seg = {n: z[n] for n in z.files}
        self.done = {r["model_id"] for r in self.rows["models"]}


@contextlib.contextmanager
def smoke_noise_floor(smoke: bool):
    """Smoke only: the noise-floor smoke files exist for fold 0 only, so other folds get placeholder tau / ceilings."""
    if not smoke:
        yield
        return
    rt, rc = G.read_tau, G.read_ceilings

    def tau(k, s, m):
        try:
            return rt(k, s, m)
        except FileNotFoundError:
            return 0.6, "smoke default"

    def ceil(k, s, m):
        try:
            return rc(k, s, m)
        except FileNotFoundError:
            return {g: 0.9 for g in G.EXPLAIN_GROUPS}

    G.read_tau, G.read_ceilings = tau, ceil
    try:
        yield
    finally:
        G.read_tau, G.read_ceilings = rt, rc


def run_test_fold(k: int, args, out: Path, harth: dict, test: dict, folds, frozen: dict) -> None:
    S = args.smoke
    gate = check_gate(out, k, args.machine_name)
    vm = pd.read_csv(out / "validation" / gate["folds"][str(k)]["file"], dtype={"worst_subject": str})
    store = TestStore(out, k, args.machine_name, args.resume, S)
    with R.tee_to(out / "logs" / f"{store.base}-test-{R.stamp()}.log"):
        t_all = time.perf_counter()
        dev = R.device_info(args.device)
        print(f"=== final test, TEST phase (9A protocol), fold {k}, {args.machine_name}, {args.device}"
              f"{' (smoke)' if S else ''}; {len(store.done)} of {len(vm)} models already done ===")
        cargs = SimpleNamespace(device=args.device, smoke=S, machine_name=args.machine_name,
                                noise_floor_machine=args.noise_floor_machine or NF_MACHINE[k])
        with smoke_noise_floor(S):
            ctx = G.Context(cargs, k, harth, test, folds, frozen)
        store.set_window_meta(G.window_meta(ctx))
        common = {"fold": k, "tau_k": ctx.tau, "tau_source": ctx.tau_src,
                  **{f"ceiling_{g}": v for g, v in ctx.ceilings.items()}, "machine_name": args.machine_name,
                  "device": args.device, "gpu_name": dev["gpu_name"], "noise_floor_machine": ctx.nf_machine}
        for r in vm.to_dict("records"):
            mid = r["model_id"]
            if mid in store.done:
                continue
            orig = r["role"] == "original"
            lv = None if orig or pd.isna(r["level"]) else float(r["level"])
            ex = {"regime": None if orig else r["regime"], "lr": None if orig else float(r["lr"]),
                  "seed": None if orig else int(r["seed"]), "role": r["role"], "size_kb": r["size_kb"]}
            entry = {"model_id": mid, "method": r["method"], "family": "original" if orig else "ultrasmall",
                     "scenario": "original" if orig else scenario(r["regime"], float(r["lr"]), int(r["seed"])),
                     "level": level_str(lv), "sparsity": lv, "bits": r.get("bits")}
            if orig:
                net = ctx.net0
            else:
                ck = torch.load(r["checkpoint"], map_location=args.device, weights_only=False)
                if ck["entry"]["model_id"] != mid:
                    raise RuntimeError(f"{r['checkpoint']} holds {ck['entry']['model_id']}, not {mid}")
                net = copy.deepcopy(ctx.net0)
                net.load_state_dict(ck["state_dict"])
                net = net.to(args.device).eval()
                if ck["masks"]:
                    check_masks(net, ck["masks"], {n: int(m.sum()) for n, m in ck["masks"].items()})
            s = G.assess(ctx, store, net, entry, {"subject_extra": {"model_id": mid, **ex}},
                         A=ctx.A_ref if orig else None)
            f1_test = s.get("macro_f1_pooled7_inner_val")
            d = abs(f1_test - r["f1"])
            if not S and d > 1e-3:
                raise RuntimeError(f"{mid}: inner-val F1 {f1_test:.6f} here vs {r['f1']:.6f} at validation: wrong "
                                   "checkpoint?")
            store.add("models", [{**entry, **ex, **common, "inner_val_f1_validation": r["f1"],
                                  "inner_val_lci_validation": r["lci"], "inner_val_f1_test_check": f1_test,
                                  "checkpoint_matches_validation": bool(d <= 1e-6) if not S else None,
                                  "size_kb_without_bitmask": r.get("size_kb_without_bitmask"),
                                  "checkpoint": r.get("checkpoint"), **s}])
            store.done.add(mid)
            store.save()
            print(f"  {mid:<52} LCI general {s.get('lci_general', np.nan):.4f}, older {s.get('lci_older', np.nan):.4f}; "
                  f"acc general {s.get('accuracy_pooled_general', np.nan):.4f}, older "
                  f"{s.get('accuracy_pooled_older', np.nan):.4f}; {s['assess_seconds']:.0f} s")
        wall = time.perf_counter() - t_all
        R.write_json(store.config, {"phase": "test", "command": " ".join(sys.argv), **dev, **R.git_info(),
                                    "bitred_code_sha256": bitred_sha256(), "har_code_sha256": R.code_sha256(),
                                    "fold": k, "smoke": S, "protocol": "9A (har.run_compression_grid.Context / assess)",
                                    "grid": G.GRID, "nsamples": ctx.nsamples, "tau_k": ctx.tau,
                                    "tau_source": ctx.tau_src, "ceilings": ctx.ceilings,
                                    "noise_floor_machine": ctx.nf_machine,
                                    "n_explain": {g: len(ctx.sets[g][1]) for g in G.EXPLAIN_GROUPS},
                                    "n_eval_windows": {g: len(w) for g, w in ctx.eval.items()},
                                    "validation_gate": gate["folds"][str(k)], "preregistration": PREREG,
                                    "seconds_wall_this_invocation": wall, "finished_at": R.now()})
        print(f"fold {k} test done: {len(store.done)} models, wall {wall / 60:.1f} min")


# ------------------------------------------------------------------------------ test analysis (pure)

def wilcoxon_greater(a, b) -> dict:
    """Paired one-sided Wilcoxon signed-rank, H_a: a > b (zero_method wilcox), plus paired_wilcoxon's two-sided p and
    effect sizes (median / mean difference, matched-pairs rank-biserial r, d_z)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    r = paired_wilcoxon(a, b)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    p1 = 1.0 if not (d != 0).any() else float(wilcoxon(a[ok], b[ok], zero_method="wilcox", alternative="greater").pvalue)
    r = {("p_two_sided" if k_ == "p" else k_): v for k_, v in r.items()}
    return {**r, "p_one_sided": p1}


def paired_gap_difference(xa, xb, g1, g2, *, absolute: bool, n_boot: int = N_BOOT, seed: int = BOOT_SEED,
                          level: float = 0.95) -> dict:
    """Two methods A, B on the same subjects (xa, xb aligned); groups g1, g2 (boolean masks). gap_M = mean_g1 - mean_g2.
    D = |gap_B| - |gap_A| (absolute: A shrinks the gap's size) or gap_A - gap_B (signed). Percentile CI with subjects
    resampled within each group, the same draws for both methods."""
    xa, xb = np.asarray(xa, float), np.asarray(xb, float)
    i1, i2 = np.flatnonzero(g1), np.flatnonzero(g2)
    rng = np.random.default_rng(seed)
    b1 = i1[rng.integers(0, len(i1), size=(n_boot, len(i1)))]
    b2 = i2[rng.integers(0, len(i2), size=(n_boot, len(i2)))]
    ga, gb = xa[b1].mean(1) - xa[b2].mean(1), xb[b1].mean(1) - xb[b2].mean(1)
    pa, pb = xa[i1].mean() - xa[i2].mean(), xb[i1].mean() - xb[i2].mean()
    dist = (np.abs(gb) - np.abs(ga)) if absolute else (ga - gb)
    q = (1 - level) / 2
    lo, hi = np.quantile(dist, [q, 1 - q])
    return {"gap_a": float(pa), "gap_b": float(pb), "d": float((abs(pb) - abs(pa)) if absolute else (pa - pb)),
            "d_ci_low": float(lo), "d_ci_high": float(hi), "n_g1": len(i1), "n_g2": len(i2), "n_boot": n_boot,
            "boot_seed": seed}


def subject_values(pooled: pd.DataFrame, method: str, level: float, regime: str, lr: float, seeds,
                   metric: str) -> pd.DataFrame:
    """Per test subject: the metric (pooled over folds) averaged over `seeds`; subjects lacking a seed are dropped."""
    d = pooled[(pooled.method == method) & np.isclose(pooled.lr.astype(float), lr) & (pooled.regime == regime)
               & np.isclose(pooled.sparsity.astype(float), level) & pooled.seed.isin(list(seeds))]
    g = d.groupby(["subject", "group", "subgroup"])
    out = g[metric].mean().rename(metric).to_frame()
    out["n_seeds"] = g.seed.nunique()
    return out[out.n_seeds == len(seeds)].reset_index()


def paired(pooled, level, regime, lr, seeds, metric, a="xgp_xb", b="lamp_recon") -> pd.DataFrame:
    va = subject_values(pooled, a, level, regime, lr, seeds, metric)
    vb = subject_values(pooled, b, level, regime, lr, seeds, metric)
    return va.merge(vb[["subject", metric]], on="subject", suffixes=("_a", "_b"))


def hypotheses(pooled: pd.DataFrame, lr: float, seeds=SEEDS, n_boot: int = N_BOOT) -> dict:
    """H1-H3 at the pre-registered cell (90 %|kd5) and lr; seeds averaged per subject."""
    out = {"lr": lr, "primary": bool(np.isclose(lr, PRIMARY_LR)), "seeds": list(seeds)}
    j = paired(pooled, H_LEVEL, H_REGIME, lr, seeds, "lci")
    if len(j) < 2:
        return {**out, "complete": False}
    w = wilcoxon_greater(j.lci_a, j.lci_b)
    o, g = j.group == "older", j.group == "general"
    fa = bootstrap_fa_lci_ci(j.lci_a[o], j.lci_a[g], j.lci_b[o], j.lci_b[g], n_boot=n_boot)
    f = paired(pooled, H_LEVEL, H_REGIME, lr, seeds, "non_stair_macro_f1")
    wf = paired_wilcoxon(f.non_stair_macro_f1_a, f.non_stair_macro_f1_b)
    out["H1"] = {"n_subjects": int(w["n"]), "n_general": int(g.sum()), "n_older": int(o.sum()),
                 "lci_xgp_xb": float(j.lci_a.mean()), "lci_lamp_recon": float(j.lci_b.mean()), **w,
                 "supported": bool(w["p_one_sided"] < ALPHA),
                 "lci_by_group": {grp: {"xgp_xb": float(j.lci_a[j.group == grp].mean()),
                                        "lamp_recon": float(j.lci_b[j.group == grp].mean())} for grp in ("general", "older")},
                 "worst_group_lci_xgp_xb": fa["fa_lci"], "worst_group_lci_lamp_recon": fa["fa_lci_b"],
                 "worst_group_lci_diff": fa["diff"], "worst_group_lci_diff_ci": [fa["diff_ci_low"], fa["diff_ci_high"]],
                 "non_stair_f1_xgp_xb": float(f.non_stair_macro_f1_a.mean()),
                 "non_stair_f1_lamp_recon": float(f.non_stair_macro_f1_b.mean()),
                 "non_stair_f1_wilcoxon_two_sided_p": wf["p"], "non_stair_f1_mean_diff": wf["mean_diff"],
                 "non_stair_f1_by_group": {grp: {"xgp_xb": float(f.non_stair_macro_f1_a[f.group == grp].mean()),
                                                 "lamp_recon": float(f.non_stair_macro_f1_b[f.group == grp].mean())}
                                           for grp in ("general", "older")}}
    a = paired(pooled, H_LEVEL, H_REGIME, lr, seeds, "aopc")
    h2 = paired_gap_difference(a.aopc_a, a.aopc_b, a.group == "older", a.group == "general", absolute=True,
                               n_boot=n_boot)
    ci = {m: bootstrap_group_gap_ci(a[f"aopc_{s}"][a.group == "older"], a[f"aopc_{s}"][a.group == "general"],
                                    n_boot=n_boot) for m, s in (("xgp_xb", "a"), ("lamp_recon", "b"))}
    out["H2"] = {"gap_older_minus_general_xgp_xb": h2["gap_a"], "gap_older_minus_general_lamp_recon": h2["gap_b"],
                 "gap_ci_xgp_xb": [ci["xgp_xb"]["ci_low"], ci["xgp_xb"]["ci_high"]],
                 "gap_ci_lamp_recon": [ci["lamp_recon"]["ci_low"], ci["lamp_recon"]["ci_high"]],
                 "D2_abs_gap_reduction": h2["d"], "D2_ci": [h2["d_ci_low"], h2["d_ci_high"]],
                 "signed_gap_difference_xgp_minus_lamp": h2["gap_a"] - h2["gap_b"],
                 "aopc_xgp_xb": float(a.aopc_a.mean()), "aopc_lamp_recon": float(a.aopc_b.mean()),
                 "n_older": h2["n_g1"], "n_general": h2["n_g2"], "supported": bool(h2["d_ci_low"] > 0)}
    d = paired(pooled, H_LEVEL, H_REGIME, lr, seeds, "d_accuracy")
    d = d[d.group == "older"]
    wa, oo = d.subgroup == "walking_aid_inferred", d.subgroup == "other_older"
    if wa.any() and oo.any():
        h3 = paired_gap_difference(d.d_accuracy_a, d.d_accuracy_b, wa, oo, absolute=False, n_boot=n_boot)
        out["H3"] = {"d_accuracy_wa_gap_xgp_xb": h3["gap_a"], "d_accuracy_wa_gap_lamp_recon": h3["gap_b"],
                     "D3": h3["d"], "D3_ci": [h3["d_ci_low"], h3["d_ci_high"]], "n_walking_aid": h3["n_g1"],
                     "n_other_older": h3["n_g2"], "exploratory": True,
                     "supported_exploratory": bool(h3["d_ci_low"] > 0)}
    else:
        out["H3"] = {"complete": False, "reason": "no walking-aid or no other-older subjects"}
    out["complete"] = True
    return out


def cell_tests(pooled: pd.DataFrame, lr: float) -> pd.DataFrame:
    """Secondary cells (xgp_xb > lamp_recon, seed 0, Holm over the 5 non-H1 cells) and the H1 cell at seed 0
    (descriptive, outside the Holm family)."""
    rows = []
    for level in LEVELS:
        for regime in REGIMES:
            j = paired(pooled, level, regime, lr, (0,), "lci")
            if len(j) < 2:
                continue
            h1 = np.isclose(level, H_LEVEL) and regime == H_REGIME
            rows.append({"level": level, "regime": regime, "lr": lr, "seed": 0,
                         "family": "H1 cell, seed 0 (descriptive)" if h1 else "secondary (Holm over 5)",
                         "lci_xgp_xb": float(j.lci_a.mean()), "lci_lamp_recon": float(j.lci_b.mean()),
                         **wilcoxon_greater(j.lci_a, j.lci_b)})
    out = pd.DataFrame(rows)
    if len(out):
        sec = out.family.str.startswith("secondary")
        out["p_holm"] = np.nan
        out.loc[sec, "p_holm"] = holm(out.loc[sec, "p_one_sided"].to_numpy())
        out["significant_holm"] = out.p_holm < ALPHA
    return out


def ablation_tests(pooled: pd.DataFrame, lr: float) -> pd.DataFrame:
    rows = []
    for abl in ABLATIONS:
        for level in LEVELS:
            for regime in REGIMES:
                j = paired(pooled, level, regime, lr, (0,), "lci", a=abl, b="lamp_recon")
                if len(j) >= 2:
                    rows.append({"test": f"{abl} > lamp_recon (one-sided)", "family": "vs_lamp_recon", "level": level,
                                 "regime": regime, "lr": lr, "lci_a": j.lci_a.mean(), "lci_b": j.lci_b.mean(),
                                 **wilcoxon_greater(j.lci_a, j.lci_b), "p_test": None})
                j = paired(pooled, level, regime, lr, (0,), "lci", a="xgp_xb", b=abl)
                if len(j) >= 2:
                    w = paired_wilcoxon(j.lci_a, j.lci_b)
                    rows.append({"test": f"xgp_xb vs {abl} (two-sided)", "family": "xgp_xb_vs_ablation",
                                 "level": level, "regime": regime, "lr": lr, "lci_a": j.lci_a.mean(),
                                 "lci_b": j.lci_b.mean(), **{("p_two_sided" if k_ == "p" else k_): v for k_, v in w.items()}})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_test"] = np.where(out.family == "vs_lamp_recon", out.get("p_one_sided"), out.p_two_sided)
        out["p_holm"] = np.nan
        for _, idx in out.groupby("family").groups.items():
            out.loc[idx, "p_holm"] = holm(out.loc[idx, "p_test"].to_numpy())
    return out


def add_d_accuracy(ps: pd.DataFrame) -> pd.DataFrame:
    o = ps[ps.model_id == "original"][["fold", "subject", "accuracy"]].rename(columns={"accuracy": "accuracy_original"})
    ps = ps.merge(o, on=["fold", "subject"], how="left")
    ps["d_accuracy"] = ps.accuracy - ps.accuracy_original
    return ps


def pool_test(ps: pd.DataFrame) -> pd.DataFrame:
    """Test subjects, one row per (model, subject) pooled over folds (9A), with seed-mean rows for the H1 cell."""
    t = ps[ps.group.isin(G.TEST_GROUPS)].copy()
    for c in ("regime", "role"):
        t[c] = t[c].astype(object).where(t[c].notna(), "-")
    for c in ("lr", "seed", "sparsity"):
        t[c] = t[c].astype(float).fillna(-1.0)
    pooled = pool_over_folds(t, [m for m in TEST_METRICS if m in t], keys=KEYS + ["sparsity"])
    pooled["subgroup"] = [G.subgroup(g, s) for g, s in zip(pooled.group, pooled.subject)]
    for c in ("lr", "seed", "sparsity"):
        pooled[c] = pooled[c].replace(-1.0, np.nan)
    pooled["regime"] = pooled.regime.replace("-", np.nan)
    pooled["role"] = pooled.role.replace("-", np.nan)
    # seed-mean settings of the H1 cell (both methods, both lrs), so the group tables show them too
    sm = []
    for method in MAIN:
        for lr in LRS:
            d = pooled[(pooled.method == method) & np.isclose(pooled.lr.astype(float), lr) & (pooled.regime == H_REGIME)
                       & np.isclose(pooled.sparsity.astype(float), H_LEVEL) & pooled.seed.isin(list(SEEDS))]
            if d.seed.nunique() != len(SEEDS):
                continue
            mets = [m for m in TEST_METRICS if m in d]
            g = d.groupby(["subject", "group", "subgroup"])[mets].mean().reset_index()
            sm.append(g.assign(method=method, family="ultrasmall", level=level_str(H_LEVEL), sparsity=H_LEVEL,
                               regime=H_REGIME, lr=lr, seed=np.nan, role="seed mean",
                               scenario=f"{H_REGIME}|lr{lr:g}|seedmean",
                               model_id=f"seedmean-{tag(method, H_LEVEL)}-channel-4bit-{H_REGIME}-lr{lr:g}",
                               n_folds=d.groupby("subject").n_folds.first().reindex(g.subject).to_numpy()))
    return pd.concat([pooled, *sm], ignore_index=True) if sm else pooled


def lci_sensitivity(seg_by_fold: dict[int, dict], lr_list=LRS, n_boot: int = N_BOOT) -> pd.DataFrame:
    """H1's test recomputed from the saved 48-segment SHAP for k x omega (seed mean, 90 %|kd5)."""
    from lci.metric import compute_lci

    rows = []
    for kk in SENS_K:
        for om in SENS_OMEGA:
            per = []
            for fold, z in seg_by_fold.items():
                ref = z["original"]
                subj, grp = z["meta_subject"], z["meta_group"]
                test = np.isin(grp, G.TEST_GROUPS)
                for method in MAIN:
                    for lr in lr_list:
                        for s in SEEDS:
                            mid = model_id(method, H_LEVEL, H_REGIME, lr, s)
                            if mid not in z:
                                continue
                            for sj in sorted(set(subj[test].tolist())):
                                m = subj == sj
                                per.append({"fold": fold, "method": method, "lr": lr, "seed": s, "subject": sj,
                                            "group": str(grp[m][0]),
                                            "lci": compute_lci(ref[m], z[mid][m], k=kk, omega=om)["lci"]})
            if not per:
                continue
            per = pd.DataFrame(per)
            pooled = pool_over_folds(per, ["lci"], keys=["method", "lr", "seed"])
            for lr in lr_list:
                v = {mt: pooled[(pooled.method == mt) & np.isclose(pooled.lr, lr)].groupby("subject")
                     .agg(lci=("lci", "mean"), n=("seed", "nunique")) for mt in MAIN}
                v = {mt: x[x.n == len(SEEDS)].lci for mt, x in v.items()}
                j = pd.concat(v, axis=1, join="inner")
                if len(j) < 2:
                    continue
                rows.append({"k": kk, "omega": om, "lr": lr, "primary_setting": kk == G.GRID["k"] and om == G.GRID["omega"],
                             "lci_xgp_xb": j["xgp_xb"].mean(), "lci_lamp_recon": j["lamp_recon"].mean(),
                             **wilcoxon_greater(j["xgp_xb"], j["lamp_recon"])})
    return pd.DataFrame(rows)


def analyze(args, out: Path) -> None:
    need = SMOKE_FOLDS if args.smoke else FOLDS
    use = {int(a.split("=")[0]): a.split("=", 1)[1] for a in (args.use_machine or [])}
    ps, chosen = load_fold_files(out, "per-subject-metrics.csv", need, use)
    models, _ = load_fold_files(out, "models.csv", need, {k: m for k, m in chosen.items()})
    complete = all(k in chosen for k in need)
    expected = {e["model_id"] for e in fold_plan()} | {"original"}
    if complete:
        complete = all(set(models[models.fold == k].model_id) == expected for k in need)
    if ps.empty:
        raise FileNotFoundError(f"no {P}-fold*-per-subject-metrics.csv in {out}")
    t0 = time.perf_counter()
    ps = add_d_accuracy(ps)
    pooled = pool_test(ps)
    H = {lr: hypotheses(pooled, lr, n_boot=args.n_boot) for lr in LRS}
    cells = pd.concat([cell_tests(pooled, lr) for lr in LRS], ignore_index=True)
    abl = pd.concat([ablation_tests(pooled, lr) for lr in LRS], ignore_index=True)
    segs = {}
    for k, mch in chosen.items():
        p = out / f"{P}-fold{k}-{mch}-shap-48-segments.npz"
        if p.exists():
            with np.load(p, allow_pickle=False) as z:
                segs[k] = {n: z[n] for n in z.files}
    sens = lci_sensitivity(segs, n_boot=args.n_boot)
    gs = G.group_summary(pooled, TEST_METRICS, args.n_boot)
    fa = G.fa_lci_table(pooled, args.n_boot)
    safe = G.safe_rate_table(ps.assign(scenario=ps.scenario.astype(str)))
    keep = [c for c in ["model_id", "method", "level", "regime", "lr", "seed", "role", "size_kb"] if c in models]
    perf = [c for c in models.columns if c.startswith(("lci_", "accuracy_pooled_", "macro_f1_pooled7_"))
            or c in ("inner_val_f1_validation", "inner_val_lci_validation", "checkpoint_matches_validation")]
    msum = (models.assign(regime=models.regime.fillna("-"), lr=models.lr.fillna(-1), seed=models.seed.fillna(-1),
                          level=models.level.astype(str))
            .groupby(keep[:-1], dropna=False)
            .agg(n_folds=("fold", "nunique"), size_kb=("size_kb", "mean"),
                 **{c: (c, "mean") for c in perf if c != "checkpoint_matches_validation"},
                 **({"checkpoints_match_validation": ("checkpoint_matches_validation", "all")}
                    if "checkpoint_matches_validation" in models and models.checkpoint_matches_validation.notna().any()
                    else {})).reset_index())
    rows = []
    for lr, h in H.items():
        for hn in ("H1", "H2", "H3"):
            if hn in h:
                rows.append({"lr": lr, "primary": h["primary"], "hypothesis": hn,
                             **{k_: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k_, v in h[hn].items()}})
    hyp = pd.DataFrame(rows)
    hp = H[PRIMARY_LR]
    dec = {"complete": bool(complete), "folds_and_machines": chosen, "primary_lr": PRIMARY_LR,
           "cell": f"{int(H_LEVEL * 100)}%|{H_REGIME}", "seeds": list(SEEDS), "preregistration": PREREG}
    if complete and hp.get("complete"):
        dec.update({"H1_supported": hp["H1"]["supported"], "H2_supported": hp["H2"]["supported"],
                    "H3_supported_exploratory": hp["H3"].get("supported_exploratory"),
                    "H1": hp["H1"], "H2": hp["H2"], "H3": hp["H3"],
                    "secondary_cells_lr1e-3": cells[np.isclose(cells.lr, PRIMARY_LR)].to_dict("records"),
                    "secondary_lr1e-4": {k_: H[1e-4].get(k_) for k_ in ("H1", "H2", "H3")}})
    else:
        dec["verdict"] = f"incomplete: needs all {len(need)} folds with every model; no H1-H3 verdict"
    files = {"hypotheses-both-lrs": hyp, "secondary-cells-holm": cells, "ablations-test-holm": abl,
             "group-summary-gaps-bootstrap-ci": gs, "fa-lci-bootstrap-ci": fa, "safe-explanation-rate": safe,
             "per-subject-pooled-over-folds": pooled, "lci-sensitivity-k-omega": sens, "models-summary": msum}
    for name, df in files.items():
        R.write_csv(out / f"{P}-{name}.csv", df)
    dec["seconds"] = time.perf_counter() - t0
    dec.update({"analyzed_at": R.now(), **R.git_info(), "n_boot": args.n_boot, "boot_seed": BOOT_SEED})
    R.write_json(out / f"{P}-decision.json", dec)
    if complete and hp.get("complete"):
        h1, h2, h3 = hp["H1"], hp["H2"], hp["H3"]
        print(f"H1: LCI xgp_xb {h1['lci_xgp_xb']:.4f} vs lamp_recon {h1['lci_lamp_recon']:.4f} over {h1['n_subjects']} "
              f"subjects, one-sided p {h1['p_one_sided']:.4g} -> {'SUPPORTED' if h1['supported'] else 'not supported'}")
        print(f"H2: D2 {h2['D2_abs_gap_reduction']:+.4f} CI [{h2['D2_ci'][0]:+.4f}, {h2['D2_ci'][1]:+.4f}] -> "
              f"{'SUPPORTED' if h2['supported'] else 'not supported'}")
        if "D3" in h3:
            print(f"H3 (exploratory): D3 {h3['D3']:+.4f} CI [{h3['D3_ci'][0]:+.4f}, {h3['D3_ci'][1]:+.4f}] -> "
                  f"{'supported' if h3['supported_exploratory'] else 'not supported'}")
    else:
        print(dec.get("verdict"))
    print(f"saved {out / f'{P}-decision.json'} and {len(files)} tables; {dec['seconds']:.1f} s")


# ------------------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--phase", choices=["validation", "test"])
    g.add_argument("--analyze-validation", action="store_true")
    g.add_argument("--analyze", action="store_true")
    ap.add_argument("--folds", type=int, nargs="+", choices=FOLDS)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--machine-name", default=R.machine_name())
    ap.add_argument("--noise-floor-machine", default=None, help="default: bitred.context.NF_MACHINE (as the 9A analysis)")
    ap.add_argument("--use-machine", nargs="*", help="k=<machine> when a fold ran on more than one machine")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = OUT_SMOKE if args.smoke else OUT
    if args.analyze_validation:
        return analyze_validation(args, out)
    if args.analyze:
        return analyze(args, out)
    if not args.folds:
        ap.error("--phase needs --folds")
    if args.smoke and any(k not in SMOKE_FOLDS for k in args.folds):
        ap.error(f"--smoke runs folds {SMOKE_FOLDS} only")
    if args.phase == "validation":
        for k in args.folds:
            run_validation_fold(k, args, out)
        return
    harth, test = load_harth_windows(), load_test_windows()
    counts = pd.Series(harth["harth_test"].subject).value_counts().to_dict()
    folds = load_folds(harth_test_counts={s: int(c) for s, c in counts.items()})
    frozen = json.loads(G.FROZEN_CONFIG.read_text(encoding="utf-8"))
    assert frozen["model"] == "tinyhar", frozen["model"]
    for k in args.folds:
        run_test_fold(k, args, out, harth, test, folds, frozen)


if __name__ == "__main__":
    main()
