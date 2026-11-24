"""Low-bit diagnostic (baselines only, no new method; 2026-11-24): below ~30 KB, does the best existing
compression silently break explanations, or hurt one group more?

Every model starts from the frozen TinyHAR of the fold (seed 0, model-selection checkpoint, 54,273 parameters,
212 KB in float32). No retraining and no fine-tuning anywhere:
    ptq_pc4, ptq_pc3, ptq_pc2   per-output-channel symmetric weight PTQ (har.compression.ptq_per_channel) of
                                every prunable tensor plus the head (the tensors of the plan's ptq_symmetric)
    lamp_recon 90%, 95%         round 4's unstructured LAMP + reconstruction baseline (har/xgp4.py): LAMP layer
                                budgets (Lee et al. 2021) through the round-4 bounds, K = 5 nested L1 steps, ridge
                                layer-wise reconstruction after each step on round 4's calibration set (<= 10
                                training windows per (subject, class), seed 46). Reconstruction is a closed-form
                                refit of kept weights on training windows, not gradient training. fp32 reference.
    lamp_recon + ptq_pc4        the same pruned models, then per-channel 4-bit PTQ (pruned weights stay 0)
    layer2bit:<tensor>          per-layer sensitivity: only that tensor per-channel 2-bit, the rest float32
Size (har.compression.low_bit_size): kept values at their bit width + one fp32 scale per quantized row, plus
(size_kb_with_bitmask) 1 bit per position of every pruned tensor.

Windows, SHAP and metrics are Step 9A's (har/run_compression_grid.py: Context, assess): explain windows
inner_val (3 HARTH subjects) / general (fold's HARTH test subjects) / older (18 HAR70+ subjects), GradientExplainer
nsamples 200, SHAP seed 0, target = the original model's predicted class; LCI 48 segments, k = 10; accuracy and
F1 on all of the group's windows; deletion faithfulness. tau_k and the ceilings come from the committed
noise-floor files (fold 0-1 nakhoul-3050, 2-4 uni-2080, as the 9A analysis).
    (1) validation = HARTH inner val, folds 0-4: LCI_val, worst-subject LCI_val, pooled F1_val, share below tau_k
    (2) test = baseline measurement as 9A: older (HAR70+) vs general (HARTH), walking-aid (inferred) split,
        subjects pooled over folds, bootstrap CIs (10,000, seed 42)

Usage (repo root, inside .venv):
    python -m har.run_diagnostic_low_bit --folds 0 1 2 3 4 --device cuda --machine-name <m>
    python -m har.run_diagnostic_low_bit --analyze
    --smoke: tiny CPU check (results/har-diagnostic-low-bit-smoke/); --resume continues a fold;
    --no-layer-sensitivity skips the 13 single-layer models.

Writes results/har-diagnostic-low-bit/, per fold k and machine (prefix har-diag-lowbit-tinyhar-fold<k>-<machine>):
    -per-subject-metrics.csv, -per-subject-class-lci.csv, -per-group-class-lci.csv, -models.csv,
    -shap-48-segments.npz (local), -config.json, logs/
and with --analyze the har-diag-lowbit-a-*.csv tables (see analyze()).
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from . import runlog as R
from .compression import low_bit_size, prunable_tensors, ptq_per_channel, size_record
from .data import find_windows, load_harth_windows, load_test_windows
from .folds import load_folds
from .paths import CHECKPOINT_DIR, DIAG_LOW_BIT_DIR, DIAG_LOW_BIT_SMOKE_DIR, NOISE_FLOOR_SMOKE_DIR
from .run_compression_grid import (FROZEN_CONFIG, GRID, SETTING, TEST_GROUPS, Context, FoldStore, assess,
                                   fa_lci_table, group_summary, load_fold_files, per_class_table, pool_subjects,
                                   safe_rate_table, stairs_table, window_meta)
from .xgp3 import layerwise_reconstruct
from .xgp4 import ROUND4, bounded_budget, iterative_prune_budget, unreconstructed_tensors

PREFIX = "har-diag-lowbit-tinyhar"
APREFIX = "har-diag-lowbit-a"
DIAG = {"ptq_bits": (4, 3, 2), "lamp_levels": (0.9, 0.95), "lamp_ptq_bits": 4, "sens_bits": 2,
        "K": ROUND4["K"], "damp": ROUND4["damp"], "cal_per_cell": ROUND4["cal_per_cell"],
        "cal_seed": ROUND4["cal_seed"], "silent_tol": 0.01, "target_kb": 30}
SMOKE_D = {"K": 2}
# whose (committed) noise-floor files give tau_k and the ceilings: as the 9A analysis (fold 3 = uni-2080)
NF_MACHINE = {0: "nakhoul-3050", 1: "nakhoul-3050", 2: "uni-2080", 3: "uni-2080", 4: "uni-2080"}
FAMILIES = ("original", "ptq_per_channel", "pruning_recon", "pruning_recon_ptq", "layer_sensitivity")
TABLES = ("per-subject-metrics", "per-subject-class-lci", "per-group-class-lci", "models")


def pct(level: float) -> int:
    return int(round(level * 100))


# ------------------------------------------------------------------------------ plan (pure)

def diag_plan(tensors: Sequence[str], with_layers: bool = True) -> list[dict]:
    """The models of one fold, in run order."""
    base = {"sparsity": np.nan, "bits": np.nan, "tensor": None, "scenario": "no_finetune"}
    plan = [{**base, "model_id": "original", "method": "original", "family": "original", "scenario": "original",
             "level": "none"}]
    for b in DIAG["ptq_bits"]:
        plan.append({**base, "model_id": f"ptq_pc{b}", "method": f"ptq_pc{b}", "family": "ptq_per_channel",
                     "level": f"{b}-bit", "bits": b})
    qb = DIAG["lamp_ptq_bits"]
    for lv in DIAG["lamp_levels"]:
        plan.append({**base, "model_id": f"lamp_recon-{pct(lv)}pct", "method": "lamp_recon", "family": "pruning_recon",
                     "level": f"{pct(lv)}%", "sparsity": lv})
        plan.append({**base, "model_id": f"lamp_recon-{pct(lv)}pct-ptq_pc{qb}", "method": f"lamp_recon_ptq_pc{qb}",
                     "family": "pruning_recon_ptq", "level": f"{pct(lv)}%+{qb}-bit", "sparsity": lv, "bits": qb})
    if with_layers:
        sb = DIAG["sens_bits"]
        for t in tensors:
            plan.append({**base, "model_id": f"layer{sb}bit-{t}", "method": f"layer{sb}bit:{t}",
                         "family": "layer_sensitivity", "level": f"{sb}-bit", "bits": sb, "tensor": t})
    return plan


# ------------------------------------------------------------------------------ one fold

class DiagStore(FoldStore):
    """Step 9A's fold store without the fine-tuning history (nothing is fine-tuned here)."""
    TABLES = TABLES


def calibration_windows(ctx: Context, smoke: bool):
    """Round 4's calibration set for this fold (<= cal_per_cell training windows per (subject, class), seed
    46), rebuilt with round 4's own function; for folds 0-1 it must equal round 4's saved table."""
    from .run_xgp_pilot4 import cal_table_path, calibration_table, out_dir

    tab = calibration_table(ctx.tr, smoke)
    idx = find_windows(ctx.tr, "harth_train", tab.window_id.tolist())
    p = cal_table_path(out_dir(smoke), ctx.k)
    match = None
    if p.exists():
        saved = pd.read_csv(p, dtype={"subject": str})
        match = saved.window_id.tolist() == tab.window_id.tolist()
        if not match and not smoke:
            raise RuntimeError(f"fold {ctx.k}: calibration windows differ from round 4's {p.name}")
    return torch.from_numpy(ctx.tr.X[idx]).to(ctx.dev), {"n_calibration": int(len(idx)),
                                                         "calibration_matches_round4_table": match}


def lamp_recon_start(ctx: Context, level: float, cal_X: torch.Tensor, K: int, ckpt: Path):
    """LAMP + reconstruction start (round 4's lamp_recon, deterministic), cached as a checkpoint."""
    if ckpt.exists():
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        net = copy.deepcopy(ctx.net0)
        net.load_state_dict(ck["state_dict"])
        net.prune_masks = {n: m.to(ctx.dev) for n, m in ck["prune_masks"].items()}
        return net.to(ctx.dev).eval(), ck["keep"], 0.0
    t0 = time.perf_counter()
    unrecon = unreconstructed_tensors(ctx.net0, cal_X[:4])
    keep = bounded_budget(ctx.net0, level, unrecon, "lamp")
    net, _ = iterative_prune_budget(ctx.net0, keep, K=K, beta=0.0, importance_fn=None,
                                    recon_fn=lambda n: layerwise_reconstruct(n, ctx.net0, cal_X, damp=DIAG["damp"]))
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {n: v.cpu() for n, v in net.state_dict().items()},
                "prune_masks": {n: m.cpu() for n, m in net.prune_masks.items()}, "keep": keep}, ckpt)
    return net.eval(), keep, time.perf_counter() - t0


def nf_machine_for(k: int, args) -> str:
    if args.noise_floor_machine:
        return args.noise_floor_machine
    if args.smoke:
        found = sorted(NOISE_FLOOR_SMOKE_DIR.glob(f"har-noise-floor-tinyhar-fold{k}-*-tau.csv"))
        if not found:
            raise FileNotFoundError(f"no smoke noise floor for fold {k}: run python -m har.run_noise_floor --folds {k} --smoke")
        return found[0].name[len(f"har-noise-floor-tinyhar-fold{k}-"):-len("-tau.csv")]
    return NF_MACHINE[k]


def run_fold(args, k: int, harth: dict, test: dict, folds, frozen: dict) -> None:
    out = DIAG_LOW_BIT_SMOKE_DIR if args.smoke else DIAG_LOW_BIT_DIR
    ckdir = CHECKPOINT_DIR / ("har-diagnostic-low-bit-smoke" if args.smoke else "har-diagnostic-low-bit")
    machine = args.machine_name
    store = DiagStore(out, k, machine, args.resume, args.smoke, prefix=PREFIX)
    log = out / "logs" / f"{store.base}-{R.stamp()}.log"
    with R.tee_to(log):
        dev = R.device_info(args.device)
        t_fold = time.perf_counter()
        print(f"=== low-bit diagnostic fold {k} | machine {machine} | device {args.device} {dev['gpu_name']} | torch "
              f"{dev['torch_version']} | started {R.now()} | {' '.join(sys.argv)}"
              + (f" | resuming, {len(store.done)} models done" if store.done else ""))
        a = copy.copy(args)
        a.noise_floor_machine = nf_machine_for(k, args)
        ctx = Context(a, k, harth, test, folds, frozen)
        store.set_window_meta(window_meta(ctx))
        tensors = [n for n, _, _ in prunable_tensors(ctx.net0, include_head=True)]
        numel = {n: int(getattr(m, p).numel()) for n, m, p in prunable_tensors(ctx.net0, include_head=True)}
        plan = diag_plan(tensors, with_layers=not args.no_layer_sensitivity)
        K = SMOKE_D["K"] if args.smoke else DIAG["K"]
        common = {"fold": k, "tau_k": ctx.tau, "tau_source": ctx.tau_src,
                  **{f"ceiling_{g}": v for g, v in ctx.ceilings.items()}, "noise_floor_machine": a.noise_floor_machine,
                  "machine_name": machine, "device": args.device, "gpu_name": dev["gpu_name"]}
        cal_X, cal_info, starts, times = None, {}, {}, {}

        def finish(entry, net, summary, quantized, extra):
            rec = size_record(net)
            row = {**{c: entry.get(c) for c in ("model_id", "method", "family", "scenario", "level", "sparsity", "bits",
                                                 "tensor")},
                   "tensor_numel": numel.get(entry.get("tensor")), "quantized_tensors": json.dumps(sorted(quantized)),
                   **{c: rec[c] for c in ("n_params", "n_nonzero", "sparsity_prunable", "sparsity_weights_incl_head",
                                          "sparsity_all_params")},
                   "per_layer_sparsity": json.dumps(rec["per_layer_sparsity"]), **low_bit_size(net, quantized),
                   **extra, **summary, **common}
            store.add("models", [row])
            store.done.add(entry["model_id"])
            store.save()
            print(f"  {entry['model_id']:<52} {row['size_kb_with_bitmask']:7.1f} KB | LCI val "
                  f"{summary.get('lci_inner_val', np.nan):.4f}, general {summary.get('lci_general', np.nan):.4f}, older "
                  f"{summary.get('lci_older', np.nan):.4f} | acc val {summary.get('accuracy_pooled_inner_val', np.nan):.4f}, "
                  f"general {summary.get('accuracy_pooled_general', np.nan):.4f}, older "
                  f"{summary.get('accuracy_pooled_older', np.nan):.4f} | {summary['assess_seconds']:.0f} s")

        for entry in plan:
            mid = entry["model_id"]
            if mid in store.done:
                continue
            if entry["family"] == "original":
                s = assess(ctx, store, ctx.net0, entry, {}, A=ctx.A_ref)
                finish(entry, ctx.net0, s, {}, {"shap_seconds_original": ctx.times["shap_original"]})
                continue
            if entry["family"] == "ptq_per_channel":
                net, q = ptq_per_channel(ctx.net0, entry["bits"])
                finish(entry, net, assess(ctx, store, net.to(ctx.dev).eval(), entry, {}), q, {})
                continue
            if entry["family"] == "layer_sensitivity":
                net, q = ptq_per_channel(ctx.net0, entry["bits"], only=[entry["tensor"]])
                finish(entry, net, assess(ctx, store, net.to(ctx.dev).eval(), entry, {}), q, {})
                continue
            lv = entry["sparsity"]
            if lv not in starts:
                if cal_X is None:
                    cal_X, cal_info = calibration_windows(ctx, args.smoke)
                ck = ckdir / f"{store.base}-lamp_recon-{pct(lv)}pct.pt"
                starts[lv] = lamp_recon_start(ctx, lv, cal_X, K, ck)
                times[f"lamp_recon_{pct(lv)}pct_build_seconds"] = starts[lv][2]
            start, keep, build_s = starts[lv]
            extra = {"lamp_keep_counts": json.dumps(keep), "K": K, "damp": DIAG["damp"], "build_seconds": build_s,
                     **cal_info}
            if entry["family"] == "pruning_recon":
                finish(entry, start, assess(ctx, store, start, entry, {}), {}, extra)
            else:
                net, q = ptq_per_channel(start, entry["bits"])
                for n, m in net.prune_masks.items():
                    w = dict(net.named_parameters())[n]
                    if torch.any(w[~m.to(w.device)] != 0):
                        raise RuntimeError(f"{mid}: a pruned weight of {n} became non-zero")
                finish(entry, net, assess(ctx, store, net.to(ctx.dev).eval(), entry, {}), q, extra)

        wall = time.perf_counter() - t_fold
        R.write_json(store.config, {"command_line": sys.argv, "machine_name": machine, "device": dev, **R.git_info(),
                                    "code_sha256": R.code_sha256(), "fold": k, "smoke": args.smoke, "diag": DIAG,
                                    "K": K, "grid_metrics": {c: GRID[c] for c in ("nsamples", "shap_seed", "k", "omega")},
                                    "nsamples": ctx.nsamples, "tau_k": ctx.tau, "tau_source": ctx.tau_src,
                                    "ceilings": ctx.ceilings, "noise_floor_machine": a.noise_floor_machine,
                                    "weights": str(ctx.ckpt), "n_explain": {g: len(ctx.sets[g][1]) for g in ctx.sets},
                                    "n_eval_windows": {g: len(w) for g, w in ctx.eval.items()}, **cal_info,
                                    "layer_sensitivity": not args.no_layer_sensitivity, "seconds_fixed": {**ctx.times, **times},
                                    "n_models": len(store.done), "seconds_wall_this_invocation": wall,
                                    "finished_at": R.now()})
        print(f"fold {k} done in {wall:.1f} s")
        for p in (*store.paths.values(), store.npz, store.config, log):
            print(f"saved {p}")


# ------------------------------------------------------------------------------ analysis (pure)

def validation_per_fold(per_subject: pd.DataFrame, models: pd.DataFrame, tol: float = DIAG["silent_tol"]) -> pd.DataFrame:
    """Per (fold, setting), inner-val windows only: LCI_val (subject mean), worst-subject LCI_val, rho, Jaccard,
    share of the inner-val subjects below tau_k, LCI_val < tau_k, pooled 7-class F1_val and accuracy and their
    change from the fold's original model, size; silent_break = F1_val drop <= tol and LCI_val < tau_k."""
    iv = per_subject[per_subject.group == "inner_val"]
    g = iv.groupby(["fold", *SETTING], dropna=False)
    v = g.agg(lci_val=("lci", "mean"), worst_subject_lci_val=("lci", "min"), rho_val=("shap_corr", "mean"),
              jaccard_val=("jaccard", "mean"), n_subjects=("lci", "size"), tau_k=("tau_k", "first"))
    v["share_subjects_below_tau_k"] = g.apply(lambda d: float((d.lci < d.tau_k).mean()), include_groups=False)
    v = v.reset_index()
    m = models[["fold", "model_id", *SETTING, "tensor", "size_kb", "size_kb_with_bitmask", "n_nonzero",
                "macro_f1_pooled7_inner_val", "accuracy_pooled_inner_val"]]
    v = v.merge(m, on=["fold", *SETTING], how="left")
    orig = m[m.family == "original"].set_index("fold")
    v["d_f1_val"] = v.macro_f1_pooled7_inner_val - v.fold.map(orig.macro_f1_pooled7_inner_val)
    v["d_accuracy_val"] = v.accuracy_pooled_inner_val - v.fold.map(orig.accuracy_pooled_inner_val)
    v["lci_val_below_tau_k"] = v.lci_val < v.tau_k
    v["silent_break_val"] = v.lci_val_below_tau_k & (v.d_f1_val >= -tol)
    return v


def validation_summary(per_fold: pd.DataFrame) -> pd.DataFrame:
    """Per setting over folds: mean / sd / min LCI_val, mean worst-subject LCI_val, F1_val, the number of folds
    below tau_k and with a silent break, the share of all inner-val subjects below tau_k, size."""
    rows = []
    for key, d in per_fold.groupby(SETTING, sort=False, dropna=False):
        rows.append({**dict(zip(SETTING, key)), "tensor": d.tensor.iloc[0], "n_folds": len(d),
                     "size_kb": d.size_kb.mean(), "size_kb_with_bitmask": d.size_kb_with_bitmask.mean(),
                     "lci_val_mean": d.lci_val.mean(), "lci_val_sd": d.lci_val.std(ddof=1) if len(d) > 1 else np.nan,
                     "lci_val_min": d.lci_val.min(), "worst_subject_lci_val_mean": d.worst_subject_lci_val.mean(),
                     "tau_k_mean": d.tau_k.mean(), "n_folds_lci_val_below_tau_k": int(d.lci_val_below_tau_k.sum()),
                     "share_subjects_below_tau_k": float((d.share_subjects_below_tau_k * d.n_subjects).sum()
                                                         / d.n_subjects.sum()),
                     "f1_val_mean": d.macro_f1_pooled7_inner_val.mean(), "d_f1_val_mean": d.d_f1_val.mean(),
                     "accuracy_val_mean": d.accuracy_pooled_inner_val.mean(), "d_accuracy_val_mean": d.d_accuracy_val.mean(),
                     "n_folds_silent_break": int(d.silent_break_val.sum())})
    return pd.DataFrame(rows)


def add_deltas(pooled: pd.DataFrame, metrics: Sequence[str] = ("accuracy", "non_stair_macro_f1", "aopc")) -> pd.DataFrame:
    """d_<metric> = the subject's value minus its value under the original model (same pooling), so the
    older - general gap of d_<metric> says whether compression costs one group more."""
    out = pooled.copy()
    orig = out[out.family == "original"].set_index("subject")
    for m in metrics:
        if m in out:
            out[f"d_{m}"] = out[m] - out.subject.map(orig[m])
    return out


def headline_table(fa: pd.DataFrame, gs: pd.DataFrame, safe: pd.DataFrame, val: pd.DataFrame, sizes: pd.DataFrame,
                   tau_global: float, tol: float = DIAG["silent_tol"]) -> pd.DataFrame:
    """One row per setting: size, validation, test accuracy / F1 / LCI per group, gaps with CIs (incl. the
    walking-aid split and the gap of the accuracy / F1 drop), FA-LCI with CI, safe / unsafe rates at tau_k
    (subject unit) and silent_break_test = FA-LCI < tau_global with the accuracy drop <= tol in both groups."""
    rows = []
    gsi = gs.set_index([*SETTING, "metric"])
    for _, f in fa.iterrows():
        key = tuple(f[c] for c in SETTING)
        r = {**dict(zip(SETTING, key))}
        s = sizes[(sizes[SETTING] == pd.Series(key, index=SETTING)).all(axis=1)]
        r.update({"size_kb": s.size_kb.mean(), "size_kb_with_bitmask": s.size_kb_with_bitmask.mean()})
        vv = val[(val[SETTING] == pd.Series(key, index=SETTING)).all(axis=1)]
        if len(vv):
            r.update({c: vv.iloc[0][c] for c in ("lci_val_mean", "worst_subject_lci_val_mean", "f1_val_mean",
                                                  "d_f1_val_mean", "n_folds_lci_val_below_tau_k",
                                                  "share_subjects_below_tau_k", "n_folds_silent_break")})
        for metric in ("accuracy", "non_stair_macro_f1", "lci", "d_accuracy", "d_non_stair_macro_f1", "aopc"):
            if (*key, metric) not in gsi.index:
                continue
            g = gsi.loc[(*key, metric)]
            g = g.iloc[0] if isinstance(g, pd.DataFrame) else g
            for grp in ("general", "older", "walking_aid_inferred", "other_older"):
                r[f"{metric}_{grp}"] = g.get(f"mean_{grp}")
            r[f"{metric}_gap"] = g.get("gap_older_minus_general")
            r[f"{metric}_gap_ci_low"], r[f"{metric}_gap_ci_high"] = g.get("gap_ci_low"), g.get("gap_ci_high")
            r[f"{metric}_wa_gap"] = g.get("wa_gap")
            r[f"{metric}_wa_gap_ci_low"], r[f"{metric}_wa_gap_ci_high"] = g.get("wa_gap_ci_low"), g.get("wa_gap_ci_high")
        r.update({"fa_lci": f.fa_lci, "fa_lci_ci_low": f.fa_lci_ci_low, "fa_lci_ci_high": f.fa_lci_ci_high,
                  "worst_group": f.worst_group})
        sf = safe[(safe[["method", "family", "scenario"]] == pd.Series(key[:3], index=["method", "family", "scenario"])
                   ).all(axis=1) & (safe.level == key[3]) & (safe.unit == "subject") & (safe.threshold == "tau_k")]
        if len(sf):
            for grp in ("general", "older"):
                r[f"safe_rate_{grp}"] = float(sf.iloc[0][f"safe_{grp}"])
                r[f"share_below_tau_k_{grp}"] = 1 - r[f"safe_rate_{grp}"]
        r["tau_global"] = tau_global
        acc_ok = all((r.get(f"d_accuracy_{g}") is not None) and r[f"d_accuracy_{g}"] >= -tol for g in TEST_GROUPS)
        r["silent_break_test"] = bool(f.fa_lci < tau_global and acc_ok)
        rows.append(r)
    return pd.DataFrame(rows)


def layer_sensitivity_table(val: pd.DataFrame, head: pd.DataFrame, numel: dict[str, int]) -> pd.DataFrame:
    """Per tensor (only that tensor per-channel 2-bit): LCI drop = 1 - LCI on validation (mean over folds) and
    test (per group), its share of the full ptq_pc2 drop, the older - general LCI gap with CI, the accuracy
    drop per group; ranked by the validation drop."""
    lv = val[val.family == "layer_sensitivity"]
    full_val = val.loc[val.method == "ptq_pc2", "lci_val_mean"]
    full_val = 1 - float(full_val.iloc[0]) if len(full_val) else np.nan
    hd = head.set_index("method")
    full_test = {g: 1 - float(hd.loc["ptq_pc2", f"lci_{g}"]) if "ptq_pc2" in hd.index else np.nan for g in TEST_GROUPS}
    tot = sum(numel.values())
    rows = []
    for _, r in lv.iterrows():
        t = r.tensor
        h = hd.loc[r.method] if r.method in hd.index else pd.Series(dtype=float)
        row = {"tensor": t, "numel": numel.get(t), "share_of_quantized_weights": numel.get(t, np.nan) / tot,
               "lci_val_drop": 1 - r.lci_val_mean, "lci_val_sd": r.lci_val_sd,
               "share_of_full_2bit_val_drop": (1 - r.lci_val_mean) / full_val if full_val else np.nan,
               "d_f1_val": r.d_f1_val_mean}
        for g in TEST_GROUPS:
            row[f"lci_drop_{g}"] = 1 - h.get(f"lci_{g}", np.nan)
            row[f"share_of_full_2bit_drop_{g}"] = row[f"lci_drop_{g}"] / full_test[g] if full_test[g] else np.nan
            row[f"d_accuracy_{g}"] = h.get(f"d_accuracy_{g}", np.nan)
        for c in ("lci_gap", "lci_gap_ci_low", "lci_gap_ci_high", "lci_wa_gap", "fa_lci"):
            row[c] = h.get(c, np.nan)
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("lci_val_drop", ascending=False, kind="stable").reset_index(drop=True)
        out.insert(0, "rank_val", np.arange(1, len(out) + 1))
    return out


def order_settings(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "family" not in df:
        return df
    fam = df.family.map({f: i for i, f in enumerate(FAMILIES)}).fillna(len(FAMILIES))
    lvl = df.level.astype(str).str.extract(r"(\d+)")[0].astype(float).fillna(-1)
    lvl = np.where(df.family == "ptq_per_channel", -lvl, lvl)
    o = pd.DataFrame({"f": fam, "l": lvl, "m": df.method.astype(str)}, index=df.index)
    extra = [c for c in ("fold", "metric", "threshold", "unit", "group", "class_index") if c in df]
    return df.loc[pd.concat([o, df[extra]], axis=1).sort_values(["f", "l", "m", *extra], kind="stable").index
                  ].reset_index(drop=True)


METRICS = ["lci", "shap_corr", "jaccard", "lci768", "lci_k16", "retention", "n_skipped", "accuracy",
           "non_stair_macro_f1", "aopc", "random_aopc", "d_accuracy", "d_non_stair_macro_f1", "d_aopc"]


def analyze(args) -> None:
    src = DIAG_LOW_BIT_SMOKE_DIR if args.smoke else DIAG_LOW_BIT_DIR
    use = {int(a.split("=")[0]): a.split("=", 1)[1] for a in (args.use_machine or [])}
    T, chosen = load_fold_files(src, use, prefix=PREFIX, tables=TABLES)
    ps, models = T["per-subject-metrics"], T["models"]
    if ps.empty:
        raise FileNotFoundError(f"no per-fold low-bit diagnostic files in {src}")
    t0 = time.perf_counter()
    n_boot = args.n_boot
    models = models.copy()
    models["tensor"] = models["tensor"].where(models["tensor"].notna(), None)
    pooled = add_deltas(pool_subjects(ps, [m for m in METRICS if not m.startswith("d_")]))
    gs = group_summary(pooled, METRICS, n_boot)
    fa = fa_lci_table(pooled, n_boot)
    safe = safe_rate_table(ps)
    for g in ("general", "older", "fa_lci"):
        safe[f"share_below_{g}"] = 1 - safe[f"safe_{g}"]
    vpf = validation_per_fold(ps, models)
    vs = validation_summary(vpf)
    tau_global = float(ps.groupby("fold").tau_k.first().min())
    sizes = models.groupby(SETTING, dropna=False)[["size_kb", "size_kb_with_bitmask"]].mean().reset_index()
    head = headline_table(fa, gs, safe, vs, sizes, tau_global)
    numel = models.dropna(subset=["tensor_numel"]).groupby("tensor").tensor_numel.first().astype(int).to_dict()
    files = {"headline-per-setting": head,
             "validation-inner-val-per-fold": vpf, "validation-inner-val-summary": vs,
             "test-fa-lci-bootstrap-ci": fa, "test-group-summary-gaps-bootstrap-ci": gs,
             "test-safe-explanation-rate": safe, "test-stairs-and-pooled-f1-per-group": stairs_table(ps, n_boot),
             "test-per-class-lci-drift": per_class_table(T["per-subject-class-lci"], T["per-group-class-lci"], n_boot),
             "test-per-subject-pooled-over-folds": pooled}
    files = {n: order_settings(d) for n, d in files.items()}
    files["layer-sensitivity-2bit"] = layer_sensitivity_table(vs, head, numel)
    for name, df in files.items():
        R.write_csv(src / f"{APREFIX}-{name}.csv", df)
    R.write_json(src / f"{APREFIX}-analysis-config.json",
                 {"folds_and_machines": chosen, "n_boot": n_boot, "metrics": METRICS, "diag": DIAG,
                  "tau_k_by_fold": ps.groupby("fold").tau_k.first().to_dict(), "tau_global": tau_global,
                  "silent_break_definitions": {
                      "validation": "per fold: LCI_val < tau_k and pooled 7-class F1_val drop <= 1 pt vs the original",
                      "test": "FA-LCI < tau_global (min_k tau_k) and subject-mean accuracy drop <= 1 pt in both groups"},
                  "smoke": args.smoke, **R.git_info(), "code_sha256": R.code_sha256(), "analyzed_at": R.now(),
                  "seconds": time.perf_counter() - t0})
    show = ["method", "level", "size_kb_with_bitmask", "lci_val_mean", "f1_val_mean", "lci_general", "lci_older",
            "lci_gap", "lci_gap_ci_low", "lci_gap_ci_high", "fa_lci", "accuracy_general", "accuracy_older",
            "d_accuracy_gap", "share_below_tau_k_older", "silent_break_test"]
    h = files["headline-per-setting"]
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(h[h.family != "layer_sensitivity"][[c for c in show if c in h]].to_string(index=False))
        print(files["layer-sensitivity-2bit"][["rank_val", "tensor", "numel", "lci_val_drop", "lci_drop_general",
                                               "lci_drop_older", "lci_gap"]].to_string(index=False))
    print(f"\nfolds {chosen}; analysis {time.perf_counter() - t0:.1f} s")
    for name in files:
        print(f"saved {src / f'{APREFIX}-{name}.csv'}")


# ------------------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--folds", type=int, nargs="+")
    mode.add_argument("--analyze", action="store_true")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--machine-name", default=R.machine_name())
    ap.add_argument("--noise-floor-machine", default=None,
                    help="whose noise-floor files give tau_k and the ceilings (default: fold 0-1 nakhoul-3050, 2-4 uni-2080)")
    ap.add_argument("--use-machine", nargs="*", default=None, help="--analyze: k=machine for a fold run on two machines")
    ap.add_argument("--no-layer-sensitivity", action="store_true")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.analyze:
        return analyze(args)
    frozen = json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))
    assert frozen["model"] == "tinyhar", frozen["model"]
    harth, test = load_harth_windows(), load_test_windows()
    counts = pd.Series(harth["harth_test"].subject).value_counts().to_dict()
    folds = load_folds(harth_test_counts={s: int(c) for s, c in counts.items()})
    t0 = time.perf_counter()
    for k in args.folds:
        run_fold(args, k, harth, test, folds, frozen)
    print(f"\nall folds {args.folds}: {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
