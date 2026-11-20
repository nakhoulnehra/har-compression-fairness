"""Step 8: noise floor per fold (plan Section 3 Step 8, Section 2.7).

What counts as a real explanation change, measured before any compression result is seen:
    retrain noise   the frozen TinyHAR retrained with seeds 1, 2, 3 (seed 0 = the main model, the
                    model-selection checkpoint) -> LCI(seed 0, seed s), SHAP seed 0 for both
    SHAP noise      the seed-0 model with SHAP seeds 1, 2 vs SHAP seed 0 -> LCI(shap 0, shap s)
on the fold's inner-validation, general (held-out HARTH) and older (HAR70+) explain windows.
LCI: per subject, 48 segments, k = 10, omega = 0.5; target = the seed-0 model's predicted class;
GradientExplainer nsamples 200 with the fold's 100-window background.

    tau_k = min over the 3 retrain seeds of LCI_val(seed 0, seed s), where LCI_val = the mean
            per-subject LCI over the fold's 3 inner-validation HARTH subjects (the gate's metric).
HAR70+ never enters tau_k (the gate stays leak-free); the general/older ranges are reported only.
Acceptance (Step 8): per group, every SHAP-noise value is above every retrain-noise value; tau_k is
written and reported next to Tulla's 0.85. The global tau = min_k tau_k is taken in Step 11.

Retraining uses the frozen setting exactly (har-frozen-model-config.json: Adam lr 1e-3, weight decay
1e-4, batch 256, <= 30 epochs, early stopping patience 5 on inner-val pooled macro-F1) and the fold's
normalisation statistics. Retrained weights are cached in data/processed/har/checkpoints/har-noise-floor/
(reused when the settings hash matches), SHAP arrays in data/processed/har/shap-cache/.

Usage (from the repo root, inside .venv):
    python -m har.run_noise_floor --folds 0 --device cuda --machine-name nakhoul-3050
    python -m har.run_noise_floor --folds 0 --smoke              # 1 epoch, 3 windows per subject, nsamples 20

Writes results/har-noise-floor/ (results/har-noise-floor-smoke/ with --smoke), per fold k and machine:
    har-noise-floor-tinyhar-fold<k>-<machine>-per-subject-lci.csv     one row per (noise, seed, group, subject)
    har-noise-floor-tinyhar-fold<k>-<machine>-group-summary.csv       per (group, noise): the 3 or 2 group means
    har-noise-floor-tinyhar-fold<k>-<machine>-tau.csv                 tau_k (read by the XGP gate and pilot)
    har-noise-floor-tinyhar-fold<k>-<machine>-retrained-models.csv    seeds 1-3: epochs, inner-val F1, time
    ...-config.json and logs/...log
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS; must precede CUDA init

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import torch

from . import runlog as R
from .attributions import mean_lci, shap_cached, subject_lci
from .data import cache_key, find_windows, load_harth_windows, load_test_windows, require_harth
from .folds import channel_stats, load_folds
from .models import build_model
from .paths import (BACKGROUND_CSV, EXPLAIN_WINDOWS_CSV, MODEL_SELECTION_DIR, NOISE_FLOOR_CKPT_DIR, NOISE_FLOOR_DIR,
                    NOISE_FLOOR_SMOKE_CKPT_DIR, NOISE_FLOOR_SMOKE_DIR, SHAP_CACHE_DIR)
from .run_model_selection import subsample
from .run_sanity_walking_aid import load_fold_model
from .train import TrainConfig, predict_proba, pooled_macro_f1, set_seed, train_model

FROZEN_CONFIG = MODEL_SELECTION_DIR / "har-frozen-model-config.json"
RETRAIN_SEEDS = (1, 2, 3)
SHAP_SEEDS = (1, 2)
NSAMPLES, K, OMEGA, TULLA_TAU = 200, 10, 0.5, 0.85
GROUPS = ("inner_val", "general", "older")
SMOKE = {"epochs": 1, "n_train": 600, "n_val": 200, "per_subject": 3, "n_older_subjects": 3, "nsamples": 20}


# ------------------------------------------------------------------------------ summaries (tested)

def tau_from_rows(df: pd.DataFrame) -> float:
    """tau_k = min over retrain seeds of the inner-validation subject mean of LCI(seed 0, seed s)."""
    r = df[(df.noise == "retrain") & (df.group == "inner_val")]
    if r.empty:
        raise ValueError("no retrain-noise rows on inner-validation subjects")
    return float(r.groupby("seed").lci.mean().min())


def group_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per (group, noise): the per-seed subject means (3 retrain, 2 SHAP), their mean / min / max, and per
    group whether every SHAP-noise value is above every retrain-noise value (Step 8 acceptance)."""
    rows = []
    for (group, noise), d in df.groupby(["group", "noise"], sort=False):
        v = d.groupby("seed").lci.mean()
        rows.append({"group": group, "noise": noise, "n_values": len(v), "values": ";".join(f"{x:.4f}" for x in v),
                     "seeds": ";".join(str(s) for s in v.index), "mean": float(v.mean()), "min": float(v.min()),
                     "max": float(v.max()), "n_subjects": int(d.subject.nunique())})
    out = pd.DataFrame(rows)
    check = {}
    for group, d in out.groupby("group"):
        r, s = d[d.noise == "retrain"], d[d.noise == "shap"]
        check[group] = bool(len(r) and len(s) and s["min"].iloc[0] > r["max"].iloc[0])
    out["shap_above_retrain"] = out.group.map(check)
    return out


# ------------------------------------------------------------------------------ helpers

def explain_sets(fold, harth: dict, test: dict, smoke: bool) -> dict:
    """{group: (Windows split, split name, window IDs, subjects)} of the fold's explain windows."""
    ex = pd.read_csv(EXPLAIN_WINDOWS_CSV, dtype={"subject": str})
    spec = {"inner_val": ("harth_test", harth["harth_test"], ex.subject.isin(fold.val)),
            "general": ("harth_test", test["harth_test"], ex.subject.isin(fold.test_general)),
            "older": ("har70plus_test", test["har70plus_test"], ex.subject.isin(fold.test_older))}
    out = {}
    for g, (split, w, sel) in spec.items():
        d = ex[(ex.split == split) & sel].sort_values(["subject", "pick_order"])
        if smoke:
            d = d.groupby("subject").head(SMOKE["per_subject"])
            if g == "older":
                d = d[d.subject.isin(sorted(d.subject.unique())[:SMOKE["n_older_subjects"]])]
        idx = find_windows(w, split, d.window_id.tolist())
        assert (w.subject[idx] == d.subject.to_numpy()).all()
        out[g] = (w.X[idx], d.window_id.tolist(), d.subject.to_numpy())
    assert set(out["inner_val"][2]) <= set(fold.val) and set(out["general"][2]) <= set(fold.test_general)
    return out


def fold_background(harth: dict, k: int) -> torch.Tensor:
    ids = pd.read_csv(BACKGROUND_CSV).query("fold == @k").window_id.tolist()
    return torch.from_numpy(harth["harth_train"].X[find_windows(harth["harth_train"], "harth_train", ids)])


def train_or_load(seed: int, k: int, fold, arch: dict, cfg: TrainConfig, tr, va, mean, std, args, ckpt_dir) -> tuple:
    settings = {"model": "tinyhar", "architecture": arch, "train_config": cfg.to_dict(), "seed": seed, "fold": k,
                "train_subjects": list(fold.train), "validation_subjects": list(fold.val), "smoke": args.smoke,
                "smoke_sizes": SMOKE if args.smoke else None, "window_cache_key": cache_key(),
                "n_train": len(tr), "n_val": len(va)}
    h = R.settings_hash(settings)
    path = ckpt_dir / f"har-noise-floor-tinyhar-fold{k}-seed{seed}.pt"
    if path.exists():
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck.get("settings_hash") == h:
            net = build_model("tinyhar", arch, ck["norm_mean"], ck["norm_std"])
            net.load_state_dict(ck["state_dict"])
            print(f"  seed {seed}: reusing {path.name} (same settings)")
            return net.eval(), ck["info"], path, True
        print(f"  seed {seed}: {path.name} has different settings; retraining")
    print(f"  seed {seed}: training (<= {cfg.max_epochs} epochs, patience {cfg.patience})")
    set_seed(seed)
    torch.manual_seed(seed)
    net = build_model("tinyhar", arch, mean, std)
    info = train_model(net, tr, va, cfg, seed, device=args.device)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": "tinyhar", "architecture": arch, "state_dict": {k_: v.cpu() for k_, v in net.state_dict().items()},
                "norm_mean": mean, "norm_std": std, "settings_hash": h, "settings": settings, "info": info}, path)
    return net.eval(), info, path, False


# ------------------------------------------------------------------------------ one fold

def run_fold(args, k: int, harth: dict, test: dict, folds, frozen: dict, dev: dict, out, ckpt_dir) -> None:
    t_fold = time.perf_counter()
    fold = folds[k]
    machine = args.machine_name
    base = f"har-noise-floor-tinyhar-fold{k}-{machine}"
    paths = {n: out / f"{base}-{n}.csv" for n in ("per-subject-lci", "group-summary", "tau", "retrained-models")}
    cfg_path, log = out / f"{base}-config.json", out / "logs" / f"{base}.log"
    log.unlink(missing_ok=True)
    with R.tee_to(log):
        print(f"=== noise floor fold {k} | machine {machine} | device {dev['device']} {dev['gpu_name']} | "
              f"torch {dev['torch_version']} | started {R.now()} | {' '.join(sys.argv)}")
        tr = harth["harth_train"].for_subjects(fold.train)
        va = harth["harth_test"].for_subjects(fold.val)
        require_harth(tr, va)
        assert not set(tr.subjects) & set(fold.test_general) and not set(va.subjects) & set(fold.test_general)
        cfg = TrainConfig(**frozen["train_config"])
        nsamples = SMOKE["nsamples"] if args.smoke else NSAMPLES
        if args.smoke:
            tr, va = subsample(tr, SMOKE["n_train"], 0), subsample(va, SMOKE["n_val"], 1)
            cfg.max_epochs = SMOKE["epochs"]
        print(f"  train {len(tr)} windows ({len(fold.train)} subjects), inner val {len(va)} ({list(fold.val)}), "
              f"test general {list(fold.test_general)} + {len(fold.test_older)} HAR70+")

        net0, ck0 = load_fold_model(k, frozen["architecture"])
        mean, std = channel_stats(harth["harth_train"], fold)
        assert np.allclose(mean, net0.norm.mean.flatten().numpy(), atol=1e-6) and \
            np.allclose(std, net0.norm.std.flatten().numpy(), atol=1e-6), "seed-0 checkpoint has other normalisation stats"
        arch = dict(net0.cfg)                       # the checkpoint's architecture (checked against the frozen config)
        print(f"  seed 0: {ck0}")
        models, info_rows, t_train = {0: net0.to(args.device).eval()}, [], time.perf_counter()
        for s in RETRAIN_SEEDS:
            net, info, path, reused = train_or_load(s, k, fold, arch, cfg, tr, va, mean, std, args, ckpt_dir)
            models[s] = net.to(args.device).eval()
            f1 = pooled_macro_f1(va.y, predict_proba(models[s], va.X, args.device).argmax(1))
            info_rows.append({"fold": k, "seed": s, "best_epoch": info["best_epoch"], "epochs_run": info["epochs_run"],
                              "best_val_macro_f1_pooled7": info["best_val_macro_f1_pooled7"], "val_macro_f1_pooled7": f1,
                              "train_seconds": info["train_seconds"], "reused_checkpoint": reused,
                              "checkpoint": str(path), "machine_name": machine, "device": dev["device"]})
        f1_0 = pooled_macro_f1(va.y, predict_proba(net0, va.X, args.device).argmax(1))
        info_rows.insert(0, {"fold": k, "seed": 0, "val_macro_f1_pooled7": f1_0, "checkpoint": str(ck0),
                             "machine_name": machine, "device": dev["device"]})
        R.write_csv(paths["retrained-models"], pd.DataFrame(info_rows))
        train_s = time.perf_counter() - t_train

        background = fold_background(harth, k).to(args.device)
        sets = explain_sets(fold, harth, test, args.smoke)
        t_shap, A = time.perf_counter(), {}
        for g, (X, ids, subj) in sets.items():
            Xd = torch.from_numpy(X).to(args.device)
            target = predict_proba(net0, X, args.device).argmax(1)
            runs = [(0, 0)] + [(0, s) for s in SHAP_SEEDS] + [(s, 0) for s in RETRAIN_SEEDS]
            for model_seed, shap_seed in runs:
                print(f"  SHAP {g}: model seed {model_seed}, SHAP seed {shap_seed}, {len(X)} windows, nsamples {nsamples}")
                A[(g, model_seed, shap_seed)] = shap_cached(models[model_seed], background, Xd, target, ids, shap_seed,
                                                            nsamples, SHAP_CACHE_DIR / ("smoke" if args.smoke else ""),
                                                            f"har-noise-floor-fold{k}-{g}")
        shap_s = time.perf_counter() - t_shap

        rows = []
        for g, (X, ids, subj) in sets.items():
            ref = A[(g, 0, 0)]
            for noise, seeds, key in (("retrain", RETRAIN_SEEDS, lambda s: (g, s, 0)), ("shap", SHAP_SEEDS, lambda s: (g, 0, s))):
                for s in seeds:
                    for r in subject_lci(ref, A[key(s)], subj, k=K, omega=OMEGA):
                        rows.append({"fold": k, "noise": noise, "seed": s, "group": g, **r, "machine_name": machine})
        df = pd.DataFrame(rows)
        lead = ["fold", "noise", "seed", "group", "subject", "lci", "shap_corr", "jaccard", "n_skipped", "n_samples"]
        df = df[lead + [c for c in df.columns if c not in lead]]
        R.write_csv(paths["per-subject-lci"], df)
        summ = group_summary(df)
        summ.insert(0, "fold", k)
        summ["machine_name"] = machine
        R.write_csv(paths["group-summary"], summ)
        tau = tau_from_rows(df)
        val_means = df[(df.noise == "retrain") & (df.group == "inner_val")].groupby("seed").lci.mean()
        checks = summ.drop_duplicates("group").set_index("group").shap_above_retrain.to_dict()
        tau_row = {"fold": k, "tau_k": tau, "tau_seed": int(val_means.idxmin()),
                   **{f"lci_val_retrain_seed{s}": float(v) for s, v in val_means.items()},
                   "inner_val_subjects": ";".join(fold.val), "tulla_tau": TULLA_TAU, "tau_k_below_tulla": tau < TULLA_TAU,
                   **{f"shap_above_retrain_{g}": checks.get(g) for g in GROUPS},
                   "nsamples": nsamples, "smoke": args.smoke, "machine_name": machine, "device": dev["device"]}
        R.write_csv(paths["tau"], pd.DataFrame([tau_row]))
        wall = time.perf_counter() - t_fold
        R.write_json(cfg_path, {"command_line": sys.argv, "machine_name": machine, "device": dev, **R.git_info(),
                                "code_sha256": R.code_sha256(), "fold": k, "smoke": args.smoke,
                                "smoke_sizes": SMOKE if args.smoke else None, "retrain_seeds": RETRAIN_SEEDS,
                                "shap_seeds": SHAP_SEEDS, "nsamples": nsamples, "k": K, "omega": OMEGA,
                                "train_config": cfg.to_dict(), "architecture": arch, "seed0_checkpoint": str(ck0),
                                "n_explain": {g: len(v[1]) for g, v in sets.items()}, "tau_k": tau,
                                "seconds": {"training": train_s, "shap": shap_s, "wall": wall}, "finished_at": R.now()})
        print("\n" + summ[["group", "noise", "values", "mean", "min", "max", "shap_above_retrain"]].to_string(index=False))
        print(f"\nfold {k}: tau_k = {tau:.4f} (retrain seed {tau_row['tau_seed']}; Tulla's tau = {TULLA_TAU}); "
              f"training {train_s:.1f} s, SHAP {shap_s:.1f} s, wall {wall:.1f} s")
        for p in [*paths.values(), cfg_path, log]:
            print(f"saved {p}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", type=int, nargs="+", required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--threads", type=int, default=None, help="torch.set_num_threads (plan: 8 Nakhoul, 10 Aya)")
    ap.add_argument("--machine-name", default=R.machine_name())
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    dev = R.device_info(args.device)
    frozen = json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))
    assert frozen["model"] == "tinyhar", frozen["model"]
    harth, test = load_harth_windows(), load_test_windows()
    counts = pd.Series(harth["harth_test"].subject).value_counts().to_dict()
    folds = load_folds(harth_test_counts={s: int(c) for s, c in counts.items()})
    out = NOISE_FLOOR_SMOKE_DIR if args.smoke else NOISE_FLOOR_DIR
    ckpt_dir = NOISE_FLOOR_SMOKE_CKPT_DIR if args.smoke else NOISE_FLOOR_CKPT_DIR
    t0 = time.perf_counter()
    for k in args.folds:
        run_fold(args, k, harth, test, folds, frozen, dev, out, ckpt_dir)
    print(f"\nall folds {args.folds}: {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
