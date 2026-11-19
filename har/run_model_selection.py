"""Plan Step 4: model choice. Tune the candidates on fold 0, then train the chosen setting on every fold.

Usage (from the repo root; build the window cache first with `python -m har.build_windows`):
    python -m har.run_model_selection --stage tune  --device cuda --models cnn1d tinyhar deepconvlstm hart xgboost
    python -m har.run_model_selection --stage final --device cuda --models cnn1d tinyhar deepconvlstm hart xgboost --folds 0 1 2
    python -m har.run_model_selection --stage all   --device cpu --smoke        # 1 fold, 1 epoch, a few hundred windows

Stages
    tune   every setting of the tuning grid (4 per model), trained on fold 0's training subjects and
           scored on fold 0's 3 validation subjects ONLY (fold 0's test subjects are never used).
    final  the best tuning setting per model, trained on each requested fold (early stopping on that
           fold's validation subjects), evaluated on that fold's held-out HARTH test subjects.
    all    tune, then final.
HAR70+ is never loaded: only the HARTH splits of the window cache are read (data.load_harth_windows).

Every unit (model x setting x fold) writes, atomically and only when it is complete:
    <out>/<stage dir>/har-<stage>-<model>[-<setting>]-fold<k>-<machine>.csv     one row, all metrics + run info
                                                         ...-config.json        exact settings + command line
                                                         ...-history.csv        per-epoch training curve
                                                         ...-per-subject.csv    (final) per-subject metrics
                                                         ...-confusion.csv      (final) 7x7 confusion matrix
    <out>/logs/<stage>-<model>[-<setting>]-fold<k>-<machine>.log                full terminal output
    data/processed/har/checkpoints/<out name>/...pt (.ubj for XGBoost)          trained weights (git-ignored)
A unit whose result row exists with the same settings hash (from any machine) is skipped, so an
interrupted run can simply be restarted; a unit with different settings is refused unless --overwrite.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS; must precede CUDA init

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import runlog as R
from .data import CLASS_NAMES, Windows, cache_key, load_harth_windows, require_harth
from .folds import FoldSplit, channel_stats, load_folds, stratified_pick, tuning_sets
from .models import ALL_MODELS, DEEP_MODELS, SOURCES, build_model, n_params, resolve_config
from .paths import (MODEL_SELECTION_CKPT_DIR, MODEL_SELECTION_DIR, MODEL_SELECTION_SMOKE_CKPT_DIR,
                    MODEL_SELECTION_SMOKE_DIR)
from .train import TrainConfig, evaluate, predict_proba, set_seed, train_model, train_xgboost, xgb_predict_proba

SEED = 0
TUNING_FOLD = 0
# Equal tuning budget (plan 2.9): 4 settings per candidate. Width 1.0 = the paper's sizes (har/models.py).
DEEP_GRID = [{"lr": 1e-3, "width": 1.0}, {"lr": 1e-3, "width": 0.5}, {"lr": 3e-4, "width": 1.0}, {"lr": 3e-4, "width": 0.5}]
# depth 3 + eta 0.1 = [HARTH Sec 4.1, p12]; depth 3 + eta 0.5 = [Ustad Sec 2.3.3, p6]; depth 6 = our choice.
XGB_GRID = [{"max_depth": 3, "learning_rate": 0.1}, {"max_depth": 3, "learning_rate": 0.5},
            {"max_depth": 6, "learning_rate": 0.1}, {"max_depth": 6, "learning_rate": 0.5}]
SMOKE = {"epochs": 1, "n_train": 300, "n_val": 200, "n_test": 200, "xgb_trees": 5}
COMPANION_SUFFIXES = ("-history.csv", "-per-subject.csv", "-confusion.csv")


def grid(model: str) -> list[dict]:
    return XGB_GRID if model == "xgboost" else DEEP_GRID


def setting_id(model: str, hp: dict) -> str:
    if model == "xgboost":
        return f"depth{hp['max_depth']}-eta{hp['learning_rate']:g}"
    return f"lr{hp['lr']:.0e}-width{hp['width']:g}".replace("e-0", "e-")


def dirs(smoke: bool) -> tuple[Path, Path]:
    return (MODEL_SELECTION_SMOKE_DIR, MODEL_SELECTION_SMOKE_CKPT_DIR) if smoke else (MODEL_SELECTION_DIR, MODEL_SELECTION_CKPT_DIR)


def unit_name(stage: str, model: str, sid: str | None, fold: int, machine: str) -> str:
    return f"har-{stage}-{model}{'-' + sid if sid else ''}-fold{fold}-{machine}"


def unit_paths(out: Path, ckpt: Path, stage: str, model: str, sid: str | None, fold: int, machine: str) -> dict:
    base = unit_name(stage, model, sid, fold, machine)
    d = out / ("tuning" if stage == "tune" else "final")
    log = out / "logs" / f"{stage}-{model}{'-' + sid if sid else ''}-fold{fold}-{machine}.log"
    return {"row": d / f"{base}.csv", "config": d / f"{base}-config.json", "history": d / f"{base}-history.csv",
            "per_subject": d / f"{base}-per-subject.csv", "confusion": d / f"{base}-confusion.csv", "log": log,
            "checkpoint": ckpt / (base + (".ubj" if model == "xgboost" else ".pt"))}


def find_done(out: Path, stage: str, model: str, sid: str | None, fold: int) -> list[pd.DataFrame]:
    d = out / ("tuning" if stage == "tune" else "final")
    pattern = unit_name(stage, model, sid, fold, "*") + ".csv"
    rows = []
    for p in sorted(d.glob(pattern)):
        if p.name.endswith(COMPANION_SUFFIXES):
            continue
        df = pd.read_csv(p, dtype={"machine_name": str})
        if len(df) == 1 and "status" in df:
            df["row_file"] = str(p)
            rows.append(df)
    return rows


def subsample(w: Windows, n: int, seed: int) -> Windows:
    """Class-stratified subset for --smoke runs (every class present when possible)."""
    if len(w) <= n:
        return w
    return w.select(np.sort(stratified_pick(w.y, n, np.random.default_rng(seed))))


def best_setting(out: Path, model: str) -> tuple[dict, str, float]:
    """Highest fold-0 validation F1 among the model's tuning settings (ties -> grid order)."""
    best = None
    for hp in grid(model):
        sid = setting_id(model, hp)
        done = [r for r in find_done(out, "tune", model, sid, TUNING_FOLD) if r.status.iloc[0] == "done"]
        if not done:
            raise RuntimeError(f"tuning of {model} setting {sid} not found in {out / 'tuning'}: run "
                               f"`--stage tune` (or merge the other machine's results) first")
        f1 = float(done[0].best_val_macro_f1_pooled7.iloc[0])
        if best is None or f1 > best[2]:
            best = (hp, sid, f1)
    return best


# ---------------------------------------------------------------------------------------- units

def run_unit(args, stage: str, model: str, hp: dict, fold: FoldSplit, folds: list[FoldSplit],
             data: dict[str, Windows], machine: str, dev: dict, saved: list[str], tuning_f1: float | None = None) -> str:
    out, ckpt = dirs(args.smoke)
    sid = setting_id(model, hp) if stage == "tune" else None
    paths = unit_paths(out, ckpt, stage, model, sid, fold.fold, machine)
    cfg = TrainConfig(**({"lr": hp["lr"]} if model != "xgboost" else {}))
    if args.smoke:
        cfg.max_epochs = SMOKE["epochs"]
    arch = resolve_config(model, hp["width"]) if model != "xgboost" else {**hp, **({"n_estimators": SMOKE["xgb_trees"]} if args.smoke else {})}

    # ---- subjects: tuning = fold 0 train / its 3 validation subjects only; final = fold k
    if stage == "tune":
        sets = tuning_sets(folds[TUNING_FOLD])
        train_subj, val_subj, test_subj = sets["tuning_train"], sets["tuning_validation"], ()
        assert not (set(train_subj) | set(val_subj)) & set(sets["never_used"]), "tuning touches fold-0 test subjects"
        assert set(val_subj) == set(folds[TUNING_FOLD].val) and len(val_subj) == 3
    else:
        train_subj, val_subj, test_subj = fold.train, fold.val, fold.test_general
    assert not (set(train_subj) & set(val_subj)) and not (set(train_subj) | set(val_subj)) & set(test_subj)
    assert not (set(train_subj) | set(val_subj) | set(test_subj)) & set(fold.test_older), "HAR70+ subject in a HARTH set"

    settings = {"stage": stage, "model": model, "setting_id": sid, "hyperparameters": hp, "architecture": arch,
                "train_config": cfg.to_dict(), "seed": SEED, "fold": fold.fold, "smoke": args.smoke,
                "smoke_sizes": SMOKE if args.smoke else None, "train_subjects": list(train_subj),
                "validation_subjects": list(val_subj), "test_subjects": list(test_subj),
                "train_windows": "harth_train (50% overlap)", "validation_windows": "harth_test (no overlap)",
                "test_windows": "harth_test (no overlap)" if test_subj else None, "window_cache_key": cache_key(),
                "selected_from_tuning_f1": tuning_f1}
    h = R.settings_hash(settings)
    label = f"{stage} {model}{' ' + sid if sid else ''} fold {fold.fold}"
    for prev in find_done(out, stage, model, sid, fold.fold):
        if prev.status.iloc[0] == "done" and prev.settings_hash.iloc[0] == h:
            print(f"  [skip] {label}: already done on {prev.machine_name.iloc[0]} ({prev.row_file.iloc[0]})")
            return "skipped"
        if not args.overwrite:
            raise RuntimeError(f"{label}: {prev.row_file.iloc[0]} exists with different settings; "
                               "use --overwrite to replace it (or move it away)")

    started, t0 = R.now(), time.perf_counter()
    paths["log"].unlink(missing_ok=True)
    with R.tee_to(paths["log"]):
        print(f"=== {label} | machine {machine} | device {dev['device']} {dev['gpu_name']} | torch {dev['torch_version']} "
              f"| started {started}")
        print(f"    train subjects {list(train_subj)}\n    validation subjects {list(val_subj)}"
              + (f"\n    test subjects {list(test_subj)}" if test_subj else "\n    test subjects: none (tuning)"))
        tr = data["harth_train"].for_subjects(train_subj)
        va = data["harth_test"].for_subjects(val_subj)
        te = data["harth_test"].for_subjects(test_subj) if test_subj else None
        require_harth(tr, va, *([te] if te is not None else []))
        if args.smoke:
            tr, va = subsample(tr, SMOKE["n_train"], SEED), subsample(va, SMOKE["n_val"], SEED + 1)
            te = subsample(te, SMOKE["n_test"], SEED + 2) if te is not None else None
        print(f"    windows: train {len(tr)}, validation {len(va)}" + (f", test {len(te)}" if te is not None else ""))
        set_seed(SEED)
        if model == "xgboost":
            booster, info = train_xgboost(tr, va, arch, SEED, device=dev["device"])
            proba = lambda X: xgb_predict_proba(booster, X)
            npar = None  # not comparable to network weights; the tree count is recorded instead
            info["n_trees"] = int(info["best_epoch"]) + 1
            paths["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
            booster.save_model(str(paths["checkpoint"]))
        else:
            mean, std = channel_stats(data["harth_train"], fold, subjects=list(train_subj)) if stage == "final" else \
                channel_stats(data["harth_train"], folds[TUNING_FOLD])
            torch.manual_seed(SEED)
            net = build_model(model, arch, mean, std)
            npar = n_params(net)
            print(f"    {model}: {npar:,} parameters; architecture {arch}")
            info = train_model(net, tr, va, cfg, SEED, device=dev["device"])
            proba = lambda X: predict_proba(net, X, dev["device"])
            paths["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model, "architecture": arch, "state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
                        "norm_mean": mean, "norm_std": std, "settings_hash": h, "settings": settings}, paths["checkpoint"])
        val_eval = evaluate(va.y, proba(va.X).argmax(1), va.subject)
        row = {"stage": stage, "model": model, "setting_id": sid or setting_id(model, hp), "fold": fold.fold,
               "status": "done", "settings_hash": h, "seed": SEED, "smoke": args.smoke,
               **{f"hp_{k}": v for k, v in hp.items()}, "n_params": npar, "n_trees": info.get("n_trees"),
               "best_epoch": info["best_epoch"], "epochs_run": info["epochs_run"],
               "best_val_macro_f1_pooled7": info["best_val_macro_f1_pooled7"],
               **{f"val_{k}": v for k, v in val_eval.items() if k not in ("per_subject", "confusion")}}
        if te is not None:
            test_eval = evaluate(te.y, proba(te.X).argmax(1), te.subject)
            row.update({f"test_{k}": v for k, v in test_eval.items() if k not in ("per_subject", "confusion")})
            row["selected_from_tuning_val_f1"] = tuning_f1
            ps = pd.DataFrame(test_eval["per_subject"])
            ps.insert(0, "fold", fold.fold); ps.insert(0, "model", model); ps.insert(0, "machine_name", machine)
            R.write_csv(paths["per_subject"], ps)
            cm = pd.DataFrame(test_eval["confusion"], index=CLASS_NAMES, columns=CLASS_NAMES).rename_axis("true\\pred").reset_index()
            R.write_csv(paths["confusion"], cm)
            print(f"    TEST (held-out HARTH): macro-F1 pooled {test_eval['macro_f1_pooled7']:.4f}, "
                  f"accuracy {test_eval['accuracy_pooled']:.4f}, non-stair F1 per subject "
                  f"{test_eval['non_stair_macro_f1_subject_mean']:.4f}, stairs F1 pooled {test_eval['stairs_f1_pooled_mean']:.4f}")
        run_s = time.perf_counter() - t0
        row.update({"train_seconds": info["train_seconds"], "run_seconds": run_s, "started_at": started,
                    "finished_at": R.now(), "machine_name": machine, **dev, **R.git_info(), "code_sha256": R.code_sha256(),
                    "train_subjects": ";".join(train_subj), "validation_subjects": ";".join(val_subj),
                    "test_subjects": ";".join(test_subj), "n_train_windows": len(tr), "n_val_windows": len(va),
                    "n_test_windows": len(te) if te is not None else 0, "source": SOURCES[model][0],
                    "source_check": SOURCES[model][1], "checkpoint": str(paths["checkpoint"]), "log": str(paths["log"]),
                    "config_file": str(paths["config"]), "command_line": " ".join(sys.argv)})
        R.write_csv(paths["history"], pd.DataFrame(info["history"]))
        R.write_json(paths["config"], {"settings": settings, "settings_hash": h, "command_line": sys.argv,
                                       "machine_name": machine, "device": dev, **R.git_info(),
                                       "code_sha256": row["code_sha256"], "started_at": started,
                                       **({"xgboost_params": info["booster_params"]} if model == "xgboost" else {})})
        R.write_csv(paths["row"], pd.DataFrame([row]))     # written last: marks the unit as done
        print(f"    done in {run_s:.1f}s; val F1 {info['best_val_macro_f1_pooled7']:.4f} (best epoch {info['best_epoch'] + 1})")
        files = [paths[k] for k in ("row", "config", "history", "per_subject", "confusion", "checkpoint", "log") if paths[k].exists()]
        for p in files:
            print(f"    saved {p}")
    saved += [str(p) for p in files]
    return "done"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["tune", "final", "all"], default="all")
    ap.add_argument("--models", nargs="+", choices=ALL_MODELS, default=list(ALL_MODELS))
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="folds for --stage final/all")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--smoke", action="store_true", help="fold 0 only, 1 epoch, a few hundred windows; "
                                                         "writes to results/har-model-selection-smoke/")
    ap.add_argument("--overwrite", action="store_true", help="replace existing units whose settings differ")
    ap.add_argument("--machine-name", default=R.machine_name())
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = torch default)")
    args = ap.parse_args()
    if args.smoke:
        args.folds = [0]
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.device == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    dev = R.device_info(args.device)
    out, ckpt = dirs(args.smoke)
    session_log = out / "logs" / f"session-{args.stage}-{args.machine_name}-{R.stamp()}.log"
    saved: list[str] = []
    with R.tee_to(session_log):
        print(f"har.run_model_selection | {' '.join(sys.argv)}")
        print(f"machine {args.machine_name} | device {dev['device']} {dev['gpu_name']} | torch {dev['torch_version']} "
              f"(CUDA {dev['cuda_version'] or 'none'}) | threads {dev['torch_threads']} | git {R.git_info()}")
        data = load_harth_windows()                                  # HARTH only
        counts = pd.Series(data["harth_test"].subject).value_counts().to_dict()
        folds = load_folds(harth_test_counts={s: int(c) for s, c in counts.items()})
        units = []
        if args.stage in ("tune", "all"):
            units += [("tune", m, hp, folds[TUNING_FOLD]) for m in args.models for hp in grid(m)]
        if args.stage in ("final", "all"):
            units += [("final", m, None, folds[k]) for m in args.models for k in args.folds]
        status = {"done": 0, "skipped": 0}
        t0 = time.perf_counter()
        for i, (stage, m, hp, fold) in enumerate(units, 1):
            print(f"\n[{i}/{len(units)}] {stage} {m} fold {fold.fold}  (elapsed {(time.perf_counter() - t0) / 60:.1f} min)")
            tf1 = None
            if stage == "final":
                hp, sid, tf1 = best_setting(out, m)
                print(f"  best tuning setting for {m}: {sid} (fold-0 validation F1 {tf1:.4f})")
            status[run_unit(args, stage, m, hp, fold, folds, data, args.machine_name, dev, saved, tf1)] += 1
        print(f"\nfinished: {status['done']} units run, {status['skipped']} skipped, "
              f"{(time.perf_counter() - t0) / 60:.1f} min")
        print("files saved in this session:")
        for p in saved:
            print(f"  {p}")
        print(f"  {session_log} (this session's full log)")


if __name__ == "__main__":
    main()
