"""Step 5: baseline gap check on HAR70+ (first HAR70+ use), with the frozen model (plan Section 3
Step 5; Section 2.9 "Sanity check after freezing"; statistics per Section 2.8).

Evaluation only: the frozen TinyHAR's five final fold checkpoints (one per fold, seed 0) are loaded
on CPU and applied unchanged (their stored normalisation) to
  - general: each HARTH subject's non-overlapping test windows, from the one fold that holds it out;
  - older:   all 18 HAR70+ subjects, with every fold's model (values averaged over the 5 folds).
Nothing is trained or tuned, and nothing about the model changes.

Groups: general (22), older (18), walking_aid_inferred (503, 505, 513, 514, 515), other_older (13),
walker_inferred (walking_aid_inferred without 503 = the 4 walker users; see WALKER_INFERRED).

Class sets (the prediction is the 7-class argmax unless stated):
  all          all 7 classes; accuracy on all windows, macro-F1 over the non-stair classes (Sec 2.7)
  non_stair    windows labelled walking, shuffling, standing, sitting, lying (NON_STAIR; Sec 2.8)
  ustad4       PRIMARY Ustad scoring: all windows, labels AND predictions merged to 4 classes, stairs
               up/down -> walking, shuffling -> standing [Ustad Sec 2.4] (walking, standing, sitting,
               lying [Ustad Sec 3.2, p7-8; plan 2.9])
  ustad4_drop_argmax7  secondary: only windows labelled walking/standing/sitting/lying; a prediction of
               shuffling or stairs counts as an error
  ustad4_drop_argmax4  secondary: the same windows, argmax over the 4 classes only
Per-subject macro-F1 averages over the set's classes present in that subject's (merged) true labels.

Tests per (comparison, class set, metric): bootstrap 95% CI of the gap (10,000 resamples, seed 42,
subjects resampled within each group) and two-sided Mann-Whitney U with rank-biserial r:
  older_minus_general, other_older_minus_walking_aid, other_older_minus_walker (503 excluded).

Usage:
    python -m har.run_sanity_walking_aid            # -> results/har-baseline-gap/
    python -m har.run_sanity_walking_aid --smoke    # 300 windows per split, 200 resamples -> results/har-baseline-gap-smoke/
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from . import runlog as R
from .data import CLASS_NAMES, NON_STAIR, STAIRS, WALKING_AID_INFERRED, load_test_windows
from .folds import load_folds
from .models import build_model
from .paths import MODEL_SELECTION_CKPT_DIR, MODEL_SELECTION_DIR, RESULTS_DIR
from .stats import BOOT_SEED, N_BOOT, bootstrap_group_gap_ci, mann_whitney
from .train import predict_proba

FROZEN_CONFIG = MODEL_SELECTION_DIR / "har-frozen-model-config.json"
STORED_PER_SUBJECT = MODEL_SELECTION_DIR / "har-model-selection-per-subject-metrics.csv"
OUT_DIR = RESULTS_DIR / "har-baseline-gap"
SMOKE_DIR = RESULTS_DIR / "har-baseline-gap-smoke"
N_FOLDS = 5

USTAD4 = (0, 4, 5, 6)   # walking, standing, sitting, lying [Ustad Sec 3.2, p7-8]
# The 4 walker users = the inferred walking-aid group without 503. Ustad et al. state 4 walker users and
# 1 walking-sticks user (outdoors) [Ustad Sec 2.1, p2] but give no IDs; 503 as the sticks user is the
# project's decision (2026-11-20), so this group is inferred too.
WALKER_INFERRED = tuple(s for s in WALKING_AID_INFERRED if s != "503")
# Ustad et al. (2023), HARTH-trained XGBoost on HAR70+, 4 classes: 87% walking aid (n=5) vs 92% other (n=13);
# their HAR70+-trained model: 93% vs 95% [Ustad Abstract; Sec 3.2, p7-8].
USTAD_REF = {"harth_model_walking_aid": 87.0, "harth_model_other": 92.0, "har70_model_walking_aid": 93.0,
             "har70_model_other": 95.0}

# Ustad's 4-class merge, applied to labels and predictions: stairs up/down -> walking, shuffling -> standing
# [Ustad Sec 2.4]. Index -> merged index (walking 0, standing 4, sitting 5, lying 6).
USTAD_MERGE = np.array([0, 4, 0, 0, 4, 5, 6])
# name: (true classes kept, F1 classes, argmax restricted to, merge map)
CLASS_SETS = {"all": (tuple(range(7)), NON_STAIR, None, None), "non_stair": (NON_STAIR, NON_STAIR, None, None),
              "ustad4": (tuple(range(7)), USTAD4, None, USTAD_MERGE),
              "ustad4_drop_argmax7": (USTAD4, USTAD4, None, None),
              "ustad4_drop_argmax4": (USTAD4, USTAD4, USTAD4, None)}
GROUPS = ("general", "older", "walking_aid_inferred", "other_older", "walker_inferred")
COMPARISONS = (("older_minus_general", "older", "general"),
               ("other_older_minus_walking_aid", "other_older", "walking_aid_inferred"),
               ("other_older_minus_walker", "other_older", "walker_inferred"))
METRICS = ("accuracy", "macro_f1")


def subgroups(group: str, subject: str) -> list[str]:
    """Every summary group a subject belongs to."""
    if group == "general":
        return ["general"]
    if subject in WALKING_AID_INFERRED:
        return ["older", "walking_aid_inferred"] + (["walker_inferred"] if subject in WALKER_INFERRED else [])
    return ["older", "other_older"]


def predictions(proba: np.ndarray, restrict: tuple[int, ...] | None) -> np.ndarray:
    if restrict is None:
        return proba.argmax(1)
    r = np.array(restrict)
    return r[proba[:, r].argmax(1)]


def subject_rows(y, proba, subject, fold, group) -> list[dict]:
    rows = []
    for cs, (keep, f1_classes, restrict, merge) in CLASS_SETS.items():
        m = np.isin(y, keep)
        pred = predictions(proba, restrict)
        yt = y if merge is None else merge[y]
        if merge is not None:
            pred = merge[pred]
        for s in sorted(set(subject[m].tolist())):
            ms = m & (subject == s)
            present = [c for c in f1_classes if (yt[ms] == c).any()]
            rows.append({"fold": fold, "subject": s, "group": group, "class_set": cs, "n_windows": int(ms.sum()),
                         "n_correct": int((pred[ms] == yt[ms]).sum()), "accuracy": float((pred[ms] == yt[ms]).mean()),
                         "macro_f1": float(f1_score(yt[ms], pred[ms], labels=present, average="macro", zero_division=0))
                         if present else np.nan,
                         "n_f1_classes_present": len(present)})
    return rows


def load_fold_model(fold: int, arch_expected: dict):
    found = sorted(MODEL_SELECTION_CKPT_DIR.glob(f"har-final-tinyhar-fold{fold}-*.pt"))
    if len(found) != 1:
        raise FileNotFoundError(f"need exactly one TinyHAR fold-{fold} checkpoint in {MODEL_SELECTION_CKPT_DIR}, found {found}")
    ck = torch.load(found[0], map_location="cpu", weights_only=False)
    if ck["model"] != "tinyhar":
        raise ValueError(f"{found[0]} is not a TinyHAR checkpoint")
    for k, v in arch_expected.items():
        if k in ck["architecture"] and ck["architecture"][k] != v:
            raise ValueError(f"{found[0]}: architecture {k}={ck['architecture'][k]} differs from the frozen config ({v})")
    net = build_model("tinyhar", ck["architecture"], ck["norm_mean"], ck["norm_std"])
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, found[0]


def pooled_stairs_f1(y, pred) -> dict:
    out = {}
    for c in STAIRS:
        if (y == c).any() or (pred == c).any():
            out[f"{CLASS_NAMES[c]}_f1_pooled"] = float(f1_score(y, pred, labels=[c], average="macro", zero_division=0))
        else:
            out[f"{CLASS_NAMES[c]}_f1_pooled"] = np.nan
        out[f"n_{CLASS_NAMES[c]}_windows"] = int((y == c).sum())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = SMOKE_DIR if args.smoke else OUT_DIR
    n_boot = 200 if args.smoke else N_BOOT
    log = out / "logs" / "har-baseline-gap-run.log"
    log.unlink(missing_ok=True)
    t0 = time.perf_counter()
    with R.tee_to(log):
        frozen = json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))
        assert frozen["model"] == "tinyhar", frozen["model"]
        print(f"har.run_sanity_walking_aid | {' '.join(sys.argv)} | machine {R.machine_name()} | torch {torch.__version__} "
              f"| frozen {frozen['model']} {frozen['setting_id']} ({frozen['n_params']} params)")
        data = load_test_windows()
        gen_w, old_w = data["harth_test"], data["har70plus_test"]
        folds = load_folds(harth_test_counts={s: int(c) for s, c in pd.Series(gen_w.subject).value_counts().items()})
        assert len(folds) == N_FOLDS
        if args.smoke:
            rng = np.random.default_rng(0)
            gen_w = gen_w.select(np.sort(rng.choice(len(gen_w), 300, replace=False)))
            old_w = old_w.select(np.sort(rng.choice(len(old_w), 300, replace=False)))
        stored = pd.read_csv(STORED_PER_SUBJECT, dtype={"subject": str}).query("model == 'tinyhar'")

        per_fold, pooled_rows, gen_parts, ckpts, recheck = [], [], [], {}, []
        for f in folds:
            net, path = load_fold_model(f.fold, frozen["architecture"])
            ckpts[f.fold] = str(path)
            g = gen_w.for_subjects(f.test_general)
            pg = predict_proba(net, g.X)
            po = predict_proba(net, old_w.X)
            per_fold += subject_rows(g.y, pg, g.subject, f.fold, "general")
            per_fold += subject_rows(old_w.y, po, old_w.subject, f.fold, "older")
            gen_parts.append((g.y, pg.argmax(1)))
            # window-pooled stairs F1 + accuracy per older subgroup, this fold
            for grp in ("older", "other_older", "walking_aid_inferred", "walker_inferred"):
                m = np.array([grp in subgroups("older", s) for s in old_w.subject])
                pred = po[m].argmax(1)
                pooled_rows.append({"fold": f.fold, "group": grp, "accuracy_window_pooled": float((pred == old_w.y[m]).mean()),
                                    **pooled_stairs_f1(old_w.y[m], pred)})
            # cross-check against the stored (GPU) model-choice per-subject numbers
            if not args.smoke:
                for r in (r for r in per_fold if r["fold"] == f.fold and r["group"] == "general" and r["class_set"] == "all"):
                    s = stored[(stored.fold == f.fold) & (stored.subject == r["subject"])]
                    recheck.append({"fold": f.fold, "subject": r["subject"],
                                    "accuracy_diff": r["accuracy"] - float(s.accuracy.iloc[0]),
                                    "macro_f1_diff": r["macro_f1"] - float(s.non_stair_macro_f1.iloc[0])})
            print(f"fold {f.fold}: {path.name} | general {len(g)} windows ({len(f.test_general)} subjects), older {len(old_w)} windows")
        yg = np.concatenate([a for a, _ in gen_parts])
        pg_all = np.concatenate([b for _, b in gen_parts])
        pooled_rows.append({"fold": "all", "group": "general", "accuracy_window_pooled": float((pg_all == yg).mean()),
                            **pooled_stairs_f1(yg, pg_all)})
        if recheck:
            rc = pd.DataFrame(recheck)
            print(f"re-check vs stored model-choice metrics (GPU predictions): max |acc diff| {rc.accuracy_diff.abs().max():.5f}, "
                  f"max |F1 diff| {rc.macro_f1_diff.abs().max():.5f}")
            if rc[["accuracy_diff", "macro_f1_diff"]].abs().max().max() > 0.005:
                raise RuntimeError("CPU predictions disagree with the stored per-subject metrics by > 0.5 pt")

        pf = pd.DataFrame(per_fold)
        pf["walking_aid_inferred"] = pf.subject.isin(WALKING_AID_INFERRED)
        pf["walker_inferred"] = pf.subject.isin(WALKER_INFERRED)
        # pool over folds: general = its one test fold; older = mean over the 5 fold models
        agg = pf.groupby(["class_set", "subject", "group"]).agg(
            accuracy=("accuracy", "mean"), macro_f1=("macro_f1", "mean"), accuracy_sd_over_folds=("accuracy", "std"),
            macro_f1_sd_over_folds=("macro_f1", "std"), n_folds=("fold", "nunique"), n_windows=("n_windows", "first"),
            n_f1_classes_present=("n_f1_classes_present", "first")).reset_index()
        assert (agg.loc[agg.group == "general", "n_folds"] == 1).all()
        assert args.smoke or (agg.loc[agg.group == "older", "n_folds"] == N_FOLDS).all()
        agg["subgroup"] = [subgroups(g, s)[-1] for g, s in zip(agg.group, agg.subject)]
        agg["walker_inferred"] = agg.subject.isin(WALKER_INFERRED)

        def members(grp: str, cs: str) -> pd.DataFrame:
            a = agg[agg.class_set == cs]
            return a[[grp in subgroups(g, s) for g, s in zip(a.group, a.subject)]]

        summary = []
        pooled = pd.DataFrame(pooled_rows)
        pooled_mean = pooled.groupby("group").mean(numeric_only=True)
        for cs in CLASS_SETS:
            for grp in GROUPS:
                a = members(grp, cs)
                row = {"class_set": cs, "group": grp, "n_subjects": len(a), "subjects": " ".join(a.subject),
                       "n_windows_per_model": int(a.n_windows.sum())}
                for mt in METRICS:
                    row.update({f"{mt}_subject_mean": a[mt].mean(), f"{mt}_subject_sd": a[mt].std(ddof=1),
                                f"{mt}_subject_median": a[mt].median(), f"{mt}_subject_min": a[mt].min(),
                                f"{mt}_subject_max": a[mt].max()})
                if cs == "all" and grp in pooled_mean.index:
                    row.update(pooled_mean.loc[grp].to_dict())   # older groups: mean over the 5 fold models
                summary.append(row)
        summary = pd.DataFrame(summary)

        tests = []
        for name, gx, gy in COMPARISONS:
            for cs in CLASS_SETS:
                x, y = members(gx, cs), members(gy, cs)
                for mt in METRICS:
                    vx, vy = x[mt].dropna().values, y[mt].dropna().values   # NaN: no scored class present
                    ci = bootstrap_group_gap_ci(vx, vy, n_boot=n_boot, seed=BOOT_SEED)
                    mw = mann_whitney(vx, vy)
                    row = {"comparison": name, "group_x": gx, "group_y": gy, "class_set": cs, "metric": mt,
                           "n_x": len(vx), "n_y": len(vy), "mean_x": vx.mean(), "mean_y": vy.mean(),
                           "gap_x_minus_y_pt": 100 * ci["gap"], "ci95_low_pt": 100 * ci["ci_low"],
                           "ci95_high_pt": 100 * ci["ci_high"], "ci_excludes_0": bool(ci["ci_low"] > 0 or ci["ci_high"] < 0),
                           "mw_u": mw["u"], "mw_p": mw["p"], "rank_biserial_r": mw["rank_biserial_r"]}
                    if gx == "other_older":   # the gap per fold model, to show it is not one fold's artefact
                        sub = pf[pf.class_set == cs]
                        per = [sub[(sub.fold == k) & sub.subject.isin(x.subject)][mt].mean()
                               - sub[(sub.fold == k) & sub.subject.isin(y.subject)][mt].mean() for k in range(N_FOLDS)]
                        row.update({"gap_pt_fold_min": 100 * np.nanmin(per), "gap_pt_fold_max": 100 * np.nanmax(per)})
                    if cs == "ustad4" and mt == "accuracy" and gx == "other_older":
                        row["ustad_harth_model_gap_pt"] = USTAD_REF["harth_model_other"] - USTAD_REF["harth_model_walking_aid"]
                        row["ustad_har70_model_gap_pt"] = USTAD_REF["har70_model_other"] - USTAD_REF["har70_model_walking_aid"]
                    tests.append(row)
        tests = pd.DataFrame(tests)

        out.mkdir(parents=True, exist_ok=True)
        R.write_csv(out / "har-baseline-gap-tinyhar-per-subject-per-fold.csv", pf)
        R.write_csv(out / "har-baseline-gap-tinyhar-per-subject.csv", agg)
        R.write_csv(out / "har-baseline-gap-tinyhar-group-summary.csv", summary)
        R.write_csv(out / "har-baseline-gap-tinyhar-group-tests.csv", tests)
        R.write_csv(out / "har-baseline-gap-tinyhar-pooled-accuracy-and-stairs-f1-per-fold.csv", pooled)
        if recheck:
            R.write_csv(out / "har-baseline-gap-tinyhar-general-recheck-vs-model-selection.csv", rc)
        elapsed = time.perf_counter() - t0
        R.write_json(out / "har-baseline-gap-tinyhar-config.json", {
            "command_line": sys.argv, "smoke": args.smoke, "frozen_config": str(FROZEN_CONFIG),
            "frozen_setting": frozen["setting_id"], "checkpoints": ckpts, "class_sets": {
                k: {"true_classes_kept": [CLASS_NAMES[c] for c in keep], "f1_classes": [CLASS_NAMES[c] for c in f1c],
                    "argmax_over": None if r is None else [CLASS_NAMES[c] for c in r],
                    "merge_labels_and_predictions": None if mg is None else
                    {CLASS_NAMES[i]: CLASS_NAMES[j] for i, j in enumerate(mg) if i != j}}
                for k, (keep, f1c, r, mg) in CLASS_SETS.items()},
            "walking_aid_inferred": WALKING_AID_INFERRED, "walker_inferred": WALKER_INFERRED, "n_boot": n_boot,
            "boot_seed": BOOT_SEED, "ustad_reference": USTAD_REF, "machine_name": R.machine_name(),
            **R.device_info("cpu"), **R.git_info(), "code_sha256": R.code_sha256(), "finished_at": R.now(),
            "elapsed_s": elapsed})

        pd.set_option("display.width", 200)
        print("\nGroup summary (subject means, %):")
        s = summary.copy()
        for mt in METRICS:
            s[mt] = (100 * s[f"{mt}_subject_mean"]).round(2)
        print(s.pivot(index="group", columns="class_set", values=["accuracy", "macro_f1"]).reindex(GROUPS).to_string())
        print("\nGaps (pt) with bootstrap 95% CI and Mann-Whitney p:")
        print(tests[["comparison", "class_set", "metric", "n_x", "n_y", "gap_x_minus_y_pt", "ci95_low_pt", "ci95_high_pt",
                     "mw_p", "rank_biserial_r"]].round(3).to_string(index=False))
        print(f"\ndone in {elapsed:.1f} s -> {out}")


if __name__ == "__main__":
    main()
