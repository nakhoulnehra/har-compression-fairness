"""Plan Steps 1-2 (data part): build the HAR windows, folds and normalisation statistics, and write
the setup summaries to results/har-setup/.

Usage (from the repo root):
    python -m har.build_windows            # uses the window cache in data/processed/har/ if valid
    python -m har.build_windows --force    # rebuild the cache from data/raw/

Writes (paths in har/paths.py):
    har-window-counts-per-subject.csv        windows per subject, split and class; single-label / mixed
    har-window-counts-per-group-and-class.csv windows per split, group and class
    har-dropped-data-summary.csv             every row or window dropped, and why
    har-fold-assignment.csv                  each fold's train / validation / test subjects
    har-inner-validation-subjects.csv        the 3 validation subjects of each fold
    har-normalisation-stats-per-fold.csv     channel mean / std from each fold's training subjects
    har-explain-windows.csv                  the fixed explain windows (IDs), up to 30 per subject
    har-explain-windows-per-subject-class-and-group.csv   their counts per subject, class and group
    har-shap-background-windows.csv          the 100 SHAP background windows (IDs) of each fold
    har-explain-and-background-windows-per-fold.csv       explain / background counts per fold
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from .data import (CLASS_NAMES, CHANNELS, LABEL_NAMES, PIPELINE_VERSION, SHARED_LABELS, STAIRS, WA_NA, WA_NO,
                   WA_YES, WALKING_AID_INFERRED, build_all, build_report, window_ids)
from .folds import (N_BACKGROUND, N_EXPLAIN_PER_SUBJECT, SEED, TEST_SPLITS, channel_stats, make_folds,
                    select_background, select_explain_windows)
from .paths import (BACKGROUND_CSV, DROPPED_DATA_CSV, EXPLAIN_BACKGROUND_PER_FOLD_CSV, EXPLAIN_COUNTS_CSV,
                    EXPLAIN_WINDOWS_CSV, FOLD_ASSIGNMENT_CSV, HAR_SETUP_DIR, INNER_VALIDATION_CSV,
                    NORMALISATION_STATS_CSV, WINDOW_COUNTS_PER_GROUP_CLASS_CSV, WINDOW_COUNTS_PER_SUBJECT_CSV)

SPLIT_LABEL = {"harth_train": "train_50pct_overlap", "harth_test": "test_no_overlap",
               "har70plus_test": "test_no_overlap"}
# Approximate counts in plan Section 1 (from the planning profile), for the +-5% acceptance check.
PLAN_ESTIMATES = {"harth_train": 84_700, "har70plus_test": 17_500}


def class_counts(y: np.ndarray) -> dict[str, int]:
    c = np.bincount(y, minlength=len(CLASS_NAMES))
    return {f"n_{name}": int(v) for name, v in zip(CLASS_NAMES, c)}


def per_subject_table(windows, wstats: pd.DataFrame, rstats: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, w in windows.items():
        st = wstats[wstats.split == key].set_index("subject")
        for s in sorted(st.index):
            r0 = rstats[(rstats.dataset == w.dataset) & (rstats.subject == s)].iloc[0]
            m = w.subject == s
            rows.append({"dataset": w.dataset, "subject": s, "group": r0.group, "walking_aid": r0.walking_aid,
                         "split": key, "windowing": SPLIT_LABEL[key], "hours_50hz": round(r0.hours_50hz, 3),
                         "n_windows": int(m.sum()), "n_single_label": int(st.loc[s, "n_single_label"]),
                         "n_mixed_kept": int(st.loc[s, "n_mixed_kept"]),
                         "n_mixed_dropped": int(st.loc[s, "n_dropped_no_majority"]),
                         **class_counts(w.y[m])})
    return pd.DataFrame(rows)


def per_group_class_table(windows) -> pd.DataFrame:
    rows = []
    for key, w in windows.items():
        groups = [("general" if w.dataset == "harth" else "older", np.ones(len(w), bool))]
        if w.dataset == "har70plus":
            groups += [("older_inferred_walking_aid", w.walking_aid == WA_YES),
                       ("older_inferred_no_walking_aid", w.walking_aid == WA_NO)]
        for g, m in groups:
            rows.append({"split": key, "windowing": SPLIT_LABEL[key], "group": g,
                         "n_subjects": len(set(w.subject[m].tolist())), "n_windows": int(m.sum()),
                         "n_mixed_kept": int((w.majority_share[m] < 1).sum()), **class_counts(w.y[m])})
    return pd.DataFrame(rows)


def dropped_table(wstats: pd.DataFrame, rstats: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in rstats.iterrows():
        base = {"dataset": r.dataset, "subject": r.subject}
        dc = r.dropped_columns if isinstance(r.dropped_columns, str) else ""
        if dc:
            rows.append({**base, "stage": "load", "reason": "extra index column dropped", "n_rows": 0,
                         "n_windows": 0, "detail": dc})
        if r.n_rows_raw != r.n_rows_50hz:
            rows.append({**base, "stage": "resample", "reason": "recorded at 100 Hz; FIR anti-alias filter and decimated to 50 Hz",
                         "n_rows": int(r.n_rows_raw - r.n_rows_50hz), "n_windows": 0,
                         "detail": f"{r.n_rows_raw} rows at {r.raw_period_ms} ms -> {r.n_rows_50hz} rows at 20 ms"})
        for code in sorted(LABEL_NAMES):
            if code in SHARED_LABELS:
                continue
            n = int(r[f"rows_label_{code}"])
            if n:
                rows.append({**base, "stage": "labels", "reason": "label not shared by HARTH and HAR70+",
                             "n_rows": n, "n_windows": 0, "detail": f"label {code} ({LABEL_NAMES[code]})"})
    for _, s in wstats.iterrows():
        base = {"dataset": s.dataset, "subject": s.subject}
        if s.split != "harth_train":  # gaps and short runs are the same for both HARTH splits; report once
            rows.append({**base, "stage": "segments", "reason": "recording gap > 40 ms (windows never cross it)",
                         "n_rows": 0, "n_windows": 0, "detail": f"{int(s.n_gaps)} gaps"})
            if s.rows_short_runs:
                rows.append({**base, "stage": "segments", "reason": "gap-free shared-label run shorter than 128 samples",
                             "n_rows": int(s.rows_short_runs), "n_windows": 0, "detail": ""})
        if s.rows_not_covered_run_tails:
            rows.append({**base, "stage": f"windows_{s.split}", "reason": "run tail shorter than one window step",
                         "n_rows": int(s.rows_not_covered_run_tails), "n_windows": 0, "detail": f"step {s.step}"})
        if s.n_dropped_no_majority:
            rows.append({**base, "stage": f"windows_{s.split}", "reason": "no label covers > 50% of the window",
                         "n_rows": 0, "n_windows": int(s.n_dropped_no_majority), "detail": f"step {s.step}"})
    return pd.DataFrame(rows)


def fold_tables(folds, windows):
    counts = {k: pd.Series(w.subject).value_counts() for k, w in windows.items()}
    rows, vrows = [], []
    for f in folds:
        for role, subjects, key, ds in [("train", f.train, "harth_train", "harth"),
                                        ("validation", f.val, "harth_test", "harth"),
                                        ("test", f.test_general, "harth_test", "harth"),
                                        ("test", f.test_older, "har70plus_test", "har70plus")]:
            for s in subjects:
                rows.append({"fold": f.fold, "dataset": ds, "subject": s,
                             "group": "general" if ds == "harth" else "older",
                             "walking_aid": WA_NA if ds == "harth" else (WA_YES if s in WALKING_AID_INFERRED else WA_NO),
                             "role": role, "windows_used": key, "n_windows": int(counts[key].get(s, 0))})
        for s in f.val:
            vrows.append({"fold": f.fold, "subject": s, "n_windows_no_overlap": int(counts["harth_test"].get(s, 0))})
    return pd.DataFrame(rows), pd.DataFrame(vrows)


def window_rows(w, split, idx, extra=None) -> pd.DataFrame:
    ids = window_ids(w, split)
    df = pd.DataFrame({"window_id": ids[idx], "split": split, "dataset": w.dataset, "subject": w.subject[idx],
                       "group": w.group[idx], "walking_aid": w.walking_aid[idx], "class_index": w.y[idx],
                       "class_name": np.array(CLASS_NAMES)[w.y[idx]], "start_row": w.start_row[idx],
                       "start_time": pd.to_datetime(w.start_time[idx]).strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3],
                       "majority_share": np.round(w.majority_share[idx], 4)})
    if extra:
        for k, v in extra.items():
            df.insert(0, k, v)
    return df


def explain_and_background(windows, folds):
    """The fixed explain windows (plan 2.2) and each fold's SHAP background, as ID tables."""
    test_fold = {s: f.fold for f in folds for s in f.test_general}
    parts = []
    for split in TEST_SPLITS:
        w = windows[split]
        idx = select_explain_windows(w, split)
        df = window_rows(w, split, idx)
        df.insert(0, "pick_order", df.groupby("subject").cumcount())
        df.insert(0, "harth_test_fold", [str(test_fold[s]) if split == "harth_test" else "all" for s in df.subject])
        parts.append(df)
    explain = pd.concat(parts, ignore_index=True)
    explain.insert(0, "pipeline_version", PIPELINE_VERSION)
    bg = []
    for f in folds:
        idx = select_background(windows["harth_train"], "harth_train", f)
        df = window_rows(windows["harth_train"], "harth_train", idx)
        df.insert(0, "pick_order", np.arange(len(df)))
        df.insert(0, "fold", f.fold)
        bg.append(df)
    background = pd.concat(bg, ignore_index=True)
    background.insert(0, "pipeline_version", PIPELINE_VERSION)
    return explain, background


def explain_count_table(explain: pd.DataFrame, windows) -> pd.DataFrame:
    rows = []
    avail = {s: w for s, w in ((sp, windows[sp]) for sp in TEST_SPLITS)}
    def counts(df):
        c = df.class_index.value_counts()
        return {f"n_{n}": int(c.get(i, 0)) for i, n in enumerate(CLASS_NAMES)}
    for (split, s), g in explain.groupby(["split", "subject"], sort=True):
        w = avail[split]
        rows.append({"level": "subject", "split": split, "subject": s, "group": g.group.iloc[0],
                     "walking_aid": g.walking_aid.iloc[0], "n_subjects": 1, "n_explain": len(g),
                     "n_available_windows": int((w.subject == s).sum()),
                     "n_classes_present": int(len(np.unique(w.y[w.subject == s]))), **counts(g)})
    for name, m in [("general", explain.group == "general"), ("older", explain.group == "older"),
                    ("older_inferred_walking_aid", explain.walking_aid == WA_YES),
                    ("older_inferred_no_walking_aid", explain.walking_aid == WA_NO), ("all", explain.group.notna())]:
        g = explain[m]
        rows.append({"level": "group", "split": ";".join(sorted(g.split.unique())), "subject": "", "group": name,
                     "walking_aid": "", "n_subjects": g.subject.nunique(), "n_explain": len(g),
                     "n_available_windows": "", "n_classes_present": "", **counts(g)})
    return pd.DataFrame(rows)


def per_fold_table(explain: pd.DataFrame, background: pd.DataFrame, folds) -> pd.DataFrame:
    rows = []
    for f in folds:
        e = explain
        b = background[background.fold == f.fold]
        c = b.class_index.value_counts()
        rows.append({"fold": f.fold,
                     "n_explain_general_test": int(e.subject.isin(f.test_general).sum()),
                     "n_explain_older_test": int(e.subject.isin(f.test_older).sum()),
                     "n_explain_total_test": int(e.subject.isin(f.test_general + f.test_older).sum()),
                     "n_explain_validation": int(e.subject.isin(f.val).sum()),
                     "n_background": len(b), "n_background_subjects": b.subject.nunique(),
                     "background_subjects": ";".join(sorted(b.subject.unique())),
                     **{f"n_background_{n}": int(c.get(i, 0)) for i, n in enumerate(CLASS_NAMES)},
                     "background_mixed_label": int((b.majority_share < 1).sum())})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="rebuild the window cache from data/raw/")
    args = ap.parse_args()
    t0 = time.perf_counter()
    windows = build_all(force=args.force)
    t_build = time.perf_counter() - t0
    wstats, rstats = build_report()

    test_counts = pd.Series(windows["harth_test"].subject).value_counts().to_dict()
    older = windows["har70plus_test"].subjects
    folds = make_folds({s: int(c) for s, c in test_counts.items()}, older)

    norm = []
    for f in folds:
        mean, std = channel_stats(windows["harth_train"], f)
        n_win = int(np.isin(windows["harth_train"].subject, f.train).sum())
        norm += [{"fold": f.fold, "channel": c, "mean_g": float(m), "std_g": float(s),
                  "n_train_subjects": len(f.train), "n_train_windows": n_win} for c, m, s in zip(CHANNELS, mean, std)]

    HAR_SETUP_DIR.mkdir(parents=True, exist_ok=True)
    subj = per_subject_table(windows, wstats, rstats)
    grp = per_group_class_table(windows)
    drop = dropped_table(wstats, rstats)
    fold_df, val_df = fold_tables(folds, windows)
    subj.to_csv(WINDOW_COUNTS_PER_SUBJECT_CSV, index=False)
    grp.to_csv(WINDOW_COUNTS_PER_GROUP_CLASS_CSV, index=False)
    drop.to_csv(DROPPED_DATA_CSV, index=False)
    fold_df.to_csv(FOLD_ASSIGNMENT_CSV, index=False)
    val_df.to_csv(INNER_VALIDATION_CSV, index=False)
    pd.DataFrame(norm).to_csv(NORMALISATION_STATS_CSV, index=False, float_format="%.6f")
    explain, background = explain_and_background(windows, folds)
    explain.to_csv(EXPLAIN_WINDOWS_CSV, index=False)
    background.to_csv(BACKGROUND_CSV, index=False)
    ecounts = explain_count_table(explain, windows)
    ecounts.to_csv(EXPLAIN_COUNTS_CSV, index=False)
    perfold = per_fold_table(explain, background, folds)
    perfold.to_csv(EXPLAIN_BACKGROUND_PER_FOLD_CSV, index=False)

    # ---- console summary and acceptance checks (plan Steps 1-2)
    print(f"build/load: {t_build:.1f} s")
    for key, w in windows.items():
        print(f"{key:15s} {len(w):7d} windows, {len(w.subjects):2d} subjects, mixed kept {(w.majority_share < 1).sum()}")
    for key, est in PLAN_ESTIMATES.items():
        n = len(windows[key])
        print(f"  {key}: {n} vs plan estimate {est} ({100 * (n - est) / est:+.1f}%, acceptance +-5%)")
    no_stairs = sorted(s for s in older if not np.isin(windows["har70plus_test"].y[windows["har70plus_test"].subject == s], STAIRS).any())
    print(f"HAR70+ subjects with no stairs rows: {sorted(rstats[(rstats.dataset == 'har70plus') & ~rstats.has_stairs_rows].subject)}")
    print(f"HAR70+ subjects with no stairs windows: {no_stairs}")
    for f in folds:
        cls_test = np.unique(windows["harth_test"].y[np.isin(windows["harth_test"].subject, f.test_general)])
        print(f"fold {f.fold}: train {len(f.train)} {list(f.train)}\n        val {list(f.val)}  test {list(f.test_general)} "
              f"(classes in HARTH test: {len(cls_test)}/7, windows {sum(test_counts[s] for s in f.test_general)})")
    print(f"explain windows: {len(explain)} ({N_EXPLAIN_PER_SUBJECT} max per subject, seed {SEED})")
    print(ecounts[ecounts.level == "group"].drop(columns=["level", "subject", "walking_aid", "n_available_windows",
                                                          "n_classes_present"]).to_string(index=False))
    print(f"backgrounds: {N_BACKGROUND} windows per fold from training subjects")
    print(perfold.drop(columns=["background_subjects"]).to_string(index=False))
    print(f"total time {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
