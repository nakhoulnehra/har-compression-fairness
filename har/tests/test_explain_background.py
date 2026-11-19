"""Tests for the fixed explain windows and SHAP backgrounds (plan Section 2.2, Step 2)."""

import numpy as np
import pandas as pd
import pytest

from har import data as D
from har import folds as F
from har.paths import BACKGROUND_CSV, EXPLAIN_WINDOWS_CSV, RAW_DIR

HAS_RAW = (RAW_DIR / "harth").is_dir() and (RAW_DIR / "har70plus").is_dir()
real = pytest.mark.skipif(not HAS_RAW, reason="raw data not downloaded")

HARTH = [f"S{i:03d}" for i in (6, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29)]
OLDER = [str(i) for i in range(501, 519)]


def synth(dataset, class_counts_per_subject: dict[str, dict[int, int]], seed=0) -> D.Windows:
    subj, y = [], []
    for s, cc in class_counts_per_subject.items():
        for c, k in cc.items():
            subj += [s] * k
            y += [c] * k
    n = len(y)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)                      # interleave, like real recordings
    subj, y = np.array(subj, "<U4")[order], np.array(y, np.int64)[order]
    start_row = np.zeros(n, np.int64)
    for s in set(subj.tolist()):
        m = subj == s
        start_row[m] = np.arange(m.sum()) * 128
    return D.Windows(X=rng.normal(size=(n, 6, 128)).astype(np.float32), y=y, subject=subj, dataset=dataset,
                     group=np.full(n, D.GROUP[dataset], "<U7"), walking_aid=np.full(n, D.WA_NA, "<U23"),
                     majority_share=np.ones(n, np.float32), start_time=np.zeros(n, "datetime64[ms]"),
                     start_row=start_row)


def balanced_except_exhausted(picked: np.ndarray, available: np.ndarray) -> bool:
    """Stratification as specified: classes not used up differ by at most 1, every used-up class
    contributed all its windows, and no used-up class has more picks than a non-exhausted one."""
    k = np.bincount(picked, minlength=7)
    a = np.bincount(available, minlength=7)
    present = a > 0
    exhausted = present & (k == a)
    open_ = present & ~exhausted
    if open_.any():
        if k[open_].max() - k[open_].min() > 1:
            return False
        if exhausted.any() and k[exhausted].max() > k[open_].max():
            return False
    return bool((k[~present] == 0).all())


# ------------------------------------------------------------------------------ stratified picks

def test_stratified_pick_round_robin_with_exhausted_classes():
    classes = np.repeat(np.arange(7), [50, 50, 2, 0, 50, 50, 50])
    pos = F.stratified_pick(classes, 30, np.random.default_rng(0))
    assert len(pos) == 30 and len(set(pos.tolist())) == 30
    k = np.bincount(classes[pos], minlength=7)
    assert k[2] == 2 and k[3] == 0                     # stairs up all used, stairs down absent
    # rounds 1-2: 6 classes (12 picks, stairs up used up); rounds 3-5: 5 classes (27 picks);
    # the last 3 picks go to the first classes in index order: walking, shuffling, standing
    assert k[[0, 1, 4, 5, 6]].tolist() == [6, 6, 6, 5, 5]


def test_stratified_pick_takes_everything_when_fewer_than_n():
    classes = np.repeat(np.arange(7), [3, 0, 1, 0, 4, 2, 0])
    pos = F.stratified_pick(classes, 30, np.random.default_rng(1))
    assert sorted(pos.tolist()) == list(range(10))


# ----------------------------------------------------------------------------- explain windows

def test_explain_windows_reproducible_and_seed_dependent():
    w = synth("har70plus", {s: {0: 60, 1: 20, 2: 3, 4: 40, 5: 40, 6: 30} for s in OLDER[:5]})
    a = F.select_explain_windows(w, "har70plus_test")
    b = F.select_explain_windows(w, "har70plus_test")
    c = F.select_explain_windows(w, "har70plus_test", seed=7)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_explain_windows_per_subject_and_stratified():
    cc = {"501": {0: 60, 1: 20, 2: 3, 4: 40, 5: 40, 6: 30}, "502": {0: 5, 4: 5, 5: 5},
          "503": {0: 100, 4: 100, 5: 100, 6: 100}}
    w = synth("har70plus", cc)
    idx = F.select_explain_windows(w, "har70plus_test")
    for s, counts in cc.items():
        picked = idx[w.subject[idx] == s]
        assert len(picked) == min(30, sum(counts.values()))
        assert len(set(picked.tolist())) == len(picked)
        assert balanced_except_exhausted(w.y[picked], w.y[w.subject == s])


def test_explain_choice_of_a_subject_does_not_depend_on_other_subjects():
    full = synth("har70plus", {s: {0: 40, 4: 40, 5: 40} for s in OLDER[:4]})
    one = full.for_subjects([OLDER[2]])
    ids_full = D.window_ids(full, "har70plus_test")[F.select_explain_windows(full, "har70plus_test")]
    ids_one = D.window_ids(one, "har70plus_test")[F.select_explain_windows(one, "har70plus_test")]
    assert [i for i in ids_full if f":{OLDER[2]}:" in i] == ids_one.tolist()


def test_explain_windows_only_from_test_splits():
    w = synth("harth", {"S006": {0: 50}})
    with pytest.raises(ValueError, match="test splits only"):
        F.select_explain_windows(w, "harth_train")


# ---------------------------------------------------------------------------------- backgrounds

@pytest.fixture
def fold_and_train():
    counts = {s: 2000 for s in HARTH}
    f = F.make_folds(counts, OLDER)[0]
    w = synth("harth", {s: {0: 40, 1: 10, 2: 4, 3: 4, 4: 20, 5: 60, 6: 10} for s in HARTH})
    return f, w


def test_background_only_from_training_subjects(fold_and_train):
    f, w = fold_and_train
    idx = F.select_background(w, "harth_train", f)
    assert len(idx) == F.N_BACKGROUND == len(set(idx.tolist()))
    assert set(w.subject[idx].tolist()) <= set(f.train)
    assert not set(w.subject[idx].tolist()) & set(f.val + f.test_general + f.test_older)


def test_background_reproducible_stratified_and_fold_dependent(fold_and_train):
    f, w = fold_and_train
    a = F.select_background(w, "harth_train", f)
    assert np.array_equal(a, F.select_background(w, "harth_train", f))
    assert not np.array_equal(a, F.select_background(w, "harth_train", f, seed=7))
    k = np.bincount(w.y[a], minlength=7)
    assert k.max() - k.min() <= 1                     # 100 = 14 x 7 + 2


def test_background_rejects_har70plus_and_non_training_splits(fold_and_train):
    f, w = fold_and_train
    with pytest.raises(ValueError, match="test-only"):
        F.select_background(synth("har70plus", {"501": {0: 200}}), "harth_train", f)
    with pytest.raises(ValueError, match="harth_train"):
        F.select_background(w, "harth_test", f)


def test_window_ids_round_trip(fold_and_train):
    _, w = fold_and_train
    ids = D.window_ids(w, "harth_train")
    assert len(set(ids.tolist())) == len(w)
    pick = np.array([5, 0, 17])
    assert np.array_equal(D.find_windows(w, "harth_train", ids[pick].tolist()), pick)
    with pytest.raises(KeyError):
        D.find_windows(w, "harth_train", ["harth_train:S999:0"])
    with pytest.raises(ValueError):
        D.window_ids(w, "har70plus_test")


# ------------------------------------------------------------------------------------ real data

@pytest.fixture(scope="module")
def real_setup():
    w = D.build_all()
    counts = pd.Series(w["harth_test"].subject).value_counts().to_dict()
    folds = F.make_folds({s: int(c) for s, c in counts.items()}, w["har70plus_test"].subjects)
    return w, folds


@real
def test_real_explain_windows_from_test_sets_stratified_and_written(real_setup):
    w, _ = real_setup
    written = pd.read_csv(EXPLAIN_WINDOWS_CSV, dtype={"subject": str}) if EXPLAIN_WINDOWS_CSV.exists() else None
    for split in F.TEST_SPLITS:
        ws = w[split]
        idx = F.select_explain_windows(ws, split)
        for s in ws.subjects:
            picked = idx[ws.subject[idx] == s]
            assert len(picked) == min(30, int((ws.subject == s).sum()))
            assert balanced_except_exhausted(ws.y[picked], ws.y[ws.subject == s])
        if written is not None:
            ids = written[written.split == split].window_id.tolist()
            assert ids == D.window_ids(ws, split)[idx].tolist()      # file matches a fresh, seeded draw
            assert np.array_equal(D.find_windows(ws, split, ids), idx)
    if written is not None:
        assert set(written.split) == set(F.TEST_SPLITS)                # never harth_train
        assert written.window_id.is_unique


@real
def test_real_backgrounds_only_from_fold_training_subjects(real_setup):
    w, folds = real_setup
    written = pd.read_csv(BACKGROUND_CSV, dtype={"subject": str}) if BACKGROUND_CSV.exists() else None
    tr = w["harth_train"]
    for f in folds:
        idx = F.select_background(tr, "harth_train", f)
        subj = set(tr.subject[idx].tolist())
        assert len(idx) == 100 and subj <= set(f.train)
        assert not subj & (set(f.val) | set(f.test_general) | set(w["har70plus_test"].subjects))
        k = np.bincount(tr.y[idx], minlength=7)
        assert k.max() - k.min() <= 1
        if written is not None:
            ids = written[written.fold == f.fold].window_id.tolist()
            assert ids == D.window_ids(tr, "harth_train")[idx].tolist()
            assert set(written[written.fold == f.fold].subject) <= set(f.train)
