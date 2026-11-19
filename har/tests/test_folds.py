"""Tests for har/folds.py (plan Section 2.10, test_folds.py and the normalisation tests)."""

import numpy as np
import pytest

from har import data as D
from har import folds as F
from har.paths import FOLD_ASSIGNMENT_CSV, RAW_DIR

HAS_RAW = (RAW_DIR / "harth").is_dir() and (RAW_DIR / "har70plus").is_dir()
real = pytest.mark.skipif(not HAS_RAW, reason="raw data not downloaded")

HARTH = [f"S{i:03d}" for i in (6, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29)]
OLDER = [str(i) for i in range(501, 519)]


def synthetic_counts(seed=0):
    rng = np.random.default_rng(seed)
    return {s: int(c) for s, c in zip(HARTH, rng.integers(100, 3200, len(HARTH)))}


def synthetic_windows(dataset, subjects, n_per=4, seed=0):
    rng = np.random.default_rng(seed)
    n = n_per * len(subjects)
    return D.Windows(X=rng.normal(size=(n, 6, 128)).astype(np.float32) * np.arange(1, 7)[None, :, None],
                     y=np.zeros(n, np.int64), subject=np.repeat(subjects, n_per).astype("<U4"), dataset=dataset,
                     group=np.full(n, D.GROUP[dataset], "<U7"), walking_aid=np.full(n, D.WA_NA, "<U23"),
                     majority_share=np.ones(n, np.float32), start_time=np.zeros(n, "datetime64[ms]"),
                     start_row=np.zeros(n, np.int64))


# ----------------------------------------------------------------------------------------- folds

def test_fold_sizes_and_every_subject_tested_once():
    counts = synthetic_counts()
    assign = F.assign_folds(counts)
    assert set(assign) == set(HARTH)
    sizes = sorted(np.bincount(list(assign.values()), minlength=5).tolist())
    assert sizes == [4, 4, 4, 5, 5]
    folds = F.make_folds(counts, OLDER)
    tested = [s for f in folds for s in f.test_general]
    assert sorted(tested) == sorted(HARTH)


def test_folds_are_deterministic_and_balanced():
    counts = synthetic_counts(1)
    assert F.make_folds(counts, OLDER) == F.make_folds(counts, OLDER)
    totals = [sum(counts[s] for s in f.test_general) for f in F.make_folds(counts, OLDER)]
    assert max(totals) - min(totals) <= max(counts.values())


def test_no_subject_in_two_sets_and_har70plus_test_only():
    for f in F.make_folds(synthetic_counts(2), OLDER):
        tr, va, te, old = map(set, (f.train, f.val, f.test_general, f.test_older))
        assert not (tr & va) and not (tr & te) and not (va & te)
        assert tr | va | te == set(HARTH)
        assert old == set(OLDER) and not (old & (tr | va))
        assert len(va) == 3


def test_validation_subjects_have_enough_windows_and_are_not_test():
    counts = synthetic_counts(3)
    for f in F.make_folds(counts, OLDER):
        assert all(counts[s] >= F.MIN_VAL_WINDOWS for s in f.val)
        assert not set(f.val) & set(f.test_general)


def test_check_rejects_leaks():
    f = F.make_folds(synthetic_counts(), OLDER)[0]
    leaky = F.FoldSplit(0, f.train + (f.test_general[0],), f.val, f.test_general, f.test_older)
    with pytest.raises(ValueError, match="both train and test_general"):
        leaky.check(HARTH, OLDER)
    older_in_train = F.FoldSplit(0, f.train + ("501",), f.val, f.test_general, f.test_older)
    with pytest.raises(ValueError):
        older_in_train.check(HARTH, OLDER)


def test_tuning_sets_of_fold0_are_disjoint_from_its_test_subjects():
    f0 = F.make_folds(synthetic_counts(), OLDER)[0]
    t = F.tuning_sets(f0)
    tr, va, never = map(set, (t["tuning_train"], t["tuning_validation"], t["never_used"]))
    assert va == set(f0.val) and len(va) == 3
    assert not (tr & va) and not (tr & never) and not (va & never)
    assert not (tr | va) & set(OLDER)


# --------------------------------------------------------------------------------- normalisation

@pytest.fixture
def fold_and_windows():
    f = F.make_folds(synthetic_counts(), OLDER)[0]
    w = synthetic_windows("harth", HARTH)
    return f, w


def test_channel_stats_use_training_subjects_only(fold_and_windows):
    f, w = fold_and_windows
    mean, std = F.channel_stats(w, f)
    Xtr = w.X[np.isin(w.subject, f.train)].astype(np.float64)
    assert np.allclose(mean, Xtr.mean(axis=(0, 2)), atol=1e-5)
    assert np.allclose(std, Xtr.std(axis=(0, 2)), atol=1e-5)


def test_channel_stats_unchanged_when_other_subjects_change(fold_and_windows):
    f, w = fold_and_windows
    mean, std = F.channel_stats(w, f)
    w2 = w.select(np.arange(len(w)))
    w2.X = w2.X.copy()
    others = np.isin(w2.subject, f.val + f.test_general)
    w2.X[others] = w2.X[others] * 100 + 50                     # change validation and test data
    m2, s2 = F.channel_stats(w2, f)
    assert np.array_equal(mean, m2) and np.array_equal(std, s2)
    w3 = w.select(np.isin(w.subject, f.train))                 # drop them entirely
    m3, s3 = F.channel_stats(w3, f)
    assert np.allclose(mean, m3) and np.allclose(std, s3)


@pytest.mark.parametrize("who", ["val", "test_general", "test_older", "unknown"])
def test_channel_stats_raise_on_non_training_subjects(fold_and_windows, who):
    f, w = fold_and_windows
    extra = {"val": f.val[0], "test_general": f.test_general[0], "test_older": "501", "unknown": "S999"}[who]
    with pytest.raises(ValueError, match="only use training subjects"):
        F.channel_stats(w, f, subjects=list(f.train) + [extra])


def test_channel_stats_raise_on_har70plus_windows(fold_and_windows):
    f, _ = fold_and_windows
    with pytest.raises(ValueError, match="test-only"):
        F.channel_stats(synthetic_windows("har70plus", OLDER), f)


# ------------------------------------------------------------------------------------- real data

@real
def test_real_folds_written_match_a_fresh_computation():
    import pandas as pd
    w = D.build_all()
    counts = pd.Series(w["harth_test"].subject).value_counts().to_dict()
    folds = F.make_folds({s: int(c) for s, c in counts.items()}, w["har70plus_test"].subjects)
    for f in folds:
        f.check(w["harth_test"].subjects, w["har70plus_test"].subjects)
        mean, std = F.channel_stats(w["harth_train"], f)
        assert mean.shape == std.shape == (6,) and (std > 0).all()
    if FOLD_ASSIGNMENT_CSV.exists():
        written = pd.read_csv(FOLD_ASSIGNMENT_CSV, dtype={"subject": str})
        for f in folds:
            g = written[written.fold == f.fold]
            assert sorted(g[g.role == "train"].subject) == list(f.train)
            assert sorted(g[g.role == "validation"].subject) == list(f.val)
            assert sorted(g[(g.role == "test") & (g.dataset == "harth")].subject) == list(f.test_general)
            assert sorted(g[(g.role == "test") & (g.dataset == "har70plus")].subject) == list(f.test_older)
            assert set(g[g.dataset == "har70plus"].role) == {"test"}
