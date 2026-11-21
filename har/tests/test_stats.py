"""Tests for har/stats.py (plan Section 2.10, test_stats.py): the parts used from Step 5 on."""

import numpy as np
import pandas as pd
import pytest

from har.run_sanity_walking_aid import WALKER_INFERRED, predictions, subgroups, subject_rows
from har.stats import bootstrap_group_gap_ci, mann_whitney, pool_over_folds


def test_pool_over_folds_harth_single_fold_har70_mean():
    df = pd.DataFrame({"fold": [0, 1, 0, 1, 2], "subject": ["S1", "S2", "501", "501", "501"],
                       "group": ["general", "general", "older", "older", "older"], "v": [0.5, 0.7, 0.2, 0.4, 0.9]})
    out = pool_over_folds(df, "v", keys=[]).set_index("subject")
    assert out.loc["S1", "v"] == 0.5 and out.loc["S2", "v"] == 0.7
    assert out.loc["501", "v"] == pytest.approx(0.5) and out.loc["501", "n_folds"] == 3


def test_pool_over_folds_rejects_harth_in_two_folds():
    df = pd.DataFrame({"fold": [0, 1], "subject": ["S1", "S1"], "group": ["general", "general"], "v": [0.5, 0.7]})
    with pytest.raises(ValueError):
        pool_over_folds(df, "v", keys=[])


def test_mann_whitney_rank_biserial_extremes():
    r = mann_whitney([5, 6, 7], [1, 2, 3])
    assert r["rank_biserial_r"] == pytest.approx(1.0) and r["u"] == 9
    assert mann_whitney([1, 2, 3], [5, 6, 7])["rank_biserial_r"] == pytest.approx(-1.0)


def test_bootstrap_ci_covers_true_gap_and_is_seeded():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0.9, 0.05, 13), rng.normal(0.8, 0.05, 5)
    ci = bootstrap_group_gap_ci(a, b, n_boot=2000)
    assert ci["ci_low"] < 0.1 < ci["ci_high"]
    assert ci["ci_low"] < ci["gap"] < ci["ci_high"]
    assert bootstrap_group_gap_ci(a, b, n_boot=2000) == ci
    assert bootstrap_group_gap_ci(a, a, n_boot=500)["gap"] == 0


def test_walker_group_and_subgroups():
    assert WALKER_INFERRED == ("505", "513", "514", "515")
    assert subgroups("older", "503") == ["older", "walking_aid_inferred"]
    assert subgroups("older", "505")[-1] == "walker_inferred"
    assert subgroups("older", "501") == ["older", "other_older"]
    assert subgroups("general", "S006") == ["general"]


def test_ustad4_filter_and_restricted_argmax():
    # true: walking, shuffling, stairs up, standing; predictions: shuffling wins on window 0
    y = np.array([0, 1, 2, 4])
    p = np.full((4, 7), 0.01)
    p[0, 1], p[0, 0] = 0.6, 0.3          # walking predicted as shuffling
    p[1, 1] = 0.9
    p[2, 2] = 0.9
    p[3, 4] = 0.9
    assert predictions(p, (0, 4, 5, 6))[0] == 0
    rows = {r["class_set"]: r for r in subject_rows(y, p, np.array(["501"] * 4), 0, "older")}
    assert rows["ustad4_drop_argmax7"]["n_windows"] == 2 and rows["ustad4_drop_argmax7"]["accuracy"] == 0.5  # shuffling pred = error
    assert rows["ustad4_drop_argmax4"]["accuracy"] == 1.0
    assert rows["all"]["n_windows"] == 4 and rows["non_stair"]["n_windows"] == 3


def test_ustad4_merge_on_labels_and_predictions():
    # [Ustad Sec 2.4]: stairs -> walking, shuffling -> standing, on both sides
    y = np.array([2, 3, 1, 0, 4, 5])               # stairs up, stairs down, shuffling, walking, standing, sitting
    pred = np.array([0, 2, 4, 3, 1, 6])            # walking, stairs up, standing, stairs down, shuffling, lying
    p = np.full((6, 7), 0.01)
    p[np.arange(6), pred] = 0.9
    r = {x["class_set"]: x for x in subject_rows(y, p, np.array(["501"] * 6), 0, "older")}["ustad4"]
    assert r["n_windows"] == 6                     # every window is scored
    assert r["n_correct"] == 5                     # only sitting -> lying is wrong
    assert r["n_f1_classes_present"] == 3          # merged truth: walking, standing, sitting


# ------------------------------------------------------------------------------ Step 9 additions (plan 2.8)

from har.stats import (bootstrap_fa_lci_ci, bootstrap_mean_ci, cohens_dz, holm, paired_wilcoxon,  # noqa: E402
                       safe_rate)


def test_holm_known_values_and_nan():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    # sorted 0.005, 0.01, 0.03, 0.04 -> x4, x3, x2, x1 = 0.02, 0.03, 0.06, 0.04 -> monotone 0.02 0.03 0.06 0.06
    assert np.allclose(holm(p), [0.03, 0.06, 0.06, 0.02])
    q = holm(np.array([0.01, np.nan, 0.02]))
    assert np.isnan(q[1]) and np.allclose(q[[0, 2]], [0.02, 0.02])
    assert holm(np.array([0.6, 0.9])).max() <= 1.0


def test_paired_wilcoxon_rank_biserial_and_dz():
    a = np.array([0.9, 0.8, 0.85, 0.95, 0.7, 0.75])
    r = paired_wilcoxon(a, a - 0.1 - np.arange(6) * 0.01)
    assert r["rank_biserial_r"] == pytest.approx(1.0) and r["n"] == 6 and r["median_diff"] > 0
    assert paired_wilcoxon(a - 0.1 - np.arange(6) * 0.01, a)["rank_biserial_r"] == pytest.approx(-1.0)
    same = paired_wilcoxon(a, a)
    assert same["p"] == 1.0 and same["median_diff"] == 0
    d = np.array([1.0, 2.0, 3.0])
    assert cohens_dz(d, np.zeros(3)) == pytest.approx(2.0)
    nanpair = paired_wilcoxon(np.array([1.0, np.nan, 2.0, 3.0]), np.array([0.0, 1.0, 1.0, 1.5]))
    assert nanpair["n"] == 3


def test_bootstrap_mean_ci_seeded():
    x = np.random.default_rng(0).normal(0.7, 0.1, 18)
    ci = bootstrap_mean_ci(x, n_boot=2000)
    assert ci["ci_low"] < ci["mean"] < ci["ci_high"] and ci["n"] == 18
    assert bootstrap_mean_ci(x, n_boot=2000) == ci
    assert np.isnan(bootstrap_mean_ci(np.array([]), n_boot=10)["mean"])


def test_vectorised_gap_equals_loop():
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=9), rng.normal(size=5)
    fast = bootstrap_group_gap_ci(a, b, n_boot=500)
    slow = bootstrap_group_gap_ci(a, b, stat=lambda v: float(np.mean(v)), n_boot=500)
    assert fast["ci_low"] == pytest.approx(slow["ci_low"]) and fast["ci_high"] == pytest.approx(slow["ci_high"])


def test_fa_lci_bootstrap_point_is_min_and_paired_difference():
    older, general = np.array([0.5, 0.6, 0.55, 0.65]), np.array([0.8, 0.85, 0.9])
    r = bootstrap_fa_lci_ci(older, general, n_boot=2000)
    assert r["fa_lci"] == pytest.approx(min(older.mean(), general.mean()))
    assert r["ci_low"] <= r["fa_lci"] <= r["ci_high"]
    r2 = bootstrap_fa_lci_ci(older, general, older - 0.1, general - 0.1, n_boot=500)
    assert r2["diff"] == pytest.approx(0.1)
    assert r2["diff_ci_low"] == pytest.approx(0.1) and r2["diff_ci_high"] == pytest.approx(0.1)  # same subjects


def test_safe_rate():
    assert safe_rate([0.9, 0.5, 0.7, np.nan], 0.7) == pytest.approx(2 / 3)
    assert np.isnan(safe_rate([], 0.5))
