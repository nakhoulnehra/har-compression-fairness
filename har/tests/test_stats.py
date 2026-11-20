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
