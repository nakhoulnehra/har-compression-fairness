"""Tests for har/train.py and har/features.py (plan Section 2.10, test_train.py)."""

import numpy as np
import pytest
import torch
from sklearn.metrics import f1_score

from har import data as D
from har.features import FEATURE_NAMES, window_features
from har.models import build_model, resolve_config
from har.train import (TrainConfig, class_weights, evaluate, f1_from_confusion, pooled_class_f1, pooled_macro_f1,
                       subject_metrics, train_model, train_xgboost, xgb_predict_proba)


def windows(dataset="harth", n_per_class=20, subjects=("S001", "S002"), seed=0) -> D.Windows:
    """Separable synthetic windows: class c has a mean offset of c on the back_x channel."""
    rng = np.random.default_rng(seed)
    y = np.tile(np.repeat(np.arange(7), n_per_class), len(subjects))
    subj = np.repeat(subjects, 7 * n_per_class).astype("<U4")
    X = rng.normal(size=(len(y), 6, 128)).astype(np.float32) * 0.3
    X[:, 0] += y[:, None]
    n = len(y)
    return D.Windows(X=X, y=y.astype(np.int64), subject=subj, dataset=dataset, group=np.full(n, D.GROUP[dataset], "<U7"),
                     walking_aid=np.full(n, D.WA_NA, "<U23"), majority_share=np.ones(n, np.float32),
                     start_time=np.zeros(n, "datetime64[ms]"), start_row=np.arange(n, dtype=np.int64))


# -------------------------------------------------------------------------------------- metrics

def test_class_weights_mean_one_inverse_frequency():
    y = np.array([0] * 80 + [1] * 15 + [2] * 5)
    w = class_weights(y)
    assert w[:3].mean() == pytest.approx(1.0) and (w[3:] == 0).all()
    assert w[2] / w[0] == pytest.approx(16.0)


def test_per_subject_macro_f1_uses_only_non_stair_classes_present():
    y = np.array([0, 0, 2, 2, 5, 5])          # walking, stairs up, sitting
    pred = np.array([0, 5, 2, 0, 5, 5])
    r = subject_metrics(y, pred, np.array(["A"] * 6))[0]
    expected = f1_score(y, pred, labels=[0, 5], average="macro")
    assert r["non_stair_macro_f1"] == pytest.approx(expected)
    assert r["n_non_stair_classes_present"] == 2 and r["accuracy"] == pytest.approx(4 / 6)


def test_pooled_class_f1_equals_sklearn_on_concatenated_windows():
    rng = np.random.default_rng(0)
    y, pred = rng.integers(0, 7, 500), rng.integers(0, 7, 500)
    r = pooled_class_f1(y, pred)
    ref = f1_score(y, pred, labels=[2, 3], average=None)
    assert r["f1_class_2"] == pytest.approx(ref[0]) and r["f1_class_3"] == pytest.approx(ref[1])
    assert r["f1_mean"] == pytest.approx(ref.mean())


def test_f1_from_confusion_matches_pooled_macro_f1():
    rng = np.random.default_rng(1)
    y, pred = rng.integers(0, 7, 400), rng.integers(0, 7, 400)
    cm = evaluate(y, pred, np.array(["A"] * 400))["confusion"]
    assert f1_from_confusion(cm).mean() == pytest.approx(pooled_macro_f1(y, pred))


# ------------------------------------------------------------------------------------- training

def test_require_harth_in_training():
    tr, va = windows(), windows("har70plus", subjects=("501",))
    m = build_model("cnn1d", resolve_config("cnn1d", 0.5), np.zeros(6), np.ones(6))
    with pytest.raises(ValueError, match="test-only"):
        train_model(m, tr, va, TrainConfig(max_epochs=1), seed=0)
    with pytest.raises(ValueError, match="test-only"):
        train_model(m, va, tr, TrainConfig(max_epochs=1), seed=0)


def test_one_epoch_run_is_reproducible_with_the_same_seed():
    tr, va = windows(), windows(seed=1, subjects=("S003",))
    outs = []
    for _ in range(2):
        torch.manual_seed(0)
        m = build_model("cnn1d", resolve_config("cnn1d", 0.5), np.zeros(6), np.ones(6))
        info = train_model(m, tr, va, TrainConfig(max_epochs=1, batch_size=32), seed=0, log=lambda s: None)
        outs.append((info["history"][0]["train_loss"], [p.detach().clone() for p in m.parameters()]))
    assert outs[0][0] == outs[1][0]
    assert all(torch.equal(a, b) for a, b in zip(outs[0][1], outs[1][1]))


def test_training_learns_and_keeps_the_best_epoch():
    tr, va = windows(n_per_class=30), windows(seed=1, subjects=("S003",))
    torch.manual_seed(0)
    # normalised with training statistics, as in the pipeline (the HARTH CNN has no BatchNorm)
    m = build_model("cnn1d", resolve_config("cnn1d", 1.0), tr.X.mean(axis=(0, 2)), tr.X.std(axis=(0, 2)))
    info = train_model(m, tr, va, TrainConfig(max_epochs=6, batch_size=32, patience=3), seed=0, log=lambda s: None)
    best = max(h["val_macro_f1_pooled7"] for h in info["history"])
    assert info["best_val_macro_f1_pooled7"] == best and best > 0.5
    from har.train import predict_proba
    assert pooled_macro_f1(va.y, predict_proba(m, va.X).argmax(1)) == pytest.approx(best)


# ----------------------------------------------------------------------------- features, XGBoost

def test_window_features_shape_and_known_values():
    t = np.arange(128) / 50.0
    X = np.zeros((2, 6, 128), np.float32)
    X[0, 0] = np.sin(2 * np.pi * 2.34375 * t)           # exactly on an FFT bin (6 / 128 * 50 Hz)
    X[1] += 1.0                                          # constant signals: pure gravity, no movement
    F = window_features(X)
    # 8 signals x (8 gravity + 9 movement) + 15 axis pairs + 1 magnitude pair + 3 = 155 [HARTH Sec 3.3]
    assert F.shape == (2, len(FEATURE_NAMES)) == (2, 155) and np.isfinite(F).all()
    col = {n: i for i, n in enumerate(FEATURE_NAMES)}
    assert F[0, col["back_x_mov_dom_freq_hz"]] == pytest.approx(2.34375)      # movement part keeps the 2.3 Hz sine
    assert abs(F[0, col["back_x_grav_mean"]]) < 0.05                           # 1 Hz low-pass removes it
    assert F[1, col["back_x_grav_mean"]] == pytest.approx(1.0, abs=1e-6)
    assert F[1, col["back_x_mov_energy"]] == pytest.approx(0.0, abs=1e-10) and F[1, col["back_x_mov_skew"]] == 0
    assert F[1, col["back_mag_grav_mean"]] == pytest.approx(np.sqrt(3), rel=1e-5)
    assert F[1, col["grav_mean_both_sensors_x"]] == pytest.approx(1.0, abs=1e-6)


def test_xgboost_trains_on_harth_only_and_predicts():
    tr, va = windows(), windows(seed=1, subjects=("S003",))
    booster, info = train_xgboost(tr, va, {"max_depth": 3, "learning_rate": 0.3, "n_estimators": 20}, seed=0,
                                  log=lambda s: None)
    p = xgb_predict_proba(booster, va.X)
    assert p.shape == (len(va), 7) and np.allclose(p.sum(1), 1, atol=1e-5)
    assert info["best_val_macro_f1_pooled7"] > 0.5
    with pytest.raises(ValueError, match="test-only"):
        train_xgboost(tr, windows("har70plus", subjects=("501",)), {"max_depth": 3, "learning_rate": 0.3}, seed=0)
