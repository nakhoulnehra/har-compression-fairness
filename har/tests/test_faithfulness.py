"""Tests for har/faithfulness.py (plan Section 2.7, deletion faithfulness)."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from har.faithfulness import (background_mean, delete_segments, deletion_curve, faithfulness_scores,
                              random_deletion_curve)


class SegmentSum(nn.Module):
    """Logit of class 0 = sum of channel 0, time segment 2 (samples 32..47); other logits 0."""

    def forward(self, x):
        z = torch.zeros(x.shape[0], 7)
        z[:, 0] = x[:, 0, 32:48].sum(dim=1)
        return z


def test_background_mean_shape():
    bg = np.random.default_rng(0).normal(size=(10, 6, 128)).astype(np.float32)
    assert background_mean(bg).shape == (6, 128)
    assert np.allclose(background_mean(bg), bg.mean(0))


def test_delete_segments_replaces_exactly_the_top_k_segments():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2, 6, 128)).astype(np.float32)
    bgm = np.full((6, 128), 9.0, np.float32)
    A = np.zeros((2, 48))
    A[0, 0 * 8 + 2] = 5.0          # channel 0, segment 2 is the top segment of window 0
    A[0, 3 * 8 + 7] = 4.0          # channel 3, segment 7 second
    A[1, 5 * 8 + 0] = 1.0
    order = np.argsort(-A, axis=1, kind="stable")
    Y = delete_segments(X, order, 2, bgm)
    assert (Y[0, 0, 32:48] == 9.0).all() and (Y[0, 3, 112:128] == 9.0).all()
    changed = (Y != X)
    assert changed[0].sum() == 32 and changed[1].sum() == 32
    assert (Y[0, 0, :32] == X[0, 0, :32]).all()
    assert (delete_segments(X, order, 0, bgm) == X).all()


def test_deletion_curve_k0_is_original_probability_and_top_segment_matters():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(4, 6, 128)).astype(np.float32)
    X[:, 0, 32:48] = 1.0                                  # class-0 logit 16 -> p(class 0) close to 1
    bgm = np.zeros((6, 128), np.float32)
    A = np.zeros((4, 48))
    A[:, 2] = 1.0                                         # SHAP points at the right segment
    target = np.zeros(4, int)
    m = SegmentSum()
    curve = deletion_curve(m, X, A, target, bgm, ks=range(0, 11))
    p0 = torch.softmax(m(torch.from_numpy(X)), 1)[:, 0].numpy()
    assert curve.shape == (4, 11)
    assert np.allclose(curve[:, 0], p0)
    assert np.allclose(curve[:, 1], 1 / 7, atol=1e-6)    # logit 0 after deletion -> uniform
    sc = faithfulness_scores(curve)
    assert np.allclose(sc["drop_k5"], p0 - 1 / 7, atol=1e-6)
    assert np.allclose(sc["aopc"], p0 - 1 / 7, atol=1e-6)


def test_faithfulness_scores_formula():
    curve = np.array([[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]])
    sc = faithfulness_scores(curve)
    assert sc["drop_k5"][0] == pytest.approx(0.5)
    assert sc["aopc"][0] == pytest.approx(np.mean(np.arange(1, 11) / 10))


def test_random_deletion_deterministic_with_seed():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(3, 6, 128)).astype(np.float32)
    bgm = np.zeros((6, 128), np.float32)
    m = SegmentSum()
    a = random_deletion_curve(m, X, np.zeros(3, int), bgm, ks=range(0, 11), n_repeats=3, seed=42)
    b = random_deletion_curve(m, X, np.zeros(3, int), bgm, ks=range(0, 11), n_repeats=3, seed=42)
    assert a.shape == (3, 11) and np.array_equal(a, b)
