"""Deletion faithfulness (plan Section 2.7).

For each window, the k segments (of the 48 = 6 axes x 8 time segments of 16 samples) with the largest
SHAP values for the target class (descending, stable ties) are replaced by the background mean at the
same (axis, time) positions, and the softmax probability of the target class is recorded for
k = 0..10. The target is c(x), the original model's predicted class (the same target as LCI).

    drop_k5 = p(k=0) - p(k=5)
    aopc    = mean over k = 1..10 of p(k=0) - p(k)

The random-deletion baseline deletes segments in a random order (n_repeats seeded orders per window).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch.nn as nn

from .train import predict_proba

SEG_LEN = 16
KS = tuple(range(0, 11))


def background_mean(background) -> np.ndarray:
    """(6, 128) mean of the background windows."""
    b = background.detach().cpu().numpy() if hasattr(background, "detach") else np.asarray(background)
    return b.mean(axis=0).astype(np.float32)


def delete_segments(X: np.ndarray, order: np.ndarray, k: int, bg_mean: np.ndarray, seg_len: int = SEG_LEN) -> np.ndarray:
    """X with, per window, the segments order[i, :k] (feature = axis * n_seg + segment) replaced by bg_mean."""
    n, C, T = X.shape
    n_seg = T // seg_len
    if k == 0:
        return X.copy()
    mask = np.zeros((n, C * n_seg), bool)
    np.put_along_axis(mask, order[:, :k], True, axis=1)
    mask = np.repeat(mask.reshape(n, C, n_seg), seg_len, axis=2)
    return np.where(mask, bg_mean[None], X).astype(X.dtype)


def _curve(model: nn.Module, X, order, target, bg_mean, ks, device) -> np.ndarray:
    rows = np.arange(len(X))
    return np.stack([predict_proba(model, delete_segments(X, order, k, bg_mean), device)[rows, target] for k in ks], 1)


def deletion_curve(model: nn.Module, X: np.ndarray, A_seg: np.ndarray, target: np.ndarray, bg_mean: np.ndarray,
                   ks: Sequence[int] = KS, device: str = "cpu") -> np.ndarray:
    """(n, len(ks)) target-class probability after deleting the top-k SHAP segments."""
    order = np.argsort(-np.asarray(A_seg), axis=1, kind="stable")
    return _curve(model, np.asarray(X, np.float32), order, np.asarray(target), bg_mean, list(ks), device)


def random_deletion_curve(model: nn.Module, X: np.ndarray, target: np.ndarray, bg_mean: np.ndarray,
                          ks: Sequence[int] = KS, n_repeats: int = 5, seed: int = 42, device: str = "cpu") -> np.ndarray:
    """(n, len(ks)) mean over n_repeats random deletion orders (numpy generator seeded with `seed`)."""
    X = np.asarray(X, np.float32)
    n, C, T = X.shape
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_repeats):
        order = np.argsort(rng.random((n, C * (T // SEG_LEN))), axis=1, kind="stable")
        out.append(_curve(model, X, order, np.asarray(target), bg_mean, list(ks), device))
    return np.mean(out, axis=0)


def faithfulness_scores(curve: np.ndarray, ks: Sequence[int] = KS) -> dict:
    """Per window: drop_k5 = p(0) - p(5) and aopc = mean over k = 1..10 of p(0) - p(k)."""
    ks = list(ks)
    c = np.asarray(curve)
    p0 = c[:, ks.index(0)]
    return {"drop_k5": p0 - c[:, ks.index(5)], "aopc": np.mean([p0 - c[:, ks.index(k)] for k in range(1, 11)], axis=0)}
