"""Older-like copies of HARTH windows (XGP pilot round 2, plan Section 2.5 (b)). HARTH windows only.

A crude, seeded stand-in for slower, smaller movement, used (1) as a second validation window set
and (2) as extra reference windows for the augmented keep-score. It is NOT HAR70+ data and never
reads it: every function that takes Windows calls require_harth, and the result stays a HARTH
Windows object.

    time_stretch    speed s: the central round(128 s) samples of each channel, linearly resampled to
                    128 (s < 1 slows the movement down)
    scale_movement  x' = mean + a (x - mean) around each window's per-channel mean, so the gravity
                    component (the mean) is unchanged and the movement shrinks by a
    older_like      time_stretch then scale_movement, one (s, a) per window
    draw_older_like_params  s ~ U(0.8, 0.9), a ~ U(0.7, 0.9), drawn once per window with a seeded
                    numpy generator (all s first, then all a)
"""

from __future__ import annotations

import numpy as np

from .data import Windows, require_harth

OLDER_LIKE = {"speed": (0.8, 0.9), "scale": (0.7, 0.9), "explain_seed": 42, "reference_seed": 43}


def draw_older_like_params(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(s, a), each (n,) float64. One generator per call: n speeds first, then n scales, so the first
    k draws do not depend on n (each array comes from its own child generator)."""
    ss, sa = np.random.SeedSequence(seed).spawn(2)
    s = np.random.default_rng(ss).uniform(*OLDER_LIKE["speed"], size=n)
    a = np.random.default_rng(sa).uniform(*OLDER_LIKE["scale"], size=n)
    return s, a


def _check(X: np.ndarray, p: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    X, p = np.asarray(X, np.float32), np.asarray(p, np.float64)
    if X.ndim != 3:
        raise ValueError(f"X must be (N, C, T), got {X.shape}")
    if p.shape != (X.shape[0],):
        raise ValueError(f"{name} must be ({X.shape[0]},), got {p.shape}")
    return X, p


def time_stretch(X: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Per window: the central L = round(T s) samples of every channel (start (T - L) // 2), linearly
    resampled to T points (np.interp at linspace(0, L - 1, T)). (N, C, T) float32."""
    X, s = _check(X, s, "s")
    N, C, T = X.shape
    out = np.empty_like(X)
    for i in range(N):
        L = int(round(T * s[i]))
        if not 2 <= L <= T:
            raise ValueError(f"speed {s[i]} gives {L} samples of {T}")
        start = (T - L) // 2
        seg = X[i, :, start:start + L]
        q = np.linspace(0.0, L - 1, T)
        for c in range(C):
            out[i, c] = np.interp(q, np.arange(L), seg[c])
    return out


def scale_movement(X: np.ndarray, a: np.ndarray) -> np.ndarray:
    """x' = mean + a (x - mean), mean = each window's per-channel mean over time. (N, C, T) float32."""
    X, a = _check(X, a, "a")
    mu = X.mean(axis=2, keepdims=True, dtype=np.float64)
    return (mu + a[:, None, None] * (X - mu)).astype(np.float32)


def older_like(X: np.ndarray, s: np.ndarray, a: np.ndarray) -> np.ndarray:
    """time_stretch with speed s, then scale_movement with a; (N, 6, 128) float32."""
    return scale_movement(time_stretch(X, s), a)


def older_like_windows(w: Windows, s: np.ndarray, a: np.ndarray) -> Windows:
    """Older-like copies of HARTH windows: same labels, subjects and metadata, X transformed. The
    result is still a HARTH Windows object (it is derived from HARTH data only)."""
    require_harth(w)
    v = w.select(np.arange(len(w)))
    v.X = older_like(w.X, s, a)
    require_harth(v)
    return v
