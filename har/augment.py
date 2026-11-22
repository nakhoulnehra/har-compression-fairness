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


# ------------------------------------------------------------------------------ round 3: + per-sensor rotation
# XGP round 3 (plan 2.5 (b), "Round 3 pre-registration", amended 2026-11-22 before any run): the older-like
# transform gains a per-sensor random 3D rotation, applied after time_stretch and scale_movement. The back
# and the thigh triplets are rotated independently: axis uniform on the unit sphere, angle ~ U(-15, 15)
# degrees, x' = R x at every time step (gravity included, so the static postures change too). Rotation as
# a wearable-data augmentation follows Um et al. (2017, ICMI); the +-15 degree bound is our a-priori choice,
# not tuned. The round-2 functions above are unchanged (round 2 stays reproducible); s and a are drawn
# exactly as in round 2, the rotation from two further, independent streams of the same seed.

OLDER_LIKE_R3 = {**OLDER_LIKE, "rotation_deg": (-15.0, 15.0), "sensors": {"back": (0, 1, 2), "thigh": (3, 4, 5)},
                 "order": "time_stretch -> scale_movement -> rotate_sensors"}


def draw_rotation_params(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(axes (n, 2, 3) unit vectors, angles (n, 2) in degrees), sensor order (back, thigh). Streams 2 and 3
    of SeedSequence(seed) (streams 0 and 1 are draw_older_like_params' s and a), so s and a do not change and
    the first k draws do not depend on n."""
    ss = np.random.SeedSequence(seed).spawn(4)
    v = np.random.default_rng(ss[2]).normal(size=(n, 2, 3))
    axes = v / np.linalg.norm(v, axis=2, keepdims=True)
    angles = np.random.default_rng(ss[3]).uniform(*OLDER_LIKE_R3["rotation_deg"], size=(n, 2))
    return axes, angles


def draw_older_like_r3_params(n: int, seed: int) -> dict[str, np.ndarray]:
    s, a = draw_older_like_params(n, seed)
    axes, angles = draw_rotation_params(n, seed)
    return {"s": s, "a": a, "axes": axes, "angles": angles}


def rotation_matrices(axes: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """Rodrigues: (..., 3) unit axes and (...) angles in degrees -> (..., 3, 3) rotation matrices."""
    k = np.asarray(axes, np.float64)
    th = np.deg2rad(np.asarray(angles_deg, np.float64))[..., None, None]
    K = np.zeros(k.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K[..., 1, 0], K[..., 2, 0], K[..., 2, 1] = k[..., 2], -k[..., 1], k[..., 0]
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def rotate_sensors(X: np.ndarray, axes: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """Rotate the back (channels 0-2) and thigh (3-5) triplets of each window independently:
    x'(t) = R_sensor x(t). axes (N, 2, 3), angles (N, 2) degrees. (N, 6, T) float32."""
    X = np.asarray(X, np.float32)
    if X.ndim != 3 or X.shape[1] != 6:
        raise ValueError(f"X must be (N, 6, T), got {X.shape}")
    axes, angles_deg = np.asarray(axes), np.asarray(angles_deg)
    if axes.shape != (X.shape[0], 2, 3) or angles_deg.shape != (X.shape[0], 2):
        raise ValueError(f"axes must be ({X.shape[0]}, 2, 3) and angles ({X.shape[0]}, 2)")
    Rm = rotation_matrices(axes, angles_deg)                         # (N, 2, 3, 3)
    out = np.empty_like(X)
    for j, ch in enumerate(OLDER_LIKE_R3["sensors"].values()):
        out[:, list(ch)] = np.einsum("nij,njt->nit", Rm[:, j], X[:, list(ch)].astype(np.float64)).astype(np.float32)
    return out


def older_like_r3(X: np.ndarray, s: np.ndarray, a: np.ndarray, axes: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """Round 3's older-like transform: time_stretch(s) -> scale_movement(a) -> rotate_sensors(axes, angles)."""
    return rotate_sensors(older_like(X, s, a), axes, angles_deg)


def older_like_windows_r3(w: Windows, s, a, axes, angles_deg) -> Windows:
    """Round-3 older-like copies of HARTH windows (still a HARTH Windows object)."""
    require_harth(w)
    v = w.select(np.arange(len(w)))
    v.X = older_like_r3(w.X, s, a, axes, angles_deg)
    require_harth(v)
    return v
