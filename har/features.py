"""Window features for the XGBoost accuracy reference, following HARTH's feature set
[HARTH Sec 3.3, p10-11] (Logacjov et al. 2021, papers/sensors-21-07853 (1).pdf); Ustad et al. (2023)
use "the same 161 features" [Ustad Sec 2.3.2, p5].

From the paper:
- 4th-order 20 Hz low-pass Butterworth on the acceleration signals          [Sec 3.3, step 2]
  (applied once, to whole gap-free stretches, in har/data.py before windowing; not repeated here)
- 8 signals: the 6 axes and each sensor's vector magnitude sqrt(x^2+y^2+z^2)  [Sec 3.3, step 4]
- gravity component: 4th-order 1 Hz low-pass Butterworth; movement = raw - gravity
- gravity features: mean, median, std, coefficient of variation, 25th and 75th percentile, min, max
- movement features: skew, kurtosis, signal energy, frequency-domain magnitudes' mean and std,
  dominant frequency, dominant frequency's magnitude, spectral centroid, total signal power
- axis correlation between all six axes and between the two vector magnitudes
- mean across the two sensors' gravity components
Our choices (not stated): the 1 Hz gravity filter is zero-phase and applied per 128-sample window
(the paper filters whole recordings, before windowing); correlations on the input signals; "mean across the two sensors'
gravity components" read as one value per axis (x, y, z); DC excluded from the frequency features;
signal energy = mean of squares; total signal power = sum of |FFT|^2 / T; coefficient of variation
= std / |mean| (0 when |mean| < 1e-8). The listed features give 8 x 17 + 15 + 1 + 3 = 155 of the
paper's 161; the remaining 6 cannot be identified from the text and are left out. Min-max scaling
[Sec 3.3, step 5] is not applied (it does not change tree models). Windows are the plan's 2.56 s, not
HARTH's 1 s [Sec 3.3] or Ustad's 5 s [Sec 2.3.2].
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .data import CHANNELS, FS

GRAVITY_HZ, ORDER = 1.0, 4
_SOS_GRAV = butter(ORDER, GRAVITY_HZ, btype="low", fs=FS, output="sos")

SIGNALS = list(CHANNELS) + ["back_mag", "thigh_mag"]
GRAVITY_STATS = ["grav_mean", "grav_median", "grav_std", "grav_cv", "grav_p25", "grav_p75", "grav_min", "grav_max"]
MOVEMENT_STATS = ["mov_skew", "mov_kurtosis", "mov_energy", "mov_fft_mag_mean", "mov_fft_mag_std", "mov_dom_freq_hz",
                  "mov_dom_freq_mag", "mov_spectral_centroid_hz", "mov_total_power"]
AXIS_PAIRS = [(i, j) for i in range(6) for j in range(i + 1, 6)]
FEATURE_NAMES = ([f"{s}_{st}" for s in SIGNALS for st in GRAVITY_STATS + MOVEMENT_STATS]
                 + [f"corr_{CHANNELS[i]}_{CHANNELS[j]}" for i, j in AXIS_PAIRS] + ["corr_back_mag_thigh_mag"]
                 + [f"grav_mean_both_sensors_{a}" for a in "xyz"])


def _corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    return np.divide((a * b).sum(axis=1), den, out=np.zeros(len(a)), where=den > 1e-12)


def window_features(X: np.ndarray, batch: int = 20_000) -> np.ndarray:
    """(n, 6, T) raw windows in g -> (n, 155) float32 features, in FEATURE_NAMES order."""
    if X.ndim != 3 or X.shape[1] != len(CHANNELS):
        raise ValueError(f"expected (n, 6, T) windows, got {X.shape}")
    if len(X) == 0:
        return np.zeros((0, len(FEATURE_NAMES)), np.float32)
    return np.concatenate([_features(X[i:i + batch].astype(np.float64))
                           for i in range(0, len(X), batch)]).astype(np.float32)


def _features(X: np.ndarray) -> np.ndarray:
    n, _, T = X.shape
    mags = np.stack([np.linalg.norm(X[:, 0:3], axis=1), np.linalg.norm(X[:, 3:6], axis=1)], axis=1)
    S = np.concatenate([X, mags], axis=1)                      # (n, 8, T)
    G = sosfiltfilt(_SOS_GRAV, S, axis=2)                      # gravity component
    M = S - G                                                  # movement component

    g_mean, g_std = G.mean(axis=2), G.std(axis=2)
    g_cv = np.divide(g_std, np.abs(g_mean), out=np.zeros_like(g_std), where=np.abs(g_mean) > 1e-8)
    g_p25, g_med, g_p75 = np.percentile(G, [25, 50, 75], axis=2)
    grav = [g_mean, g_med, g_std, g_cv, g_p25, g_p75, G.min(axis=2), G.max(axis=2)]

    mc = M - M.mean(axis=2, keepdims=True)
    m_std = mc.std(axis=2)
    safe = np.where(m_std > 1e-12, m_std, 1.0)
    skew = np.where(m_std > 1e-12, (mc ** 3).mean(axis=2) / safe ** 3, 0.0)
    kurt = np.where(m_std > 1e-12, (mc ** 4).mean(axis=2) / safe ** 4 - 3.0, 0.0)
    mag = np.abs(np.fft.rfft(M, axis=2))[..., 1:]             # exclude DC
    freqs = np.fft.rfftfreq(T, d=1.0 / FS)[1:]
    dom = mag.argmax(axis=2)
    tot = mag.sum(axis=2)
    centroid = np.divide((mag * freqs).sum(axis=2), tot, out=np.zeros_like(tot), where=tot > 1e-12)
    mov = [skew, kurt, (M ** 2).mean(axis=2), mag.mean(axis=2), mag.std(axis=2), freqs[dom],
           np.take_along_axis(mag, dom[..., None], axis=2)[..., 0], centroid, (mag ** 2).sum(axis=2) / T]

    per_signal = np.stack(grav + mov, axis=2).reshape(n, -1)  # (n, 8 x 17), signal-major
    corrs = [_corr(X[:, i], X[:, j]) for i, j in AXIS_PAIRS] + [_corr(mags[:, 0], mags[:, 1])]
    both = [(G[:, a].mean(axis=1) + G[:, 3 + a].mean(axis=1)) / 2 for a in range(3)]
    return np.concatenate([per_signal, np.stack(corrs + both, axis=1)], axis=1)
