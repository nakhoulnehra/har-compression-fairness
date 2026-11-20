"""Logic Consistency Index (LCI), Tulla et al. (2026), Eq. 1 / Defs 3.1-3.2, 3.4.

    LCI(T, S) = omega * rho_bar + (1 - omega) * J_k

rho_bar : mean over samples of the per-sample Pearson r between teacher and student
          attribution vectors (correlation across the F features of ONE sample).
J_k     : Jaccard index |A n B| / |A u B| of the global top-k feature sets, where
          global importance = mean_i |A[i, :]|.

Pure numpy on purpose: no torch / shap imports, so attributions from any source work.
"""

from __future__ import annotations

import numpy as np

COLLAPSE_TAU = 0.85  # paper's empirical Logic Collapse threshold
_STD_EPS = 1e-8


def _as_2d(A, name: str) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    if A.ndim < 2:
        raise ValueError(f"{name} must have shape (N, ...) with at least 2 dims, got {A.shape}")
    return A.reshape(A.shape[0], -1)


def global_ranking(A: np.ndarray) -> np.ndarray:
    """Feature indices ordered by mean |attribution|, descending; ties -> lowest index first."""
    importance = np.abs(A).mean(axis=0)
    return np.argsort(-importance, kind="stable")


def top_k(A: np.ndarray, k: int) -> set[int]:
    return set(global_ranking(A)[:k].tolist())


def per_sample_corr(A_T: np.ndarray, A_S: np.ndarray) -> tuple[float, int]:
    """Mean per-sample Pearson r and the number of skipped samples."""
    rs, skipped = [], 0
    for t, s in zip(A_T, A_S):
        if t.std() < _STD_EPS or s.std() < _STD_EPS:
            skipped += 1
            continue
        r = np.corrcoef(t, s)[0, 1]
        if np.isnan(r):
            skipped += 1
            continue
        rs.append(r)
    return (float(np.mean(rs)) if rs else 0.0), skipped


def jaccard_topk(A_T: np.ndarray, A_S: np.ndarray, k: int) -> float:
    a, b = top_k(A_T, k), top_k(A_S, k)
    return len(a & b) / len(a | b)


def mprf(A_T: np.ndarray, A_S: np.ndarray) -> tuple[int, int]:
    """Most-Perturbed Rank Feature: argmax_j |rank_T(j) - rank_S(j)|. Returns (feature, shift)."""
    F = A_T.shape[1]
    rank_T = np.empty(F, dtype=int)
    rank_S = np.empty(F, dtype=int)
    rank_T[global_ranking(A_T)] = np.arange(F)
    rank_S[global_ranking(A_S)] = np.arange(F)
    shift = np.abs(rank_T - rank_S)
    j = int(np.argmax(shift))  # first (lowest-index) feature on ties
    return j, int(shift[j])


def compute_lci(A_T, A_S, k: int = 10, omega: float = 0.5) -> dict:
    """LCI between teacher attributions A_T and student attributions A_S.

    Both arrays have shape (N, ...) over the same N inputs in the same order; trailing dims
    are flattened to F features. LCI is NOT clipped to [0, 1] (rho_bar can be negative).
    """
    A_T, A_S = _as_2d(A_T, "A_T"), _as_2d(A_S, "A_S")
    if A_T.shape != A_S.shape:
        raise ValueError(f"shape mismatch: A_T {A_T.shape} vs A_S {A_S.shape}")
    n, F = A_T.shape
    if not 1 <= k <= F:
        raise ValueError(f"k={k} must be in [1, F={F}]")

    rho, n_skipped = per_sample_corr(A_T, A_S)
    jac = jaccard_topk(A_T, A_S, k)
    feat, shift = mprf(A_T, A_S)
    return {
        "lci": omega * rho + (1.0 - omega) * jac,
        "shap_corr": rho,
        "jaccard": jac,
        "n_skipped": n_skipped,
        "n_samples": n,
        "k": k,
        "mprf_feature": feat,
        "mprf_shift": shift,
    }


def is_collapsed(lci: float, tau: float = COLLAPSE_TAU) -> bool:
    return lci < tau


# ------------------------------------------------------------------ window extensions (HAR plan 2.6)

def aggregate_time_segments(A, seg_len: int = 16) -> np.ndarray:
    """(N, C, T) -> (N, C * T // seg_len): signed sum per (channel, segment), channel-major order
    (feature = channel * (T // seg_len) + segment). Raises if T % seg_len != 0."""
    A = np.asarray(A)
    if A.ndim != 3:
        raise ValueError(f"expected (N, C, T), got {A.shape}")
    N, C, T = A.shape
    if T % seg_len:
        raise ValueError(f"T={T} is not divisible by seg_len={seg_len}")
    return A.reshape(N, C, T // seg_len, seg_len).sum(axis=-1).reshape(N, C * (T // seg_len))


def compute_lci_by_subject(A_T, A_S, subjects, k: int = 10, omega: float = 0.5) -> dict[str, dict]:
    """compute_lci on each subject's rows (subjects in sorted order); same keys plus "subject"."""
    subjects = np.asarray(subjects)
    if len(subjects) != len(A_T) or len(A_T) != len(A_S):
        raise ValueError(f"lengths differ: A_T {len(A_T)}, A_S {len(A_S)}, subjects {len(subjects)}")
    A_T, A_S = np.asarray(A_T), np.asarray(A_S)
    out = {}
    for s in sorted(set(subjects.tolist())):
        m = subjects == s
        out[s] = {**compute_lci(A_T[m], A_S[m], k=k, omega=omega), "subject": s}
    return out
