"""Statistics over subjects (plan Section 2.8).

Unit = subject. Implemented so far: pooling over folds, Mann-Whitney U with rank-biserial r, and the
percentile bootstrap CI of a group gap (first used by Step 5, the baseline-gap check). The paired
tests, Holm, d_z and the FA-LCI bootstrap follow with the compression grid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

N_BOOT = 10_000
BOOT_SEED = 42


def pool_over_folds(per_fold: pd.DataFrame, value: str | list[str], keys: list[str]) -> pd.DataFrame:
    """(fold, *keys, subject, group, value...) -> one row per (*keys, subject, group). A HARTH subject
    must appear in exactly one fold (its test fold) and keeps that value; a HAR70+ subject is
    averaged over its folds. Adds `n_folds`."""
    values = [value] if isinstance(value, str) else list(value)
    by = [*keys, "subject", "group"]
    g = per_fold.groupby(by, dropna=False)
    out = g[values].mean()
    out["n_folds"] = g.fold.nunique()
    out = out.reset_index()
    if (out.loc[out.group == "general", "n_folds"] != 1).any():
        raise ValueError("a general (HARTH) subject appears in more than one fold")
    return out


def mann_whitney(x, y) -> dict:
    """Two-sided Mann-Whitney U of x vs y, plus the rank-biserial r = 2 U_x / (n_x n_y) - 1
    (r > 0: x tends to be larger)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    res = mannwhitneyu(x, y, alternative="two-sided")
    u = float(res.statistic)
    return {"u": u, "p": float(res.pvalue), "rank_biserial_r": 2 * u / (len(x) * len(y)) - 1,
            "n_x": len(x), "n_y": len(y)}


def bootstrap_group_gap_ci(older, general, stat=np.mean, n_boot: int = N_BOOT, seed: int = BOOT_SEED,
                           level: float = 0.95) -> dict:
    """Point estimate and percentile CI of stat(older) - stat(general) (generally: first group minus
    second), resampling subjects with replacement within each group."""
    a, b = np.asarray(older, float), np.asarray(general, float)
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(a), size=(n_boot, len(a)))
    ib = rng.integers(0, len(b), size=(n_boot, len(b)))
    diffs = np.array([stat(a[i]) - stat(b[j]) for i, j in zip(ia, ib)])
    q = (1 - level) / 2
    lo, hi = np.quantile(diffs, [q, 1 - q])
    return {"gap": float(stat(a) - stat(b)), "ci_low": float(lo), "ci_high": float(hi),
            "n_boot": n_boot, "seed": seed}
