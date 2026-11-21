"""Statistics over subjects (plan Section 2.8).

Unit = subject. Pooling over folds, Mann-Whitney U with rank-biserial r, the percentile bootstrap CI
of a group gap (Step 5), and for the compression grid (Step 9): the paired Wilcoxon with the
matched-pairs rank-biserial r and Cohen's d_z, Holm, the bootstrap CI of a mean and of FA-LCI (with
the paired difference between two methods), and the safe-explanation rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata, wilcoxon

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
    if stat is np.mean:                      # vectorised; same draws, same result as the loop
        diffs = a[ia].mean(1) - b[ib].mean(1)
    else:
        diffs = np.array([stat(a[i]) - stat(b[j]) for i, j in zip(ia, ib)])
    q = (1 - level) / 2
    lo, hi = np.quantile(diffs, [q, 1 - q])
    return {"gap": float(stat(a) - stat(b)), "ci_low": float(lo), "ci_high": float(hi),
            "n_boot": n_boot, "seed": seed}


def bootstrap_mean_ci(x, n_boot: int = N_BOOT, seed: int = BOOT_SEED, level: float = 0.95) -> dict:
    """Mean of the finite values of x and its percentile CI, resampling subjects with replacement."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0}
    m = np.random.default_rng(seed).integers(0, len(x), size=(n_boot, len(x)))
    q = (1 - level) / 2
    lo, hi = np.quantile(x[m].mean(1), [q, 1 - q])
    return {"mean": float(x.mean()), "ci_low": float(lo), "ci_high": float(hi), "n": int(len(x))}


def cohens_dz(a, b) -> float:
    """mean(a - b) / sd(a - b, ddof = 1); NaN when the differences are constant."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    sd = d.std(ddof=1) if len(d) > 1 else 0.0
    return float(d.mean() / sd) if sd > 0 else np.nan


def paired_wilcoxon(a, b) -> dict:
    """Two-sided Wilcoxon signed-rank of a vs b (pairs with a NaN dropped; zero_method="wilcox"), with the
    median and mean difference, the matched-pairs rank-biserial r = (W+ - W-) / (W+ + W-) over the
    non-zero differences (r > 0: a tends to be larger) and Cohen's d_z [plan 2.8]."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    d = a - b
    nz = d[d != 0]
    out = {"n": int(len(d)), "n_nonzero_diff": int(len(nz)), "median_diff": float(np.median(d)) if len(d) else np.nan,
           "mean_diff": float(d.mean()) if len(d) else np.nan, "cohens_dz": cohens_dz(a, b)}
    if len(nz) == 0:
        return {**out, "stat": np.nan, "p": 1.0, "rank_biserial_r": 0.0}
    ranks = rankdata(np.abs(nz))
    wp, wm = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    res = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return {**out, "stat": float(res.statistic), "p": float(res.pvalue), "rank_biserial_r": float((wp - wm) / (wp + wm))}


def holm(pvals) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (NaN entries stay NaN and do not count towards m)."""
    p = np.asarray(pvals, float)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    m = len(ok)
    if m == 0:
        return out
    order = ok[np.argsort(p[ok], kind="stable")]
    adj = np.minimum(1.0, np.maximum.accumulate(p[order] * (m - np.arange(m))))
    out[order] = adj
    return out


def bootstrap_fa_lci_ci(older_A, general_A, older_B=None, general_B=None, lam: float = 0.5,
                        n_boot: int = N_BOOT, seed: int = BOOT_SEED, level: float = 0.95) -> dict:
    """FA-LCI = (LCI_older + LCI_general) / 2 - lam |LCI_older - LCI_general| of method A, with a percentile
    CI over subjects resampled within each group. With B (same subjects, same order), also the paired
    difference FA_A - FA_B, using the same resampled subjects for both methods in every draw."""
    oa, ga = np.asarray(older_A, float), np.asarray(general_A, float)
    rng = np.random.default_rng(seed)
    io = rng.integers(0, len(oa), size=(n_boot, len(oa)))
    ig = rng.integers(0, len(ga), size=(n_boot, len(ga)))
    fa = lambda o, g: (o + g) / 2 - lam * np.abs(o - g)
    q = (1 - level) / 2
    dist_a = fa(oa[io].mean(1), ga[ig].mean(1))
    lo, hi = np.quantile(dist_a, [q, 1 - q])
    out = {"fa_lci": float(fa(oa.mean(), ga.mean())), "ci_low": float(lo), "ci_high": float(hi), "n_boot": n_boot,
           "seed": seed}
    if older_B is not None:
        ob, gb = np.asarray(older_B, float), np.asarray(general_B, float)
        if ob.shape != oa.shape or gb.shape != ga.shape:
            raise ValueError("method B must have the same subjects as method A")
        dist = dist_a - fa(ob[io].mean(1), gb[ig].mean(1))
        dlo, dhi = np.quantile(dist, [q, 1 - q])
        out.update({"fa_lci_b": float(fa(ob.mean(), gb.mean())),
                    "diff": float(fa(oa.mean(), ga.mean()) - fa(ob.mean(), gb.mean())),
                    "diff_ci_low": float(dlo), "diff_ci_high": float(dhi)})
    return out


def safe_rate(lci_values, tau: float) -> float:
    """Share of the (finite) settings with LCI >= tau [plan 2.8]."""
    v = np.asarray(lci_values, float)
    v = v[np.isfinite(v)]
    return float((v >= tau).mean()) if len(v) else np.nan
