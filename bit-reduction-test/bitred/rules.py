"""Pre-registered design constants and decision rules (fixed before any real run; see README.md).

Units: folds (deterministic quantization, no seeds). Metrics on the inner-validation windows of each fold:
LCI_val (subject mean over the 3 inner-val subjects), worst-subject LCI_val, F1_val (pooled 7-class macro-F1).
"""

from __future__ import annotations

from math import ceil

import numpy as np
import pandas as pd

BIT_OPTIONS = (2, 3, 4, 8)            # candidate bit widths per tensor (mixed precision)
SENS_BITS = (2, 3, 4, 8)              # bit widths at which every sensitivity is measured
PROBE_BITS = (8, 4, 3, 2)             # uniform reference models (8-bit: per-channel only)
METHODS = ("xgq", "zeroq_kl", "hawq")  # ours first
# Budgets = size(uniform per-channel 2-bit) + f * (size(uniform 3-bit) - size(uniform 2-bit)). 2 bits is the smallest
# option, so a budget equal to the uniform 2-bit size would force every method to all-2-bit: the lowest budget is 2.25.
BUDGETS = ("3bit", "2.5bit", "2.25bit")
BUDGET_FRACTION = {"3bit": 1.0, "2.5bit": 0.5, "2.25bit": 0.25}
BUDGET_BITWIDTHS = {"3bit": (3,), "2.5bit": (3, 2), "2.25bit": (2,)}
UNIFORM_AT_BUDGET = {"3bit": "uniform-channel-3bit"}   # the only budget that a uniform model meets exactly
HUTCHINSON = {"n_iter": 100, "seed": 49}
SMOKE_HUTCHINSON = {"n_iter": 4, "seed": 49}
FOLDS = (0, 1, 2, 3, 4)

HEADROOM_DROP = 0.05                  # R1
WIN = {"lci_gain": 0.02, "fold_share": 0.8, "f1_tol": 0.01, "worst_tol": 0.01}   # R2

RULES = {
    "R1_headroom": "At b in {3, 2}: explanation headroom exists at b bits iff the mean over folds of "
                   "LCI_val(uniform-channel-4bit) - LCI_val(uniform-channel-b bit) >= 0.05.",
    "R2_beats": "XGQ beats a comparator M at budget B iff all hold over the folds: (a) mean LCI_val gain >= 0.02; "
                "(b) LCI_val strictly higher in >= ceil(0.8 * n_folds) folds (an identical allocation counts as not "
                "higher); (c) mean F1_val >= M's - 0.01; (d) mean worst-subject LCI_val >= M's - 0.01.",
    "R3_claim": "XGQ beats existing allocation at B iff it beats every comparator at B: zeroq_kl, hawq, and (3bit "
                "budget only) the uniform per-channel 3-bit model of the same size. The contribution is supported iff "
                "R3 holds at >= 1 budget whose bit width has headroom under R1 (3bit -> 3; 2.5bit -> 3 or 2; "
                "2.25bit -> 2). Otherwise it is reported as not supported; nothing is retuned.",
    "R4_no_tuning": "No method has a hyperparameter chosen on inner-val. Bit options, budgets, calibration windows, "
                    "Hutchinson iterations and seeds are fixed in advance; every fold is reported.",
}


def comparators(budget: str) -> list[str]:
    out = [f"alloc-{m}-{budget}" for m in METHODS if m != "xgq"]
    if budget in UNIFORM_AT_BUDGET:
        out.append(UNIFORM_AT_BUDGET[budget])
    return out


def headroom(per_fold: pd.DataFrame) -> pd.DataFrame:
    """R1 from per-fold rows (model_id, fold, lci, f1). One row per bit width in {3, 2}."""
    rows = []
    p = per_fold.pivot_table(index="fold", columns="model_id", values=["lci", "f1"])
    ref = "uniform-channel-4bit"
    for b in (3, 2):
        mid = f"uniform-channel-{b}bit"
        if ("lci", ref) not in p or ("lci", mid) not in p:
            continue
        d_lci = p["lci"][ref] - p["lci"][mid]
        d_f1 = p["f1"][ref] - p["f1"][mid]
        rows.append({"bits": b, "n_folds": int(d_lci.notna().sum()), "lci_4bit": float(p["lci"][ref].mean()),
                     f"lci_{b}bit": float(p["lci"][mid].mean()), "lci_drop_mean": float(d_lci.mean()),
                     "lci_drop_min": float(d_lci.min()), "lci_drop_max": float(d_lci.max()),
                     "f1_drop_mean": float(d_f1.mean()), "headroom": bool(d_lci.mean() >= HEADROOM_DROP)})
    return pd.DataFrame(rows)


def compare(per_fold: pd.DataFrame, ours: str, other: str) -> dict:
    """R2 for one pair, from per-fold rows (model_id, fold, lci, f1, worst_subject_lci)."""
    a = per_fold[per_fold.model_id == ours].set_index("fold")
    b = per_fold[per_fold.model_id == other].set_index("fold")
    folds = sorted(set(a.index) & set(b.index))
    if not folds:
        return {"ours": ours, "comparator": other, "n_folds": 0, "beats": False}
    a, b = a.loc[folds], b.loc[folds]
    d_lci = a.lci - b.lci
    d_f1 = a.f1 - b.f1
    d_w = a.worst_subject_lci - b.worst_subject_lci
    need = ceil(WIN["fold_share"] * len(folds))
    n_higher = int((d_lci > 0).sum())
    crit = {"a_gain": bool(d_lci.mean() >= WIN["lci_gain"]), "b_folds": bool(n_higher >= need),
            "c_f1": bool(d_f1.mean() >= -WIN["f1_tol"]), "d_worst": bool(d_w.mean() >= -WIN["worst_tol"])}
    return {"ours": ours, "comparator": other, "n_folds": len(folds), "lci_ours": float(a.lci.mean()),
            "lci_comparator": float(b.lci.mean()), "d_lci_mean": float(d_lci.mean()),
            "d_lci_per_fold": ";".join(f"{x:+.4f}" for x in d_lci), "n_folds_higher": n_higher, "n_folds_needed": need,
            "f1_ours": float(a.f1.mean()), "f1_comparator": float(b.f1.mean()), "d_f1_mean": float(d_f1.mean()),
            "d_worst_mean": float(d_w.mean()), **crit, "beats": all(crit.values())}


def decide(per_fold: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """R1-R3 from per-fold rows. Returns (decision, headroom table, comparison table)."""
    hr = headroom(per_fold)
    hr_bits = {int(r.bits): bool(r.headroom) for r in hr.itertuples()}
    comps, by_budget = [], {}
    for B in BUDGETS:
        ours = f"alloc-xgq-{B}"
        rs = [compare(per_fold, ours, m) for m in comparators(B)]
        comps += [{"budget": B, **r} for r in rs]
        present = [r for r in rs if r["n_folds"] > 0]
        beats_all = bool(present) and len(present) == len(rs) and all(r["beats"] for r in present)
        hr_ok = any(hr_bits.get(b, False) for b in BUDGET_BITWIDTHS[B])
        by_budget[B] = {"beats_all_comparators": beats_all, "headroom_at_bitwidth": hr_ok,
                        "supported": beats_all and hr_ok,
                        "beaten": [r["comparator"] for r in rs if r.get("beats")],
                        "not_beaten": [r["comparator"] for r in rs if not r.get("beats")]}
    dec = {"headroom_by_bits": hr_bits, "by_budget": by_budget,
           "contribution_supported": any(v["supported"] for v in by_budget.values()), "rules": RULES}
    return dec, hr, pd.DataFrame(comps)


def spearman(x, y) -> float:
    x, y = pd.Series(np.asarray(x, float)), pd.Series(np.asarray(y, float))
    return float(x.rank().corr(y.rank()))
