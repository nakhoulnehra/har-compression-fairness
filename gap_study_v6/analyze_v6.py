"""Gap study v6 analysis (validation only). Merges the v6 models CSVs, writes per-fold and mean tables, fold-paired
gaps (t-based 95 % CI over folds) and the KL-decoupling check: for each fold, LCI is regressed on log(cal KL) over v5's
in-place pruned models (one-shot and fine-tuned, 80 / 90 %); a v6 student's residual says how far its explanation
falls below what its output fidelity predicts for in-basin compression.

    python -m gap_study_v6.analyze_v6
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
V6 = ROOT / "results" / "gap-study-v6" / "validation"
V5 = ROOT / "results" / "gap-study-v5" / "validation"
P = "gap-study-v6-validation-tinyhar"
PAIRS = [   # (reference / oracle, baseline): x - y
    ("capfit-w12-lr0.001-e30", "fmkd-w12-lr0.001-e30"), ("capfit-w12-lr0.001-e30", "slice_fmkd-w12-lr0.0003-e30"),
    ("slice_capfit-w12-lr0.0003-e30", "slice_fmkd-w12-lr0.0003-e30"),
    ("slice_fmkd-w20-lr0.0003-e30", "fmkd-w20-lr0.001-e30"),
    ("gptq-3bit", "rtn-3bit"), ("gptq-2bit", "rtn-2bit"),
    ("qat_fmkd_from_gptq-2bit-lr0.0001-e20", "gptq-2bit"),
    ("qat_fmkd_p8_from_gptq-2bit-lr0.0001-e20", "gptq_p8-2bit"),
    ("qat_fmkd_p8_from_gptq-2bit-lr3e-05-e20", "gptq_p8-2bit"),
    ("qat_fmkd_p8_from_gptq-2bit-lr0.0003-e20", "gptq_p8-2bit"),
    ("qat_fmkd_p8_from_gptq-2bit-lr0.0001-e20", "gptq-2bit"),
    ("qat_fmkd_from_gptq-2bit-lr0.0001-e20", "gptq_p8-2bit"),
]


def load() -> pd.DataFrame:
    fs = [f for f in glob.glob(str(V6 / f"{P}-fold*-models.csv"))]
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    d = d.drop_duplicates(["fold", "run_id"], keep="last")
    d["key"] = np.where(d.stage == "ref", "original", d.run_id.str.replace("-seed0", "", regex=False))
    return d


def paired(d: pd.DataFrame, a: str, b: str, col: str = "lci_indep") -> dict:
    p = d.pivot_table(index="fold", columns="key", values=col)
    if a not in p or b not in p:
        return {}
    x = (p[a] - p[b]).dropna()
    n = len(x)
    if n < 2:
        return {"a": a, "b": b, "n_folds": n, "mean": float(x.mean()) if n else np.nan}
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return {"a": a, "b": b, "metric": col, "n_folds": n, "mean": float(x.mean()), "ci_lo": float(x.mean() - h),
            "ci_hi": float(x.mean() + h), "n_positive": int((x > 0).sum()), "min": float(x.min()),
            "folds": " ".join(str(int(f)) for f in x.index)}


def v5_inplace() -> pd.DataFrame:
    fs = glob.glob(str(V5 / "*-models.csv"))
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    d = d[d.get("cal_kl").notna() & (d.stage != "ref")] if "cal_kl" in d else d.iloc[0:0]
    d = d[~d.run_id.str.startswith(("swap", "unitonly", "capfit", "jfm", "capS"))]
    return d.drop_duplicates(["fold", "run_id"])


def decoupling(d6: pd.DataFrame) -> pd.DataFrame:
    v5 = v5_inplace()
    rows = []
    for k, g in v5.groupby("fold"):
        g = g[g.cal_kl > 0]
        if len(g) < 5:
            continue
        sl, ic, r, _, _ = stats.linregress(np.log(g.cal_kl), g.lci_indep)
        for _, s in d6[(d6.fold == k) & (d6.stage == "stu")].iterrows():
            if not np.isfinite(s.get("cal_kl", np.nan)) or s.cal_kl <= 0:
                continue
            pred = ic + sl * np.log(s.cal_kl)
            near = g[(g.cal_kl > s.cal_kl / 1.5) & (g.cal_kl < s.cal_kl * 1.5)]
            rows.append({"fold": k, "key": s.key, "cal_kl": s.cal_kl, "lci_indep": s.lci_indep,
                         "lci_pred_inplace": pred, "residual": s.lci_indep - pred, "fit_r": r, "n_inplace": len(g),
                         "inplace_same_kl_mean_lci": near.lci_indep.mean() if len(near) else np.nan,
                         "inplace_same_kl_n": len(near)})
    return pd.DataFrame(rows)


def main() -> None:
    d = load()
    cols = ["lci_indep", "f1", "accuracy", "occ_lci", "self_lci", "agree", "cal_kl", "val_kl", "worst_subject_lci"]
    per_fold = d.pivot_table(index="key", columns="fold", values="lci_indep")
    per_fold.to_csv(V6 / f"{P}-lci-indep-per-fold.csv")
    orig = d[d.key == "original"].set_index("fold")
    d["acc_delta"] = d.accuracy - d.fold.map(orig.accuracy)
    d["gap_to_ceiling"] = d.fold.map(orig.lci_indep) - d.lci_indep
    summ = d.groupby("key").agg(n_folds=("fold", "nunique"), **{c: (c, "mean") for c in cols + ["acc_delta",
                                                                                                "gap_to_ceiling"]})
    summ = summ.sort_values("lci_indep", ascending=False)
    summ.to_csv(V6 / f"{P}-summary-mean-over-folds.csv")
    print(summ.round(4).to_string())
    print(per_fold.round(3).to_string())
    keys = list(summ.index)
    pairs = []
    for o, b in PAIRS:
        for col in ("lci_indep", "occ_lci", "accuracy"):
            pairs.append(paired(d, o, b, col))
    for b in keys:
        if b != "original":
            pairs.append(paired(d, "original", b))
    pr = pd.DataFrame([p for p in pairs if p])
    pr.to_csv(V6 / f"{P}-paired-gaps.csv", index=False)
    if len(pr):
        print(pr.round(4).to_string())
    dc = decoupling(d)
    if len(dc):
        dc.to_csv(V6 / f"{P}-kl-decoupling.csv", index=False)
        print(dc.groupby("key")[["residual", "lci_indep", "lci_pred_inplace", "inplace_same_kl_mean_lci"]]
              .agg(["mean", "count"]).round(4).to_string())


if __name__ == "__main__":
    main()
