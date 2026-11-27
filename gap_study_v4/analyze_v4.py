"""Gap study v4: validation analysis (v3 rows + v4 rows, HARTH inner val, 5 folds).

    python -m gap_study_v4.analyze_v4

Writes results/gap-study-v4/validation/:
  gap-study-v4-validation-tinyhar-every-model-long.csv               one row per (fold, model) from v3 and v4
  gap-study-v4-validation-tinyhar-summary-mean-over-folds-seeds.csv  per method x level: means, n, fold-level SDs
  gap-study-v4-validation-tinyhar-per-fold-lci-indep-80pct.csv        80 %: per fold, seed-mean lci_indep per method
  gap-study-v4-validation-tinyhar-paired-gaps.csv                     fold-paired differences between named pairs

The primary explanation metric is lci_indep (model SHAP seed 1 vs original seed 0), comparable with u = the
original's SHAP-noise ceiling (seed 1 vs seed 0) on the same 150 explain windows per fold.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "results" / "gap-study-v3" / "phase2"
V4 = ROOT / "results" / "gap-study-v4" / "validation"
P = "gap-study-v4-validation-tinyhar"
NUM = ["f1", "accuracy", "lci", "lci_indep", "self_lci", "occ_lci", "agree", "lci_agree"]


def label(r) -> str:
    m = r["method"]
    lr = r.get("lr")
    if r["src"] == "v3":
        if r["stage"] in ("ref",):
            return "original"
        if r["stage"] in ("rescore", "nr", "refs2"):
            return m + (f" [pdamp {r['pdamp']:g}]" if pd.notna(r.get("pdamp")) else "")
        if r["stage"] == "rescore9a":
            return m
        return f"{m} [lr {lr:g}]"
    if r["stage"] == "ref":
        return "original"
    if r["stage"] == "nr4":
        return m
    ep = r.get("epochs")
    return f"{m}_from_{r['start']} [lr {lr:g}]" + (f" e{int(ep)}" if pd.notna(ep) else "")


def load() -> pd.DataFrame:
    parts = []
    for f in sorted(V3.glob("gap-study-v3-phase2-tinyhar-fold*-models.csv")):
        d = pd.read_csv(f)
        d["src"] = "v3"
        parts.append(d)
    for f in sorted(V4.glob(f"{P}-fold*-models.csv")):
        d = pd.read_csv(f)
        d["src"] = "v4"
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d["label"] = d.apply(label, axis=1)
    d = d.drop_duplicates(["src", "fold", "run_id", "label"])
    d = d[~((d.label == "original") & (d.src == "v4"))]
    return d


def ci95(x) -> str:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return ""
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return f"[{x.mean() - h:.3f}, {x.mean() + h:.3f}]"


def main() -> None:
    d = load()
    keep = ["src", "fold", "stage", "label", "method", "level", "seed", "lr", "start", "ceiling50", "n_nonzero"] + NUM
    d[[c for c in keep if c in d.columns]].round(4).to_csv(V4 / f"{P}-every-model-long.csv", index=False,
                                                           lineterminator="\n")
    u = d[d.label == "original"].groupby("fold").lci_indep.mean()
    acc0 = d[d.label == "original"].groupby("fold").accuracy.mean()
    f10 = d[d.label == "original"].groupby("fold").f1.mean()
    # per fold: seed mean
    pf = d.groupby(["label", "level", "fold"])[NUM].mean().reset_index()
    pf["u"] = pf.fold.map(u)
    pf["d_acc_pt"] = 100 * (pf.accuracy - pf.fold.map(acc0))
    pf["d_f1_pt"] = 100 * (pf.f1 - pf.fold.map(f10))
    pf["gap_to_u"] = pf.u - pf.lci_indep
    n = d.groupby(["label", "level"]).agg(n_folds=("fold", "nunique"), n_runs=("fold", "size")).reset_index()
    s = pf.groupby(["label", "level"]).agg(**{c: (c, "mean") for c in NUM + ["u", "d_acc_pt", "d_f1_pt", "gap_to_u"]},
                                            lci_indep_fold_sd=("lci_indep", "std")).reset_index()
    s = s.merge(n, on=["label", "level"])
    wsd = d.groupby(["label", "level", "fold"]).lci_indep.std().groupby(["label", "level"]).mean()
    s["within_fold_seed_sd"] = s.set_index(["label", "level"]).index.map(wsd)
    s = s.sort_values(["level", "lci_indep"], ascending=[True, False])
    s.round(4).to_csv(V4 / f"{P}-summary-mean-over-folds-seeds.csv", index=False, lineterminator="\n")
    w = pf[pf.level == 0.8].pivot(index="fold", columns="label", values="lci_indep")
    w["u"] = u
    w.round(4).to_csv(V4 / f"{P}-per-fold-lci-indep-80pct.csv", lineterminator="\n")

    pairs = []
    labels = set(pf.label)

    def pair(a: str, b: str, lev: float, note: str):
        if a not in labels or b not in labels:
            return
        A = pf[(pf.label == a) & (pf.level == lev)].set_index("fold")
        B = pf[(pf.label == b) & (pf.level == lev)].set_index("fold")
        f = A.index.intersection(B.index)
        if len(f) == 0:
            return
        dl = A.loc[f, "lci_indep"] - B.loc[f, "lci_indep"]
        do = A.loc[f, "occ_lci"] - B.loc[f, "occ_lci"]
        da = 100 * (A.loc[f, "accuracy"] - B.loc[f, "accuracy"])
        dfi = 100 * (A.loc[f, "f1"] - B.loc[f, "f1"])
        pairs.append({"level": lev, "a": a, "b": b, "note": note, "n_folds": len(f), "a_lci_indep": A.loc[f, "lci_indep"].mean(),
                      "b_lci_indep": B.loc[f, "lci_indep"].mean(), "diff_lci_indep": dl.mean(), "ci95": ci95(dl),
                      "per_fold": " ".join(f"{v:.3f}" for v in dl), "folds_ge_0.05": int((dl >= 0.05).sum()),
                      "folds_positive": int((dl > 0).sum()), "diff_occ_lci": do.mean(), "occ_ci95": ci95(do),
                      "diff_acc_pt": da.mean(), "diff_f1_pt": dfi.mean(),
                      "a_acc": A.loc[f, "accuracy"].mean(), "b_acc": B.loc[f, "accuracy"].mean()})

    B80 = "l1_fullrecon"
    for x in sorted(l for l in labels if ("_from_" in l or l.startswith("l1_kd [") or l.startswith("l1_fullrecon_kd ["))):
        pair(x, B80, 0.8, "same-size reference vs best no-retrain baseline")
        pair(x, "lamp_recon_kd [lr 0.0003]", 0.9, "vs v3 best fine-tuned 90 % baseline")
    for b in ("ria_fullrecon", "l1_fullrecon_mixcal", "wanda_fullrecon", "l1_fullrecon_cal5x", "taylor_fullrecon"):
        pair(b, B80, 0.8, "no-retrain baseline check vs l1_fullrecon")
    # label-free scenario (decision points 2 and 3): full-data same-size references vs label-free baselines
    lf = sorted(l for l in labels if l.startswith(("calfm", "calkd")))
    for xl, lev in (("fmkd_from_l1_fullrecon [lr 0.0003]", 0.8), ("fmkd_from_l1_fullrecon [lr 0.0003] e40", 0.8),
                    ("lamp_recon_kd [lr 0.0003]", 0.9), ("fmkd_from_lamp_recon [lr 0.0003]", 0.9)):
        for b in lf + ["l1_fullrecon", "l1_fullrecon_mixcal", "wanda_fullrecon"]:
            pair(xl, b, lev, "label-free scenario: full-data reference vs label-free / no-retrain baseline")
    pt = pd.DataFrame(pairs)
    pt.round(4).to_csv(V4 / f"{P}-paired-gaps.csv", index=False, lineterminator="\n")

    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 48)
    cols = ["label", "level", "n_folds", "n_runs", "accuracy", "d_acc_pt", "f1", "d_f1_pt", "lci", "lci_indep",
            "gap_to_u", "self_lci", "occ_lci", "agree", "within_fold_seed_sd"]
    print(s[s.level.isin([0.8, 0.9])][cols].round(3).to_string(index=False))
    print(w.round(3).T.to_string())
    print(pt.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
