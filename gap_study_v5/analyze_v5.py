"""Gap study v5: validation analysis (v3 + v4 + v5 rows, HARTH inner val, 5 folds).

    python -m gap_study_v5.analyze_v5

Writes results/gap-study-v5/validation/:
  gap-study-v5-validation-tinyhar-summary-mean-over-folds-seeds.csv  per method x level (v3 + v4 + v5 rows)
  gap-study-v5-validation-tinyhar-per-fold-lci-indep.csv              per fold, seed-mean lci_indep, 80 % and 90 %
  gap-study-v5-validation-tinyhar-paired-gaps.csv                     fold-paired differences between named pairs
  gap-study-v5-validation-tinyhar-ablation-where.csv                  unit-only pruning and same-mask weight swaps
  gap-study-v5-validation-tinyhar-shift-probe-summary.csv             shift probe (noise / rotation), per model
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gap_study_v4.analyze_v4 as A4  # noqa: E402

V5 = ROOT / "results" / "gap-study-v5" / "validation"
P = "gap-study-v5-validation-tinyhar"
NUM = A4.NUM


def label5(r) -> str:
    if r["stage"] == "ref":
        return "original"
    if r["stage"] == "orc":
        ep = r.get("epochs")
        return f"jfm_from_{r['start']} [lr {r['lr']:g}] lam{r['lam']:g}" + (f" e{int(ep)}" if pd.notna(ep) else "")
    if r["stage"] in ("orc2", "cap"):
        return f"{r['method']}_from_{r['start'].split('-')[0]} [lr {r['lr']:g}] lam{r['lam']:g} e{int(r['epochs'])}"
    if r["stage"] == "fmo":
        st = r["start"].split("-")[0]
        if st.startswith("alloc_"):          # started from the fold's KL-chosen allocation (a3, or a4 on fold 3 at 90 %)
            st = "alloc_kl"
        return f"fmkd_from_{st} [lr {r['lr']:g}] e{int(r['epochs'])}"
    return r["method"]


def load() -> pd.DataFrame:
    d4 = A4.load()
    parts = [d4]
    for f in sorted(V5.glob(f"{P}-fold*-models.csv")):
        if "-shift-" in f.name:
            continue
        d = pd.read_csv(f)
        d["src"] = "v5"
        d["label"] = d.apply(label5, axis=1)
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d = d.drop_duplicates(["src", "fold", "run_id", "label"])
    # the label-free, backprop-free allocation choice: per (fold, level, calibration seed) the allocation with the lowest calibration KL
    al = d[d.label.astype(str).str.startswith("alloc_a") & d.cal_kl.notna()]
    if len(al):
        pick = al.loc[al.groupby(["fold", "level", "seed"]).cal_kl.idxmin()].copy()
        pick["chosen"] = pick.label
        pick["label"] = "alloc_kl (OBC + refit, KL-chosen allocation)"
        d = pd.concat([d, pick], ignore_index=True)
    d = d[~((d.label == "original") & (d.src != "v3"))]
    return d


def main() -> None:
    V5.mkdir(parents=True, exist_ok=True)
    d = load()
    u = d[d.label == "original"].groupby("fold").lci_indep.mean()
    acc0 = d[d.label == "original"].groupby("fold").accuracy.mean()
    f10 = d[d.label == "original"].groupby("fold").f1.mean()
    pf = d.groupby(["label", "level", "fold"])[NUM].mean().reset_index()
    pf["u"] = pf.fold.map(u)
    pf["d_acc_pt"] = 100 * (pf.accuracy - pf.fold.map(acc0))
    pf["d_f1_pt"] = 100 * (pf.f1 - pf.fold.map(f10))
    pf["gap_to_u"] = pf.u - pf.lci_indep
    n = d.groupby(["label", "level"]).agg(n_folds=("fold", "nunique"), n_runs=("fold", "size"),
                                          src=("src", "first")).reset_index()
    s = pf.groupby(["label", "level"]).agg(**{c: (c, "mean") for c in NUM + ["u", "d_acc_pt", "d_f1_pt", "gap_to_u"]},
                                            lci_indep_fold_sd=("lci_indep", "std")).reset_index()
    s = s.merge(n, on=["label", "level"])
    wsd = d.groupby(["label", "level", "fold"]).lci_indep.std().groupby(["label", "level"]).mean()
    s["within_fold_seed_sd"] = s.set_index(["label", "level"]).index.map(wsd)
    s = s.sort_values(["level", "lci_indep"], ascending=[True, False])
    abl = s.label.str.startswith(("unitonly_", "swapin_", "swapout_"))
    s[~abl].round(4).to_csv(V5 / f"{P}-summary-mean-over-folds-seeds.csv", index=False, lineterminator="\n")
    s[abl].round(4).to_csv(V5 / f"{P}-ablation-where.csv", index=False, lineterminator="\n")
    w = pf[pf.level.isin([0.8, 0.9]) & ~pf.label.str.startswith(("unitonly_", "swapin_", "swapout_"))]
    w = w.pivot_table(index=["level", "label"], columns="fold", values="lci_indep")
    w.round(4).to_csv(V5 / f"{P}-per-fold-lci-indep.csv", lineterminator="\n")

    labels = set(pf.label)
    pairs = []

    def pair(a, b, lev, note):
        if a not in labels or b not in labels:
            return
        A = pf[(pf.label == a) & (pf.level == lev)].set_index("fold")
        B = pf[(pf.label == b) & (pf.level == lev)].set_index("fold")
        f = A.index.intersection(B.index)
        if len(f) == 0:
            return
        dl = A.loc[f, "lci_indep"] - B.loc[f, "lci_indep"]
        do = A.loc[f, "occ_lci"] - B.loc[f, "occ_lci"]
        pairs.append({"level": lev, "a": a, "b": b, "note": note, "n_folds": len(f),
                      "a_lci_indep": A.loc[f, "lci_indep"].mean(), "b_lci_indep": B.loc[f, "lci_indep"].mean(),
                      "diff_lci_indep": dl.mean(), "ci95": A4.ci95(dl), "per_fold": " ".join(f"{v:.3f}" for v in dl),
                      "folds_ge_0.05": int((dl >= 0.05).sum()), "folds_positive": int((dl > 0).sum()),
                      "diff_occ_lci": do.mean(), "occ_ci95": A4.ci95(do),
                      "diff_acc_pt": 100 * (A.loc[f, "accuracy"] - B.loc[f, "accuracy"]).mean(),
                      "diff_f1_pt": 100 * (A.loc[f, "f1"] - B.loc[f, "f1"]).mean(),
                      "a_acc": A.loc[f, "accuracy"].mean(), "b_acc": B.loc[f, "accuracy"].mean()})

    refs80 = ["fmkd_from_l1_fullrecon [lr 0.0003] e40", "calfm_from_l1_fullrecon [lr 0.001]"]
    refs80 += sorted(l for l in labels if l.startswith("jfm_from_l1_fullrecon"))
    nr80 = ["obc_fullrecon", "sparsegpt_fullrecon", "dsnot_fullrecon", "l1_fullrecon_bnrecal", "l1_fullrecon_k20",
            "l1_fullrecon_mixcal", "l1_fullrecon", "obc", "l1_fullrecon_probecal"]
    for x in refs80:
        for b in nr80:
            pair(x, b, 0.8, "same-size reference vs no-retrain baseline")
    for b in nr80:
        pair(b, "l1_fullrecon", 0.8, "no-retrain baseline vs l1_fullrecon")
        pair(b, "l1_fullrecon_mixcal", 0.8, "no-retrain baseline vs v4's best (mixcal)")
    for x in sorted(l for l in labels if l.startswith("jfm_from_lamp_recon")) + ["fmkd_from_lamp_recon [lr 0.0003]"]:
        for b in ("lamp_recon_kd [lr 0.0003]", "fmkd_from_lamp_recon [lr 0.0003]", "calfm_from_lamp_recon [lr 0.0003]",
                  "obc_fullrecon", "l1_fullrecon", "wanda_fullrecon"):
            if x != b:
                pair(x, b, 0.9, "90 %: reference vs baseline")
    for x in sorted(l for l in labels if l.startswith("jfm_from_l1_fullrecon")):
        pair(x, "fmkd_from_l1_fullrecon [lr 0.0003] e40", 0.8, "oracle vs fmkd (same masks)")
    AK = "alloc_kl (OBC + refit, KL-chosen allocation)"
    FA = "fmkd_from_alloc_kl [lr 0.0003] e20"
    CP = "capfit_from_obc_fullrecon [lr 0.0003] lam10 e20"
    C9 = "capfit_from_lamp_recon [lr 0.0003] lam10 e20"
    for b in ("l1_fullrecon_mixcal", "obc_fullrecon", "alloc_a0", "fmkd_from_l1_fullrecon [lr 0.0003] e40"):
        pair(AK, b, 0.8, "KL-chosen allocation vs baseline / reference")
    pair(CP, AK, 0.8, "capacity oracle vs strongest existing one-shot")
    pair(CP, "fmkd_from_l1_fullrecon [lr 0.0003] e40", 0.8, "capacity oracle vs fmkd")
    pair("alloc_a2", AK, 0.8, "LCI-friendly fixed allocation vs KL choice")
    for b in ("lamp_recon_kd [lr 0.0003]", "fmkd_from_lamp_recon [lr 0.0003]", "calfm_from_lamp_recon [lr 0.0003]",
              "l1_kd [lr 0.0003]", AK, "obc_fullrecon"):
        pair(FA, b, 0.9, "strongest existing 90 % combination vs baseline")
    pair(C9, FA, 0.9, "capacity oracle vs strongest existing 90 %")
    pair(AK, "obc_fullrecon", 0.9, "KL-chosen allocation vs uniform, 90 % one shot")
    pair("obc_fullrecon", "wanda_fullrecon", 0.9, "90 % one shot")
    pair("obc_fullrecon", "l1_fullrecon", 0.9, "90 % one shot")
    pair("obc_fullrecon", "l1_fullrecon", 0.7, "70 % one shot")
    pt = pd.DataFrame(pairs)
    pt.round(4).to_csv(V5 / f"{P}-paired-gaps.csv", index=False, lineterminator="\n")

    sh = sorted(V5.glob(f"{P}-fold*-Ansel-shift-models.csv"))
    if sh:
        z = pd.concat([pd.read_csv(f) for f in sh])
        z["gap_to_ceiling"] = z.ceiling - z.lci_indep
        zs = z.groupby(["condition", "model"]).agg(n_folds=("fold", "nunique"), lci_indep=("lci_indep", "mean"),
                                                   ceiling=("ceiling", "mean"), gap=("gap_to_ceiling", "mean"),
                                                   accuracy=("accuracy", "mean"), f1=("f1", "mean"),
                                                   agree_original=("agree_original", "mean")).reset_index()
        zs.round(4).to_csv(V5 / f"{P}-shift-probe-summary.csv", index=False, lineterminator="\n")
        print(zs.round(3).to_string(index=False))

    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 50)
    cols = ["label", "level", "src", "n_folds", "n_runs", "accuracy", "d_acc_pt", "f1", "lci_indep", "gap_to_u",
            "self_lci", "occ_lci", "agree", "within_fold_seed_sd"]
    print(s[~abl & s.level.isin([0.7, 0.8, 0.9])][cols].round(3).head(60).to_string(index=False))
    print(s[abl][cols].round(3).to_string(index=False))
    if len(pt):
        print(pt[["level", "a", "b", "n_folds", "a_lci_indep", "b_lci_indep", "diff_lci_indep", "ci95", "per_fold",
                  "folds_positive", "diff_occ_lci", "diff_acc_pt"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
