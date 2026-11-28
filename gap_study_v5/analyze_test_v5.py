"""Gap study v5: held-out test summary (HARTH test subjects = "general", HAR70+ = "older"), from the per-subject
files of gap_study_v4.test_confirm run with --out results/gap-study-v5/test. v4's --analyze fails on its pooled-F1
label strings (one subject's string is one character short), so pooled F1 is left out; everything else is v4's
definition: subject level (HAR70+ subjects averaged over the 5 folds), lci_indep vs the per-subject ceiling.

    python -m gap_study_v5.analyze_test_v5 --models OBC OBC5X FMKD ... [--pairs A:B ...]
Writes results/gap-study-v5/test/gap-study-v5-test-tinyhar-{summary-per-group-and-model,paired-differences}.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "gap-study-v5" / "test"
NUM = ["lci_same", "lci_indep", "ceiling", "occ_lci", "agree", "accuracy"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--pairs", nargs="*", default=[])
    a = ap.parse_args()
    d = pd.concat([pd.read_csv(f, dtype={"subject": str}, usecols=lambda c: c not in ("y", "pred"))
                   for f in sorted(OUT.glob("*-fold*-per-subject.csv"))])
    folds = d.groupby("label").fold.nunique()
    subj = d.groupby(["label", "group", "subject"])[NUM].mean().reset_index()
    rows = []
    for g in ("general", "older", "all"):
        sg = subj if g == "all" else subj[subj.group == g]
        for l in ["original"] + a.models:
            x = sg[sg.label == l]
            gap = x.ceiling - x.lci_indep
            rows.append({"group": g, "label": l, "n_folds": int(folds.get(l, 0)), "n_subjects": len(x),
                         "lci_indep": x.lci_indep.mean(), "ceiling": x.ceiling.mean(), "gap_to_ceiling": gap.mean(),
                         "subjects_gap_ge_0.10": int((gap >= 0.10).sum()), "lci_same": x.lci_same.mean(),
                         "occ_lci": x.occ_lci.mean(), "agree": x.agree.mean(), "accuracy": x.accuracy.mean()})
    t = pd.DataFrame(rows)
    t.round(4).to_csv(OUT / "gap-study-v5-test-tinyhar-summary-per-group-and-model.csv", index=False, lineterminator="\n")
    pr = []
    for spec in a.pairs:
        A, B = spec.split(":")
        for g in ("general", "older", "all"):
            sg = subj if g == "all" else subj[subj.group == g]
            xa = sg[sg.label == A].set_index(["group", "subject"])
            xb = sg[sg.label == B].set_index(["group", "subject"])
            ix = xa.index.intersection(xb.index)
            if len(ix) < 3:
                continue
            dl = xa.loc[ix, "lci_indep"] - xb.loc[ix, "lci_indep"]
            do = xa.loc[ix, "occ_lci"] - xb.loc[ix, "occ_lci"]
            pr.append({"group": g, "a": A, "b": B, "n_subjects": len(ix), "a_lci_indep": xa.loc[ix, "lci_indep"].mean(),
                       "b_lci_indep": xb.loc[ix, "lci_indep"].mean(), "diff_lci_indep": dl.mean(),
                       "subjects_positive": int((dl > 0).sum()),
                       "wilcoxon_p_greater": float(stats.wilcoxon(dl, alternative="greater").pvalue),
                       "diff_occ_lci": do.mean(),
                       "diff_accuracy_pt": 100 * (xa.loc[ix, "accuracy"] - xb.loc[ix, "accuracy"]).mean()})
    p = pd.DataFrame(pr)
    if len(p):
        p.round(4).to_csv(OUT / "gap-study-v5-test-tinyhar-paired-differences.csv", index=False, lineterminator="\n")
    pd.set_option("display.width", 250)
    print(t.round(3).to_string(index=False))
    if len(p):
        print(p.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
