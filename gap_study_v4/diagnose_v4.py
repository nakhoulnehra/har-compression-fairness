"""Gap study v4: where does the best no-retraining baseline fail, and where does the same-size reference recover?
Diagnostics only, from saved validation attributions (inner val; no new model).

    python -m gap_study_v4.diagnose_v4 --models B=v3:l1_fullrecon-80pct-seed{s} X=v4:fmkd_from_l1_fullrecon-80pct-seed{s}-lr0.0003 ...

Per model (independent SHAP seed: model seed 1 vs original seed 0; the original's own seed 1 = "noise"):
  LCI and its two parts (per-window Pearson, global top-10 Jaccard), per true class (subject x class cells with >= 3
  windows, mean over cells), occlusion LCI per class, and the window-level gain of X over B by class.
Writes results/gap-study-v4/validation/gap-study-v4-validation-tinyhar-diagnostics-{per-class,parts}.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from har.data import CLASS_NAMES, find_windows, load_harth_windows
from lci.metric import compute_lci, per_sample_corr

ROOT = Path(__file__).resolve().parents[1]
ATTR = {"v3": ROOT / "data" / "processed" / "har" / "gap-study-v3" / "attributions",
        "v4": ROOT / "data" / "processed" / "har" / "gap-study-v4" / "attributions"}
OUT = ROOT / "results" / "gap-study-v4" / "validation"
P = "gap-study-v4-validation-tinyhar-diagnostics"


def cell_lci(Aref, A, subj, mask, key="lci"):
    vals = []
    for s in sorted(set(subj[mask].tolist())):
        m = mask & (subj == s)
        if m.sum() >= 3:
            vals.append(compute_lci(Aref[m], A[m], k=10, omega=0.5)[key])
    return float(np.mean(vals)) if vals else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="label=src:run_id ({s} = seed)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    harth = load_harth_windows()
    va = harth["harth_test"]
    rows, parts = [], []
    for k in a.folds:
        r = np.load(ATTR["v3"] / f"fold{k}-original-refs.npz", allow_pickle=False)
        A0, subj = r["A0"], r["subj"].astype(str)
        y = va.y[find_windows(va, "harth_test", r["ex_ids"].astype(str).tolist())]
        models = {"noise (original, SHAP seed 1)": (r["A0s1"], r["O0"] * 0 + r["O0"], None)}
        for spec in a.models:
            lab, ref = spec.split("=", 1)
            src, rid = ref.split(":", 1)
            for s in a.seeds:
                p = ATTR[src] / f"fold{k}-{rid.format(s=s)}.npz"
                if not p.exists():
                    continue
                z = np.load(p)
                models[f"{lab} seed{s}"] = (z["A1"], z["O"], z["pred"])
        for name, (A, O, pred) in models.items():
            lab = name.rsplit(" seed", 1)[0] if " seed" in name and not name.startswith("noise") else name
            rho, _ = per_sample_corr(A0, A)
            allm = np.ones(len(A0), bool)
            parts.append({"fold": k, "label": lab, "model": name, "lci": cell_lci(A0, A, subj, allm),
                          "pearson_part": cell_lci(A0, A, subj, allm, "shap_corr"),
                          "jaccard_part": cell_lci(A0, A, subj, allm, "jaccard"),
                          "occ_lci": np.nan if pred is None else cell_lci(r["O0"], O, subj, allm),
                          "agree": np.nan if pred is None else float((pred == r["target"]).mean())})
            for c in sorted(set(y.tolist())):
                m = y == c
                w = np.array([np.corrcoef(A0[i], A[i])[0, 1] for i in np.flatnonzero(m)])
                rows.append({"fold": k, "label": lab, "model": name, "class": CLASS_NAMES[c], "n_windows": int(m.sum()),
                             "lci": cell_lci(A0, A, subj, m), "window_pearson_mean": float(np.nanmean(w)),
                             "occ_lci": np.nan if pred is None else cell_lci(r["O0"], O, subj, m)})
    d, p = pd.DataFrame(rows), pd.DataFrame(parts)
    d.round(4).to_csv(OUT / f"{P}-per-class.csv", index=False, lineterminator="\n")
    p.round(4).to_csv(OUT / f"{P}-parts.csv", index=False, lineterminator="\n")
    pd.set_option("display.width", 250)
    print(p.groupby("label")[["lci", "pearson_part", "jaccard_part", "occ_lci", "agree"]].mean().round(3).to_string())
    t = d.groupby(["class", "label"]).agg(lci=("lci", "mean"), wcorr=("window_pearson_mean", "mean"),
                                          occ=("occ_lci", "mean"), n=("n_windows", "sum")).unstack("label")
    print(t.round(3).to_string())


if __name__ == "__main__":
    main()
