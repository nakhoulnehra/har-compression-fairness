"""Phase 3 diagnostics (no method, no tuning): WHERE and HOW explanations change in the candidate cells.

    python -m gap_study_v3.diagnose

Reads the attributions saved by gap_study_v3.confirm (data/processed/har/gap-study-v3/attributions/, 48 features =
6 channels x 8 time segments, SHAP seed 0 and 1, occlusion, predicted class) and the phase-2 model CSVs.

Groups (one per cell; every fold present, seeds pooled):
  orig_shap_noise      the original under another SHAP seed (the noise ceiling)
  nr50_l1_fullrecon    no retraining, 50 % (near lossless reference)
  nr80_l1_fullrecon    candidate 1 best baseline (no retraining, 80 %), seeds 0-2
  nr80_lamp_recon      candidate 1 most accurate baseline, seeds 0-2
  nr90_l1_fullrecon    no retraining, 90 %
  ft90_l1_kd           candidate 2 (KD fine-tuning, 90 %, lr 1e-4), seeds 0-2
  ft90_l1_fullrecon_kd candidate 2, reconstruction start + KD, seeds 0-2
  ft80_*               candidate 2 neighbour
  retrain_seed1        an independently retrained TinyHAR (folds 0, 1, 3): the Rashomon floor
Outputs in results/gap-study-v3/phase3/.
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

for p in (ROOT,):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from har.data import CHANNELS, CLASS_NAMES  # noqa: E402
from har.paths import PROCESSED_DIR  # noqa: E402
from lci.metric import compute_lci  # noqa: E402

ATTR = PROCESSED_DIR / "gap-study-v3" / "attributions"
OUT = ROOT / "results" / "gap-study-v3" / "phase3"
P2 = ROOT / "results" / "gap-study-v3" / "phase2"
P = "gap-study-v3-phase3-tinyhar"
GROUPS = {
    "nr50_l1_fullrecon": r"^l1_fullrecon-50pct-seed0$",
    "nr70_l1_fullrecon": r"^l1_fullrecon-70pct-seed0$",
    "nr80_l1_fullrecon": r"^l1_fullrecon-80pct-seed\d$",
    "nr80_lamp_recon": r"^lamp_recon-80pct-seed\d$",
    "nr80_wanda_fullrecon": r"^wanda_fullrecon-80pct-seed0$",
    "nr80_taylor_fullrecon": r"^taylor_fullrecon-80pct-seed\d$",
    "nr90_l1_fullrecon": r"^l1_fullrecon-90pct-seed0$",
    "ft80_l1_kd": r"^l1_kd-80pct-seed0-lr0.0001-e20$",
    "ft80_l1_fullrecon_kd": r"^l1_fullrecon_kd-80pct-seed0-lr0.0001-e20$",
    "ft90_l1_kd": r"^l1_kd-90pct-seed\d-lr0.0001-e20$",
    "ft90_l1_fullrecon_kd": r"^l1_fullrecon_kd-90pct-seed\d-lr0.0001-e20$",
    "ft90_lamp_recon_kd": r"^lamp_recon_kd-90pct-seed\d-lr0.0003-e20$",
    "ft90_9a_park_kd": r"^9a_park_kd-90pct-seed0$",
    "retrain_seed1": r"^retrain_seed1-0pct-seed1$",
}
SEGS = 8


def norm1(A):
    return A / np.maximum(np.abs(A).sum(1, keepdims=True), 1e-12)


def window_corr(A, B):
    a = A - A.mean(1, keepdims=True)
    b = B - B.mean(1, keepdims=True)
    return (a * b).sum(1) / np.maximum(np.sqrt((a * a).sum(1) * (b * b).sum(1)), 1e-12)


def model_stats(ref: dict, A: np.ndarray, pred: np.ndarray, proba: np.ndarray | None) -> dict:
    A0, tgt, subj = ref["A0"], ref["target"], ref["subj"]
    r0, rm = norm1(A0), norm1(A)
    err = np.abs(rm - r0)                                       # per window, scale of the reference = 1
    ch_err = err.reshape(len(A), 6, SEGS).sum(2).sum(0)
    ch_mass_ref = np.abs(r0).reshape(len(A), 6, SEGS).sum(2).sum(0)
    ch_mass_m = np.abs(rm).reshape(len(A), 6, SEGS).sum(2).sum(0)
    seg_err = err.reshape(len(A), 6, SEGS).sum(1).sum(0)
    top10 = lambda X: np.sort(np.abs(norm1(X)), 1)[:, -10:].sum(1)
    mag = np.abs(A).sum(1) / np.maximum(np.abs(A0).sum(1), 1e-12)
    wc = window_corr(A0, A)
    out = {"magnitude_ratio_median": float(np.median(mag)),
           "top10_mass_ref": float(top10(A0).mean()), "top10_mass_model": float(top10(A).mean()),
           "window_corr_mean": float(wc.mean()),
           "sign_flip_top10_ref": float(np.mean([np.mean(np.sign(A[i, np.argsort(-np.abs(A0[i]))[:10]])
                                                         != np.sign(A0[i, np.argsort(-np.abs(A0[i]))[:10]]))
                                                 for i in range(len(A))])),
           "agree": float((pred == tgt).mean())}
    # global top-10 per subject (the Jaccard half of LCI): share of the reference's top-10 slots that change channel
    js, chs = [], []
    for s in sorted(set(subj.tolist())):
        m = subj == s
        g0 = np.argsort(-np.abs(A0[m]).mean(0), kind="stable")[:10]
        g1 = np.argsort(-np.abs(A[m]).mean(0), kind="stable")[:10]
        js.append(len(set(g0) & set(g1)) / len(set(g0) | set(g1)))
        chs.append(np.abs(np.bincount(g0 // SEGS, minlength=6) - np.bincount(g1 // SEGS, minlength=6)).sum() / 2)
    out["jaccard_global_top10"] = float(np.mean(js))
    out["top10_slots_changing_channel"] = float(np.mean(chs))
    if proba is not None:
        p0 = None
    for c, ch in enumerate(CHANNELS):
        out[f"err_share_{ch}"] = float(ch_err[c] / ch_err.sum())
        out[f"mass_ref_{ch}"] = float(ch_mass_ref[c] / ch_mass_ref.sum())
        out[f"mass_model_{ch}"] = float(ch_mass_m[c] / ch_mass_m.sum())
    for s_ in range(SEGS):
        out[f"err_share_seg{s_}"] = float(seg_err[s_] / seg_err.sum())
    per_class = {}
    for c, name in enumerate(CLASS_NAMES):
        m = tgt == c
        if m.sum() >= 5:
            per_class[name] = (compute_lci(A0[m], A[m], k=10, omega=0.5)["lci"], int(m.sum()))
    return out, per_class, wc


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, cls_rows, win_rows = [], [], []
    for k in range(5):
        rp = ATTR / f"fold{k}-original-refs.npz"
        if not rp.exists():
            continue
        z = np.load(rp, allow_pickle=False)
        ref = {"A0": z["A0"], "target": z["target"], "subj": z["subj"].astype(str)}
        items = [("orig_shap_noise", "original-seed1", z["A0s1"], z["target"], None)]
        for f in sorted(ATTR.glob(f"fold{k}-*.npz")):
            rid = f.stem[len(f"fold{k}-"):]
            if rid == "original-refs":
                continue
            for g, pat in GROUPS.items():
                if re.match(pat, rid):
                    d = np.load(f, allow_pickle=False)
                    items.append((g, rid, d["A0"], d["pred"], d["proba"]))
                    if "O" in d.files:
                        items.append((g + "__occlusion", rid, d["O"], d["pred"], None))
        for g, rid, A, pred, proba in items:
            if g.endswith("__occlusion"):
                o, pc, wc = model_stats({**ref, "A0": z["O0"]}, A, pred, proba)
            else:
                o, pc, wc = model_stats(ref, A, pred, proba)
            rows.append({"group": g, "fold": k, "run_id": rid, **o})
            for name, (l, n) in pc.items():
                cls_rows.append({"group": g, "fold": k, "run_id": rid, "class": name, "lci_class": l, "n": n})
            if proba is not None:
                # window level: does explanation change track the confidence change on the explained class?
                pt = proba[np.arange(len(pred)), ref["target"]]
                win_rows.append({"group": g, "fold": k, "run_id": rid,
                                 "spearman_windowcorr_vs_p_target": float(stats.spearmanr(wc, pt).correlation),
                                 "window_corr_agree": float(wc[pred == ref["target"]].mean()),
                                 "window_corr_disagree": float(wc[pred != ref["target"]].mean()) if (pred != ref["target"]).any() else np.nan})
    # cross-seed agreement: do compressed models of different seeds agree with EACH OTHER more than with the original?
    xs = []
    for k in range(5):
        rp = ATTR / f"fold{k}-original-refs.npz"
        if not rp.exists():
            continue
        subj = np.load(rp)["subj"].astype(str)
        A0 = np.load(rp)["A0"]
        for g, pat in GROUPS.items():
            fs = [f for f in sorted(ATTR.glob(f"fold{k}-*.npz")) if re.match(pat, f.stem[len(f"fold{k}-"):])]
            if len(fs) < 2:
                continue
            As = [np.load(f)["A0"] for f in fs]
            def lci_s(X, Y):
                return np.mean([compute_lci(X[subj == s], Y[subj == s], k=10, omega=0.5)["lci"] for s in sorted(set(subj))])
            pairs = [lci_s(As[i], As[j]) for i in range(len(As)) for j in range(i + 1, len(As))]
            to_orig = [lci_s(A0, A) for A in As]
            xs.append({"group": g, "fold": k, "n_seeds": len(As), "lci_between_seeds": float(np.mean(pairs)),
                       "lci_to_original": float(np.mean(to_orig))})
    if xs:
        x = pd.DataFrame(xs)
        x.round(4).to_csv(OUT / f"{P}-cross-seed-agreement.csv", index=False, lineterminator="\n")
        print(x.groupby("group")[["n_seeds", "lci_between_seeds", "lci_to_original"]].mean().round(3).to_string())
    d = pd.DataFrame(rows)
    d.round(4).to_csv(OUT / f"{P}-per-model-attribution-diagnostics.csv", index=False, lineterminator="\n")
    num = [c for c in d.columns if c not in ("group", "fold", "run_id")]
    g = d.groupby(["group", "fold"])[num].mean().groupby("group").agg(["mean"]).droplevel(1, axis=1)
    g["n_folds"] = d.groupby("group").fold.nunique()
    g.round(4).to_csv(OUT / f"{P}-attribution-diagnostics-mean-over-folds.csv", lineterminator="\n")
    c = pd.DataFrame(cls_rows)
    ref_noise = c[c.group == "orig_shap_noise"].groupby(["fold", "class"]).lci_class.mean().rename("lci_class_shap_noise")
    c = c.merge(ref_noise.reset_index(), on=["fold", "class"], how="left")
    cg = c.groupby(["group", "class"]).agg(lci_class=("lci_class", "mean"), lci_class_noise=("lci_class_shap_noise", "mean"),
                                           n_folds=("fold", "nunique"), n_windows=("n", "mean")).reset_index()
    cg["drop_vs_noise"] = cg.lci_class_noise - cg.lci_class
    cg.round(4).to_csv(OUT / f"{P}-per-class-lci.csv", index=False, lineterminator="\n")
    w = pd.DataFrame(win_rows)
    if len(w):
        w.groupby("group")[[c for c in w.columns if c not in ("group", "fold", "run_id")]].mean().round(4).to_csv(
            OUT / f"{P}-window-level-explanation-vs-confidence.csv", lineterminator="\n")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    show = ["n_folds", "agree", "window_corr_mean", "jaccard_global_top10", "top10_slots_changing_channel",
            "magnitude_ratio_median", "top10_mass_ref", "top10_mass_model", "sign_flip_top10_ref"]
    print(g[show].round(3).to_string())
    print(g[[f"err_share_{ch}" for ch in CHANNELS] + [f"mass_ref_{ch}" for ch in CHANNELS]].round(3).to_string())
    print(cg.pivot(index="group", columns="class", values="lci_class").round(3).to_string())
    if len(w):
        print(w.groupby("group").mean(numeric_only=True).round(3).to_string())


if __name__ == "__main__":
    main()
