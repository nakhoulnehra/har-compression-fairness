"""Phase 2 analysis: the gate for each candidate cell, the full sweep, the noise floor and baseline trade-offs.

    python -m gap_study_v3.analyze

Gate (fixed 2026-11-26 ~10:05, before any fine-tuned result of this study was read):
  reference u      ceiling50 = LCI(original SHAP seed 0, original SHAP seed 1) on the same 50 explain windows per
                   inner-val subject, mean over the 5 folds (independent SHAP noise; the uncompressed model's score)
  best baseline b  the method with the highest mean same-seed LCI at the cell (mean over folds x seeds; only methods
                   run on all 5 folds compete). Its independent-seed LCI (lci_indep) is the apples-to-apples value;
                   the same-seed value (agent B's protocol) is reported as the conservative one
  G1 size          u - b >= 0.10 or (u - b) / u >= 15 %, on BOTH the same-seed and the independent-seed value
  G2 variance      fold-paired 95 % t CI of (u_k - b_k) excludes 0, and u - b > 2 x the SHAP-noise SD and > 2 x the
                   within-fold seed SD of b
  G3 scope         gap >= 0.10 in >= 4 of 5 folds (validation); test-set confirmation on HARTH test and HAR70+ is
                   reported once, from the existing one-shot test runs (9A for l1_kd, agent B's for no retraining)
  G4 not collapse  accuracy within 5 pt of the original, or the relative LCI drop >= 2x the relative accuracy drop
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "gap-study-v3" / "phase2"
P = "gap-study-v3-phase2-tinyhar"
GATE = {"abs": 0.10, "rel": 0.15, "acc_pt": 0.05, "ratio": 2.0}


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    v3 = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(OUT / f"{P}-fold*-models.csv")))], ignore_index=True)
    orig = v3[v3.method == "original"].groupby("fold").agg(acc0=("accuracy", "first"), f10=("f1", "first"),
                                                         ceiling50=("ceiling50", "first"))
    b = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(RES / "gap-study-v3" / "agentB-no-retrain-baselines" / "gap-study-phase2-tinyhar-fold*-models.csv")))],
                  ignore_index=True)
    b = b[(b.method != "original") & (b.damp.round(4) == 0.01)].copy()
    b["variant"], b["scenario"], b["source"] = "agent B phase 2", "no_retrain", "agentB"
    # attach this study's artefact metrics to agent B's runs (re-scored checkpoints)
    rs = v3[v3.stage == "rescore"].copy()
    rs["key"] = rs.method + "|" + rs.level.round(2).astype(str) + "|" + rs.seed.astype(int).astype(str) + "|" + rs.fold.astype(str)
    b["key"] = b.method + "|" + b.level.round(2).astype(str) + "|" + b.seed.astype(int).astype(str) + "|" + b.fold.astype(str)
    for c in ("lci_indep", "self_lci", "occ_lci", "agree", "lci_agree"):
        b[c] = b.key.map(rs.set_index("key")[c])
    n = v3[v3.stage.isin(["nr", "ftcore", "ftextra", "rescore9a", "refs2"])].copy()
    n["scenario"] = n.stage.map({"nr": "no_retrain", "ftcore": "kd_finetune", "ftextra": "kd_finetune",
                                 "rescore9a": "kd_finetune", "refs2": "reference"})
    n["variant"] = n.apply(lambda r: (f"pdamp {r.pdamp}" if pd.notna(r.get("pdamp")) else "") +
                           (f"lr {r.lr:g}" if pd.notna(r.get("lr")) else ""), axis=1)
    n.loc[n.method.str.startswith("9a_"), "variant"] = "9A checkpoint (folds 0 1 3)"
    n["source"] = "v3"
    d = pd.concat([b, n], ignore_index=True)
    d["label"] = d.method + np.where(d.variant.isin(["", "agent B phase 2"]), "", " [" + d.variant + "]")
    d = d.drop(columns=["ceiling50"], errors="ignore").join(orig, on="fold")
    d["d_acc"] = d.accuracy - d.acc0
    d["d_f1"] = d.f1 - d.f10
    return d, orig


def summarize(d: pd.DataFrame) -> pd.DataFrame:
    keys = ["scenario", "level", "label"]
    g = d.groupby(keys).agg(n_folds=("fold", "nunique"), n_runs=("fold", "size"), accuracy=("accuracy", "mean"),
                            d_acc=("d_acc", "mean"), f1=("f1", "mean"), d_f1=("d_f1", "mean"), lci=("lci", "mean"),
                            lci_sd=("lci", "std"), lci_indep=("lci_indep", "mean"), self_lci=("self_lci", "mean"),
                            occ_lci=("occ_lci", "mean"), agree=("agree", "mean"), lci_agree=("lci_agree", "mean"),
                            worst=("worst_subject_lci", "mean"), ceiling50=("ceiling50", "mean")).reset_index()
    seed_sd = d.groupby(keys + ["fold"]).lci.std().groupby(keys).mean().rename("within_fold_seed_sd")
    return g.join(seed_sd, on=keys)


def gate_cell(d, s, scenario, level, orig) -> dict:
    cell = s[(s.scenario == scenario) & (s.level.round(2) == level) & (s.n_folds == 5)]
    if cell.empty:
        return {"scenario": scenario, "level": level, "status": "no complete baseline"}
    best = cell.sort_values("lci", ascending=False).iloc[0]
    most_acc = cell.sort_values("accuracy", ascending=False).iloc[0]
    x = d[(d.scenario == scenario) & (d.level.round(2) == level) & (d.label == best.label)]
    pf = x.groupby("fold").agg(b=("lci", "mean"), bi=("lci_indep", "mean"), acc=("accuracy", "mean"))
    pf = pf.join(orig)
    gap_k = pf.ceiling50 - pf.b
    gap_ki = pf.ceiling50 - pf.bi
    u = float(orig.ceiling50.mean())
    ci = stats.t.interval(0.95, len(gap_k) - 1, loc=gap_k.mean(), scale=gap_k.std(ddof=1) / np.sqrt(len(gap_k)))
    cii = (stats.t.interval(0.95, len(gap_ki.dropna()) - 1, loc=gap_ki.mean(), scale=gap_ki.std(ddof=1) / np.sqrt(len(gap_ki.dropna())))
           if gap_ki.notna().sum() >= 3 else (np.nan, np.nan))
    gap, gapi = u - best.lci, (u - best.lci_indep) if pd.notna(best.lci_indep) else np.nan
    shap_sd = float((1 - orig.ceiling50).std(ddof=1))
    rel_lci = gap / u
    rel_acc = -best.d_acc / float(orig.acc0.mean())
    row = {"scenario": scenario, "level": level, "best_baseline": best.label, "n_folds": int(best.n_folds),
           "n_runs": int(best.n_runs), "u_ceiling50": u, "b_lci_same_seed": best.lci, "b_lci_sd": best.lci_sd,
           "b_lci_indep": best.lci_indep, "gap_same": gap, "gap_rel_same": rel_lci, "gap_indep": gapi,
           "gap_rel_indep": gapi / u if pd.notna(gapi) else np.nan, "gap_ci95": f"[{ci[0]:.3f}, {ci[1]:.3f}]",
           "gap_indep_ci95": f"[{cii[0]:.3f}, {cii[1]:.3f}]", "folds_gap_ge_0.10": int((gap_k >= GATE["abs"]).sum()),
           "per_fold_gap": " ".join(f"{v:.3f}" for v in gap_k),
           "folds_gap_indep_ge_0.10": int((gap_ki >= GATE["abs"]).sum()) if gap_ki.notna().all() else np.nan,
           "per_fold_gap_indep": " ".join(f"{v:.3f}" for v in gap_ki), "shap_noise_sd": shap_sd,
           "within_fold_seed_sd": best.within_fold_seed_sd, "b_accuracy": best.accuracy, "acc_original": float(orig.acc0.mean()),
           "b_d_acc_pt": 100 * best.d_acc, "b_d_f1_pt": 100 * best.d_f1, "rel_lci_drop_over_rel_acc_drop": rel_lci / rel_acc if rel_acc > 0 else np.inf,
           "b_self_lci": best.self_lci, "b_occ_lci": best.occ_lci, "b_agree": best.agree, "b_lci_agree": best.lci_agree,
           "most_accurate_baseline": most_acc.label, "most_accurate_acc": most_acc.accuracy, "most_accurate_lci": most_acc.lci}
    g1 = (gap >= GATE["abs"] or rel_lci >= GATE["rel"]) and (pd.isna(gapi) or gapi >= GATE["abs"] or gapi / u >= GATE["rel"])
    sd_seed = best.within_fold_seed_sd if pd.notna(best.within_fold_seed_sd) else 0.0
    g2 = bool(ci[0] > 0 and gap > 2 * shap_sd and gap > 2 * sd_seed)
    g3 = bool((gap_k >= GATE["abs"]).sum() >= 4)
    row["G3_scope_folds_indep"] = bool((gap_ki >= GATE["abs"]).sum() >= 4) if gap_ki.notna().all() else None
    g4 = bool(-best.d_acc <= GATE["acc_pt"] or row["rel_lci_drop_over_rel_acc_drop"] >= GATE["ratio"])
    row.update({"G1_size": bool(g1), "G2_variance": g2, "G3_scope_folds": g3, "G4_not_collapse": g4,
                "gate_pass_validation": bool(g1 and g2 and g3 and g4), "seeds_flag": "" if best.n_runs >= 15 else "best baseline has < 3 seeds per fold"})
    return row


def main() -> None:
    d, orig = load()
    s = summarize(d)
    s.round(4).to_csv(OUT / f"{P}-summary-every-method-level-mean-over-folds-seeds.csv", index=False, lineterminator="\n")
    cells = [("no_retrain", lv) for lv in (0.5, 0.7, 0.8, 0.9)] + [("kd_finetune", lv) for lv in (0.7, 0.8, 0.9)]
    gt = pd.DataFrame([gate_cell(d, s, sc, lv, orig) for sc, lv in cells])
    gt.round(4).to_csv(OUT / f"{P}-gate-per-cell.csv", index=False, lineterminator="\n")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    cols = ["scenario", "level", "label", "n_folds", "n_runs", "accuracy", "d_acc", "f1", "lci", "lci_indep", "self_lci",
            "occ_lci", "agree", "lci_agree", "within_fold_seed_sd"]
    print(s[cols].sort_values(["scenario", "level", "lci"], ascending=[True, True, False]).round(3).to_string())
    print(gt.T.to_string())
    print(json.dumps({"ceiling50_per_fold": orig.ceiling50.round(4).tolist(), "acc0": orig.acc0.round(4).tolist()}))


if __name__ == "__main__":
    main()
