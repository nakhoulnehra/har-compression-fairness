"""MASTER_TABLE.csv: every method x evaluation set x compression x level x scenario, from existing files only.

    python -m gap_study_v3.build_master_table

Long format, one row per cell. Columns: source, eval_set (harth_inner_val | test_harth_general | test_har70plus |
test_all), compression, level, scenario (no_retrain | retrained_full | retrained_kd5 | retrained_calibration |
reference), method, variant, n_folds, n_runs, accuracy / f1 / lci (mean, sd over runs), fa_lci (test only: min of
the general and older group LCI, plan's lambda = 0.5 definition), flags.

Sources:
  validation  agent B's long table (results/gap-study-v3/phase0-master-table-inputs-agentB-long.csv; F1 + LCI, no accuracy)
              for rounds 3-4, low-bit, bitred, PAIR; re-parsed here with accuracy: 9A models, agent B's phase 2 / 3
              (GMR), this study's phase 2
  test        9A analysis (FA-LCI file + group accuracy), low-bit diagnostic FA-LCI, agent B's test means (GMR,
              l1_fullrecon, lamp_recon), agent A's PAIR calibration / scarce test means
Flags: n_folds < 5, < 3 runs per fold, LCI sd > 0.10, F1 < 0.6 (collapsed), no accuracy in the source, previous agent (A or B).
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "gap_study_v3" / "MASTER_TABLE.csv"
SCEN = {"no_retrain": "no_retrain", "full_ft": "retrained_full", "kd5": "retrained_kd5",
        "calib_train": "retrained_calibration", "reference": "reference", "no_finetune": "no_retrain",
        "finetune": "retrained_full", "original": "reference"}


def agg(df: pd.DataFrame, keys: list[str], acc="accuracy", f1="f1", lci="lci") -> pd.DataFrame:
    g = df.groupby(keys, dropna=False)
    out = g.agg(n_folds=("fold", "nunique"), n_runs=("fold", "size"),
                accuracy_mean=(acc, "mean"), accuracy_sd=(acc, "std"), f1_mean=(f1, "mean"), f1_sd=(f1, "std"),
                lci_mean=(lci, "mean"), lci_sd=(lci, "std")).reset_index()
    return out


def flags(r) -> str:
    f = []
    if r.get("n_folds", 5) < 5 and not str(r.get("eval_set", "")).startswith("test"):
        f.append(f"only {int(r['n_folds'])} fold(s)")
    if r.get("n_runs", 0) < 3 * r.get("n_folds", 1) and not str(r.get("eval_set", "")).startswith("test"):
        f.append("<3 seeds/fold")
    if pd.notna(r.get("lci_sd")) and r["lci_sd"] > 0.10:
        f.append("LCI SD > 0.10")
    if pd.notna(r.get("f1_mean")) and r["f1_mean"] < 0.6:
        f.append("F1 collapsed (<0.6)")
    if pd.isna(r.get("accuracy_mean")):
        f.append("no accuracy in source")
    return "; ".join(f)


def validation_rows() -> list[pd.DataFrame]:
    parts = []
    b = pd.read_csv(RES / "gap-study-v3" / "phase0-master-table-inputs-agentB-long.csv")
    b = b[b.source != "9A"].copy()                       # 9A re-parsed below with accuracy
    b = b.rename(columns={"f1_mean": "f1_mean", "lci_mean": "lci_mean"})
    b["scenario"] = b.scenario.map(SCEN).fillna(b.scenario)
    b["eval_set"] = "harth_inner_val"
    b["accuracy_mean"] = np.nan
    b["accuracy_sd"] = np.nan
    b["uncommitted"] = b.source.str.startswith("pair")
    parts.append(b[["source", "eval_set", "compression", "level", "scenario", "method", "variant", "n_folds", "n_runs",
                    "accuracy_mean", "accuracy_sd", "f1_mean", "f1_sd", "lci_mean", "lci_sd", "uncommitted"]])
    # 9A models (fold 3 = uni-2080's copy, as the 9A analysis)
    fs = [f for f in sorted(glob.glob(str(RES / "har-compression-grid" / "har-grid-tinyhar-fold*-models.csv")))
          if "fold3-nakhoul" not in f]
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    d["compression"] = d.family.map({"pruning": "unstructured_pruning", "ptq": "ptq", "original": "none"}).fillna(d.family)
    d["scenario"] = d.scenario.map(SCEN).fillna(d.scenario)
    a = agg(d, ["compression", "level", "scenario", "method"], acc="accuracy_pooled_inner_val",
            f1="macro_f1_pooled7_inner_val", lci="lci_inner_val")
    a["source"], a["variant"], a["eval_set"], a["uncommitted"] = "9A", "30 explain windows/subject", "harth_inner_val", False
    parts.append(a)
    # agent B phase 2 + 3 (uncommitted)
    for ph, extra, folder in (("phase2", ["damp"], RES / "gap-study-v3" / "agentB-no-retrain-baselines"),
                              ("phase3", ["mu"], RES / "gap-study" / "phase3")):
        fs = sorted(glob.glob(str(folder / f"gap-study-{ph}-tinyhar-fold*-models.csv")))
        if not fs:
            continue      # deleted source: rows kept frozen from the previous MASTER_TABLE.csv
        d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
        d = d[d.method != "original"].copy()
        d["variant"] = d[extra[0]].astype(str).radd(f"{extra[0]}=")
        a = agg(d, ["level", "method", "variant"])
        a["source"], a["compression"], a["scenario"] = f"gap-study-{ph} (agent B)", "unstructured_pruning", "no_retrain"
        a["eval_set"], a["uncommitted"] = "harth_inner_val", True
        parts.append(a)
    # this study's phase 2
    fs = sorted(glob.glob(str(RES / "gap-study-v3" / "phase2" / "gap-study-v3-phase2-tinyhar-fold*-models.csv")))
    if fs:
        d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
        d = d[d.method != "original"].copy()
        d["variant"] = d.apply(lambda r: _v3_variant(r), axis=1)
        d["scenario"] = d.stage.map({"rescore": "no_retrain", "nr": "no_retrain", "ftcore": "retrained_full",
                                     "ftextra": "retrained_full", "rescore9a": "retrained_full"})
        a = agg(d, ["level", "method", "variant", "scenario"])
        for c in ("lci_indep", "self_lci", "occ_lci", "agree", "lci_agree"):
            a[c + "_mean"] = d.groupby(["level", "method", "variant", "scenario"], dropna=False)[c].mean().values \
                if c in d else np.nan
        a["source"], a["compression"], a["eval_set"], a["uncommitted"] = "gap-study-v3 phase2", "unstructured_pruning", "harth_inner_val", False
        parts.append(a)
    return parts


def _v3_variant(r) -> str:
    v = []
    if pd.notna(r.get("pdamp")):
        v.append(f"percdamp={r['pdamp']}")
    if pd.notna(r.get("lr")):
        v.append(f"lr={r['lr']:g} E=20")
    if r.get("stage") == "rescore":
        v.append("agent B checkpoint, re-scored")
    if r.get("stage") == "rescore9a":
        v.append("9A checkpoint, re-scored (folds 0 1 3)")
    return ", ".join(v) if v else "-"


def test_rows() -> list[pd.DataFrame]:
    parts = []

    def tidy(df, source, uncommitted):
        df = df.copy()
        df["source"], df["uncommitted"], df["n_runs"] = source, uncommitted, 1
        return df

    # 9A
    fa = pd.read_csv(RES / "har-compression-grid-analysis" / "har-grid-a-fa-lci-bootstrap-ci.csv")
    gs = pd.read_csv(RES / "har-compression-grid-analysis" / "har-grid-a-group-summary-gaps-bootstrap-ci.csv")
    acc = gs[gs.metric == "accuracy"][["method", "scenario", "level", "mean_general", "mean_older"]]
    m = fa.merge(acc, on=["method", "scenario", "level"], how="left")
    rows = []
    for _, r in m.iterrows():
        comp = {"pruning": "unstructured_pruning", "ptq": "ptq"}.get(r.family, r.family)
        for es, l, ac in (("test_harth_general", r.lci_general, r.mean_general), ("test_har70plus", r.lci_older, r.mean_older)):
            rows.append({"eval_set": es, "compression": comp, "level": r.level, "scenario": SCEN.get(r.scenario, r.scenario),
                         "method": r.method, "variant": "9A test, seed 0, 5 fold models", "n_folds": 5,
                         "accuracy_mean": ac, "lci_mean": l, "fa_lci": r.fa_lci, "fa_lci_ci": f"[{r.fa_lci_ci_low:.3f}, {r.fa_lci_ci_high:.3f}]"})
    parts.append(tidy(pd.DataFrame(rows), "9A test", False))
    # low-bit diagnostic
    lb = pd.read_csv(RES / "har-diagnostic-low-bit" / "har-diag-lowbit-a-test-fa-lci-bootstrap-ci.csv")
    lb = lb[~lb.method.str.startswith("layer2bit")]
    rows = []
    for _, r in lb.iterrows():
        for es, l in (("test_harth_general", r.lci_general), ("test_har70plus", r.lci_older)):
            rows.append({"eval_set": es, "compression": r.family, "level": r.level, "scenario": SCEN.get(r.scenario, r.scenario),
                         "method": r.method, "variant": "low-bit diagnostic test", "n_folds": 5, "lci_mean": l,
                         "fa_lci": r.fa_lci, "fa_lci_ci": f"[{r.fa_lci_ci_low:.3f}, {r.fa_lci_ci_high:.3f}]"})
    parts.append(tidy(pd.DataFrame(rows), "low-bit test", False))
    # agent B test
    t = pd.read_csv(RES / "gap-study-v3" / "agentB-no-retrain-test" / "gap-study-test-tinyhar-means-per-model.csv")
    rows = []
    for _, r in t.iterrows():
        lv = next((x for x in r.model_id.split("-") if x.endswith("pct")), "none")
        meth = r.model_id.split("-")[0] + ("-mu0.1" if r.model_id.startswith("gmr") else "")
        for es, l, ac in (("test_harth_general", r.lci_general, r.accuracy_general), ("test_har70plus", r.lci_older, r.accuracy_older),
                          ("test_all", r.lci_all, r.accuracy_all)):
            rows.append({"eval_set": es, "compression": "unstructured_pruning", "level": lv, "scenario": "no_retrain",
                         "method": meth, "variant": "agent B test, seed 0", "n_folds": 5, "accuracy_mean": ac,
                         "lci_mean": l, "fa_lci": min(r.lci_general, r.lci_older)})
    parts.append(tidy(pd.DataFrame(rows), "gap-study test (agent B)", True))
    # agent A tests (deleted after this study; rows then kept frozen from the previous MASTER_TABLE.csv)
    if not (RES / "har-pair-calibration").exists():
        return parts
    g = pd.read_csv(RES / "har-pair-calibration" / "test" / "har-pair-calibration-tinyhar-test-group-means.csv")
    rows = []
    for mid, gg in g.groupby("model_id"):
        gg = gg.set_index("group")
        fa_ = min(gg.lci.get("general", np.nan), gg.lci.get("older", np.nan))
        for grp, es in (("general", "test_harth_general"), ("older", "test_har70plus")):
            if grp in gg.index:
                rows.append({"eval_set": es, "compression": "ptq/qat 2-bit per channel", "level": "20.7 KB",
                             "scenario": "retrained_calibration", "method": mid.split("-")[0], "variant": mid,
                             "n_folds": 5, "accuracy_mean": gg.accuracy[grp], "lci_mean": gg.lci[grp], "fa_lci": fa_})
    parts.append(tidy(pd.DataFrame(rows), "har-pair-calibration test (agent A)", True))
    s = pd.read_csv(RES / "har-pair-scarce" / "test" / "har-pair-scarce-tinyhar-test-means-per-population-and-model.csv")
    rows = []
    for mid, gg in s.groupby("model_id"):
        gg = gg.set_index("population")
        fa_ = min(gg.lci.get("harth_all", np.nan), gg.lci.get("har70plus", np.nan))
        for pop, es in (("harth_all", "test_harth_general"), ("har70plus", "test_har70plus"), ("primary_untouched", "test_primary_untouched")):
            if pop in gg.index:
                rows.append({"eval_set": es, "compression": "ptq/qat 2-bit per channel", "level": "20.7 KB",
                             "scenario": "retrained_calibration (~100 windows)", "method": mid.split("-")[0], "variant": mid,
                             "n_folds": 5, "accuracy_mean": gg.accuracy[pop], "lci_mean": gg.lci[pop], "fa_lci": fa_})
    parts.append(tidy(pd.DataFrame(rows), "har-pair-scarce test (agent A)", True))
    return parts


def main() -> None:
    v = pd.concat(validation_rows(), ignore_index=True)
    t = pd.concat(test_rows(), ignore_index=True)
    d = pd.concat([v, t], ignore_index=True)
    if OUT.exists():   # sources whose files were deleted (agent A's PAIR, agent B's GMR phase 3): keep their rows frozen
        prev = pd.read_csv(OUT)
        prev = prev[~prev.source.isin(d.source.unique())]
        d = pd.concat([d, prev.drop(columns=["flags"], errors="ignore")], ignore_index=True)
    d["flags"] = [flags(r) for r in d.to_dict("records")]
    d.loc[d.uncommitted.fillna(False).astype(bool), "flags"] = (d["flags"] + "; previous agent").str.strip("; ")
    cols = ["source", "eval_set", "compression", "level", "scenario", "method", "variant", "n_folds", "n_runs",
            "accuracy_mean", "accuracy_sd", "f1_mean", "f1_sd", "lci_mean", "lci_sd", "fa_lci", "fa_lci_ci",
            "lci_indep_mean", "self_lci_mean", "occ_lci_mean", "agree_mean", "lci_agree_mean", "flags"]
    for c in cols:
        if c not in d:
            d[c] = np.nan
    d = d[cols].sort_values(["eval_set", "compression", "scenario", "level", "lci_mean"], ascending=[True, True, True, True, False])
    d.round(4).to_csv(OUT, index=False, lineterminator="\n")
    print(f"{len(d)} cells -> {OUT}")
    print(d.groupby(["eval_set", "source"]).size().to_string())


if __name__ == "__main__":
    main()
