"""Test-set confirmation of the candidate gaps from EXISTING one-shot test runs (no new test run, nothing selected on test).

    python -m gap_study_v3.test_confirmation

Per subject (22 HARTH test subjects, 18 HAR70+ subjects, every fold's model pooled as in 9A), the gap is
ceiling_s - LCI_s, with ceiling_s = the 9A SHAP-noise LCI of the original (SHAP seed 1 vs seed 0, same windows).
  candidate 1 (no retraining, 80 %; also 70 / 90 %): agent B's test of l1_fullrecon and lamp_recon (seed 0)
  candidate 2 (KD fine-tuning, 90 %; also 70 %): 9A's test of l1_kd, park_kd, park, snip_kd (seed 0)
Reported per group: mean LCI, mean ceiling, gap, subjects with gap >= 0.10, one-sided Wilcoxon (gap > 0), accuracy
and its change vs the original, FA-LCI = min over groups.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "gap-study-v3" / "test-confirmation"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    a = pd.read_csv(RES / "har-compression-grid-analysis" / "har-grid-a-per-subject-pooled-over-folds.csv")
    ceil = a[a.method == "noise_shap_seed1"].set_index("subject").lci.rename("ceiling")
    orig = a[a.method == "original"].set_index("subject").accuracy.rename("acc0")
    rows = []
    nine = a[a.scenario == "finetune"]
    for (m, lv), g in nine.groupby(["method", "level"]):
        rows.append(("kd_finetune (9A)" if m.endswith("_kd") or m == "park_kd" else "ce_finetune (9A)", m, lv, g))
    b = pd.read_csv(RES / "gap-study-v3" / "agentB-no-retrain-test" / "gap-study-test-tinyhar-per-subject-pooled-over-folds.csv")
    b = b[~b.model_id.str.startswith("gmr") & b.model_id.str.contains("pct")]
    for mid, g in b.groupby("model_id"):
        rows.append(("no_retrain (agent B test)", mid.split("-")[0], mid.split("-")[1], g))
    out = []
    for scen, m, lv, g in rows:
        g = g.set_index("subject").join(ceil).join(orig)
        g["gap"] = g.ceiling - g.lci
        rec = {"scenario": scen, "method": m, "level": lv}
        for grp in ("general", "older", "all"):
            x = g if grp == "all" else g[g.group == grp]
            w = stats.wilcoxon(x.gap, alternative="greater") if len(x) > 5 else None
            rec.update({f"lci_{grp}": x.lci.mean(), f"ceiling_{grp}": x.ceiling.mean(), f"gap_{grp}": x.gap.mean(),
                        f"n_{grp}": len(x), f"subjects_gap_ge_0.10_{grp}": int((x.gap >= 0.10).sum()),
                        f"wilcoxon_p_{grp}": float(w.pvalue) if w else np.nan,
                        f"accuracy_{grp}": x.accuracy.mean(), f"d_acc_pt_{grp}": 100 * (x.accuracy - x.acc0).mean()})
        rec["fa_lci"] = min(rec["lci_general"], rec["lci_older"])
        rec["fa_ceiling"] = min(rec["ceiling_general"], rec["ceiling_older"])
        rec["fa_gap"] = rec["fa_ceiling"] - rec["fa_lci"]
        out.append(rec)
    t = pd.DataFrame(out).sort_values(["scenario", "level", "lci_all"], ascending=[True, True, False])
    t.round(4).to_csv(OUT / "gap-study-v3-test-confirmation-per-subject-gaps-existing-test-runs.csv", index=False,
                      lineterminator="\n")
    pd.set_option("display.width", 250)
    cols = ["scenario", "method", "level", "lci_general", "gap_general", "subjects_gap_ge_0.10_general",
            "wilcoxon_p_general", "d_acc_pt_general", "lci_older", "gap_older", "subjects_gap_ge_0.10_older",
            "wilcoxon_p_older", "d_acc_pt_older", "fa_lci", "fa_gap"]
    print(t[cols].round(3).to_string())


if __name__ == "__main__":
    main()
