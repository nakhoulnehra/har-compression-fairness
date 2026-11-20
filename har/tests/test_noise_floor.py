"""Noise floor summaries (har/run_noise_floor.py; plan Section 2.7): tau_k and the per-group check."""

import numpy as np
import pandas as pd
import pytest

from har.run_noise_floor import group_summary, tau_from_rows


def rows():
    out = []
    lci = {("retrain", 1): 0.80, ("retrain", 2): 0.70, ("retrain", 3): 0.75, ("shap", 1): 0.95, ("shap", 2): 0.93}
    for (kind, seed), base in lci.items():
        for group, subjects, shift in (("inner_val", ("S1", "S2", "S3"), 0.0), ("general", ("S4", "S5"), 0.02),
                                       ("older", ("501", "502"), -0.05)):
            for i, s in enumerate(subjects):
                out.append({"fold": 0, "noise": kind, "seed": seed, "group": group, "subject": s,
                            "lci": base + shift + (i - 1) * 0.01})
    return pd.DataFrame(out)


def test_tau_is_min_over_retrain_seeds_of_inner_val_subject_mean():
    df = rows()
    tau = tau_from_rows(df)
    # inner-val subject means per retrain seed: 0.80, 0.70, 0.75 (offsets -0.01, 0, +0.01 cancel)
    assert tau == pytest.approx(0.70)
    df.loc[(df.group == "older") & (df.noise == "retrain"), "lci"] = -1.0    # HAR70+ never moves tau
    df.loc[(df.group == "general") & (df.noise == "retrain"), "lci"] = -1.0
    assert tau_from_rows(df) == pytest.approx(0.70)
    df.loc[(df.group == "inner_val") & (df.noise == "shap"), "lci"] = -1.0   # neither does SHAP noise
    assert tau_from_rows(df) == pytest.approx(0.70)


def test_group_summary_and_shap_above_retrain_check():
    s = group_summary(rows())
    r = s.set_index(["group", "noise"])
    assert r.loc[("inner_val", "retrain"), "n_values"] == 3 and r.loc[("inner_val", "shap"), "n_values"] == 2
    assert r.loc[("inner_val", "retrain"), "min"] == pytest.approx(0.70)
    assert r.loc[("inner_val", "retrain"), "max"] == pytest.approx(0.80)
    assert r.loc[("older", "shap"), "min"] == pytest.approx(0.875)   # 0.93 - 0.05 - 0.005
    checks = s.drop_duplicates("group").set_index("group")["shap_above_retrain"]
    assert checks.all()
    bad = rows()
    bad.loc[(bad.group == "general") & (bad.noise == "shap"), "lci"] = 0.5
    s2 = group_summary(bad).drop_duplicates("group").set_index("group")["shap_above_retrain"]
    assert not s2["general"] and s2["inner_val"] and s2["older"]
