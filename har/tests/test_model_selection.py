"""Tests for the model-choice step: tuning subjects, no HAR70+ loading, resume, merging and the
selection rule (plan 2.9, Step 4 acceptance)."""

import numpy as np
import pandas as pd
import pytest

from har import data as D
from har import folds as F
from har import run_model_selection as RMS
from har.paths import RAW_DIR, WINDOW_CACHE_NPZ
from har.select_model import apply_rule, dedupe

HAS_CACHE = WINDOW_CACHE_NPZ.exists()
HARTH = [f"S{i:03d}" for i in (6, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29)]
OLDER = [str(i) for i in range(501, 519)]


# ------------------------------------------------------------------------------- selection rule

# measured CPU SHAP s/window (torch 2.7.1) and parameter counts of the source-checked models (environment/time.md)
SHAP = {"cnn1d": 0.45, "tinyhar": 0.08, "deepconvlstm": 0.70, "hart": 0.22}
PARAMS = {"cnn1d": 1_254_919, "tinyhar": 54_273, "deepconvlstm": 458_183, "hart": 1_389_555}


def cand(cnn, tiny, dcl, hart, params=None, shap=None):
    params, shap = params or PARAMS, shap or SHAP
    f = {"cnn1d": cnn, "tinyhar": tiny, "deepconvlstm": dcl, "hart": hart}
    return {m: {"f1": f[m], "n_params": params[m], "shap_s": shap[m]} for m in f}


def test_rule_best_f1_wins_when_nothing_else_is_within_2_points():
    assert apply_rule(cand(80, 81, 90, 83))[0] == "deepconvlstm"      # slowest SHAP, but alone in the margin
    assert apply_rule(cand(85, 70, 70, 70))[0] == "cnn1d"


def test_rule_lowest_shap_time_within_2_points_of_the_best():
    assert apply_rule(cand(84, 83.2, 85, 83.5))[0] == "tinyhar"       # all within 2 pt of 85 -> fastest SHAP
    assert apply_rule(cand(84, 82.9, 85, 83.5))[0] == "hart"          # TinyHAR 2.1 pt behind -> out; HART next fastest
    choice, trace = apply_rule(cand(80, 78.0, 70, 70))                # exactly 2.0 pt behind is still eligible
    assert choice == "tinyhar" and "eligible" in trace[2] and "tinyhar" in trace[2]


def test_rule_parameter_count_breaks_exact_shap_ties():
    shap = {"cnn1d": 0.2, "tinyhar": 0.2, "deepconvlstm": 1.0, "hart": 0.2}
    choice, trace = apply_rule(cand(80, 80, 80, 80, shap=shap))
    assert choice == "tinyhar" and "tie" in trace[-1]                 # 54K < 1.25M < 1.39M parameters
    shap["tinyhar"] = 0.2000001                                       # no exact tie -> SHAP time decides
    assert apply_rule(cand(80, 80, 80, 80, shap=shap))[0] == "cnn1d"


def test_rule_needs_all_deep_candidates_and_their_shap_times():
    c = cand(80, 80, 80, 80)
    del c["hart"]
    with pytest.raises(ValueError, match="missing"):
        apply_rule(c)
    c = cand(80, 80, 80, 80)
    c["tinyhar"]["shap_s"] = None
    with pytest.raises(ValueError, match="SHAP time"):
        apply_rule(c)


# ------------------------------------------------------------------------------ tuning subjects

def test_tuning_uses_fold0_validation_subjects_only():
    folds = F.make_folds({s: 2000 for s in HARTH}, OLDER)
    t = F.tuning_sets(folds[RMS.TUNING_FOLD])
    assert set(t["tuning_validation"]) == set(folds[0].val) and len(t["tuning_validation"]) == 3
    assert not set(t["tuning_train"] + t["tuning_validation"]) & set(folds[0].test_general)
    assert not set(t["tuning_train"] + t["tuning_validation"]) & set(OLDER)


def test_grid_has_four_settings_per_model_and_unique_ids():
    for m in ("cnn1d", "tinyhar", "deepconvlstm", "hart", "xgboost"):
        ids = [RMS.setting_id(m, hp) for hp in RMS.grid(m)]
        assert len(ids) == 4 == len(set(ids))


# ------------------------------------------------------------------------- no HAR70+ loading

@pytest.mark.skipif(not HAS_CACHE, reason="window cache not built")
def test_load_harth_windows_never_reads_har70plus_arrays(monkeypatch):
    seen = []
    real_load = np.load

    class Spy:
        def __init__(self, z):
            self.z = z
        def __enter__(self):
            return self
        def __exit__(self, *a):
            self.z.close()
        def __getitem__(self, k):
            seen.append(k)
            return self.z[k]

    monkeypatch.setattr(D.np, "load", lambda p, *a, **k: Spy(real_load(p, *a, **k)))
    monkeypatch.setattr(D, "load_recording", lambda *a, **k: (_ for _ in ()).throw(AssertionError("raw CSV opened")))
    w = D.load_harth_windows()
    assert set(w) == {"harth_train", "harth_test"}
    assert seen and not [k for k in seen if k.startswith("har70plus")]


def test_runner_never_references_har70plus_windows():
    src = open(RMS.__file__, encoding="utf-8").read()
    assert "har70plus_test" not in src and "build_all" not in src and "load_harth_windows" in src


# -------------------------------------------------------------------------------- resume / merge

def write_row(d, stage, model, sid, fold, machine, h, status="done", f1=0.5):
    p = RMS.unit_paths(d, d, stage, model, sid, fold, machine)["row"]
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"stage": stage, "model": model, "setting_id": sid, "fold": fold, "status": status,
                   "settings_hash": h, "machine_name": machine, "best_val_macro_f1_pooled7": f1}]).to_csv(p, index=False)
    # companion files must not be mistaken for result rows
    pd.DataFrame({"epoch": [0]}).to_csv(p.with_name(p.stem + "-history.csv"), index=False)
    return p


def test_find_done_sees_rows_from_any_machine_and_ignores_companions(tmp_path):
    write_row(tmp_path, "final", "cnn1d", None, 2, "gpu-a", "abc")
    found = RMS.find_done(tmp_path, "final", "cnn1d", None, 2)
    assert len(found) == 1 and found[0].machine_name.iloc[0] == "gpu-a"
    assert RMS.find_done(tmp_path, "final", "cnn1d", None, 3) == []


def test_best_setting_reads_all_tuning_rows(tmp_path):
    for i, hp in enumerate(RMS.grid("cnn1d")):
        write_row(tmp_path, "tune", "cnn1d", RMS.setting_id("cnn1d", hp), 0, "m1" if i % 2 else "m2", f"h{i}",
                  f1=[0.5, 0.7, 0.6, 0.7][i])
    hp, sid, f1 = RMS.best_setting(tmp_path, "cnn1d")
    assert f1 == 0.7 and hp == RMS.grid("cnn1d")[1]                 # tie -> earlier in the grid
    (tmp_path / "tuning" / f"har-tune-cnn1d-{RMS.setting_id('cnn1d', RMS.grid('cnn1d')[3])}-fold0-m1.csv").unlink()
    with pytest.raises(RuntimeError, match="tuning of cnn1d"):
        RMS.best_setting(tmp_path, "cnn1d")


def test_dedupe_resolves_or_rejects_duplicate_machines():
    df = pd.DataFrame([{"model": "cnn1d", "fold": 0, "settings_hash": "x", "machine_name": "a", "row_file": "1"},
                       {"model": "cnn1d", "fold": 0, "settings_hash": "x", "machine_name": "b", "row_file": "2"}])
    with pytest.raises(RuntimeError, match="several machines"):
        dedupe(df, ["model", "fold"], None)
    assert dedupe(df, ["model", "fold"], "b").machine_name.tolist() == ["b"]
    df.loc[1, "settings_hash"] = "y"
    with pytest.raises(RuntimeError, match="different settings"):
        dedupe(df, ["model", "fold"], "b")


def test_smoke_subsample_keeps_every_class():
    rng = np.random.default_rng(0)
    n = 5000
    y = rng.choice(7, n, p=[0.3, 0.05, 0.02, 0.02, 0.15, 0.4, 0.06])
    w = D.Windows(X=np.zeros((n, 6, 128), np.float32), y=y, subject=np.full(n, "S001", "<U4"), dataset="harth",
                  group=np.full(n, "general", "<U7"), walking_aid=np.full(n, D.WA_NA, "<U23"),
                  majority_share=np.ones(n, np.float32), start_time=np.zeros(n, "datetime64[ms]"),
                  start_row=np.arange(n, dtype=np.int64))
    s = RMS.subsample(w, 300, 0)
    assert len(s) == 300 and set(s.y) == set(range(7))
