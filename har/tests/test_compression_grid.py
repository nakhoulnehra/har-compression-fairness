"""Tests for the Step 9 grid runner har/run_compression_grid.py (plan Section 3 Step 9, part A).

    plan        method registry and grid plan: Park only at 70/90%, every fine-tuned method E_ft = 20,
                one no_finetune start per score and level, unique model IDs, XGP can be registered later
    sizes       identical n_nonzero for every pruning method at a level; head never pruned
    PTQ         Tulla-style = lci.compression.ptq unchanged; quantizes every parameter
    metrics     per-subject and per-class LCI, retention, confusion round trip and pooled stairs F1
    HAR70+      never reaches fine-tuning (finetune refuses non-HARTH windows)
    analysis    pooling over folds, FA-LCI = worst group, gaps, Wilcoxon families, safe rate, per-class
"""

import inspect

import numpy as np
import pandas as pd
import pytest
import torch

import har.run_compression_grid as G
from har.compression import finetune, l1_scores, prune_by_score, ptq_tulla, size_record, snip_scores
from har.tests.test_compression import fake_windows, tinyhar


# ------------------------------------------------------------------------------ plan

def test_design_constants_part_a():
    assert G.GRID["levels"] == (0.3, 0.5, 0.7, 0.9) and G.GRID["park_levels"] == (0.7, 0.9)
    assert G.GRID["epochs"] == 20 and G.GRID["lr"] == 1e-4 and G.GRID["ptq_bits"] == (8, 4)
    assert set(G.PART_A) == {"l1", "l1_kd", "snip", "snip_kd", "park", "park_kd"}
    assert G.METHODS["l1"].recipe == "ce" and G.METHODS["l1_kd"].recipe == "kd"
    assert G.METHODS["snip"].score == "snip" and G.METHODS["park_kd"].park and G.METHODS["park_kd"].recipe == "kd"
    assert set(G.PTQ) == {"ptq8", "ptq4", "ptq_tulla8", "ptq_tulla4"}


def test_grid_plan_structure():
    plan = G.grid_plan()
    ids = [p["model_id"] for p in plan]
    assert len(ids) == len(set(ids))
    ft = [p for p in plan if p["scenario"] == "finetune"]
    nft = [p for p in plan if p["family"] == "pruning" and p["scenario"] == "no_finetune"]
    assert {p["sparsity"] for p in ft if p["method"].startswith("park")} == {0.7, 0.9}
    assert len(ft) == 4 * 4 + 2 * 2                          # l1, l1_kd, snip, snip_kd at 4 levels; park x2 at 2
    assert len(nft) == 2 * 4                                 # one start per score (l1, snip) per level
    for p in ft:                                             # every fine-tuned model has its start assessed
        assert any(q["score"] == p["score"] and q["sparsity"] == p["sparsity"] for q in nft)
    assert all(p["epochs"] == 20 for p in ft)
    assert [p["model_id"] for p in plan if p["family"] == "ptq"] == ["ptq8", "ptq4", "ptq_tulla8", "ptq_tulla4"]
    assert plan[0]["model_id"] == "original"
    # order: PTQ first, then by level; within a level the starts come before the fine-tuned models
    lv = [p["sparsity"] for p in plan if p["family"] == "pruning"]
    assert lv == sorted(lv)


def test_grid_plan_method_subset_and_pending():
    plan = G.grid_plan(methods=("l1",), levels=(0.5,), ptq=())
    assert [p["model_id"] for p in plan] == ["original", "l1-50pct-no_finetune", "l1-50pct-finetune"]
    assert [p["model_id"] for p in G.pending(plan, {"original", "l1-50pct-no_finetune"})] == ["l1-50pct-finetune"]


def test_a_new_method_can_be_registered_like_xgp():
    calls = []

    def fake_scorer(ctx):
        calls.append(ctx)
        return {"w": torch.ones(1)}

    try:
        G.register_method(G.Method("xgp_kd", score="xgp", recipe="kd"), scorer=fake_scorer)
        plan = G.grid_plan(methods=("xgp_kd",), levels=(0.7,), ptq=())
        assert [p["model_id"] for p in plan] == ["original", "xgp-70pct-no_finetune", "xgp_kd-70pct-finetune"]
        assert G.SCORERS["xgp"]({"ctx": 1}) == {"w": torch.ones(1)} and calls
    finally:
        G.METHODS.pop("xgp_kd", None)
        G.SCORERS.pop("xgp", None)


# ------------------------------------------------------------------------------ sizes and PTQ

def test_identical_n_nonzero_across_scores_and_head_unpruned():
    net, tr = tinyhar(), fake_windows(64)
    head = net.head.weight.detach().clone()
    for lv in G.GRID["levels"]:
        a = prune_by_score(net, l1_scores(net), lv)
        b = prune_by_score(net, snip_scores(net, tr, n_batches=2, batch_size=32), lv)
        assert size_record(a)["n_nonzero"] == size_record(b)["n_nonzero"]
        assert torch.equal(a.head.weight, head) and torch.equal(b.head.weight, head)
        assert size_record(a)["sparsity_prunable"] == pytest.approx(lv, abs=1e-3)


def test_ptq_tulla_equals_lci_ptq():
    from lci.compression import ptq

    net = tinyhar()
    q, quantized = ptq_tulla(net, 4)
    ref, _ = ptq(net, 4)
    for (n1, p1), (n2, p2) in zip(q.named_parameters(), ref.named_parameters()):
        assert n1 == n2 and torch.equal(p1, p2)
    assert set(quantized) == {n for n, _ in net.named_parameters()} and set(quantized.values()) == {4}
    assert size_record(q, quantized)["size_kb_nominal"] == pytest.approx(size_record(q)["n_params"] * 0.5 / 1024)


def test_finetune_refuses_har70plus():
    net, tr = tinyhar(), fake_windows(32)
    older = fake_windows(32, dataset="har70plus")
    with pytest.raises(Exception):
        finetune(prune_by_score(net, l1_scores(net), 0.5), older, tr, epochs=1)


def test_grid_training_side_reads_only_harth():
    src = inspect.getsource(G.Context.prepare_training)
    assert "har70plus" not in src and "test[" not in src


# ------------------------------------------------------------------------------ metrics

def _attr(n, seed):
    return np.random.default_rng(seed).normal(size=(n, 6, 128)).astype(np.float32)


def test_explanation_metrics_identity_and_per_class():
    A = _attr(24, 0)
    subj = np.array(["S1"] * 12 + ["501"] * 12)
    grp = np.array(["general"] * 12 + ["older"] * 12)
    lab = np.array([0, 0, 0, 1, 1, 1, 2, 2, 4, 4, 4, 4] * 2)
    s_rows, c_rows, g_rows = G.explanation_metrics(A, A.copy(), subj, grp, lab, min_class_windows=3)
    assert {r["subject"] for r in s_rows} == {"S1", "501"}
    assert all(r["lci"] == pytest.approx(1.0) and r["lci768"] == pytest.approx(1.0) for r in s_rows)
    # class 2 has 2 windows per subject < 3 -> no per-subject row, but the pooled group rows have 2 + ... windows
    assert {(r["subject"], r["class_index"]) for r in c_rows} == {(s, c) for s in ("S1", "501") for c in (0, 1, 4)}
    assert {r["n_windows"] for r in c_rows if r["class_index"] == 4} == {4}
    gk = {(r["group"], r["class_index"]): r for r in g_rows}
    assert ("older", 2) not in gk and gk[("general", 0)]["n_windows"] == 3 and gk[("older", 4)]["lci"] == pytest.approx(1)
    # walking-aid subgroup rows appear for HAR70+ subjects in that group
    s2, _, g2 = G.explanation_metrics(A, A.copy(), np.array(["S1"] * 12 + ["503"] * 12), grp, lab)
    assert any(r["group"] == "walking_aid_inferred" for r in g2) and not any(r["group"] == "other_older" for r in g2)


def test_explanation_metrics_detects_drift():
    A = _attr(20, 1)
    B = A.copy()
    B[10:] = _attr(10, 2)                                    # older windows replaced by noise
    s_rows, _, _ = G.explanation_metrics(A, B, np.array(["S1"] * 10 + ["501"] * 10),
                                         np.array(["general"] * 10 + ["older"] * 10), np.zeros(20, int))
    by = {r["subject"]: r for r in s_rows}
    assert by["S1"]["lci"] == pytest.approx(1.0) and by["501"]["lci"] < 0.5


def test_retention():
    assert G.retention(0.66, 0.66, 0.95) == pytest.approx(0.0)
    assert G.retention(0.95, 0.66, 0.95) == pytest.approx(1.0)


def test_confusion_round_trip_and_pooled_stairs_f1():
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(0)
    y1, p1 = rng.integers(0, 7, 200), rng.integers(0, 7, 200)
    y2, p2 = rng.integers(0, 7, 150), rng.integers(0, 7, 150)
    rows = G.accuracy_rows(np.concatenate([y1, y2]), np.concatenate([p1, p2]), np.array(["A"] * 200 + ["B"] * 150))
    assert [r["subject"] for r in rows] == ["A", "B"]
    cms = [G.parse_confusion(r["confusion7"]) for r in rows]
    assert cms[0].sum() == 200 and cms[1].sum() == 150
    pooled = G.pooled_class_f1_from_confusion(cms[0] + cms[1], (2, 3))
    y, p = np.concatenate([y1, y2]), np.concatenate([p1, p2])
    assert pooled[2] == pytest.approx(f1_score(y, p, labels=[2], average="macro"))
    assert pooled[3] == pytest.approx(f1_score(y, p, labels=[3], average="macro"))
    assert not any("stairs" in k and "f1" in k for k in rows[0])          # never per-subject stairs F1


# ------------------------------------------------------------------------------ analysis

def synthetic_per_subject(n_folds=2):
    rows = []
    general = {0: ["S1", "S2", "S3"], 1: ["S4", "S5", "S6"]}
    older = ["501", "502", "503", "505", "506"]
    rng = np.random.default_rng(0)
    for k in range(n_folds):
        for method, base in (("l1", 0.8), ("snip", 0.7)):
            for s in general[k] + older:
                g = "general" if s.startswith("S") else "older"
                lci = base + (0 if g == "general" else -0.25) + rng.normal(0, 0.01)
                rows.append({"fold": k, "model_id": f"{method}-70pct-finetune", "method": method, "family": "pruning",
                             "scenario": "finetune", "level": "70%", "subject": s, "group": g, "lci": lci,
                             "shap_corr": lci, "jaccard": lci, "accuracy": 0.9, "non_stair_macro_f1": 0.8,
                             "aopc": 0.1, "retention": 0.5, "tau_k": 0.6 + 0.01 * k,
                             "confusion7": G.confusion_str(np.eye(7, dtype=int) * 3)})
    return pd.DataFrame(rows)


def test_pool_and_fa_lci_and_gap():
    ps = synthetic_per_subject()
    pooled = G.pool_subjects(ps, ["lci", "accuracy"])
    assert len(pooled[pooled.method == "l1"]) == 6 + 5          # 6 general (one fold each) + 5 older (averaged)
    fa = G.fa_lci_table(pooled, n_boot=500).set_index("method")
    assert fa.loc["l1", "fa_lci"] == pytest.approx(min(fa.loc["l1", "lci_older"], fa.loc["l1", "lci_general"]))
    assert fa.loc["l1", "worst_group"] == "older" and fa.loc["l1", "gap_older_minus_general"] == pytest.approx(-0.25, abs=0.02)
    gs = G.group_summary(pooled, ["lci"], n_boot=500)
    r = gs[(gs.method == "l1") & (gs.metric == "lci")].iloc[0]
    assert r.gap_ci_high < 0 and r.n_general == 6 and r.n_older == 5 and r.n_walking_aid_inferred == 2


def test_wilcoxon_table_pairs_within_level_and_scenario_with_holm():
    pooled = G.pool_subjects(synthetic_per_subject(), ["lci", "non_stair_macro_f1", "aopc"])
    w = G.wilcoxon_table(pooled, ["lci"])
    prim = w[w.population == "all"]
    assert len(prim) == 1 and prim.iloc[0].n == 11 and prim.iloc[0].median_diff == pytest.approx(0.1, abs=0.02)
    assert {"p_holm", "rank_biserial_r", "cohens_dz"} <= set(w.columns)
    assert set(w.population) == {"all", "general", "older"}


def test_safe_rate_table():
    ps = synthetic_per_subject()
    sr = G.safe_rate_table(ps, tulla_tau=0.85)
    r = sr[(sr.method == "l1") & (sr.unit == "fold_setting") & (sr.threshold == "tau_k")].iloc[0]
    assert r.safe_general == 1.0 and r.safe_older == 0.0 and r.safe_fa_lci == 0.0
    s = sr[(sr.method == "snip") & (sr.unit == "subject") & (sr.threshold == "tau_k")].iloc[0]
    assert s.safe_general == 1.0 and s.safe_older == 0.0


def test_per_class_table():
    rows = []
    for k in (0, 1):
        for s, g in (("S1" if k == 0 else "S2", "general"), ("501", "older"), ("502", "older")):
            for c, v in ((0, 0.9), (4, 0.5)):
                rows.append({"fold": k, "model_id": "l1-70pct-finetune", "method": "l1", "family": "pruning",
                             "scenario": "finetune", "level": "70%", "subject": s, "group": g, "class_index": c,
                             "class_name": "walking" if c == 0 else "standing",
                             "lci": v - (0.2 if g == "older" else 0), "n_windows": 5})
    pc = G.per_class_table(pd.DataFrame(rows), None, n_boot=200)
    r = pc[(pc.class_index == 4)].iloc[0]
    assert r.lci_general == pytest.approx(0.5) and r.lci_older == pytest.approx(0.3)
    assert r.gap_older_minus_general == pytest.approx(-0.2) and r.n_general == 2 and r.n_older == 2


def test_stairs_table_pooled_never_per_subject():
    ps = synthetic_per_subject()
    st = G.stairs_table(ps, n_boot=200)
    assert {"stairs_up_f1_pooled", "stairs_down_f1_pooled"} <= set(st.columns)
    assert (st.stairs_up_f1_pooled == 1.0).all()                # diagonal confusion


def test_fold3_defaults_to_uni_2080_copy(tmp_path):
    for k, m in ((2, "uni-2080"), (3, "uni-2080"), (3, "nakhoul-3050"), (1, "nakhoul-3050")):
        pd.DataFrame({"fold": [k], "model_id": ["original"], "machine_name": [m]}).to_csv(
            tmp_path / f"har-grid-tinyhar-fold{k}-{m}-models.csv", index=False)
    assert G.FOLD_MACHINE_DEFAULT == {3: "uni-2080"}
    _, chosen = G.load_fold_files(tmp_path, dict(G.FOLD_MACHINE_DEFAULT))
    assert chosen == {1: "nakhoul-3050", 2: "uni-2080", 3: "uni-2080"}
    _, chosen = G.load_fold_files(tmp_path, {**G.FOLD_MACHINE_DEFAULT, 3: "nakhoul-3050"})
    assert chosen[3] == "nakhoul-3050"
    with pytest.raises(RuntimeError):
        G.load_fold_files(tmp_path, {})


def test_npz_keeps_window_metadata_and_checks_it_on_resume(tmp_path):
    meta = {"meta_window_id": np.array(["a", "b"]), "meta_subject": np.array(["S1", "501"]),
            "meta_group": np.array(["general", "older"]), "meta_class_index": np.array([0, 4])}
    st = G.FoldStore(tmp_path, 0, "m", resume=False, smoke=False)
    st.set_window_meta(meta)
    st.seg["original"] = np.zeros((2, 48), np.float32)
    st.add("models", [{"model_id": "original"}])
    st.save()
    st2 = G.FoldStore(tmp_path, 0, "m", resume=True, smoke=False)
    assert st2.done == {"original"} and list(st2.seg["meta_subject"]) == ["S1", "501"]
    st2.set_window_meta(meta)                                    # same windows: fine
    with pytest.raises(RuntimeError):
        st2.set_window_meta({**meta, "meta_subject": np.array(["S1", "502"])})
