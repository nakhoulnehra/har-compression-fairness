"""Tests for XGP round 4 (plan Section 2.5 (b), "Round 4 pre-registration"; har/xgp4.py, har/run_xgp_pilot4.py).

    groups       default group = subject, integer codes, calibration pick per (group, class) cell, class-balanced
                 weights, weighted importance = round 3's with unit weights, worst-group mean (k = 1 max,
                 k >= G mean), worst-group importance / shares, group DRO over several subject groups
    budgets      water-filling (sum, bounds, proportionality, infeasible), LAMP counts, bounds protect the
                 tensors the reconstruction does not refit, XGP shares sum to 1
    unstructured budgeted iterative pruning: exact per-tensor budgets, uniform total n_nonzero, >= 1 per row,
                 nested masks, reconstruction keeps the masks
    structured   masks (rows + consumer columns), physical slicing = masked model (also after fine-tuning),
                 nested iterative filter removal, identical widths across criteria, Taylor / L1 scores, MACs
    rules        beats (gain, paired count, F1 tolerance, worst-subject LCI), final on LCI_val with tie order,
                 round-3 config lookup / xgp_r3 recipe
    guards       round 3 files untouched by the runner; HAR70+ never loaded; older-like code only in xgp_r3
"""

import copy
import inspect

import numpy as np
import pytest
import torch
import torch.nn as nn

from har.compression import finetune, l1_scores, prunable_tensors, prune_by_score, ptq_symmetric, size_record
from har.tests.test_compression import fake_windows, tinyhar
from har.xgp3 import alignment_importance_ref, layerwise_reconstruct
from har.xgp4 import (ROUND4, S_DRO_CONTROLS, S_METHODS, U_BASELINES, U_CANDIDATES, U_CELLS, U_DRO_CONTROLS,
                      apply_structured, beats, bounded_budget, budget_bounds, calibration_pick, class_balanced_weights,
                      count_macs, cpu_latency, default_groups, filter_scores_l1, filter_scores_taylor, group_index,
                      iterative_prune_budget, iterative_prune_structured, lamp_keep_counts, per_group_importance,
                      pick_final, slice_structured, structured_layers, uniform_keep_counts, unreconstructed_tensors,
                      water_fill, weighted_alignment_importance, worst_group_importance, worst_group_mean,
                      worst_group_shares)


def X_(n=16, seed=0):
    return torch.randn(n, 6, 128, generator=torch.Generator().manual_seed(seed))


def test_design_constants():
    assert ROUND4["levels_u"] == (0.8, 0.9, 0.95) and ROUND4["levels_u_kd5"] == (0.9, 0.95)
    assert ROUND4["levels_s"] == (0.5, 0.75) and ROUND4["seeds"] == (0, 1, 2) and ROUND4["folds"] == (0, 1)
    assert ROUND4["gain"] == 0.02 and ROUND4["min_better"] == 5 and ROUND4["f1_tolerance"] == 0.01
    assert ROUND4["beta_u"] == 0.1 and ROUND4["beta_s"] == 1.0 and ROUND4["K"] == 5 and ROUND4["epochs"] == 20
    assert ROUND4["curve_epochs"] == (2, 5) and ROUND4["kd5_fraction"] == 0.05 and ROUND4["floor_frac"] == 0.02
    assert set(U_CELLS.values()) == {(b, s) for b in ("uniform", "lamp", "xb") for s in ("l1", "xgp")}
    assert U_CANDIDATES == ("xgp", "l1_xb_recon", "xgp_lamp", "xgp_xb")
    assert set(S_METHODS.values()) == {"l1", "taylor", "xgp"}
    assert ROUND4["top_k"] == 3 and ROUND4["cal_per_cell"] == 10 and ROUND4["wsl_tolerance"] == 0.01
    assert ROUND4["dro_eta"] == 0.01 and not any("older" in k or "aug" in k for k in ROUND4)
    assert set(U_DRO_CONTROLS.values()) == set(U_BASELINES) and all(k.endswith("_kd_dro") for k in U_DRO_CONTROLS)
    assert set(S_DRO_CONTROLS.values()) == {"l1_filter", "l1_filter_recon", "taylor_recon"}


# ------------------------------------------------------------------------------ groups and importance

def test_default_groups_index_and_calibration_pick():
    subj = np.array(["S2", "S1", "S2", "S1", "S3"])
    assert default_groups(subj).tolist() == subj.tolist()
    assert default_groups(subj, ["a", "a", "b", "b", "b"]).tolist() == ["a", "a", "b", "b", "b"]
    with pytest.raises(ValueError):
        default_groups(subj, ["a"])
    codes, labels = group_index(subj)
    assert labels == ["S1", "S2", "S3"] and codes.tolist() == [1, 0, 1, 0, 2]
    codes, _ = group_index(["S3"], labels)
    assert codes.tolist() == [2]
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 300)
    g = np.array([f"S{i}" for i in rng.integers(0, 4, 300)])
    pick = calibration_pick(y, g, 5, 46)
    assert len(set(pick.tolist())) == len(pick)
    for gg in set(g.tolist()):
        for c in range(3):
            n_cell = int(((g == gg) & (y == c)).sum())
            assert int(((g[pick] == gg) & (y[pick] == c)).sum()) == min(5, n_cell)
    assert np.array_equal(pick, calibration_pick(y, g, 5, 46))
    w = class_balanced_weights(np.array([0, 0, 0, 1, 2, 2]))
    assert abs(w.sum() - 1) < 1e-12 and abs(w[:3].sum() - w[3]) < 1e-12 and abs(w[4:].sum() - 1 / 3) < 1e-12


def test_weighted_importance_with_unit_weights_is_round3_importance():
    net = tinyhar()
    X, bg = X_(10), X_(6, 1)
    t = torch.arange(10) % 7
    a = weighted_alignment_importance(net, net, X, t, np.ones(10), bg, m=2, eg_seeds=(0, 1), batch_size=4)
    b = alignment_importance_ref(net, net, X, t, bg, m=2, eg_seeds=(0, 1), batch_size=4)
    for n in b:
        torch.testing.assert_close(a[n], b[n], rtol=1e-4, atol=1e-7)


def test_worst_group_mean_k():
    G = [{"w": torch.tensor([1.0, 5.0])}, {"w": torch.tensor([3.0, 1.0])}, {"w": torch.tensor([2.0, 3.0])}]
    torch.testing.assert_close(worst_group_mean(G, 1)["w"], torch.tensor([3.0, 5.0]))
    torch.testing.assert_close(worst_group_mean(G, 2)["w"], torch.tensor([2.5, 4.0]))
    torch.testing.assert_close(worst_group_mean(G, 3)["w"], torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(worst_group_mean(G, 10)["w"], torch.tensor([2.0, 3.0]))


def test_worst_group_importance_and_shares():
    net = tinyhar()
    X, bg = X_(18), X_(6, 1)
    y = np.arange(18) % 3
    t = torch.from_numpy(y)
    g = np.array(["A"] * 6 + ["B"] * 6 + ["C"] * 6)
    raw = per_group_importance(net, net, X, t, y, g, bg, m=2, eg_seeds=(0,), batch_size=7)
    assert set(raw) == {"A", "B", "C"}
    I = worst_group_importance(net, net, X, t, y, g, bg, k=2, m=2, eg_seeds=(0,), batch_size=7)
    names = {n for n, _, _ in prunable_tensors(net)}
    assert set(I) == names and all(torch.all(v >= 0) for v in I.values())
    I1 = worst_group_importance(net, net, X, t, y, None, bg, k=2, m=2, eg_seeds=(0,), batch_size=7)
    assert set(I1) == names                                    # no groups -> one group
    sh = worst_group_shares(net, net, X, t, y, g, bg, k=2, m=2, eg_seeds=(0,), batch_size=7)
    assert set(sh) == names and abs(sum(sh.values()) - 1) < 1e-9 and min(sh.values()) >= 0
    with pytest.raises(ValueError):
        per_group_importance(net, net, X, t, y, g[:5], bg, m=2, eg_seeds=(0,))


def test_group_dro_over_several_subject_groups():
    net = tinyhar()
    tr, va = fake_windows(96), fake_windows(32, seed=1)
    groups, _ = group_index(np.array([f"S{i % 5}" for i in range(len(tr))]))
    info = finetune(net, tr, va, epochs=1, seed=0, groups=groups, dro_eta=0.01, log=lambda s: None)
    q = [info["history"][-1][f"dro_q{j}"] for j in range(5)]
    assert abs(sum(q) - 1) < 1e-6 and min(q) > 0


# ------------------------------------------------------------------------------ budgets

def test_water_fill_sum_bounds_and_proportional():
    lo, hi = {"a": 0, "b": 0, "c": 0}, {"a": 100, "b": 100, "c": 100}
    k = water_fill({"a": 0.5, "b": 0.3, "c": 0.2}, 100, lo, hi)
    assert k == {"a": 50, "b": 30, "c": 20}
    k = water_fill({"a": 0.9, "b": 0.05, "c": 0.05}, 120, {"a": 0, "b": 10, "c": 0}, {"a": 60, "b": 100, "c": 100})
    assert sum(k.values()) == 120 and k["a"] == 60 and k["b"] >= 10 and k["b"] == k["c"] == 30
    with pytest.raises(ValueError):
        water_fill({"a": 1.0}, 5, {"a": 10}, {"a": 20})


def test_unreconstructed_tensors_are_lstm_and_attention_projections():
    net = tinyhar()
    names = unreconstructed_tensors(net, X_(4))
    assert any("lstm.weight_ih" in n for n in names) and any("lstm.weight_hh" in n for n in names)
    assert any(n.endswith("in_proj_weight") for n in names) and any(n.endswith("out_proj.weight") for n in names)
    assert not any("convs" in n or "fusion" in n or "linear1" in n for n in names)


@pytest.mark.parametrize("level", [0.8, 0.95])
def test_lamp_and_xgp_budgets_keep_total_floor_and_protection(level):
    net = tinyhar()
    protect = unreconstructed_tensors(net, X_(4))
    uni = uniform_keep_counts(net, level)
    total = sum(uni.values())
    cnt = lamp_keep_counts(net, total)
    assert sum(cnt.values()) == total and min(cnt.values()) >= 1
    rng = np.random.default_rng(0)
    shares = {n: float(rng.random()) for n in uni}
    z = sum(shares.values())
    shares = {n: v / z for n, v in shares.items()}
    params = dict(net.named_parameters())
    for kind, sh in (("lamp", None), ("xb", shares)):
        keep = bounded_budget(net, level, protect, kind, sh)
        assert sum(keep.values()) == total
        for n in uni:
            assert keep[n] >= params[n].shape[0] and keep[n] >= int(np.ceil(0.02 * params[n].numel() - 1e-9))
            assert keep[n] <= params[n].numel()
            if n in protect:
                assert keep[n] >= uni[n]
    lo, _, _ = budget_bounds(net, level, protect)
    assert all(lo[n] >= uni[n] for n in protect)


# ------------------------------------------------------------------------------ budgeted iterative pruning

def test_budgeted_iterative_prune_exact_budget_rows_and_total():
    net = tinyhar()
    level = 0.9
    protect = unreconstructed_tensors(net, X_(4))
    keep = bounded_budget(net, level, protect, "lamp")
    X = X_(24)
    pruned, rows = iterative_prune_budget(net, keep, K=2, beta=0.0, importance_fn=None,
                                          recon_fn=lambda m: layerwise_reconstruct(m, net, X), log=lambda s: None)
    l1 = prune_by_score(net, l1_scores(net), level)
    assert size_record(pruned)["n_nonzero"] == size_record(l1)["n_nonzero"]
    for n, mk in pruned.prune_masks.items():
        assert int(mk.sum()) == keep[n]
        assert bool(mk.reshape(mk.shape[0], -1).any(1).all()), f"{n}: an output row lost every weight"
    params = dict(pruned.named_parameters())
    for n, mk in pruned.prune_masks.items():
        assert torch.all(params[n][~mk] == 0)
    assert len(rows) == 2 and rows[0]["n_nonzero"] > rows[1]["n_nonzero"]


# ------------------------------------------------------------------------------ structured

def test_structured_layers_exclude_last_conv():
    net = tinyhar()
    lay = structured_layers(net)
    assert len(lay) == 3
    convs = [n for n, m in net.named_modules() if isinstance(m, nn.Conv2d)]
    assert [p for p, _, _ in lay] == convs[:3] and lay[-1][2] == convs[3]


def _l1_removed(net, level):
    sc = filter_scores_l1(net)
    return {p: sorted(int(i) for i in torch.argsort(v, stable=True)[:int(round(level * len(v)))]) for p, v in sc.items()}


@pytest.mark.parametrize("level", [0.5, 0.75])
def test_slicing_equals_masked_model(level):
    net = tinyhar()
    masked = apply_structured(net, _l1_removed(net, level))
    for prod, _, cons in structured_layers(masked):
        r = masked.structured_removed[prod]
        W, Wc = dict(masked.named_modules())[prod].weight, dict(masked.named_modules())[cons].weight
        assert torch.all(W[r] == 0) and torch.all(Wc[:, r] == 0)
    sl = slice_structured(masked)
    X = X_(10)
    with torch.no_grad():
        torch.testing.assert_close(masked.eval()(X), sl(X), rtol=1e-5, atol=1e-5)
    n_dense = sum(p.numel() for p in net.parameters())
    assert sum(p.numel() for p in sl.parameters()) < n_dense
    x1 = torch.zeros(1, 6, 128)
    assert 0 < count_macs(sl, x1) < count_macs(copy.deepcopy(net), x1)
    kept = {p: dict(sl.named_modules())[p].weight.shape[0] for p, _, _ in structured_layers(sl)}
    assert set(kept.values()) == {28 - int(round(level * 28))}


def test_slicing_still_equal_after_finetune_and_ptq_keeps_zeros():
    net = tinyhar()
    masked = apply_structured(net, _l1_removed(net, 0.5))
    tr, va = fake_windows(64), fake_windows(32, seed=1)
    finetune(masked, tr, va, epochs=1, seed=0, log=lambda s: None)
    sl = slice_structured(masked)
    X = X_(8)
    with torch.no_grad():
        torch.testing.assert_close(masked.eval()(X), sl(X), rtol=1e-5, atol=1e-5)
    q, _ = ptq_symmetric(masked, 4)
    params = dict(q.named_parameters())
    for n, mk in masked.prune_masks.items():
        assert torch.all(params[n][~mk] == 0)


def test_iterative_structured_nested_identical_widths_and_recon():
    net = tinyhar()
    X = X_(24)
    tr = fake_windows(64)
    batches = [(torch.from_numpy(tr.X[:32]), torch.from_numpy(tr.y[:32]))]
    fns = {"l1": filter_scores_l1, "taylor": lambda m: filter_scores_taylor(m, batches, nn.CrossEntropyLoss())}
    widths = set()
    for name, fn in fns.items():
        pruned, rows = iterative_prune_structured(net, 0.75, K=3, score_fn=fn,
                                                  recon_fn=lambda m: layerwise_reconstruct(m, net, X), log=lambda s: None)
        assert [r["removed"][structured_layers(net)[0][0]] for r in rows] == [7, 14, 21]
        sl = slice_structured(pruned)
        widths.add(tuple(dict(sl.named_modules())[p].weight.shape[0] for p, _, _ in structured_layers(sl)))
        with torch.no_grad():
            torch.testing.assert_close(pruned.eval()(X[:4]), sl(X[:4]), rtol=1e-4, atol=1e-5)
    assert widths == {(7, 7, 7)}
    t = filter_scores_taylor(net, batches, nn.CrossEntropyLoss())
    assert all(v.shape == (28,) and torch.all(v >= 0) for v in t.values())


def test_cpu_latency_keys():
    r = cpu_latency(tinyhar(), batch=2, threads=1, warmup=1, reps=3)
    assert {"median_ms", "p10_ms", "p90_ms", "batch", "threads"} <= set(r) and r["median_ms"] > 0


# ------------------------------------------------------------------------------ rules

def _runs(vals):
    return {(f, s): v for (f, s), v in zip([(f, s) for f in (0, 1) for s in (0, 1, 2)], vals)}


F1 = {0.9: {"f1": 0.70, "worst_subject_lci": 0.60}}


def test_beats_rule():
    b = _runs([0.5] * 6)
    assert beats(_runs([0.53, 0.53, 0.53, 0.53, 0.53, 0.49]), b, F1, F1)["beats"]          # gain 0.025, 5/6
    assert not beats(_runs([0.56, 0.56, 0.53, 0.53, 0.49, 0.49]), b, F1, F1)["beats"]      # 4/6
    assert not beats(_runs([0.519] * 6), b, F1, F1)["beats"]                               # gain 0.019
    lower = {0.9: {"f1": 0.689, "worst_subject_lci": 0.60}}
    assert not beats(_runs([0.6] * 6), b, lower, F1)["beats"]                              # F1 drop 1.1 pt
    ok = {0.9: {"f1": 0.691, "worst_subject_lci": 0.591}}
    assert beats(_runs([0.6] * 6), b, ok, F1)["beats"]                                     # both within tolerance
    worse = {0.9: {"f1": 0.75, "worst_subject_lci": 0.585}}
    r = beats(_runs([0.6] * 6), b, worse, F1)
    assert not r["beats"] and not r["worst_subject_within_tol"]                            # worst subject -1.5 pt
    with pytest.raises(ValueError):
        beats({(0, 0): 0.6}, b, F1, F1)


def test_pick_final_eligibility_and_tie_order():
    best = {0.9: {"f1": 0.70}}
    st = lambda lci, f1=0.70: {0.9: {"lci": lci, "f1": f1, "worst_subject_lci": 0.5}}  # noqa: E731
    stats = {"xgp": st(0.800), "l1_xb_recon": st(0.803), "xgp_lamp": st(0.9, f1=0.68), "xgp_xb": st(0.804)}
    final, tab = pick_final(stats, best, U_CANDIDATES)
    assert final == "xgp"                                   # xgp_xb within 0.005 -> earlier in order
    assert not next(t for t in tab if t["method"] == "xgp_lamp")["eligible"]
    final, _ = pick_final({"xgp": st(0.5, f1=0.6)}, best, U_CANDIDATES)
    assert final is None


def test_round3_config_lookup_and_xgp_r3_recipe():
    from har.run_xgp_pilot4 import R3_FT_START, r3_config, r3_ft_recipe
    dec = {"decision": {"no_finetune": {"final": "xgp", "config_by_level": {"50pct": {"beta": 0.5, "r": None},
                                                                             "70pct": {"beta": 0.1, "r": None},
                                                                             "90pct": {"beta": 0.2, "r": None}}},
                        "finetune": {"final": "xgp_ft", "config_by_level": {"70pct": {"beta": 0.1, "r": 0.25},
                                                                            "90pct": {"beta": 0.1, "r": 0.5}}}}}
    assert r3_config(dec, "no_finetune", 0.8)["beta"] == 0.2         # 70 and 90 equidistant -> higher
    assert r3_config(dec, "no_finetune", 0.95)["round3_level"] == "90pct"
    assert r3_ft_recipe(dec, 0.9) == {"start": "xgp", "recipe": "kdaug", "r": 0.5, "dro": True,
                                      "source": "round 3 finetune final xgp_ft at 90pct"}
    dec["decision"]["finetune"]["final"] = "xgp_ft_no_aug"
    assert r3_ft_recipe(dec, 0.9)["recipe"] == "kd" and r3_ft_recipe(dec, 0.9)["r"] is None
    dec["decision"]["finetune"] = {"final": None, "reason": "x"}
    assert r3_ft_recipe(dec, 0.9) is None and r3_config(dec, "finetune", 0.9) is None
    from har.xgp3 import COMPONENTS
    assert set(R3_FT_START) == set(COMPONENTS["finetune"])


def test_runner_never_loads_har70_and_never_writes_round3():
    import har.run_xgp_pilot4 as m
    src = inspect.getsource(m)
    assert "load_test_windows" not in src and "load_har70" not in src.lower()
    assert m.out_dir(False).name == "round4" and m.out_dir(True).name == "round4"
    assert m.out_dir(False).parent.name == "har-xgp-pilot" and m.ckpt_dir(False).name == "har-xgp-pilot4"
    writes = [ln for ln in src.splitlines() if "write_csv(" in ln or "write_json(" in ln or "torch.save(" in ln]
    assert writes and not any("P3." in ln or "XGP_PILOT3" in ln for ln in writes)


def test_round4_uses_no_synthetic_older_like_data_outside_xgp_r3():
    import har.run_xgp_pilot4 as m
    import har.xgp4 as x
    assert "older" not in inspect.getsource(x).replace("no synthetic (older-like) data", "")
    src = inspect.getsource(m).replace(inspect.getsource(m.Context4.r3_aug_set), "")
    code = [ln for ln in src.splitlines() if ln.strip() and not ln.strip().startswith(("#", '"', "'"))]
    hits = [ln for ln in code if "older_like" in ln and "gkind" not in ln]
    assert not hits, hits
    assert not issubclass(m.Context4, m.P3.Context)
    assert "draw_older_like" not in inspect.getsource(m.calibration_table)
    assert m.KD["dro"] is False and m.KD_DRO["dro"] is True
