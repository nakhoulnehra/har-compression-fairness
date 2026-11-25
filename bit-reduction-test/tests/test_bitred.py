"""Tests for the bit-reduction test (quantizer, sizes, knapsack, sensitivities, rules, data isolation)."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

import bitred  # noqa: F401  (adds the repo root to sys.path)
from bitred.allocate import BIT_OPTIONS, allocate, mckp_min, objective
from bitred.quant import (fixed_bits, mean_bits, quant_tensors, quantize_model, quantize_weight, size_bits,
                          tensor_cost_bits, tensor_info, uniform_bits)
from bitred.rules import RULES, comparators, compare, decide, headroom
from bitred.sensitivity import hawq_sensitivity, hutchinson_traces, kl_from_logits, kl_sensitivity, lci_pooled
from har.models import build_model, resolve_config

MEAN = np.array([-0.9, 0.0, -0.2, -0.5, 0.0, 0.4], np.float32)
STD = np.array([0.3, 0.2, 0.3, 0.5, 0.3, 0.6], np.float32)


def tinyhar():
    torch.manual_seed(0)
    return build_model("tinyhar", resolve_config("tinyhar", 1.0), MEAN, STD).eval()


# ------------------------------------------------------------------------------ quantizer

@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_per_channel_levels_and_extremes(bits):
    torch.manual_seed(1)
    w = torch.randn(6, 4, 5)
    q = quantize_weight(w, bits, "channel")
    qmax = 2 ** (bits - 1) - 1
    W, Q = w.reshape(6, -1), q.reshape(6, -1)
    for r in range(6):
        s = W[r].abs().max() / qmax
        levels = torch.round(Q[r] / s)
        assert torch.allclose(Q[r], levels * s, atol=1e-6)
        assert levels.abs().max() <= qmax
        assert torch.isclose(Q[r].abs().max(), W[r].abs().max(), atol=1e-6)   # the row max is a level
    assert ((q - w).abs() <= (W.abs().amax(1) / qmax / 2 + 1e-6).reshape(6, 1, 1)).all()


def test_two_bit_is_ternary_and_zeros_stay_zero():
    w = torch.tensor([[0.0, 0.1, -0.9, 0.5], [0.0, 0.0, 0.0, 0.0]])
    q = quantize_weight(w, 2, "channel")
    s = 0.9
    assert set(torch.round(q[0] / s).tolist()) <= {-1.0, 0.0, 1.0}
    assert q[0, 0] == 0 and torch.equal(q[1], torch.zeros(4))
    with pytest.raises(ValueError):
        quantize_weight(w, 1)


def test_per_tensor_uses_one_scale():
    w = torch.tensor([[1.0, 0.2], [0.1, 0.05]])
    q = quantize_weight(w, 3, "tensor")
    s = 1.0 / 3
    assert torch.allclose(q, torch.round(w / s).clamp(-3, 3) * s)


def test_quantize_model_touches_only_named_tensors():
    net = tinyhar()
    names = [n for n, _, _ in quant_tensors(net)]
    assert len(names) == 13 and "backbone.head.weight" in names
    q = quantize_model(net, {names[0]: 2, names[5]: 32}, "channel")
    p0, pq = dict(net.named_parameters()), dict(q.named_parameters())
    for n in p0:
        if n == names[0]:
            assert not torch.equal(p0[n], pq[n])
        else:
            assert torch.equal(p0[n], pq[n]), n
    with pytest.raises(KeyError):
        quantize_model(net, {"backbone.convs.1.bias": 4})


def test_sizes_tinyhar():
    net = tinyhar()
    info = tensor_info(net)
    n_q = sum(v["numel"] for v in info.values())
    rows = sum(v["rows"] for v in info.values())
    n_all = sum(p.numel() for p in net.parameters())
    assert (n_q, n_all, rows) == (53116, 54273, 820)
    assert fixed_bits(net) == (n_all - n_q) * 32
    for b in (2, 3, 4, 8):
        assert size_bits(net, uniform_bits(net, b), "channel") == (n_all - n_q) * 32 + n_q * b + rows * 32
        assert size_bits(net, uniform_bits(net, b), "tensor") == (n_all - n_q) * 32 + n_q * b + 13 * 32
    assert size_bits(net, {}, "channel") == n_all * 32
    # per-tensor 4-bit = the plan's ptq4 nominal size (30.455 KB) plus 13 scales
    assert size_bits(net, uniform_bits(net, 4), "tensor") / 8 / 1024 == pytest.approx(30.455078125 + 13 * 4 / 1024)
    assert mean_bits(net, uniform_bits(net, 3)) == pytest.approx(3.0)
    assert tensor_cost_bits(10, 2, 32) == 320


# ------------------------------------------------------------------------------ knapsack

def brute(costs, values, budget):
    best = None
    for combo in itertools.product(*[range(len(c)) for c in costs]):
        c = sum(costs[g][j] for g, j in enumerate(combo))
        if c > budget:
            continue
        v = sum(values[g][j] for g, j in enumerate(combo))
        if best is None or v < best[0] - 1e-12:
            best = (v, combo)
    return best


def test_mckp_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(200):
        G = rng.integers(1, 5)
        costs = [list(rng.integers(0, 20, size=rng.integers(1, 4)) * 4) for _ in range(G)]
        values = [list(rng.random(len(c))) for c in costs]
        budget = int(rng.integers(0, 120))
        got = mckp_min(costs, values, budget)
        ref = brute(costs, values, budget)
        if ref is None:
            assert got is None
        else:
            assert got is not None
            assert sum(costs[g][j] for g, j in enumerate(got)) <= budget
            assert sum(values[g][j] for g, j in enumerate(got)) == pytest.approx(ref[0])


def test_mckp_infeasible_and_validation():
    assert mckp_min([[5, 6]], [[0.0, 0.0]], 4) is None
    with pytest.raises(ValueError):
        mckp_min([[1]], [[0.0, 1.0]], 3)


def test_allocate_at_uniform_budget_not_worse_than_uniform():
    net = tinyhar()
    names = [n for n, _, _ in quant_tensors(net)]
    rng = np.random.default_rng(3)
    for b in (2, 3):
        budget = size_bits(net, uniform_bits(net, b), "channel")
        # random monotone sensitivities (fewer bits -> larger drop)
        om = {n: dict(zip(BIT_OPTIONS, sorted(rng.random(4), reverse=True))) for n in names}
        bits = allocate(net, om, budget)
        assert size_bits(net, bits, "channel") <= budget
        assert objective(om, bits) <= objective(om, uniform_bits(net, b)) + 1e-12
    # zero sensitivity everywhere -> any feasible choice; all-8 is infeasible at the 2-bit budget
    assert allocate(net, {n: {b: 0.0 for b in BIT_OPTIONS} for n in names},
                    size_bits(net, uniform_bits(net, 2), "channel")) is not None
    assert allocate(net, {n: {b: 0.0 for b in BIT_OPTIONS} for n in names}, fixed_bits(net)) is None


def test_allocate_spends_bits_on_the_sensitive_tensor():
    net = tinyhar()
    names = [n for n, _, _ in quant_tensors(net)]
    om = {n: {2: 0.01, 3: 0.005, 4: 0.002, 8: 0.0} for n in names}
    om["backbone.head.weight"] = {2: 5.0, 3: 2.0, 4: 1.0, 8: 0.0}
    b2, b3 = (size_bits(net, uniform_bits(net, b), "channel") for b in (2, 3))
    # the uniform 2-bit budget leaves no room (2 bits is the minimum): all-2 is the only feasible choice
    assert set(allocate(net, om, b2).values()) == {2}
    bits = allocate(net, om, (b2 + b3) // 2)
    assert bits["backbone.head.weight"] == 8


# ------------------------------------------------------------------------------ sensitivities

def test_hutchinson_on_linear_least_squares():
    """loss = mean_i 0.5 ||W x_i - y_i||^2 -> H_W = I_out (x) mean x x^T, Tr = out * mean ||x||^2."""
    torch.manual_seed(0)
    lin = nn.Linear(5, 3, bias=False)
    X, Y = torch.randn(64, 5), torch.randn(64, 3)
    loss = lambda m, x, y: 0.5 * ((m(x) - y) ** 2).sum(1).mean()   # noqa: E731
    tr = hutchinson_traces(lin, loss, X, Y, ["weight"], n_iter=300, seed=0, batch_size=20)
    exact = 3 * float((X ** 2).sum(1).mean())
    assert tr["weight"]["trace"] == pytest.approx(exact, rel=0.08)


def test_kl_and_hawq_sensitivities_are_monotone_in_expectation():
    net = tinyhar()
    X = torch.randn(32, 6, 128) * 0.3
    kl = kl_sensitivity(net, X, (2, 8))
    assert all(v[2] >= 0 and v[8] >= 0 for v in kl.values())
    assert sum(v[2] for v in kl.values()) > sum(v[8] for v in kl.values())
    z = torch.randn(10, 7)
    assert kl_from_logits(z, z) == pytest.approx(0.0, abs=1e-7)
    traces = {n: {"trace": 1.0} for n, _, _ in quant_tensors(net)}
    hq = hawq_sensitivity(net, traces, (2, 3, 4, 8))
    for v in hq.values():
        assert v[2] >= v[3] >= v[4] >= v[8] >= 0
    neg = hawq_sensitivity(net, {n: {"trace": -1.0} for n in traces}, (2, 8))
    assert all(v[2] == 0.0 and v[8] == 0.0 for v in neg.values())     # negative trace clipped


def test_hutchinson_runs_through_tinyhar():
    net = tinyhar()
    X, y = torch.randn(16, 6, 128) * 0.3, torch.randint(0, 7, (16,))
    names = [n for n, _, _ in quant_tensors(net)]
    from bitred.sensitivity import ce_loss
    tr = hutchinson_traces(net, ce_loss, X, y, names, n_iter=2, seed=0)
    assert set(tr) == set(names) and all(np.isfinite(v["trace"]) for v in tr.values())


def test_lci_pooled_identity():
    A = np.random.default_rng(0).normal(size=(12, 6, 128))
    assert lci_pooled(A, A)["lci"] == pytest.approx(1.0)


# ------------------------------------------------------------------------------ rules

def per_fold_rows(spec: dict) -> pd.DataFrame:
    rows = []
    for mid, vals in spec.items():
        for k, (lci, f1, w) in enumerate(vals):
            rows.append({"model_id": mid, "fold": k, "lci": lci, "f1": f1, "worst_subject_lci": w})
    return pd.DataFrame(rows)


def test_compare_rule():
    d = per_fold_rows({"a": [(0.80, 0.70, 0.75)] * 4 + [(0.70, 0.70, 0.65)],
                       "b": [(0.75, 0.70, 0.70)] * 5})
    r = compare(d, "a", "b")
    assert r["n_folds_needed"] == 4 and r["n_folds_higher"] == 4
    assert r["d_lci_mean"] == pytest.approx(0.03) and r["beats"]
    d1 = per_fold_rows({"a": [(0.80, 0.70, 0.75)] * 3 + [(0.70, 0.70, 0.65)] * 2,
                        "b": [(0.75, 0.70, 0.70)] * 5})
    assert compare(d1, "a", "b")["n_folds_higher"] == 3 and not compare(d1, "a", "b")["beats"]   # 3 < 4 folds
    d2 = per_fold_rows({"a": [(0.80, 0.68, 0.75)] * 5, "b": [(0.75, 0.70, 0.70)] * 5})
    assert not compare(d2, "a", "b")["beats"]          # F1 more than 1 pt lower
    d3 = per_fold_rows({"a": [(0.75, 0.70, 0.70)] * 5, "b": [(0.75, 0.70, 0.70)] * 5})
    assert not compare(d3, "a", "b")["beats"]          # identical allocation is not a win


def test_decide_requires_headroom_and_all_comparators():
    budgets = ("3bit", "2.5bit", "2.25bit")
    base = {"uniform-channel-4bit": [(0.90, 0.80, 0.85)] * 5,
            "uniform-channel-3bit": [(0.80, 0.79, 0.75)] * 5,    # headroom at 3 bits (0.10)
            "uniform-channel-2bit": [(0.88, 0.70, 0.80)] * 5}    # no headroom at 2 bits (0.02)
    others = {f"alloc-{m}-{B}": [(0.80, 0.79, 0.75)] * 5 for m in ("zeroq_kl", "hawq") for B in budgets}
    others["alloc-hawq-3bit"] = [(0.84, 0.79, 0.80)] * 5          # XGQ's gain over HAWQ at 3bit is only 0.01
    ours = {f"alloc-xgq-{B}": [(0.85, 0.79, 0.80)] * 5 for B in budgets}
    dec, hr, comp = decide(per_fold_rows({**base, **others, **ours}))
    assert dec["headroom_by_bits"] == {3: True, 2: False}
    assert not dec["by_budget"]["3bit"]["supported"] and dec["by_budget"]["3bit"]["not_beaten"] == ["alloc-hawq-3bit"]
    assert dec["by_budget"]["2.5bit"]["supported"]                 # beats both, headroom at 3 bits
    b225 = dec["by_budget"]["2.25bit"]
    assert b225["beats_all_comparators"] and not b225["supported"]  # no headroom at 2 bits
    assert dec["contribution_supported"]
    assert comparators("2.5bit") == ["alloc-zeroq_kl-2.5bit", "alloc-hawq-2.5bit"]
    assert comparators("3bit")[-1] == "uniform-channel-3bit"
    assert set(RULES) == {"R1_headroom", "R2_beats", "R3_claim", "R4_no_tuning"}


# ------------------------------------------------------------------------------ isolation

BANNED_TEST_DATA = ("load_test_windows", "har70plus_test", "test_older", "explain_sets")


def test_har70plus_never_loaded():
    """No bitred module loads HAR70+ or the HARTH test subjects, except the final test's test phase
    (run_final_test.py, pre-registered 2026-11-25), which is checked separately below."""
    src = "\n".join(p.read_text(encoding="utf-8") for p in (Path(bitred.__file__).parent).glob("*.py")
                    if p.name != "run_final_test.py")
    for banned in BANNED_TEST_DATA:
        assert banned not in src.replace("fold.test_older", ""), banned


def test_final_test_validation_phase_never_loads_test_data():
    import inspect
    from bitred import run_final_test as F
    for fn in (F.run_validation_fold, F.analyze_validation, F.fold_plan):     # get_start is nested in the first
        src = inspect.getsource(fn)
        for banned in BANNED_TEST_DATA:
            assert banned not in src, (fn.__name__, banned)
    main = inspect.getsource(F.main)
    # test windows are loaded only after the validation branch has returned
    assert main.index("run_validation_fold") < main.index("load_test_windows")
    assert "check_gate(out, k, args.machine_name)" in inspect.getsource(F.run_test_fold)


# ------------------------------------------------------------------------------ 2-bit QAT

def synthetic_windows(n: int, seed: int = 0):
    from har.data import Windows
    rng = np.random.default_rng(seed)
    return Windows(X=(rng.normal(size=(n, 6, 128)) * 0.3).astype(np.float32), y=np.arange(n) % 7,
                   subject=np.array(["S006"] * n), dataset="harth", group=np.array(["general"] * n),
                   walking_aid=np.array(["not_applicable"] * n), majority_share=np.ones(n, np.float32),
                   start_time=np.zeros(n, "datetime64[ms]"), start_row=np.arange(n))


def test_qat_returns_quantized_model_and_trains_masters():
    from bitred.qat import qat_kd_finetune
    net = tinyhar()
    ref = {n: p.detach().clone() for n, p in net.named_parameters()}
    tr, va = synthetic_windows(64), synthetic_windows(21, 1)
    zt = torch.randn(64, 7)
    bits = uniform_bits(net, 2)
    q, hist = qat_kd_finetune(net, tr, va, zt, bits, epochs=2, lr=1e-3, batch_size=16, log=lambda s: None)
    assert len(hist) == 2 and all(np.isfinite(h["train_loss"]) for h in hist)
    # masters moved and are not themselves quantized; the returned model = quantize(masters)
    head = dict(net.named_parameters())["backbone.head.weight"]
    assert not torch.equal(head, ref["backbone.head.weight"])
    qd = dict(q.named_parameters())
    for n, b in bits.items():
        m = dict(net.named_parameters())[n].detach()
        assert torch.equal(qd[n].detach(), quantize_weight(m, b, "channel")), n
        assert len(torch.unique(qd[n].detach().reshape(m.shape[0], -1)[0])) <= 3   # ternary rows
    # non-quantized parameters are the trained masters
    assert torch.equal(qd["backbone.head.bias"], dict(net.named_parameters())["backbone.head.bias"])


def test_qat_gradient_is_taken_at_the_quantized_weights():
    """One SGD-like step with Adam replaced by a check: the gradient used equals the gradient of the quantized model."""
    from bitred.qat import qat_kd_finetune
    torch.manual_seed(0)
    net = tinyhar()
    tr, va = synthetic_windows(16), synthetic_windows(7, 1)
    zt = torch.randn(16, 7)
    bits = uniform_bits(net, 2)
    # expected gradient: same batch order (seed 0, one batch of 16), on the quantized model in train mode
    from har.compression import kd_loss
    from har.train import class_weights, set_seed
    set_seed(0)
    qm = quantize_model(net, bits)
    qm.train()
    perm = torch.randperm(16, generator=torch.Generator().manual_seed(0))
    ce = nn.CrossEntropyLoss(weight=class_weights(tr.y))
    loss = kd_loss(qm(torch.from_numpy(tr.X)[perm]), zt[perm], torch.from_numpy(tr.y)[perm], ce)
    loss.backward()
    g_exp = dict(qm.named_parameters())["backbone.head.weight"].grad.clone()
    # QAT with lr tiny so Adam's first step = -lr * sign(g) (bias-corrected): compare signs
    before = dict(net.named_parameters())["backbone.head.weight"].detach().clone()
    qat_kd_finetune(net, tr, va, zt, bits, epochs=1, lr=1e-6, batch_size=16, log=lambda s: None)
    step = dict(net.named_parameters())["backbone.head.weight"].detach() - before
    nz = g_exp.abs() > 1e-6
    assert torch.equal(torch.sign(step[nz]), -torch.sign(g_exp[nz]))


# ------------------------------------------------------------------------------ ultra-small gap test

def l1_pruned(net, level):
    from har.compression import l1_scores, prune_by_score
    return prune_by_score(net, l1_scores(net), level)


def test_pruned_size_record_by_hand():
    from bitred.quant import pruned_size_record
    from har.xgp4 import uniform_keep_counts
    net = tinyhar()
    info = tensor_info(net)
    rows = sum(v["rows"] for v in info.values())
    for b, g in ((2, "tensor"), (4, "channel"), (3, "channel")):
        assert pruned_size_record(net, uniform_bits(net, b), {}, g)["size_kb"] == pytest.approx(
            size_bits(net, uniform_bits(net, b), g) / 8 / 1024)
    assert pruned_size_record(net, uniform_bits(net, 2), {}, "tensor")["size_kb"] == pytest.approx(17.5, abs=0.05)
    pr = l1_pruned(net, 0.9)
    masks = pr.prune_masks
    kept = {n: int(m.sum()) for n, m in masks.items()}
    pos = sum(info[n]["numel"] for n in masks)
    n_kept = sum(kept.get(n, v["numel"]) for n, v in info.items())
    s = pruned_size_record(net, uniform_bits(net, 4), masks, "channel")
    assert s["n_kept_quantizable"] == n_kept
    assert s["size_kb"] == pytest.approx((n_kept * 4 + rows * 32 + pos + fixed_bits(net)) / 8 / 1024)
    assert s["size_kb_bitmask"] == pytest.approx(pos / 8 / 1024) and s["size_kb_without_bitmask"] < s["size_kb"]
    # fp32 pruned (no quantization): kept values at 32 bits, no scales, plus the bitmask
    f = pruned_size_record(net, {}, masks)
    assert f["size_kb"] == pytest.approx((n_kept * 32 + pos + fixed_bits(net)) / 8 / 1024) and f["size_kb_scales"] == 0
    # the README sizes: the uniform total is what LAMP keeps, so the size does not depend on the split
    for level, kb in ((0.9, 16.9), (0.95, 15.6)):
        m = l1_pruned(net, level).prune_masks
        assert sum(int(x.sum()) for x in m.values()) == sum(uniform_keep_counts(net, level).values())
        assert pruned_size_record(net, uniform_bits(net, 4), m, "channel")["size_kb"] == pytest.approx(kb, abs=0.05)
    with pytest.raises(KeyError):
        pruned_size_record(net, {"backbone.gamma": 4}, {})


def test_qat_keeps_masks_fixed():
    from bitred.qat import qat_kd_finetune
    net = l1_pruned(tinyhar(), 0.9)
    masks = {n: m.clone() for n, m in net.prune_masks.items()}
    ref = {n: p.detach().clone() for n, p in net.named_parameters()}
    tr, va = synthetic_windows(64), synthetic_windows(21, 1)
    bits = uniform_bits(net, 4)
    q, _ = qat_kd_finetune(net, tr, va, torch.randn(64, 7), bits, epochs=2, lr=1e-2, batch_size=16, log=lambda s: None)
    P, Q = dict(net.named_parameters()), dict(q.named_parameters())
    for n, m in masks.items():
        assert torch.all(P[n].detach()[~m] == 0) and torch.all(Q[n].detach()[~m] == 0), n
        assert torch.equal(q.prune_masks[n], m)
    moved = [n for n, m in masks.items() if not torch.equal(P[n].detach()[m], ref[n][m])]
    assert len(moved) == len(masks)                                    # kept weights are trained
    assert not any(n.endswith("head.weight") for n in q.prune_masks)    # head unpruned (4-bit may round some to 0)
    for n, b in bits.items():
        assert torch.equal(Q[n].detach(), quantize_weight(P[n].detach(), b, "channel")), n


def test_lamp_recon_start_and_settings():
    from bitred.run_qat_small import SETTINGS, check_masks, lamp_recon_start
    from har.compression import head_unpruned
    from har.xgp4 import uniform_keep_counts
    net0 = tinyhar()
    before = {n: p.detach().clone() for n, p in net0.named_parameters()}
    X = torch.from_numpy(synthetic_windows(48).X)
    net, keep, unrecon, _ = lamp_recon_start(net0, 0.9, X, log=lambda s: None)
    assert all(torch.equal(p, before[n]) for n, p in net0.named_parameters())       # original untouched
    assert sum(keep.values()) == sum(uniform_keep_counts(net0, 0.9).values())
    uni = uniform_keep_counts(net0, 0.9)
    assert all(keep[n] >= uni[n] for n in unrecon)                                  # protected tensors
    assert head_unpruned(net)
    check_masks(net, net.prune_masks, keep)
    q = quantize_model(net, uniform_bits(net, 4))
    check_masks(q, net.prune_masks, keep)
    assert [s["setting"] for s in SETTINGS] == ["tensor-2bit", "lamp_recon90-channel-4bit",
                                                "lamp_recon95-channel-4bit"]


def test_regime_runner_settings_and_rule():
    import bitred.run_qat_regimes as G
    from bitred.run_qat import GAP, decide
    assert [s["setting"] for s in G.SETTINGS] == ["channel-2bit", "tensor-2bit", "lamp_recon90-channel-4bit",
                                                  "lamp_recon95-channel-4bit"]
    assert G.REGIMES == ("kd5", "calibration") and G.QAT["lrs"] == (1e-4, 1e-3) and G.QAT["epochs"] == 20
    assert GAP == {"f1_recovered_tol": 0.03, "lci_gap_threshold": 0.85}
    rows = [{"model_id": "a", "f1": 0.79, "lci": 0.85, "worst_subject_lci": 0.8},
            {"model_id": "b", "f1": 0.70, "lci": 0.95, "worst_subject_lci": 0.9}]
    assert decide(rows, 0.818)["verdict"].startswith("gap")            # b does not recover, a is at the threshold
    assert decide(rows, 0.83)["verdict"].startswith("inconclusive")
    rows[0]["lci"] = 0.851
    assert decide(rows, 0.818)["verdict"].startswith("no gap")


def test_steps_per_epoch():
    from bitred.qat import steps_per_epoch
    assert steps_per_epoch(51451, 256) == 201 and 20 * steps_per_epoch(51451, 256) == 4020    # fold 0 full data
    assert steps_per_epoch(2583, 256) == 11 and steps_per_epoch(978, 256) == 4               # kd5, calibration
    assert steps_per_epoch(257, 256) == 1 and steps_per_epoch(258, 256) == 2                  # last batch of 1 skipped


def test_qat_step_mode_matches_epoch_mode_and_stops_mid_epoch():
    from bitred.qat import qat_kd_finetune
    tr, va = synthetic_windows(40), synthetic_windows(14, 1)
    zt = torch.randn(40, 7)
    bits = uniform_bits(tinyhar(), 2)
    kw = dict(lr=1e-3, batch_size=16, log=lambda s: None)          # 40 windows -> 3 steps per epoch
    qa, ha = qat_kd_finetune(tinyhar(), tr, va, zt, bits, epochs=2, **kw)
    qb, hb = qat_kd_finetune(tinyhar(), tr, va, zt, bits, steps=6, eval_every_steps=3, **kw)
    assert all(torch.equal(a, b) for a, b in zip(qa.state_dict().values(), qb.state_dict().values()))
    assert [h["step"] for h in ha] == [h["step"] for h in hb] == [3, 6]
    assert [h["val_f1_quantized"] for h in ha] == [h["val_f1_quantized"] for h in hb]
    # 7 steps = 2 epochs + 1 step of the third; validated at 3, 6 and the last step
    _, hc = qat_kd_finetune(tinyhar(), tr, va, zt, bits, steps=7, eval_every_steps=3, **kw)
    assert [h["step"] for h in hc] == [3, 6, 7] and hc[-1]["epoch"] == pytest.approx(7 / 3)
    # masks stay fixed over many epochs on a small set
    net = l1_pruned(tinyhar(), 0.9)
    q, _ = qat_kd_finetune(net, tr, va, zt, uniform_bits(net, 4), steps=30, eval_every_steps=10, **kw)
    assert all(torch.all(dict(q.named_parameters())[n].detach()[~m] == 0) for n, m in net.prune_masks.items())


# ------------------------------------------------------------------------------ final comparison (XGP vs LAMP)

def _compare_frame(d_lci=0.03, d_f1=0.0, d_worst=0.0, lamp_f1=(0.70, 0.75), n_folds=5, lower_fold=None):
    import bitred.run_xgp_compare as X
    rows = []
    for k in range(n_folds):
        for level in X.LEVELS:
            for regime in X.REGIMES:
                for lr, f1 in zip(X.QAT["lrs"], lamp_f1):
                    base = {"fold": k, "level": level, "regime": regime, "lr": lr, "role": "qat", "size_kb": 16.9}
                    d = -0.01 if k == lower_fold else d_lci
                    rows.append({**base, "method": "lamp_recon", "f1": f1, "lci": 0.80, "worst_subject_lci": 0.75})
                    rows.append({**base, "method": "xgp_xb", "f1": f1 + d_f1, "lci": 0.80 + d,
                                 "worst_subject_lci": 0.75 + d_worst})
    return pd.DataFrame(rows)


def test_primary_lr_is_chosen_on_lamp_f1_only():
    import bitred.run_xgp_compare as X
    p = X.primary_lrs(_compare_frame(lamp_f1=(0.70, 0.75)))
    assert p.primary_lr.tolist() == [1e-3] * 3
    assert X.primary_lrs(_compare_frame(lamp_f1=(0.75, 0.70))).primary_lr.tolist() == [1e-4] * 3
    assert X.primary_lrs(_compare_frame(lamp_f1=(0.72, 0.72))).primary_lr.tolist() == [1e-4] * 3   # tie
    # XGP's F1 does not move the choice
    m = _compare_frame(lamp_f1=(0.75, 0.70), d_f1=0.0)
    m.loc[(m.method == "xgp_xb") & np.isclose(m.lr, 1e-3), "f1"] = 0.99
    assert X.primary_lrs(m).primary_lr.tolist() == [1e-4] * 3


def test_r2_verdicts():
    import bitred.run_xgp_compare as X
    def verdicts(**kw):
        m = _compare_frame(**kw)
        c = X.comparisons(m, X.primary_lrs(m))
        return c[c.primary].verdict.unique().tolist(), c
    assert verdicts(d_lci=0.03)[0] == ["XGP beats LAMP"]
    assert verdicts(d_lci=0.015)[0] == ["XGP does not beat LAMP"]                      # (a) gain < 0.02
    assert verdicts(d_lci=0.03, lower_fold=0)[0] == ["XGP beats LAMP"]                 # 4 of 5 higher, mean 0.022
    v, c = verdicts(d_lci=0.03, d_f1=-0.011)
    assert v == ["XGP does not beat LAMP"] and not c.c_f1.any()                        # (c)
    v, c = verdicts(d_lci=0.03, d_worst=-0.011)
    assert v == ["XGP does not beat LAMP"] and not c.d_worst.any()                     # (d)
    v, c = verdicts(d_lci=0.03, n_folds=4)
    assert v == ["incomplete"]
    assert len(c) == len(X.LEVELS) * len(X.REGIMES) * len(X.QAT["lrs"]) and c.primary.sum() == 6


def test_r2_needs_four_of_five_higher():
    import bitred.run_xgp_compare as X
    m = _compare_frame(d_lci=0.06)
    m.loc[(m.method == "xgp_xb") & m.fold.isin([0, 1]), "lci"] = 0.79                   # 3 of 5 higher, mean 0.032
    c = X.comparisons(m, X.primary_lrs(m))
    assert (c[c.primary].verdict == "XGP does not beat LAMP").all() and not c.b_folds.any()


def test_xgp_xb_start_uses_round4_functions():
    import bitred.run_xgp_compare as X
    from har.xgp4 import U_CELLS, uniform_keep_counts
    assert U_CELLS["xgp_xb"] == ("xb", "xgp") and U_CELLS["lamp_recon"] == ("lamp", "l1")
    net0 = tinyhar()
    w = synthetic_windows(28)

    class Cal:
        X = torch.from_numpy(w.X)
        t = torch.from_numpy(w.y.astype(np.int64))
        y = w.y
        g = np.array(["S006", "S008"] * 14)
        unrecon = []
    bg = torch.from_numpy(synthetic_windows(4, 3).X)
    xc = {**X.XGP, "m": 2, "K": 2}
    calls = []

    def shares_fn():
        from har.xgp4 import worst_group_shares
        calls.append(1)
        return worst_group_shares(net0, net0, Cal.X, Cal.t, Cal.y, Cal.g, bg, k=xc["top_k"], m=2, eg_seeds=(0, 1))
    before = {n: p.detach().clone() for n, p in net0.named_parameters()}
    nx, kx, sx, _ = X.build_start(net0, "xgp_xb", 0.9, Cal, bg, xc, shares_fn, log=lambda s: None)
    nl, kl, sl, _ = X.build_start(net0, "lamp_recon", 0.9, Cal, bg, xc, shares_fn, log=lambda s: None)
    assert calls == [1] and sl is None and abs(sum(sx.values()) - 1) < 1e-9
    total = sum(uniform_keep_counts(net0, 0.9).values())
    assert sum(kx.values()) == sum(kl.values()) == total and kx != kl                  # same size, other split
    assert all(torch.equal(p, before[n]) for n, p in net0.named_parameters())
    from bitred.quant import pruned_size_record
    sz = [pruned_size_record(net0, uniform_bits(net0, 4), n.prune_masks)["size_kb"] for n in (nx, nl)]
    assert sz[0] == sz[1]


# ------------------------------------------------------------------------------ final test on HAR70+ (run_final_test)

def test_final_test_plan_and_seeds():
    from bitred import run_final_test as F
    plan = F.fold_plan()
    ids = [e["model_id"] for e in plan]
    assert len(plan) == 62 and len(set(ids)) == 62
    roles = pd.Series([e["role"] for e in plan]).value_counts().to_dict()
    assert roles == {"reference": 6, "main": 24, "ablation": 24, "seed": 8}
    extra = [e for e in plan if e["seed"] != 0]
    assert {(e["method"], e["level"], e["regime"]) for e in extra} == {(m, 0.9, "kd5") for m in F.MAIN}
    assert {e["lr"] for e in plan} == {1e-4, 1e-3} and F.PRIMARY_LR == 1e-3
    # round 4's seed convention: only starts with XGP budgets or selection depend on the seed
    assert F.start_seed("lamp_recon", 2) == 0
    assert F.start_seed("xgp_xb", 2) == 2 and F.start_seed("xgp_lamp", 1) == 1 and F.start_seed("l1_xb_recon", 1) == 1


def test_final_test_reuses_compare_pipeline_unchanged():
    from bitred import run_final_test as F, run_xgp_compare as C
    assert F.build_start is C.build_start and F.Calibration is C.Calibration
    assert F.QAT is C.QAT and F.QAT["steps"] == 4020 and F.REGIMES == C.REGIMES and F.LEVELS == C.LEVELS
    assert F.model_id("xgp_xb", 0.9, "kd5", 1e-3, 0) == "qat-xgp_xb90-channel-4bit-kd5-lr0.001-seed0"
    import har.run_compression_grid as G
    assert F.G is G and F.TestStore.__mro__[1] is G.FoldStore


def test_wilcoxon_greater_direction():
    from bitred.run_final_test import wilcoxon_greater
    rng = np.random.default_rng(0)
    b = rng.uniform(0.6, 0.8, 40)
    up = wilcoxon_greater(b + 0.03 + rng.normal(0, 0.01, 40), b)
    down = wilcoxon_greater(b - 0.03, b)
    assert up["p_one_sided"] < 1e-6 and down["p_one_sided"] > 0.99
    assert abs(up["p_one_sided"] - up["p_two_sided"] / 2) < 1e-12
    assert wilcoxon_greater(b, b)["p_one_sided"] == 1.0


def test_paired_gap_difference():
    from bitred.run_final_test import paired_gap_difference
    g1 = np.r_[np.ones(18, bool), np.zeros(22, bool)]
    xb = np.where(g1, 0.5, 0.7) + np.random.default_rng(1).normal(0, 0.01, 40)   # B: gap -0.2
    xa = np.where(g1, 0.65, 0.7) + np.random.default_rng(2).normal(0, 0.01, 40)  # A: gap -0.05
    r = paired_gap_difference(xa, xb, g1, ~g1, absolute=True, n_boot=2000)
    assert abs(r["d"] - 0.15) < 0.02 and r["d_ci_low"] > 0
    s = paired_gap_difference(xa, xb, g1, ~g1, absolute=False, n_boot=2000)
    assert abs(s["d"] - 0.15) < 0.02          # gap_A - gap_B = -0.05 - (-0.2)
    rev = paired_gap_difference(xb, xa, g1, ~g1, absolute=True, n_boot=2000)
    assert rev["d_ci_high"] < 0


def _synthetic_pooled(delta_lci: float) -> pd.DataFrame:
    from har.data import WALKING_AID_INFERRED
    subjects = [(f"S{i:03d}", "general", "general") for i in range(22)] + [
        (str(s), "older", "walking_aid_inferred" if str(s) in WALKING_AID_INFERRED else "other_older")
        for s in range(501, 519)]
    rng = np.random.default_rng(3)
    rows = []
    for method in ("lamp_recon", "xgp_xb"):
        for seed in (0, 1, 2):
            for subj, grp, sub in subjects:
                base = 0.75 + rng.normal(0, 0.02)
                rows.append({"method": method, "lr": 1e-3, "regime": "kd5", "sparsity": 0.9, "seed": seed,
                             "subject": subj, "group": grp, "subgroup": sub,
                             "lci": base + (delta_lci if method == "xgp_xb" else 0), "non_stair_macro_f1": 0.8,
                             "aopc": 0.3 + (0.05 if grp == "older" and method == "lamp_recon" else 0),
                             "d_accuracy": -0.05 - (0.05 if sub == "walking_aid_inferred" and method == "lamp_recon" else 0)})
    return pd.DataFrame(rows)


def test_final_test_hypotheses_on_synthetic_subjects():
    from bitred.run_final_test import hypotheses
    h = hypotheses(_synthetic_pooled(0.03), 1e-3, n_boot=2000)
    assert h["complete"] and h["H1"]["n_subjects"] == 40 and h["H1"]["n_older"] == 18
    assert h["H1"]["supported"] and h["H1"]["p_one_sided"] < 0.05
    assert h["H2"]["supported"] and abs(h["H2"]["D2_abs_gap_reduction"] - 0.05) < 1e-9
    assert h["H3"]["n_walking_aid"] == 5 and h["H3"]["n_other_older"] == 13
    assert h["H3"]["supported_exploratory"] and abs(h["H3"]["D3"] - 0.05) < 1e-9
    assert not hypotheses(_synthetic_pooled(-0.03), 1e-3, n_boot=500)["H1"]["supported"]
    # a subject missing one seed is dropped, not averaged over fewer seeds
    d = _synthetic_pooled(0.03)
    d = d[~((d.subject == "S000") & (d.seed == 2) & (d.method == "xgp_xb"))]
    assert hypotheses(d, 1e-3, n_boot=500)["H1"]["n_subjects"] == 39


def test_final_test_gate_enforces_validation_first(tmp_path):
    import json
    from bitred.run_final_test import P, check_gate, sha256_file
    v = tmp_path / "validation"
    v.mkdir()
    f = v / f"{P}-fold0-m1-models.csv"
    f.write_text("model_id\noriginal\n")
    with pytest.raises(SystemExit, match="analyze-validation"):
        check_gate(tmp_path, 0, "m1")
    (v / f"{P}-validation-complete.json").write_text(json.dumps(
        {"complete": True, "folds": {"0": {"machine": "m1", "file": f.name, "sha256": sha256_file(f)}}}))
    assert check_gate(tmp_path, 0, "m1")["complete"]
    with pytest.raises(SystemExit, match="ran on m1"):
        check_gate(tmp_path, 0, "m2")
    with pytest.raises(SystemExit, match="not in the completed"):
        check_gate(tmp_path, 1, "m1")
    f.write_text("model_id\nchanged\n")
    with pytest.raises(SystemExit, match="changed"):
        check_gate(tmp_path, 0, "m1")
