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

def test_har70plus_never_loaded():
    src = "\n".join(p.read_text(encoding="utf-8") for p in (Path(bitred.__file__).parent).glob("*.py"))
    for banned in ("load_test_windows", "har70plus_test", "test_older", "explain_sets"):
        assert banned not in src.replace("fold.test_older", ""), banned


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
