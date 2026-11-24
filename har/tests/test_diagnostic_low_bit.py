"""Tests for the low-bit diagnostic (har/run_diagnostic_low_bit.py) and per-channel PTQ / low_bit_size.

    PTQ        per output row at most 2^b - 1 levels, the row max kept, zeros and pruned weights stay 0,
               `only` quantizes one tensor, 2 bits = ternary, the original model is untouched
    size       low_bit_size by hand: values at their bit width + one fp32 scale per row + bitmask
    plan       unique model IDs, one single-layer model per quantizable tensor, no fine-tuning anywhere
    LAMP       lamp_recon start on a random TinyHAR: uniform total kept, head unpruned, masks survive PTQ
    analysis   validation per fold (below tau_k, silent break), deltas vs original, layer sensitivity
"""

import numpy as np
import pandas as pd
import pytest
import torch

import har.run_diagnostic_low_bit as D
from har.compression import head_unpruned, l1_scores, low_bit_size, prunable_tensors, prune_by_score, ptq_per_channel, size_record
from har.tests.test_compression import fake_windows, tinyhar


# ------------------------------------------------------------------------------ PTQ

@pytest.mark.parametrize("bits", [4, 3, 2])
def test_per_channel_levels_and_row_max(bits):
    net = tinyhar()
    q, quant = ptq_per_channel(net, bits)
    names = [n for n, _, _ in prunable_tensors(net, include_head=True)]
    assert sorted(quant) == sorted(names) and set(quant.values()) == {bits}
    p0, p1 = dict(net.named_parameters()), dict(q.named_parameters())
    qmax = 2 ** (bits - 1) - 1
    for n in names:
        W0, W1 = p0[n].detach().reshape(p0[n].shape[0], -1), p1[n].detach().reshape(p1[n].shape[0], -1)
        for r0, r1 in zip(W0, W1):
            assert len(torch.unique(r1)) <= 2 * qmax + 1
            assert torch.isclose(r1.abs().max(), r0.abs().max(), rtol=1e-5)
            s = r0.abs().max() / qmax
            assert torch.all((r1 / s - torch.round(r1 / s)).abs() < 1e-3)
    for n, p in p0.items():           # biases and norms unchanged
        if n not in names:
            assert torch.equal(p, p1[n])
    assert not torch.equal(p0[names[0]], p1[names[0]])  # deepcopy, original untouched


def test_two_bit_is_ternary_and_only_one_tensor():
    net = tinyhar()
    t = "backbone.fusion.0.weight"
    q, quant = ptq_per_channel(net, 2, only=[t])
    assert quant == {t: 2}
    W = dict(q.named_parameters())[t].detach().reshape(56, -1)
    for r in W:
        assert set(torch.round(r / r.abs().max()).tolist()) <= {-1.0, 0.0, 1.0}
    for n, p in net.named_parameters():
        if n != t:
            assert torch.equal(p, dict(q.named_parameters())[n])
    with pytest.raises(KeyError):
        ptq_per_channel(net, 2, only=["backbone.gamma"])
    with pytest.raises(ValueError):
        ptq_per_channel(net, 1)


def test_pruned_weights_stay_zero_and_zero_row():
    net = prune_by_score(tinyhar(), l1_scores(tinyhar()), 0.9)
    with torch.no_grad():
        dict(net.named_parameters())["backbone.head.weight"][0].zero_()
    q, _ = ptq_per_channel(net, 4)
    P = dict(q.named_parameters())
    for n, m in q.prune_masks.items():
        assert torch.all(P[n][~m] == 0)
    assert torch.all(P["backbone.head.weight"][0] == 0) and torch.isfinite(P["backbone.head.weight"]).all()
    assert size_record(q)["n_nonzero"] <= size_record(net)["n_nonzero"]


# ------------------------------------------------------------------------------ size

def test_low_bit_size_by_hand():
    net = tinyhar()
    params = dict(net.named_parameters())
    n_all = sum(p.numel() for p in params.values())
    assert low_bit_size(net)["size_kb"] == pytest.approx(n_all * 4 / 1024)
    q, quant = ptq_per_channel(net, 4)
    qn = sum(params[n].numel() for n in quant)
    rows = sum(params[n].shape[0] for n in quant)
    s = low_bit_size(q, quant)
    assert s["size_kb"] == pytest.approx((qn * 4 + rows * 32 + (n_all - qn) * 32) / 8 / 1024)
    assert s["size_kb_bitmask"] == 0 and s["size_kb_with_bitmask"] == s["size_kb"]
    assert s["size_kb_dense_fp32"] == pytest.approx(n_all * 4 / 1024)
    # pruned + quantized: kept values only, plus one bit per position of each pruned tensor
    pr = prune_by_score(net, l1_scores(net), 0.9)
    q2, quant2 = ptq_per_channel(pr, 4)
    kept = {n: int(m.sum()) for n, m in pr.prune_masks.items()}
    pos = sum(params[n].numel() for n in pr.prune_masks)
    vals = sum(kept.get(n, p.numel()) * (4 if n in quant2 else 32) for n, p in params.items())
    s2 = low_bit_size(q2, quant2)
    assert s2["size_kb_values"] == pytest.approx(vals / 8 / 1024)
    assert s2["size_kb_with_bitmask"] == pytest.approx((vals + rows * 32 + pos) / 8 / 1024)
    assert s2["size_kb_with_bitmask"] < s["size_kb"]
    assert 33 < s["size_kb"] < 34            # 4-bit per channel with fp32 scales is just above 30 KB
    assert low_bit_size(*ptq_per_channel(net, 3))["size_kb"] < 30


# ------------------------------------------------------------------------------ plan

def test_plan():
    names = [n for n, _, _ in prunable_tensors(tinyhar(), include_head=True)]
    plan = D.diag_plan(names)
    ids = [p["model_id"] for p in plan]
    assert len(ids) == len(set(ids)) == 1 + 3 + 2 * 2 + len(names) == 21
    assert [p["bits"] for p in plan if p["family"] == "ptq_per_channel"] == [4, 3, 2]
    assert {p["tensor"] for p in plan if p["family"] == "layer_sensitivity"} == set(names)
    assert all(p["scenario"] in ("original", "no_finetune") for p in plan)
    assert len({tuple(p[c] for c in D.SETTING) for p in plan}) == len(plan)
    assert len(D.diag_plan(names, with_layers=False)) == 8


def test_lamp_recon_start_small(tmp_path):
    from har.xgp4 import uniform_keep_counts

    class Ctx:
        net0, dev = tinyhar(), "cpu"
    cal = torch.from_numpy(fake_windows(32).X)
    net, keep, _ = D.lamp_recon_start(Ctx, 0.9, cal, K=2, ckpt=tmp_path / "s.pt")
    assert sum(keep.values()) == sum(uniform_keep_counts(Ctx.net0, 0.9).values())
    assert head_unpruned(net) and all(int(m.sum()) == keep[n] for n, m in net.prune_masks.items())
    again, keep2, sec = D.lamp_recon_start(Ctx, 0.9, cal, K=2, ckpt=tmp_path / "s.pt")
    assert sec == 0.0 and keep2 == keep
    assert all(torch.equal(a, b) for a, b in zip(net.state_dict().values(), again.state_dict().values()))
    q, _ = ptq_per_channel(net, 4)
    P = dict(q.named_parameters())
    assert all(torch.all(P[n][~m] == 0) for n, m in q.prune_masks.items())


# ------------------------------------------------------------------------------ analysis

def _frames():
    rows, models = [], []
    for fold, tau in ((0, 0.6), (1, 0.65)):
        for method, family, level, lci, f1 in (("original", "original", "none", 1.0, 0.80),
                                               ("ptq_pc2", "ptq_per_channel", "2-bit", 0.5, 0.795),
                                               ("ptq_pc4", "ptq_per_channel", "4-bit", 0.9, 0.70)):
            for s, d in (("a", 0.0), ("b", 0.1), ("c", -0.1)):
                rows.append({"fold": fold, "method": method, "family": family, "scenario": "no_finetune",
                             "level": level, "subject": f"{fold}{s}", "group": "inner_val", "lci": lci + d,
                             "shap_corr": lci, "jaccard": lci, "tau_k": tau})
            models.append({"fold": fold, "model_id": method, "method": method, "family": family,
                           "scenario": "no_finetune", "level": level, "tensor": None, "size_kb": 10.0,
                           "size_kb_with_bitmask": 11.0, "n_nonzero": 1, "macro_f1_pooled7_inner_val": f1,
                           "accuracy_pooled_inner_val": f1})
    return pd.DataFrame(rows), pd.DataFrame(models)


def test_validation_tables():
    ps, models = _frames()
    v = D.validation_per_fold(ps, models)
    p2 = v[v.method == "ptq_pc2"]
    assert p2.lci_val.tolist() == pytest.approx([0.5, 0.5]) and p2.worst_subject_lci_val.tolist() == pytest.approx([0.4, 0.4])
    assert p2.lci_val_below_tau_k.all() and p2.silent_break_val.all()             # F1 drop 0.5 pt only
    p4 = v[v.method == "ptq_pc4"]
    assert not p4.lci_val_below_tau_k.any() and not p4.silent_break_val.any()
    assert p4.d_f1_val.tolist() == pytest.approx([-0.1, -0.1])
    # ptq_pc2 subjects 0.5 / 0.6 / 0.4: fold 0 (tau 0.6) 2 of 3 below, fold 1 (tau 0.65) all 3
    assert p2.share_subjects_below_tau_k.tolist() == pytest.approx([2 / 3, 1.0])
    s = D.validation_summary(v).set_index("method")
    assert s.loc["ptq_pc2", "n_folds_silent_break"] == 2 and s.loc["ptq_pc4", "n_folds_lci_val_below_tau_k"] == 0
    assert s.loc["original", "lci_val_mean"] == pytest.approx(1.0)


def test_add_deltas():
    pooled = pd.DataFrame({"method": ["original", "original", "x", "x"], "family": ["original", "original", "p", "p"],
                           "subject": ["a", "b", "a", "b"], "accuracy": [0.9, 0.8, 0.85, 0.6]})
    out = D.add_deltas(pooled, ("accuracy",))
    assert out.d_accuracy.tolist() == pytest.approx([0, 0, -0.05, -0.2])


def test_layer_sensitivity_table():
    vs = pd.DataFrame({"method": ["ptq_pc2", "layer2bit:a", "layer2bit:b"],
                       "family": ["ptq_per_channel", "layer_sensitivity", "layer_sensitivity"],
                       "tensor": [None, "a", "b"], "lci_val_mean": [0.5, 0.9, 0.6], "lci_val_sd": [0, 0, 0],
                       "d_f1_val_mean": [0, 0, 0]})
    head = pd.DataFrame({"method": ["ptq_pc2", "layer2bit:a", "layer2bit:b"], "lci_general": [0.6, 0.95, 0.7],
                         "lci_older": [0.4, 0.9, 0.5], "d_accuracy_general": [0, 0, 0], "d_accuracy_older": [0, 0, 0],
                         "lci_gap": [-0.2, -0.05, -0.2], "lci_gap_ci_low": [0, 0, 0], "lci_gap_ci_high": [0, 0, 0],
                         "lci_wa_gap": [0, 0, 0], "fa_lci": [0.4, 0.9, 0.5]})
    t = D.layer_sensitivity_table(vs, head, {"a": 100, "b": 300})
    assert t.tensor.tolist() == ["b", "a"] and t.rank_val.tolist() == [1, 2]
    assert t.set_index("tensor").loc["b", "share_of_full_2bit_val_drop"] == pytest.approx(0.8)
    assert t.set_index("tensor").loc["b", "lci_drop_older"] == pytest.approx(0.5)
    assert t.set_index("tensor").loc["a", "share_of_quantized_weights"] == pytest.approx(0.25)
