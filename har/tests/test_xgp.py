"""Tests for XGP, explanation-guided pruning (har/xgp.py; plan Sections 2.5 (b) and 2.10).

Keep-score (beta = 0 ranks like L1), the importance I(w) as |w * grad| of the dot-product alignment,
the reason for the dot product (the Pearson / cosine alignment has zero gradient at theta_0), L_cons,
the gate branch, the tuning selection rule, and matched n_nonzero for every method and ablation.
"""

import copy

import numpy as np
import pytest
import torch
import torch.nn as nn

from har.compression import (eg_attributions, l1_scores, prunable_tensors, prune_by_score, segment_attributions,
                             size_record, snip_scores)
from har.tests.test_compression import fake_windows, tinyhar, toy
from har.xgp import (ABLATIONS, METHODS, XGP, alignment_importance, cosine_sim, gate_branch, keep_scores,
                     lcons_from_attributions, lcons_loss, pearson_rows, pruning_scores, reference_windows,
                     select_config)


class Tanh7(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.fc1, self.head = nn.Linear(6 * 128, 16), nn.Linear(16, 7)

    def forward(self, x):
        return self.head(torch.tanh(self.fc1(x.flatten(1))))


def data(n=6, seed=0):
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.normal(size=(n, 6, 128)).astype(np.float32))


def test_constants_match_the_design():
    assert XGP["n_ref"] == 1000 and XGP["m_score"] == 8 and tuple(XGP["eg_seeds"]) == (0, 1)
    assert XGP["m_cons"] == 4 and XGP["betas"] == (0.3, 0.7) and XGP["gammas"] == (0.1, 1.0)
    assert XGP["tune_level"] == 0.7 and XGP["ablation_levels"] == (0.7, 0.9) and XGP["f1_tolerance"] == 0.01
    assert ABLATIONS == ("xgp_score_only", "l1_plus_lcons", "xgp_no_gate", "xgp")


# ------------------------------------------------------------------------------ keep-score

def test_keep_score_beta0_ranks_like_l1():
    m = tinyhar()
    rng = torch.Generator().manual_seed(0)
    imp = {n: torch.rand(getattr(mod, p).shape, generator=rng) for n, mod, p in prunable_tensors(m)}
    s0 = keep_scores(m, imp, beta=0.0)
    for level in (0.3, 0.5, 0.7, 0.9):
        a, b = prune_by_score(m, l1_scores(m), level), prune_by_score(m, s0, level)
        assert all(torch.equal(a.prune_masks[k], b.prune_masks[k]) for k in a.prune_masks)
    s1 = keep_scores(m, imp, beta=1.0)               # beta = 1 ranks like I(w)
    a, b = prune_by_score(m, imp, 0.5), prune_by_score(m, s1, 0.5)
    assert all(torch.equal(a.prune_masks[k], b.prune_masks[k]) for k in a.prune_masks)


def test_keep_score_formula_per_layer_means():
    m = toy()
    w = m.fc.weight.detach()
    I = torch.rand_like(w)
    s = keep_scores(m, {"fc.weight": I}, beta=0.3)["fc.weight"]
    ref = 0.7 * w.abs() / w.abs().mean() + 0.3 * I / I.mean()
    assert torch.allclose(s, ref)
    s_zero = keep_scores(m, {"fc.weight": torch.zeros_like(w)}, beta=0.3)["fc.weight"]   # all-zero I: no NaN
    assert torch.isfinite(s_zero).all() and torch.allclose(s_zero, 0.7 * w.abs() / w.abs().mean())
    with pytest.raises(ValueError):
        keep_scores(m, {"fc.weight": I}, beta=1.5)


# ------------------------------------------------------------------------------ importance I(w)

def test_importance_is_abs_w_times_dot_alignment_gradient():
    m = Tanh7()
    X, bg = data(5), data(10, seed=1)
    t = torch.tensor([0, 1, 2, 3, 4])
    I = alignment_importance(m, X, t, bg, m=4, eg_seeds=(3,), batch_size=2)
    assert set(I) == {"fc1.weight"}
    ref = copy.deepcopy(m).eval()
    g = torch.Generator().manual_seed(3)
    total = torch.zeros_like(ref.fc1.weight)
    for i in range(0, 5, 2):                         # same batches, same draws
        A = segment_attributions(eg_attributions(ref, X[i:i + 2], t[i:i + 2], bg, 4, create_graph=True, generator=g))
        (gw,) = torch.autograd.grad((A * A.detach()).sum(), ref.fc1.weight)
        total += gw
    assert torch.allclose(I["fc1.weight"], (ref.fc1.weight * total).abs(), rtol=1e-4, atol=1e-8)
    assert I["fc1.weight"].abs().sum() > 0
    assert all(p.grad is None for p in m.parameters())   # model left untouched


def test_importance_averages_over_eg_seeds_and_is_reproducible():
    m = Tanh7()
    X, bg, t = data(4), data(10, seed=1), torch.tensor([0, 1, 2, 3])
    a = alignment_importance(m, X, t, bg, m=4, eg_seeds=(0,), batch_size=4)
    b = alignment_importance(m, X, t, bg, m=4, eg_seeds=(1,), batch_size=4)
    ab = alignment_importance(m, X, t, bg, m=4, eg_seeds=(0, 1), batch_size=4)
    assert torch.allclose(ab["fc1.weight"], (a["fc1.weight"] + b["fc1.weight"]) / 2, rtol=1e-5, atol=1e-9)
    again = alignment_importance(m, X, t, bg, m=4, eg_seeds=(0, 1), batch_size=4)
    assert torch.equal(ab["fc1.weight"], again["fc1.weight"])


def test_pearson_and_cosine_alignment_have_zero_gradient_at_theta0():
    """Why I(w) uses the dot product: Pearson(A_theta, sg(A_theta0)) and cos(A_theta, sg(A_theta0)) are
    at their maximum (= 1) at theta_0, so their gradient there is zero and the Taylor score would be
    zero for every weight. The dot-product alignment has a non-zero gradient."""
    m = Tanh7().eval()
    X, bg, t = data(6), data(10, seed=1), torch.tensor([0, 1, 2, 3, 4, 5])
    A = segment_attributions(eg_attributions(m, X, t, bg, 4, create_graph=True,
                                             generator=torch.Generator().manual_seed(0)))
    A0 = A.detach()
    g_dot = torch.autograd.grad((A * A0).sum(), m.fc1.weight, retain_graph=True)[0]
    g_pear = torch.autograd.grad(pearson_rows(A, A0).sum(), m.fc1.weight, retain_graph=True)[0]
    cos_rows = (A * A0).sum(1) / (A.norm(dim=1) * A0.norm(dim=1))
    g_cos = torch.autograd.grad(cos_rows.sum(), m.fc1.weight)[0]
    scale = g_dot.abs().max()
    assert scale > 1e-6
    assert g_pear.abs().max() < 1e-4 * scale
    assert g_cos.abs().max() < 1e-4 * scale


def test_importance_on_tinyhar_has_every_prunable_tensor():
    m = tinyhar()
    X, bg = data(4), data(10, seed=1)
    with torch.no_grad():
        t = m(X).argmax(1)
    I = alignment_importance(m, X, t, bg, m=2, eg_seeds=(0, 1), batch_size=4)
    params = dict(m.named_parameters())
    assert set(I) == {n for n, _, _ in prunable_tensors(m)}
    for k, v in I.items():
        assert v.shape == params[k].shape and torch.isfinite(v).all() and (v >= 0).all(), k
    assert sum(float(v.sum()) for v in I.values()) > 0


def test_reference_windows_fixed_stratified_harth_only():
    w = fake_windows(140)
    a, b = reference_windows(w, n=70, seed=42), reference_windows(w, n=70, seed=42)
    assert np.array_equal(a, b) and len(a) == 70 and len(set(a.tolist())) == 70
    assert np.bincount(w.y[a], minlength=7).tolist() == [10] * 7       # class-stratified
    with pytest.raises(ValueError):
        reference_windows(fake_windows(140, dataset="har70plus"), n=70)


# ------------------------------------------------------------------------------ L_cons

def test_pearson_rows_matches_numpy_and_handles_zero_rows():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(5, 48)), rng.normal(size=(5, 48))
    got = pearson_rows(torch.tensor(a), torch.tensor(b)).numpy()
    ref = [np.corrcoef(x, y)[0, 1] for x, y in zip(a, b)]
    assert np.allclose(got, ref, atol=1e-6)
    z = torch.zeros(2, 48, requires_grad=True)
    r = pearson_rows(z, torch.tensor(b[:2], dtype=torch.float32))
    r.sum().backward()
    assert torch.isfinite(r).all() and torch.isfinite(z.grad).all() and torch.allclose(r, torch.zeros(2))
    assert torch.isfinite(cosine_sim(torch.zeros(48), torch.ones(48)))


def test_lcons_parts():
    rng = np.random.default_rng(1)
    A0 = torch.tensor(rng.normal(size=(8, 48)), dtype=torch.float32)
    loss, parts = lcons_from_attributions(A0.clone(), A0)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)
    loss, parts = lcons_from_attributions(3.0 * A0 + 1.0, A0)       # Pearson is affine-invariant per row
    assert parts["pearson_term"] == pytest.approx(0.0, abs=1e-5)
    g, g0 = (3.0 * A0 + 1.0).abs().mean(0), A0.abs().mean(0)        # global profile: mean |A| per segment
    assert parts["profile_term"] == pytest.approx(1 - float(torch.dot(g, g0) / (g.norm() * g0.norm())), abs=1e-5)
    loss, parts = lcons_from_attributions(-A0, A0)
    assert parts["pearson_term"] == pytest.approx(2.0, abs=1e-5)    # 1 - (-1)
    assert parts["profile_term"] == pytest.approx(0.0, abs=1e-5)    # |A| profile unchanged by a sign flip
    assert loss.item() == pytest.approx(parts["pearson_term"] + parts["profile_term"], abs=1e-6)


def test_lcons_loss_zero_for_original_and_differentiable():
    teacher = tinyhar()
    X, bg = data(6, seed=2), data(10, seed=3)
    with torch.no_grad():
        t = teacher(X).argmax(1)
    A0 = segment_attributions(eg_attributions(teacher, X, t, bg, 4, create_graph=False,
                                              generator=torch.Generator().manual_seed(0)))
    student = copy.deepcopy(teacher)
    same = lcons_loss(student, X, t, A0, bg, m=4, generator=torch.Generator().manual_seed(0))
    assert same.item() < 1e-4
    with torch.no_grad():
        for p in student.parameters():
            p.add_(0.05 * torch.randn_like(p))
    loss = lcons_loss(student, X, t, A0, bg, m=4, generator=torch.Generator().manual_seed(0))
    assert loss.item() > 1e-3
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters())


# ------------------------------------------------------------------------------ gate and tuning

def test_gate_picks_penalty_exactly_when_below_tau():
    assert gate_branch(0.80, 0.85) == "ce_plus_lcons"
    assert gate_branch(0.85, 0.85) == "ce"
    assert gate_branch(0.90, 0.85) == "ce"
    assert gate_branch(np.nextafter(0.85, 0), 0.85) == "ce_plus_lcons"
    for v in np.linspace(-0.5, 1.0, 31):
        assert (gate_branch(v, 0.6) == "ce_plus_lcons") == (v < 0.6)


def test_select_config_rule():
    rows = [{"config": "a", "val_f1": 0.80, "lci_val": 0.95},   # F1 too low (L1 = 0.82)
            {"config": "b", "val_f1": 0.811, "lci_val": 0.90},
            {"config": "c", "val_f1": 0.83, "lci_val": 0.88},
            {"config": "d", "val_f1": 0.8099, "lci_val": 0.99}]   # just outside 1 pt
    chosen, fallback = select_config(rows, l1_f1=0.82)
    assert chosen["config"] == "b" and not fallback
    chosen, fallback = select_config(rows[:1], l1_f1=0.82)      # nobody within 1 pt: best F1, flagged
    assert chosen["config"] == "a" and fallback


# ------------------------------------------------------------------------------ matched n_nonzero

@pytest.mark.parametrize("level", (0.3, 0.5, 0.7, 0.9))
def test_matched_n_nonzero_every_method_and_ablation(level):
    m = tinyhar()
    w = fake_windows(64)
    X = torch.from_numpy(w.X[:6])
    bg = torch.from_numpy(fake_windows(10, seed=7).X)
    with torch.no_grad():
        t = m(X).argmax(1)
    snip = snip_scores(m, w, n_batches=2, batch_size=16)
    imp = alignment_importance(m, X, t, bg, m=2, eg_seeds=(0,), batch_size=6)
    assert set(METHODS) >= {"l1", "snip", "park", *ABLATIONS}
    records = {}
    for name in METHODS:
        for beta in (XGP["betas"] if METHODS[name]["score"] == "keep" else (None,)):
            s = pruning_scores(name, m, snip=snip, importance=imp, beta=beta)
            records[(name, beta)] = size_record(prune_by_score(m, s, level))
    nnz = {k: r["n_nonzero"] for k, r in records.items()}
    assert len(set(nnz.values())) == 1, nnz
    assert all(abs(r["sparsity_prunable"] - level) < 1e-3 for r in records.values())
    with pytest.raises(ValueError):
        pruning_scores("xgp", m, snip=snip, importance=None, beta=0.3)


def test_pilot_reads_harth_only():
    """The pilot never loads HAR70+: only the HARTH splits of the window cache."""
    import inspect

    import har.run_xgp_pilot as pilot
    src = inspect.getsource(pilot)
    assert "load_harth_windows" in src
    assert "load_test_windows" not in src and "har70plus" not in src
