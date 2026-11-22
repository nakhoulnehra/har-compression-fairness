"""Tests for XGP round 3 (plan Section 2.5 (b), "Round 3 pre-registration"; har/xgp3.py, the DRO / epoch
callback additions to compression.finetune, compression.head_unpruned, and har/run_xgp_pilot3.py).

    importance      at theta0 with one group and one class: equals round 2's alignment_importance
    keep-score      pruned weights score -1; beta 0 ranks like |w|
    reconstruction  recovers an unpruned layer exactly; never writes a pruned weight; n_nonzero unchanged;
                    lowers each refitted layer's output error; ridge solve = lstsq on the support
    iterative       nested masks; exact uniform target, same n_nonzero as one-shot L1
    finetune        callback leaves training bit-identical; DRO moves weight to the worse group
    head check      True for pruned / fine-tuned models; False if a head weight is zero
    rules           pre-registered config and final selection; HAR70+ never loaded by the runner
"""

import copy
import inspect

import numpy as np
import pytest
import torch
import torch.nn as nn

from har.compression import finetune, head_unpruned, l1_scores, prune_by_score, size_record
from har.tests.test_compression import fake_windows, tinyhar
from har.xgp import alignment_importance
from har.xgp3 import (COMPONENTS, ROUND3, STARTS, alignment_importance_ref, best_baseline, group_class_importance,
                      iterative_prune, layerwise_reconstruct, masked_keep_scores, refit_modules, ridge_refit_rows,
                      select_config, select_final)


def X_bg(n=12, nb=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 6, 128, generator=g), torch.randn(nb, 6, 128, generator=g)


def test_design_constants():
    assert ROUND3["levels"] == (0.5, 0.7, 0.9) and ROUND3["seeds"] == (0, 1, 2) and ROUND3["folds"] == (0, 1)
    assert ROUND3["betas"] == (0.1, 0.3, 0.5) and ROUND3["rs"] == (0.25, 0.5) and ROUND3["K"] == 5
    assert ROUND3["epochs"] == 20 and ROUND3["curve_epochs"] == (2, 5) and ROUND3["damp"] == 0.01
    assert ROUND3["f1_tolerance"] == 0.01 and ROUND3["tie"] == 0.005 and ROUND3["dro_eta"] == 0.01
    assert len(COMPONENTS["no_finetune"]["xgp"]) == 4 and len(COMPONENTS["finetune"]["xgp_ft"]) == 6
    assert len(COMPONENTS["finetune"]["xgp_ft_no_aug"]) == 4
    assert STARTS["xgp_no_iter"]["K"] == 1 and STARTS["xgp_no_group"]["groups"] == ("normal",)


# ------------------------------------------------------------------------------ importance and keep-score

def test_importance_at_theta0_equals_round2():
    net = tinyhar()
    X, bg = X_bg()
    t = torch.arange(len(X)) % 7
    a = alignment_importance(net, X, t, bg, m=2, eg_seeds=(0, 1), batch_size=5)
    b = alignment_importance_ref(net, net, X, t, bg, m=2, eg_seeds=(0, 1), batch_size=5)
    for n in a:
        torch.testing.assert_close(a[n], b[n], rtol=1e-4, atol=1e-7)


def test_group_class_importance_one_group_one_class_is_layer_normalised():
    net = tinyhar()
    X, bg = X_bg(8)
    t = torch.zeros(8, dtype=torch.long)
    I = group_class_importance(net, net, {"normal": (X, t, np.zeros(8, int))}, bg, m=2, eg_seeds=(0,), batch_size=4)
    ref = alignment_importance_ref(net, net, X, t, bg, m=2, eg_seeds=(0,), batch_size=4)
    for n in I:
        mu = ref[n].mean()
        torch.testing.assert_close(I[n], ref[n] / mu if mu > 1e-12 else torch.zeros_like(ref[n]), rtol=1e-4, atol=1e-6)


def test_group_importance_is_max_over_groups():
    net = tinyhar()
    X, bg = X_bg(8)
    t = torch.zeros(8, dtype=torch.long)
    y = np.zeros(8, int)
    kw = dict(m=2, eg_seeds=(0,), batch_size=4)
    a = group_class_importance(net, net, {"normal": (X[:4], t[:4], y[:4])}, bg, **kw)
    b = group_class_importance(net, net, {"older_like": (X[4:], t[4:], y[4:])}, bg, **kw)
    both = group_class_importance(net, net, {"normal": (X[:4], t[:4], y[:4]), "older_like": (X[4:], t[4:], y[4:])},
                                  bg, **kw)
    for n in both:
        torch.testing.assert_close(both[n], torch.maximum(a[n], b[n]))


def test_masked_keep_scores_pruned_get_minus_one_and_beta0_ranks_like_l1():
    net = prune_by_score(tinyhar(), l1_scores(tinyhar()), 0.5)
    sc = masked_keep_scores(net, None, 0.0)
    w = l1_scores(net)
    for n, mk in net.prune_masks.items():
        assert torch.all(sc[n][~mk] == -1)
        assert torch.equal(torch.argsort(sc[n][mk]), torch.argsort(w[n][mk]))


# ------------------------------------------------------------------------------ reconstruction

def test_ridge_rows_match_lstsq_on_support():
    g = torch.Generator().manual_seed(0)
    X = torch.randn(200, 5, generator=g, dtype=torch.float64)
    W = torch.randn(3, 5, generator=g, dtype=torch.float64)
    Y = X @ W.T + 0.5
    keep = torch.tensor([[1, 1, 0, 1, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]], dtype=torch.bool)
    X1 = torch.cat([X, torch.ones(200, 1, dtype=X.dtype)], 1)
    sol = ridge_refit_rows(X1.T @ X1, X1.T @ Y, keep, True, damp=0.0)
    for i in range(3):
        S = torch.cat([torch.nonzero(keep[i]).flatten(), torch.tensor([5])])
        ref = torch.linalg.lstsq(X1[:, S], Y[:, i:i + 1]).solution.squeeze(1)
        torch.testing.assert_close(sol[i, S], ref, rtol=1e-6, atol=1e-8)
        assert torch.all(sol[i, [j for j in range(5) if not keep[i, j]]] == 0)
    torch.testing.assert_close(sol[1, :5], W[1], rtol=1e-6, atol=1e-8)        # dense row recovered


def test_refit_modules_tinyhar():
    names = refit_modules(tinyhar(), X_bg(2)[0])
    convs = [n for n in names if "convs" in n]
    assert len(convs) == 4
    assert any(n.endswith("linear1") for n in names) and any(n.endswith("linear2") for n in names)
    assert any("fusion" in n for n in names) and any(n.endswith("attn_score") for n in names)
    assert not any("out_proj" in n for n in names) and not any(n.endswith("head") for n in names)


def test_reconstruction_keeps_masks_and_reduces_error():
    net0 = tinyhar()
    X, _ = X_bg(40)
    net = prune_by_score(net0, l1_scores(net0), 0.7)
    nnz = size_record(net)["n_nonzero"]
    log = layerwise_reconstruct(net, net0, X, damp=0.01, batch_size=16)
    assert size_record(net)["n_nonzero"] == nnz
    params = dict(net.named_parameters())
    for n, mk in net.prune_masks.items():
        assert torch.all(params[n][~mk] == 0)
    assert len(log["refit"]) == 8
    assert all(r["rel_error_after"] <= r["rel_error_before"] + 1e-9 for r in log["refit"])
    assert any("lstm" in n for n in log["not_refit"]) and any("in_proj" in n for n in log["not_refit"])
    assert torch.equal(net.head.weight, net0.head.weight)


def test_reconstruction_of_unpruned_model_changes_little():
    net0 = tinyhar()
    X, _ = X_bg(40)
    net = prune_by_score(net0, l1_scores(net0), 0.0)
    layerwise_reconstruct(net, net0, X, damp=1e-8, batch_size=16)
    with torch.no_grad():
        torch.testing.assert_close(net(X), net0(X), rtol=1e-3, atol=1e-3)


# ------------------------------------------------------------------------------ iterative pruning

def test_iterative_prune_nested_and_exact_target():
    net0 = tinyhar()
    X, _ = X_bg(20)
    net, rows = iterative_prune(net0, 0.7, K=5, beta=0.0, importance_fn=None,
                                recon_fn=lambda n: layerwise_reconstruct(n, net0, X, batch_size=10), log=lambda s: None)
    one = prune_by_score(net0, l1_scores(net0), 0.7)
    assert size_record(net)["n_nonzero"] == size_record(one)["n_nonzero"]
    assert [r["step"] for r in rows] == [1, 2, 3, 4, 5]
    assert np.allclose([r["sparsity"] for r in rows], [0.14, 0.28, 0.42, 0.56, 0.7])
    assert all(a["n_nonzero"] > b["n_nonzero"] for a, b in zip(rows, rows[1:]))
    rec = size_record(net)
    for v in rec["per_layer_sparsity"].values():
        assert abs(v - 0.7) < 0.01


def test_iterative_k1_no_recon_beta0_equals_l1():
    net0 = tinyhar()
    net, _ = iterative_prune(net0, 0.5, K=1, beta=0.0, importance_fn=None, recon_fn=None, log=lambda s: None)
    one = prune_by_score(net0, l1_scores(net0), 0.5)
    for n in one.prune_masks:
        assert torch.equal(net.prune_masks[n], one.prune_masks[n])


# ------------------------------------------------------------------------------ finetune additions

def _ft(net, tr, va, **kw):
    return finetune(net, tr, va, epochs=2, lr=1e-3, batch_size=16, seed=0, log=lambda s: None, **kw)


def test_epoch_callback_leaves_training_identical():
    tr, va = fake_windows(48, seed=1), fake_windows(14, seed=2)
    a = prune_by_score(tinyhar(), l1_scores(tinyhar()), 0.5)
    b = copy.deepcopy(a)
    _ft(a, tr, va)
    seen = []

    def cb(ep, net):
        seen.append(ep)
        torch.rand(100)
        np.random.rand(10)
        with torch.no_grad():
            net(torch.randn(3, 6, 128))
    _ft(b, tr, va, on_epoch_end=cb)
    assert seen == [1, 2]
    for (n, p), (_, q) in zip(a.state_dict().items(), b.state_dict().items()):
        assert torch.equal(p, q), n


def test_dro_moves_weight_to_the_worse_group_and_keeps_masks():
    tr, va = fake_windows(64, seed=1), fake_windows(14, seed=2)
    tr.X[32:] *= 5.0                                     # group 1 = harder
    groups = np.r_[np.zeros(32, int), np.ones(32, int)]
    net = prune_by_score(tinyhar(), l1_scores(tinyhar()), 0.5)
    info = finetune(net, tr, va, epochs=2, lr=1e-3, batch_size=16, seed=0, log=lambda s: None,
                    groups=groups, dro_eta=0.5)
    h = info["history"][-1]
    assert abs(h["dro_q0"] + h["dro_q1"] - 1) < 1e-9
    params = dict(net.named_parameters())
    for n, mk in net.prune_masks.items():
        assert torch.all(params[n][~mk] == 0)
    with pytest.raises(ValueError):
        finetune(net, tr, va, epochs=1, groups=groups, log=lambda s: None)


def test_dro_single_group_equals_plain_loss():
    from har.compression import _dro_loss, kd_loss
    torch.manual_seed(0)
    z, zt, y, cw = torch.randn(16, 7), torch.randn(16, 7), torch.randint(0, 7, (16,)), torch.rand(7) + 0.5
    ce = nn.CrossEntropyLoss(weight=cw)
    one = torch.zeros(16, dtype=torch.long)
    q = torch.ones(1, dtype=torch.float64)
    torch.testing.assert_close(_dro_loss(z, y, None, one, cw, q.clone(), 0.01), ce(z, y))
    torch.testing.assert_close(_dro_loss(z, y, zt, one, cw, q.clone(), 0.01), kd_loss(z, zt, y, ce))
    # two groups: q moves towards the group with the larger loss
    two = torch.r_[torch.zeros(8, dtype=torch.long), torch.ones(8, dtype=torch.long)] if hasattr(torch, "r_") else         torch.cat([torch.zeros(8, dtype=torch.long), torch.ones(8, dtype=torch.long)])
    z2 = z.clone()
    z2[8:] = 0.0
    z2[8:, 0] = -10.0
    q2 = torch.full((2,), 0.5, dtype=torch.float64)
    _dro_loss(z2, y, None, two, cw, q2, 1.0)
    l0 = (cw[y[:8]] * nn.functional.cross_entropy(z2[:8], y[:8], reduction="none")).sum() / cw[y[:8]].sum()
    l1 = (cw[y[8:]] * nn.functional.cross_entropy(z2[8:], y[8:], reduction="none")).sum() / cw[y[8:]].sum()
    assert (q2[1] > q2[0]) == bool(l1 > l0)


def test_head_unpruned_check():
    net0 = tinyhar()
    net = prune_by_score(net0, l1_scores(net0), 0.9)
    assert head_unpruned(net)
    tr, va = fake_windows(32, seed=1), fake_windows(14, seed=2)
    _ft(net, tr, va)
    assert not torch.equal(net.head.weight, net0.head.weight)     # the head trains ...
    assert head_unpruned(net)                                     # ... and is still unpruned
    with torch.no_grad():
        net.head.weight[0, 0] = 0.0
    assert not head_unpruned(net)


# ------------------------------------------------------------------------------ rules

def test_select_config_rule():
    best = {"f1_normal": 0.80, "f1_older_like": 0.70}
    cfg = {(0.1, None): {"fa_lci": 0.60, "f1_normal": 0.80, "f1_older_like": 0.70},
           (0.3, None): {"fa_lci": 0.603, "f1_normal": 0.795, "f1_older_like": 0.695},     # tie -> smaller beta
           (0.5, None): {"fa_lci": 0.90, "f1_normal": 0.80, "f1_older_like": 0.68}}        # older-like F1 too low
    pick, table = select_config(cfg, best)
    assert pick == (0.1, None)
    assert [t["eligible"] for t in table] == [True, True, False]
    cfg[(0.3, None)]["fa_lci"] = 0.62
    assert select_config(cfg, best)[0] == (0.3, None)
    ft = {(0.1, 0.25): {"fa_lci": 0.8, "f1_normal": 0.8, "f1_older_like": 0.7},
          (0.1, 0.5): {"fa_lci": 0.802, "f1_normal": 0.8, "f1_older_like": 0.7}}
    assert select_config(ft, best)[0] == (0.1, 0.25)
    none = {(0.1, None): {"fa_lci": 0.9, "f1_normal": 0.5, "f1_older_like": 0.7}}
    assert select_config(none, best)[0] is None


def test_select_final_rule():
    comp = COMPONENTS["no_finetune"]
    best = {lv: {"f1_normal": 0.8, "f1_older_like": 0.7} for lv in (0.5, 0.7, 0.9)}
    ok = {"f1_normal": 0.8, "f1_older_like": 0.7}
    c = {"xgp": {lv: {"fa_lci": 0.70, **ok} for lv in best},
         "xgp_no_recon": {lv: {"fa_lci": 0.697, **ok} for lv in best},                # within tie, fewer components
         "xgp_no_iter": {lv: {"fa_lci": 0.90, **ok} for lv in best}}
    c["xgp_no_iter"][0.9]["f1_normal"] = 0.7                                         # not eligible at 90%
    final, table = select_final(c, best, comp)
    assert final == "xgp_no_recon"
    assert {t["variant"]: t["eligible"] for t in table} == {"xgp": True, "xgp_no_recon": True, "xgp_no_iter": False}
    c["xgp_no_recon"] = {lv: {"fa_lci": 0.60, **ok} for lv in best}
    assert select_final(c, best, comp)[0] == "xgp"


def test_best_baseline_includes_r_variants():
    st = {"l1": {"f1_normal": 0.7, "f1_older_like": 0.6}, "l1_kdaug(r=0.25)": {"f1_normal": 0.75, "f1_older_like": 0.66},
          "xgp_ft(beta=0.1,r=0.25)": {"f1_normal": 0.99, "f1_older_like": 0.99}}
    assert best_baseline(st, ROUND3["baselines"]["finetune"]) == {"f1_normal": 0.75, "f1_older_like": 0.66}


def test_runner_never_loads_har70plus():
    import har.run_xgp_pilot3 as r3
    src = inspect.getsource(r3)
    assert "load_test_windows" not in src and "har70plus_test" not in src
    assert "require_harth" in src


# ------------------------------------------------------------------------------ round-3 older-like rotation

def test_rotation_params_fixed_and_round2_draws_unchanged():
    from har.augment import OLDER_LIKE_R3, draw_older_like_params, draw_older_like_r3_params
    assert OLDER_LIKE_R3["rotation_deg"] == (-15.0, 15.0)
    assert OLDER_LIKE_R3["sensors"] == {"back": (0, 1, 2), "thigh": (3, 4, 5)}
    q = draw_older_like_r3_params(200, 42)
    s, a = draw_older_like_params(200, 42)
    assert np.array_equal(q["s"], s) and np.array_equal(q["a"], a)            # round 2's s, a unchanged
    assert q["axes"].shape == (200, 2, 3) and q["angles"].shape == (200, 2)
    np.testing.assert_allclose(np.linalg.norm(q["axes"], axis=2), 1.0, atol=1e-12)
    assert (np.abs(q["angles"]) <= 15).all() and q["angles"].min() < -10 and q["angles"].max() > 10
    q2 = draw_older_like_r3_params(10, 42)
    assert np.array_equal(q2["axes"], q["axes"][:10]) and np.array_equal(q2["angles"], q["angles"][:10])
    assert not np.array_equal(q["angles"][:, 0], q["angles"][:, 1])            # sensors drawn independently


def test_rotation_matrices_are_rotations_with_the_right_angle():
    from har.augment import rotation_matrices
    rng = np.random.default_rng(0)
    k = rng.normal(size=(50, 3))
    k /= np.linalg.norm(k, axis=1, keepdims=True)
    th = rng.uniform(-15, 15, 50)
    R = rotation_matrices(k, th)
    np.testing.assert_allclose(R @ np.swapaxes(R, 1, 2), np.broadcast_to(np.eye(3), R.shape), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.einsum("nij,nj->ni", R, k), k, atol=1e-12)    # the axis is fixed
    np.testing.assert_allclose(np.rad2deg(np.arccos((np.trace(R, axis1=1, axis2=2) - 1) / 2)), np.abs(th), atol=1e-6)
    # +90 deg about z maps x to y
    np.testing.assert_allclose(rotation_matrices(np.array([0, 0, 1.0]), np.array(90.0)) @ [1, 0, 0], [0, 1, 0],
                               atol=1e-12)


def test_rotate_sensors_per_triplet_and_order_of_steps():
    from har.augment import older_like, older_like_r3, rotate_sensors
    rng = np.random.default_rng(1)
    X = rng.normal(size=(6, 6, 128)).astype(np.float32)
    X[:, 0] -= 1.0                                                              # gravity on back_x
    axes = rng.normal(size=(6, 2, 3))
    axes /= np.linalg.norm(axes, axis=2, keepdims=True)
    ang = rng.uniform(-15, 15, (6, 2))
    Y = rotate_sensors(X, axes, ang)
    for ch in ((0, 1, 2), (3, 4, 5)):                                           # per-sample norm kept per sensor
        np.testing.assert_allclose(np.linalg.norm(Y[:, ch], axis=1), np.linalg.norm(X[:, ch], axis=1), rtol=1e-5)
    assert not np.allclose(Y[:, :3].mean(2), X[:, :3].mean(2), atol=1e-3)       # the gravity direction moves
    # zero angle on the thigh only: thigh unchanged, back rotated
    ang0 = ang.copy()
    ang0[:, 1] = 0.0
    Z = rotate_sensors(X, axes, ang0)
    np.testing.assert_allclose(Z[:, 3:], X[:, 3:], atol=1e-6)
    s, a = np.full(6, 0.85), np.full(6, 0.8)
    np.testing.assert_allclose(older_like_r3(X, s, a, axes, ang), rotate_sensors(older_like(X, s, a), axes, ang),
                               atol=1e-6)
    np.testing.assert_allclose(older_like_r3(X, s, a, axes, np.zeros((6, 2))), older_like(X, s, a), atol=1e-6)


def test_round3_runner_uses_the_rotated_transform():
    import har.run_xgp_pilot3 as r3
    src = inspect.getsource(r3)
    assert "older_like_r3" in src and "older_like_windows_r3" in src and "draw_older_like_r3_params" in src
    assert "draw_older_like_params(" not in src and "older_like_windows(" not in src
