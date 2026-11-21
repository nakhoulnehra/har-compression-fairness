"""Tests for XGP pilot round 2 (plan Section 2.5 (b), round 2; har/augment.py, the KD recipe in
har/compression.py, the worst-class / augmented importance and the pre-registered rule in har/xgp.py,
and the runner har/run_xgp_pilot2.py).

    augmentation   per-channel mean kept by the movement scaling, deterministic with the seed,
                   shape (N, 6, 128), HARTH windows in and out
    KD             the KL term is 0 when student = teacher (the loss is then 0.3 x CE)
    worst-class    with one class it gives exactly the plain keep-score
    n_nonzero      identical for every round-2 method and variant at a level
    HAR70+         never loaded by the round-2 code
    rule           pre-registered selection of beta and of the final XGP
"""

import inspect

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from har.augment import (OLDER_LIKE, draw_older_like_params, older_like, older_like_windows, scale_movement,
                         time_stretch)
from har.compression import (KD, finetune, kd_kl_term, kd_loss, l1_scores, predict_logits, prune_by_score,
                             size_record, snip_scores)
from har.tests.test_compression import fake_windows, tinyhar
from har.xgp import (XGP2, alignment_importance, choose_beta, classwise_alignment_importance, keep_scores,
                     select_final_xgp, worst_class_importance)


def windows_x(n=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6, 128)).astype(np.float32)
    X[:, 2] += 1.0                                  # a gravity-like offset on one axis
    return X


# ------------------------------------------------------------------------------ augmentation

def test_design_constants():
    assert OLDER_LIKE["speed"] == (0.8, 0.9) and OLDER_LIKE["scale"] == (0.7, 0.9)
    assert XGP2["betas"] == (0.05, 0.1, 0.3) and XGP2["levels"] == (0.7, 0.9) and XGP2["seeds"] == (0, 1, 2)
    assert XGP2["baselines"] == ("l1", "l1_kd", "snip_kd", "park", "park_kd")
    assert XGP2["f1_tolerance"] == 0.01 and XGP2["tie"] == 0.005 and XGP2["epochs"] == 10
    assert set(XGP2["variants"]) == {"xgp_kd", "xgp_worstclass_kd", "xgp_aug_kd", "xgp_worstclass_aug_kd"}
    assert KD == {"T": 3.0, "alpha": 0.7}


def test_older_like_shape_dtype_and_ranges():
    X = windows_x(20)
    s, a = draw_older_like_params(20, seed=42)
    assert s.shape == a.shape == (20,)
    assert (s >= 0.8).all() and (s < 0.9).all() and (a >= 0.7).all() and (a < 0.9).all()
    Y = older_like(X, s, a)
    assert Y.shape == (20, 6, 128) and Y.dtype == np.float32 and np.isfinite(Y).all()


def test_older_like_deterministic_with_seed():
    s1, a1 = draw_older_like_params(50, seed=42)
    s2, a2 = draw_older_like_params(50, seed=42)
    s3, _ = draw_older_like_params(50, seed=43)
    assert np.array_equal(s1, s2) and np.array_equal(a1, a2) and not np.array_equal(s1, s3)
    X = windows_x(50)
    assert np.array_equal(older_like(X, s1, a1), older_like(X, s2, a2))
    # the first n draws do not depend on how many windows are drawn after them
    s4, a4 = draw_older_like_params(10, seed=42)
    assert np.array_equal(s4, s1[:10]) and np.array_equal(a4, a1[:10])


def test_movement_scaling_keeps_per_channel_mean():
    X = windows_x(8)
    a = np.linspace(0.7, 0.9, 8)
    Y = scale_movement(X, a)
    np.testing.assert_allclose(Y.mean(axis=2), X.mean(axis=2), atol=1e-5)
    # movement (deviation from the mean) is scaled by a
    np.testing.assert_allclose(Y - Y.mean(2, keepdims=True), a[:, None, None] * (X - X.mean(2, keepdims=True)),
                               atol=1e-5)
    # full older-like transform: per-channel mean equal to the mean of the stretched window
    s = np.full(8, 0.85)
    np.testing.assert_allclose(older_like(X, s, a).mean(axis=2), time_stretch(X, s).mean(axis=2), atol=1e-5)
    # a constant (gravity-only) channel is left exactly as it is
    Xc = np.ones((2, 6, 128), np.float32) * np.arange(6, dtype=np.float32)[None, :, None]
    np.testing.assert_allclose(older_like(Xc, np.array([0.8, 0.9]), np.array([0.7, 0.9])), Xc, atol=1e-6)


def test_time_stretch_takes_central_samples_and_resamples_linearly():
    t = np.arange(128, dtype=np.float32)
    X = np.broadcast_to(t, (1, 6, 128)).copy()
    s = np.array([0.8])
    L = round(128 * 0.8)                      # 102 central samples
    start = (128 - L) // 2
    Y = time_stretch(X, s)
    np.testing.assert_allclose(Y[0, 0], np.linspace(start, start + L - 1, 128), atol=1e-4)
    np.testing.assert_allclose(time_stretch(X, np.array([1.0])), X, atol=1e-6)   # s = 1: unchanged


def test_older_like_windows_harth_only():
    w = fake_windows(12)
    s, a = draw_older_like_params(12, seed=1)
    v = older_like_windows(w, s, a)
    assert v.dataset == "harth" and v.X.shape == w.X.shape
    assert np.array_equal(v.y, w.y) and np.array_equal(v.subject, w.subject)
    with pytest.raises(ValueError):
        older_like_windows(fake_windows(12, dataset="har70plus"), s, a)
    with pytest.raises(ValueError):
        older_like_windows(w, s[:5], a[:5])


# ------------------------------------------------------------------------------ KD recipe

def test_kd_kl_term_zero_when_student_equals_teacher():
    z = torch.randn(32, 7)
    assert float(kd_kl_term(z, z.clone(), T=3.0)) == pytest.approx(0.0, abs=1e-7)
    y = torch.randint(0, 7, (32,))
    ce = nn.CrossEntropyLoss(weight=torch.rand(7) + 0.5)
    assert float(kd_loss(z, z.clone(), y, ce)) == pytest.approx(0.3 * float(ce(z, y)), rel=1e-6)
    zs = torch.randn(32, 7)
    ref = 0.7 * F.kl_div(F.log_softmax(zs / 3, 1), F.softmax(z / 3, 1), reduction="batchmean") * 9 + 0.3 * ce(zs, y)
    assert float(kd_loss(zs, z, y, ce)) == pytest.approx(float(ref), rel=1e-6)
    assert float(kd_kl_term(zs, z, T=3.0)) > 0


def test_finetune_with_kd_runs_and_keeps_masks():
    m = tinyhar()
    w, v = fake_windows(64), fake_windows(32, seed=3)
    p = prune_by_score(m, l1_scores(m), 0.7)
    logits = torch.from_numpy(predict_logits(m, w.X))
    assert logits.shape == (64, 7)
    info = finetune(p, w, v, epochs=1, batch_size=32, kd_teacher_logits=logits, log=lambda s: None)
    assert info["epochs_run"] == 1 and info["history"][0]["train_kd"] is not None
    params = dict(p.named_parameters())
    assert all(torch.all(params[n][~mk] == 0) for n, mk in p.prune_masks.items())
    with pytest.raises(ValueError):
        finetune(p, w, v, epochs=1, kd_teacher_logits=logits[:10], log=lambda s: None)


# ------------------------------------------------------------------------------ worst-class importance

def _ref(n=14, seed=5):
    m = tinyhar()
    X = torch.from_numpy(fake_windows(n, seed=seed).X)
    bg = torch.from_numpy(fake_windows(10, seed=7).X)
    with torch.no_grad():
        t = m(X).argmax(1)
    return m, X, t, bg


def test_worst_class_equals_plain_with_one_class():
    m, X, t, bg = _ref()
    plain = alignment_importance(m, X, t, bg, m=2, eg_seeds=(0, 1), batch_size=7)
    per = classwise_alignment_importance(m, X, t, np.zeros(len(X), int), bg, m=2, eg_seeds=(0, 1), batch_size=7)
    assert list(per) == [0]
    for name in plain:
        assert torch.allclose(per[0][name], plain[name])
    worst = worst_class_importance(per)
    for beta in (0.05, 0.3, 1.0):
        a, b = keep_scores(m, plain, beta), keep_scores(m, worst, beta)
        for name in a:
            assert torch.allclose(a[name], b[name], rtol=1e-5, atol=1e-6)


def test_worst_class_is_max_of_layer_rescaled_class_scores():
    per = {0: {"w": torch.tensor([1.0, 3.0])}, 1: {"w": torch.tensor([10.0, 0.0])}, 2: {"w": torch.zeros(2)}}
    out = worst_class_importance(per)["w"]
    # class 0 / mean 2 -> [0.5, 1.5]; class 1 / mean 5 -> [2, 0]; class 2 all zero -> contributes 0
    assert torch.allclose(out, torch.tensor([2.0, 1.5]))


def test_classwise_uses_only_each_class_windows():
    m, X, t, bg = _ref()
    cls = np.array([0] * 7 + [3] * 7)
    per = classwise_alignment_importance(m, X, t, cls, bg, m=2, eg_seeds=(0,), batch_size=7)
    only3 = alignment_importance(m, X[7:], t[7:], bg, m=2, eg_seeds=(0,), batch_size=7)
    assert sorted(per) == [0, 3]
    assert all(torch.allclose(per[3][k], only3[k]) for k in only3)


# ------------------------------------------------------------------------------ matched n_nonzero

@pytest.mark.parametrize("level", (0.7, 0.9))
def test_identical_n_nonzero_every_round2_variant(level):
    m, X, t, bg = _ref()
    w = fake_windows(64)
    snip = snip_scores(m, w, n_batches=2, batch_size=16)
    plain = alignment_importance(m, X, t, bg, m=2, eg_seeds=(0,), batch_size=7)
    worst = worst_class_importance(classwise_alignment_importance(m, X, t, np.arange(len(X)) % 3, bg, m=2,
                                                                  eg_seeds=(0,), batch_size=7))
    Xa = torch.from_numpy(older_like(X.numpy(), *draw_older_like_params(len(X), 43)))
    Xb = torch.cat([X, Xa])
    with torch.no_grad():
        tb = m(Xb).argmax(1)
    aug = alignment_importance(m, Xb, tb, bg, m=2, eg_seeds=(0,), batch_size=7)
    scores = {"l1": l1_scores(m), "snip": snip}
    for b in XGP2["betas"]:
        for name, imp in (("plain", plain), ("worst", worst), ("aug", aug)):
            scores[f"{name}_{b}"] = keep_scores(m, imp, b)
    nnz = {k: size_record(prune_by_score(m, s, level))["n_nonzero"] for k, s in scores.items()}
    assert len(set(nnz.values())) == 1, nnz


# ------------------------------------------------------------------------------ pre-registered rule

def _stats(**kw):
    """{(method, level): {"lci_mean_sets", "f1_normal"}} from method=(lci70, f1_70, lci90, f1_90)."""
    out = {}
    for meth, (l7, f7, l9, f9) in kw.items():
        out[(meth, 0.7)] = {"lci_mean_sets": l7, "f1_normal": f7}
        out[(meth, 0.9)] = {"lci_mean_sets": l9, "f1_normal": f9}
    return out


BASE = dict(l1=(0.80, 0.73, 0.60, 0.60), l1_kd=(0.85, 0.78, 0.65, 0.66), snip_kd=(0.70, 0.75, 0.50, 0.62),
            park=(0.75, 0.74, 0.55, 0.61), park_kd=(0.80, 0.77, 0.60, 0.65))


def test_rule_picks_highest_lci_among_eligible():
    st = _stats(**BASE, xgp_kd=(0.90, 0.775, 0.70, 0.655), xgp_worstclass_kd=(0.95, 0.76, 0.80, 0.66),
                xgp_aug_kd=(0.905, 0.78, 0.704, 0.66), xgp_worstclass_aug_kd=(0.99, 0.70, 0.90, 0.70))
    r = select_final_xgp(st)
    # worstclass fails at 70% (0.76 < 0.78 - 0.01); worstclass_aug fails at 70%; aug 0.8045 vs xgp_kd 0.80: tie
    assert r["best_baseline_f1"] == {0.7: 0.78, 0.9: 0.66}
    elig = {row["variant"]: row["eligible"] for row in r["table"]}
    assert elig == {"xgp_kd": True, "xgp_worstclass_kd": False, "xgp_aug_kd": True, "xgp_worstclass_aug_kd": False}
    assert r["final"] == "xgp_kd"                      # within 0.005 of aug -> fewer components
    st[("xgp_aug_kd", 0.7)]["lci_mean_sets"] = 0.95    # aug now clearly better
    assert select_final_xgp(st)["final"] == "xgp_aug_kd"


def test_rule_no_eligible_variant_gives_no_fallback():
    st = _stats(**BASE, xgp_kd=(0.99, 0.70, 0.99, 0.70), xgp_worstclass_kd=(0.99, 0.70, 0.99, 0.70),
                xgp_aug_kd=(0.99, 0.70, 0.99, 0.70), xgp_worstclass_aug_kd=(0.99, 0.70, 0.99, 0.70))
    r = select_final_xgp(st)
    assert r["final"] is None and not any(row["eligible"] for row in r["table"])


def test_rule_missing_cell_raises():
    st = _stats(**BASE, xgp_kd=(0.9, 0.78, 0.7, 0.66))
    with pytest.raises(KeyError):
        select_final_xgp(st)


def test_choose_beta():
    rows = {0.05: {"lci_mean_sets": 0.80, "f1_normal": 0.78}, 0.1: {"lci_mean_sets": 0.803, "f1_normal": 0.775},
            0.3: {"lci_mean_sets": 0.90, "f1_normal": 0.70}}
    assert choose_beta(rows, best_baseline_f1=0.78) == (0.05, False)   # 0.3 ineligible; 0.1 within 0.005
    rows[0.1]["lci_mean_sets"] = 0.82
    assert choose_beta(rows, best_baseline_f1=0.78) == (0.1, False)
    assert choose_beta(rows, best_baseline_f1=0.95) == (0.05, True)    # none eligible: highest F1, flagged


# ------------------------------------------------------------------------------ HAR70+ never loaded

def test_round2_code_never_loads_har70plus():
    import har.augment as aug
    import har.run_xgp_pilot2 as pilot2
    for mod in (pilot2, aug):
        src = inspect.getsource(mod)
        assert "load_test_windows" not in src and "har70plus" not in src and "_load_cache" not in src
    assert "load_harth_windows" in inspect.getsource(pilot2)
    assert "require_harth" in inspect.getsource(pilot2) and "require_harth" in inspect.getsource(aug)


def test_preregistration_hash_ignores_line_endings(tmp_path):
    from har.run_xgp_pilot2 import file_sha
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_bytes(b"x,y\r\n1,2\r\n")
    b.write_bytes(b"x,y\n1,2\n")
    assert file_sha(a) == file_sha(b)
