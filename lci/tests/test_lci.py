"""Unit tests for compute_lci (plan/LCI_IMPLEMENTATION.md, Section 2.7)."""

import numpy as np
import pytest

from lci.metric import compute_lci

RNG = np.random.default_rng(0)


def _base(n=40, F=30):
    # Distinct column scales give a clear, tie-free global ranking.
    scales = np.linspace(3.0, 0.1, F)
    return RNG.normal(size=(n, F)) * scales


def test_metric_does_not_import_torch_or_shap():
    import lci.metric as m
    src = open(m.__file__, encoding="utf-8").read()
    assert "import torch" not in src and "import shap" not in src


def test_identical():
    A = _base()
    r = compute_lci(A, A.copy())
    assert r["lci"] == pytest.approx(1.0)
    assert r["shap_corr"] == pytest.approx(1.0)
    assert r["jaccard"] == pytest.approx(1.0)
    assert r["n_skipped"] == 0 and r["mprf_shift"] == 0


def test_positive_scaling():
    A = _base()
    assert compute_lci(A, 3 * A)["lci"] == pytest.approx(1.0)


def test_negation():
    A = _base()
    r = compute_lci(A, -A)
    assert r["shap_corr"] == pytest.approx(-1.0)
    assert r["jaccard"] == pytest.approx(1.0)
    assert r["lci"] == pytest.approx(0.0)


def test_same_ranking_with_noise():
    A = _base(n=200)
    noisy = A + RNG.normal(scale=0.05, size=A.shape)
    r = compute_lci(A, noisy)
    assert r["jaccard"] == pytest.approx(1.0)
    assert r["shap_corr"] < 1.0


def test_disjoint_topk():
    n, F = 20, 30
    A_T = np.full((n, F), 0.01)
    A_S = np.full((n, F), 0.01)
    A_T[:, :10] = 1.0   # teacher top-10 = features 0..9
    A_S[:, 20:] = 1.0   # student top-10 = features 20..29
    A_T += RNG.normal(scale=1e-3, size=A_T.shape)
    A_S += RNG.normal(scale=1e-3, size=A_S.shape)
    assert compute_lci(A_T, A_S)["jaccard"] == 0.0


def test_jaccard_not_overlap_over_k():
    n, F = 20, 30
    imp_T = np.zeros(F); imp_T[0:10] = np.arange(10, 0, -1) + 10   # top-10 = 0..9
    imp_S = np.zeros(F); imp_S[5:15] = np.arange(10, 0, -1) + 10   # top-10 = 5..14
    A_T = np.tile(imp_T, (n, 1)) + RNG.normal(scale=1e-3, size=(n, F))
    A_S = np.tile(imp_S, (n, 1)) + RNG.normal(scale=1e-3, size=(n, F))
    j = compute_lci(A_T, A_S)["jaccard"]
    assert j == pytest.approx(5 / 15)
    assert j != pytest.approx(0.5)


def test_constant_sample_skipped():
    A = _base(n=10)
    B = A.copy()
    B[3, :] = 7.0  # constant vector in the student
    r = compute_lci(A, B)
    assert r["n_skipped"] == 1
    assert r["n_samples"] == 10
    assert r["shap_corr"] == pytest.approx(1.0)


def test_all_skipped_gives_zero_corr():
    A = np.ones((5, 12))
    r = compute_lci(A, A, k=3)
    assert r["n_skipped"] == 5 and r["shap_corr"] == 0.0


def test_errors():
    A = _base(n=10, F=30)
    with pytest.raises(ValueError):
        compute_lci(A, A[:, :29])
    with pytest.raises(ValueError):
        compute_lci(A, A[:9])
    with pytest.raises(ValueError):
        compute_lci(A, A, k=31)


def test_trailing_dims_flattened():
    A = RNG.normal(size=(25, 5, 6)) * np.linspace(2, 0.1, 30).reshape(5, 6)
    B = A + RNG.normal(scale=0.5, size=A.shape)
    r3 = compute_lci(A, B)
    r2 = compute_lci(A.reshape(25, 30), B.reshape(25, 30))
    for key in ("lci", "shap_corr", "jaccard", "n_skipped", "mprf_feature", "mprf_shift"):
        assert r3[key] == pytest.approx(r2[key])


def test_lci_not_clipped():
    # B[i] = -A[i] + c (per-sample constant) -> r = -1 for every sample. With A > 0 the
    # largest-|B| features are A's smallest ones, so the top-10 sets are disjoint -> LCI = -0.5.
    A = np.abs(_base()) + 0.01
    B = -A + 100.0
    r = compute_lci(A, B)
    assert r["shap_corr"] == pytest.approx(-1.0)
    assert r["jaccard"] == 0.0
    assert r["lci"] == pytest.approx(-0.5)
