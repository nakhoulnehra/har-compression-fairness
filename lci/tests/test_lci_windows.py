"""Window extensions of lci/metric.py used by the HAR noise floor and the XGP gate (plan 2.6, 2.10).

Covered: aggregate_time_segments and compute_lci_by_subject. fa_lci and its tests come with Step 11.
"""

import numpy as np
import pytest

from lci.metric import aggregate_time_segments, compute_lci, compute_lci_by_subject

RNG = np.random.default_rng(0)


def test_aggregate_sums_keeps_sign_and_is_channel_major():
    A = np.zeros((2, 6, 128))
    A[0, 1, 16:32] = 1.0            # axis 1, segment 1 -> feature 9
    A[0, 1, 20] = -3.0              # sign kept: 15 * 1 - 3 = 12
    A[1, 5, 0:16] = -2.0            # axis 5, segment 0 -> feature 40
    S = aggregate_time_segments(A)
    assert S.shape == (2, 48)
    assert S[0, 9] == pytest.approx(12.0) and np.count_nonzero(S[0]) == 1
    assert S[1, 40] == pytest.approx(-32.0)
    X = RNG.normal(size=(3, 6, 128))
    assert np.allclose(aggregate_time_segments(X).sum(1), X.sum(axis=(1, 2)))


def test_aggregate_raises_when_not_divisible():
    with pytest.raises(ValueError):
        aggregate_time_segments(np.zeros((1, 6, 100)))


def test_lci_on_windows_equals_flattened():
    A, B = RNG.normal(size=(20, 6, 128)), RNG.normal(size=(20, 6, 128))
    assert compute_lci(A, B)["lci"] == pytest.approx(compute_lci(A.reshape(20, 768), B.reshape(20, 768))["lci"])


def test_lci_by_subject_equals_per_slice():
    A = RNG.normal(size=(30, 48))
    B = A + RNG.normal(scale=0.5, size=A.shape)
    subj = np.array(["S1"] * 10 + ["S2"] * 12 + ["S3"] * 8)
    out = compute_lci_by_subject(A, B, subj, k=10)
    assert list(out) == ["S1", "S2", "S3"]
    for s in out:
        ref = compute_lci(A[subj == s], B[subj == s], k=10)
        assert out[s]["subject"] == s
        assert all(out[s][k] == ref[k] for k in ref)
    with pytest.raises(ValueError):
        compute_lci_by_subject(A, B, subj[:-1])
