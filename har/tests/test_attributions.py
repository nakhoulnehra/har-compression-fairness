"""Tests for the expected-gradients surrogate (plan Section 2.10, test_attributions.py).

Covered now: `eg_attributions` (in har/compression.py). The SHAP completeness and per-class ordering
tests of this file come with the SHAP cache (Step 8).
"""

import numpy as np
import torch
import torch.nn as nn

from har.compression import eg_attributions


class Linear7(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.fc = nn.Linear(6 * 128, 7)

    def forward(self, x):
        return self.fc(x.flatten(1))


class Tanh7(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.fc1, self.fc2 = nn.Linear(6 * 128, 16), nn.Linear(16, 7)

    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x.flatten(1))))


def data(n=5, seed=0):
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.normal(size=(n, 6, 128)).astype(np.float32))


def test_eg_linear_model_single_baseline_is_exact():
    m = Linear7()
    X, b = data(4), data(1, seed=1)
    target = torch.tensor([0, 3, 6, 3])
    A = eg_attributions(m, X, target, b, m=5, create_graph=False, generator=torch.Generator().manual_seed(0))
    w = m.fc.weight[target].reshape(4, 6, 128)
    assert A.shape == (4, 6, 128)
    assert torch.allclose(A, (X - b) * w, atol=1e-6)
    with torch.no_grad():   # completeness: sum = f_c(x) - f_c(b)
        gap = m(X).gather(1, target[:, None])[:, 0] - m(b)[0, target]
    assert torch.allclose(A.sum(dim=(1, 2)), gap, atol=1e-4)


def test_eg_seeded_and_no_graph_by_default():
    m = Tanh7()
    X, bg = data(3), data(10, seed=1)
    t = torch.tensor([1, 2, 3])
    a = eg_attributions(m, X, t, bg, m=4, create_graph=False, generator=torch.Generator().manual_seed(7))
    b = eg_attributions(m, X, t, bg, m=4, create_graph=False, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b) and not a.requires_grad


def test_eg_create_graph_gives_second_order_gradients():
    m = Tanh7()
    X, bg = data(3), data(10, seed=1)
    t = torch.tensor([1, 2, 3])
    A = eg_attributions(m, X, t, bg, m=4, create_graph=True, generator=torch.Generator().manual_seed(7))
    assert A.requires_grad
    (A ** 2).sum().backward()          # d/dtheta of a function of d f / dx
    g1 = m.fc1.weight.grad
    assert g1 is not None and g1.abs().sum() > 0
    assert m.fc2.weight.grad is not None and m.fc2.weight.grad.abs().sum() > 0


def test_shap_cache_hit_and_key(tmp_path):
    from har.attributions import shap_cached
    m = Linear7().eval()
    X, bg = data(4), data(8, seed=1)
    t = np.array([0, 1, 1, 2])
    ids = [f"w{i}" for i in range(4)]
    logs = []
    a = shap_cached(m, bg, X, t, ids, 0, 10, tmp_path, "t", log=logs.append)
    b = shap_cached(m, bg, X, t, ids, 0, 10, tmp_path, "t", log=logs.append)
    assert np.array_equal(a, b) and len(logs) == 1 and "hit" in logs[0]
    shap_cached(m, bg, X, t, ids, 1, 10, tmp_path, "t", log=logs.append)       # other SHAP seed: new entry
    with torch.no_grad():
        m.fc.weight.mul_(2)
    shap_cached(m, bg, X, t, ids, 0, 10, tmp_path, "t", log=logs.append)       # other weights: new entry
    assert len(logs) == 1 and len(list(tmp_path.glob("t-*.npz"))) == 3
