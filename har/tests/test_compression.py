"""Tests for the baseline parts of har/compression.py (plan Section 2.10, test_compression.py).

Covered now: prunable tensors (head excluded), ptq_symmetric, SNIP-score pruning scores and Park's
attribution-matching loss. Not yet: matched n_nonzero across methods, finetune, the keep-score and
the gate (they come with the pruning code and our method, before Step 9).
"""

import copy

import numpy as np
import pytest
import torch
import torch.nn as nn

from har.compression import (park_am_distance, park_am_loss, park_teacher_attributions, prunable_tensors,
                             ptq_symmetric, segment_attributions, snip_scores, snip_scores_from_batches,
                             unit_norm)
from har.data import Windows
from har.models import build_model, resolve_config

MEAN = np.array([-0.9, 0.0, -0.2, -0.5, 0.0, 0.4], np.float32)
STD = np.array([0.3, 0.2, 0.3, 0.5, 0.3, 0.6], np.float32)


class Toy(nn.Module):
    """2-layer network: Linear -> BatchNorm -> ReLU -> head."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(6 * 128, 16)
        self.bn = nn.BatchNorm1d(16)
        self.head = nn.Linear(16, 7)

    def forward(self, x):
        return self.head(torch.relu(self.bn(self.fc(x.flatten(1)))))


def toy(seed=0):
    torch.manual_seed(seed)
    m = Toy()
    with torch.no_grad():          # non-trivial BN parameters and running stats
        m.bn.weight.uniform_(0.5, 1.5); m.bn.bias.uniform_(-0.2, 0.2)
        m.bn.running_mean.uniform_(-0.1, 0.1); m.bn.running_var.uniform_(0.5, 1.5)
    return m.eval()


def tinyhar(seed=0):
    torch.manual_seed(seed)
    return build_model("tinyhar", resolve_config("tinyhar", 1.0), MEAN, STD).eval()


def fake_windows(n=64, dataset="harth", seed=0):
    rng = np.random.default_rng(seed)
    return Windows(X=rng.normal(size=(n, 6, 128)).astype(np.float32), y=np.arange(n) % 7,
                   subject=np.array(["S006"] * n), dataset=dataset, group=np.array(["general"] * n),
                   walking_aid=np.array(["not_applicable"] * n), majority_share=np.ones(n, np.float32),
                   start_time=np.zeros(n, "datetime64[ms]"), start_row=np.arange(n))


# ------------------------------------------------------------------------------ prunable tensors

def test_prunable_excludes_head_and_non_weights():
    names = [n for n, _, _ in prunable_tensors(toy())]
    assert names == ["fc.weight"]


def test_prunable_tinyhar_scope():
    names = {n for n, _, _ in prunable_tensors(tinyhar())}
    assert "backbone.head.weight" not in names
    assert not any(n.endswith("bias") or ".bn" in n or "gamma" in n or n.startswith("norm.") for n in names)
    for expected in ("backbone.convs.0.weight", "backbone.lstm.weight_ih_l0", "backbone.lstm.weight_hh_l0",
                     "backbone.channel_attn.self_attn.in_proj_weight",
                     "backbone.channel_attn.self_attn.out_proj.weight", "backbone.channel_attn.linear1.weight",
                     "backbone.fusion.0.weight"):
        assert expected in names, expected
    with_head = {n for n, _, _ in prunable_tensors(tinyhar(), include_head=True)}
    assert with_head == names | {"backbone.head.weight"}


# ------------------------------------------------------------------------------------- PTQ

@pytest.mark.parametrize("bits", [8, 4])
def test_ptq_symmetric(bits):
    m = tinyhar()
    with torch.no_grad():                         # make sure some weights are exactly zero
        m.backbone.fusion[0].weight[0, :5] = 0.0
    q, quantized = ptq_symmetric(m, bits)
    assert set(quantized) == {n for n, _, _ in prunable_tensors(m, include_head=True)}
    assert "backbone.head.weight" in quantized and all(b == bits for b in quantized.values())
    before, after = dict(m.named_parameters()), dict(q.named_parameters())
    for name, p in after.items():
        w = before[name]
        if name in quantized:
            assert len(torch.unique(p)) <= 2 ** bits - 1, name
            assert torch.isclose(p.abs().max(), w.abs().max(), rtol=1e-6), name
            assert torch.all(p[w == 0] == 0), name
            assert not torch.equal(p, w), name
        else:                                       # biases, BN, gamma: unchanged
            assert torch.equal(p, w), name
    for (n1, b1), (n2, b2) in zip(m.named_buffers(), q.named_buffers()):
        assert n1 == n2 and torch.equal(b1, b2), n1    # BN running stats, normalisation statistics
    assert torch.equal(m.backbone.head.weight, before["backbone.head.weight"])   # original left untouched


def test_ptq_symmetric_levels_hand_example():
    m = toy()
    with torch.no_grad():
        m.fc.weight.zero_()
        m.fc.weight[0, :4] = torch.tensor([1.0, -0.55, 0.26, -1.0])
    q, _ = ptq_symmetric(m, 4)       # s = 1 / 7; levels k/7, k = -7..7
    got = q.fc.weight[0, :4]
    # 1 -> 7, -0.55 * 7 = -3.85 -> -4, 0.26 * 7 = 1.82 -> 2, -1 -> -7 (no exact halves: float ties are fragile)
    assert torch.allclose(got, torch.tensor([7, -4, 2, -7]) / 7.0)


# ------------------------------------------------------------------------------------- SNIP

def test_snip_scores_equal_abs_w_times_grad():
    m = toy()
    rng = np.random.default_rng(1)
    batches = [(torch.from_numpy(rng.normal(size=(8, 6, 128)).astype(np.float32)), torch.arange(8) % 7)
               for _ in range(3)]
    loss_fn = nn.CrossEntropyLoss()
    scores = snip_scores_from_batches(m, batches, loss_fn)
    ref = copy.deepcopy(m).eval()
    ref.zero_grad()
    for X, y in batches:
        loss_fn(ref(X), y).backward()
    assert set(scores) == {"fc.weight"}
    assert torch.allclose(scores["fc.weight"], (ref.fc.weight * ref.fc.weight.grad).abs())
    assert all(p.grad is None for p in m.parameters())          # model left clean
    assert torch.equal(m.bn.running_mean, ref.bn.running_mean)  # eval mode: BN stats not updated


def test_snip_scores_on_windows_fixed_batches_and_harth_only():
    m = tinyhar()
    w = fake_windows(64)
    a = snip_scores(m, w, n_batches=3, batch_size=16, seed=42)
    b = snip_scores(m, w, n_batches=3, batch_size=16, seed=42)
    assert set(a) == {n for n, _, _ in prunable_tensors(m)}
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert all(a[k].shape == dict(m.named_parameters())[k].shape for k in a)
    with pytest.raises(ValueError):
        snip_scores(m, fake_windows(64, dataset="har70plus"), n_batches=1, batch_size=16)


# ------------------------------------------------------------------------------ Park's L_AM

def test_segment_attributions_channel_major_signed_sum():
    A = torch.zeros(2, 6, 128)
    A[0, 1, 16:32] = 1.0          # axis 1, segment 1 -> feature 1 * 8 + 1 = 9
    A[1, 5, 0:16] = -2.0          # axis 5, segment 0 -> feature 40
    S = segment_attributions(A)
    assert S.shape == (2, 48)
    assert S[0, 9] == 16.0 and S[0].abs().sum() == 16.0
    assert S[1, 40] == -32.0


def test_park_am_distance_unsquared_and_scale_free():
    a_t = unit_norm(torch.tensor([[3.0, 4.0, 0.0]]))
    assert torch.isclose(park_am_distance(torch.tensor([[-6.0, -8.0, 0.0]]), a_t), torch.tensor(2.0))
    assert torch.isclose(park_am_distance(torch.tensor([[30.0, 40.0, 0.0]]), a_t), torch.tensor(0.0), atol=1e-7)
    # unsquared: orthogonal unit vectors are sqrt(2) apart
    assert torch.isclose(park_am_distance(torch.tensor([[0.0, 0.0, 5.0]]), a_t), torch.tensor(2 ** 0.5))


def test_park_loss_zero_for_teacher_and_differentiable():
    teacher = tinyhar()
    rng = np.random.default_rng(2)
    X = torch.from_numpy(rng.normal(size=(6, 6, 128)).astype(np.float32))
    bg = torch.from_numpy(rng.normal(size=(10, 6, 128)).astype(np.float32))
    with torch.no_grad():
        target = teacher(X).argmax(1)
    A_T = park_teacher_attributions(teacher, X, target, bg, m=4, seed=0)
    assert A_T.shape == (6, 48) and not A_T.requires_grad
    assert torch.allclose(A_T.norm(dim=1), torch.ones(6), atol=1e-5)
    student = copy.deepcopy(teacher)
    same = park_am_loss(student, X, target, A_T, bg, m=4, generator=torch.Generator().manual_seed(0))
    assert same.item() < 1e-4                         # identical model, identical draws
    with torch.no_grad():
        for p in student.parameters():
            p.add_(0.05 * torch.randn_like(p))
    loss = park_am_loss(student, X, target, A_T, bg, m=4, generator=torch.Generator().manual_seed(0))
    assert loss.item() > 1e-3
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters())
