"""Compression baselines and the attribution surrogate (plan Section 2.5).

Written so far (the baseline parts the Step 6 timing needs):
    prunable_tensors      the weight tensors pruning and PTQ act on (head excluded unless asked)
    ptq_symmetric         per-tensor symmetric weight PTQ, the primary PTQ            [plan 2.5, Q1]
    snip_scores           SNIP saliency |w * dL/dw| on the trained model             [Lee et al. 2019; plan Q2]
    eg_attributions       differentiable expected-gradients surrogate of SHAP        [plan 2.5]
    segment_attributions  (B, 6, 128) -> (B, 48) signed sums, channel-major           [plan 2.6]
    park_teacher_attributions, park_am_distance, park_am_loss
                          L1 + attribution matching, adapted                         [Park Sec 3.2, Eq. 2]
Not written yet: L1 scores, prune_by_score, size_record, finetune, ptq_tulla, and our method
(keep-score, L_cons, gate); our method is designed and reviewed separately before Step 9.
"""

from __future__ import annotations

import copy
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from .data import Windows, require_harth
from .train import class_weights

PRUNABLE_MODULES = (nn.Conv1d, nn.Conv2d, nn.Linear)   # nn.MultiheadAttention.out_proj is an nn.Linear
SEG_LEN = 16


# ------------------------------------------------------------------------------ scope

def _head(model: nn.Module) -> nn.Module | None:
    return getattr(model, "head", None)


def prunable_tensors(model: nn.Module, include_head: bool = False) -> list[tuple[str, nn.Module, str]]:
    """(full parameter name, owning module, parameter name) of every prunable weight tensor [plan 2.5, Q3]:
    Conv1d / Conv2d / Linear `weight`, LSTM `weight_ih_l*` / `weight_hh_l*`, attention `in_proj_weight`
    (out_proj is a Linear). Biases, normalization parameters and other parameters are never included.
    The classification head (`model.head`) is excluded unless include_head (PTQ quantizes it)."""
    head = _head(model)
    out = []
    for mname, mod in model.named_modules():
        if mod is head and not include_head:
            continue
        prefix = f"{mname}." if mname else ""
        if isinstance(mod, PRUNABLE_MODULES):
            out.append((prefix + "weight", mod, "weight"))
        elif isinstance(mod, nn.LSTM):
            for pname, _ in mod.named_parameters(recurse=False):
                if pname.startswith(("weight_ih_l", "weight_hh_l")):
                    out.append((prefix + pname, mod, pname))
        elif isinstance(mod, nn.MultiheadAttention) and mod.in_proj_weight is not None:
            out.append((prefix + "in_proj_weight", mod, "in_proj_weight"))
    return out


# ------------------------------------------------------------------------------ PTQ (primary)

def ptq_symmetric(model: nn.Module, bits: int) -> tuple[nn.Module, dict[str, int]]:
    """Per-tensor symmetric weight-only PTQ [plan 2.5, Q1] on a deepcopy: for every prunable tensor
    plus the head, s = max|W| / (2^(b-1) - 1), W_q = clamp(round(W / s), -(2^(b-1) - 1), 2^(b-1) - 1) * s.
    Stored back as float (fake quantization); biases, normalization parameters and activations untouched.
    Returns (quantized model in eval mode, {tensor name: bits})."""
    q = copy.deepcopy(model)
    qmax = 2 ** (bits - 1) - 1
    quantized = {}
    with torch.no_grad():
        for name, mod, pname in prunable_tensors(q, include_head=True):
            w = getattr(mod, pname)
            s = w.abs().max() / qmax
            if s > 0:
                w.copy_(torch.clamp(torch.round(w / s), -qmax, qmax) * s)
            quantized[name] = bits
    q.eval()
    return q, quantized


# ------------------------------------------------------------------------------ SNIP-score pruning

def snip_scores_from_batches(model: nn.Module, batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
                             loss_fn: nn.Module) -> dict[str, torch.Tensor]:
    """|w * g| per prunable tensor, g = dL/dw summed over the batches. Computed on a deepcopy in eval
    mode (no dropout, BatchNorm running statistics used and not updated), so `model` is left as is."""
    m = copy.deepcopy(model).eval()
    m.zero_grad(set_to_none=True)
    dev = next(m.parameters()).device
    # cuDNN is switched off here: its RNN backward only runs in training mode, so gradients through
    # TinyHAR's LSTM in eval mode fail on CUDA ("cudnn RNN backward can only be called in training
    # mode"). No effect on CPU.
    with torch.backends.cudnn.flags(enabled=False):
        for X, y in batches:
            loss_fn(m(X.to(dev)), y.to(dev)).backward()  # gradients accumulate = sum over batches
    return {name: (getattr(mod, p).detach() * getattr(mod, p).grad).abs()
            for name, mod, p in prunable_tensors(m)}


def snip_scores(model: nn.Module, train: Windows, n_batches: int = 10, seed: int = 42,
                batch_size: int = 256) -> dict[str, torch.Tensor]:
    """SNIP saliency |w * dL/dw| (Lee et al., 2019) on the trained model [plan 2.5, Q2]: L = the training
    loss (class-weighted CE, weights from `train`), gradients summed over `n_batches` fixed batches of
    `batch_size` HARTH training windows drawn without replacement with numpy seed `seed`."""
    require_harth(train)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train))[:n_batches * batch_size]
    X, y = torch.from_numpy(train.X), torch.from_numpy(train.y)
    batches = [(X[idx[i:i + batch_size]], y[idx[i:i + batch_size]]) for i in range(0, len(idx), batch_size)]
    dev = next(model.parameters()).device
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(train.y).to(dev))
    return snip_scores_from_batches(model, batches, loss_fn)


# ------------------------------------------------------------------------------ EG surrogate

def eg_attributions(model: nn.Module, X: torch.Tensor, target: torch.Tensor, baselines: torch.Tensor,
                    m: int, create_graph: bool, generator: torch.Generator | None = None) -> torch.Tensor:
    """Expected gradients of the target-class logit [plan 2.5], (B, 6, 128):
    mean over m draws (baseline b ~ the given baselines, alpha ~ U(0, 1)) of
    (x - b) * d f_target(b + alpha (x - b)) / dx. The m draws are one batch of m * B inputs.
    The draws come from `generator` (a CPU torch.Generator; seed it for reproducible draws).
    create_graph=True keeps the graph, so a loss on the result has gradients w.r.t. the parameters.
    The model's train/eval mode is the caller's choice and is not changed here.
    Attention runs on PyTorch's math SDPA kernel: the fused CPU/CUDA kernels (flash, memory-efficient)
    have no double backward, so a loss on these attributions (create_graph=True) would fail on
    TinyHAR's and HART's attention.
    cuDNN is switched off here: its RNN backward only runs in training mode, so gradients through
    TinyHAR's LSTM in eval mode fail on CUDA ("cudnn RNN backward can only be called in training
    mode"). No effect on CPU."""
    with sdpa_kernel(SDPBackend.MATH), torch.backends.cudnn.flags(enabled=False):
        return _eg(model, X, target, baselines, m, create_graph, generator)


def _eg(model, X, target, baselines, m, create_graph, generator):
    dev = X.device
    B = X.shape[0]
    target = torch.as_tensor(target, device=dev).long()
    idx = torch.randint(len(baselines), (m, B), generator=generator).to(dev)
    alpha = torch.rand((m, B) + (1,) * (X.dim() - 1), generator=generator).to(dev)
    b = baselines.to(dev)[idx]                                   # (m, B, 6, 128)
    delta = X.unsqueeze(0) - b
    xi = (b + alpha * delta).reshape((m * B,) + tuple(X.shape[1:])).detach().requires_grad_(True)
    out = model(xi).gather(1, target.repeat(m)[:, None]).sum()
    (g,) = torch.autograd.grad(out, xi, create_graph=create_graph)
    A = (delta * g.reshape(delta.shape)).mean(dim=0)
    return A if create_graph else A.detach()


def segment_attributions(A: torch.Tensor, seg_len: int = SEG_LEN) -> torch.Tensor:
    """(B, C, T) -> (B, C * T // seg_len): signed sum per (channel, segment), channel-major
    (feature = axis * 8 + segment) [plan 2.6]; the torch twin of lci.metric's aggregation."""
    B, C, T = A.shape
    if T % seg_len:
        raise ValueError(f"T={T} is not divisible by seg_len={seg_len}")
    return A.reshape(B, C, T // seg_len, seg_len).sum(dim=-1).reshape(B, C * (T // seg_len))


def unit_norm(a: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return a / a.norm(dim=1, keepdim=True).clamp_min(eps)


# ------------------------------------------------------------------------------ Park et al. (2020), adapted

def park_teacher_attributions(teacher: nn.Module, X: torch.Tensor, target: torch.Tensor, baselines: torch.Tensor,
                              m: int = 4, seed: int = 0, batch_size: int = 256) -> torch.Tensor:
    """The original model's fixed matching targets [plan 2.5 (a)]: unit-L2-norm 48-segment EG attributions
    (m draws) of the original model's top-1 class, computed once in eval mode without a graph. (N, 48)."""
    was_training = teacher.training
    teacher.eval()
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(0, len(X), batch_size):
        A = eg_attributions(teacher, X[i:i + batch_size], target[i:i + batch_size], baselines, m,
                            create_graph=False, generator=g)
        out.append(unit_norm(segment_attributions(A)))
    teacher.train(was_training)
    return torch.cat(out)


def park_am_distance(A_S_seg: torch.Tensor, A_T_unit: torch.Tensor) -> torch.Tensor:
    """L_AM = mean_i || a_S(x_i) / ||a_S(x_i)|| - a_T(x_i) ||_2, unsquared as in [Park Sec 3.2, Eq. 2];
    A_T_unit is already unit-normalised."""
    return (unit_norm(A_S_seg) - A_T_unit).norm(dim=1).mean()


def park_am_loss(student: nn.Module, X: torch.Tensor, target: torch.Tensor, A_T_unit: torch.Tensor,
                 baselines: torch.Tensor, m: int = 4, generator: torch.Generator | None = None) -> torch.Tensor:
    """Park's attribution-matching term on a batch [plan 2.5 (a)]: student 48-segment EG attributions
    (m draws, create_graph=True) of the original model's top-1 class vs the fixed teacher targets.
    Deviation (our choice, plan A.7): input attributions instead of Grad-CAM at the last conv layer.
    The fine-tuning loss is CE + gamma_P * this."""
    A_S = segment_attributions(eg_attributions(student, X, target, baselines, m, create_graph=True,
                                               generator=generator))
    return park_am_distance(A_S, A_T_unit.to(A_S.device))
