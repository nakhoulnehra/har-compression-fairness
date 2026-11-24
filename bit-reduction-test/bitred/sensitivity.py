"""Per-tensor, per-bit-width sensitivities: the only thing that differs between the allocation methods.

All three are measured on the fold's calibration windows (HARTH training windows only), with exactly one tensor
quantized (per-channel, b bits) and every other tensor in float32:

    xgq   (ours)  Omega_t(b) = 1 - LCI(original, single-tensor model) on the calibration SHAP subset
                  (48-segment SHAP of the original model's predicted-class logit, LCI pooled over the windows)
    zeroq_kl      Omega_t(b) = mean_x KL(p_original(x) || p_single-tensor(x))         [ZeroQ, Cai et al. 2020,
                  Sec 3.3; real calibration windows instead of ZeroQ's distilled data]
    hawq          Omega_t(b) = Tr(H_t) / n_t * ||Q_b(W_t) - W_t||^2, Tr(H_t) of the cross-entropy loss on the
                  calibration windows by Hutchinson's estimator                     [HAWQ-V2, Dong et al. 2020;
                  HAWQ-V3, Yao et al. 2021]
(The ZeroQ / HAWQ PDFs are not in papers/: verify the cited sections before submission.)
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from har.compression import attribution_grad_context

from .quant import quant_tensors, quantize_model, quantize_weight


@torch.no_grad()
def logits(model: nn.Module, X: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
    model.eval()
    return torch.cat([model(X[i:i + batch_size]) for i in range(0, len(X), batch_size)])


def kl_from_logits(z_ref: torch.Tensor, z: torch.Tensor) -> float:
    """mean_x KL(softmax(z_ref) || softmax(z)); float32 round-off below 0 is set to 0 (KL >= 0)."""
    lp_ref, lp = F.log_softmax(z_ref.double(), 1), F.log_softmax(z.double(), 1)
    return max(float((lp_ref.exp() * (lp_ref - lp)).sum(1).mean()), 0.0)


def kl_sensitivity(net0: nn.Module, X: torch.Tensor, bit_options: Sequence[int], granularity: str = "channel"
                   ) -> dict[str, dict[int, float]]:
    z0 = logits(net0, X)
    out = {}
    for n, _, _ in quant_tensors(net0):
        out[n] = {}
        for b in bit_options:
            out[n][int(b)] = kl_from_logits(z0, logits(quantize_model(net0, {n: b}, granularity), X))
    return out


def hutchinson_traces(model: nn.Module, loss_fn: Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor],
                      X: torch.Tensor, y: torch.Tensor, names: Sequence[str], n_iter: int, seed: int,
                      batch_size: int = 1024) -> dict[str, dict]:
    """Hutchinson estimate of Tr(H_t) of the mean loss for each named parameter t: E_v[v^T H_t v], v Rademacher
    per tensor (one draw for all tensors per iteration, the block-diagonal traces). Returns, per tensor,
    {"trace": mean, "trace_se": standard error over iterations, "n_iter"}. The model stays in eval mode."""
    model.eval()
    params = dict(model.named_parameters())
    P = [params[n] for n in names]
    gen = torch.Generator().manual_seed(int(seed))
    N = len(X)
    samples = {n: [] for n in names}
    with attribution_grad_context():
        for _ in range(int(n_iter)):
            vs = [(torch.randint(0, 2, p.shape, generator=gen) * 2 - 1).to(p.device, p.dtype) for p in P]
            hv = [torch.zeros_like(p) for p in P]
            for i in range(0, N, batch_size):
                xb, yb = X[i:i + batch_size], y[i:i + batch_size]
                loss = loss_fn(model, xb, yb) * (len(xb) / N)
                g = torch.autograd.grad(loss, P, create_graph=True)
                gv = sum((gi * vi).sum() for gi, vi in zip(g, vs))
                h = torch.autograd.grad(gv, P)
                hv = [a + b.detach() for a, b in zip(hv, h)]
            for n, v, h in zip(names, vs, hv):
                samples[n].append(float((v * h).sum()))
    out = {}
    for n in names:
        s = np.asarray(samples[n])
        out[n] = {"trace": float(s.mean()), "trace_se": float(s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else np.nan,
                  "n_iter": int(len(s))}
    return out


def ce_loss(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(model(X), y)


def hawq_sensitivity(net0: nn.Module, traces: Mapping[str, Mapping], bit_options: Sequence[int],
                     granularity: str = "channel") -> dict[str, dict[int, float]]:
    """max(Tr(H_t), 0) / n_t * ||Q_b(W_t) - W_t||^2 per tensor and bit width. HAWQ assumes positive curvature;
    the cross-entropy Hessian away from a minimum is not PSD, and a negative trace estimate would make fewer bits look
    better, so it is clipped at 0 (no measurable curvature = insensitive). The raw estimate is reported."""
    out = {}
    for n, mod, p in quant_tensors(net0):
        w = getattr(mod, p).detach()
        tr = max(float(traces[n]["trace"]), 0.0) / w.numel()
        out[n] = {int(b): tr * float(((quantize_weight(w, int(b), granularity) - w) ** 2).sum()) for b in bit_options}
    return out


def lci_pooled(A_ref: np.ndarray, A: np.ndarray, k: int = 10, omega: float = 0.5) -> dict:
    """LCI over all rows together (calibration windows come from many subjects, a few each), 48 segments."""
    from lci.metric import aggregate_time_segments, compute_lci
    return compute_lci(aggregate_time_segments(A_ref), aggregate_time_segments(A), k=k, omega=omega)


def check_omega(omega: Mapping[str, Mapping[int, float]], names: Sequence[str], bit_options: Sequence[int]) -> None:
    for n in names:
        for b in bit_options:
            v = omega[n][int(b)]
            if not np.isfinite(v):
                raise ValueError(f"sensitivity of {n} at {b} bits is not finite: {v}")
