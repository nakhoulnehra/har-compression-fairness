"""Candidate D (gap study v6): one-shot weight-only quantization, the strongest existing backprop-free route.
rtn   per-output-channel symmetric round-to-nearest (har.compression.ptq_per_channel; every prunable tensor + head)
gptq  GPTQ (Frantar et al., ICLR 2023; OBQ from Frantar et al. 2022): the same per-row grid (s_r = max|W_r| / qmax,
      fixed from the original), columns quantized in order, the error of each column spread over the remaining
      columns with the inverse input Hessian of the unit (H = X^T X on the calibration windows, the CURRENT model's
      inputs, units in forward order; dampening 1 % of the mean diagonal). The head (8 x 56) uses RTN.
"""

from __future__ import annotations

import copy

import numpy as np
import torch

import gap_study_v3.baselines as GB
from har.compression import prunable_tensors, ptq_per_channel

GPTQ_DAMP = 0.01
# p8: the common practice of keeping the first layer, tiny sensitive tensors and the classifier at 8 bits (e.g.
# Zhou et al. 2016 DoReFa, Choi et al. 2018 PACT keep first / last layers in high precision); here conv0 (140 weights),
# attn_score (56) and the head (448): +0.47 KB over uniform 2-bit (20.7 KB), disclosed.
P8 = ("convs.0.weight", "attn_score.weight", "head.weight")


def bits_of(name: str, bits: int, protect: bool) -> int:
    return 8 if (protect and name.endswith(P8)) else bits


def gptq_matrix(W: np.ndarray, H: np.ndarray, S: np.ndarray, qmax: int, damp: float = GPTQ_DAMP) -> np.ndarray:
    """W (r, c), H (c, c), S (r, c) per-entry scale (row x tensor group). Returns the quantized W."""
    W = W.astype(np.float64).copy()
    H = H.astype(np.float64).copy()
    dead = np.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    H += damp * np.mean(np.diag(H)) * np.eye(len(H))
    Hinv = np.linalg.cholesky(np.linalg.inv(H)).T          # upper Cholesky factor of H^-1 (GPTQ, Alg. 1)
    Q = np.zeros_like(W)
    for j in range(W.shape[1]):
        w = W[:, j]
        q = np.clip(np.round(w / S[:, j]), -qmax, qmax) * S[:, j]
        Q[:, j] = q
        err = (w - q) / Hinv[j, j]
        W[:, j + 1:] -= np.outer(err, Hinv[j, j + 1:])
    return Q


@torch.no_grad()
def gptq_quantize(original: torch.nn.Module, bits: int, X_cal: torch.Tensor, protect: bool = False) -> torch.nn.Module:
    net = copy.deepcopy(original).eval()
    GB.check_units_cover(net)
    for kind, name in GB.UNITS:
        qmax = 2 ** (bits_of(GB.unit_tensors(kind, name)[0], bits, protect) - 1) - 1
        W, _ = GB.unit_weight(net, kind, name)
        H, _, _ = GB.unit_grams(net, None, X_cal, kind, name, False)
        params = dict(net.named_parameters())
        S = np.zeros(tuple(W.shape))
        c0 = 0
        for t in GB.unit_tensors(kind, name):
            Wt = params[t].detach().reshape(params[t].shape[0], -1).cpu().numpy()
            s = np.abs(Wt).max(1) / qmax
            s[s == 0] = 1.0
            S[:, c0:c0 + Wt.shape[1]] = s[:, None]
            c0 += Wt.shape[1]
        Qm = gptq_matrix(W.cpu().numpy(), H.cpu().numpy(), S, qmax)
        GB.set_unit_weight(net, kind, name, torch.from_numpy(Qm).to(W.device, W.dtype), None)
    head = [n for n, _, _ in prunable_tensors(net, include_head=True) if n.endswith("head.weight")]
    net, _ = ptq_per_channel(net, bits_of(head[0], bits, protect), only=head)
    return net.eval()


def build_quant(method: str, net0, bits: int, X_cal: torch.Tensor):
    if method == "rtn":
        net, _ = ptq_per_channel(net0, bits)
    elif method == "gptq":
        net = gptq_quantize(net0, bits, X_cal)
    elif method == "gptq_p8":
        net = gptq_quantize(net0, bits, X_cal, protect=True)
    else:
        raise ValueError(method)
    net.prune_masks = {}
    return net.eval()


# ------------------------------------------------------------------------------ same-size maximum: QAT + fmkd

def _qtensors(net):
    return [(n, getattr(m, p)) for n, m, p in prunable_tensors(net, include_head=True)]


@torch.no_grad()
def _project(w: torch.Tensor, qmax: int) -> torch.Tensor:
    W = w.reshape(w.shape[0], -1)
    s = W.abs().amax(1, keepdim=True) / qmax
    s = torch.where(s > 0, s, torch.ones_like(s))
    return (torch.clamp(torch.round(W / s), -qmax, qmax) * s).reshape(w.shape)


def qat_fmkd(net, ctx, bits: int, epochs: int, lr: float, seed: int, device: str, log=print, protect: bool = False):
    """Quantization-aware training with the straight-through estimator (latent fp32 weights; each step quantizes
    them per row in place for the forward / backward, restores the latent copy, then steps on it; Courbariaux et al.
    2015 / Jacob et al. 2018) with the v4 fmkd objective (KD T 3 alpha 0.7 + function matching on mixup blends).
    The final weights are projected on the grid. Batch 256, Adam."""
    import time
    import torch.nn as nn
    import gap_study_v4.run_v4 as V4
    from har.compression import KD, kd_kl_term, predict_logits
    from har.train import class_weights, pooled_macro_f1, predict_proba
    dev = torch.device(device)
    net.to(dev)
    torch.manual_seed(seed)
    teacher = copy.deepcopy(ctx.net0).to(dev).eval()
    for q in teacher.parameters():
        q.requires_grad_(False)
    Xtr = torch.from_numpy(ctx.tr.X).to(dev)
    ytr = torch.from_numpy(ctx.tr.y).to(dev)
    zt = torch.from_numpy(predict_logits(ctx.net0, ctx.tr.X, ctx.dev)).to(dev)
    fm = V4.fm_extra_loss(teacher, Xtr, seed)
    ce_fn = nn.CrossEntropyLoss(weight=class_weights(ctx.tr.y).to(dev))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    qs = _qtensors(net)
    qm = [2 ** (bits_of(n, bits, protect) - 1) - 1 for n, _ in qs]
    hist, t_all = [], time.perf_counter()
    for ep in range(epochs):
        t0 = time.perf_counter()
        perm = torch.randperm(len(Xtr), generator=g).to(dev)
        for i in range(0, len(Xtr), 256):
            idx = perm[i:i + 256]
            if len(idx) < 2:
                continue
            latent = [w.detach().clone() for _, w in qs]
            with torch.no_grad():
                for (_, w), qmax in zip(qs, qm):
                    w.copy_(_project(w, qmax))
            opt.zero_grad(set_to_none=True)
            net.train()
            z = net(Xtr[idx])
            loss = KD["alpha"] * kd_kl_term(z, zt[idx], KD["T"]) * KD["T"] ** 2 + (1 - KD["alpha"]) * ce_fn(z, ytr[idx])
            net.eval()
            loss = loss + fm(net, idx)
            loss.backward()
            with torch.no_grad():
                for (_, w), lt in zip(qs, latent):
                    w.copy_(lt)
            opt.step()
        with torch.no_grad():
            saved = [w.detach().clone() for _, w in qs]
            for (_, w), qmax in zip(qs, qm):
                w.copy_(_project(w, qmax))
        pred = predict_proba(net, ctx.va.X, device).argmax(1)
        f1 = pooled_macro_f1(ctx.va.y, pred)
        with torch.no_grad():
            for (_, w), sv in zip(qs, saved):
                w.copy_(sv)
        hist.append({"epoch": ep, "val_macro_f1_pooled7": f1, "seconds": time.perf_counter() - t0})
        log(f"    qat epoch {ep + 1}/{epochs} val F1 {f1:.4f} {time.perf_counter() - t0:.1f}s")
    with torch.no_grad():
        for (_, w), qmax in zip(qs, qm):
            w.copy_(_project(w, qmax))
    net.eval()
    return net, {"history": hist, "seconds": time.perf_counter() - t_all, "epochs_run": epochs}
