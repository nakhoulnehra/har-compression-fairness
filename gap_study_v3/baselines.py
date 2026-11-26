# Moved here unchanged from gap_study/baselines.py (agent B, 2026-11-26): the no-retraining baselines of the confirmed gap.
"""Phase 2 baselines for post-training (no retraining) unstructured pruning of the frozen TinyHAR.

Existing (round 4, reused unchanged from har/xgp3.py, har/xgp4.py):
    l1_recon      uniform per-tensor budget, K = 5 nested magnitude steps, ridge layer-wise reconstruction of the
                  Conv / Linear modules only (LSTM and attention in/out projections are pruned without refit)
    lamp_recon    the same with LAMP keep counts (Lee et al. 2021) bounded so the non-refit tensors stay uniform
New here (all prunable tensors are refit, so the jury's "why not reconstruct every layer?" is answered):
    l1_fullrecon      l1_recon, but the reconstruction refits EVERY linear map: Conv / Linear as before, plus the
                      attention in_proj (tokens -> q, k, v), out_proj (attention output -> out) and the LSTM gates
                      ([x_t, h_{t-1}, 1] -> W_ih x_t + W_hh h_{t-1} + b), each by the same row-wise ridge on the
                      pruned model's own inputs against the original model's outputs (SparseGPT / OBC style
                      error-correcting order).
    wanda_fullrecon   Wanda selection (Sun et al. 2024, |W_ij| * ||X_j||_2, ranked within each tensor) recomputed on
                      the current model at each of the K = 5 steps, with the full reconstruction after every step.
    sparsegpt         SparseGPT (Frantar & Alistarh 2023, Alg. 1): one shot, layers in forward order on the partly
                      pruned model's inputs, OBS saliency w^2 / [chol(H^-1)]_jj^2 for the mask (one block = all
                      columns, so every tensor keeps exactly the uniform count), OBS weight updates, 1 % dampening;
                      biases unchanged. The LSTM's W_ih and W_hh are processed as one matrix on [x_t, h_{t-1}].
Every method keeps exactly the same number of weights per tensor at a level (uniform counts), except lamp_recon
(same total). No gradient training anywhere.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from har.compression import head_unpruned, prunable_tensors, size_record
from har.xgp3 import _as_rows, _out_rows, layerwise_reconstruct, ridge_refit_rows
from har.xgp4 import (ROUND4, bounded_budget, iterative_prune_budget, uniform_keep_counts,
                      unreconstructed_tensors)

P = "backbone."
MHA = P + "channel_attn.self_attn"
LSTM = P + "lstm"
# forward order of every linear map of TinyHAR (the head is never pruned)
UNITS = (
    ("lin", P + "convs.0"), ("lin", P + "convs.3"), ("lin", P + "convs.6"), ("lin", P + "convs.9"),
    ("mha_in", MHA), ("mha_out", MHA), ("lin", P + "channel_attn.linear1"), ("lin", P + "channel_attn.linear2"),
    ("lin", P + "fusion.0"), ("lstm", LSTM), ("lin", P + "attn_score"),
)
DAMP = ROUND4["damp"]          # 0.01, round 3's ridge dampening
K_STEPS = ROUND4["K"]          # 5
SPARSEGPT_DAMP = 0.01          # SparseGPT's default percdamp


def unit_tensors(kind: str, mod_name: str) -> list[str]:
    if kind == "lin":
        return [f"{mod_name}.weight"]
    if kind == "mha_in":
        return [f"{mod_name}.in_proj_weight"]
    if kind == "mha_out":
        return [f"{mod_name}.out_proj.weight"]
    if kind == "lstm":
        return [f"{mod_name}.weight_ih_l0", f"{mod_name}.weight_hh_l0"]
    raise ValueError(kind)


def check_units_cover(model: nn.Module) -> None:
    names = {n for n, _, _ in prunable_tensors(model)}
    covered = [t for k, m in UNITS for t in unit_tensors(k, m)]
    if set(covered) != names or len(covered) != len(names):
        raise RuntimeError(f"UNITS do not cover the prunable tensors: {sorted(names ^ set(covered))}")


def attention_output(mha: nn.MultiheadAttention, x: torch.Tensor) -> torch.Tensor:
    """The input of out_proj for self-attention on x (N, L, E), batch_first, 1 head or more, no masks."""
    E, h = mha.embed_dim, mha.num_heads
    qkv = F.linear(x, mha.in_proj_weight, mha.in_proj_bias)
    q, k, v = qkv.split(E, dim=-1)
    N, L, _ = x.shape
    q, k, v = (t.reshape(N, L, h, E // h).transpose(1, 2) for t in (q, k, v))
    a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(E // h), dim=-1)
    return (a @ v).transpose(1, 2).reshape(N, L, E)


def _forward_capture(net: nn.Module, xb: torch.Tensor, kind: str, mod_name: str) -> dict:
    """Run net on xb (eval, grad enabled so TransformerEncoderLayer takes its slow path) and capture what the unit
    needs: module input (and LSTM output)."""
    mod = dict(net.named_modules())[mod_name]
    cap = {}
    hs = [mod.register_forward_pre_hook(lambda _m, a: cap.__setitem__("x", a[0].detach()))]
    if kind in ("lin", "lstm"):
        hs.append(mod.register_forward_hook(lambda _m, _i, o: cap.__setitem__(
            "y", (o[0] if isinstance(o, tuple) else o).detach())))
    try:
        with torch.enable_grad():
            net(xb)
    finally:
        for h in hs:
            h.remove()
    cap["mod"] = mod
    return cap


def unit_rows(net: nn.Module, xb: torch.Tensor, kind: str, mod_name: str, own_output: bool = True):
    """(X rows in the unit's column order, Y rows = the unit's output computed from those inputs by `net`)."""
    c = _forward_capture(net, xb, kind, mod_name)
    mod, x = c["mod"], c["x"]
    if kind == "lin":
        return _as_rows(mod, x), _out_rows(mod, c["y"])
    if kind == "mha_in":
        Xr = x.reshape(-1, x.shape[-1])
        return Xr, F.linear(Xr, mod.in_proj_weight, mod.in_proj_bias)
    if kind == "mha_out":
        o = attention_output(mod, x).reshape(-1, x.shape[-1])
        return o, F.linear(o, mod.out_proj.weight, mod.out_proj.bias)
    if kind == "lstm":
        h = c["y"]
        hprev = torch.cat([torch.zeros_like(h[:, :1]), h[:, :-1]], 1)
        Xr = torch.cat([x, hprev], -1).reshape(-1, x.shape[-1] + h.shape[-1])
        W = torch.cat([mod.weight_ih_l0, mod.weight_hh_l0], 1)
        return Xr, F.linear(Xr, W, mod.bias_ih_l0 + mod.bias_hh_l0)
    raise ValueError(kind)


def unit_weight(net: nn.Module, kind: str, mod_name: str) -> tuple[torch.Tensor, torch.Tensor | None]:
    """(W as (out, in) matrix in the unit's column order, combined bias or None)."""
    mod = dict(net.named_modules())[mod_name]
    if kind == "lin":
        return mod.weight.detach().reshape(mod.weight.shape[0], -1), (None if mod.bias is None else mod.bias.detach())
    if kind == "mha_in":
        return mod.in_proj_weight.detach(), mod.in_proj_bias.detach()
    if kind == "mha_out":
        return mod.out_proj.weight.detach(), mod.out_proj.bias.detach()
    if kind == "lstm":
        return (torch.cat([mod.weight_ih_l0, mod.weight_hh_l0], 1).detach(),
                (mod.bias_ih_l0 + mod.bias_hh_l0).detach())
    raise ValueError(kind)


@torch.no_grad()
def set_unit_weight(net: nn.Module, kind: str, mod_name: str, W: torch.Tensor, b: torch.Tensor | None) -> None:
    mod = dict(net.named_modules())[mod_name]
    if kind == "lin":
        mod.weight.copy_(W.reshape(mod.weight.shape).to(mod.weight.dtype))
        if b is not None:
            mod.bias.copy_(b.to(mod.bias.dtype))
    elif kind == "mha_in":
        mod.in_proj_weight.copy_(W.to(mod.in_proj_weight.dtype))
        if b is not None:
            mod.in_proj_bias.copy_(b.to(mod.in_proj_bias.dtype))
    elif kind == "mha_out":
        mod.out_proj.weight.copy_(W.to(mod.out_proj.weight.dtype))
        if b is not None:
            mod.out_proj.bias.copy_(b.to(mod.out_proj.bias.dtype))
    elif kind == "lstm":
        I = mod.weight_ih_l0.shape[1]
        mod.weight_ih_l0.copy_(W[:, :I].to(mod.weight_ih_l0.dtype))
        mod.weight_hh_l0.copy_(W[:, I:].to(mod.weight_hh_l0.dtype))
        if b is not None:   # keep b_hh, put the change into b_ih (n_nonzero of the biases unchanged)
            mod.bias_ih_l0.copy_((b - mod.bias_hh_l0.detach()).to(mod.bias_ih_l0.dtype))
    if hasattr(mod, "flatten_parameters") and kind == "lstm":
        mod.flatten_parameters()


def unit_mask(net: nn.Module, kind: str, mod_name: str) -> torch.Tensor:
    masks = getattr(net, "prune_masks", None) or {}
    W, _ = unit_weight(net, kind, mod_name)
    parts = []
    for t in unit_tensors(kind, mod_name):
        p = dict(net.named_parameters())[t]
        m = masks.get(t, torch.ones_like(p, dtype=torch.bool)).to(p.device)
        parts.append(m.reshape(p.shape[0], -1))
    return torch.cat(parts, 1)


def unit_grams(pruned: nn.Module, original: nn.Module | None, X_cal: torch.Tensor, kind: str, mod_name: str,
               with_bias: bool, batch_size: int = 256):
    """H = [X 1]^T [X 1] of the pruned model's unit inputs; G = [X 1]^T Y_orig and YY = ||Y_orig||^2 with the
    original's unit output on the original's inputs (None if original is None)."""
    H = G = None
    YY = 0.0
    for i in range(0, len(X_cal), batch_size):
        xb = X_cal[i:i + batch_size]
        Xr, _ = unit_rows(pruned, xb, kind, mod_name)
        Xr = Xr.double()
        if with_bias:
            Xr = torch.cat([Xr, torch.ones(len(Xr), 1, dtype=Xr.dtype, device=Xr.device)], 1)
        H = Xr.T @ Xr if H is None else H + Xr.T @ Xr
        if original is not None:
            _, Yo = unit_rows(original, xb, kind, mod_name)
            Yo = Yo.double()
            G = Xr.T @ Yo if G is None else G + Xr.T @ Yo
            YY += float((Yo * Yo).sum())
    return H, G, YY


# ------------------------------------------------------------------------------ full reconstruction

def full_reconstruct(pruned: nn.Module, original: nn.Module, X_cal: torch.Tensor, damp: float = DAMP,
                     batch_size: int = 256) -> dict:
    """In place on `pruned`: every unit of UNITS in forward order is refit by row-wise ridge (masks fixed) so its
    output on the pruned model's own inputs matches the original unit's output on the original's inputs."""
    pruned.eval()
    original.eval()
    nnz0 = prunable_nonzero(pruned)
    log = {"refit": []}
    for kind, name in UNITS:
        W, b = unit_weight(pruned, kind, name)
        has_bias = b is not None
        H, G, YY = unit_grams(pruned, original, X_cal, kind, name, has_bias, batch_size)
        keep = unit_mask(pruned, kind, name)
        theta = (torch.cat([W, b[:, None]], 1) if has_bias else W).double()
        err = lambda th: float(((th @ H @ th.T).trace() - 2 * (th * G.T).sum() + YY) / max(YY, 1e-30))
        sol = ridge_refit_rows(H, G, keep, has_bias, damp, old=theta)
        d = keep.shape[1]
        Wn = sol[:, :d].to(W.dtype)
        Wn = torch.where(keep & (Wn == 0), W, Wn) * keep.to(W.dtype)
        bn = None
        if has_bias:
            bn = sol[:, -1].to(b.dtype)
            bn = torch.where(bn == 0, b, bn)
        e0, e1 = err(theta), err(sol)
        set_unit_weight(pruned, kind, name, Wn, bn)
        log["refit"].append({"unit": f"{kind}:{name}", "rel_error_before": e0, "rel_error_after": e1})
    _check_masks_fixed(pruned, nnz0)
    return log


def prunable_nonzero(net: nn.Module) -> int:
    """Non-zero count over the prunable weight tensors (biases are outside the pruning budget)."""
    return int(sum(int((getattr(m, p).detach() != 0).sum()) for _, m, p in prunable_tensors(net)))


def _check_masks_fixed(net: nn.Module, nnz0: int) -> None:
    nnz1 = prunable_nonzero(net)
    if nnz1 != nnz0:
        raise RuntimeError(f"reconstruction changed the prunable non-zero count ({nnz0} -> {nnz1})")
    params = dict(net.named_parameters())
    for n, mk in (getattr(net, "prune_masks", None) or {}).items():
        if torch.any(params[n].detach()[~mk.to(params[n].device)] != 0):
            raise RuntimeError(f"reconstruction wrote a pruned weight in {n}")


# ------------------------------------------------------------------------------ Wanda scores

def wanda_scores(net: nn.Module, X_cal: torch.Tensor, batch_size: int = 256) -> dict[str, torch.Tensor]:
    """|W_ij| * ||X_j||_2 per prunable tensor, X = the current model's unit inputs on the calibration windows."""
    out = {}
    for kind, name in UNITS:
        H, _, _ = unit_grams(net, None, X_cal, kind, name, False, batch_size)
        norm = torch.sqrt(torch.diagonal(H).clamp_min(0)).float()
        W, _ = unit_weight(net, kind, name)
        S = W.abs().float() * norm[None, :].to(W.device)
        c0 = 0
        for t in unit_tensors(kind, name):
            p = dict(net.named_parameters())[t]
            w = p.numel() // p.shape[0]
            out[t] = S[:, c0:c0 + w].reshape(p.shape).clone()
            c0 += w
    return out


# ------------------------------------------------------------------------------ SparseGPT

@torch.no_grad()
def sparsegpt_matrix(W: torch.Tensor, H: torch.Tensor, prune_counts: list[tuple[int, int, int]],
                     percdamp: float = SPARSEGPT_DAMP) -> tuple[torch.Tensor, torch.Tensor]:
    """SparseGPT (Frantar & Alistarh 2023, Alg. 1) on W (rows, cols) with Hessian H = X^T X (cols, cols), one block
    of all columns. prune_counts = [(col_start, col_end, k)]: exactly k weights are pruned inside each column group
    (one group per original tensor). Returns (new W, keep mask)."""
    W = W.clone().double()
    H = H.clone().double()
    cols = W.shape[1]
    dead = torch.diagonal(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H += percdamp * torch.mean(torch.diagonal(H)) * torch.eye(cols, dtype=H.dtype, device=H.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    sal = W ** 2 / torch.diagonal(Hinv)[None, :] ** 2
    keep = torch.ones_like(W, dtype=torch.bool)
    for a, b, k in prune_counts:
        s = sal[:, a:b].flatten().cpu()
        idx = torch.argsort(s, stable=True)[:k]
        km = torch.ones(s.numel(), dtype=torch.bool)
        km[idx] = False
        keep[:, a:b] = km.reshape(W.shape[0], b - a).to(W.device)
    for i in range(cols):
        w = W[:, i]
        d = Hinv[i, i]
        q = torch.where(keep[:, i], w, torch.zeros_like(w))
        err = (w - q) / d
        W[:, i:] -= err[:, None] @ Hinv[i, i:][None, :]
        W[:, i] = q
    W = W * keep
    return W, keep


def sparsegpt_prune(original: nn.Module, sparsity: float, X_cal: torch.Tensor, batch_size: int = 256,
                    log: Callable[[str], None] = print) -> nn.Module:
    import copy
    net = copy.deepcopy(original).eval()
    check_units_cover(net)
    uni = uniform_keep_counts(net, sparsity)
    masks = {}
    for kind, name in UNITS:
        W, b = unit_weight(net, kind, name)
        H, _, _ = unit_grams(net, None, X_cal, kind, name, False, batch_size)
        groups, c0 = [], 0
        for t in unit_tensors(kind, name):
            p = dict(net.named_parameters())[t]
            w = p.numel() // p.shape[0]
            groups.append((c0, c0 + w, p.numel() - uni[t]))
            c0 += w
        Wn, keep = sparsegpt_matrix(W, H, groups)
        set_unit_weight(net, kind, name, Wn.to(W.dtype), None)
        for (a, bb, _), t in zip(groups, unit_tensors(kind, name)):
            p = dict(net.named_parameters())[t]
            masks[t] = keep[:, a:bb].reshape(p.shape).cpu()
    net.prune_masks = masks
    net.prune_sparsity = sparsity
    for t, m in masks.items():
        if int(m.sum()) != uni[t]:
            raise RuntimeError(f"sparsegpt {t}: kept {int(m.sum())}, uniform {uni[t]}")
    log(f"      sparsegpt one shot: n_nonzero {size_record(net)['n_nonzero']}")
    return net


# ------------------------------------------------------------------------------ method dispatch

BASELINES = ("l1_recon", "lamp_recon", "l1_fullrecon", "wanda_fullrecon", "sparsegpt")


def build(method: str, net0: nn.Module, level: float, X_cal: torch.Tensor, damp: float = DAMP,
          log: Callable[[str], None] = print) -> nn.Module:
    """A pruned copy of net0 (net0 unchanged) with prune_masks; no gradient training."""
    net0 = net0.eval()
    check_units_cover(net0)
    unrecon = unreconstructed_tensors(net0, X_cal[:4])
    if method == "l1_recon":
        keep = uniform_keep_counts(net0, level)
        net, _ = iterative_prune_budget(net0, keep, K=K_STEPS, beta=0.0, importance_fn=None,
                                        recon_fn=lambda n: layerwise_reconstruct(n, net0, X_cal, damp=damp), log=log)
    elif method == "lamp_recon":
        keep = bounded_budget(net0, level, unrecon, "lamp")
        net, _ = iterative_prune_budget(net0, keep, K=K_STEPS, beta=0.0, importance_fn=None,
                                        recon_fn=lambda n: layerwise_reconstruct(n, net0, X_cal, damp=damp), log=log)
    elif method == "l1_fullrecon":
        keep = uniform_keep_counts(net0, level)
        net, _ = iterative_prune_budget(net0, keep, K=K_STEPS, beta=0.0, importance_fn=None,
                                        recon_fn=lambda n: full_reconstruct(n, net0, X_cal, damp=damp), log=log)
    elif method == "wanda_fullrecon":
        keep = uniform_keep_counts(net0, level)
        net, _ = iterative_prune_budget(net0, keep, K=K_STEPS, beta=1.0,
                                        importance_fn=lambda n: wanda_scores(n, X_cal),
                                        recon_fn=lambda n: full_reconstruct(n, net0, X_cal, damp=damp), log=log)
    elif method == "sparsegpt":
        net = sparsegpt_prune(net0, level, X_cal, log=log)
    else:
        raise ValueError(method)
    if not head_unpruned(net):
        raise RuntimeError(f"{method} pruned the head")
    net.prune_sparsity = level
    return net.eval()
