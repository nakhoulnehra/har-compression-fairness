"""Gap study v5: the backprop-free post-training pruning baselines that v3 / v4 listed as "considered but not run"
(v3 GAP_REPORT.md Sec. 5), plus BatchNorm re-estimation. EXISTING techniques only, no new method, no gradient training.

    obc              Optimal Brain Compression (Frantar, Singh & Alistarh, NeurIPS 2022, Sec. 3-4, pruning): exact
                     greedy OBS per row (remove the weight with the smallest w_q^2 / [H^-1]_qq, update the remaining
                     weights by -w_q / [H^-1]_qq * H^-1[:, q] and downdate H^-1), with the global selection across
                     rows of a layer (the next removal is the cheapest one over all rows); exact per-tensor counts
                     (uniform, as every v3 baseline). Layers in forward order on the partly pruned model's inputs,
                     H = X^T X + 1 % mean-diagonal dampening (SparseGPT / OBC default).
    obc_fullrecon    obc's masks + v3's full layer-wise ridge refit against the original model's outputs
                     (gap_study_v3.baselines.full_reconstruct, unchanged; the refit every v3 "fullrecon" method uses).
    sparsegpt_fullrecon  v3's SparseGPT masks (percdamp 0.01) + the same full refit. v3's plain SparseGPT keeps its own
                     OBS weights and collapses (F1 <= 0.39); this checks whether its masks are the problem.
    dsnot_fullrecon  DSnoT (Zhang et al., ICLR 2024, Alg. 1, training-free mask refinement): from Wanda's masks
                     (wanda_fullrecon, K = 5), per output row, repeatedly grow the pruned weight that most reduces the
                     expected reconstruction error Delta = sum_{j pruned} w_j E[x_j] and prune the kept weight with the
                     smallest Wanda score |w_j| ||x_j|| among those that do not push Delta further in the same
                     direction; a swap is kept only if |Delta| decreases; at most 50 cycles. Per-row and per-tensor
                     counts unchanged. Then the full refit.
    l1_fullrecon_bnrecal  l1_fullrecon + BatchNorm statistics re-estimated on the calibration windows (cumulative
                     average, train-mode forward of the BatchNorm layers only), the standard training-free step after
                     pruning (e.g. Li et al. 2020, EagleEye, Sec. 3.2 "adaptive BN").
    l1_fullrecon_k20  l1_fullrecon with K = 20 nested magnitude steps (v3: K = 5): more gradual one-shot pruning.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

import gap_study_v3.baselines as GB
from har.compression import head_unpruned, prunable_tensors
from har.xgp4 import iterative_prune_budget, uniform_keep_counts

OBC_DAMP = 0.01
DSNOT_CYCLES = 50


# ------------------------------------------------------------------------------ OBC

def obc_matrix(W: np.ndarray, H: np.ndarray, groups: list[tuple[int, int, int]], damp: float = OBC_DAMP):
    """Exact OBS with global greedy selection over the rows of W (rows, cols). groups = [(c0, c1, k_remove)]: exactly
    k_remove weights are removed inside columns c0:c1 (summed over rows). Returns (new W, keep mask)."""
    W = W.astype(np.float64).copy()
    H = H.astype(np.float64).copy()
    r, d = W.shape
    dead = np.diag(H) == 0
    H[dead, dead] = 1.0
    H += damp * np.mean(np.diag(H)) * np.eye(d)
    Hinv0 = np.linalg.inv(H)
    Hinv = np.repeat(Hinv0[None], r, axis=0)                     # one inverse per row
    alive = np.ones((r, d), dtype=bool)
    col_group = np.zeros(d, dtype=int)
    quota = []
    for gi, (a, b, k) in enumerate(groups):
        col_group[a:b] = gi
        quota.append(int(k))
    quota = np.array(quota)
    eligible_col = quota[col_group] > 0

    def row_loss(i):
        L = W[i] ** 2 / np.maximum(np.diagonal(Hinv[i]), 1e-300)
        L[~alive[i] | ~eligible_col] = np.inf
        return L

    loss = np.stack([row_loss(i) for i in range(r)])
    best_c = loss.argmin(1)
    best_v = loss[np.arange(r), best_c]
    for _ in range(int(quota.sum())):
        i = int(best_v.argmin())
        q = int(best_c[i])
        if not np.isfinite(best_v[i]):
            raise RuntimeError("OBC ran out of eligible weights")
        Hi = Hinv[i]
        hqq = Hi[q, q]
        W[i] -= (W[i, q] / hqq) * Hi[:, q]
        W[i, q] = 0.0
        Hinv[i] = Hi - np.outer(Hi[:, q], Hi[q, :]) / hqq
        alive[i, q] = False
        g = col_group[q]
        quota[g] -= 1
        if quota[g] == 0:                                        # this tensor is done: recompute every row
            eligible_col = quota[col_group] > 0
            loss = np.stack([row_loss(j) for j in range(r)])
            best_c = loss.argmin(1)
            best_v = loss[np.arange(r), best_c]
        else:
            L = row_loss(i)
            best_c[i] = L.argmin()
            best_v[i] = L[best_c[i]]
    W[~alive] = 0.0
    return W, alive


def obc_prune(original: nn.Module, sparsity: float, X_cal: torch.Tensor, damp: float = OBC_DAMP,
              keep_counts: dict | None = None) -> nn.Module:
    """keep_counts: per-tensor keep counts (default uniform at `sparsity`). Weights that are already zero (an earlier
    pruning step) cost 0 and are removed first, so a nested schedule never revives them (checked by the caller)."""
    net = copy.deepcopy(original).eval()
    GB.check_units_cover(net)
    uni = keep_counts if keep_counts is not None else uniform_keep_counts(net, sparsity)
    masks = {}
    for kind, name in GB.UNITS:
        W, _ = GB.unit_weight(net, kind, name)
        H, _, _ = GB.unit_grams(net, None, X_cal, kind, name, False)
        groups, c0 = [], 0
        params = dict(net.named_parameters())
        for t in GB.unit_tensors(kind, name):
            w = params[t].numel() // params[t].shape[0]
            groups.append((c0, c0 + w, params[t].numel() - uni[t]))
            c0 += w
        Wn, keep = obc_matrix(W.detach().cpu().numpy(), H.detach().cpu().numpy(), groups, damp)
        GB.set_unit_weight(net, kind, name, torch.from_numpy(Wn).to(W.device, W.dtype), None)
        params = dict(net.named_parameters())
        for (a, b, _), t in zip(groups, GB.unit_tensors(kind, name)):
            masks[t] = torch.from_numpy(keep[:, a:b].copy()).reshape(params[t].shape)
    net.prune_masks = masks
    net.prune_sparsity = sparsity
    for t, m in masks.items():
        if int(m.sum()) != uni[t]:
            raise RuntimeError(f"obc {t}: kept {int(m.sum())}, uniform {uni[t]}")
    return net


# ------------------------------------------------------------------------------ DSnoT

def unit_input_stats(net: nn.Module, X_cal: torch.Tensor, kind: str, name: str, batch_size: int = 256):
    """(mean, l2 norm) of each input column of the unit on the calibration windows (current model's inputs)."""
    s = sq = None
    n = 0
    for i in range(0, len(X_cal), batch_size):
        Xr, _ = GB.unit_rows(net, X_cal[i:i + batch_size], kind, name)
        Xr = Xr.double()
        s = Xr.sum(0) if s is None else s + Xr.sum(0)
        sq = (Xr * Xr).sum(0) if sq is None else sq + (Xr * Xr).sum(0)
        n += len(Xr)
    return (s / n).detach().cpu().numpy(), torch.sqrt(sq).detach().cpu().numpy()


def dsnot_matrix(W0: np.ndarray, keep: np.ndarray, mu: np.ndarray, xnorm: np.ndarray,
                 groups: list[tuple[int, int]], cycles: int = DSNOT_CYCLES):
    """DSnoT swaps within each (row, column group): W0 dense weights, keep initial mask. Returns the new mask."""
    keep = keep.copy()
    contrib = W0 * mu[None, :]                    # w_j E[x_j]
    wanda = np.abs(W0) * xnorm[None, :]
    r = W0.shape[0]
    n_swaps = 0
    for a, b in groups:
        for i in range(r):
            kp = keep[i, a:b]
            c = contrib[i, a:b]
            sc = wanda[i, a:b]
            if kp.all() or (~kp).all():
                continue
            delta = float(c[~kp].sum())          # expected error of row i from this group's pruned weights
            for _ in range(cycles):
                if abs(delta) < 1e-12:
                    break
                sgn = np.sign(delta)
                cand_g = np.where(~kp, sgn * c, -np.inf)        # growing j changes delta by -c_j
                g = int(cand_g.argmax())
                if not np.isfinite(cand_g[g]) or cand_g[g] <= 0:
                    break
                d1 = delta - c[g]
                s1 = np.sign(d1) if d1 != 0 else sgn
                ok = kp & (s1 * c <= 0)                          # pruning p adds c_p: must not push d1 further
                ok[g] = False
                if not ok.any():
                    break
                p = int(np.where(ok, sc, np.inf).argmin())
                d2 = d1 + c[p]
                if abs(d2) >= abs(delta):
                    break
                kp[g], kp[p] = True, False
                delta = d2
                n_swaps += 1
            keep[i, a:b] = kp
    return keep, n_swaps


def dsnot_refine(net: nn.Module, original: nn.Module, X_cal: torch.Tensor) -> tuple[nn.Module, int]:
    """In forward order: refine each unit's mask with DSnoT on the dense original weights and the current model's
    inputs, then set the unit's weights to the original weights under the new mask (the full refit follows)."""
    total = 0
    params0 = dict(original.named_parameters())
    for kind, name in GB.UNITS:
        mu, xn = unit_input_stats(net, X_cal, kind, name)
        W0, _ = GB.unit_weight(original, kind, name)
        keep0 = GB.unit_mask(net, kind, name).detach().cpu().numpy()
        groups, c0 = [], 0
        for t in GB.unit_tensors(kind, name):
            w = params0[t].numel() // params0[t].shape[0]
            groups.append((c0, c0 + w))
            c0 += w
        keep, ns = dsnot_matrix(W0.detach().cpu().numpy().astype(np.float64), keep0, mu, xn, groups)
        total += ns
        _, b_cur = GB.unit_weight(net, kind, name)
        Wn = torch.from_numpy(W0.detach().cpu().numpy() * keep).to(W0.device, W0.dtype)
        GB.set_unit_weight(net, kind, name, Wn, None)
        params = dict(net.named_parameters())
        for (a, b), t in zip(groups, GB.unit_tensors(kind, name)):
            net.prune_masks[t] = torch.from_numpy(keep[:, a:b].copy()).reshape(params[t].shape)
    return net, total


# ------------------------------------------------------------------------------ BatchNorm re-estimation

@torch.no_grad()
def bn_recalibrate(net: nn.Module, X_cal: torch.Tensor, batch_size: int = 256) -> nn.Module:
    bns = [m for m in net.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    net.eval()
    for m in bns:
        m.reset_running_stats()
        m.momentum = None                            # cumulative average over all calibration batches
        m.train()
    for i in range(0, len(X_cal), batch_size):
        net(X_cal[i:i + batch_size])
    for m in bns:
        m.momentum = 0.1
    return net.eval()


# ------------------------------------------------------------------------------ dispatch

def obc_fullrecon_k(net0: nn.Module, level: float, X_cal: torch.Tensor, K: int = 5) -> nn.Module:
    """OBC selection + OBS update in K nested steps (step k prunes round((n - keep) k / K) per tensor), the full
    ridge refit after every step (the schedule of v3's l1_fullrecon, with OBC in place of magnitude)."""
    final = uniform_keep_counts(net0, level)
    sizes = {n: getattr(m, p).numel() for n, m, p in prunable_tensors(net0)}
    net, prev = net0, None
    for k in range(1, K + 1):
        keep_k = {n: sizes[n] - int(round((sizes[n] - final[n]) * k / K)) for n in sizes}
        net = obc_prune(net, level, X_cal, keep_counts=keep_k)
        if prev is not None:
            for n, mk in prev.items():
                if torch.any(net.prune_masks[n] & ~mk):
                    raise RuntimeError(f"obc step {k}: a pruned weight of {n} was revived")
        GB.full_reconstruct(net, net0, X_cal, damp=GB.DAMP)
        prev = {n: m.clone() for n, m in net.prune_masks.items()}
    return net


# ------------------------------------------------------------------------------ layer budget allocation (added 01:00)
# unit groups of TinyHAR's prunable tensors (52,724 weights): attention block (mha in / out, ff1, ff2, attn_score,
# 6,328), convs (11,900), fusion (9,408), LSTM (25,088). Keep fraction per group; the total is 20 % (80 % sparsity).
ALLOC_GROUPS = {"attn": ("channel_attn.", "attn_score"), "conv": ("convs.",), "fus": ("fusion.",), "lstm": ("lstm.",)}
ALLOCS = {   # keep fractions (attn, conv, fus); lstm takes the rest of the 20 % budget
    "a0": None,                  # uniform (= obc_fullrecon)
    "a1": (0.05, 0.15, 0.10),    # protect the LSTM
    "a2": (0.10, 0.35, 0.15),    # protect the convs
    "a3": (0.05, 0.25, 0.25),    # protect convs + fusion
    "a4": (0.05, 0.2205, 0.2205),  # sacrifice the attention block only
    "a5": (0.245, 0.245, 0.245),   # sacrifice the LSTM (keep 15 %)
    "a6": (0.05, 0.10, 0.05),    # protect the LSTM strongly
}


def group_of(t: str) -> str:
    for g, keys in ALLOC_GROUPS.items():
        if any(k in t for k in keys):
            return g
    raise KeyError(t)


def alloc_keep_counts(net0: nn.Module, level: float, name: str) -> dict:
    uni = uniform_keep_counts(net0, level)
    fr = ALLOCS[name]
    if fr is None:
        return uni
    sizes = {n: getattr(m, p).numel() for n, m, p in prunable_tensors(net0)}
    total_keep = sum(uni.values())
    scale = (1.0 - level) / 0.2          # fractions are written for 80 %; at 90 % they are halved (same proportions)
    frac = {g: f * scale for g, f in zip(("attn", "conv", "fus"), fr)}
    keep = {n: max(1, int(round(sizes[n] * frac[group_of(n)]))) for n in sizes if group_of(n) != "lstm"}
    for n in sizes:          # the two tiny, sensitive tensors stay dense (conv0: 140 weights, attn_score: 56)
        if n.endswith(("convs.0.weight", "attn_score.weight")):
            keep[n] = sizes[n]
    lstm = [n for n in sizes if group_of(n) == "lstm"]
    rest = total_keep - sum(keep.values())
    lsz = sum(sizes[n] for n in lstm)
    for i, n in enumerate(lstm):
        keep[n] = int(round(rest * sizes[n] / lsz)) if i < len(lstm) - 1 else rest - sum(keep[m] for m in lstm[:-1])
    if sum(keep.values()) != total_keep or any(keep[n] > sizes[n] or keep[n] < 1 for n in sizes):
        raise RuntimeError(f"allocation {name} infeasible")
    return keep


V5_BASELINES = ("obc", "obc_fullrecon", "sparsegpt_fullrecon", "dsnot_fullrecon", "l1_fullrecon_bnrecal",
                "l1_fullrecon_k20")


def build(method: str, net0: nn.Module, level: float, X_cal: torch.Tensor) -> tuple[nn.Module, dict]:
    net0 = net0.eval()
    info = {}
    if method == "obc":
        net = obc_prune(net0, level, X_cal)
    elif method == "obc_fullrecon":
        net = obc_prune(net0, level, X_cal)
        GB.full_reconstruct(net, net0, X_cal, damp=GB.DAMP)
    elif method.startswith("obc_fullrecon_damp"):       # OBC dampening check (added 00:05)
        net = obc_prune(net0, level, X_cal, damp=float(method.split("damp")[1]))
        GB.full_reconstruct(net, net0, X_cal, damp=GB.DAMP)
    elif method.startswith("alloc_"):                   # obc_fullrecon with a fixed group allocation
        keep = alloc_keep_counts(net0, level, method.split("_")[1])
        net = obc_prune(net0, level, X_cal, keep_counts=keep)
        GB.full_reconstruct(net, net0, X_cal, damp=GB.DAMP)
        info["keep_lstm"] = sum(v for n, v in keep.items() if group_of(n) == "lstm")
    elif method == "obc_fullrecon_k5":
        net = obc_fullrecon_k(net0, level, X_cal, K=5)
    elif method == "sparsegpt_fullrecon":
        net = GB.sparsegpt_prune(net0, level, X_cal, log=lambda x: None)
        GB.full_reconstruct(net, net0, X_cal, damp=GB.DAMP)
    elif method == "dsnot_fullrecon":
        net = GB.build("wanda_fullrecon", net0, level, X_cal, log=lambda x: None)
        net, info["dsnot_swaps"] = dsnot_refine(net, net0, X_cal)
        GB.full_reconstruct(net, net0, X_cal, damp=GB.DAMP)
    elif method == "l1_fullrecon_bnrecal":
        net = GB.build("l1_fullrecon", net0, level, X_cal, log=lambda x: None)
        bn_recalibrate(net, X_cal)
    elif method == "l1_fullrecon_k20":
        keep = uniform_keep_counts(net0, level)
        net, _ = iterative_prune_budget(net0, keep, K=20, beta=0.0, importance_fn=None,
                                        recon_fn=lambda n: GB.full_reconstruct(n, net0, X_cal, damp=GB.DAMP),
                                        log=lambda x: None)
    else:
        raise ValueError(method)
    uni = alloc_keep_counts(net0, level, method.split("_")[1]) if method.startswith("alloc_") else uniform_keep_counts(net0, level)
    for t, m in net.prune_masks.items():
        if int(m.sum()) != uni[t]:
            raise RuntimeError(f"{method} {t}: kept {int(m.sum())}, uniform {uni[t]}")
    params = dict(net.named_parameters())
    for t, m in net.prune_masks.items():
        if torch.any(params[t].detach()[~m.to(params[t].device)] != 0):
            raise RuntimeError(f"{method}: pruned weight non-zero in {t}")
    if not head_unpruned(net):
        raise RuntimeError(f"{method} pruned the head")
    net.prune_sparsity = level
    return net.eval(), info
