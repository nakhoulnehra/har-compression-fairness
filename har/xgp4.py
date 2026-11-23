"""XGP round 4 (final round, validation only; plan Section 2.5 (b), "Round 4 pre-registration").
Round 3 (har/xgp3.py) is imported, never changed. HARTH data only; no synthetic (older-like) data.

    groups        group-agnostic and dataset-agnostic: an optional group label per calibration / training
                  window, default = subject (available in any wearable dataset); no age or HAR70+ information
    importance    per group g, the class-balanced alignment objective J_g (one backward pass per group),
                  I_g = |w dJ_g/dw| layer-normalised; I = mean of the k largest I_g(w) (worst-group emphasis)
    fine-tuning   group DRO over the same groups (Sagawa et al. 2020) for XGP; every baseline also gets the
                  same DRO as a control (*_kd_dro)
    unstructured  layer budgets: XGP budgets (worst-group importance shares at theta0) or LAMP
                  (Lee et al. 2021), both through the same bounds (floor 2% / 1 per row; every tensor the
                  reconstruction does not refit keeps at least its uniform count) and water-filling to the
                  uniform total; budgeted iterative pruning (K nested steps, row floor, reconstruction)
    structured    whole output filters of convs.0 / .3 / .6 (+ the next conv's input channels); filter
                  scores L1 (Li et al. 2017), Taylor (Molchanov et al. 2019), XGP-S (summed I); K nested
                  steps with reconstruction; physical slicing, MACs, CPU latency
    rules         on real inner-val windows: eligibility (F1), final (LCI_val), "beats" (LCI gain >= 0.02,
                  >= 5 of 6 paired runs, F1 within 1 pt, worst-subject LCI within 0.01)
"""

from __future__ import annotations

import copy
import math
import statistics
import time
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn

from .compression import prunable_tensors, size_record
from .compression import attribution_grad_context, eg_attributions, segment_attributions
from .xgp3 import _layer_normalise, masked_keep_scores, refit_modules

ROUND4 = {
    "folds": (0, 1), "seeds": (0, 1, 2), "levels_u": (0.8, 0.9, 0.95), "levels_u_kd5": (0.9, 0.95),
    "levels_s": (0.5, 0.75), "beta_u": 0.1, "beta_s": 1.0, "K": 5, "m_score": 8, "score_batch": 125, "damp": 0.01,
    "epochs": 20, "lr": 1e-4, "curve_epochs": (2, 5), "curve_levels_u": (0.9, 0.95),
    "cal_per_cell": 10, "cal_seed": 46, "kd5_fraction": 0.05, "kd5_seed": 48, "top_k": 3,
    "floor_frac": 0.02, "taylor_batches": 10, "taylor_batch_size": 256, "taylor_seed": 42,
    "ptq_levels_u": (0.9, 0.95), "ptq_levels_s": (0.75,), "ptq8_level_u": 0.9, "ptq8_level_s": 0.75,
    "nsamples": 200, "shap_seed": 0, "dro_eta": 0.01,
    "gain": 0.02, "min_better": 5, "f1_tolerance": 0.01, "wsl_tolerance": 0.01, "tie": 0.005,
    "latency": {"threads": (1, None), "batches": (1, 256), "warmup": 20, "reps": 200},
}
# unstructured cells: (budget, selection); all K-step iterative + reconstruction
U_CELLS = {"l1_recon": ("uniform", "l1"), "xgp": ("uniform", "xgp"), "lamp_recon": ("lamp", "l1"),
           "xgp_lamp": ("lamp", "xgp"), "l1_xb_recon": ("xb", "l1"), "xgp_xb": ("xb", "xgp")}
U_BASELINES = ("l1", "l1_recon", "lamp_recon")
U_DRO_CONTROLS = {"l1_kd_dro": "l1", "l1_recon_kd_dro": "l1_recon", "lamp_recon_kd_dro": "lamp_recon"}  # name: start
U_CANDIDATES = ("xgp", "l1_xb_recon", "xgp_lamp", "xgp_xb")          # tie order
U_CURVE = ("l1_recon", "lamp_recon", "xgp", "xgp_xb")
U_PTQ = ("l1_recon", "lamp_recon", "xgp", "xgp_xb")
S_METHODS = {"l1_filter_recon": "l1", "taylor_recon": "taylor", "xgp_s": "xgp"}
S_BASELINES = ("l1_filter", "l1_filter_recon", "taylor_recon")
S_DRO_CONTROLS = {"l1_filter_kd_dro": "l1_filter", "l1_filter_recon_kd_dro": "l1_filter_recon",
                  "taylor_recon_kd_dro": "taylor_recon"}
S_CANDIDATES = ("xgp_s",)
S_CURVE = ("l1_filter_recon", "taylor_recon", "xgp_s")
S_PTQ = ("l1_filter_recon", "taylor_recon", "xgp_s")


# ------------------------------------------------------------------------------ groups and importance

def default_groups(subjects: Sequence, groups: Sequence | None = None) -> np.ndarray:
    """The group label of every window: `groups` if given, else the subject (the dataset-agnostic default)."""
    g = np.asarray(subjects if groups is None else groups).astype(str)
    if len(g) != len(subjects):
        raise ValueError(f"{len(g)} group labels for {len(subjects)} windows")
    return g


def group_index(groups: Sequence, labels: Sequence[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """Integer codes 0..G-1 (for group DRO) in the order of `labels` (default: sorted) and the labels."""
    g = np.asarray(groups).astype(str)
    labels = sorted(set(g.tolist())) if labels is None else list(labels)
    pos = {x: i for i, x in enumerate(labels)}
    return np.array([pos[x] for x in g.tolist()], dtype=np.int64), labels


def calibration_pick(labels: np.ndarray, groups: np.ndarray, per_cell: int, seed: int) -> np.ndarray:
    """Up to `per_cell` windows per (group, class) cell, without replacement (one generator, cells in sorted
    order); indices returned in (group, class, pick) order."""
    rng = np.random.default_rng(seed)
    labels, groups = np.asarray(labels), np.asarray(groups).astype(str)
    out = []
    for g in sorted(set(groups.tolist())):
        for c in sorted(set(labels[groups == g].tolist())):
            cell = np.flatnonzero((groups == g) & (labels == c))
            out += rng.permutation(cell)[:per_cell].tolist()
    return np.asarray(out, dtype=np.int64)


def class_balanced_weights(labels: np.ndarray) -> np.ndarray:
    """w_i = 1 / (C n_c(i)) over the C classes present: each class weighs equally and the weights sum to 1."""
    labels = np.asarray(labels)
    cls, cnt = np.unique(labels, return_counts=True)
    n = dict(zip(cls.tolist(), cnt.tolist()))
    return np.array([1.0 / (len(cls) * n[c]) for c in labels.tolist()])


def weighted_alignment_importance(model: nn.Module, ref_model: nn.Module, X: torch.Tensor, target: torch.Tensor,
                                  weights: np.ndarray, baselines: torch.Tensor, m: int, eg_seeds,
                                  batch_size: int = ROUND4["score_batch"]) -> dict[str, torch.Tensor]:
    """xgp3.alignment_importance_ref with a weight per window: J = sum_i w_i <A_theta(x_i), sg(A_theta0(x_i))>
    (the same EG draws for both models), I_s = |w * dJ/dw|, I = mean over the EG seeds. Pruned weights get 0."""
    net = copy.deepcopy(model).eval()
    ref = ref_model.eval()
    named = prunable_tensors(net)
    params = [getattr(mod, p) for _, mod, p in named]
    for p in params:
        p.requires_grad_(True)
    dev = params[0].device
    target = torch.as_tensor(target).long()
    wts = torch.as_tensor(np.asarray(weights), dtype=torch.float32)
    total = [torch.zeros_like(p) for p in params]
    for s in eg_seeds:
        g = torch.Generator().manual_seed(int(s))
        acc = [torch.zeros_like(p) for p in params]
        for i in range(0, len(X), batch_size):
            Xb, tb = X[i:i + batch_size].to(dev), target[i:i + batch_size].to(dev)
            state = g.get_state()
            A0 = segment_attributions(eg_attributions(ref, Xb, tb, baselines, m, create_graph=False, generator=g))
            g.set_state(state)
            with attribution_grad_context():
                A = segment_attributions(eg_attributions(net, Xb, tb, baselines, m, create_graph=True, generator=g))
                wb = wts[i:i + batch_size].to(dev, A.dtype).reshape(-1, *([1] * (A.dim() - 1)))
                grads = torch.autograd.grad((wb * A * A0.detach()).sum(), params, allow_unused=True)
            for a, gr in zip(acc, grads):
                if gr is not None:
                    a += gr.detach()
        for t, p, a in zip(total, params, acc):
            t += (p.detach() * a).abs()
    return {name: t / len(eg_seeds) for (name, _, _), t in zip(named, total)}


def per_group_importance(model: nn.Module, ref_model: nn.Module, X: torch.Tensor, target: torch.Tensor,
                         labels: np.ndarray, groups: np.ndarray | None, baselines: torch.Tensor, m: int, eg_seeds,
                         batch_size: int = ROUND4["score_batch"]) -> dict[str, dict[str, torch.Tensor]]:
    """{group: raw I_g}, one class-balanced alignment objective per group (groups None -> a single group)."""
    labels = np.asarray(labels)
    groups = np.full(len(labels), "all") if groups is None else np.asarray(groups).astype(str)
    if len(groups) != len(labels) or len(labels) != len(X):
        raise ValueError("X, labels and groups must have one entry per calibration window")
    target = torch.as_tensor(target)
    out = {}
    for g in sorted(set(groups.tolist())):
        idx = np.flatnonzero(groups == g)
        ti = torch.from_numpy(idx)
        out[g] = weighted_alignment_importance(model, ref_model, X[ti.to(X.device)], target[ti.to(target.device)],
                                               class_balanced_weights(labels[idx]), baselines, m=m,
                                               eg_seeds=eg_seeds, batch_size=batch_size)
    return out


def worst_group_mean(per_group: Sequence[dict[str, torch.Tensor]], k: int) -> dict[str, torch.Tensor]:
    """Elementwise mean of the k largest values over the groups (k = 1 -> max; k >= G -> mean)."""
    per_group = list(per_group)
    kk = max(1, min(int(k), len(per_group)))
    return {n: torch.stack([g[n] for g in per_group]).topk(kk, dim=0).values.mean(0) for n in per_group[0]}


def worst_group_importance(model: nn.Module, ref_model: nn.Module, X: torch.Tensor, target: torch.Tensor,
                           labels: np.ndarray, groups: np.ndarray | None, baselines: torch.Tensor, *,
                           k: int = ROUND4["top_k"], m: int, eg_seeds,
                           batch_size: int = ROUND4["score_batch"]) -> dict[str, torch.Tensor]:
    """I(w) = mean of the k largest layer-normalised I_g(w) over the groups (worst-group emphasis: the
    groups whose explanations pruning w would change most)."""
    raw = per_group_importance(model, ref_model, X, target, labels, groups, baselines, m, eg_seeds, batch_size)
    return worst_group_mean([{n: _layer_normalise(v) for n, v in I.items()} for I in raw.values()], k)


def worst_group_shares(model: nn.Module, ref_model: nn.Module, X: torch.Tensor, target: torch.Tensor,
                       labels: np.ndarray, groups: np.ndarray | None, baselines: torch.Tensor, *,
                       k: int = ROUND4["top_k"], m: int, eg_seeds,
                       batch_size: int = ROUND4["score_batch"]) -> dict[str, float]:
    """XGP budget shares: per group, raw I_g summed per tensor as a share over the tensors; per tensor the
    mean of the k largest group shares; renormalised to sum 1."""
    raw = per_group_importance(model, ref_model, X, target, labels, groups, baselines, m, eg_seeds, batch_size)
    parts = []
    for I in raw.values():
        tot = {n: float(v.double().sum()) for n, v in I.items()}
        z = sum(tot.values())
        parts.append({n: torch.tensor(v / z if z > 0 else 0.0, dtype=torch.float64) for n, v in tot.items()})
    agg = {n: float(v) for n, v in worst_group_mean(parts, k).items()}
    z = sum(agg.values())
    return {n: v / z for n, v in agg.items()} if z > 0 else agg


# ------------------------------------------------------------------------------ budgets

def uniform_keep_counts(model: nn.Module, sparsity: float) -> dict[str, int]:
    """Per tensor n - round(sparsity n), exactly as prune_by_score."""
    return {n: getattr(m, p).numel() - int(round(sparsity * getattr(m, p).numel())) for n, m, p in prunable_tensors(model)}


def unreconstructed_tensors(model: nn.Module, X_probe: torch.Tensor) -> list[str]:
    """Prunable tensors whose module layerwise_reconstruct does not refit (LSTM, attention in/out proj)."""
    refit = set(refit_modules(model, X_probe))
    return [n for n, _, _ in prunable_tensors(model) if n.rsplit(".", 1)[0] not in refit]


def budget_bounds(model: nn.Module, sparsity: float, protect: Sequence[str],
                  floor_frac: float = ROUND4["floor_frac"]) -> tuple[dict[str, int], dict[str, int], int]:
    """(lo, hi, total): lo = max(ceil(floor_frac n), rows), raised to the uniform count for `protect`;
    hi = n; total = the uniform total of kept weights."""
    uni = uniform_keep_counts(model, sparsity)
    lo, hi = {}, {}
    for n, m, p in prunable_tensors(model):
        w = getattr(m, p)
        lo[n] = max(int(math.ceil(floor_frac * w.numel() - 1e-9)), int(w.shape[0]))
        if n in protect:
            lo[n] = max(lo[n], uni[n])
        hi[n] = w.numel()
        lo[n] = min(lo[n], hi[n])
    return lo, hi, int(sum(uni.values()))


def water_fill(shares: dict[str, float], total: int, lo: dict[str, int], hi: dict[str, int]) -> dict[str, int]:
    """Integer keep counts with lo <= keep <= hi, sum = total, keep = clip(lambda share, lo, hi) (lambda by
    bisection), rounded by largest remainder. Raises if infeasible."""
    names = list(lo)
    if sum(lo.values()) > total or sum(hi.values()) < total:
        raise ValueError(f"infeasible budget: sum lo {sum(lo.values())}, sum hi {sum(hi.values())}, total {total}")
    s = {n: max(float(shares.get(n, 0.0)), 0.0) for n in names}

    def f(lam):
        return {n: min(hi[n], max(lo[n], lam * s[n])) for n in names}
    a, b = 0.0, 1.0
    while sum(f(b).values()) < total and b < 1e30:
        b *= 2
    for _ in range(200):
        mid = (a + b) / 2
        if sum(f(mid).values()) < total:
            a = mid
        else:
            b = mid
    x = f(b)
    keep = {n: int(min(hi[n], max(lo[n], math.floor(x[n] + 1e-9)))) for n in names}
    rem = total - sum(keep.values())
    order = sorted(names, key=lambda n: (-(x[n] - math.floor(x[n] + 1e-9)), names.index(n)))
    while rem != 0:
        moved = False
        for n in (order if rem > 0 else order[::-1]):
            if rem > 0 and keep[n] < hi[n]:
                keep[n] += 1
                rem -= 1
                moved = True
            elif rem < 0 and keep[n] > lo[n]:
                keep[n] -= 1
                rem += 1
                moved = True
            if rem == 0:
                break
        if not moved:
            raise RuntimeError("water_fill could not distribute the remainder")
    assert sum(keep.values()) == total
    return keep


def lamp_keep_counts(model: nn.Module, total: int) -> dict[str, int]:
    """LAMP (Lee et al. 2021): per tensor, weights sorted by w^2 ascending, score_u = w_u^2 / sum_{v >= u} w_v^2;
    the `total` highest scores over all prunable tensors are kept (ties -> stable order). Returns counts."""
    names, scores = [], []
    for n, m, p in prunable_tensors(model):
        w2 = getattr(m, p).detach().double().flatten().cpu() ** 2
        srt, _ = torch.sort(w2, stable=True)
        tail = torch.flip(torch.cumsum(torch.flip(srt, [0]), 0), [0])
        scores.append(srt / tail.clamp_min(1e-300))
        names.append(n)
    allsc = torch.cat(scores)
    owner = torch.cat([torch.full((len(s),), i) for i, s in enumerate(scores)])
    order = torch.argsort(-allsc, stable=True)[:total]
    cnt = torch.bincount(owner[order], minlength=len(names))
    return {n: int(c) for n, c in zip(names, cnt)}


def bounded_budget(model: nn.Module, sparsity: float, protect: Sequence[str], kind: str,
                   shares: dict[str, float] | None = None) -> dict[str, int]:
    """Keep counts for kind 'lamp' (LAMP counts as shares) or 'xb' (given XGP shares), through the same
    bounds and water-filling; total = the uniform total."""
    lo, hi, total = budget_bounds(model, sparsity, protect)
    if kind == "lamp":
        cnt = lamp_keep_counts(model, total)
        shares = {n: c / total for n, c in cnt.items()}
    elif kind != "xb" or shares is None:
        raise ValueError(kind)
    return water_fill(shares, total, lo, hi)


# ------------------------------------------------------------------------------ budgeted iterative pruning

def prune_by_counts(model: nn.Module, scores: dict[str, torch.Tensor], prune_counts: dict[str, int]) -> nn.Module:
    """prune_by_score with a per-tensor number of pruned weights (lowest score first, ties -> lowest index)."""
    pruned = copy.deepcopy(model)
    masks = {}
    with torch.no_grad():
        for name, mod, p in prunable_tensors(pruned):
            w = getattr(mod, p)
            sc = scores[name].detach().to("cpu", torch.float64).flatten()
            k = int(prune_counts[name])
            if not 0 <= k < w.numel():
                raise ValueError(f"{name}: cannot prune {k} of {w.numel()}")
            keep = torch.ones(w.numel(), dtype=torch.bool)
            keep[torch.argsort(sc, stable=True)[:k]] = False
            masks[name] = keep.reshape(w.shape)
            w.mul_(masks[name].to(w.device, w.dtype))
    pruned.prune_masks = masks
    return pruned


def protect_row_max(scores: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Give the highest-scoring surviving weight of every output row a score above everything, so each row
    keeps >= 1 weight (pruned weights score -1 and are never chosen while a survivor exists)."""
    out = {}
    for n, s in scores.items():
        r = s.detach().clone().reshape(s.shape[0], -1)
        rows = torch.arange(r.shape[0], device=r.device)
        top = r.argmax(1)
        v = r[rows, top]
        r[rows, top] = torch.where(v >= 0, torch.full_like(v, 1e30), v)
        out[n] = r.reshape(s.shape)
    return out


def iterative_prune_budget(original: nn.Module, keep: dict[str, int], *, K: int, beta: float,
                           importance_fn: Callable | None, recon_fn: Callable | None,
                           log: Callable[[str], None] = print) -> tuple[nn.Module, list[dict]]:
    """K nested steps to the per-tensor keep counts (step k prunes round((n - keep) k / K)), scores =
    masked_keep_scores (beta, I recomputed on the current model) with the row floor, reconstruction after
    every step. No gradient training."""
    sizes = {n: getattr(m, p).numel() for n, m, p in prunable_tensors(original)}
    net, rows = original, []
    for k in range(1, K + 1):
        I = importance_fn(net) if (importance_fn is not None and beta > 0) else None
        sc = protect_row_max(masked_keep_scores(net, I, beta))
        pc = {n: int(round((sizes[n] - keep[n]) * k / K)) for n in sizes}
        prev = getattr(net, "prune_masks", None)
        net = prune_by_counts(net, sc, pc)
        if prev:
            for n, mk in prev.items():
                if torch.any(net.prune_masks[n] & ~mk):
                    raise RuntimeError(f"step {k}: a pruned weight of {n} was revived")
        rec = recon_fn(net) if recon_fn is not None else None
        rows.append({"step": k, "n_nonzero": size_record(net)["n_nonzero"], "recon": rec["refit"] if rec else None})
        log(f"      budgeted prune step {k}/{K}: n_nonzero {rows[-1]['n_nonzero']}" + (", reconstructed" if rec else ""))
    for n, mk in net.prune_masks.items():
        if int(mk.sum()) != keep[n]:
            raise RuntimeError(f"{n}: kept {int(mk.sum())}, budget {keep[n]}")
    return net, rows


# ------------------------------------------------------------------------------ structured (conv filters)

def structured_layers(model: nn.Module) -> list[tuple[str, str, str]]:
    """(producer conv, its BatchNorm, consumer conv) module names for the structured filters: every Conv2d
    in the conv stack except the last (whose filters are the transformer's d_model)."""
    convs = [n for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    mods = dict(model.named_modules())
    out = []
    for a, b in zip(convs[:-1], convs[1:]):
        parent, idx = a.rsplit(".", 1)
        bn = f"{parent}.{int(idx) + 1}"
        if not isinstance(mods.get(bn), nn.BatchNorm2d):
            raise ValueError(f"no BatchNorm2d after {a}")
        out.append((a, bn, b))
    return out


def structured_masks(model: nn.Module, removed: dict[str, Sequence[int]]) -> dict[str, torch.Tensor]:
    """Keep-masks for the producer and consumer conv weights: producer rows of removed filters and the
    consumer's matching input channels are False; every other prunable tensor is unmasked."""
    params = dict(model.named_parameters())
    masks = {}
    for prod, _, cons in structured_layers(model):
        r = list(removed.get(prod, ()))
        for name in (f"{prod}.weight", f"{cons}.weight"):
            masks.setdefault(name, torch.ones_like(params[name], dtype=torch.bool))
        if r:
            masks[f"{prod}.weight"][r] = False
            masks[f"{cons}.weight"][:, r] = False
    return masks


def apply_structured(model: nn.Module, removed: dict[str, Sequence[int]]) -> nn.Module:
    net = copy.deepcopy(model)
    net.prune_masks = structured_masks(net, removed)
    params = dict(net.named_parameters())
    with torch.no_grad():
        for n, mk in net.prune_masks.items():
            params[n].mul_(mk.to(params[n].device, params[n].dtype))
    net.structured_removed = {k: sorted(int(i) for i in v) for k, v in removed.items()}
    return net


def filter_scores_l1(model: nn.Module) -> dict[str, torch.Tensor]:
    mods = dict(model.named_modules())
    return {p: mods[p].weight.detach().abs().reshape(mods[p].weight.shape[0], -1).sum(1).cpu().double()
            for p, _, _ in structured_layers(model)}


def filter_scores_from_weight_scores(model: nn.Module, I: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Sum of per-weight scores over each producer filter's own conv weights."""
    return {p: I[f"{p}.weight"].detach().reshape(I[f"{p}.weight"].shape[0], -1).sum(1).cpu().double()
            for p, _, _ in structured_layers(model)}


def filter_scores_taylor(model: nn.Module, batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
                         loss_fn: nn.Module) -> dict[str, torch.Tensor]:
    """Molchanov et al. (2019), first-order Taylor for a parameter group: mean over batches of
    (sum_{w in filter} w dL/dw)^2, L = class-weighted CE, eval mode, cuDNN off (LSTM backward)."""
    m = copy.deepcopy(model).eval()
    dev = next(m.parameters()).device
    mods = dict(m.named_modules())
    lay = [p for p, _, _ in structured_layers(m)]
    acc = {p: torch.zeros(mods[p].weight.shape[0], dtype=torch.float64) for p in lay}
    with torch.backends.cudnn.flags(enabled=False):
        for X, y in batches:
            m.zero_grad(set_to_none=True)
            loss_fn(m(X.to(dev)), y.to(dev)).backward()
            for p in lay:
                w = mods[p].weight
                acc[p] += ((w.detach() * w.grad).reshape(w.shape[0], -1).sum(1) ** 2).double().cpu()
    return {p: v / max(len(batches), 1) for p, v in acc.items()}


def iterative_prune_structured(original: nn.Module, level: float, *, K: int, score_fn: Callable,
                               recon_fn: Callable | None, log: Callable[[str], None] = print
                               ) -> tuple[nn.Module, list[dict]]:
    """K nested steps; step k removes round(level k / K F) filters per structured layer (lowest score among
    the survivors; removed filters score -inf; ties -> lowest index), then reconstructs."""
    layers = structured_layers(original)
    mods = dict(original.named_modules())
    F_ = {p: mods[p].weight.shape[0] for p, _, _ in layers}
    removed = {p: [] for p, _, _ in layers}
    net, rows = original, []
    for k in range(1, K + 1):
        sc = score_fn(net)
        new = {}
        for p in F_:
            s = sc[p].clone().double()
            s[removed[p]] = -math.inf
            r = int(round(level * k / K * F_[p]))
            new[p] = sorted(int(i) for i in torch.argsort(s, stable=True)[:r])
            if not set(removed[p]) <= set(new[p]):
                raise RuntimeError(f"step {k}: a removed filter of {p} came back")
        removed = new
        net = apply_structured(original if k == 1 else net, removed)
        rec = recon_fn(net) if recon_fn is not None else None
        rows.append({"step": k, "n_nonzero": size_record(net)["n_nonzero"],
                     "removed": {p: len(v) for p, v in removed.items()}, "recon": rec["refit"] if rec else None})
        log(f"      structured step {k}/{K}: removed {rows[-1]['removed']}" + (", reconstructed" if rec else ""))
    net.prune_sparsity = level
    return net, rows


def slice_structured(model: nn.Module) -> nn.Module:
    """Physically smaller copy: producer convs, their BatchNorm and the consumers' input channels keep
    only the surviving filters (from prune_masks). The output equals the masked model's (tested)."""
    net = copy.deepcopy(model).cpu()
    mods = dict(net.named_modules())

    def setmod(name, new):
        parent, child = name.rsplit(".", 1)
        setattr(mods[parent], child, new)
        mods[name] = new
    masks = model.prune_masks
    keep_prev = None
    for prod, bn, cons in structured_layers(net):
        keep = masks[f"{prod}.weight"].reshape(mods[prod].weight.shape[0], -1).any(1).cpu()
        kidx = torch.nonzero(keep).flatten()
        c = mods[prod]
        cin = c.in_channels if keep_prev is None else len(keep_prev)
        nc = nn.Conv2d(cin, len(kidx), c.kernel_size, c.stride, c.padding, c.dilation, bias=c.bias is not None)
        with torch.no_grad():
            W = c.weight.detach()[kidx]
            if keep_prev is not None:
                W = W[:, keep_prev]
            nc.weight.copy_(W)
            if c.bias is not None:
                nc.bias.copy_(c.bias.detach()[kidx])
        setmod(prod, nc)
        b = mods[bn]
        nb = nn.BatchNorm2d(len(kidx), b.eps, b.momentum, b.affine, b.track_running_stats)
        with torch.no_grad():
            if b.affine:
                nb.weight.copy_(b.weight.detach()[kidx])
                nb.bias.copy_(b.bias.detach()[kidx])
            nb.running_mean.copy_(b.running_mean[kidx])
            nb.running_var.copy_(b.running_var[kidx])
            nb.num_batches_tracked.copy_(b.num_batches_tracked)
        setmod(bn, nb)
        keep_prev = kidx
    # the last consumer: input channels only
    last = structured_layers(model)[-1][2]
    c = mods[last]
    nc = nn.Conv2d(len(keep_prev), c.out_channels, c.kernel_size, c.stride, c.padding, c.dilation,
                   bias=c.bias is not None)
    with torch.no_grad():
        nc.weight.copy_(c.weight.detach()[:, keep_prev])
        if c.bias is not None:
            nc.bias.copy_(c.bias.detach())
    setmod(last, nc)
    for a in ("prune_masks", "structured_removed"):
        if hasattr(net, a):
            delattr(net, a)
    return net.eval()


def count_macs(model: nn.Module, x: torch.Tensor) -> int:
    """Multiply-accumulates of one forward (our count): Conv2d / Linear called as modules, MultiheadAttention
    (in-proj, scores, weighted sum, out-proj), LSTM (T x 4H x (I + H))."""
    total = [0]
    hooks = []

    def conv(m, i, o):
        total[0] += o.numel() * (m.in_channels // m.groups) * int(np.prod(m.kernel_size))

    def lin(m, i, o):
        total[0] += o.numel() * m.in_features

    def mha(m, i, o):
        q = i[0]
        L, E = q.shape[-2], q.shape[-1]
        B = q.numel() // (L * E)
        total[0] += B * (3 * L * E * E + 2 * L * L * E + L * E * E)

    def lstm(m, i, o):
        x_ = i[0]
        B, T, I = x_.shape if m.batch_first else (x_.shape[1], x_.shape[0], x_.shape[2])
        total[0] += B * T * 4 * m.hidden_size * (I + m.hidden_size)
    inside_mha = set()
    for n, m in model.named_modules():
        if isinstance(m, nn.MultiheadAttention):
            inside_mha |= {n + "." + c for c, _ in m.named_modules() if c}
    for n, m in model.named_modules():
        if n in inside_mha:
            continue
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin))
        elif isinstance(m, nn.MultiheadAttention):
            hooks.append(m.register_forward_hook(mha))
        elif isinstance(m, nn.LSTM):
            hooks.append(m.register_forward_hook(lstm))
    was = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in hooks:
            h.remove()
        model.train(was)
    return int(total[0] // x.shape[0])


def cpu_latency(model: nn.Module, batch: int, threads: int | None, warmup: int, reps: int) -> dict:
    """Median / p10 / p90 milliseconds of one forward on CPU (inference_mode)."""
    old = torch.get_num_threads()
    if threads:
        torch.set_num_threads(threads)
    m = copy.deepcopy(model).cpu().eval()
    x = torch.randn(batch, 6, 128, generator=torch.Generator().manual_seed(0))
    ts = []
    try:
        with torch.inference_mode():
            for _ in range(warmup):
                m(x)
            for _ in range(reps):
                t0 = time.perf_counter()
                m(x)
                ts.append((time.perf_counter() - t0) * 1e3)
    finally:
        torch.set_num_threads(old)
    q = statistics.quantiles(ts, n=10)
    return {"batch": batch, "threads": threads or old, "median_ms": statistics.median(ts), "p10_ms": q[0],
            "p90_ms": q[-1], "reps": reps}


# ------------------------------------------------------------------------------ pre-registered rules

def eligible(st: dict[float, dict], best: dict[float, dict], tol: float = ROUND4["f1_tolerance"]) -> bool:
    """st, best: {level: {"f1"}} (inner-val F1); eligible = within tol of `best` at every level of `best`."""
    return all(lv in st and st[lv]["f1"] >= best[lv]["f1"] - tol - 1e-12 for lv in best)


def pick_final(stats: dict[str, dict[float, dict]], best: dict[float, dict], order: Sequence[str],
               tie: float = ROUND4["tie"]) -> tuple[str | None, list[dict]]:
    """stats = {method: {level: {"lci", "f1", ...}}} (means over runs). Final = the eligible candidate (in
    `order`) with the highest mean LCI_val over the levels; ties within `tie` -> earlier in order."""
    table = []
    for v in order:
        if v not in stats:
            continue
        lci = float(np.mean([stats[v][lv]["lci"] for lv in best if lv in stats[v]]))
        table.append({"method": v, "lci_mean_levels": lci, "eligible": eligible(stats[v], best)})
    ok = [t for t in table if t["eligible"]]
    final = None
    if ok:
        top = max(t["lci_mean_levels"] for t in ok)
        final = next(t["method"] for t in ok if t["lci_mean_levels"] >= top - tie - 1e-12)
    for t in table:
        t["selected"] = t["method"] == final
    return final, table


def beats(a_runs: dict[tuple, float], b_runs: dict[tuple, float], a_st: dict[float, dict], b_st: dict[float, dict],
          gain: float = ROUND4["gain"], min_better: int = ROUND4["min_better"],
          tol: float = ROUND4["f1_tolerance"], wsl_tol: float = ROUND4["wsl_tolerance"]) -> dict:
    """Rule 4. a_runs / b_runs: {(fold, seed): LCI_val averaged over the levels}; a_st / b_st: {level: means of
    "f1" and "worst_subject_lci"} (the levels of b_st count). A beats B iff (a) mean LCI gain >= gain, (b) A is
    higher in >= min_better paired runs, (c) F1 within tol at every level, (d) mean worst-subject LCI over
    the levels >= B's - wsl_tol."""
    if set(a_runs) != set(b_runs):
        raise ValueError(f"unpaired runs: {sorted(set(a_runs) ^ set(b_runs))}")
    keys = sorted(a_runs)
    d = [a_runs[k] - b_runs[k] for k in keys]
    n_better = int(sum(x > 0 for x in d))
    f1_ok = all(a_st[lv]["f1"] >= b_st[lv]["f1"] - tol - 1e-12 for lv in b_st)
    wa = float(np.mean([a_st[lv]["worst_subject_lci"] for lv in b_st]))
    wb = float(np.mean([b_st[lv]["worst_subject_lci"] for lv in b_st]))
    wsl_ok = wa >= wb - wsl_tol - 1e-12
    mean_d = float(np.mean(d)) if d else float("nan")
    return {"mean_gain": mean_d, "n_better": n_better, "n_pairs": len(keys), "f1_within_tol": bool(f1_ok),
            "worst_subject_lci_a": wa, "worst_subject_lci_b": wb, "worst_subject_within_tol": bool(wsl_ok),
            "beats": bool(d and mean_d >= gain - 1e-12 and n_better >= min_better and f1_ok and wsl_ok)}
