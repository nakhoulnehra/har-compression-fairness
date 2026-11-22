"""XGP round 3 (final design, validation only; plan Section 2.5 (b), "Round 3 pre-registration").
HARTH data only: every function that takes Windows calls require_harth; tensors come from HARTH windows.

    alignment_importance_ref   I(w) = | w * d/dw sum_i < A_theta(x_i), sg(A_theta0(x_i)) > | at the CURRENT
                               model theta against the ORIGINAL model's attributions, with the same EG draws
                               for both (at theta = theta0 this is round 2's alignment_importance)
    group_class_importance     I_g = mean_c I_{g,c} / mean_layer I_{g,c} (class-balanced) per group g,
                               I = max over groups
    masked_keep_scores         (1 - beta) |w| / mean + beta I / mean, means over the surviving weights;
                               pruned weights get -1 so they stay pruned
    layerwise_reconstruct      ridge least-squares refit of the kept weights (and bias) of every hooked
                               Conv / Linear layer, in forward order, to the original layer outputs
                               [OBC: Frantar, Singh & Alistarh 2022, Sec 3, Eq. 1; SparseGPT: Frantar &
                               Alistarh 2023, Sec 3, Eq. 2; 1% dampening]
    iterative_prune            K steps to the target (s k / K), rescoring on the current (reconstructed) model
    select_config, select_final   the pre-registered rules
"""

from __future__ import annotations

import copy
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .compression import (attribution_grad_context, eg_attributions, l1_scores, prunable_tensors, prune_by_score,
                          segment_attributions, size_record)

ROUND3 = {
    "folds": (0, 1), "levels": (0.5, 0.7, 0.9), "seeds": (0, 1, 2), "betas": (0.1, 0.3, 0.5), "rs": (0.25, 0.5),
    "K": 5, "m_score": 8, "score_batch": 125, "damp": 0.01, "dro_eta": 0.01, "epochs": 20, "lr": 1e-4,
    "curve_epochs": (2, 5), "cal_per_class": 70, "cal_seed": 44, "explain_per_subject": 100, "nsamples": 200,
    "shap_seed": 0, "aug_seed_offset": 1000, "f1_tolerance": 0.01, "tie": 0.005, "eps": 1e-12,
    "baselines": {"no_finetune": ("l1", "snip", "l1_recon"),
                  "finetune": ("l1", "l1_kd", "snip_kd", "l1_kdaug")},
}
# Components of each candidate in the final rule (phase 2). no_aug removes aug and DRO together.
COMPONENTS = {
    "no_finetune": {"xgp": ("score", "group", "iter", "recon"), "xgp_no_score": ("group", "iter", "recon"),
                    "xgp_no_group": ("score", "iter", "recon"), "xgp_no_iter": ("score", "group", "recon"),
                    "xgp_no_recon": ("score", "group", "iter")},
    "finetune": {"xgp_ft": ("score", "group", "iter", "recon", "aug", "dro"),
                 "xgp_no_score_ft": ("group", "iter", "recon", "aug", "dro"),
                 "xgp_no_group_ft": ("score", "iter", "recon", "aug", "dro"),
                 "xgp_no_iter_ft": ("score", "group", "recon", "aug", "dro"),
                 "xgp_no_recon_ft": ("score", "group", "iter", "aug", "dro"),
                 "xgp_ft_no_aug": ("score", "group", "iter", "recon"),
                 "xgp_ft_no_dro": ("score", "group", "iter", "recon", "aug")},
}
# The pruning options of each start (score = use the I term, groups = calibration groups, K, recon).
STARTS = {
    "xgp": {"score": True, "groups": ("normal", "older_like"), "K": ROUND3["K"], "recon": True},
    "xgp_no_score": {"score": False, "groups": ("normal", "older_like"), "K": ROUND3["K"], "recon": True},
    "xgp_no_group": {"score": True, "groups": ("normal",), "K": ROUND3["K"], "recon": True},
    "xgp_no_iter": {"score": True, "groups": ("normal", "older_like"), "K": 1, "recon": True},
    "xgp_no_recon": {"score": True, "groups": ("normal", "older_like"), "K": ROUND3["K"], "recon": False},
}


# ------------------------------------------------------------------------------ importance

def alignment_importance_ref(model: nn.Module, ref_model: nn.Module, X: torch.Tensor, target: torch.Tensor,
                             baselines: torch.Tensor, m: int = ROUND3["m_score"], eg_seeds=(0, 1),
                             batch_size: int = ROUND3["score_batch"]) -> dict[str, torch.Tensor]:
    """I(w) per prunable tensor of `model` (a deepcopy in eval mode is used; neither model changes):
    for each EG seed s (one CPU generator drawn batch after batch), A_theta = 48-segment EG attributions
    of `model` with create_graph=True and A_0 = those of `ref_model` with the SAME draws (the generator
    state is restored between the two); alignment = sum_i <A_theta_i, A_0_i>; I_s = |w * grad|;
    I = mean over seeds. Pruned weights (w = 0) get I = 0."""
    net = copy.deepcopy(model).eval()
    ref = ref_model.eval()
    named = prunable_tensors(net)
    params = [getattr(mod, p) for _, mod, p in named]
    for p in params:
        p.requires_grad_(True)
    dev = params[0].device
    target = torch.as_tensor(target).long()
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
                grads = torch.autograd.grad((A * A0.detach()).sum(), params, allow_unused=True)
            for a, gr in zip(acc, grads):
                if gr is not None:
                    a += gr.detach()
        for t, p, a in zip(total, params, acc):
            t += (p.detach() * a).abs()
    return {name: t / len(eg_seeds) for (name, _, _), t in zip(named, total)}


def _layer_normalise(I: torch.Tensor, eps: float = ROUND3["eps"]) -> torch.Tensor:
    mu = I.mean()
    return I / mu if mu > eps else torch.zeros_like(I)


def group_class_importance(model: nn.Module, ref_model: nn.Module, sets: dict[str, tuple], baselines: torch.Tensor,
                           m: int = ROUND3["m_score"], eg_seeds=(0, 1),
                           batch_size: int = ROUND3["score_batch"]) -> dict[str, torch.Tensor]:
    """Group-aware, class-balanced I(w). sets = {group: (X, target, labels)}. Per group g and class c:
    I_{g,c} on that group's windows of class c (true label); I_g = mean_c I_{g,c} / mean_layer I_{g,c};
    I = elementwise max over groups (with one group, I = I_g)."""
    per_group = []
    for X, target, labels in sets.values():
        labels = np.asarray(labels)
        target = torch.as_tensor(target)
        parts = []
        for c in np.unique(labels):
            idx = torch.from_numpy(np.flatnonzero(labels == c))
            imp = alignment_importance_ref(model, ref_model, X[idx.to(X.device)], target[idx.to(target.device)],
                                           baselines, m=m, eg_seeds=eg_seeds, batch_size=batch_size)
            parts.append({n: _layer_normalise(v) for n, v in imp.items()})
        per_group.append({n: torch.stack([p[n] for p in parts]).mean(0) for n in parts[0]})
    return {n: torch.stack([g[n] for g in per_group]).amax(0) for n in per_group[0]}


def masked_keep_scores(model: nn.Module, importance: dict[str, torch.Tensor] | None, beta: float,
                       eps: float = ROUND3["eps"]) -> dict[str, torch.Tensor]:
    """score = (1 - beta) |w| / mean + beta I / mean per prunable tensor, both means over the weights that
    are still unpruned (model.prune_masks, or all if none); pruned weights get -1, so with the
    lowest-score-first rule they stay pruned. importance None or beta 0 -> |w| only."""
    masks = getattr(model, "prune_masks", None) or {}
    out = {}
    for name, w in l1_scores(model).items():
        keep = masks.get(name, torch.ones_like(w, dtype=torch.bool)).to(w.device)
        mw = w[keep].mean() if keep.any() else w.new_tensor(0.0)
        sc = (1 - beta) * (w / mw if mw > eps else torch.zeros_like(w))
        if importance is not None and beta > 0:
            I = importance[name].detach().to(w.device, w.dtype)
            mi = I[keep].mean() if keep.any() else I.new_tensor(0.0)
            sc = sc + beta * (I / mi if mi > eps else torch.zeros_like(I))
        out[name] = torch.where(keep, sc, torch.full_like(sc, -1.0))
    return out


# ------------------------------------------------------------------------------ layer-wise reconstruction

def refit_modules(model: nn.Module, X_probe: torch.Tensor) -> list[str]:
    """Names of the prunable Conv1d / Conv2d / Linear modules (head excluded) whose forward is actually
    called, in call order (one forward of X_probe with grad enabled, so nn.TransformerEncoderLayer takes
    its slow path and calls linear1 / linear2; MultiheadAttention's out_proj is applied without its
    forward and is therefore not listed)."""
    head = getattr(model, "head", None)
    cand = {n: mod for n, mod in model.named_modules()
            if isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Linear)) and mod is not head}
    order, hooks = [], []
    for n, mod in cand.items():
        hooks.append(mod.register_forward_hook(lambda _m, _i, _o, n=n: order.append(n) if n not in order else None))
    was = model.training
    model.eval()
    try:
        with torch.enable_grad():
            model(X_probe)
    finally:
        for h in hooks:
            h.remove()
        model.train(was)
    return order


def _as_rows(mod: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """The layer's input as (samples, features) in the weight's flattened column order."""
    if isinstance(mod, nn.Linear):
        return x.reshape(-1, x.shape[-1])
    if isinstance(mod, nn.Conv1d):
        x = x.unsqueeze(-1)
        k, st, pd, dl = (mod.kernel_size[0], 1), (mod.stride[0], 1), (mod.padding[0], 0), (mod.dilation[0], 1)
    else:
        k, st, pd, dl = mod.kernel_size, mod.stride, mod.padding, mod.dilation
    if mod.groups != 1 or isinstance(pd, str):
        raise NotImplementedError(f"reconstruction supports groups=1 and numeric padding only ({mod})")
    cols = F.unfold(x, k, dilation=dl, padding=pd, stride=st)          # (B, C_in * prod(k), L)
    return cols.transpose(1, 2).reshape(-1, cols.shape[1])


def _out_rows(mod: nn.Module, y: torch.Tensor) -> torch.Tensor:
    if isinstance(mod, nn.Linear):
        return y.reshape(-1, y.shape[-1])
    return y.reshape(y.shape[0], y.shape[1], -1).transpose(1, 2).reshape(-1, y.shape[1])


def ridge_refit_rows(H: torch.Tensor, G: torch.Tensor, keep: torch.Tensor, has_bias: bool, damp: float,
                     old: torch.Tensor | None = None) -> torch.Tensor:
    """Solve, per output row i, min ||w_i^T x + b_i - y_i||^2 + lam ||w_i||^2 over the kept columns of row i.
    H = [X 1]^T [X 1] ((d+1) x (d+1) with bias, else d x d), G = [X 1]^T Y ((d(+1)) x C_out), keep (C_out, d)
    bool. lam = damp * mean diag(X^T X) (bias not damped). Returns (C_out, d(+1)) float64, zeros off-support.
    A kept column whose input is identically zero on the calibration set (diag(H) = 0, e.g. a dead unit
    upstream; relative energy < 1e-12) has no effect on the fit; it is left out of the solve and keeps its value from `old`
    ((C_out, d(+1)); 0 if old is None), so the mask and n_nonzero do not change."""
    d = keep.shape[1]
    lam = damp * float(torch.diagonal(H)[:d].mean())
    lam = max(lam, 1e-10)
    diag = torch.diagonal(H)[:d]
    live = diag > 1e-12 * float(diag.max()) if d and float(diag.max()) > 0 else torch.zeros(d, dtype=torch.bool,
                                                                                            device=H.device)
    out = torch.zeros(keep.shape[0], H.shape[0], dtype=torch.float64, device=H.device)
    if old is not None:
        out[:, :d] = old[:, :d].to(out) * (keep & ~live[None, :])
    for i in range(keep.shape[0]):
        S = torch.nonzero(keep[i] & live).flatten()
        if has_bias:
            S = torch.cat([S, torch.tensor([d], device=S.device)])
        if len(S) == 0:
            continue
        Hs = H[S][:, S].clone()
        idx = torch.arange(len(S) - (1 if has_bias else 0), device=H.device)
        Hs[idx, idx] += lam
        out[i, S] = torch.linalg.solve(Hs, G[S, i])
    return out


@torch.no_grad()
def _stats(pruned, original, name, X_cal, batch_size):
    mp, mo = dict(pruned.named_modules())[name], dict(original.named_modules())[name]
    cap = {}
    h1 = mp.register_forward_pre_hook(lambda _m, inp: cap.__setitem__("x", inp[0].detach()))
    h2 = mo.register_forward_hook(lambda _m, _i, out: cap.__setitem__("y", out.detach()))
    H = G = None
    YY = 0.0
    try:
        for i in range(0, len(X_cal), batch_size):
            xb = X_cal[i:i + batch_size]
            with torch.enable_grad():          # slow path of TransformerEncoderLayer (see refit_modules)
                pruned(xb)
                original(xb)
            Xr = _as_rows(mp, cap["x"]).double()
            if mp.bias is not None:
                Xr = torch.cat([Xr, torch.ones(len(Xr), 1, dtype=Xr.dtype, device=Xr.device)], 1)
            Yr = _out_rows(mo, cap["y"]).double()
            H = Xr.T @ Xr if H is None else H + Xr.T @ Xr
            G = Xr.T @ Yr if G is None else G + Xr.T @ Yr
            YY += float((Yr * Yr).sum())
    finally:
        h1.remove()
        h2.remove()
    return mp, H, G, YY


def layerwise_reconstruct(pruned: nn.Module, original: nn.Module, X_cal: torch.Tensor,
                          damp: float = ROUND3["damp"], batch_size: int = 256) -> dict:
    """In place on `pruned` (eval mode): for every refit module in forward order, refit its kept weights
    and bias by ridge least squares so that its output on the pruned model's own inputs matches the
    original model's layer output on the original inputs (calibration windows X_cal). Masks are fixed:
    pruned weights stay exactly 0 and n_nonzero is unchanged (asserted). Returns a log with the relative
    output error of each layer before and after the refit."""
    masks = pruned.prune_masks
    pruned.eval()
    original.eval()
    names = refit_modules(pruned, X_cal[:min(4, len(X_cal))])
    nnz0 = size_record(pruned)["n_nonzero"]
    log = {"refit": [], "not_refit": [n for n, _, _ in prunable_tensors(pruned)
                                     if n.rsplit(".", 1)[0] not in names]}
    for name in names:
        mp, H, G, YY = _stats(pruned, original, name, X_cal, batch_size)
        wname = f"{name}.weight"
        W = mp.weight
        keep = masks.get(wname, torch.ones_like(W, dtype=torch.bool)).to(W.device).reshape(W.shape[0], -1)
        has_bias = mp.bias is not None
        theta = torch.cat([W.detach().reshape(W.shape[0], -1),
                           mp.bias.detach()[:, None]], 1).double() if has_bias else W.detach().reshape(W.shape[0], -1).double()

        def err(th):     # relative output error ||[X 1] th^T - Y||^2 / ||Y||^2 from the Gram matrices
            return float(((th @ H @ th.T).trace() - 2 * (th * G.T).sum() + YY) / max(YY, 1e-30))
        sol = ridge_refit_rows(H, G, keep, has_bias, damp, old=theta)
        e_before, e_after = err(theta), err(sol)
        with torch.no_grad():
            new = sol[:, :keep.shape[1]].reshape(W.shape).to(W.dtype)
            # a kept weight whose refit value is exactly 0 in float32 (underflow) keeps its old value, so the
            # mask and n_nonzero stay fixed
            km = keep.reshape(W.shape)
            zero_kept = km & (new == 0)
            new = torch.where(zero_kept, W.detach(), new) * km.to(W.dtype)
            W.copy_(new)
            if has_bias:
                b_new = sol[:, -1].to(mp.bias.dtype)
                mp.bias.copy_(torch.where(b_new == 0, mp.bias.detach(), b_new))
        log["refit"].append({"module": name, "rel_error_before": e_before, "rel_error_after": e_after,
                             "kept_weights_left_unchanged_zero_refit": int(zero_kept.sum())})
    nnz1 = size_record(pruned)["n_nonzero"]
    if nnz1 != nnz0:
        raise RuntimeError(f"reconstruction changed n_nonzero ({nnz0} -> {nnz1})")
    for n, mk in masks.items():
        p = dict(pruned.named_parameters())[n]
        if torch.any(p[~mk.to(p.device)] != 0):
            raise RuntimeError(f"reconstruction wrote a pruned weight in {n}")
    return log


# ------------------------------------------------------------------------------ iterative pruning

def iterative_prune(original: nn.Module, target: float, *, K: int, beta: float, importance_fn: Callable | None,
                    recon_fn: Callable | None, log: Callable[[str], None] = print) -> tuple[nn.Module, list[dict]]:
    """K steps to `target` (step k at target k / K). At each step: importance_fn(current) -> I (None if
    beta = 0 or no I term), scores = masked_keep_scores(current, I, beta), prune_by_score(current, ..., s_k)
    (masks nested: pruned weights score -1), then recon_fn(current) if given. No gradient training.
    Returns (pruned model with prune_masks, one log row per step)."""
    net, rows = original, []
    for k in range(1, K + 1):
        s_k = target * k / K
        I = importance_fn(net) if (importance_fn is not None and beta > 0) else None
        prev = getattr(net, "prune_masks", None)
        net = prune_by_score(net, masked_keep_scores(net, I, beta), s_k)
        if prev:
            for n, mk in prev.items():
                if torch.any(net.prune_masks[n] & ~mk):
                    raise RuntimeError(f"step {k}: a pruned weight of {n} was revived")
        rec = recon_fn(net) if recon_fn is not None else None
        rows.append({"step": k, "sparsity": s_k, "n_nonzero": size_record(net)["n_nonzero"],
                     "recon": rec["refit"] if rec else None})
        log(f"      prune step {k}/{K}: sparsity {s_k:.3f}, n_nonzero {rows[-1]['n_nonzero']}"
            + (", reconstructed" if rec else ""))
    net.prune_sparsity = target
    return net, rows


# ------------------------------------------------------------------------------ pre-registered rules

def _eligible(st: dict, best: dict, tol: float) -> bool:
    return bool(st["f1_normal"] >= best["f1_normal"] - tol - 1e-12
                and st["f1_older_like"] >= best["f1_older_like"] - tol - 1e-12)


def best_baseline(stats: dict[str, dict], baselines: Sequence[str]) -> dict:
    """B_set = the highest mean F1 on each window set among the baselines present in stats."""
    have = [b for b in stats if any(b == x or b.startswith(x + "(") for x in baselines)]
    if not have:
        raise KeyError(f"no baseline among {baselines} in {list(stats)}")
    return {s: max(stats[b][s] for b in have) for s in ("f1_normal", "f1_older_like")}


def select_config(configs: dict[tuple, dict], best: dict, tol: float = ROUND3["f1_tolerance"],
                  tie: float = ROUND3["tie"]) -> tuple[tuple | None, list[dict]]:
    """Phase-1 rule at one (scenario, level). configs = {(beta, r): {"fa_lci", "f1_normal", "f1_older_like"}}
    (r = None without fine-tuning). Eligible: both F1 >= B_set - tol. Selected: highest fa_lci; ties within
    `tie` -> the smaller beta, then the smaller r. None eligible -> (None, table)."""
    table = [{"beta": b, "r": r, **v, "eligible": _eligible(v, best, tol)} for (b, r), v in configs.items()]
    ok = [t for t in table if t["eligible"]]
    if not ok:
        return None, table
    top = max(t["fa_lci"] for t in ok)
    near = [t for t in ok if t["fa_lci"] >= top - tie - 1e-12]
    pick = min(near, key=lambda t: (t["beta"], -1 if t["r"] is None else t["r"], -t["fa_lci"]))
    for t in table:
        t["selected"] = t is pick
    return (pick["beta"], pick["r"]), table


def select_final(cands: dict[str, dict[float, dict]], best: dict[float, dict], components: dict[str, tuple],
                 tol: float = ROUND3["f1_tolerance"], tie: float = ROUND3["tie"]) -> tuple[str | None, list[dict]]:
    """Phase-2 rule for one scenario. cands = {variant: {level: {"fa_lci", "f1_normal", "f1_older_like"}}},
    best = {level: B_set}. Eligible = eligible at every level; final = the highest mean fa_lci over levels,
    ties within `tie` -> fewer components, then the higher fa_lci. None eligible -> (None, table)."""
    table = []
    for v, by_lv in cands.items():
        row = {"variant": v, "n_components": len(components[v])}
        ok = True
        for lv, st in by_lv.items():
            e = _eligible(st, best[lv], tol)
            row.update({f"fa_lci_{int(round(lv * 100))}": st["fa_lci"], f"eligible_{int(round(lv * 100))}": e})
            ok &= e
        row["fa_lci_mean_levels"] = float(np.mean([st["fa_lci"] for st in by_lv.values()]))
        row["eligible"] = bool(ok)
        table.append(row)
    ok = [t for t in table if t["eligible"]]
    final = None
    if ok:
        top = max(t["fa_lci_mean_levels"] for t in ok)
        near = [t for t in ok if t["fa_lci_mean_levels"] >= top - tie - 1e-12]
        final = min(near, key=lambda t: (t["n_components"], -t["fa_lci_mean_levels"]))["variant"]
    for t in table:
        t["selected"] = t["variant"] == final
    return final, table
