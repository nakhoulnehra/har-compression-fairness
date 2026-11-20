"""XGP, explanation-guided pruning: our method (plan Section 2.5 (b)). HARTH data only.

    keep-score   score(w) = (1 - beta) |w| / mean_layer |w| + beta I(w) / mean_layer I(w)
    I(w)         | w * d/dw sum_i < A_theta(x_i), sg(A_theta0(x_i)) > |  at theta_0, a first-order (Taylor)
                 estimate of how much removing w changes the alignment of the explanation with the
                 original one. A = 48-segment EG attribution of the original predicted class (m = 8),
                 on 1,000 fixed class-stratified HARTH training windows, averaged over 2 EG seeds.
                 The DOT product is on purpose: Pearson / cosine alignment is at its maximum (1) at
                 theta_0, so its gradient there is zero and would score every weight 0
                 (test_pearson_and_cosine_alignment_have_zero_gradient_at_theta0).
    L_cons       (1 - mean_i Pearson(A_theta(x_i), A_theta0(x_i))) + (1 - cos(g_theta, g_theta0)),
                 g = mean over the batch of |A| per segment (the 48-dim global importance profile that
                 LCI's top-k / Jaccard term ranks). EG m = 4; eps in Pearson and cosine for all-zero rows.
    gate         real SHAP on the fold's inner-validation explain windows after pruning, before
                 fine-tuning: LCI_val < tau_k -> CE + gamma L_cons, else CE. E_ft = 10 epochs either way.
    tuning       (beta, gamma) in {0.3, 0.7} x {0.1, 1.0} at 70%, inner validation only: highest LCI_val
                 among configurations with inner-val macro-F1 within 1 pt of L1 (>= L1 - 0.01).
Every attribution gradient runs inside compression.attribution_grad_context (math SDPA, cuDNN off).

Ablations (70% and 90% only, same E_ft, matched n_nonzero): xgp_score_only (keep-score, CE),
l1_plus_lcons (L1 + always-on repair), xgp_no_gate (keep-score + always-on repair), xgp (full).
"""

from __future__ import annotations

import copy
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from .compression import (attribution_grad_context, eg_attributions, l1_scores, park_am_loss, prunable_tensors,
                          segment_attributions)
from .data import Windows, require_harth
from .folds import stratified_pick

XGP = {"n_ref": 1000, "ref_seed": 42, "m_score": 8, "eg_seeds": (0, 1), "score_batch": 125, "m_cons": 4,
       "betas": (0.3, 0.7), "gammas": (0.1, 1.0), "tune_level": 0.7, "ablation_levels": (0.7, 0.9),
       "f1_tolerance": 0.01, "eps": 1e-12}
# Park's equal tuning budget [plan 2.5]: gamma_P in {1, 50} x lr in {1e-4, 3e-4} (4 configurations, as ours).
PARK_TUNING = {"gamma_p": (1.0, 50.0), "lr": (1e-4, 3e-4), "m": 4}

# score: which pruning score; repair: the fine-tuning term ("none", "park", "lcons", "gated_lcons").
METHODS = {
    "l1": {"score": "l1", "repair": "none"},
    "snip": {"score": "snip", "repair": "none"},
    "park": {"score": "l1", "repair": "park"},
    "xgp_score_only": {"score": "keep", "repair": "none"},
    "l1_plus_lcons": {"score": "l1", "repair": "lcons"},
    "xgp_no_gate": {"score": "keep", "repair": "lcons"},
    "xgp": {"score": "keep", "repair": "gated_lcons"},
}
ABLATIONS = ("xgp_score_only", "l1_plus_lcons", "xgp_no_gate", "xgp")


# ------------------------------------------------------------------------------ keep-score

def reference_windows(train: Windows, n: int = XGP["n_ref"], seed: int = XGP["ref_seed"]) -> np.ndarray:
    """Indices (sorted) of the n fixed, class-stratified HARTH training windows the importance uses."""
    require_harth(train)
    idx = stratified_pick(train.y, n, np.random.default_rng(seed))
    if len(idx) < n:
        raise ValueError(f"only {len(idx)} training windows for {n} reference windows")
    return np.sort(idx)


def alignment_importance(model: nn.Module, X: torch.Tensor, target: torch.Tensor, baselines: torch.Tensor,
                         m: int = XGP["m_score"], eg_seeds=XGP["eg_seeds"],
                         batch_size: int = XGP["score_batch"]) -> dict[str, torch.Tensor]:
    """I(w) per prunable tensor, on a deepcopy of `model` in eval mode (`model` is not changed).

    For each EG seed s (one CPU generator, drawn batch after batch): A = 48-segment EG attributions
    with create_graph=True; at theta_0, sg(A_theta0) is A itself (same draws), so the alignment is
    sum_i <A_i, A_i.detach()> and its gradient is summed over the batches. I_s = |w * grad|, and
    I = mean over the seeds of I_s."""
    net = copy.deepcopy(model).eval()
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
            with attribution_grad_context():
                A = segment_attributions(eg_attributions(net, Xb, tb, baselines, m, create_graph=True, generator=g))
                grads = torch.autograd.grad((A * A.detach()).sum(), params, allow_unused=True)
            for a, gr in zip(acc, grads):
                if gr is not None:
                    a += gr.detach()
        for t, p, a in zip(total, params, acc):
            t += (p.detach() * a).abs()
    return {name: t / len(eg_seeds) for (name, _, _), t in zip(named, total)}


def keep_scores(model: nn.Module, importance: dict[str, torch.Tensor], beta: float,
                eps: float = XGP["eps"]) -> dict[str, torch.Tensor]:
    """score(w) = (1 - beta) |w| / mean_layer |w| + beta I(w) / mean_layer I(w), per prunable tensor.
    A tensor whose I is all zero gets only the |w| term (no division by zero)."""
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta {beta} not in [0, 1]")
    out = {}
    for name, w in l1_scores(model).items():
        I = importance[name].detach().to(w.device, w.dtype)
        mw, mi = w.mean(), I.mean()
        term_w = w / mw if mw > eps else torch.zeros_like(w)
        term_i = I / mi if mi > eps else torch.zeros_like(I)
        out[name] = (1 - beta) * term_w + beta * term_i
    return out


def pruning_scores(method: str, model: nn.Module, *, snip: dict | None = None, importance: dict | None = None,
                   beta: float | None = None) -> dict[str, torch.Tensor]:
    """The pruning scores of a method in METHODS (l1, snip or the keep-score)."""
    kind = METHODS[method]["score"]
    if kind == "l1":
        return l1_scores(model)
    if kind == "snip":
        if snip is None:
            raise ValueError(f"{method}: SNIP scores required")
        return snip
    if importance is None or beta is None:
        raise ValueError(f"{method}: the keep-score needs the importance I(w) and beta")
    return keep_scores(model, importance, beta)


# ------------------------------------------------------------------------------ L_cons

def pearson_rows(a: torch.Tensor, b: torch.Tensor, eps: float = XGP["eps"]) -> torch.Tensor:
    """Per-row Pearson r of (B, F) tensors; eps inside the square roots, so an all-zero row gives r = 0
    with finite gradients."""
    ac, bc = a - a.mean(dim=1, keepdim=True), b - b.mean(dim=1, keepdim=True)
    return (ac * bc).sum(1) / (torch.sqrt((ac * ac).sum(1) + eps) * torch.sqrt((bc * bc).sum(1) + eps))


def cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = XGP["eps"]) -> torch.Tensor:
    return (a * b).sum() / (torch.sqrt((a * a).sum() + eps) * torch.sqrt((b * b).sum() + eps))


def lcons_from_attributions(A: torch.Tensor, A0: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """L_cons on 48-segment attributions A (student, with graph) and A0 (original, fixed), (B, 48)."""
    A0 = A0.detach()
    pear = 1 - pearson_rows(A, A0).mean()
    prof = 1 - cosine_sim(A.abs().mean(0), A0.abs().mean(0))
    return pear + prof, {"pearson_term": float(pear.detach()), "profile_term": float(prof.detach())}


def lcons_loss(student: nn.Module, X: torch.Tensor, target: torch.Tensor, A0_seg: torch.Tensor,
               baselines: torch.Tensor, m: int = XGP["m_cons"], generator: torch.Generator | None = None
               ) -> torch.Tensor:
    """L_cons on a batch: the student's 48-segment EG attributions (m draws, create_graph=True) of the
    original model's predicted class vs the original model's fixed ones (A0_seg, (B, 48))."""
    A = segment_attributions(eg_attributions(student, X, target, baselines, m, create_graph=True, generator=generator))
    return lcons_from_attributions(A, A0_seg.to(A.device))[0]


def make_extra_loss(repair: str, *, X: torch.Tensor, target: torch.Tensor, reference: torch.Tensor,
                    baselines: torch.Tensor, gamma: float, m: int, seed: int = 1
                    ) -> Callable[[nn.Module, torch.Tensor], torch.Tensor]:
    """The fine-tuning term for compression.finetune(extra_loss=...): gamma * L_cons ("lcons";
    `reference` = the raw 48-segment A0) or gamma_P * L_AM ("park"; `reference` = unit-norm targets).
    One EG generator seeded with `seed`, drawn batch after batch. X, target, reference on the device."""
    g = torch.Generator().manual_seed(seed)
    if repair == "lcons":
        return lambda net, idx: gamma * lcons_loss(net, X[idx], target[idx], reference[idx], baselines, m=m, generator=g)
    if repair == "park":
        return lambda net, idx: gamma * park_am_loss(net, X[idx], target[idx], reference[idx], baselines, m=m, generator=g)
    raise ValueError(repair)


# ------------------------------------------------------------------------------ gate and tuning

def gate_branch(lci_val: float, tau: float) -> str:
    """The gate [plan 2.5 (b)]: the penalty branch exactly when LCI_val < tau_k."""
    return "ce_plus_lcons" if lci_val < tau else "ce"


def select_config(rows: list[dict], l1_f1: float, f1_key: str = "val_f1", lci_key: str = "lci_val",
                  tol: float = XGP["f1_tolerance"]) -> tuple[dict, bool]:
    """Tuning rule [plan 2.5]: the highest LCI_val among configurations with inner-val macro-F1 >= the L1
    baseline's - tol (1 pt). If none qualifies (not covered by the plan; our choice): the highest F1,
    flagged as a fallback. Returns (chosen row, fallback)."""
    ok = [r for r in rows if r[f1_key] >= l1_f1 - tol - 1e-12]
    if ok:
        return max(ok, key=lambda r: r[lci_key]), False
    return max(rows, key=lambda r: r[f1_key]), True
