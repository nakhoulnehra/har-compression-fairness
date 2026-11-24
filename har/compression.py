"""Compression methods and the attribution surrogate (plan Section 2.5).

    prunable_tensors      the weight tensors pruning and PTQ act on (head excluded unless asked)
    l1_scores, prune_by_score, apply_masks, size_record, finetune
                          uniform per-tensor unstructured pruning and the shared fine-tuning loop
    KD, kd_kl_term, kd_loss, predict_logits
                          the shared KD fine-tuning recipe (XGP pilot round 2): every method gets it
    ptq_symmetric         per-tensor symmetric weight PTQ, the primary PTQ            [plan 2.5, Q1]
    ptq_per_channel, low_bit_size
                          per-output-channel symmetric PTQ and its storage (values, scales, bitmask), for
                          the low-bit diagnostic (har/run_diagnostic_low_bit.py)
    snip_scores           SNIP saliency |w * dL/dw| on the trained model             [Lee et al. 2019; plan Q2]
    eg_attributions       differentiable expected-gradients surrogate of SHAP        [plan 2.5]
    attribution_grad_context  math SDPA + cuDNN off, for every gradient through attributions
    segment_attributions  (B, 6, 128) -> (B, 48) signed sums, channel-major           [plan 2.6]
    teacher_segment_attributions  the original model's fixed 48-segment EG attributions
    park_teacher_attributions, park_am_distance, park_am_loss
                          L1 + attribution matching, adapted                         [Park Sec 3.2, Eq. 2]
    ptq_tulla             Tulla-style PTQ = lci.compression.ptq unchanged (secondary)  [plan 2.5, Q1]
Our method (XGP: keep-score, L_cons, gate) is in har/xgp.py.

Pruning masks (deviation from the plan's torch.nn.utils.prune.custom_from_mask, decided 2026-11-20):
the masks are a plain dict `model.prune_masks` {tensor name: bool keep-mask}, applied in place to the
parameters. `prune.custom_from_mask` re-computes `weight` in a forward pre-hook of the owning module,
and nn.MultiheadAttention reads `out_proj.weight` without calling out_proj's forward, so the hook
would never refresh it. In-place masks work for every module (LSTM flat weights included);
`finetune` zeroes the masked gradients before and re-applies the masks after every optimizer step.
"""

from __future__ import annotations

import contextlib
import copy
import time
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.nn.attention import SDPBackend, sdpa_kernel

from .data import Windows, require_harth
from .train import class_weights, pooled_macro_f1, predict_proba, set_seed

PRUNABLE_MODULES = (nn.Conv1d, nn.Conv2d, nn.Linear)   # nn.MultiheadAttention.out_proj is an nn.Linear
SEG_LEN = 16
E_FT = 10            # fine-tuning epochs, every pruning method [plan 2.5]
FT_LR = 1e-4         # Adam [plan 2.5]
FT_BATCH = 256
# The shared KD fine-tuning recipe (XGP pilot round 2, plan 2.5): every method gets it, it is not part
# of XGP. alpha KL(T) T^2 to the original model's logits + (1 - alpha) class-weighted CE on the labels,
# as in lci.compression.distill.
KD = {"T": 3.0, "alpha": 0.7}


# ------------------------------------------------------------------------------ scope

def _head(model: nn.Module) -> nn.Module | None:
    return getattr(model, "head", None)


def head_unpruned(model: nn.Module) -> bool:
    """Step 9 acceptance (fixed 2026-11-22): the classification head carries no pruning mask and none of
    its weights is zero. (The head is trained during fine-tuning, so "unchanged" is the wrong test.)"""
    head = _head(model)
    masks = getattr(model, "prune_masks", None) or {}
    return bool(not any(n.endswith("head.weight") for n in masks) and torch.all(head.weight.detach() != 0))


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


# ------------------------------------------------------------------------------ pruning

def l1_scores(model: nn.Module) -> dict[str, torch.Tensor]:
    """|w| per prunable tensor (L1 magnitude pruning)."""
    return {name: getattr(mod, p).detach().abs().clone() for name, mod, p in prunable_tensors(model)}


def prune_by_score(model: nn.Module, scores: dict[str, torch.Tensor], sparsity: float) -> nn.Module:
    """A deepcopy of `model` with, per prunable tensor, the round(sparsity * numel) weights of lowest
    score set to zero (ties -> lowest flat index first) [plan 2.5]. Uniform per-tensor sparsity, so
    every method gets the same non-zero count at a level. The keep-masks are stored in
    `pruned.prune_masks` and re-applied by `finetune`."""
    names = [n for n, _, _ in prunable_tensors(model)]
    if set(scores) != set(names):
        raise ValueError(f"scores must cover exactly the prunable tensors; missing {sorted(set(names) - set(scores))}, "
                         f"extra {sorted(set(scores) - set(names))}")
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity {sparsity} not in [0, 1)")
    pruned = copy.deepcopy(model)
    masks = {}
    with torch.no_grad():
        for name, mod, p in prunable_tensors(pruned):
            w = getattr(mod, p)
            sc = scores[name].detach().to("cpu", torch.float64).flatten()
            if sc.numel() != w.numel():
                raise ValueError(f"{name}: score shape {tuple(scores[name].shape)} vs weight {tuple(w.shape)}")
            k = int(round(sparsity * w.numel()))
            order = torch.argsort(sc, stable=True)
            keep = torch.ones(w.numel(), dtype=torch.bool)
            keep[order[:k]] = False
            masks[name] = keep.reshape(w.shape)
            w.mul_(masks[name].to(w.device, w.dtype))
    pruned.prune_masks = masks
    pruned.prune_sparsity = sparsity
    return pruned


def _masked_params(model: nn.Module) -> list[tuple[torch.Tensor, torch.Tensor]]:
    masks = getattr(model, "prune_masks", None) or {}
    out = []
    for name, mod, p in prunable_tensors(model):
        if name in masks:
            w = getattr(mod, p)
            out.append((w, masks[name].to(w.device)))
    return out


def apply_masks(model: nn.Module) -> None:
    """Zero the pruned weights in place (no-op for an unpruned model)."""
    with torch.no_grad():
        for w, keep in _masked_params(model):
            w.mul_(keep.to(w.dtype))


def size_record(model: nn.Module, quantized: dict[str, int] | None = None) -> dict:
    """Sizes and sparsities [plan 2.5]: n_params, n_nonzero (over all parameters), sparsity over the
    prunable tensors (= the nominal level), over all weight tensors incl. the head, over all
    parameters, per layer, and dense / nominal size in KB (pruned: n_nonzero x 4 B, index overhead not
    counted; PTQ: the quantized tensors at `bits` per value + the rest at 4 B)."""
    params = dict(model.named_parameters())
    n_params = sum(p.numel() for p in params.values())
    n_nonzero = int(sum(int((p != 0).sum()) for p in params.values()))
    prun = [n for n, _, _ in prunable_tensors(model)]
    with_head = [n for n, _, _ in prunable_tensors(model, include_head=True)]
    zeros = lambda names: sum(int((params[n] == 0).sum()) for n in names)
    size = lambda names: sum(params[n].numel() for n in names)
    if quantized:
        q_values = sum(params[n].numel() * b / 8 for n, b in quantized.items())
        nominal = q_values + 4 * (n_params - size(list(quantized)))
    else:
        nominal = 4 * n_nonzero
    return {"n_params": n_params, "n_nonzero": n_nonzero,
            "sparsity_prunable": zeros(prun) / size(prun),
            "sparsity_weights_incl_head": zeros(with_head) / size(with_head),
            "sparsity_all_params": 1 - n_nonzero / n_params,
            "per_layer_sparsity": {n: zeros([n]) / size([n]) for n in prun},
            "size_kb_dense": n_params * 4 / 1024, "size_kb_nominal": nominal / 1024}


def kd_kl_term(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float = KD["T"]) -> torch.Tensor:
    """KL(softmax(teacher / T) || softmax(student / T)), batch mean (without the T^2 factor)."""
    return F.kl_div(F.log_softmax(student_logits / T, 1), F.softmax(teacher_logits.detach() / T, 1),
                    reduction="batchmean")


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, y: torch.Tensor, ce_fn: nn.Module,
            T: float = KD["T"], alpha: float = KD["alpha"]) -> torch.Tensor:
    """The shared KD recipe: alpha KL(T) T^2 + (1 - alpha) CE(student, y), with ce_fn the class-weighted CE."""
    return alpha * kd_kl_term(student_logits, teacher_logits, T) * T * T + (1 - alpha) * ce_fn(student_logits, y)


@torch.no_grad()
def predict_logits(model: nn.Module, X: np.ndarray, device: str = "cpu", batch_size: int = 1024) -> np.ndarray:
    """Logits in eval mode, (n, n_classes) float32 (the KD teacher outputs)."""
    model.eval()
    dev = torch.device(device)
    out = [model(torch.from_numpy(np.ascontiguousarray(X[i:i + batch_size])).to(dev)).float().cpu().numpy()
           for i in range(0, len(X), batch_size)]
    return np.concatenate(out)


def finetune(model: nn.Module, train: Windows, val: Windows, *, epochs: int = E_FT, lr: float = FT_LR,
             batch_size: int = FT_BATCH, seed: int = 0, device: str = "cpu",
             extra_loss: Callable[[nn.Module, torch.Tensor], torch.Tensor] | None = None,
             kd_teacher_logits: torch.Tensor | None = None,
             groups: np.ndarray | None = None, dro_eta: float | None = None,
             on_epoch_end: Callable[[int, nn.Module], None] | None = None,
             log: Callable[[str], None] = print) -> dict:
    """The fine-tuning loop shared by every pruning method [plan 2.5]: exactly `epochs` epochs (no early
    stopping; the last state is kept), Adam(lr), class-weighted CE from `train`, batch order from a
    generator seeded with `seed`. `extra_loss(model, idx)` adds an attribution term for the batch
    `idx` (indices into the training windows, on `device`); following Park's convention (Step 6), CE
    is computed in train mode and the extra term in eval mode, so BatchNorm running statistics are not
    updated by EG's interpolated inputs. Masked gradients are zeroed before each step and the masks
    re-applied after it, so pruned weights stay exactly zero. Inner-validation pooled 7-class macro-F1
    is logged after every epoch. The model is left on `device` in eval mode.
    With kd_teacher_logits ((n_train, n_classes), the original model's logits on `train`), the
    supervised term is the shared KD recipe kd_loss (KD["alpha"] KL(T) T^2 + (1 - alpha) CE) instead of
    CE; `train_ce` then logs the CE part and `train_kd` the KL part.
    Group DRO (XGP round 3; Sagawa et al. 2020, Algorithm 1): with `groups` (one int per training
    window, 0..G-1) and `dro_eta`, the batch loss is sum_g q_g l_g over the groups present in the batch
    (renormalised), l_g = the supervised loss (CE or the KD recipe) on group g's windows, and before
    each step q_g <- q_g exp(dro_eta l_g), normalised; q starts uniform. q is logged per epoch.
    `on_epoch_end(epoch_1based, model)` is called after each epoch's validation (e.g. the round-3
    budget curve); every RNG state is saved before and restored after it, so the training run is
    identical with or without the callback."""
    require_harth(train, val)
    if kd_teacher_logits is not None and tuple(kd_teacher_logits.shape)[:1] != (len(train),):
        raise ValueError(f"kd_teacher_logits must have one row per training window ({len(train)}), "
                         f"got {tuple(kd_teacher_logits.shape)}")
    if (groups is None) != (dro_eta is None):
        raise ValueError("group DRO needs both groups and dro_eta")
    dev = torch.device(device)
    model.to(dev)
    set_seed(seed)
    apply_masks(model)
    masked = _masked_params(model)
    Xtr, ytr = torch.from_numpy(train.X).to(dev), torch.from_numpy(train.y).to(dev)
    cw = class_weights(train.y).to(dev)
    loss_fn = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    zt = kd_teacher_logits.to(dev) if kd_teacher_logits is not None else None
    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape != (len(train),):
            raise ValueError(f"groups must have one entry per training window ({len(train)}), got {groups.shape}")
        gtr = torch.from_numpy(groups.astype(np.int64)).to(dev)
        n_groups = int(groups.max()) + 1
        q = torch.full((n_groups,), 1.0 / n_groups, dtype=torch.float64, device=dev)
    g = torch.Generator().manual_seed(seed)
    n, history, t_start = len(train), [], time.perf_counter()
    for epoch in range(epochs):
        t0 = time.perf_counter()
        perm = torch.randperm(n, generator=g).to(dev)
        tot_ce = tot_extra = tot_kd = 0.0
        seen = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:          # BatchNorm cannot train on a batch of one
                continue
            opt.zero_grad(set_to_none=True)
            model.train()
            z = model(Xtr[idx])
            ce = loss_fn(z, ytr[idx])
            if zt is None:
                loss = ce
            else:
                kl = kd_kl_term(z, zt[idx], KD["T"])
                loss = KD["alpha"] * kl * KD["T"] ** 2 + (1 - KD["alpha"]) * ce
                tot_kd += float(kl.detach()) * len(idx)
            if groups is not None:
                loss = _dro_loss(z, ytr[idx], zt[idx] if zt is not None else None, gtr[idx], cw, q, dro_eta)
            if extra_loss is not None:
                model.eval()
                ex = extra_loss(model, idx)
                loss = loss + ex
                tot_extra += float(ex.detach()) * len(idx)
            loss.backward()
            with torch.no_grad():
                for w, keep in masked:
                    if w.grad is not None:
                        w.grad.mul_(keep.to(w.grad.dtype))
            opt.step()
            apply_masks(model)
            tot_ce += float(ce.detach()) * len(idx)
            seen += len(idx)
        pred = predict_proba(model, val.X, device).argmax(axis=1)
        f1, acc = pooled_macro_f1(val.y, pred), float(accuracy_score(val.y, pred))
        dt = time.perf_counter() - t0
        history.append({"epoch": epoch, "train_ce": tot_ce / max(seen, 1),
                        "train_extra": tot_extra / max(seen, 1) if extra_loss is not None else None,
                        "train_kd": tot_kd / max(seen, 1) if zt is not None else None,
                        "val_macro_f1_pooled7": f1, "val_accuracy": acc, "seconds": dt,
                        **({f"dro_q{j}": float(q[j]) for j in range(len(q))} if groups is not None else {})})
        log(f"    ft epoch {epoch + 1:2d}/{epochs}  CE {tot_ce / max(seen, 1):.4f}"
            + (f"  KL {tot_kd / max(seen, 1):.4f}" if zt is not None else "")
            + (f"  extra {tot_extra / max(seen, 1):.4f}" if extra_loss is not None else "")
            + (f"  DRO q {[round(float(v), 3) for v in q]}" if groups is not None else "")
            + f"  val F1 {f1:.4f}  acc {acc:.4f}  {dt:.1f}s")
        if on_epoch_end is not None:
            states = _rng_states()
            on_epoch_end(epoch + 1, model)
            _restore_rng_states(states)
            model.to(dev)
    model.eval()
    return {"epochs_run": len(history), "history": history, "seconds": time.perf_counter() - t_start,
            "val_macro_f1_pooled7": history[-1]["val_macro_f1_pooled7"] if history else None}


def _dro_loss(z, y, zt, grp, cw, q, eta) -> torch.Tensor:
    """Online group DRO step (Sagawa et al. 2020, Algorithm 1) on one batch; updates q in place.
    l_g = (1 - alpha) class-weighted CE (weighted mean over g's windows, as nn.CrossEntropyLoss(weight))
    + alpha T^2 KL (mean over g's windows) when zt is given, else the class-weighted CE alone."""
    ce_i = F.cross_entropy(z, y, reduction="none")
    w_i = cw[y]
    if zt is not None:
        T = KD["T"]
        p_t = F.softmax(zt.detach() / T, 1)
        kl_i = (p_t * (torch.log(p_t.clamp_min(1e-30)) - F.log_softmax(z / T, 1))).sum(1)
    losses, present = [], []
    for j in range(len(q)):
        m = grp == j
        if not bool(m.any()):
            continue
        l_g = (w_i[m] * ce_i[m]).sum() / w_i[m].sum()
        if zt is not None:
            l_g = KD["alpha"] * kl_i[m].mean() * T * T + (1 - KD["alpha"]) * l_g
        losses.append(l_g)
        present.append(j)
    with torch.no_grad():
        for j, l_g in zip(present, losses):
            q[j] = q[j] * torch.exp(eta * l_g.detach().double())
        q /= q.sum()
    qp = q[present].to(z.dtype)
    return (qp * torch.stack(losses)).sum() / qp.sum()


def _rng_states() -> dict:
    import random
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng_states(s: dict) -> None:
    import random
    torch.set_rng_state(s["torch"])
    np.random.set_state(s["numpy"])
    random.setstate(s["python"])
    if s["cuda"] is not None:
        torch.cuda.set_rng_state_all(s["cuda"])


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


def ptq_per_channel(model: nn.Module, bits: int, only: Iterable[str] | None = None
                    ) -> tuple[nn.Module, dict[str, int]]:
    """Per-output-channel symmetric weight-only PTQ (low-bit diagnostic, 2026-11-24) on a deepcopy: the same
    tensors as ptq_symmetric (every prunable tensor plus the head), or only the tensors named in `only`.
    Per row r of W reshaped to (out, -1) (conv filter, linear / LSTM gate / attention projection row):
    s_r = max|W_r| / (2^(b-1) - 1), W_q = clamp(round(W / s_r), -(2^(b-1) - 1), 2^(b-1) - 1) * s_r. A row of
    zeros stays zero, and pruned weights (0) stay exactly 0, so prune_masks remain valid. Fake quantization
    (stored as float); biases, normalization parameters and activations untouched.
    Returns (quantized model in eval mode, {tensor name: bits})."""
    q = copy.deepcopy(model)
    qmax = 2 ** (bits - 1) - 1
    if qmax < 1:
        raise ValueError(f"bits must be >= 2 for symmetric quantization, got {bits}")
    names = {n for n, _, _ in prunable_tensors(q, include_head=True)}
    sel = names if only is None else set(only)
    if not sel <= names:
        raise KeyError(f"not quantizable: {sorted(sel - names)}")
    quantized = {}
    with torch.no_grad():
        for name, mod, pname in prunable_tensors(q, include_head=True):
            if name not in sel:
                continue
            w = getattr(mod, pname)
            W = w.reshape(w.shape[0], -1)
            s = W.abs().amax(1, keepdim=True) / qmax
            s = torch.where(s > 0, s, torch.ones_like(s))
            w.copy_((torch.clamp(torch.round(W / s), -qmax, qmax) * s).reshape(w.shape))
            quantized[name] = bits
    q.eval()
    return q, quantized


def low_bit_size(model: nn.Module, quantized: dict[str, int] | None = None, scale_bits: int = 32) -> dict:
    """Storage of a pruned and / or per-channel quantized model (low-bit diagnostic). Per parameter tensor:
    stored values = the kept weights for a tensor in `model.prune_masks` (all values otherwise), at `bits`
    if quantized, else 32; a quantized tensor adds one `scale_bits` scale per output row. KB = 1024 B.
        size_kb               values + scales (no index overhead, as size_kb_nominal in the plan)
        size_kb_with_bitmask  + 1 bit per position of every pruned tensor (a simple sparse index)
        size_kb_dense_fp32    n_params x 4 B;  compression_ratio = dense / size_kb_with_bitmask"""
    quantized = quantized or {}
    masks = getattr(model, "prune_masks", None) or {}
    params = dict(model.named_parameters())
    if not set(quantized) <= set(params):
        raise KeyError(f"unknown quantized tensors: {sorted(set(quantized) - set(params))}")
    values = scales = mask = 0
    for n, p in params.items():
        stored = int(masks[n].sum()) if n in masks else p.numel()
        values += stored * quantized.get(n, 32)
        if n in quantized:
            scales += int(p.shape[0]) * scale_bits
        if n in masks:
            mask += p.numel()
    kb = lambda bits: bits / 8 / 1024  # noqa: E731
    dense = kb(32 * sum(p.numel() for p in params.values()))
    with_mask = kb(values + scales + mask)
    return {"size_kb": kb(values + scales), "size_kb_with_bitmask": with_mask, "size_kb_values": kb(values),
            "size_kb_scales": kb(scales), "size_kb_bitmask": kb(mask), "size_kb_dense_fp32": dense,
            "compression_ratio": dense / with_mask}


def ptq_tulla(model: nn.Module, bits: int) -> tuple[nn.Module, dict[str, int]]:
    """Tulla-style PTQ (secondary, always labelled so) [plan 2.5, Q1]: lci.compression.ptq unchanged, i.e.
    every learnable parameter (normalization parameters and biases included) rounded to a fixed
    1 / (2^(b-1) - 1) grid, no scale, no clamping. Returns (model in eval mode, {parameter name: bits})."""
    from lci.compression import ptq

    q, _ = ptq(model, bits)
    return q, {n: bits for n, _ in q.named_parameters()}


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
    with attribution_grad_context():
        return _eg(model, X, target, baselines, m, create_graph, generator)


@contextlib.contextmanager
def attribution_grad_context():
    """Math SDPA kernel + cuDNN off: wrap every gradient taken through attributions (EG forward, and the
    backward of a loss on them when the model is in eval mode). See eg_attributions."""
    with sdpa_kernel(SDPBackend.MATH), torch.backends.cudnn.flags(enabled=False):
        yield


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

def teacher_segment_attributions(teacher: nn.Module, X: torch.Tensor, target: torch.Tensor,
                                 baselines: torch.Tensor, m: int = 4, seed: int = 0,
                                 batch_size: int = 256) -> torch.Tensor:
    """The original model's fixed 48-segment EG attributions (m draws, seeded generator) of the original
    model's predicted class, computed once in eval mode without a graph. (N, 48), signed, not normalised.
    Shared by Park's targets (unit-normalised) and XGP's L_cons reference."""
    was_training = teacher.training
    teacher.eval()
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(0, len(X), batch_size):
        A = eg_attributions(teacher, X[i:i + batch_size], target[i:i + batch_size], baselines, m,
                            create_graph=False, generator=g)
        out.append(segment_attributions(A))
    teacher.train(was_training)
    return torch.cat(out)


def park_teacher_attributions(teacher: nn.Module, X: torch.Tensor, target: torch.Tensor, baselines: torch.Tensor,
                              m: int = 4, seed: int = 0, batch_size: int = 256) -> torch.Tensor:
    """The original model's fixed matching targets [plan 2.5 (a)]: unit-L2-norm 48-segment EG attributions
    (m draws) of the original model's top-1 class, computed once in eval mode without a graph. (N, 48)."""
    return unit_norm(teacher_segment_attributions(teacher, X, target, baselines, m, seed, batch_size))


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
