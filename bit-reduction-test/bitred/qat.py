"""Quantization-aware KD fine-tuning (QAT) with the straight-through estimator, the standard way to recover a
low-bit model's accuracy. Used by the 2-bit gap test (run_qat.py).

Each step: the quantized tensors' fp32 master weights are replaced by their quantized values (per-channel, b bits,
scales from the current weights), forward + backward run on the quantized model, the masters are restored, and Adam
updates them with that gradient (STE: d loss / d W := d loss / d Q(W)). The supervised loss is the project's shared KD
recipe (har.compression.kd_loss: 0.7 KL(T = 3) T^2 + 0.3 class-weighted CE to the original model's logits), with
Adam, batch 256 and a seeded batch order, as har.compression.finetune. After training the returned model holds the
quantized weights (the deployed model). BatchNorm statistics are updated in train mode on the quantized forward.

Pruned models (model.prune_masks, {tensor name: bool keep-mask}): the masks are fixed. They are applied before
training, the masked gradients are zeroed before every Adam step and the masks are re-applied after it (as
har.compression.finetune), so pruned weights stay exactly 0; the quantizer maps 0 to 0, so the quantized model keeps
them at 0 too (asserted at the end).
"""

from __future__ import annotations

import time
from typing import Callable, Mapping

import numpy as np
import torch
import torch.nn as nn

from har.compression import kd_loss
from har.data import Windows, require_harth
from har.train import class_weights, pooled_macro_f1, predict_proba, set_seed

from .quant import quant_tensors, quantize_model, quantize_weight


def steps_per_epoch(n: int, batch_size: int) -> int:
    """Adam steps in one pass over n windows (a final batch of < 2 windows is skipped, as in the loop)."""
    return sum(1 for i in range(0, int(n), int(batch_size)) if min(int(batch_size), int(n) - i) >= 2)


def qat_kd_finetune(model: nn.Module, train: Windows, val: Windows, teacher_logits: torch.Tensor,
                    bits: Mapping[str, int], *, granularity: str = "channel", epochs: int = 20, lr: float = 1e-4,
                    batch_size: int = 256, seed: int = 0, device: str = "cpu", steps: int | None = None,
                    eval_every_steps: int | None = None,
                    log: Callable[[str], None] = print) -> tuple[nn.Module, list[dict]]:
    """Trains `model` in place (fp32 masters) and returns (quantized copy in eval mode, history).

    steps None: `epochs` passes over `train`, validated after every epoch (the original behaviour).
    steps given: exactly `steps` Adam steps, epochs after epoch (a fresh seeded permutation each), stopping
    mid-epoch if needed; `epochs` is ignored; validated every `eval_every_steps` steps (default one epoch) and
    after the last step. Both modes draw the same batch order, so steps = epochs x steps_per_epoch with
    eval_every_steps = steps_per_epoch reproduces the epoch mode exactly. Validation F1 is the quantized model's."""
    require_harth(train, val)
    if tuple(teacher_logits.shape)[:1] != (len(train),):
        raise ValueError("teacher_logits must have one row per training window")
    n = len(train)
    spe = steps_per_epoch(n, batch_size)
    if spe == 0:
        raise ValueError("need at least 2 training windows")
    total = int(epochs) * spe if steps is None else int(steps)
    every = spe if steps is None else int(eval_every_steps or spe)
    if total < 1 or every < 1:
        raise ValueError(f"steps {total} and eval_every_steps {every} must be >= 1")
    dev = torch.device(device)
    model.to(dev)
    set_seed(seed)
    params = dict(model.named_parameters())
    masks = {n_: m.to(dev) for n_, m in (getattr(model, "prune_masks", None) or {}).items()}
    unknown = set(masks) - set(params)
    if unknown:
        raise KeyError(f"masks for unknown tensors: {sorted(unknown)}")
    mp = [(params[n_], m) for n_, m in masks.items()]
    with torch.no_grad():
        for p, m in mp:
            p.mul_(m.to(p.dtype))
    qp = [(params[n_], int(bits[n_])) for n_, _, _ in quant_tensors(model) if n_ in bits and int(bits[n_]) < 32]
    Xtr, ytr = torch.from_numpy(train.X).to(dev), torch.from_numpy(train.y).to(dev)
    zt = teacher_logits.to(dev)
    ce_fn = nn.CrossEntropyLoss(weight=class_weights(train.y).to(dev))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    history, step, epoch = [], 0, 0
    tot, seen, t0 = 0.0, 0, time.perf_counter()
    while step < total:
        perm = torch.randperm(n, generator=g).to(dev)
        epoch += 1
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                masters = [p.detach().clone() for p, _ in qp]
                for p, b in qp:
                    p.copy_(quantize_weight(p, b, granularity))
            model.train()
            loss = kd_loss(model(Xtr[idx]), zt[idx], ytr[idx], ce_fn)
            loss.backward()
            with torch.no_grad():
                for (p, _), m in zip(qp, masters):
                    p.copy_(m)
                for p, m in mp:
                    if p.grad is not None:
                        p.grad.mul_(m.to(p.grad.dtype))
            opt.step()
            with torch.no_grad():
                for p, m in mp:
                    p.mul_(m.to(p.dtype))
            tot += float(loss.detach()) * len(idx)
            seen += len(idx)
            step += 1
            if step % every == 0 or step == total:
                qm = quantize_model(model, bits, granularity)
                pred = predict_proba(qm, val.X, device).argmax(1)
                f1 = pooled_macro_f1(val.y, pred)
                acc = float((pred == val.y).mean())
                dt = time.perf_counter() - t0
                history.append({"epoch": epoch if steps is None else step / spe, "step": step,
                                "train_loss": tot / max(seen, 1), "val_f1_quantized": f1,
                                "val_accuracy_quantized": acc, "seconds": dt})
                where = (f"epoch {epoch:2d}/{epochs}" if steps is None
                         else f"step {step:5d}/{total} (epoch {step / spe:.1f})")
                log(f"    QAT {where}  loss {tot / max(seen, 1):.4f}  val F1 (quantized) {f1:.4f}  "
                    f"acc {acc:.4f}  {dt:.1f}s")
                tot, seen, t0 = 0.0, 0, time.perf_counter()
            if step == total:
                break
    model.eval()
    qm = quantize_model(model, bits, granularity)
    qparams = dict(qm.named_parameters())
    for n_, m in masks.items():
        if torch.any(qparams[n_].detach()[~m] != 0):
            raise RuntimeError(f"{n_}: a pruned weight is non-zero after QAT")
    return qm, history
