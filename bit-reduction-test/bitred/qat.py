"""Quantization-aware KD fine-tuning (QAT) with the straight-through estimator, the standard way to recover a
low-bit model's accuracy. Used by the 2-bit gap test (run_qat.py).

Each step: the quantized tensors' fp32 master weights are replaced by their quantized values (per-channel, b bits,
scales from the current weights), forward + backward run on the quantized model, the masters are restored, and Adam
updates them with that gradient (STE: d loss / d W := d loss / d Q(W)). The supervised loss is the project's shared KD
recipe (har.compression.kd_loss: 0.7 KL(T = 3) T^2 + 0.3 class-weighted CE to the original model's logits), with
Adam, batch 256 and a seeded batch order, as har.compression.finetune. After training the returned model holds the
quantized weights (the deployed model). BatchNorm statistics are updated in train mode on the quantized forward.
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


def qat_kd_finetune(model: nn.Module, train: Windows, val: Windows, teacher_logits: torch.Tensor,
                    bits: Mapping[str, int], *, granularity: str = "channel", epochs: int = 20, lr: float = 1e-4,
                    batch_size: int = 256, seed: int = 0, device: str = "cpu",
                    log: Callable[[str], None] = print) -> tuple[nn.Module, list[dict]]:
    """Trains `model` in place (fp32 masters) and returns (quantized copy in eval mode, per-epoch history).
    The per-epoch validation F1 is that of the quantized model."""
    require_harth(train, val)
    if tuple(teacher_logits.shape)[:1] != (len(train),):
        raise ValueError("teacher_logits must have one row per training window")
    dev = torch.device(device)
    model.to(dev)
    set_seed(seed)
    params = dict(model.named_parameters())
    qp = [(params[n], int(bits[n])) for n, _, _ in quant_tensors(model) if n in bits and int(bits[n]) < 32]
    Xtr, ytr = torch.from_numpy(train.X).to(dev), torch.from_numpy(train.y).to(dev)
    zt = teacher_logits.to(dev)
    ce_fn = nn.CrossEntropyLoss(weight=class_weights(train.y).to(dev))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    n, history = len(train), []
    for epoch in range(epochs):
        t0 = time.perf_counter()
        perm = torch.randperm(n, generator=g).to(dev)
        tot, seen = 0.0, 0
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
            opt.step()
            tot += float(loss.detach()) * len(idx)
            seen += len(idx)
        qm = quantize_model(model, bits, granularity)
        pred = predict_proba(qm, val.X, device).argmax(1)
        f1 = pooled_macro_f1(val.y, pred)
        acc = float((pred == val.y).mean())
        dt = time.perf_counter() - t0
        history.append({"epoch": epoch + 1, "train_loss": tot / max(seen, 1), "val_f1_quantized": f1,
                        "val_accuracy_quantized": acc, "seconds": dt})
        log(f"    QAT epoch {epoch + 1:2d}/{epochs}  loss {tot / max(seen, 1):.4f}  val F1 (quantized) {f1:.4f}  "
            f"acc {acc:.4f}  {dt:.1f}s")
    model.eval()
    return quantize_model(model, bits, granularity), history
