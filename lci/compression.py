"""The three compression families (spec 2.4). Each starts from a deepcopy of the teacher."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune

from .models import make_student, n_nonzero, n_params
from .train import make_loader


def ptq(teacher: nn.Module, bits: int) -> tuple[nn.Module, dict]:
    """Authors' `fake_ptq`: round every learnable parameter to a 1/(2^(b-1)-1) grid.
    No clamping, no calibration, no fine-tuning; BN running stats untouched; stays float."""
    model = copy.deepcopy(teacher)
    scale = float(2 ** (bits - 1) - 1)
    with torch.no_grad():
        for p in model.parameters():
            p.data = torch.round(p.data * scale) / scale
    model.eval()
    n = n_params(model)
    return model, {"n_params": n, "n_nonzero": n_nonzero(model), "eff_size_bytes": n * bits / 8}


def prune_l1(teacher: nn.Module, amount: float) -> tuple[nn.Module, dict]:
    """Local L1-unstructured pruning of every nn.Linear weight (incl. head); no fine-tuning."""
    model = copy.deepcopy(teacher)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            prune.l1_unstructured(m, name="weight", amount=amount)
            prune.remove(m, "weight")
    model.eval()
    n = n_params(model)
    return model, {"n_params": n, "n_nonzero": n_nonzero(model), "eff_size_bytes": n * 4}


def distill(teacher: nn.Module, X_tr, y_tr, d: int, seed: int, T: float = 3.0, alpha: float = 0.7,
            epochs: int = 15, batch_size: int = 512, lr: float = 1e-3) -> tuple[nn.Module, dict]:
    """KD into the 32-16 student: alpha*KL(T)*T^2 + (1-alpha)*CE. Returns the final-epoch student."""
    teacher = copy.deepcopy(teacher).eval()
    torch.manual_seed(seed)
    student = make_student(d)
    loader = make_loader(X_tr, y_tr, batch_size, seed)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    kl = nn.KLDivLoss(reduction="batchmean")
    ce = nn.CrossEntropyLoss()
    for _ in range(epochs):
        student.train()
        for xb, yb in loader:
            with torch.no_grad():
                zt = teacher(xb)
            zs = student(xb)
            loss = alpha * kl(F.log_softmax(zs / T, 1), F.softmax(zt / T, 1)) * T * T + (1 - alpha) * ce(zs, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    student.eval()
    n = n_params(student)
    return student, {"n_params": n, "n_nonzero": n_nonzero(student), "eff_size_bytes": n * 4}
