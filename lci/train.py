"""Seeding, teacher training (spec 2.3) and evaluation."""

from __future__ import annotations

import copy
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, seed: int) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g)


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    return model(torch.from_numpy(X)).argmax(1).numpy()


def evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray) -> dict:
    p = predict(model, X)
    return {"acc": float(accuracy_score(y, p)), "f1_macro": float(f1_score(y, p, average="macro"))}


def train_teacher(model: nn.Module, X_tr, y_tr, X_va, y_va, seed: int,
                  epochs: int = 30, batch_size: int = 512, lr: float = 1e-3, wd: float = 1e-4) -> dict:
    """Train in place; load the best-validation-accuracy state dict at the end."""
    loader = make_loader(X_tr, y_tr, batch_size, seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    best_acc, best_state, best_epoch = -1.0, None, -1
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        sched.step()
        val_acc = evaluate(model, X_va, y_va)["acc"]
        if val_acc > best_acc:
            best_acc, best_state, best_epoch = val_acc, copy.deepcopy(model.state_dict()), epoch
    model.load_state_dict(best_state)
    model.eval()
    return {"best_val_acc": best_acc, "best_epoch": best_epoch}
