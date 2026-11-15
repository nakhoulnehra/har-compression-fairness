"""ResidualMLP teacher (spec 2.2) and the small KD student (spec 2.4c)."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, h: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h, h), nn.BatchNorm1d(h),
        )

    def forward(self, x):
        return F.relu(self.net(x) + x)


class ResidualMLP(nn.Module):
    def __init__(self, d: int, h: int = 128, n_classes: int = 2):
        super().__init__()
        self.proj = nn.Linear(d, h)
        self.r1 = ResBlock(h)
        self.r2 = ResBlock(h)
        self.head = nn.Linear(h, n_classes)

    def forward(self, x):
        return self.head(self.r2(self.r1(F.relu(self.proj(x)))))


def make_student(d: int, n_classes: int = 2) -> nn.Module:
    return nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, n_classes))


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def n_nonzero(model: nn.Module) -> int:
    return int(sum((p != 0).sum().item() for p in model.parameters()))
