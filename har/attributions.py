"""SHAP attributions for one target output per window (plan Section 2.6).

So far only what the model-choice step needs: the single-output GradientExplainer call used to
measure SHAP time per window. The cache and the differentiable surrogate come with later steps.
"""

from __future__ import annotations

import numpy as np
import shap
import torch
import torch.nn as nn


class SingleOutput(nn.Module):
    """x -> logits[:, [c]], so GradientExplainer computes gradients for one output only."""

    def __init__(self, model: nn.Module, c: int):
        super().__init__()
        self.model, self.c = model, int(c)

    def forward(self, x):
        return self.model(x)[:, [self.c]]


def _as_array(sv) -> np.ndarray:
    if isinstance(sv, list):
        sv = sv[0]
    sv = np.asarray(sv)
    return sv[..., 0] if sv.ndim == 4 else sv


def shap_target_class(model: nn.Module, background: torch.Tensor, X: torch.Tensor, target: np.ndarray,
                      shap_seed: int = 0, nsamples: int = 200) -> np.ndarray:
    """(n, 6, 128) SHAP values of the target-class logit, one GradientExplainer call per class group,
    seeded with shap_seed + class, reassembled in the original order."""
    model.eval()
    target = np.asarray(target)
    out = np.zeros(tuple(X.shape), dtype=np.float32)
    for c in np.unique(target):
        idx = np.flatnonzero(target == c)
        explainer = shap.GradientExplainer(SingleOutput(model, c), background)
        np.random.seed(shap_seed + int(c))
        torch.manual_seed(shap_seed + int(c))
        out[idx] = _as_array(explainer.shap_values(X[idx], nsamples=nsamples))
    return out
