"""SHAP attributions for one target output per window (plan Section 2.6).

    shap_target_class   single-output GradientExplainer, one call per target-class group
    shap_cached         the same, cached in data/processed/har/shap-cache/ (git-ignored), keyed by the
                        model weights, window IDs, background, targets, SHAP seed and nsamples
    subject_lci, mean_lci   per-subject LCI on 48 segments (k = 10, omega = 0.5) and its subject mean
The differentiable surrogate (expected gradients) is in har/compression.py.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Sequence

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
        # cuDNN off: its RNN backward only runs in training mode, and the model is in eval mode, so the
        # input gradients through TinyHAR's LSTM fail on CUDA. No effect on CPU.
        with torch.backends.cudnn.flags(enabled=False):
            out[idx] = _as_array(explainer.shap_values(X[idx], nsamples=nsamples))
    return out


# ------------------------------------------------------------------------------ cache

def state_hash(model: nn.Module) -> str:
    """sha256 of the model's state dict (names, shapes, values), 16 hex characters."""
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(np.ascontiguousarray(v.detach().cpu().numpy()).tobytes())
    return h.hexdigest()[:16]


def array_hash(a) -> str:
    a = a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)
    return hashlib.sha256(np.ascontiguousarray(a).tobytes() + str(a.dtype).encode()).hexdigest()[:16]


def shap_cached(model: nn.Module, background: torch.Tensor, X: torch.Tensor, target: np.ndarray,
                window_ids: Sequence[str], shap_seed: int, nsamples: int, cache_dir: Path, tag: str,
                log: Callable[[str], None] = print) -> np.ndarray:
    """shap_target_class with a disk cache: <cache_dir>/<tag>-<key>.npz, key = hash of the config
    (weights, window IDs, background, targets, SHAP seed, nsamples, shap version)."""
    cfg = {"model_state": state_hash(model), "window_ids": hashlib.sha256("|".join(window_ids).encode()).hexdigest()[:16],
           "n_windows": len(window_ids), "background": array_hash(background), "target": array_hash(np.asarray(target)),
           "target_rule": "original model predicted class", "shap_seed": int(shap_seed), "nsamples": int(nsamples),
           "explainer": "GradientExplainer, single output per class group", "shap_version": shap.__version__}
    key = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
    path = Path(cache_dir) / f"{tag}-{key}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            if json.loads(str(z["config"])) == cfg:
                log(f"    SHAP cache hit {path.name}")
                return z["A"]
    A = shap_target_class(model, background, X, target, shap_seed=shap_seed, nsamples=nsamples)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, A=A, config=json.dumps(cfg, sort_keys=True), window_ids=np.asarray(window_ids))
    tmp.replace(path)
    return A


# ------------------------------------------------------------------------------ LCI per subject

def subject_lci(A_ref: np.ndarray, A: np.ndarray, subjects, k: int = 10, omega: float = 0.5) -> list[dict]:
    """Per-subject LCI(reference, model) on the 48-segment features [plan 2.6]; (n, 6, 128) inputs
    are aggregated first. One dict per subject (sorted), keys of compute_lci plus "subject"."""
    from lci.metric import aggregate_time_segments, compute_lci_by_subject

    seg = lambda a: aggregate_time_segments(a) if np.asarray(a).ndim == 3 else np.asarray(a)
    return list(compute_lci_by_subject(seg(A_ref), seg(A), subjects, k=k, omega=omega).values())


def mean_lci(rows: list[dict]) -> float:
    """Subject mean of per-subject LCI (LCI_val = this over the fold's 3 inner-validation subjects)."""
    return float(np.mean([r["lci"] for r in rows]))
