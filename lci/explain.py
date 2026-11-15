"""SHAP GradientExplainer attributions (spec 2.5), cached on disk.

The cache is two files instead of one pair per model: an .npz holding every attribution
array (one entry per key, e.g. "seed0_ptq4_shapseed0") and a .json mapping each key to the
config it was computed with. A cached array is reused only if its config, including the
model's weight hash, matches exactly.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import shap
import torch
import torch.nn as nn

CLASS_INDEX = 1  # legitimate, as in the authors' code
NSAMPLES = 200


def model_hash(model: nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def normalize_shap(sv, class_index: int = CLASS_INDEX) -> np.ndarray:
    """list-of-classes | (N, d, C) | (N, d)  ->  (N, d) for `class_index`."""
    if isinstance(sv, list):
        sv = sv[class_index]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[..., class_index]
    return sv.astype(np.float32)


def shap_attributions(model: nn.Module, background: torch.Tensor, X_explain: torch.Tensor,
                      shap_seed: int = 0, nsamples: int = NSAMPLES) -> np.ndarray:
    model.eval()
    explainer = shap.GradientExplainer(model, background)
    np.random.seed(shap_seed)
    torch.manual_seed(shap_seed)
    return normalize_shap(explainer.shap_values(X_explain, nsamples=nsamples))


class AttributionCache:
    def __init__(self, npz_path: Path, config_path: Path):
        self.npz_path, self.config_path = npz_path, config_path
        self.arrays: dict[str, np.ndarray] = {}
        self.configs: dict[str, dict] = {}
        if npz_path.exists() and config_path.exists():
            with np.load(npz_path) as z:
                self.arrays = {k: z[k] for k in z.files}
            self.configs = json.loads(config_path.read_text(encoding="utf-8"))

    def get(self, model, background, X_explain, key: str, config: dict, shap_seed: int) -> np.ndarray:
        config = {**config, "shap_seed": shap_seed, "nsamples": NSAMPLES, "class_index": CLASS_INDEX,
                  "n_background": int(background.shape[0]), "n_explain": int(X_explain.shape[0]),
                  "explainer": "shap.GradientExplainer", "shap_version": shap.__version__,
                  "torch_version": torch.__version__, "model_sha256": model_hash(model)}
        if key in self.arrays and self.configs.get(key) == config:
            return self.arrays[key]
        A = shap_attributions(model, background, X_explain, shap_seed)
        self.arrays[key], self.configs[key] = A, config
        self._save()  # after every new array, so an interrupted run keeps its progress
        return A

    def _save(self) -> None:
        self.npz_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.npz_path.with_name(self.npz_path.stem + ".tmp.npz")
        np.savez(tmp, **self.arrays)
        os.replace(tmp, self.npz_path)
        self.config_path.write_text(json.dumps(self.configs, indent=1, sort_keys=True), encoding="utf-8")
