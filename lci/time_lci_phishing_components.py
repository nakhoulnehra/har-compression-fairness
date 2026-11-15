"""Per-stage wall-clock timing of the LCI Phishing pipeline for one seed (for environment/time.md).

Usage (from the repo root):
    python -m lci.time_lci_phishing_components --seed 0

Times data loading, teacher training, each compression method, each SHAP call and the LCI
computation. SHAP is always recomputed (the results cache is bypassed) and nothing is written.
"""

from __future__ import annotations

import argparse
import time

import torch

from .data import load_phishing_split
from .explain import shap_attributions
from .metric import compute_lci
from .models import ResidualMLP
from .run_lci_phishing import METHODS
from .train import set_seed, train_teacher


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rows = []

    def timed(label, fn):
        t = time.perf_counter()
        out = fn()
        rows.append((label, time.perf_counter() - t))
        return out

    split = timed("data: load + preprocess + split", load_phishing_split)
    d = split.X_train.shape[1]
    bg, X = torch.from_numpy(split.X_train[:50]), torch.from_numpy(split.X_test[:300])
    set_seed(args.seed)
    teacher = ResidualMLP(d)
    timed("train: teacher (30 epochs)", lambda: train_teacher(teacher, split.X_train, split.y_train,
                                                               split.X_val, split.y_val, seed=args.seed))
    A_T = timed("shap: teacher", lambda: shap_attributions(teacher, bg, X, 0))
    for name, _, _, build in METHODS:
        model, _ = timed(f"compress: {name}", lambda: build(teacher, split, d, args.seed))
        A_S = timed(f"shap: {name}", lambda: shap_attributions(model, bg, X, 0))
        timed(f"lci: {name}", lambda: compute_lci(A_T, A_S))

    total = sum(t for _, t in rows)
    for label, t in rows:
        print(f"{label:34s} {t:8.2f} s")
    for group in ("data", "train", "compress", "shap", "lci"):
        g = sum(t for label, t in rows if label.startswith(group))
        print(f"TOTAL {group:28s} {g:8.2f} s  ({100 * g / total:4.1f}%)")
    print(f"TOTAL {'all':28s} {total:8.2f} s")


if __name__ == "__main__":
    main()
