"""Gap study v4: third-explainer check on saved validation checkpoints (inner val only).

    $env:PYTHONPATH = "bit-reduction-test"
    python -m gap_study_v4.explainer_check --folds 0 1 2 3 4 --models label=pattern ... --threads 4

Explainer: gradient x input at the window itself, d f_c / d x * x (c = the original's predicted class), summed into
the same 48 features (6 axes x 8 segments) and scored with the same LCI. It is deterministic (no explainer noise,
ceiling 1) and LOCAL: unlike GradientExplainer (blends with background windows) and occlusion (segments replaced by
the background mean), it never evaluates the model off the data. A method that only matched the model on the SHAP
probe distribution would not gain here.

{k} in a pattern is replaced by the fold. Writes results/gap-study-v4/validation/
gap-study-v4-validation-tinyhar-gradient-x-input-lci-per-fold.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "bit-reduction-test"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import bitred.context as BC  # noqa: E402
import gap_study_v3.confirm as C3  # noqa: E402
from har.compression import attribution_grad_context  # noqa: E402
from har.train import predict_proba  # noqa: E402

OUT = ROOT / "results" / "gap-study-v4" / "validation" / "gap-study-v4-validation-tinyhar-gradient-x-input-lci-per-fold.csv"
SEG, NSEG = 16, 8


def grad_x_input(net, X: torch.Tensor, target: np.ndarray, bs: int = 64) -> np.ndarray:
    net.eval()
    out = []
    with attribution_grad_context():
        for i in range(0, len(X), bs):
            x = X[i:i + bs].clone().requires_grad_(True)
            t = torch.as_tensor(target[i:i + bs]).long()
            f = net(x).gather(1, t[:, None]).sum()
            g, = torch.autograd.grad(f, x)
            a = (g * x).detach()                                         # (n, 6, 128)
            out.append(a.reshape(len(a), a.shape[1], NSEG, SEG).sum(-1).reshape(len(a), -1).numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    BC.DESIGN["explain_per_subject"] = C3.EXPLAIN_PER_SUBJECT
    rows = []
    for k in a.folds:
        ctx = BC.FoldContext(k, "cpu", False, None)
        G0 = grad_x_input(ctx.net0, ctx.X_ex, ctx.target_ex)
        for spec in a.models:
            lab, pat = spec.split("=", 1)
            paths = sorted(Path(pat.format(k=k)).parent.glob(Path(pat.format(k=k)).name))
            if not paths:
                print(f"fold {k} {lab}: no checkpoint for {pat}", flush=True)
                continue
            for path in paths:
                net = C3.load_ckpt(ctx, path)
                G = grad_x_input(net, ctx.X_ex, ctx.target_ex)
                pred = predict_proba(net, ctx.X_ex.cpu().numpy(), "cpu").argmax(1)
                rows.append({"fold": k, "label": lab, "checkpoint": path.name,
                             "gxi_lci": C3.lci_mean(G0, G, ctx.ex_subj), "agree": float((pred == ctx.target_ex).mean())})
                print(f"fold {k} {lab} {path.name}: gradient x input LCI {rows[-1]['gxi_lci']:.3f}", flush=True)
    d = pd.DataFrame(rows)
    d.round(4).to_csv(OUT, index=False, lineterminator="\n")
    s = d.groupby(["label", "fold"]).gxi_lci.mean().unstack()
    s["mean"] = s.mean(1)
    print(s.round(3).to_string())


if __name__ == "__main__":
    main()
