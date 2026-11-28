"""Gap study v5, shift probe (data only, existing checkpoints, no training): does a deployment condition open a LARGER
or EARLIER explanation gap than the clean one? HARTH inner val only.

    $env:PYTHONPATH = "bit-reduction-test"
    python -m gap_study_v5.shift_probe --folds 0 --threads 3 --resume

Conditions (fixed a priori, not tuned): the fold's inner-val explain windows (150) and F1 windows, perturbed once
with a fixed seed per fold:
  noise05  additive white Gaussian noise, sigma 0.05 g on every axis (sensor / ADC noise)
  rot15    each sensor (back, thigh) rotated independently about a random axis by U(-15, 15) degrees
           (har.augment.rotate_sensors, round 3's placement-shift magnitude)
Per model and condition: lci_indep (model SHAP seed 1 vs the ORIGINAL's SHAP seed 0 on the same perturbed windows),
the uncompressed ceiling on those windows (original seed 0 vs 1), accuracy / F1 on the perturbed F1 windows, and
agreement with the original's prediction. Models: agent B's l1_fullrecon 50 / 70 / 80 % (seed 0), v4's
l1_fullrecon_mixcal 80 % (seed 0), v4's fmkd 80 % (40 epochs) and calfm 80 % (lr 1e-3).
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
from har.attributions import shap_cached  # noqa: E402
from har.augment import draw_rotation_params, rotate_sensors  # noqa: E402
from har.train import pooled_macro_f1, predict_proba  # noqa: E402

OUT = ROOT / "results" / "gap-study-v5" / "validation"
P = "gap-study-v5-validation-tinyhar"
V4 = C3.CHECKPOINT_DIR / "gap-study-v4"
MODELS = {
    "l1_fullrecon-50": lambda k: C3.B_CKPT / f"fold{k}-Ansel-l1_fullrecon-50pct-seed0-damp0.01.pt",
    "l1_fullrecon-70": lambda k: C3.B_CKPT / f"fold{k}-Ansel-l1_fullrecon-70pct-seed0-damp0.01.pt",
    "l1_fullrecon-80": lambda k: C3.B_CKPT / f"fold{k}-Ansel-l1_fullrecon-80pct-seed0-damp0.01.pt",
    "mixcal-80": lambda k: V4 / f"fold{k}-Ansel-l1_fullrecon_mixcal-80pct-seed0.pt",
    "fmkd_e40-80": lambda k: V4 / f"fold{k}-Ansel-fmkd_from_l1_fullrecon-80pct-seed0-lr0.0003-e40.pt",
    "calfm-80": lambda k: V4 / f"fold{k}-Ansel-calfm_from_l1_fullrecon-80pct-seed0-lr0.001.pt",
}
SIGMA = 0.05
ROT = 15.0


def perturb(X: np.ndarray, cond: str, seed: int) -> np.ndarray:
    if cond == "noise05":
        rng = np.random.default_rng(seed)
        return (X + rng.normal(0.0, SIGMA, size=X.shape)).astype(np.float32)
    if cond == "rot15":
        axes, _ = draw_rotation_params(len(X), seed)
        ang = np.random.default_rng(seed + 1).uniform(-ROT, ROT, size=(len(X), 2))
        return rotate_sensors(X, axes, ang)
    raise ValueError(cond)


def run_fold(k: int, a) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    csv = OUT / f"{P}-fold{k}-Ansel-shift-models.csv"
    done = set(zip(*[pd.read_csv(csv)[c] for c in ("model", "condition")])) if (a.resume and csv.exists()) else set()
    BC.DESIGN["explain_per_subject"] = C3.EXPLAIN_PER_SUBJECT
    ctx = BC.FoldContext(k, "cpu", False, None)
    seg = C3.aggregate_time_segments
    for cond in ("noise05", "rot15"):
        sd = 20_000 + 10 * k + (0 if cond == "noise05" else 5)
        Xex = torch.from_numpy(perturb(ctx.X_ex.cpu().numpy(), cond, sd))
        Xva = perturb(ctx.va.X, cond, sd + 2)
        ids = [f"{i}|{cond}" for i in ctx.ex_ids]
        A0 = seg(shap_cached(ctx.net0, ctx.bg, Xex, ctx.target_ex, ids, 0, ctx.nsamples, ctx.cache,
                             f"gs5shift-fold{k}-{cond}-original-seed0", log=lambda x: None))
        A0s1 = seg(shap_cached(ctx.net0, ctx.bg, Xex, ctx.target_ex, ids, 1, ctx.nsamples, ctx.cache,
                               f"gs5shift-fold{k}-{cond}-original-seed1", log=lambda x: None))
        ceil = C3.lci_mean(A0, A0s1, ctx.ex_subj)
        p0 = predict_proba(ctx.net0, Xva, "cpu").argmax(1)
        pex0 = predict_proba(ctx.net0, Xex.numpy(), "cpu").argmax(1)
        rows = []
        if ("original", cond) not in done:
            rows.append({"fold": k, "model": "original", "condition": cond, "lci_indep": ceil, "ceiling": ceil,
                         "f1": pooled_macro_f1(ctx.va.y, p0), "accuracy": float((p0 == ctx.va.y).mean()),
                         "agree_target": float((pex0 == ctx.target_ex).mean())})
            C3._append(csv, rows)
        for name, path_fn in MODELS.items():
            if (name, cond) in done:
                continue
            p = path_fn(k)
            if not p.exists():
                print(f"fold {k} {name}: missing {p.name}", flush=True)
                continue
            net = C3.load_ckpt(ctx, p)
            A1 = seg(shap_cached(net, ctx.bg, Xex, ctx.target_ex, ids, 1, ctx.nsamples, ctx.cache,
                                 f"gs5shift-fold{k}-{cond}-{name}-seed1", log=lambda x: None))
            pv = predict_proba(net, Xva, "cpu").argmax(1)
            pex = predict_proba(net, Xex.numpy(), "cpu").argmax(1)
            r = {"fold": k, "model": name, "condition": cond, "lci_indep": C3.lci_mean(A0, A1, ctx.ex_subj),
                 "ceiling": ceil, "f1": pooled_macro_f1(ctx.va.y, pv), "accuracy": float((pv == ctx.va.y).mean()),
                 "agree_target": float((pex == ctx.target_ex).mean()), "agree_original": float((pex == pex0).mean())}
            rows.append(r)
            print(f"fold {k} {cond} {name}: lci_indep {r['lci_indep']:.3f} ceiling {ceil:.3f} acc {r['accuracy']:.3f} "
                  f"(original {float((p0 == ctx.va.y).mean()):.3f})", flush=True)
            C3._append(csv, [r])
    print(f"fold {k} shift probe done", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", type=int, nargs="+", default=[0])
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    for k in a.folds:
        run_fold(k, a)


if __name__ == "__main__":
    main()
