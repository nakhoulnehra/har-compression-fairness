"""Gap study v4, validation runs: is v3's best no-retraining baseline the strongest, and what is achievable at the
same size? EXISTING techniques only (no new method). HARTH inner val only (HAR70+ and test subjects never loaded).

    $env:PYTHONPATH = "bit-reduction-test"
    python -m gap_study_v4.run_v4 --folds 0 --stages nr4 ft4 --threads 3 --machine-name Ansel --resume
    python -m gap_study_v4.run_v4 --folds 0 --stages nr4 ft4 --smoke --out <scratchpad>

Protocol = gap_study_v3.confirm (reused, unchanged): 50 explain windows per inner-val subject, GradientExplainer
nsamples 200, target = original predicted class, background = the fold's 100 training windows; every model is scored
with same-seed LCI, independent-seed LCI (lci_indep, the reported one), explainer self-consistency, occlusion LCI and
prediction agreement.

Hypothesis tested (from the explainer's definition, not from data): GradientExplainer (expected gradients) evaluates
the model at blends x' + a (x - x') of a background (training) window x' and the explained window x, and occlusion
at windows with a segment replaced. Every existing compressor fits the original only on real windows (calibration or
training windows), so the off-data behaviour that the explanations read is left free.

Stages
  nr4  no retraining, 80 % (the v3 gap cell): two strongest-baseline checks
       ria_fullrecon       RIA selection (Zhang et al., ICLR 2024, Eq. 1-2: |W_ij| / sum_row + |W_ij| / sum_col, times
                           ||X_j||^0.5), K = 5 steps, full layer-wise ridge refit (as wanda_fullrecon)
       l1_fullrecon_mixcal l1_fullrecon whose ~980 unlabeled calibration windows are complemented by as many blends
                           lam x_i + (1 - lam) x_j (lam ~ U(0, 1), both from the calibration set): label-free, no
                           training; a calibration-set choice that matches the explainer's probe distribution
  ft4  training-based references (outside the no-retraining scenario) at the same masks
       fmkd   the 9A KD recipe (KD T 3 alpha 0.7 + class-weighted CE on training windows, train mode) PLUS function
              matching (Beyer et al., CVPR 2022, Sec. 3: teacher and student see the same mixup input) on blends of
              two training windows, lam ~ U(0, 1): alpha T^2 KL(teacher || student) at T 3, eval mode (repo
              convention for interpolated inputs: BatchNorm statistics not updated by them). 20 epochs, lr {1e-4,
              3e-4}. Starts: agent B's l1_fullrecon 80 % seed-0 checkpoint (the v3 best baseline's exact masks) and
              its lamp_recon 90 % seed-0 checkpoint (v3's best fine-tuned 90 % start).
       kd     plain 9A KD from the same 80 % start at lr 3e-4 (v3 ran only 1e-4 at 80 %)
       calkd  label-free KD on the ~980 calibration windows only (alpha 1, T 3, no labels), 200 epochs, lr {1e-4,
              3e-4}, from the same 80 % start: light, label-free training a deployer could run
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "bit-reduction-test"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import bitred.context as BC  # noqa: E402
import gap_study_v3.baselines as GB  # noqa: E402
import gap_study_v3.confirm as C3  # noqa: E402
from har import runlog as R  # noqa: E402
from har.compression import (KD, _masked_params, apply_masks, attribution_grad_context, finetune,  # noqa: E402
                             head_unpruned, kd_kl_term, predict_logits, size_record)
from har.train import set_seed  # noqa: E402
from har.xgp4 import iterative_prune_budget, uniform_keep_counts  # noqa: E402

OUT = ROOT / "results" / "gap-study-v4" / "validation"
P = "gap-study-v4-validation-tinyhar"
CKPT = C3.CHECKPOINT_DIR / "gap-study-v4"
ATTR = C3.PROCESSED_DIR / "gap-study-v4" / "attributions"
B_CKPT = C3.B_CKPT
FT_EPOCHS = 20
CALKD_EPOCHS = 200
RIA_A = 0.5


def plan(stages: list[str], smoke: bool) -> list[dict]:
    S = (0, 1) if smoke else (0, 1, 2)
    rows = []
    for st in stages:
        if st == "ref":
            rows.append({"stage": "ref", "method": "original", "level": 0.0, "seed": 0})
        elif st == "nr4":
            for m in ("ria_fullrecon", "l1_fullrecon_mixcal"):
                for s in S:
                    rows.append({"stage": "nr4", "method": m, "level": 0.8, "seed": s})
        elif st == "ft4":
            for m, start, lev, lrs in (("fmkd", "l1_fullrecon", 0.8, (3e-4, 1e-4)), ("kd", "l1_fullrecon", 0.8, (3e-4,)),
                                       ("calkd", "l1_fullrecon", 0.8, (1e-4, 3e-4)),
                                       ("fmkd", "lamp_recon", 0.9, (3e-4,))):   # lr 1e-4 dropped 13:52 (time)
                for lr in lrs:
                    rows.append({"stage": "ft4", "method": m, "start": start, "level": lev, "seed": 0, "lr": lr})
        elif st == "ft4b":     # fold 2 restart after its CUDA crash (14:26): ft4 without fmkd 80 % lr 1e-4 (time)
            rows += [r for r in plan(["ft4"], smoke) if not (r["method"] == "fmkd" and r["level"] == 0.8
                                                              and r["lr"] == 1e-4)]
        elif st == "x4":       # added 15:23 (decision point 2): calfm lr 1e-3, longer fmkd 80 %, then the missing fold 0 / 2 runs
            rows.append({"stage": "ft4", "method": "calfm", "start": "l1_fullrecon", "level": 0.8, "seed": 0, "lr": 1e-3})
            for lr in (1e-4, 3e-4):
                rows.append({"stage": "ft4", "method": "calkd", "start": "l1_fullrecon", "level": 0.8, "seed": 0, "lr": lr})
            rows.append({"stage": "ft4", "method": "fmkd", "start": "l1_fullrecon", "level": 0.8, "seed": 0, "lr": 3e-4,
                         "epochs": 40})
            rows.append({"stage": "ft4", "method": "kd", "start": "l1_fullrecon", "level": 0.8, "seed": 0, "lr": 3e-4})
            rows.append({"stage": "ft4", "method": "fmkd", "start": "lamp_recon", "level": 0.9, "seed": 0, "lr": 3e-4})
        elif st == "lf90":     # added 16:30 (decision point 3): label-free calibration-only fine-tuning at 90 %
            for m, start, lr in (("calfm", "lamp_recon", 1e-3), ("calfm", "lamp_recon", 3e-4), ("calfm", "l1_fullrecon", 1e-3),
                                 ("calkd", "lamp_recon", 1e-3), ("calfm", "lamp_recon", 3e-3)):
                rows.append({"stage": "ft4", "method": m, "start": start, "level": 0.9, "seed": 0, "lr": lr})
        elif st == "cal5":     # added 15:33: lr grid edge checks for the label-free baselines (run on CPU)
            rows.append({"stage": "ft4", "method": "calfm", "start": "l1_fullrecon", "level": 0.8, "seed": 0, "lr": 3e-3})
            rows.append({"stage": "ft4", "method": "calkd", "start": "l1_fullrecon", "level": 0.8, "seed": 0, "lr": 1e-3})
        elif st == "cal4":     # added 14:15: label-free function matching on the calibration windows only
            for lr in (1e-4, 3e-4):
                rows.append({"stage": "ft4", "method": "calfm", "start": "l1_fullrecon", "level": 0.8, "seed": 0, "lr": lr})
        elif st.startswith("ftseeds:"):   # ftseeds:method:start:level:lr  -> seeds 1, 2 of a selected setting
            _, m, start, lev, lr = st.split(":")
            for s in (1, 2):
                rows.append({"stage": "ft4", "method": m, "start": start, "level": float(lev), "seed": s,
                             "lr": float(lr)})
        else:
            raise ValueError(st)
    return rows


def run_id(r: dict) -> str:
    rid = f"{r['method']}-{int(round(r['level'] * 100))}pct-seed{r['seed']}"
    if "start" in r:
        rid = f"{r['method']}_from_{r['start']}-{int(round(r['level'] * 100))}pct-seed{r['seed']}"
    if "lr" in r:
        rid += f"-lr{r['lr']:g}"
    if "epochs" in r:
        rid += f"-e{r['epochs']}"
    return rid


# ------------------------------------------------------------------------------ existing no-retraining techniques

def ria_scores(net, X_cal: torch.Tensor, a: float = RIA_A) -> dict[str, torch.Tensor]:
    """RIA (Zhang et al. 2024): (|W_ij| / sum_k |W_kj| + |W_ij| / sum_k |W_ik|) * ||X_j||_2^a per prunable tensor,
    relative importance over the unit's whole (out, in) matrix, X = the current model's unit inputs."""
    out = {}
    for kind, name in GB.UNITS:
        H, _, _ = GB.unit_grams(net, None, X_cal, kind, name, False)
        norm = torch.sqrt(torch.diagonal(H).clamp_min(0)).float()
        W, _ = GB.unit_weight(net, kind, name)
        A = W.abs().float()
        RI = A / A.sum(0, keepdim=True).clamp_min(1e-12) + A / A.sum(1, keepdim=True).clamp_min(1e-12)
        S = RI * norm[None, :].to(W.device) ** a
        c0 = 0
        params = dict(net.named_parameters())
        for t in GB.unit_tensors(kind, name):
            w = params[t].numel() // params[t].shape[0]
            out[t] = S[:, c0:c0 + w].reshape(params[t].shape).clone()
            c0 += w
    return out


def blend(X: torch.Tensor, seed: int) -> torch.Tensor:
    """X plus len(X) blends lam x_i + (1 - lam) x_j (lam ~ U(0, 1), j a random other window of X)."""
    g = torch.Generator().manual_seed(7_000 + seed)
    j = torch.randint(0, len(X), (len(X),), generator=g)
    lam = torch.rand(len(X), 1, 1, generator=g).to(X.dtype)
    return torch.cat([X, lam.to(X.device) * X + (1 - lam.to(X.device)) * X[j.to(X.device)]])


def build_nr4(r: dict, ctx, smoke: bool):
    Xc = C3.calibration(ctx, r["seed"], C3.CAL["per_cell"], smoke)
    if r["method"] == "ria_fullrecon":
        keep = uniform_keep_counts(ctx.net0, r["level"])
        net, _ = iterative_prune_budget(ctx.net0, keep, K=GB.K_STEPS, beta=1.0,
                                        importance_fn=lambda n: ria_scores(n, Xc),
                                        recon_fn=lambda n: GB.full_reconstruct(n, ctx.net0, Xc, damp=GB.DAMP),
                                        log=lambda x: None)
        net.prune_sparsity = r["level"]
        return net.eval()
    if r["method"] == "l1_fullrecon_mixcal":
        return GB.build("l1_fullrecon", ctx.net0, r["level"], blend(Xc, r["seed"]), log=lambda x: None)
    raise ValueError(r["method"])


# ------------------------------------------------------------------------------ training-based references

def fm_extra_loss(teacher, X_train: torch.Tensor, seed: int):
    """Function matching on mixup blends of training windows (Beyer et al. 2022): alpha T^2 KL at T = KD['T']."""
    g = torch.Generator().manual_seed(9_000 + seed)
    T, alpha = KD["T"], KD["alpha"]

    def f(model, idx):
        n = len(idx)
        j = torch.randint(0, len(X_train), (n,), generator=g).to(X_train.device)
        lam = torch.rand(n, 1, 1, generator=g).to(X_train.device)
        xm = lam * X_train[idx] + (1 - lam) * X_train[j]
        with torch.no_grad():
            zt = teacher(xm)
        with attribution_grad_context():     # eval-mode student through the LSTM on CUDA (see har.compression)
            zs = model(xm)
        return alpha * T ** 2 * kd_kl_term(zs, zt, T)
    return f


def calkd(net, teacher, Xc: torch.Tensor, *, epochs: int, lr: float, seed: int, device: str,
          batch_size: int = 64, fm: bool = False) -> dict:
    """Label-free KD on the calibration windows only: T^2 KL(teacher || student) at T = KD['T'], train mode,
    masks fixed (gradients masked, masks re-applied after every step). fm=True (calfm) adds function matching on
    blends of two calibration windows (lam ~ U(0, 1)), T^2 KL in eval mode, as fmkd but label-free."""
    t0 = time.perf_counter()
    dev = torch.device(device)
    net.to(dev)
    teacher = teacher.to(dev).eval()
    set_seed(seed)
    apply_masks(net)
    masked = _masked_params(net)
    X = Xc.to(dev)
    with torch.no_grad():
        zt = teacher(X)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    gm = torch.Generator().manual_seed(9_000 + seed)
    T = KD["T"]
    for _ in range(epochs):
        perm = torch.randperm(len(X), generator=g).to(dev)
        for i in range(0, len(X), batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            net.train()
            loss = T ** 2 * kd_kl_term(net(X[idx]), zt[idx], T)
            if fm:
                j = torch.randint(0, len(X), (len(idx),), generator=gm).to(dev)
                lam = torch.rand(len(idx), 1, 1, generator=gm).to(dev)
                xm = lam * X[idx] + (1 - lam) * X[j]
                with torch.no_grad():
                    ztm = teacher(xm)
                net.eval()
                with attribution_grad_context():
                    loss = loss + T ** 2 * kd_kl_term(net(xm), ztm, T)
            loss.backward()
            with torch.no_grad():
                for w, keep in masked:
                    if w.grad is not None:
                        w.grad.mul_(keep.to(w.grad.dtype))
            opt.step()
            apply_masks(net)
    net.eval()
    return {"seconds": time.perf_counter() - t0, "epochs_run": epochs}


def load_start(ctx, start: str, level: float, smoke: bool, seed: int):
    """Agent B's no-retraining checkpoint (seed 0) = the exact masks of the v3 best baseline; rebuilt if absent
    (smoke) or for seeds > 0 (calibration seed s, as v3)."""
    p = B_CKPT / f"fold{ctx.k}-Ansel-{start}-{int(round(level * 100))}pct-seed{seed}-damp0.01.pt"
    if p.exists() and not smoke:
        return C3.load_ckpt(ctx, p), p.name
    Xc = C3.calibration(ctx, seed, C3.CAL["per_cell"], smoke)
    return GB.build(start, ctx.net0, level, Xc, log=lambda x: None), "rebuilt"


def build_ft4(r: dict, ctx, a, zt):
    net, src = load_start(ctx, r["start"], r["level"], a.smoke, r["seed"])
    epochs = 1 if a.smoke else r.get("epochs", FT_EPOCHS)
    if r["method"] in ("calkd", "calfm"):
        Xc = C3.calibration(ctx, r["seed"], C3.CAL["per_cell"], a.smoke)
        info = calkd(net, copy.deepcopy(ctx.net0), Xc, epochs=2 if a.smoke else CALKD_EPOCHS, lr=r["lr"],
                     seed=r["seed"], device=a.ft_device, fm=r["method"] == "calfm")
    else:
        extra = None
        if r["method"] == "fmkd":
            teacher = copy.deepcopy(ctx.net0).to(a.ft_device).eval()
            extra = fm_extra_loss(teacher, torch.from_numpy(ctx.tr.X).to(a.ft_device), r["seed"])
        info = finetune(net, ctx.tr, ctx.va, epochs=epochs, lr=r["lr"], seed=r["seed"], device=a.ft_device,
                        kd_teacher_logits=zt, extra_loss=extra, log=lambda x: None)
    info["start_source"] = src
    return net.to(ctx.dev).eval(), info


# ------------------------------------------------------------------------------ fold loop

def run_fold(k: int, a) -> None:
    out = Path(a.out) if a.out else OUT
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{P}-fold{k}-{a.machine_name}-{a.worker}"
    models_csv, subj_csv = Path(f"{stem}-models.csv"), Path(f"{stem}-per-subject-lci.csv")
    done = set(pd.read_csv(models_csv).run_id) if (a.resume and models_csv.exists()) else set()
    if not a.resume and models_csv.exists():
        raise SystemExit(f"{models_csv} exists; pass --resume")
    BC.DESIGN["explain_per_subject"] = C3.EXPLAIN_PER_SUBJECT
    if a.smoke:
        BC.SMOKE["explain_per_subject"] = 3
    t0 = time.perf_counter()
    ctx = BC.FoldContext(k, a.device, a.smoke, None)
    refs = C3.Refs(ctx, ctx.nsamples)
    attr_dir = (Path(a.out) / "attributions") if a.out else ATTR
    attr_dir.mkdir(parents=True, exist_ok=True)
    cfgp = Path(f"{stem}-config.json")
    if not cfgp.exists():
        cfgp.write_text(json.dumps({"command": " ".join(sys.argv), "fold": k, "device": a.device,
                                    "ft_device": a.ft_device, "machine": a.machine_name, "torch": torch.__version__,
                                    "explain_per_subject": C3.EXPLAIN_PER_SUBJECT, "window_hashes": ctx.hashes,
                                    "ceiling50": refs.ceiling50, "calibration": C3.CAL, "ft_epochs": FT_EPOCHS,
                                    "calkd_epochs": CALKD_EPOCHS, "kd": KD, "ria_a": RIA_A,
                                    "plan": plan(a.stages, a.smoke)}, indent=2, default=str), encoding="utf-8")
    base = {"fold": k, "machine_name": a.machine_name, "device": a.device, "ceiling50": refs.ceiling50}
    zt = None
    ck_dir = (Path(a.out) / "ckpt") if a.out else CKPT
    for r in plan(a.stages, a.smoke):
        rid = run_id(r)
        if rid in done:
            continue
        ts = time.perf_counter()
        info, t_build, t_ft = {}, 0.0, 0.0
        if r["stage"] == "ref":
            f1, acc = C3._f1acc(ctx, ctx.net0)
            o = {"f1": f1, "accuracy": acc, "lci": 1.0, "lci_indep": refs.ceiling50, "self_lci": refs.ceiling50,
                 "occ_lci": 1.0, "agree": 1.0}
            rows, net = [], ctx.net0
        else:
            if r["stage"] == "nr4":
                net = build_nr4(r, ctx, a.smoke)
                t_build = time.perf_counter() - ts
            else:
                if zt is None:
                    zt = torch.from_numpy(predict_logits(ctx.net0, ctx.tr.X, ctx.dev)).to(a.ft_device)
                net, info = build_ft4(r, ctx, a, zt)
                t_ft = info.get("seconds", time.perf_counter() - ts)
            if not head_unpruned(net):
                raise RuntimeError(f"{rid}: head pruned")
            o, rows = C3.score(ctx, refs, net, f"gs4-{rid}", True, ctx.nsamples, attr_dir / f"fold{k}-{rid}.npz")
            ck = ck_dir / f"fold{k}-{a.machine_name}-{rid}.pt"
            ck.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"run_id": rid, "fold": k, "state_dict": {n: t.cpu() for n, t in net.state_dict().items()},
                        "prune_masks": {n: m.cpu() for n, m in net.prune_masks.items()}}, ck)
        sz = size_record(net)
        C3._append(models_csv, [{**base, "run_id": rid, **r, **o, "build_seconds": t_build, "ft_seconds": t_ft,
                                 "start_source": info.get("start_source"), "epochs_run": info.get("epochs_run"),
                                 "n_nonzero": sz["n_nonzero"], "seconds_total": time.perf_counter() - ts}])
        if rows:
            C3._append(subj_csv, [{**base, "run_id": rid, **r, **x} for x in rows])
        print(f"fold {k} {rid}: F1 {o['f1']:.3f} acc {o['accuracy']:.3f} LCI {o['lci']:.3f} "
              f"indep {o.get('lci_indep', float('nan')):.3f} self {o.get('self_lci', float('nan')):.3f} "
              f"occ {o.get('occ_lci', float('nan')):.3f} agree {o.get('agree', float('nan')):.3f} nnz {sz['n_nonzero']} "
              f"[{time.perf_counter() - ts:.0f}s, total {time.perf_counter() - t0:.0f}s]", flush=True)
    print(f"fold {k} stages {a.stages} done in {time.perf_counter() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", type=int, nargs="+", default=[0])
    ap.add_argument("--stages", nargs="+", default=["ref"])
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="scoring / building device")
    ap.add_argument("--ft-device", choices=["cpu", "cuda"], default="cuda", help="fine-tuning device")
    ap.add_argument("--machine-name", default=R.machine_name())
    ap.add_argument("--worker", default="all")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    for k in a.folds:
        run_fold(k, a)


if __name__ == "__main__":
    main()
