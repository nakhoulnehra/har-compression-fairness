"""Gap study v5 (2026-11-27, Ansel): solidify the v4 gap (one-shot pruning without retraining at 80 %), measure how
high LCI can go at that size, and locate where the problem lives. No new method. HARTH inner val only (HAR70+ and
the folds' test subjects are never loaded here).

    $env:PYTHONPATH = "bit-reduction-test"
    python -m gap_study_v5.run_v5 --folds 0 --stages ref nr5 --threads 3 --worker nr --machine-name Ansel --resume
    python -m gap_study_v5.run_v5 --folds 0 --stages ref orc --ft-device cuda --worker orc --machine-name Ansel --resume
    python -m gap_study_v5.run_v5 --folds 0 --stages ref nr5 abl swap orc --smoke --out <scratchpad>

Protocol = gap_study_v3.confirm.score (unchanged): 50 explain windows per inner-val subject, GradientExplainer nsamples
200, target = original predicted class, background = the fold's 100 training windows; reported metric lci_indep
(model SHAP seed 1 vs original seed 0), plus occlusion LCI, self-consistency, agreement.

Stages
  nr5    the backprop-free baselines v3 listed as "not run" (gap_study_v5.baselines_v5): obc, obc_fullrecon,
         sparsegpt_fullrecon, dsnot_fullrecon, l1_fullrecon_bnrecal, l1_fullrecon_k20; 80 %, calibration seed 0
  nr5s:<method>  calibration seeds 1, 2 of one nr5 method
  abl    where the loss comes from (diagnostic, not a method): unitonly_<u> = only unit u pruned at 80 % (magnitude,
         K = 5, full refit of every unit), the 10 other units dense; and l1_fullrecon_probecal = l1_fullrecon whose
         calibration set is the ~980 windows plus, for each, one blend b + a (x - b) with a random window b of the
         fold's SHAP background (training windows; a ~ U(0, 1)) and one copy with one random (axis, 16-sample segment)
         replaced by the background mean: the explainer's own probe distribution, built from training windows only
  swap   same masks, which weights matter (diagnostic): agent B's l1_fullrecon 80 % seed-0 model and v4's fmkd 80 %
         (40 epochs, lr 3e-4, trained from that exact model, so the masks are identical; checked) exchange the weights of
         one unit u (its weights and biases): swapin_<u> = l1_fullrecon with fmkd's u, swapout_<u> = fmkd with
         l1_fullrecon's u; u = the 11 prunable units and "other" (BatchNorm, LayerNorm, gamma, head)
  orc    upper bound at the same size (oracle, uses the full labelled training set and gradient training, outside the
         one-shot scenario): fmkd (v4's recipe) PLUS gradient-field matching on SHAP path points: for a batch of
         training windows x, a random training window b and a ~ U(0, 1), z = b + a (x - b); loss term
         lam * mean ||d f_c^S(z)/dz - d f_c^T(z)/dz||^2 / ||d f_c^T(z)/dz||^2, c = the teacher's predicted class on x.
         GradientExplainer's attribution is (x - b) * E[d f_c(z)/dz], so matching this field on these points targets
         exactly what the explainer reads (training windows only; the explained inner-val windows are never seen).
         Starts: agent B's l1_fullrecon 80 % seed 0 (same masks as the gap's baselines) and lamp_recon 90 % seed 0.
         lam {1, 10}, lr 3e-4, 20 epochs, seed 0. Selection among the oracle's settings uses inner val (disclosed: an
         upper-bound probe, not a method)
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
import gap_study_v4.run_v4 as V4  # noqa: E402
import gap_study_v5.baselines_v5 as B5  # noqa: E402
from har import runlog as R  # noqa: E402
from har.compression import (KD, attribution_grad_context, finetune, head_unpruned, kd_kl_term,  # noqa: E402
                             predict_logits, size_record)
from har.xgp4 import iterative_prune_budget, uniform_keep_counts  # noqa: E402

OUT = ROOT / "results" / "gap-study-v5" / "validation"
P = "gap-study-v5-validation-tinyhar"
CKPT = C3.CHECKPOINT_DIR / "gap-study-v5"
ATTR = C3.PROCESSED_DIR / "gap-study-v5" / "attributions"
V4_CKPT = C3.CHECKPOINT_DIR / "gap-study-v4"
LEVEL = 0.8
UNIT_TAGS = ["conv0", "conv3", "conv6", "conv9", "mha_in", "mha_out", "ff1", "ff2", "fusion", "lstm", "attn_score"]
ORC_EPOCHS = 20
ORC_SUB = 64        # path points per batch for the gradient-field term


def unit_by_tag(tag: str):
    return dict(zip(UNIT_TAGS, GB.UNITS))[tag]


def plan(stages: list[str], smoke: bool) -> list[dict]:
    rows = []
    for st in stages:
        if st == "ref":
            rows.append({"stage": "ref", "method": "original", "level": 0.0, "seed": 0})
        elif st == "nr5":
            for m in B5.V5_BASELINES:
                rows.append({"stage": "nr5", "method": m, "level": LEVEL, "seed": 0})
        elif st.startswith("nr5s:"):
            for s in (1, 2):
                rows.append({"stage": "nr5", "method": st.split(":")[1], "level": LEVEL, "seed": s})
        elif st.startswith("nr5x:"):     # nr5x:method:level:seed,seed  (added 22:10 after obc_fullrecon's 80 % result)
            _, m, lev, ss = st.split(":")
            for s in (int(x) for x in ss.split(",")):
                rows.append({"stage": "nr5", "method": m, "level": float(lev), "seed": s})
        elif st == "abl":
            rows.append({"stage": "abl", "method": "l1_fullrecon_probecal", "level": LEVEL, "seed": 0})
            for u in UNIT_TAGS:
                rows.append({"stage": "abl", "method": f"unitonly_{u}", "level": LEVEL, "seed": 0})
        elif st == "swap":
            for u in UNIT_TAGS + ["other"]:
                for d in ("swapin", "swapout"):
                    rows.append({"stage": "swap", "method": f"{d}_{u}", "level": LEVEL, "seed": 0})
        elif st in ("alloc", "alloc90"):   # group budget allocations (OBC + full refit, calibration seed 0)
            for name in B5.ALLOCS:
                rows.append({"stage": "nr5", "method": f"alloc_{name}", "level": LEVEL if st == "alloc" else 0.9,
                             "seed": 0})
        elif st.startswith("allocs:"):   # allocs:level:seed,seed  (added 03:45: calibration seeds of the allocations)
            _, lev, ss = st.split(":")
            for sd in (int(x) for x in ss.split(",")):
                for name in B5.ALLOCS:
                    rows.append({"stage": "nr5", "method": f"alloc_{name}", "level": float(lev), "seed": sd})
        elif st == "swap2":   # added 00:05: conv units swapped together with their BatchNorm (swap broke on BN coupling)
            for u in UNIT_TAGS + ["other"]:
                for d in ("swapin", "swapout"):
                    rows.append({"stage": "swap", "method": f"{d}2_{u}", "level": LEVEL, "seed": 0})
        elif st == "orc":
            for start, lev in (("l1_fullrecon", 0.8), ("lamp_recon", 0.9)):
                for lam in (1.0, 10.0):
                    rows.append({"stage": "orc", "method": "jfm", "start": start, "level": lev, "seed": 0,
                                 "lr": 3e-4, "lam": lam})
        elif st.startswith("fmo:"):      # fmo:start_run_id:level[:epochs]  fmkd (v4 recipe, lr 3e-4) from a v5 checkpoint
            f = st.split(":")
            rows.append({"stage": "fmo", "method": "fmkd", "start": f[1], "level": float(f[2]), "seed": 0, "lr": 3e-4,
                         "epochs": int(f[3]) if len(f) > 3 else 40})
        elif st.startswith(("orc2:", "cap:")):   # orc2|cap:start_run_id:level:lam:epochs  (added 23:30)
            f = st.split(":")
            rows.append({"stage": f[0], "method": "jfm2" if f[0] == "orc2" else "capfit", "start": f[1],
                         "level": float(f[2]), "seed": 0, "lr": 3e-4, "lam": float(f[3]), "epochs": int(f[4])})
        elif st.startswith("orcx:"):    # orcx:start:level:lr:lam[:epochs]  one extra oracle setting
            f = st.split(":")
            r = {"stage": "orc", "method": "jfm", "start": f[1], "level": float(f[2]), "seed": 0, "lr": float(f[3]),
                 "lam": float(f[4])}
            if len(f) > 5:
                r["epochs"] = int(f[5])
            rows.append(r)
        else:
            raise ValueError(st)
    return rows


def run_id(r: dict) -> str:
    rid = V4.run_id({k: v for k, v in r.items() if k != "lam"})
    if "lam" in r:
        rid += f"-lam{r['lam']:g}"
    return rid


# ------------------------------------------------------------------------------ diagnostics

def probe_calibration(ctx, Xc: torch.Tensor, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(11_000 + seed)
    bg = ctx.bg.to(Xc.device)
    n = len(Xc)
    b = bg[torch.randint(0, len(bg), (n,), generator=g).to(Xc.device)]
    a = torch.rand(n, 1, 1, generator=g).to(Xc.device, Xc.dtype)
    blends = b + a * (Xc - b)
    occ = Xc.clone()
    ax = torch.randint(0, C3.C, (n,), generator=g)
    sg = torch.randint(0, C3.NSEG, (n,), generator=g)
    bgm = bg.mean(0)
    for i in range(n):
        s0 = int(sg[i]) * C3.SEG
        occ[i, int(ax[i]), s0:s0 + C3.SEG] = bgm[int(ax[i]), s0:s0 + C3.SEG]
    return torch.cat([Xc, blends, occ])


def build_abl(r: dict, ctx, smoke: bool):
    Xc = C3.calibration(ctx, r["seed"], C3.CAL["per_cell"], smoke)
    if r["method"] == "l1_fullrecon_probecal":
        return GB.build("l1_fullrecon", ctx.net0, LEVEL, probe_calibration(ctx, Xc, r["seed"]), log=lambda x: None)
    u = r["method"].split("_", 1)[1]
    kind, name = unit_by_tag(u)
    own = set(GB.unit_tensors(kind, name))
    uni = uniform_keep_counts(ctx.net0, LEVEL)
    sizes = {n: getattr(m, p).numel() for n, m, p in GB.prunable_tensors(ctx.net0)}
    keep = {n: (uni[n] if n in own else sizes[n]) for n in sizes}
    net, _ = iterative_prune_budget(ctx.net0, keep, K=GB.K_STEPS, beta=0.0, importance_fn=None,
                                    recon_fn=lambda n: GB.full_reconstruct(n, ctx.net0, Xc, damp=GB.DAMP),
                                    log=lambda x: None)
    net.prune_sparsity = LEVEL
    return net.eval()


def unit_state_keys(tag: str, sd_keys: list[str], with_bn: bool = False) -> list[str]:
    if tag == "other":
        own = set()
        for u in UNIT_TAGS:
            own |= set(unit_state_keys(u, sd_keys, with_bn))
        return [k for k in sd_keys if k not in own]
    kind, name = unit_by_tag(tag)
    if kind == "lin" and with_bn and ".convs." in name:
        i = int(name.rsplit(".", 1)[1])
        bn = f"{name.rsplit('.', 1)[0]}.{i + 1}"
        ks = [f"{name}.weight", f"{name}.bias"] + [k for k in sd_keys if k.startswith(bn + ".")]
    elif kind == "lin":
        ks = [f"{name}.weight", f"{name}.bias"]
    elif kind == "mha_in":
        ks = [f"{name}.in_proj_weight", f"{name}.in_proj_bias"]
    elif kind == "mha_out":
        ks = [f"{name}.out_proj.weight", f"{name}.out_proj.bias"]
    else:
        ks = [f"{name}.{p}_l0" for p in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")]
    missing = [k for k in ks if k not in sd_keys]
    if missing:
        raise KeyError(missing)
    return ks


def swap_pair(ctx, smoke: bool):
    """(l1_fullrecon 80 % seed 0, fmkd 80 % e40 lr 3e-4 from it); masks must be identical."""
    pa = C3.B_CKPT / f"fold{ctx.k}-Ansel-l1_fullrecon-80pct-seed0-damp0.01.pt"
    pb = V4_CKPT / f"fold{ctx.k}-Ansel-fmkd_from_l1_fullrecon-80pct-seed0-lr0.0003-e40.pt"
    if smoke or not (pa.exists() and pb.exists()):
        Xc = C3.calibration(ctx, 0, C3.CAL["per_cell"], smoke)
        a = GB.build("l1_fullrecon", ctx.net0, LEVEL, Xc, log=lambda x: None)
        b = copy.deepcopy(a)
        with torch.no_grad():
            for n, p in b.named_parameters():
                p.add_(0.01 * torch.randn_like(p))
        from har.compression import apply_masks
        apply_masks(b)
        return a, b
    a, b = C3.load_ckpt(ctx, pa), C3.load_ckpt(ctx, pb)
    for t in a.prune_masks:
        if not torch.equal(a.prune_masks[t].cpu(), b.prune_masks[t].cpu()):
            raise RuntimeError(f"swap pair masks differ in {t}")
    return a, b


def build_swap(r: dict, ctx, pair):
    a, b = pair
    d, u = r["method"].split("_", 1)
    with_bn = d.endswith("2")
    base, donor = (a, b) if d.startswith("swapin") else (b, a)
    net = copy.deepcopy(base)
    sd = net.state_dict()
    dsd = donor.state_dict()
    for k in unit_state_keys(u, list(sd.keys()), with_bn):
        sd[k] = dsd[k].clone()
    net.load_state_dict(sd)
    net.prune_masks = {n: m.clone() for n, m in base.prune_masks.items()}
    net.prune_sparsity = LEVEL
    return net.eval()


# ------------------------------------------------------------------------------ oracle (upper bound)

def jfm_extra_loss(teacher, X_train: torch.Tensor, lam: float, seed: int):
    """v4's function-matching term (fmkd) + lam * relative squared error of the explained logit's input gradient on
    SHAP path points z = b + a (x - b) (b a random training window, a ~ U(0, 1))."""
    fm = V4.fm_extra_loss(teacher, X_train, seed)
    g = torch.Generator().manual_seed(13_000 + seed)

    def f(model, idx):
        loss = fm(model, idx)
        sub = idx[:ORC_SUB]
        n = len(sub)
        x = X_train[sub]
        j = torch.randint(0, len(X_train), (n,), generator=g).to(X_train.device)
        a = torch.rand(n, 1, 1, generator=g).to(X_train.device)
        z = (X_train[j] + a * (x - X_train[j])).detach()
        with torch.no_grad():
            c = teacher(x).argmax(1)
        with attribution_grad_context():
            zt = z.clone().requires_grad_(True)
            gt, = torch.autograd.grad(teacher(zt).gather(1, c[:, None]).sum(), zt)
            zs = z.clone().requires_grad_(True)
            gs, = torch.autograd.grad(model(zs).gather(1, c[:, None]).sum(), zs, create_graph=True)
        gt = gt.detach()
        num = ((gs - gt) ** 2).flatten(1).sum(1)
        den = (gt ** 2).flatten(1).sum(1).clamp_min(1e-12)
        return loss + lam * (num / den).mean()
    return f


def build_fmo(r: dict, ctx, a):
    """v4's fmkd (KD + function matching on mixup blends of training windows) from a v5 one-shot checkpoint (masks
    kept): the same-mask training reference for that baseline."""
    p = CKPT / f"fold{ctx.k}-Ansel-{r['start']}.pt"
    if a.smoke or not p.exists():
        if not a.smoke:
            raise FileNotFoundError(p)
        Xc = C3.calibration(ctx, 0, C3.CAL["per_cell"], True)
        net, _ = B5.build(r["start"].split("-")[0], ctx.net0, r["level"], Xc)
    else:
        net = C3.load_ckpt(ctx, p)
    epochs = 1 if a.smoke else r["epochs"]
    zt = torch.from_numpy(predict_logits(ctx.net0, ctx.tr.X, ctx.dev)).to(a.ft_device)
    teacher = copy.deepcopy(ctx.net0).to(a.ft_device).eval()
    extra = V4.fm_extra_loss(teacher, torch.from_numpy(ctx.tr.X).to(a.ft_device), r["seed"])
    info = finetune(net, ctx.tr, ctx.va, epochs=epochs, lr=r["lr"], seed=r["seed"], device=a.ft_device,
                    kd_teacher_logits=zt, extra_loss=extra, log=lambda x: None)
    info["start_source"] = p.name
    return net.to(ctx.dev).eval(), info


def grad_match_loss(teacher, X: torch.Tensor, B: torch.Tensor, c_all: torch.Tensor | None, lam: float, seed: int,
                    n_sub: int = ORC_SUB):
    """lam * sum ||g_S - g_T||^2 / sum ||g_T||^2 over n_sub path points z = b + a (x - b), x from X, b from B,
    a ~ U(0, 1), g = d f_c / dz, c = the teacher's class on x (batch-normalised: v5's jfm divided per point, which
    lets near-zero teacher gradients dominate)."""
    g = torch.Generator().manual_seed(15_000 + seed)

    def f(model):
        n = n_sub
        i = torch.randint(0, len(X), (n,), generator=g).to(X.device)
        j = torch.randint(0, len(B), (n,), generator=g).to(X.device)
        a = torch.rand(n, 1, 1, generator=g).to(X.device)
        x = X[i]
        z = (B[j] + a * (x - B[j])).detach()
        with torch.no_grad():
            c = teacher(x).argmax(1) if c_all is None else c_all[i]
        with attribution_grad_context():
            zt = z.clone().requires_grad_(True)
            gt, = torch.autograd.grad(teacher(zt).gather(1, c[:, None]).sum(), zt)
            zs = z.clone().requires_grad_(True)
            gs, = torch.autograd.grad(model(zs).gather(1, c[:, None]).sum(), zs, create_graph=True)
        gt = gt.detach()
        return lam * ((gs - gt) ** 2).sum() / (gt ** 2).sum().clamp_min(1e-12)
    return f


def build_orc2(r: dict, ctx, a):
    """jfm2: fmkd (v4) + batch-normalised gradient-field matching on SHAP path points of TRAINING windows (b = random
    training windows). capfit (capacity oracle, NOT a method and not a fair comparison: it fits the evaluated inputs):
    fmkd + the same term on path points of the fold's 150 inner-val explain windows with the explainer's own
    background and target class; it bounds what an 80 % / 90 % model can represent, not what can be learned."""
    p = CKPT / f"fold{ctx.k}-Ansel-{r['start']}.pt"
    if a.smoke:
        Xc = C3.calibration(ctx, 0, C3.CAL["per_cell"], True)
        net, _ = B5.build(r["start"].split("-")[0], ctx.net0, r["level"], Xc)
    elif p.exists():
        net = C3.load_ckpt(ctx, p)
    else:
        net, _ = V4.load_start(ctx, r["start"].split("-")[0], r["level"], False, 0)
    epochs = 1 if a.smoke else r["epochs"]
    dev = a.ft_device
    zt = torch.from_numpy(predict_logits(ctx.net0, ctx.tr.X, ctx.dev)).to(dev)
    teacher = copy.deepcopy(ctx.net0).to(dev).eval()
    for q in teacher.parameters():
        q.requires_grad_(False)
    Xtr = torch.from_numpy(ctx.tr.X).to(dev)
    fm = V4.fm_extra_loss(teacher, Xtr, r["seed"])
    if r["stage"] == "orc2":
        gm = grad_match_loss(teacher, Xtr, Xtr, None, r["lam"], r["seed"])
    else:
        gm = grad_match_loss(teacher, ctx.X_ex.to(dev), ctx.bg.to(dev),
                             torch.as_tensor(ctx.target_ex, device=dev).long(), r["lam"], r["seed"])
    extra = lambda model, idx: fm(model, idx) + gm(model)
    info = finetune(net, ctx.tr, ctx.va, epochs=epochs, lr=r["lr"], seed=r["seed"], device=dev,
                    kd_teacher_logits=zt, extra_loss=extra, log=lambda x: None)
    info["start_source"] = p.name
    return net.to(ctx.dev).eval(), info


def build_orc(r: dict, ctx, a):
    net, src = V4.load_start(ctx, r["start"], r["level"], a.smoke, r["seed"])
    epochs = 1 if a.smoke else r.get("epochs", ORC_EPOCHS)
    zt = torch.from_numpy(predict_logits(ctx.net0, ctx.tr.X, ctx.dev)).to(a.ft_device)
    teacher = copy.deepcopy(ctx.net0).to(a.ft_device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    extra = jfm_extra_loss(teacher, torch.from_numpy(ctx.tr.X).to(a.ft_device), r["lam"], r["seed"])
    info = finetune(net, ctx.tr, ctx.va, epochs=epochs, lr=r["lr"], seed=r["seed"], device=a.ft_device,
                    kd_teacher_logits=zt, extra_loss=extra, log=lambda x: None)
    info["start_source"] = src
    return net.to(ctx.dev).eval(), info


@torch.no_grad()
def output_kl(ctx, net) -> dict:
    """KL(original || model) of the softmax outputs, on the ~980 calibration windows (label-free, what a deployer can
    compute: the criterion of reconstruction-driven allocation such as SPDY / OWL) and on the inner-val F1 windows."""
    out = {}
    for nm, X in (("cal_kl", ctx.X_cal), ("val_kl", torch.from_numpy(ctx.va.X[:4000]).to(ctx.dev))):
        p0 = torch.log_softmax(ctx.net0.eval()(X), 1)
        p1 = torch.log_softmax(net.eval()(X), 1)
        out[nm] = float((p0.exp() * (p0 - p1)).sum(1).mean())
    return out


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
                                    "ceiling50": refs.ceiling50, "calibration": C3.CAL, "level": LEVEL,
                                    "orc_epochs": ORC_EPOCHS, "orc_sub": ORC_SUB, "kd": KD,
                                    "obc_damp": B5.OBC_DAMP, "dsnot_cycles": B5.DSNOT_CYCLES,
                                    "plan": plan(a.stages, a.smoke)}, indent=2, default=str), encoding="utf-8")
    base = {"fold": k, "machine_name": a.machine_name, "device": a.device, "ceiling50": refs.ceiling50}
    ck_dir = (Path(a.out) / "ckpt") if a.out else CKPT
    pair = None
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
            if r["stage"] == "nr5":
                m = r["method"]
                if m.endswith("_cal5x"):
                    Xc = C3.calibration(ctx, r["seed"], C3.CAL["per_cell_5x"], a.smoke)
                else:
                    Xc = C3.calibration(ctx, r["seed"], C3.CAL["per_cell"], a.smoke)
                if m.endswith("_mixcal"):
                    Xc = V4.blend(Xc, r["seed"])
                net, info = B5.build(m.replace("_cal5x", "").replace("_mixcal", ""), ctx.net0, r["level"], Xc)
                t_build = time.perf_counter() - ts
            elif r["stage"] == "abl":
                net = build_abl(r, ctx, a.smoke)
                t_build = time.perf_counter() - ts
            elif r["stage"] == "swap":
                if pair is None:
                    pair = swap_pair(ctx, a.smoke)
                net = build_swap(r, ctx, pair)
            elif r["stage"] == "orc":
                net, info = build_orc(r, ctx, a)
                t_ft = info.get("seconds", time.perf_counter() - ts)
            elif r["stage"] in ("orc2", "cap"):
                net, info = build_orc2(r, ctx, a)
                t_ft = info.get("seconds", time.perf_counter() - ts)
            elif r["stage"] == "fmo":
                net, info = build_fmo(r, ctx, a)
                t_ft = info.get("seconds", time.perf_counter() - ts)
            else:
                raise ValueError(r["stage"])
            if not head_unpruned(net):
                raise RuntimeError(f"{rid}: head pruned")
            o, rows = C3.score(ctx, refs, net, f"gs5-{rid}", True, ctx.nsamples, attr_dir / f"fold{k}-{rid}.npz")
            o.update(output_kl(ctx, net))
            if r["stage"] in ("nr5", "orc", "abl", "fmo", "orc2", "cap"):
                ck = ck_dir / f"fold{k}-{a.machine_name}-{rid}.pt"
                ck.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"run_id": rid, "fold": k, "state_dict": {n: t.cpu() for n, t in net.state_dict().items()},
                            "prune_masks": {n: m.cpu() for n, m in net.prune_masks.items()}}, ck)
        sz = size_record(net)
        C3._append(models_csv, [{**base, "run_id": rid, **r, **o, "build_seconds": t_build, "ft_seconds": t_ft,
                                 "start_source": info.get("start_source"), "epochs_run": info.get("epochs_run"),
                                 "dsnot_swaps": info.get("dsnot_swaps"), "keep_lstm": info.get("keep_lstm"),
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
