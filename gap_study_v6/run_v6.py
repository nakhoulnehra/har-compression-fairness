"""Gap study v6 (2026-11-28, Ansel): search for a NEW explanation gap after compression. No new method.
HARTH inner val only (HAR70+ and the folds' test subjects are never loaded here).

Candidate A (theory in gap_study_v6/GAP_REPORT.md section 1): compression that LEAVES the original's weight basin.
Every gap so far was in-place pruning, where the compressed model starts from the original's weights and inherits its
per-feature evidence split; v5 found explanation fidelity then tracks output fidelity. A dense narrow student (TinyHAR
with fewer filters: real MAC / latency reduction, no sparse kernels, the MCU deployment form) has no such inheritance.
Output matching on blends (fmkd) pins the SUM of the GradientExplainer attribution along each path (completeness),
not its split over the 48 features, so the split is a Rashomon choice unless something else pins it.

    $env:PYTHONPATH = "bit-reduction-test"
    python -m gap_study_v6.run_v6 --folds 0 --stages ref stu:12 --device cuda --worker s12 --machine-name Ansel --resume
    python -m gap_study_v6.run_v6 --folds 0 --stages ref stu:12 --smoke --out <scratchpad>

Scoring = gap_study_v3.confirm.score (unchanged protocol): 50 explain windows per inner-val subject, GradientExplainer
nsamples 200, target = the original's predicted class, background = the fold's 100 training windows; reported metric
lci_indep (model SHAP seed 1 vs the original's seed 0); occlusion LCI as a gradient-free check.

Stage stu:F[:E]  (student width F filters, E epochs, default 30; all seed 0, batch 256, Adam, last state kept)
  ce          TinyHAR(F) from scratch, class-weighted CE on labels (the label-only floor), lr 1e-3
  kd          Hinton et al. 2015 KD (T 3, alpha 0.7: the repo's shared recipe) from scratch, lr 1e-3
  fmkd        KD + function matching on mixup blends of training windows (Beyer et al. CVPR 2022; v4's recipe; the
              strongest training recipe of v4 / v5), from scratch, lr 1e-3
  fmjm        fmkd + Jacobian matching (Srinivas & Fleuret, ICML 2018) of the explained logit's input gradient on
              SHAP-style path points of TRAINING windows (v5's grad_match_loss, lam 10), from scratch, lr 1e-3
  slice_fmkd  structured filter pruning of the original to width F (Li et al. ICLR 2017: L1 filter / row norms,
              every dependent tensor sliced), then fmkd, lr 3e-4 (v4 / v5 fine-tuning lr): the existing structured
              route that keeps part of the original's weights
  capfit      ORACLE, not a method: fmkd + the same gradient-field term on path points of the 150 EVALUATED inner-val
              windows with the explainer's background and target class (v5's capacity oracle), from scratch,
              lr 1e-3; bounds what a width-F model can represent
  slice_capfit  ORACLE from the sliced start, lr 3e-4
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
import gap_study_v3.confirm as C3  # noqa: E402
import gap_study_v4.run_v4 as V4  # noqa: E402
import gap_study_v5.baselines_v5 as B5  # noqa: E402
import gap_study_v5.run_v5 as V5  # noqa: E402
import gap_study_v6.quant_v6 as Q6  # noqa: E402
from har import runlog as R  # noqa: E402
from har.compression import KD, finetune, predict_logits, size_record  # noqa: E402
from har.models import HarModel, TinyHAR  # noqa: E402

OUT = ROOT / "results" / "gap-study-v6" / "validation"
P = "gap-study-v6-validation-tinyhar"
CKPT = C3.CHECKPOINT_DIR / "gap-study-v6"
ATTR = C3.PROCESSED_DIR / "gap-study-v6" / "attributions"
STU_METHODS = ["ce", "kd", "fmkd", "fmjm", "slice_fmkd", "capfit", "slice_capfit"]
LAM = 10.0


# ------------------------------------------------------------------------------ structured slicing (Li et al. 2017)

def _top(scores: torch.Tensor, k: int) -> torch.Tensor:
    return torch.sort(torch.topk(scores, k).indices).values


@torch.no_grad()
def slice_tinyhar(net0, F: int):
    """Width-F TinyHAR whose every tensor is a slice of the original's: per conv layer the F filters with the largest
    L1 norm; the last conv's subset is the transformer's residual subset; FF hidden, fusion outputs and LSTM hidden
    units by the L1 norm of their incoming rows (LSTM: all four gates). BatchNorm / LayerNorm sliced with their
    channels. Attention scale changes from 1/sqrt(F0) to 1/sqrt(F) (fine-tuning absorbs it)."""
    wrapper = net0 if isinstance(net0, HarModel) else None
    if wrapper is not None:
        net0 = net0.backbone
    F0 = net0.head.in_features // 2
    sd0 = {k: v.detach().cpu() for k, v in net0.state_dict().items()}
    stu = TinyHAR(filters=F)
    sd = stu.state_dict()
    out = {}
    prev = None
    for ci in (0, 3, 6, 9):
        w = sd0[f"convs.{ci}.weight"]                       # (F0, in, 5, 1)
        if prev is not None:
            w = w[:, prev]
        S = _top(w.abs().flatten(1).sum(1), F)
        out[f"convs.{ci}.weight"] = w[S]
        out[f"convs.{ci}.bias"] = sd0[f"convs.{ci}.bias"][S]
        for nm in ("weight", "bias", "running_mean", "running_var"):
            out[f"convs.{ci + 1}.{nm}"] = sd0[f"convs.{ci + 1}.{nm}"][S]
        out[f"convs.{ci + 1}.num_batches_tracked"] = sd0[f"convs.{ci + 1}.num_batches_tracked"]
        prev = S
    S = prev
    a = "channel_attn."
    W, b = sd0[a + "self_attn.in_proj_weight"], sd0[a + "self_attn.in_proj_bias"]
    rows = torch.cat([S + j * F0 for j in range(3)])
    out[a + "self_attn.in_proj_weight"] = W[rows][:, S]
    out[a + "self_attn.in_proj_bias"] = b[rows]
    out[a + "self_attn.out_proj.weight"] = sd0[a + "self_attn.out_proj.weight"][S][:, S]
    out[a + "self_attn.out_proj.bias"] = sd0[a + "self_attn.out_proj.bias"][S]
    l1 = sd0[a + "linear1.weight"][:, S]
    Hf = _top(l1.abs().sum(1), 2 * F)
    out[a + "linear1.weight"] = l1[Hf]
    out[a + "linear1.bias"] = sd0[a + "linear1.bias"][Hf]
    out[a + "linear2.weight"] = sd0[a + "linear2.weight"][S][:, Hf]
    out[a + "linear2.bias"] = sd0[a + "linear2.bias"][S]
    for nm in ("norm1", "norm2"):
        for q in ("weight", "bias"):
            out[f"{a}{nm}.{q}"] = sd0[f"{a}{nm}.{q}"][S]
    cols = torch.cat([c * F0 + S for c in range(net0.fusion[0].in_features // F0)])
    fw = sd0["fusion.0.weight"][:, cols]
    H1 = _top(fw.abs().sum(1), 2 * F)
    out["fusion.0.weight"] = fw[H1]
    out["fusion.0.bias"] = sd0["fusion.0.bias"][H1]
    H0 = 2 * F0
    wih, whh = sd0["lstm.weight_ih_l0"], sd0["lstm.weight_hh_l0"]
    unit = sum(wih[g * H0:(g + 1) * H0][:, H1].abs().sum(1) for g in range(4))
    H2 = _top(unit, 2 * F)
    r2 = torch.cat([H2 + g * H0 for g in range(4)])
    out["lstm.weight_ih_l0"] = wih[r2][:, H1]
    out["lstm.weight_hh_l0"] = whh[r2][:, H2]
    out["lstm.bias_ih_l0"] = sd0["lstm.bias_ih_l0"][r2]
    out["lstm.bias_hh_l0"] = sd0["lstm.bias_hh_l0"][r2]
    out["attn_score.weight"] = sd0["attn_score.weight"][:, H2]
    out["attn_score.bias"] = sd0["attn_score.bias"]
    out["gamma"] = sd0["gamma"]
    out["head.weight"] = sd0["head.weight"][:, H2]
    out["head.bias"] = sd0["head.bias"]
    missing = set(sd) - set(out)
    if missing:
        raise KeyError(sorted(missing))
    for k_, v in out.items():
        if tuple(v.shape) != tuple(sd[k_].shape):
            raise ValueError(f"{k_}: {tuple(v.shape)} vs {tuple(sd[k_].shape)}")
    stu.load_state_dict(out)
    return wrap_like(wrapper, stu, F) if wrapper is not None else stu


def wrap_like(net0: HarModel, backbone: TinyHAR, F: int) -> HarModel:
    """Width-F student with the original's (fixed) input normalisation."""
    m = HarModel("tinyhar", {**net0.cfg, "filters": F}, backbone, torch.zeros(1), torch.ones(1))
    m.norm = copy.deepcopy(net0.norm).cpu()
    return m


# ------------------------------------------------------------------------------ plan / build

def plan(stages: list[str], smoke: bool) -> list[dict]:
    rows = []
    for st in stages:
        if st == "ref":
            rows.append({"stage": "ref", "method": "original", "width": 28, "seed": 0})
        elif st.startswith("stu:"):      # stu:F[:E[:methods,comma]]
            f = st.split(":")
            F = int(f[1])
            E = int(f[2]) if len(f) > 2 and f[2] else 30
            ms = f[3].split(",") if len(f) > 3 else STU_METHODS
            for m in ms:
                if m not in STU_METHODS:
                    raise ValueError(m)
                rows.append({"stage": "stu", "method": m, "width": F, "seed": 0, "epochs": E,
                             "lr": 3e-4 if m.startswith("slice") else 1e-3})
        elif st.startswith("scarce:"):   # scarce:N:seed,seed[:methods]  candidate B (one-shot, 80 %)
            f = st.split(":")
            ms = f[3].split(",") if len(f) > 3 else ["akl", "akl_mix", "obcu"]
            for sd in (int(x) for x in f[2].split(",")):
                for m in ms:
                    rows.append({"stage": "scarce", "method": m, "n_cal": int(f[1]), "level": 0.8, "seed": sd,
                                 "width": 28})
        elif st.startswith("quant:"):    # quant:bits[:methods]  candidate D (one-shot weight-only PTQ)
            f = st.split(":")
            for m in (f[2].split(",") if len(f) > 2 else ["rtn", "gptq"]):
                rows.append({"stage": "quant", "method": m, "bits": int(f[1]), "seed": 0, "width": 28})
        elif st.startswith("qat:"):      # qat:bits:epochs:lr[:p8]  same-size maximum for D (QAT + fmkd from gptq)
            f = st.split(":")
            p8 = len(f) > 4 and f[4] == "p8"
            rows.append({"stage": "qat", "method": "qat_fmkd_p8" if p8 else "qat_fmkd", "bits": int(f[1]),
                         "epochs": int(f[2]), "lr": float(f[3]), "seed": 0, "width": 28})
        else:
            raise ValueError(st)
    return rows


def scarce_calibration(ctx, n: int, seed: int) -> torch.Tensor:
    """n training windows drawn uniformly at random, no labels used (an unlabelled on-device stream)."""
    g = np.random.default_rng(21_000 + seed)
    pick = np.sort(g.choice(len(ctx.tr.X), size=n, replace=False))
    return torch.from_numpy(ctx.tr.X[pick]).to(ctx.dev)


@torch.no_grad()
def kl_on(net0, net, X) -> float:
    p0 = torch.log_softmax(net0.eval()(X), 1)
    p1 = torch.log_softmax(net.eval()(X), 1)
    return float((p0.exp() * (p0 - p1)).sum(1).mean())


def build_scarce(r: dict, ctx, a):
    """Candidate B: the strongest existing one-shot combination of v5 with only n_cal unlabelled windows.
    akl: OBC + full refit for each of v5's 7 group allocations, the one with the lowest output KL on the SAME n_cal
    windows is kept (label-free, backprop-free; SPDY / OWL-style). akl_mix: the same on the n_cal windows plus n_cal
    mixup blends of them (v4's blended calibration; the KL choice still on the n_cal real windows). obcu: uniform
    OBC + full refit."""
    n = 6 if a.smoke else r["n_cal"]
    Xc = scarce_calibration(ctx, n, r["seed"])
    Xb = V4.blend(Xc, r["seed"]) if r["method"] == "akl_mix" else Xc
    names = ["a0"] if r["method"] == "obcu" else list(B5.ALLOCS)
    best, info = None, {"kl_per_alloc": {}}
    for nm in names:
        net, _ = B5.build(f"alloc_{nm}", ctx.net0, r["level"], Xb)
        kl = kl_on(ctx.net0, net, Xc)
        info["kl_per_alloc"][nm] = kl
        if best is None or kl < best[0]:
            best = (kl, nm, net)
    info["alloc"] = best[1]
    return best[2], info


def run_id(r: dict) -> str:
    if r["stage"] == "ref":
        return "original-w28-seed0"
    if r["stage"] == "scarce":
        return f"{r['method']}-n{r['n_cal']}-80pct-seed{r['seed']}"
    if r["stage"] == "quant":
        return f"{r['method']}-{r['bits']}bit-seed{r['seed']}"
    if r["stage"] == "qat":
        return f"{r['method']}_from_gptq-{r['bits']}bit-seed{r['seed']}-lr{r['lr']:g}-e{r['epochs']}"
    return f"{r['method']}-w{r['width']}-seed{r['seed']}-lr{r['lr']:g}-e{r['epochs']}"


def build_student(r: dict, ctx, a):
    dev = a.ft_device
    F, m = r["width"], r["method"]
    torch.manual_seed(r["seed"])
    net = slice_tinyhar(ctx.net0, F) if m.startswith("slice") else wrap_like(ctx.net0, TinyHAR(filters=F), F)
    net.prune_masks = {}
    epochs = 1 if a.smoke else r["epochs"]
    teacher = copy.deepcopy(ctx.net0).to(dev).eval()
    for q in teacher.parameters():
        q.requires_grad_(False)
    zt = None if m == "ce" else torch.from_numpy(predict_logits(ctx.net0, ctx.tr.X, ctx.dev)).to(dev)
    Xtr = torch.from_numpy(ctx.tr.X).to(dev)
    extra = None
    if m in ("fmkd", "slice_fmkd", "fmjm", "capfit", "slice_capfit"):
        fm = V4.fm_extra_loss(teacher, Xtr, r["seed"])
        extra = fm
        if m == "fmjm":
            gm = V5.grad_match_loss(teacher, Xtr, Xtr, None, LAM, r["seed"])
            extra = lambda model, idx: fm(model, idx) + gm(model)
        elif m in ("capfit", "slice_capfit"):
            gm = V5.grad_match_loss(teacher, ctx.X_ex.to(dev), ctx.bg.to(dev),
                                    torch.as_tensor(ctx.target_ex, device=dev).long(), LAM, r["seed"])
            extra = lambda model, idx: fm(model, idx) + gm(model)
    info = finetune(net, ctx.tr, ctx.va, epochs=epochs, lr=r["lr"], seed=r["seed"], device=dev,
                    kd_teacher_logits=zt, extra_loss=extra, log=lambda x: print(x, flush=True))
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
                                    "ceiling50": refs.ceiling50, "kd": KD, "lam": LAM,
                                    "plan": plan(a.stages, a.smoke)}, indent=2, default=str), encoding="utf-8")
    base = {"fold": k, "machine_name": a.machine_name, "device": a.device, "ceiling50": refs.ceiling50}
    ck_dir = (Path(a.out) / "ckpt") if a.out else CKPT
    for r in plan(a.stages, a.smoke):
        rid = run_id(r)
        if rid in done:
            continue
        ts = time.perf_counter()
        info, t_ft = {}, 0.0
        if r["stage"] == "ref":
            f1, acc = C3._f1acc(ctx, ctx.net0)
            o = {"f1": f1, "accuracy": acc, "lci": 1.0, "lci_indep": refs.ceiling50, "self_lci": refs.ceiling50,
                 "occ_lci": 1.0, "agree": 1.0}
            rows, net = [], ctx.net0
        else:
            if r["stage"] == "scarce":
                net, info = build_scarce(r, ctx, a)
            elif r["stage"] == "qat":
                p8 = r["method"].endswith("_p8")
                start = Q6.build_quant("gptq_p8" if p8 else "gptq", ctx.net0, r["bits"],
                                       C3.calibration(ctx, 0, C3.CAL["per_cell"], a.smoke))
                net, info = Q6.qat_fmkd(start, ctx, r["bits"], 1 if a.smoke else r["epochs"], r["lr"], r["seed"],
                                        a.ft_device, protect=p8)
                net = net.to(ctx.dev).eval()
            elif r["stage"] == "quant":
                net = Q6.build_quant(r["method"], ctx.net0, r["bits"], C3.calibration(ctx, 0, C3.CAL["per_cell"], a.smoke))
            else:
                net, info = build_student(r, ctx, a)
            t_ft = info.get("seconds", time.perf_counter() - ts)
            o, rows = C3.score(ctx, refs, net, f"gs6-{rid}", True, ctx.nsamples, attr_dir / f"fold{k}-{rid}.npz")
            o.update(V5.output_kl(ctx, net))
            ck = ck_dir / f"fold{k}-{a.machine_name}-{rid}.pt"
            ck.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"run_id": rid, "fold": k, "width": r["width"],
                        "prune_masks": {n: m.cpu() for n, m in (getattr(net, "prune_masks", None) or {}).items()},
                        "state_dict": {n: t.cpu() for n, t in net.state_dict().items()}}, ck)
            if info.get("history"):
                pd.DataFrame(info["history"]).assign(run_id=rid, fold=k).to_csv(
                    f"{stem}-history.csv", mode="a", header=not Path(f"{stem}-history.csv").exists(), index=False)
        n_params = sum(p.numel() for p in net.parameters())
        C3._append(models_csv, [{**base, "run_id": rid, **r, **o, "ft_seconds": t_ft, "n_params": n_params,
                                 "epochs_run": info.get("epochs_run"), "alloc": info.get("alloc"),
                                 "kl_per_alloc": json.dumps(info.get("kl_per_alloc")) if info.get("kl_per_alloc") else None,
                                 "seconds_total": time.perf_counter() - ts}])
        if rows:
            C3._append(subj_csv, [{**base, "run_id": rid, **r, **x} for x in rows])
        print(f"fold {k} {rid}: F1 {o['f1']:.3f} acc {o['accuracy']:.3f} LCI {o['lci']:.3f} "
              f"indep {o.get('lci_indep', float('nan')):.3f} self {o.get('self_lci', float('nan')):.3f} "
              f"occ {o.get('occ_lci', float('nan')):.3f} agree {o.get('agree', float('nan')):.3f} "
              f"calKL {o.get('cal_kl', float('nan')):.4f} params {n_params} "
              f"[{time.perf_counter() - ts:.0f}s, total {time.perf_counter() - t0:.0f}s]", flush=True)
    print(f"fold {k} stages {a.stages} done in {time.perf_counter() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", type=int, nargs="+", default=[0])
    ap.add_argument("--stages", nargs="+", default=["ref"])
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="scoring device")
    ap.add_argument("--ft-device", choices=["cpu", "cuda"], default="cuda", help="training device")
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
