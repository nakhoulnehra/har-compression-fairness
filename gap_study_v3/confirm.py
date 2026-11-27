"""Gap study v3, Phase 2: confirm the candidate gaps with fairly tuned EXISTING baselines + metric-artefact checks.

No new method is built here. Every model is an existing technique, or a checkpoint of an earlier run re-scored.
Validation only: HARTH inner val of each fold (HAR70+ and the fold's test subjects are never loaded).

    $env:PYTHONPATH = "bit-reduction-test"
    python -m gap_study_v3.confirm --folds 0 --stages ref rescore ftcore nr ftextra rescore9a --threads 3 --machine-name Ansel --resume
    python -m gap_study_v3.confirm --folds 0 1 --smoke ...       (tiny sizes, output to --out)

Protocol (identical to agent B's Phase 2, so its numbers and SHAP cache are reused): bitred.context.FoldContext with
50 explain windows per inner-val subject, GradientExplainer nsamples 200, target = the original model's predicted
class, background = the fold's 100 training windows, calibration = round 4's <= 10 training windows per
(training subject, class), seed s = rng seed 46 + s.

Stages
  ref        original model: SHAP seed 1 on the same windows (SHAP-noise ceiling on these windows), occlusion
  rescore    agent B's no-retraining checkpoints (gap_study Phase 2) re-scored with the artefact checks
  nr         no-retraining baselines agent B did not run fairly: sparsegpt at percdamp 0.1 / 1.0 (agent B: 0.01 only),
             taylor_fullrecon (explanation-target first-order Taylor selection, |w g|^2 per mini-batch with
             g = d mean f_c / d w, c = original predicted class; Molchanov et al. 2019, cf. Yeom et al. 2021) with
             the full layer-wise refit, l1_fullrecon with 5x calibration windows (<= 50 per subject-class)
  ftcore     KD fine-tuning (9A recipe: 20 epochs, Adam lr 1e-4, KD T 3 alpha 0.7 + class-weighted CE, full HARTH
             training windows of the fold) from three starts: l1 one shot (= 9A l1_kd), l1_fullrecon, lamp_recon;
             90 %, seeds 0-2 (lamp_recon seed 0)
  ftextra    lr check (3e-5, 3e-4; seed 0) for l1_kd and l1_fullrecon_kd at 90 %, and the 80 % neighbour (seed 0)
  rescore9a  9A checkpoints present on this laptop (folds 0, 1, 3): l1_kd 70 / 90 %, park_kd and park 90 %

Artefact metrics per model (besides F1, accuracy, LCI as agent B):
  lci_indep  LCI(original SHAP seed 0, model SHAP seed 1): independent SHAP noise, comparable with ceiling50
  self_lci   LCI(model seed 0, model seed 1): is the explainer itself stable on the compressed model?
  occ_lci    LCI on occlusion attributions (48 segments replaced by the background mean; gradient-free)
  agree      share of explain windows where the model predicts the original's class; lci_agree = LCI on those only
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
import gap_study_v3.baselines as GB  # noqa: E402  (agent B's baselines, moved into this folder)
from har import runlog as R  # noqa: E402
from har.attributions import shap_cached  # noqa: E402
from har.compression import (attribution_grad_context, finetune, head_unpruned, l1_scores, predict_logits,  # noqa: E402
                             prunable_tensors, prune_by_score, size_record)
from har.data import window_ids  # noqa: E402
from har.paths import CHECKPOINT_DIR, PROCESSED_DIR  # noqa: E402
from har.train import predict_proba  # noqa: E402
from har.xgp4 import calibration_pick, iterative_prune_budget, uniform_keep_counts  # noqa: E402
from lci.metric import aggregate_time_segments, compute_lci_by_subject  # noqa: E402

OUT = ROOT / "results" / "gap-study-v3" / "phase2"
P = "gap-study-v3-phase2-tinyhar"
CKPT = CHECKPOINT_DIR / "gap-study-v3"
ATTR = PROCESSED_DIR / "gap-study-v3" / "attributions"      # git-ignored; Phase 3 diagnostics read these
B_CKPT = CHECKPOINT_DIR / "gap-study" / "phase2"
G9A_CKPT = CHECKPOINT_DIR / "har-compression-grid"
EXPLAIN_PER_SUBJECT = 50
CAL = {"per_cell": 10, "seed0": 46, "per_cell_5x": 50}
FT = {"epochs": 20, "lr": 1e-4}
SEG, NSEG, C = 16, 8, 6
FT_SELECTED: list[str] | None = None     # set by --ft-selected method:lr ...


# ------------------------------------------------------------------------------ plan

def plan(k: int, stages: list[str], smoke: bool) -> list[dict]:
    rows = []
    S = (0, 1) if smoke else (0, 1, 2)
    for st in stages:
        if st == "ref":
            rows.append({"stage": "ref", "method": "original", "level": 0.0, "seed": 0})
        elif st == "rescore":
            for lev, seeds in ((0.8, S), (0.7, (0,)), (0.9, (0,))):
                for m in ("l1_fullrecon", "lamp_recon", "wanda_fullrecon", "l1_recon"):
                    for s in (seeds if (m in ("l1_fullrecon", "lamp_recon") or lev != 0.8) else (0,)):
                        if lev != 0.8 and m in ("wanda_fullrecon", "l1_recon"):
                            continue
                        rows.append({"stage": "rescore", "method": m, "level": lev, "seed": s, "damp": 0.01})
        elif st == "nr":
            for d in (0.1, 1.0):
                for s in S:
                    rows.append({"stage": "nr", "method": "sparsegpt", "level": 0.8, "seed": s, "pdamp": d})
                for lev in (0.7, 0.9):
                    rows.append({"stage": "nr", "method": "sparsegpt", "level": lev, "seed": 0, "pdamp": d})
            for s in S:
                rows.append({"stage": "nr", "method": "taylor_fullrecon", "level": 0.8, "seed": s})
            for lev in (0.7, 0.9):
                rows.append({"stage": "nr", "method": "taylor_fullrecon", "level": lev, "seed": 0})
            rows.append({"stage": "nr", "method": "l1_fullrecon_cal5x", "level": 0.8, "seed": 0})
        elif st == "ftcore":
            for m in ("l1_kd", "l1_fullrecon_kd"):
                for s in S:
                    rows.append({"stage": "ftcore", "method": m, "level": 0.9, "seed": s, "lr": FT["lr"]})
            rows.append({"stage": "ftcore", "method": "lamp_recon_kd", "level": 0.9, "seed": 0, "lr": FT["lr"]})
        elif st == "ftextra":
            for m in ("l1_kd", "l1_fullrecon_kd"):
                for lr in (3e-4, 3e-5):
                    rows.append({"stage": "ftextra", "method": m, "level": 0.9, "seed": 0, "lr": lr})
            for m in ("l1_kd", "l1_fullrecon_kd"):
                rows.append({"stage": "ftextra", "method": m, "level": 0.8, "seed": 0, "lr": FT["lr"]})
        elif st == "rescore9a":
            for m, lev in (("l1_kd", 0.9), ("park_kd", 0.9), ("park", 0.9), ("l1_kd", 0.7)):
                rows.append({"stage": "rescore9a", "method": f"9a_{m}", "level": lev, "seed": 0})
        elif st == "ftseeds":   # seeds 1-2 at the validation-selected lr (gap_study_v3.analyze --select-lr; rule in RUNS.md)
            for spec in (FT_SELECTED or []):     # method:lr:seed,seed
                m, lr, ss = spec.split(":")
                for s in (int(x) for x in ss.split(",")):
                    rows.append({"stage": "ftextra", "method": m, "level": 0.9, "seed": s, "lr": float(lr)})
        elif st == "refs2":   # reference points for the artefact metrics (occlusion has no noise ceiling)
            rows.append({"stage": "rescore", "method": "l1_fullrecon", "level": 0.5, "seed": 0, "damp": 0.01})
            rows.append({"stage": "refs2", "method": "retrain_seed1", "level": 0.0, "seed": 1})
        else:
            raise ValueError(st)
    return rows


def run_id(r: dict) -> str:
    rid = f"{r['method']}-{int(round(r['level'] * 100))}pct-seed{r['seed']}"
    if "pdamp" in r:
        rid += f"-pdamp{r['pdamp']}"
    if "lr" in r:
        rid += f"-lr{r['lr']:g}-e{FT['epochs']}"
    return rid


# ------------------------------------------------------------------------------ existing techniques not in agent B's code

def sparsegpt_prune_damp(original, sparsity: float, X_cal: torch.Tensor, percdamp: float):
    """Agent B's sparsegpt_prune (SparseGPT, Frantar & Alistarh 2023, Alg. 1) with the dampening as a parameter."""
    net = copy.deepcopy(original).eval()
    GB.check_units_cover(net)
    uni = uniform_keep_counts(net, sparsity)
    masks = {}
    for kind, name in GB.UNITS:
        W, _ = GB.unit_weight(net, kind, name)
        H, _, _ = GB.unit_grams(net, None, X_cal, kind, name, False)
        groups, c0 = [], 0
        params = dict(net.named_parameters())
        for t in GB.unit_tensors(kind, name):
            w = params[t].numel() // params[t].shape[0]
            groups.append((c0, c0 + w, params[t].numel() - uni[t]))
            c0 += w
        Wn, keep = GB.sparsegpt_matrix(W, H, groups, percdamp=percdamp)
        GB.set_unit_weight(net, kind, name, Wn.to(W.dtype), None)
        params = dict(net.named_parameters())
        for (a, bb, _), t in zip(groups, GB.unit_tensors(kind, name)):
            masks[t] = keep[:, a:bb].reshape(params[t].shape).cpu()
    net.prune_masks = masks
    net.prune_sparsity = sparsity
    return net


def taylor_scores(net, X_cal: torch.Tensor, target: torch.Tensor, batch_size: int = 32) -> dict:
    """First-order Taylor importance of each weight for the EXPLAINED logit f_c (c = original predicted class):
    sum over mini-batches of (w * d mean_b f_c / d w)^2 (Molchanov et al. 2019 criterion, label-free target)."""
    net.eval()
    names = [n for n, _, _ in prunable_tensors(net)]
    params = dict(net.named_parameters())
    acc = {n: torch.zeros_like(params[n]) for n in names}
    with attribution_grad_context():
        for i in range(0, len(X_cal), batch_size):
            xb, tb = X_cal[i:i + batch_size], target[i:i + batch_size]
            net.zero_grad(set_to_none=True)
            out = net(xb).gather(1, tb[:, None]).mean()
            out.backward()
            for n in names:
                g = params[n].grad
                if g is not None:
                    acc[n] += (params[n].detach() * g.detach()) ** 2
    net.zero_grad(set_to_none=True)
    return acc


def calibration(ctx, seed: int, per_cell: int, smoke: bool):
    if seed == 0 and per_cell == CAL["per_cell"]:
        return ctx.X_cal
    tr = ctx.tr
    per = 2 if smoke else per_cell
    pick = calibration_pick(tr.y, tr.subject, per, CAL["seed0"] + seed + (0 if per_cell == CAL["per_cell"] else 100))
    return torch.from_numpy(tr.X[pick]).to(ctx.dev)


def build_nr(r: dict, ctx, smoke: bool):
    m, lev, s = r["method"], r["level"], r["seed"]
    if m == "sparsegpt":
        return sparsegpt_prune_damp(ctx.net0, lev, calibration(ctx, s, CAL["per_cell"], smoke), r["pdamp"])
    if m == "taylor_fullrecon":
        Xc = calibration(ctx, s, CAL["per_cell"], smoke)
        tgt = torch.from_numpy(predict_proba(ctx.net0, Xc.cpu().numpy(), ctx.dev).argmax(1)).long().to(ctx.dev)
        keep = uniform_keep_counts(ctx.net0, lev)
        net, _ = iterative_prune_budget(ctx.net0, keep, K=GB.K_STEPS, beta=1.0,
                                        importance_fn=lambda n: taylor_scores(n, Xc, tgt),
                                        recon_fn=lambda n: GB.full_reconstruct(n, ctx.net0, Xc, damp=GB.DAMP),
                                        log=lambda x: None)
        return net
    if m == "l1_fullrecon_cal5x":
        return GB.build("l1_fullrecon", ctx.net0, lev, calibration(ctx, s, CAL["per_cell_5x"], smoke), log=lambda x: None)
    raise ValueError(m)


def build_ft(r: dict, ctx, a, zt):
    m, lev, s = r["method"], r["level"], r["seed"]
    if m == "l1_kd":
        net = prune_by_score(ctx.net0, l1_scores(ctx.net0), lev)              # 9A's one-shot l1 start
    elif m in ("l1_fullrecon_kd", "lamp_recon_kd"):
        net = GB.build(m.replace("_kd", ""), ctx.net0, lev, calibration(ctx, s, CAL["per_cell"], a.smoke),
                       log=lambda x: None)
    else:
        raise ValueError(m)
    epochs = 1 if a.smoke else FT["epochs"]
    info = finetune(net, ctx.tr, ctx.va, epochs=epochs, lr=r["lr"], seed=s, device=a.ft_device,
                    kd_teacher_logits=zt, log=lambda x: None)
    net = net.to(ctx.dev).eval()
    return net, info


def load_ckpt(ctx, path: Path):
    c = torch.load(path, map_location="cpu", weights_only=False)
    net = copy.deepcopy(ctx.net0).cpu()
    net.load_state_dict(c["state_dict"])
    net.prune_masks = {n: m.bool() for n, m in c.get("prune_masks", {}).items()}
    return net.to(ctx.dev).eval()


# ------------------------------------------------------------------------------ scoring

@torch.no_grad()
def occlusion(net, X: torch.Tensor, target: np.ndarray, bg_mean: torch.Tensor, bs: int = 64) -> np.ndarray:
    """f_c(x) - f_c(x with segment (axis, 16 samples) replaced by the background mean), 48 features (agent A's
    definition in bitred/run_pair_occlusion.py)."""
    net.eval()
    out = np.zeros((len(X), C * NSEG), dtype=np.float32)
    for i in range(0, len(X), bs):
        x = X[i:i + bs]
        t = torch.as_tensor(target[i:i + bs], device=x.device).long()
        base = net(x).gather(1, t[:, None]).squeeze(1)
        for ax in range(C):
            for sg in range(NSEG):
                xo = x.clone()
                xo[:, ax, sg * SEG:(sg + 1) * SEG] = bg_mean[ax, sg * SEG:(sg + 1) * SEG]
                out[i:i + bs, ax * NSEG + sg] = (base - net(xo).gather(1, t[:, None]).squeeze(1)).cpu().numpy()
    return out


def lci_mean(A_ref, A, subj, mask=None) -> float:
    A_ref, A, subj = np.asarray(A_ref), np.asarray(A), np.asarray(subj)
    if mask is not None:
        A_ref, A, subj = A_ref[mask], A[mask], subj[mask]
    vals = []
    for s in sorted(set(subj.tolist())):
        if (subj == s).sum() >= 3:
            vals.append(compute_lci_by_subject(A_ref[subj == s], A[subj == s], subj[subj == s], k=10, omega=0.5)[s]["lci"])
    return float(np.mean(vals)) if vals else float("nan")


class Refs:
    def __init__(self, ctx, nsamples):
        self.seg = lambda A: aggregate_time_segments(A)
        self.A0 = self.seg(ctx.A_ref_ex)
        t = time.perf_counter()
        self.A0s1 = self.seg(shap_cached(ctx.net0, ctx.bg, ctx.X_ex, ctx.target_ex, ctx.ex_ids, 1, nsamples,
                                         ctx.cache, f"bitred-fold{ctx.k}-eval-original-seed1", log=lambda x: None))
        self.bg_mean = ctx.bg.mean(0)
        self.O0 = occlusion(ctx.net0, ctx.X_ex, ctx.target_ex, self.bg_mean)
        self.seconds = time.perf_counter() - t
        self.ceiling50 = lci_mean(self.A0, self.A0s1, ctx.ex_subj)


def score(ctx, refs: Refs, net, tag: str, extras: bool, nsamples: int, attr_path: Path | None):
    o, rows = ctx.assess(net, tag)
    A = refs.seg(ctx.shap_eval(net, tag))            # cache hit (computed by assess)
    logits = predict_proba(net, ctx.X_ex.cpu().numpy(), ctx.dev)
    pred = logits.argmax(1)
    agree = pred == ctx.target_ex
    o.update({"agree": float(agree.mean()), "lci_agree": lci_mean(refs.A0, A, ctx.ex_subj, agree),
              "lci_disagree": lci_mean(refs.A0, A, ctx.ex_subj, ~agree)})
    save = {"A0": A.astype(np.float32), "pred": pred, "proba": logits.astype(np.float32)}
    if extras:
        t = time.perf_counter()
        A1 = refs.seg(shap_cached(net, ctx.bg, ctx.X_ex, ctx.target_ex, ctx.ex_ids, 1, nsamples, ctx.cache,
                                  f"bitred-fold{ctx.k}-eval-{tag}-seed1", log=lambda x: None))
        O = occlusion(net, ctx.X_ex, ctx.target_ex, refs.bg_mean)
        o.update({"lci_indep": lci_mean(refs.A0, A1, ctx.ex_subj), "self_lci": lci_mean(A, A1, ctx.ex_subj),
                  "occ_lci": lci_mean(refs.O0, O, ctx.ex_subj), "extras_seconds": time.perf_counter() - t})
        save.update({"A1": A1.astype(np.float32), "O": O})
    if attr_path is not None:
        attr_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(attr_path, **save)
    return o, rows


# ------------------------------------------------------------------------------ fold loop

def run_fold(k: int, a) -> None:
    out = Path(a.out) if a.out else OUT
    out.mkdir(parents=True, exist_ok=True)
    tagw = "-".join(a.stages)
    stem = out / f"{P}-fold{k}-{a.machine_name}-{a.worker or tagw}"
    models_csv, subj_csv = Path(f"{stem}-models.csv"), Path(f"{stem}-per-subject-lci.csv")
    done = set(pd.read_csv(models_csv).run_id) if (a.resume and models_csv.exists()) else set()
    if not a.resume and models_csv.exists():
        raise SystemExit(f"{models_csv} exists; pass --resume")
    BC.DESIGN["explain_per_subject"] = EXPLAIN_PER_SUBJECT
    if a.smoke:
        BC.SMOKE["explain_per_subject"] = 3
    t0 = time.perf_counter()
    ctx = BC.FoldContext(k, a.device, a.smoke, None)
    refs = Refs(ctx, ctx.nsamples)
    attr_dir = (Path(a.out) / "attributions") if a.out else ATTR
    np.savez_compressed(attr_dir.mkdir(parents=True, exist_ok=True) or attr_dir / f"fold{k}-original-refs.npz",
                        A0=refs.A0, A0s1=refs.A0s1, O0=refs.O0, target=ctx.target_ex, subj=np.asarray(ctx.ex_subj),
                        ex_ids=np.asarray(ctx.ex_ids))
    cfgp = Path(f"{stem}-config.json")
    if not cfgp.exists():
        cfgp.write_text(json.dumps({"command": " ".join(sys.argv), "fold": k, "device": a.device,
                                    "ft_device": a.ft_device, "machine": a.machine_name, "torch": torch.__version__,
                                    "explain_per_subject": EXPLAIN_PER_SUBJECT, "window_hashes": ctx.hashes,
                                    "tau_k": ctx.tau, "ceiling_committed": ctx.ceiling, "ceiling50": refs.ceiling50,
                                    "calibration": CAL, "ft": FT, "plan": plan(k, a.stages, a.smoke)},
                                   indent=2, default=str), encoding="utf-8")
    base = {"fold": k, "machine_name": a.machine_name, "device": a.device, "tau_k": ctx.tau,
            "ceiling_committed": ctx.ceiling, "ceiling50": refs.ceiling50}
    zt = None
    ck_dir = (Path(a.out) / "ckpt") if a.out else CKPT
    for r in plan(k, a.stages, a.smoke):
        rid = run_id(r)
        if rid in done:
            continue
        ts = time.perf_counter()
        info, t_build, t_ft, extras, tag = {}, 0.0, 0.0, True, f"gs3-{rid}"
        if r["stage"] == "ref":
            o = {"f1": None, "lci": 1.0, "lci_indep": refs.ceiling50, "self_lci": refs.ceiling50, "occ_lci": 1.0,
                 "agree": 1.0, "shap_seconds": refs.seconds}
            o.update(dict(zip(("f1", "accuracy"), _f1acc(ctx, ctx.net0))))
            rows, net = [], ctx.net0
        else:
            if r["stage"] == "rescore":
                brid = f"{r['method']}-{int(round(r['level'] * 100))}pct-seed{r['seed']}-damp{r['damp']}"
                p = B_CKPT / f"fold{k}-Ansel-{brid}.pt"
                if not p.exists():
                    print(f"fold {k} {rid}: missing {p.name}, skipped", flush=True)
                    continue
                net, tag = load_ckpt(ctx, p), f"gs-p2-{brid}"
            elif r["stage"] == "rescore9a":
                lv = int(round(r["level"] * 100))
                found = sorted(G9A_CKPT.glob(f"har-grid-tinyhar-fold{k}-*-{r['method'][3:]}-{lv}pct-finetune.pt"))
                if not found:
                    print(f"fold {k} {rid}: no 9A checkpoint here, skipped", flush=True)
                    continue
                net = load_ckpt(ctx, found[0])
            elif r["stage"] == "refs2":
                p = CHECKPOINT_DIR / "har-noise-floor" / f"har-noise-floor-tinyhar-fold{k}-seed1.pt"
                if not p.exists():
                    print(f"fold {k} {rid}: no retrain checkpoint here, skipped", flush=True)
                    continue
                net = load_ckpt(ctx, p)
            elif r["stage"] == "nr":
                net = build_nr(r, ctx, a.smoke)
                t_build = time.perf_counter() - ts
                extras = r["method"] != "sparsegpt"
            else:   # ftcore / ftextra
                if zt is None:
                    zt = torch.from_numpy(predict_logits(ctx.net0, ctx.tr.X, ctx.dev)).to(a.ft_device)
                net, info = build_ft(r, ctx, a, zt)
                t_ft = info.get("seconds", time.perf_counter() - ts)
            if not head_unpruned(net):
                raise RuntimeError(f"{rid}: head pruned")
            o, rows = score(ctx, refs, net, tag, extras, ctx.nsamples, attr_dir / f"fold{k}-{rid}.npz")
            if r["stage"] in ("nr", "ftcore", "ftextra"):
                ck = ck_dir / f"fold{k}-{a.machine_name}-{rid}.pt"
                ck.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"run_id": rid, "fold": k, "state_dict": {n: t.cpu() for n, t in net.state_dict().items()},
                            "prune_masks": {n: m.cpu() for n, m in net.prune_masks.items()}}, ck)
        sz = size_record(net)
        _append(models_csv, [{**base, "run_id": rid, **r, **o, "build_seconds": t_build, "ft_seconds": t_ft,
                              "epochs_run": info.get("epochs_run"), "n_nonzero": sz["n_nonzero"],
                              "seconds_total": time.perf_counter() - ts}])
        if rows:
            _append(subj_csv, [{**base, "run_id": rid, **r, **x} for x in rows])
        print(f"fold {k} {rid}: F1 {o.get('f1') or float('nan'):.3f} acc {o.get('accuracy') or float('nan'):.3f} "
              f"LCI {o['lci']:.3f} indep {o.get('lci_indep', float('nan')):.3f} self {o.get('self_lci', float('nan')):.3f} "
              f"occ {o.get('occ_lci', float('nan')):.3f} agree {o.get('agree', float('nan')):.3f} "
              f"lci_agree {o.get('lci_agree', float('nan')):.3f} nnz {sz['n_nonzero']} "
              f"[{time.perf_counter() - ts:.0f}s, total {time.perf_counter() - t0:.0f}s]", flush=True)
    print(f"fold {k} stages {a.stages} done in {time.perf_counter() - t0:.0f}s", flush=True)


def _f1acc(ctx, net):
    from har.train import pooled_macro_f1
    pred = predict_proba(net, ctx.va.X, ctx.dev).argmax(1)
    return pooled_macro_f1(ctx.va.y, pred), float((pred == ctx.va.y).mean())


def _append(path: Path, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    if path.exists():
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    tmp = path.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False, lineterminator="\n")
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", type=int, nargs="+", default=[0])
    ap.add_argument("--stages", nargs="+", default=["ref"])
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="scoring / building device")
    ap.add_argument("--ft-device", choices=["cpu", "cuda"], default="cuda", help="fine-tuning device")
    ap.add_argument("--machine-name", default=R.machine_name())
    ap.add_argument("--worker", default=None, help="file-name suffix (one CSV per worker)")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default=None, help="output folder (smoke checks)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ft-selected", nargs="*", default=None, help="method:lr for the ftseeds stage")
    a = ap.parse_args()
    global FT_SELECTED
    FT_SELECTED = a.ft_selected
    if a.threads:
        torch.set_num_threads(a.threads)
    for k in a.folds:
        run_fold(k, a)


if __name__ == "__main__":
    main()
