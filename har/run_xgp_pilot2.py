"""XGP pilot round 2: fold 0, HARTH inner validation only. The last validation-only design iteration
before XGP is frozen (plan Section 2.5 (b), round 2). HAR70+ is never loaded: only the HARTH splits
of the window cache are read, and every window set (older-like copies included) goes through
require_harth.

Design (pre-registered by `--prepare`, before any GPU run; the rule is in har/xgp.py, XGP2):
    recipe     KD for every method = 0.7 KL(T = 3) T^2 to the original model's logits + 0.3
               class-weighted CE on the labels; E_ft = 10, Adam lr 1e-4, batch 256. KD is shared, it is
               not part of XGP.
    baselines  l1 (CE), l1_kd, snip_kd, park (CE + gamma_P L_AM, gamma_P = 1, lr 1e-4), park_kd
               (KD + gamma_P L_AM). park and park_kd: seed 0 only.
    epochs     l1_kd_e20 (L1 start, KD, E_ft = 20), seed 0 only.
    XGP        keep-score only (no L_cons, no gate), all + KD: xgp_kd (beta in {0.05, 0.1, 0.3}),
               xgp_worstclass_kd (I = max over classes of the layer-rescaled class-wise I), xgp_aug_kd
               (I on the 1,000 real + 1,000 older-like reference windows), xgp_worstclass_aug_kd (both);
               the variants use the beta of xgp_kd chosen at that level.
    seeds      0, 1, 2 = the fine-tuning batch-order seed (EG seed of Park's term = 1 + seed).
    windows    100 inner-val explain windows per subject (select_explain_windows, non-overlapping
               split) + their older-like copies (har/augment.py; s ~ U(0.8, 0.9), a ~ U(0.7, 0.9),
               seeded, saved by window ID). F1 uses all inner-val windows and their older-like copies.
    SHAP       GradientExplainer nsamples 200, SHAP seed 0, target = the original model's predicted
               class on each (normal or older-like) window, background = the fold's 100 HARTH windows.
    metrics    per window set (normal, older-like): LCI_val (subject mean) with its rho and Jaccard
               parts, worst-subject LCI, pooled 7-class macro-F1, retention = (LCI - tau_0) /
               (ceiling_0 - tau_0) with ceiling_0 = the inner-val SHAP-noise LCI mean of the noise
               floor; n_nonzero (asserted equal within a level), finetune_seconds. Every pruned start is
               also assessed before fine-tuning (scenario "no_finetune").

Usage (repo root, inside .venv):
    python -m har.run_xgp_pilot2 --prepare                                 # once, CPU, before any run
    python -m har.run_xgp_pilot2 --levels 0.7 --diagnostic --device cuda --machine-name nakhoul-3050
    python -m har.run_xgp_pilot2 --levels 0.9 --device cuda --machine-name uni-2080 --noise-floor-machine nakhoul-3050
    python -m har.run_xgp_pilot2 --analyze                                 # merges both machines' files
    add --smoke to any of them for the CPU check (tiny sizes, results/har-xgp-pilot-smoke/round2/)
    --resume continues an interrupted level run (finished run IDs are skipped).

Writes results/har-xgp-pilot/round2/ (never the round-1 files):
    har-xgp-pilot2-tinyhar-fold0-inner-val-windows-older-like-params.csv   --prepare
    har-xgp-pilot2-tinyhar-fold0-reference-windows-older-like-params.csv   --prepare
    har-xgp-pilot2-tinyhar-fold0-preregistration.json                      --prepare
    har-xgp-pilot2-tinyhar-fold0-<70|90>pct-inner-val-<machine>-{runs,per-subject-lci,finetune-history}.csv, -config.json
    har-xgp-pilot2-tinyhar-fold0-eg-noise-diagnostic-<machine>.csv         --diagnostic
    har-xgp-pilot2-tinyhar-fold0-{summary-mean-sd-over-seeds,decision-table}.csv, -decision.json   --analyze
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS; must precede CUDA init

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import runlog as R
from .attributions import shap_cached, subject_lci
from .augment import OLDER_LIKE, draw_older_like_params, older_like, older_like_windows
from .compression import (KD, finetune, l1_scores, predict_logits, prune_by_score, size_record, snip_scores,
                          teacher_segment_attributions, unit_norm)
from .data import Windows, find_windows, load_harth_windows, require_harth, window_ids
from .folds import load_folds, select_explain_windows
from .paths import (EXPLAIN_WINDOWS_CSV, MODEL_SELECTION_DIR, NOISE_FLOOR_DIR, NOISE_FLOOR_SMOKE_DIR, SHAP_CACHE_DIR,
                    XGP_PILOT2_DIR, XGP_PILOT2_SMOKE_DIR)
from .run_model_selection import subsample
from .run_noise_floor import fold_background
from .run_sanity_walking_aid import load_fold_model
from .run_xgp_pilot import read_tau
from .train import pooled_macro_f1, predict_proba
from .xgp import (PARK_TUNING, XGP, XGP2, alignment_importance, choose_beta, classwise_alignment_importance,
                  keep_scores, make_extra_loss, reference_windows, select_final_xgp, worst_class_importance)

FROZEN_CONFIG = MODEL_SELECTION_DIR / "har-frozen-model-config.json"
FOLD = 0
SMOKE = {"epochs": 1, "epochs_check": 2, "n_train": 600, "n_val": 200, "per_subject": 3, "nsamples": 20,
         "n_ref": 35, "m_score": 2, "m_park": 2, "seeds": (0, 1), "eg_diagnostic_m": (2, 4, 8)}
SETS = ("normal", "older_like")
RULE_TEXT = (
    "Final XGP = the eligible XGP variant with the highest mean LCI_val over {70%, 90%} x {normal, older-like} "
    "windows (mean over seeds). Eligible = mean pooled 7-class macro-F1 on the normal inner-val windows >= the best "
    "baseline's mean F1 - 1 pt, at both levels. Baselines: l1, l1_kd, snip_kd, park, park_kd. Ties within 0.005 -> "
    "the variant with fewer components (keep-score 1, + worst-class 1, + augmentation 1). If no variant is eligible, "
    "no fallback: report and stop. Variants: xgp_kd (beta in {0.05, 0.1, 0.3}), xgp_worstclass_kd, xgp_aug_kd, "
    "xgp_worstclass_aug_kd; the variants' beta at a level = the beta of xgp_kd picked at that level by the same "
    "eligibility and LCI criterion on that level alone (ties within 0.005 -> the smaller beta; none eligible -> the "
    "highest F1, flagged), and xgp_kd enters the rule with that beta.")
FAIRNESS_TEXT = (
    "Step 9: whatever recipe and E_ft the final XGP uses, L1 and SNIP-score pruning get the same in the grid; Park "
    "runs as published (CE + gamma_P L_AM) and as park_kd. E_ft = 20 for every method in Step 9 if l1_kd_e20 beats "
    "l1_kd (seed 0, same level) by more than 1 pt pooled macro-F1 (normal windows) at either level; otherwise 10.")
METRICS = [f"{m}_{s}" for s in SETS for m in ("lci", "rho", "jaccard", "worst_subject_lci", "retention", "f1",
                                                "accuracy", "n_skipped")] + ["lci_mean_sets", "finetune_seconds"]


def prefix(k: int = FOLD) -> str:
    return f"har-xgp-pilot2-tinyhar-fold{k}"


def file_sha(p: Path) -> str:
    """sha256 of the file with CRLF normalised to LF (pandas writes CRLF on Windows and git may convert
    line endings on checkout, so the raw bytes can differ between machines for the same content)."""
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def pct(level: float) -> int:
    return int(round(level * 100))


# ------------------------------------------------------------------------------ data (HARTH only)

def load_setup(smoke: bool):
    """HARTH training and inner-validation windows of the fold. HAR70+ is never read."""
    harth = load_harth_windows()
    counts = pd.Series(harth["harth_test"].subject).value_counts().to_dict()
    fold = load_folds(harth_test_counts={s: int(c) for s, c in counts.items()})[FOLD]
    tr = harth["harth_train"].for_subjects(fold.train)
    va_full = harth["harth_test"].for_subjects(fold.val)
    require_harth(tr, va_full, harth["harth_train"], harth["harth_test"])
    assert not (set(tr.subjects) | set(va_full.subjects)) & (set(fold.test_general) | set(fold.test_older))
    if smoke:
        tr = subsample(tr, SMOKE["n_train"], 0)
    return harth, fold, tr, va_full


def window_tables(tr: Windows, va_full: Windows, smoke: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The fixed, seeded window tables (deterministic; --prepare saves them, every run re-derives and
    checks them): (1) every inner-val window with its older-like (s, a) and the explain flag / pick order;
    (2) the reference windows of the keep-score with their older-like (s, a)."""
    require_harth(tr, va_full)
    per = SMOKE["per_subject"] if smoke else XGP2["explain_per_subject"]
    ids = window_ids(va_full, "harth_test")
    s, a = draw_older_like_params(len(va_full), OLDER_LIKE["explain_seed"])
    t1 = pd.DataFrame({"window_id": ids, "subject": va_full.subject, "label": va_full.y, "speed_s": s, "scale_a": a,
                       "explain": False, "pick_order": -1})
    ex = select_explain_windows(va_full, "harth_test", per_subject=per)
    for subj in va_full.subjects:                       # pick order within each subject
        sel = ex[va_full.subject[ex] == subj]
        t1.loc[sel, "explain"] = True
        t1.loc[sel, "pick_order"] = np.arange(len(sel))
    ref = reference_windows(tr, n=SMOKE["n_ref"] if smoke else XGP["n_ref"])
    rs, ra = draw_older_like_params(len(ref), OLDER_LIKE["reference_seed"])
    t2 = pd.DataFrame({"window_id": window_ids(tr, "harth_train")[ref], "subject": tr.subject[ref], "label": tr.y[ref],
                       "train_index": ref, "speed_s": rs, "scale_a": ra})
    return t1, t2


def table_paths(out: Path) -> dict[str, Path]:
    p = prefix()
    return {"inner_val": out / f"{p}-inner-val-windows-older-like-params.csv",
            "reference": out / f"{p}-reference-windows-older-like-params.csv",
            "prereg": out / f"{p}-preregistration.json"}


def check_tables(out: Path, t1: pd.DataFrame, t2: pd.DataFrame) -> dict:
    """The saved tables must exist (written by --prepare) and equal the re-derived ones."""
    P = table_paths(out)
    for p in P.values():
        if not p.exists():
            raise FileNotFoundError(f"{p} missing: run `python -m har.run_xgp_pilot2 --prepare` first")
    prereg = json.loads(P["prereg"].read_text(encoding="utf-8"))
    for key in ("inner_val", "reference"):
        if file_sha(P[key]) != prereg["sha256"][key]:
            raise RuntimeError(f"{P[key].name} changed after the pre-registration")
    for key, t in (("inner_val", t1), ("reference", t2)):
        saved = pd.read_csv(P[key], dtype={"subject": str})
        same = (list(saved.columns) == list(t.columns) and len(saved) == len(t)
                and (saved.window_id.to_numpy() == t.window_id.to_numpy()).all()
                and np.allclose(saved.speed_s, t.speed_s, rtol=0, atol=1e-12)
                and np.allclose(saved.scale_a, t.scale_a, rtol=0, atol=1e-12)
                and all((saved[c].to_numpy() == t[c].to_numpy()).all() for c in t.columns
                        if c not in ("speed_s", "scale_a")))
        if not same:
            raise RuntimeError(f"{P[key].name} differs from the windows re-derived on this machine")
    return prereg


def prepare(args) -> None:
    out = XGP_PILOT2_SMOKE_DIR if args.smoke else XGP_PILOT2_DIR
    P = table_paths(out)
    if not args.smoke and any(p.exists() for p in P.values()):
        raise FileExistsError(f"pre-registration already written in {out}; it is fixed (delete it by hand only if "
                              "no GPU run used it)")
    harth, fold, tr, va_full = load_setup(args.smoke)
    t1, t2 = window_tables(tr, va_full, args.smoke)
    R.write_csv(P["inner_val"], t1)
    R.write_csv(P["reference"], t2)
    old = pd.read_csv(EXPLAIN_WINDOWS_CSV, dtype={"subject": str})
    old = old[(old.split == "harth_test") & old.subject.isin(fold.val)]
    new = set(t1.window_id[t1.explain])
    first30 = set(t1.window_id[t1.explain & (t1.pick_order < 30)])
    prereg = {"what": "XGP pilot round 2, fold 0, inner validation only; pre-registered before any GPU run",
              "written_at": R.now(), **R.git_info(), "code_sha256": R.code_sha256(), "smoke": args.smoke,
              "selection_rule": RULE_TEXT, "fairness_rule_step9": FAIRNESS_TEXT, "xgp2": XGP2, "kd": KD,
              "older_like": OLDER_LIKE, "park_tuning_used": {"gamma_p": XGP2["park_gamma_p"], "lr": XGP2["lr"],
                                                             "m": PARK_TUNING["m"]},
              "xgp_importance": {k: XGP[k] for k in ("n_ref", "ref_seed", "m_score", "eg_seeds", "score_batch")},
              "fold": FOLD, "inner_val_subjects": list(fold.val), "n_inner_val_windows": len(t1),
              "n_explain_windows": int(t1.explain.sum()), "n_reference_windows": len(t2),
              "explain_windows_first30_equal_round1": bool(first30 == set(old.window_id)) if not args.smoke else None,
              "round1_explain_windows_contained": int(len(set(old.window_id) & new)),
              "sha256": {"inner_val": file_sha(P["inner_val"]), "reference": file_sha(P["reference"])}}
    R.write_json(P["prereg"], prereg)
    print(f"fold {FOLD}: {len(t1)} inner-val windows ({int(t1.explain.sum())} explain windows, "
          f"{t1[t1.explain].groupby('subject').size().to_dict()}), {len(t2)} reference windows; round-1 explain windows "
          f"contained: {prereg['round1_explain_windows_contained']} / {len(old)}")
    for p in P.values():
        print(f"saved {p}")


# ------------------------------------------------------------------------------ level runs

def read_ceiling(smoke: bool, machine: str | None) -> tuple[float, str]:
    d = NOISE_FLOOR_SMOKE_DIR if smoke else NOISE_FLOOR_DIR
    found = sorted(d.glob(f"har-noise-floor-tinyhar-fold{FOLD}-*-group-summary.csv"))
    if machine:
        found = [p for p in found if p.name == f"har-noise-floor-tinyhar-fold{FOLD}-{machine}-group-summary.csv"]
    if len(found) != 1:
        raise FileNotFoundError(f"need exactly one fold-{FOLD} noise-floor group summary in {d}; found {found}")
    g = pd.read_csv(found[0])
    row = g[(g.group == "inner_val") & (g.noise == "shap")]
    assert len(row) == 1
    return float(row["mean"].iloc[0]), str(found[0])


class Context:
    """Everything fixed for fold 0 (shared by both levels and the diagnostic)."""

    def __init__(self, args):
        self.args, S = args, (SMOKE if args.smoke else {})
        self.S, self.dev = S, args.device
        self.out = XGP_PILOT2_SMOKE_DIR if args.smoke else XGP_PILOT2_DIR
        self.nsamples = S.get("nsamples", XGP2["nsamples"])
        self.cache = SHAP_CACHE_DIR / ("smoke" if args.smoke else "")
        frozen = json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))
        assert frozen["model"] == "tinyhar", frozen["model"]
        harth, fold, tr, va_full = load_setup(args.smoke)
        t1, t2 = window_tables(tr, va_full, args.smoke)
        self.prereg = check_tables(self.out, t1, t2)
        self.fold, self.tr = fold, tr
        # F1 sets: every inner-val window (subsampled for --smoke) and its older-like copy
        va = subsample(va_full, SMOKE["n_val"], 1) if args.smoke else va_full
        par = t1.set_index("window_id").loc[window_ids(va, "harth_test")]
        self.va = va
        self.va_o = older_like_windows(va, par.speed_s.to_numpy(), par.scale_a.to_numpy())
        # explain windows (normal + older-like copies), subjects sorted, then pick order
        ex = t1[t1.explain].sort_values(["subject", "pick_order"])
        X_n = va_full.X[find_windows(va_full, "harth_test", ex.window_id.tolist())]
        X_o = older_like(X_n, ex.speed_s.to_numpy(), ex.scale_a.to_numpy())
        self.n_ex = len(ex)
        self.ids = ex.window_id.tolist() + [f"{i}|older_like" for i in ex.window_id]
        self.subj = np.concatenate([ex.subject.to_numpy(), ex.subject.to_numpy()])
        self.sl = {"normal": slice(0, self.n_ex), "older_like": slice(self.n_ex, 2 * self.n_ex)}
        # reference windows of the keep-score, real and older-like
        ref_idx = t2.train_index.to_numpy()
        self.ref = tr.select(ref_idx)
        self.ref_o = older_like_windows(self.ref, t2.speed_s.to_numpy(), t2.scale_a.to_numpy())
        require_harth(self.tr, self.va, self.va_o, self.ref, self.ref_o)
        assert set(self.subj) == set(fold.val)
        self.tau, self.tau_src = ((args.tau, "--tau") if args.tau is not None
                                  else read_tau(FOLD, args.smoke, args.noise_floor_machine))
        self.ceiling, self.ceiling_src = read_ceiling(args.smoke, args.noise_floor_machine)
        net0, self.ckpt = load_fold_model(FOLD, frozen["architecture"])
        self.net0 = net0.to(self.dev).eval()
        self.bg = fold_background(harth, FOLD).to(self.dev)
        X_all = np.concatenate([X_n, X_o])
        self.X_all = torch.from_numpy(X_all).to(self.dev)
        self.target = predict_proba(self.net0, X_all, self.dev).argmax(1)
        self.times = {}
        t0 = time.perf_counter()
        self.A_ref = self.shap(self.net0, "original")
        self.times["shap_original"] = time.perf_counter() - t0
        print(f"fold {FOLD}: train {len(tr)}, inner val {len(va)} ({list(fold.val)}), explain {self.n_ex} normal + "
              f"{self.n_ex} older-like, reference {len(ref_idx)} + {len(ref_idx)} older-like; tau_0 {self.tau:.4f} "
              f"({self.tau_src}), ceiling_0 {self.ceiling:.4f} ({self.ceiling_src}); original-model SHAP "
              f"{self.times['shap_original']:.1f} s")

    def shap(self, net, tag: str) -> np.ndarray:
        return shap_cached(net, self.bg, self.X_all, self.target, self.ids, XGP2["shap_seed"], self.nsamples,
                           self.cache, f"har-xgp-pilot2-fold{FOLD}-{tag}")

    def prepare_training(self) -> None:
        """Teacher logits (KD), Park's targets, SNIP scores and the four importances (all from the
        original model on HARTH training windows; computed once, shared by both levels)."""
        S, dev, net0, tr = self.S, self.dev, self.net0, self.tr
        T = self.times
        self.Xtr = torch.from_numpy(tr.X).to(dev)
        t0 = time.perf_counter()
        self.kd_logits = torch.from_numpy(predict_logits(net0, tr.X, dev)).to(dev)
        self.target_tr = self.kd_logits.argmax(1)
        self.park_ref = unit_norm(teacher_segment_attributions(net0, self.Xtr, self.target_tr, self.bg,
                                                               m=S.get("m_park", PARK_TUNING["m"]), seed=0).to(dev))
        T["teacher_logits_and_park_targets"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.snip = snip_scores(net0, tr)
        T["snip_scores"] = time.perf_counter() - t0
        m = S.get("m_score", XGP["m_score"])
        X_r = torch.from_numpy(self.ref.X).to(dev)
        X_ra = torch.from_numpy(np.concatenate([self.ref.X, self.ref_o.X])).to(dev)
        t_r = torch.from_numpy(predict_proba(net0, self.ref.X, dev).argmax(1)).to(dev)
        t_ra = torch.from_numpy(predict_proba(net0, X_ra.cpu().numpy(), dev).argmax(1)).to(dev)
        y_ra = np.concatenate([self.ref.y, self.ref_o.y])
        kw = {"m": m, "eg_seeds": XGP["eg_seeds"]}
        self.imp = {}
        for name, fn in (("plain", lambda: alignment_importance(net0, X_r, t_r, self.bg, **kw)),
                         ("worstclass", lambda: worst_class_importance(
                             classwise_alignment_importance(net0, X_r, t_r, self.ref.y, self.bg, **kw))),
                         ("aug", lambda: alignment_importance(net0, X_ra, t_ra, self.bg, **kw)),
                         ("worstclass_aug", lambda: worst_class_importance(
                             classwise_alignment_importance(net0, X_ra, t_ra, y_ra, self.bg, **kw)))):
            t0 = time.perf_counter()
            self.imp[name] = fn()
            T[f"importance_{name}"] = time.perf_counter() - t0
        print("training-side references: " + ", ".join(f"{k} {v:.1f} s" for k, v in T.items()))

    def assess(self, net, tag: str, f1_only: bool = False) -> tuple[dict, list[dict]]:
        out, rows = {}, []
        for name, w in (("normal", self.va), ("older_like", self.va_o)):
            pred = predict_proba(net, w.X, self.dev).argmax(1)
            out[f"f1_{name}"] = pooled_macro_f1(w.y, pred)
            out[f"accuracy_{name}"] = float((pred == w.y).mean())
        if f1_only:
            return out, rows
        t = time.perf_counter()
        A = self.shap(net, tag)
        for name in SETS:
            sl = self.sl[name]
            sub = subject_lci(self.A_ref[sl], A[sl], self.subj[sl])
            rows += [{"window_set": name, **r} for r in sub]
            lci = float(np.mean([r["lci"] for r in sub]))
            out.update({f"lci_{name}": lci, f"rho_{name}": float(np.mean([r["shap_corr"] for r in sub])),
                        f"jaccard_{name}": float(np.mean([r["jaccard"] for r in sub])),
                        f"worst_subject_lci_{name}": float(min(r["lci"] for r in sub)),
                        f"retention_{name}": (lci - self.tau) / (self.ceiling - self.tau),
                        f"n_skipped_{name}": int(sum(r["n_skipped"] for r in sub))})
        out["lci_mean_sets"] = float(np.mean([out[f"lci_{s}"] for s in SETS]))
        out["shap_seconds"] = time.perf_counter() - t
        return out, rows


def run_level(ctx: Context, level: float) -> None:
    args, S, dev = ctx.args, ctx.S, ctx.dev
    base = f"{prefix()}-{pct(level)}pct-inner-val-{args.machine_name}"
    paths = {n: ctx.out / f"{base}-{n}.csv" for n in ("runs", "per-subject-lci", "finetune-history")}
    cfg_path = ctx.out / f"{base}-config.json"
    if paths["runs"].exists() and not (args.resume or args.smoke):
        raise FileExistsError(f"{paths['runs']} exists; pass --resume to continue it (results are never overwritten)")
    keep = args.resume and paths["runs"].exists()
    runs = pd.read_csv(paths["runs"]).to_dict("records") if keep else []
    subj_rows = pd.read_csv(paths["per-subject-lci"], dtype={"subject": str}).to_dict("records") if keep else []
    hist_rows = pd.read_csv(paths["finetune-history"]).to_dict("records") if keep else []
    done = {r["run_id"] for r in runs}
    epochs = S.get("epochs", XGP2["epochs"])
    seeds = tuple(args.seeds) if args.seeds else S.get("seeds", XGP2["seeds"])
    t_level = time.perf_counter()
    print(f"\n######## level {level:.0%} ({'resuming, ' + str(len(done)) + ' runs done' if keep else 'new'}); seeds "
          f"{seeds}; E_ft {epochs}")

    def save():
        R.write_csv(paths["runs"], pd.DataFrame(runs))
        R.write_csv(paths["per-subject-lci"], pd.DataFrame(subj_rows))
        R.write_csv(paths["finetune-history"], pd.DataFrame(hist_rows))

    common = {"level": level, "fold": FOLD, "tau_0": ctx.tau, "ceiling_0": ctx.ceiling,
              "machine_name": args.machine_name, "device": dev}
    starts, nnz = {}, {}

    def start(name: str, scores: dict, **meta) -> None:
        net = prune_by_score(ctx.net0, scores, level)
        starts[name] = net
        rec = size_record(net)
        nnz[name] = rec["n_nonzero"]
        assert len(set(nnz.values())) == 1, f"n_nonzero differs across methods at {level:.0%}: {nnz}"
        rid = f"start_{name}"
        if rid in done:
            return
        a, rows = ctx.assess(net, f"{pct(level)}pct-{rid}")
        subj_rows.extend({"run_id": rid, "scenario": "no_finetune", **r} for r in rows)
        runs.append({"run_id": rid, "method": f"start_{name}", "scenario": "no_finetune", "start": name, "seed": None,
                     **meta, **a, **{k: v for k, v in rec.items() if k != "per_layer_sparsity"}, **common})
        save()
        print(f"  start {name:<24} no fine-tuning: LCI normal {a['lci_normal']:.4f}, older-like {a['lci_older_like']:.4f}"
              f", F1 normal {a['f1_normal']:.4f}")

    def run(method: str, start_name: str, seed: int, *, kd: bool, park: bool = False, ep: int | None = None, **meta):
        rid = f"{method}_seed{seed}"
        if rid in done:
            return
        ep = ep or epochs
        print(f"\n=== {rid}: start {start_name}, {'KD' if kd else 'CE'}" + (" + gamma_P L_AM" if park else "")
              + f", E_ft {ep}, lr {XGP2['lr']:g}")
        net = copy.deepcopy(starts[start_name])
        extra = (make_extra_loss("park", X=ctx.Xtr, target=ctx.target_tr, reference=ctx.park_ref, baselines=ctx.bg,
                                 gamma=XGP2["park_gamma_p"], m=S.get("m_park", PARK_TUNING["m"]), seed=1 + seed)
                 if park else None)
        info = finetune(net, ctx.tr, ctx.va, epochs=ep, lr=XGP2["lr"], seed=seed, device=dev, extra_loss=extra,
                        kd_teacher_logits=ctx.kd_logits if kd else None)
        hist_rows.extend({"run_id": rid, **h} for h in info["history"])
        rec = size_record(net)
        assert rec["n_nonzero"] == nnz[start_name], (rec["n_nonzero"], nnz[start_name])
        params = dict(net.named_parameters())
        assert all(torch.all(params[n][~mk.to(dev)] == 0) for n, mk in net.prune_masks.items())
        a, rows = ctx.assess(net, f"{pct(level)}pct-{rid}")
        subj_rows.extend({"run_id": rid, "scenario": "finetune", **r} for r in rows)
        pre = next(r for r in runs if r["run_id"] == f"start_{start_name}")
        runs.append({"run_id": rid, "method": method, "scenario": "finetune", "start": start_name, "seed": seed,
                     "recipe": "kd" if kd else "ce", "park_term": park, "epochs": ep, "lr": XGP2["lr"], **meta, **a,
                     "pre_ft_lci_normal": pre["lci_normal"], "pre_ft_lci_older_like": pre["lci_older_like"],
                     "pre_ft_f1_normal": pre["f1_normal"], "finetune_seconds": info["seconds"],
                     **{k: v for k, v in rec.items() if k != "per_layer_sparsity"}, **common})
        save()
        print(f"    -> LCI normal {a['lci_normal']:.4f}, older-like {a['lci_older_like']:.4f}; F1 normal "
              f"{a['f1_normal']:.4f}, older-like {a['f1_older_like']:.4f}; {info['seconds']:.1f} s")

    if "original" not in done:
        a, _ = ctx.assess(ctx.net0, "original", f1_only=True)
        runs.append({"run_id": "original", "method": "original", "scenario": "original", **a, "lci_normal": 1.0,
                     "lci_older_like": 1.0, **{k: v for k, v in size_record(ctx.net0).items() if k != "per_layer_sparsity"},
                     **common})
        save()

    # ---- pruned starts: L1, SNIP, the plain keep-score per beta
    start("l1", l1_scores(ctx.net0), score="l1")
    start("snip", ctx.snip, score="snip")
    for b in XGP2["betas"]:
        start(f"plain_beta{b:g}", keep_scores(ctx.net0, ctx.imp["plain"], b), score="keep_plain", beta=b)

    # ---- baselines and the epochs check
    for s in seeds:
        run("l1", "l1", s, kd=False)
        run("l1_kd", "l1", s, kd=True)
        run("snip_kd", "snip", s, kd=True)
    run("park", "l1", 0, kd=False, park=True, gamma_p=XGP2["park_gamma_p"])
    run("park_kd", "l1", 0, kd=True, park=True, gamma_p=XGP2["park_gamma_p"])
    run("l1_kd_e20", "l1", 0, kd=True, ep=S.get("epochs_check", XGP2["epochs_check"]))

    # ---- xgp_kd over beta, then beta* at this level (pre-registered), then the variants with beta*
    for b in XGP2["betas"]:
        for s in seeds:
            run(f"xgp_kd_beta{b:g}", f"plain_beta{b:g}", s, kd=True, variant="xgp_kd", beta=b)
    df = pd.DataFrame(runs)
    ft = df[df.scenario == "finetune"]
    best_f1 = max(ft[ft.method == m].f1_normal.mean() for m in XGP2["baselines"])
    by_beta = {b: {"lci_mean_sets": ft[ft.method == f"xgp_kd_beta{b:g}"].lci_mean_sets.mean(),
                   "f1_normal": ft[ft.method == f"xgp_kd_beta{b:g}"].f1_normal.mean()} for b in XGP2["betas"]}
    beta_star, beta_fb = choose_beta(by_beta, best_f1)
    print(f"\nlevel {level:.0%}: best baseline mean F1 {best_f1:.4f}; xgp_kd by beta "
          + ", ".join(f"{b:g}: LCI {v['lci_mean_sets']:.4f} F1 {v['f1_normal']:.4f}" for b, v in by_beta.items())
          + f" -> beta* = {beta_star:g}" + ("  (FLAGGED: no beta within 1 pt; highest F1)" if beta_fb else ""))
    for v, (imp_name, _) in XGP2["variants"].items():
        if v == "xgp_kd":
            continue
        start(f"{imp_name}_beta{beta_star:g}", keep_scores(ctx.net0, ctx.imp[imp_name], beta_star),
              score=f"keep_{imp_name}", beta=beta_star)
        for s in seeds:
            run(v, f"{imp_name}_beta{beta_star:g}", s, kd=True, variant=v, beta=beta_star)

    df = pd.DataFrame(runs)
    df["beta_star"], df["beta_star_flagged"], df["best_baseline_f1_level"] = beta_star, beta_fb, best_f1
    runs[:] = df.to_dict("records")
    save()
    sec = time.perf_counter() - t_level
    R.write_json(cfg_path, {"command_line": sys.argv, "machine_name": args.machine_name, "device": R.device_info(dev),
                            **R.git_info(), "code_sha256": R.code_sha256(), "finished_at": R.now(), "fold": FOLD,
                            "level": level, "seeds": seeds, "epochs": epochs, "nsamples": ctx.nsamples,
                            "smoke": args.smoke, "smoke_sizes": S or None, "selection_rule": RULE_TEXT,
                            "fairness_rule_step9": FAIRNESS_TEXT, "preregistration_written_at": ctx.prereg["written_at"],
                            "preregistration_sha256": ctx.prereg["sha256"], "tau_0": ctx.tau, "tau_source": ctx.tau_src,
                            "ceiling_0": ctx.ceiling, "ceiling_source": ctx.ceiling_src, "weights": str(ctx.ckpt),
                            "n_nonzero": nnz, "beta_star": beta_star, "beta_star_flagged": beta_fb,
                            "best_baseline_f1": best_f1, "xgp_kd_by_beta": by_beta, "seconds_fixed": ctx.times,
                            "seconds_level": sec})
    print(f"level {level:.0%} done in {sec:.1f} s")
    for p in (*paths.values(), cfg_path):
        print(f"saved {p}")


# ------------------------------------------------------------------------------ EG-noise diagnostic

def diagnostic(ctx: Context) -> None:
    """LCI between real SHAP and EG (m = 4, 16, 200) on the original model's normal inner-val explain
    windows, plus EG m = 4 vs EG m = 4 with another seed, and SHAP seed 1 vs SHAP seed 0 for scale."""
    S, sl = ctx.S, ctx.sl["normal"]
    X, t, subj = ctx.X_all[sl], torch.from_numpy(ctx.target[sl]).to(ctx.dev), ctx.subj[sl]
    A_shap = ctx.A_ref[sl]
    ms = S.get("eg_diagnostic_m", XGP2["eg_diagnostic_m"])
    s0, s1 = XGP2["eg_diagnostic_seeds"]
    rows = []

    def add(name, A_a, A_b, seconds, **meta):
        sub = subject_lci(A_a, A_b, subj)
        for r in sub:
            rows.append({"comparison": name, **meta, **r, "seconds": seconds})
        rows.append({"comparison": name, **meta, "subject": "mean", "lci": float(np.mean([r["lci"] for r in sub])),
                     "shap_corr": float(np.mean([r["shap_corr"] for r in sub])),
                     "jaccard": float(np.mean([r["jaccard"] for r in sub])),
                     "n_skipped": int(sum(r["n_skipped"] for r in sub)), "seconds": seconds})
        print(f"  {name:<34} LCI {rows[-1]['lci']:.4f} (rho {rows[-1]['shap_corr']:.4f}, J {rows[-1]['jaccard']:.4f}), "
              f"{seconds:.1f} s")

    print(f"\nEG-noise diagnostic, original model, {len(subj)} normal inner-val explain windows")
    eg = {}
    for m in ms:
        t0 = time.perf_counter()
        eg[(m, s0)] = teacher_segment_attributions(ctx.net0, X, t, ctx.bg, m=m, seed=s0,
                                                   batch_size=max(1, 2048 // m)).cpu().numpy()
        add(f"shap_seed0_vs_eg_m{m}_seed{s0}", A_shap, eg[(m, s0)], time.perf_counter() - t0, m=m, eg_seed=s0)
    m4 = ms[0]
    t0 = time.perf_counter()
    eg[(m4, s1)] = teacher_segment_attributions(ctx.net0, X, t, ctx.bg, m=m4, seed=s1,
                                                batch_size=max(1, 2048 // m4)).cpu().numpy()
    add(f"eg_m{m4}_seed{s0}_vs_eg_m{m4}_seed{s1}", eg[(m4, s0)], eg[(m4, s1)], time.perf_counter() - t0, m=m4,
        eg_seed=s1)
    t0 = time.perf_counter()
    A1 = shap_cached(ctx.net0, ctx.bg, X, ctx.target[sl], ctx.ids[sl], 1, ctx.nsamples, ctx.cache,
                     f"har-xgp-pilot2-fold{FOLD}-original-shapseed1")
    add("shap_seed0_vs_shap_seed1", A_shap, A1, time.perf_counter() - t0, m=None, eg_seed=None)
    p = ctx.out / f"{prefix()}-eg-noise-diagnostic-{ctx.args.machine_name}.csv"
    if p.exists() and not ctx.args.smoke:
        raise FileExistsError(f"{p} exists (results are never overwritten)")
    R.write_csv(p, pd.DataFrame(rows))
    print(f"saved {p}")


# ------------------------------------------------------------------------------ analysis

def analyze(args) -> int:
    out = XGP_PILOT2_SMOKE_DIR if args.smoke else XGP_PILOT2_DIR
    files = sorted(out.glob(f"{prefix()}-*pct-inner-val-*-runs.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()
    for lv in XGP2["levels"]:
        got = [f.name for f in files if f"-{pct(lv)}pct-" in f.name]
        if len(got) != 1:
            raise FileNotFoundError(f"need exactly one runs file for {lv:.0%} in {out}, found {got}")
    df = df[df.level.isin(XGP2["levels"])]
    for lv, g in df[df.scenario != "original"].groupby("level"):
        if g.n_nonzero.nunique() != 1:
            raise RuntimeError(f"n_nonzero differs within {lv:.0%}: {sorted(g.n_nonzero.unique())}")
    ft = df[df.scenario == "finetune"]
    metrics = [m for m in METRICS if m in df.columns]
    agg = []
    for (lv, meth), g in df[df.scenario.isin(["finetune", "no_finetune"])].groupby(["level", "method"]):
        row = {"level": lv, "method": meth, "scenario": g.scenario.iloc[0], "n_seeds": len(g),
               "seeds": ";".join(str(int(s)) for s in g.seed.dropna()), "start": g.start.iloc[0],
               "n_nonzero": int(g.n_nonzero.iloc[0]), "machine_name": g.machine_name.iloc[0]}
        for m in metrics:
            row[f"{m}_mean"] = float(g[m].mean())
            row[f"{m}_sd"] = float(g[m].std(ddof=1)) if len(g) > 1 else float("nan")
        agg.append(row)
    summ = pd.DataFrame(agg)
    stats, betas = {}, {}
    for lv in XGP2["levels"]:
        f = ft[ft.level == lv]
        best = max(f[f.method == m].f1_normal.mean() for m in XGP2["baselines"])
        by_beta = {b: {"lci_mean_sets": f[f.method == f"xgp_kd_beta{b:g}"].lci_mean_sets.mean(),
                       "f1_normal": f[f.method == f"xgp_kd_beta{b:g}"].f1_normal.mean()} for b in XGP2["betas"]}
        b_star, flag = choose_beta(by_beta, best)
        recorded = set(f.beta_star.dropna())
        if recorded != {b_star}:
            raise RuntimeError(f"{lv:.0%}: beta* recomputed {b_star} differs from the run's {recorded}")
        betas[lv] = {"beta_star": b_star, "flagged": flag, "by_beta": by_beta}
        for meth in (*XGP2["baselines"], *[v for v in XGP2["variants"] if v != "xgp_kd"]):
            g = f[f.method == meth]
            stats[(meth, lv)] = {"lci_mean_sets": g.lci_mean_sets.mean(), "f1_normal": g.f1_normal.mean()}
        g = f[f.method == f"xgp_kd_beta{b_star:g}"]
        stats[("xgp_kd", lv)] = {"lci_mean_sets": g.lci_mean_sets.mean(), "f1_normal": g.f1_normal.mean()}
    res = select_final_xgp(stats)
    table = pd.DataFrame(res["table"])
    for lv in XGP2["levels"]:
        table[f"beta_{pct(lv)}"] = betas[lv]["beta_star"]
    base_rows = []
    for meth in XGP2["baselines"]:
        r = {"variant": meth, "n_components": 0, "eligible": None, "selected": False}
        for lv in XGP2["levels"]:
            r[f"lci_mean_sets_{pct(lv)}"] = stats[(meth, lv)]["lci_mean_sets"]
            r[f"f1_normal_{pct(lv)}"] = stats[(meth, lv)]["f1_normal"]
        r["lci_mean_levels_sets"] = float(np.mean([stats[(meth, lv)]["lci_mean_sets"] for lv in XGP2["levels"]]))
        base_rows.append(r)
    table = pd.concat([table, pd.DataFrame(base_rows)], ignore_index=True)
    table.insert(1, "role", ["xgp_variant"] * len(res["table"]) + ["baseline"] * len(base_rows))
    # epochs check: l1_kd_e20 vs l1_kd, seed 0, same level
    ep = {}
    for lv in XGP2["levels"]:
        f = ft[(ft.level == lv) & (ft.seed == 0)]
        e20, e10 = float(f[f.method == "l1_kd_e20"].f1_normal.iloc[0]), float(f[f.method == "l1_kd"].f1_normal.iloc[0])
        ep[lv] = {"f1_e20": e20, "f1_e10": e10, "gain": e20 - e10}
    e_ft = XGP2["epochs_check"] if any(v["gain"] > XGP2["epochs_check_margin"] for v in ep.values()) else XGP2["epochs"]
    p = prefix()
    R.write_csv(out / f"{p}-summary-mean-sd-over-seeds.csv", summ)
    R.write_csv(out / f"{p}-decision-table.csv", table)
    decision = {"final_xgp": res["final"], "selection_rule": RULE_TEXT, "fairness_rule_step9": FAIRNESS_TEXT,
                "beta_by_level": {f"{pct(lv)}pct": betas[lv]["beta_star"] for lv in XGP2["levels"]},
                "beta_flagged_by_level": {f"{pct(lv)}pct": betas[lv]["flagged"] for lv in XGP2["levels"]},
                "best_baseline_f1": {f"{pct(lv)}pct": v for lv, v in res["best_baseline_f1"].items()},
                "epochs_check": {f"{pct(lv)}pct": v for lv, v in ep.items()}, "e_ft_step9": e_ft,
                "files": [f.name for f in files], "analyzed_at": R.now(), "smoke": args.smoke}
    R.write_json(out / f"{p}-decision.json", decision)
    show = ["variant", "role", "n_components", "lci_mean_sets_70", "f1_normal_70", "lci_mean_sets_90", "f1_normal_90",
            "lci_mean_levels_sets", "eligible", "selected"]
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.4f}".format):
        print(table[[c for c in show if c in table.columns]].to_string(index=False))
    print(f"\nbest baseline mean F1: " + ", ".join(f"{lv:.0%} {v:.4f}" for lv, v in res["best_baseline_f1"].items())
          + f"; beta*: " + ", ".join(f"{lv:.0%} {betas[lv]['beta_star']:g}" + (" (flagged)" if betas[lv]["flagged"] else "")
                                     for lv in XGP2["levels"]))
    print("epochs check (l1_kd_e20 - l1_kd, seed 0, F1 normal): "
          + ", ".join(f"{lv:.0%} {v['gain']:+.4f}" for lv, v in ep.items()) + f" -> E_ft = {e_ft} in Step 9")
    for n in ("summary-mean-sd-over-seeds.csv", "decision-table.csv", "decision.json"):
        print(f"saved {out / f'{p}-{n}'}")
    if res["final"] is None:
        print("\nNO ELIGIBLE XGP VARIANT: no variant is within 1 pt of the best baseline's mean F1 at both levels. "
              "Per the pre-registered rule there is no fallback; XGP is NOT frozen. Stop here and review.")
        return 2
    print(f"\nFINAL XGP (pre-registered rule): {res['final']}  (beta "
          + ", ".join(f"{lv:.0%} {betas[lv]['beta_star']:g}" for lv in XGP2["levels"]) + ")")
    return 0


# ------------------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true", help="write the window tables and the pre-registration")
    mode.add_argument("--levels", type=float, nargs="+", choices=XGP2["levels"], help="sparsity levels to run")
    mode.add_argument("--analyze", action="store_true", help="merge both machines' files and apply the rule")
    ap.add_argument("--diagnostic", action="store_true", help="also run the EG-noise diagnostic (with --levels)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--machine-name", default=R.machine_name())
    ap.add_argument("--tau", type=float, default=None, help="override tau_0 (default: the noise-floor tau file)")
    ap.add_argument("--noise-floor-machine", default=None, help="whose fold-0 noise-floor files give tau_0 and ceiling_0")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.prepare:
        return prepare(args)
    if args.analyze:
        sys.exit(analyze(args))
    out = XGP_PILOT2_SMOKE_DIR if args.smoke else XGP_PILOT2_DIR
    tag = "-".join(f"{pct(lv)}pct" for lv in args.levels)
    log = out / "logs" / f"{prefix()}-{tag}-{args.machine_name}-{R.stamp()}.log"
    with R.tee_to(log):
        dev = R.device_info(args.device)
        t_all = time.perf_counter()
        print(f"har.run_xgp_pilot2 | {' '.join(sys.argv)} | machine {args.machine_name} | device {args.device} "
              f"{dev['gpu_name']} | torch {dev['torch_version']} | started {R.now()}")
        ctx = Context(args)
        if args.diagnostic:
            diagnostic(ctx)
        ctx.prepare_training()
        for lv in args.levels:
            run_level(ctx, lv)
        print(f"\nwall clock {time.perf_counter() - t_all:.1f} s; log {log}")


if __name__ == "__main__":
    main()
