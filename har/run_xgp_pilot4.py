"""XGP pilot round 4 (final round, validation only; plan Section 2.5 (b), "Round 4 pre-registration", with
amendment 1). Folds 0 and 1, HARTH only (HAR70+ is never loaded). Round 3 stays reproducible: this file only
imports from har/run_xgp_pilot3.py and har/xgp3.py; outputs go to results/har-xgp-pilot/round4/.

No synthetic (older-like) data and no age information. XGP is group-agnostic: its groups are the calibration /
training windows' subjects (the dataset-agnostic default; har/xgp4.py takes any group label). Older vs general
and walking-aid vs not are evaluation-only, on the test grid in 9B. The one exception is the reference row
xgp_r3: round 3's frozen method exactly as it was (its own older-like calibration and augmentation); only its
scores use the round-4 metrics.

Disclosure: the round-4 design was informed by round 3's validation results, a read-only per-layer importance
diagnostic at theta0 (folds 0-1) and the Step 9A test-set analysis; all round-4 tuning and selection use HARTH
inner-validation windows only. The 9A-ext baseline rows run in 9B, after this decision.

    --prepare                 calibration (<= 10 training windows per (subject, class) cell) + KD-5% tables,
                              pre-registration JSON (needs round 3's decision.json and the tag xgp-round3-final)
    --mode unstructured       3 x 2 budget x selection design (uniform / LAMP / XGP budgets x L1 / XGP score),
                              l1 and xgp_r3 references, DRO controls; no fine-tuning, KD 100%, KD 5%; curve; PTQ
    --mode structured         conv-filter pruning: l1_filter, l1_filter_recon, taylor_recon, xgp_s, DRO controls
    --analyze --mode M        the pre-registered rules -> har-xgp-pilot4-tinyhar-{M}-decision.json
    --latency                 CPU latency of the dense and the physically sliced structured models

    validation (real inner-val windows only): F1_val on every inner-val window; LCI_val = mean over the fold's
    3 inner-val subjects of the per-subject LCI on round 3's explain windows; worst-subject LCI_val = the min.
    fine-tuning: baselines plain KD; XGP candidates KD + group DRO over the training subjects; every baseline
    also KD + the same DRO (*_kd_dro controls).

Usage (repo root, inside .venv):
    python -m har.run_xgp_pilot4 --prepare
    python -m har.run_xgp_pilot4 --mode unstructured --folds 0 --device cuda --machine-name uni-2080 --noise-floor-machine nakhoul-3050
    python -m har.run_xgp_pilot4 --mode structured --folds 1 --device cuda --machine-name uni-2080 --noise-floor-machine nakhoul-3050
    python -m har.run_xgp_pilot4 --analyze --mode unstructured
    python -m har.run_xgp_pilot4 --latency --machine-name Aya
    add --smoke for the CPU check (tiny sizes, results/har-xgp-pilot-smoke/round4/); --resume continues a run.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import runlog as R
from . import run_xgp_pilot3 as P3
from .attributions import shap_cached, subject_lci
from .compression import (finetune, head_unpruned, l1_scores, predict_logits, prunable_tensors, prune_by_score,
                          ptq_symmetric, size_record)
from .data import Windows, find_windows, require_harth, window_ids
from .folds import select_explain_windows
from .paths import CHECKPOINT_DIR, REPO_ROOT, SHAP_CACHE_DIR, XGP_PILOT_DIR, XGP_PILOT_SMOKE_DIR
from .run_noise_floor import fold_background
from .run_sanity_walking_aid import load_fold_model
from .run_xgp_pilot import read_tau
from .train import class_weights, pooled_macro_f1, predict_proba
from .xgp3 import COMPONENTS, ROUND3, iterative_prune, layerwise_reconstruct
from .xgp4 import (ROUND4, S_BASELINES, S_CANDIDATES, S_CURVE, S_DRO_CONTROLS, S_METHODS, S_PTQ, U_BASELINES,
                   U_CANDIDATES, U_CELLS, U_CURVE, U_DRO_CONTROLS, U_PTQ, apply_structured, beats, bounded_budget,
                   calibration_pick, count_macs, cpu_latency, default_groups, filter_scores_from_weight_scores,
                   filter_scores_l1, filter_scores_taylor, group_index, iterative_prune_budget,
                   iterative_prune_structured, pick_final, slice_structured, structured_layers, uniform_keep_counts,
                   unreconstructed_tensors, worst_group_importance, worst_group_shares)

P = "har-xgp-pilot4-tinyhar"
TAG = "xgp-round3-final"
SMOKE4 = {"levels_u": (0.9,), "levels_u_kd5": (0.9,), "levels_s": (0.5,), "seeds": (0,), "K": 2, "epochs": 2,
          "curve_epochs": (1,), "curve_levels_u": (0.9,), "cal_per_cell": 2, "m_score": 2, "taylor_batches": 2,
          "ptq_levels_u": (0.9,), "ptq_levels_s": (0.5,), "ptq8_level_u": 0.9, "ptq8_level_s": 0.5,
          "latency": {"threads": (1,), "batches": (1, 8), "warmup": 1, "reps": 5}}
REGIMES = ("no_finetune", "kd100", "kd5")
KD = {"recipe": "kd", "r": None, "dro": False}
KD_DRO = {"recipe": "kd", "r": None, "dro": True}
METRICS = ("lci", "worst_subject_lci", "f1", "accuracy", "rho", "jaccard", "retention")
DISCLOSURE = ("The round-4 design was informed by round 3's validation results (folds 0-1), a read-only validation "
              "diagnostic of per-layer importance at theta0, and the Step 9A test-set analysis (as round 3). All "
              "round-4 tuning and selection use real HARTH inner-validation windows of folds 0 and 1 only; HAR70+, "
              "the folds' test subjects, synthetic older-like data and age information are never used by the method "
              "or the selection. The 9A-ext baseline-only settings run on the test grid in 9B, after this decision, "
              "and cannot change it.")
AMENDMENT = ("Amendment 1 (2026-11-23, before --prepare and before any round-4 run): all synthetic older-like data "
             "removed from round 4; XGP is group-agnostic (optional group label per window, default = subject): "
             "importance = mean of the k = 3 worst groups, fine-tuning = group DRO over the training subjects, every "
             "baseline gets the same DRO as a control (*_kd_dro); validation = LCI_val, worst-subject LCI_val and "
             "F1_val on real inner-val windows; older vs general and walking-aid vs not are evaluation-only in 9B. "
             "Reason: the older-like proxy added nothing in round 3, and the method must not depend on one "
             "dataset's groups.")
RULES = (
    "Per mode and regime (no_finetune, kd100, kd5), levels = the mode's levels in that regime, means over folds 0-1 x "
    "seeds 0-2 (6 paired runs; deterministic starts count once per seed). Metrics on real inner-val windows only. "
    "1 Baselines: unstructured l1, l1_recon, lamp_recon; structured l1_filter, l1_filter_recon, taylor_recon; in the "
    "fine-tuned regimes also their DRO controls (*_kd_dro). B_F1 = the highest mean F1_val among them per level. "
    "2 Eligible = mean F1_val >= B_F1 - 1 pt at every level. "
    "3 Final: unstructured candidates xgp, l1_xb_recon, xgp_lamp, xgp_xb (tie order), structured xgp_s; the eligible "
    "candidate with the highest mean LCI_val over levels, ties within 0.005 -> earlier in order; none -> no final. "
    "4 A beats B iff (a) mean LCI_val gain (over levels and runs) >= 0.02, (b) A higher in >= 5 of 6 paired (fold, "
    "seed) runs (each run averaged over levels), (c) at every level mean F1_val of A >= mean F1_val of B - 1 pt, "
    "(d) mean worst-subject LCI_val of A (over levels) >= B's - 0.01. "
    "5 Replacement (unstructured, per regime): round 4's final replaces round 3's (xgp_r3, scored with these metrics) "
    "only if it beats it; otherwise round 3 is final there; with no round-3 final there, round 4's final is adopted "
    "if it exists. 6 Claim: 'beats the baselines' only if the adopted final beats the baseline (DRO controls "
    "included) with the highest mean LCI_val. 7 Structured: xgp_s frozen if eligible; enters 9B only if its pilot "
    "finished and was decided by 27 Nov morning. 8 Seed SD per method reported; a regime with missing runs has no "
    "round-4 decision (round 3 stays final).")
R3_FT_START = {"xgp_ft": ("xgp", "kdaug", True), "xgp_no_score_ft": ("xgp_no_score", "kdaug", True),
               "xgp_no_group_ft": ("xgp_no_group", "kdaug", True), "xgp_no_iter_ft": ("xgp_no_iter", "kdaug", True),
               "xgp_no_recon_ft": ("xgp_no_recon", "kdaug", True), "xgp_ft_no_aug": ("xgp", "kd", False),
               "xgp_ft_no_dro": ("xgp", "kdaug", False)}


def cfg(key: str, smoke: bool):
    return SMOKE4[key] if smoke and key in SMOKE4 else ROUND4[key]


def pct(level: float) -> int:
    return int(round(level * 100))


def out_dir(smoke: bool) -> Path:
    return (XGP_PILOT_SMOKE_DIR if smoke else XGP_PILOT_DIR) / "round4"


def ckpt_dir(smoke: bool) -> Path:
    return CHECKPOINT_DIR / "har-xgp-pilot4" / ("smoke" if smoke else "")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# ------------------------------------------------------------------------------ round 3 (read only)

def r3_decision(smoke: bool) -> tuple[dict, Path]:
    p = P3.out_dir(smoke) / f"{P3.P}-decision.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: round 3's decision must exist first "
                                "(`python -m har.run_xgp_pilot3 --analyze --phase ablation`)")
    return json.loads(p.read_text(encoding="utf-8")), p


def r3_config(dec: dict, scen: str, level: float) -> dict | None:
    """Round 3's configuration in `scen` at the round-3 level nearest to `level` (ties -> the higher)."""
    d = dec["decision"].get(scen) or {}
    by = d.get("config_by_level") or {}
    if not d.get("final") or not by:
        return None
    lv = sorted(by, key=lambda s: (abs(int(s.replace("pct", "")) - pct(level)), -int(s.replace("pct", ""))))[0]
    return {**by[lv], "round3_level": lv, "final": d["final"]}


def r3_ft_recipe(dec: dict, level: float) -> dict | None:
    """xgp_r3 only: round 3's fine-tuned final at the nearest round-3 level -> its start and recipe."""
    c = r3_config(dec, "finetune", level)
    if c is None:
        return None
    start, recipe, dro = R3_FT_START[c["final"]]
    assert ("aug" in COMPONENTS["finetune"][c["final"]]) == (recipe == "kdaug")
    return {"start": start, "recipe": recipe, "r": c["r"] if recipe == "kdaug" else None, "dro": dro,
            "source": f"round 3 finetune final {c['final']} at {c['round3_level']}"}


def tag_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", f"{TAG}^{{commit}}"], cwd=REPO_ROOT, capture_output=True, text=True,
                             timeout=20)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


# ------------------------------------------------------------------------------ prepare

def cal_table_path(out: Path, k: int) -> Path:
    return out / f"{P}-fold{k}-calibration-train-windows-by-subject-class.csv"


def kd5_table_path(out: Path, k: int) -> Path:
    return out / f"{P}-fold{k}-kd5pct-train-window-ids.csv"


def prereg_path(out: Path) -> Path:
    return out / f"{P}-preregistration.json"


def calibration_table(tr: Windows, smoke: bool, groups=None) -> pd.DataFrame:
    """Up to cal_per_cell training windows per (group, class) cell, seed 46; group = subject by default."""
    require_harth(tr)
    g = default_groups(tr.subject, groups)
    pick = calibration_pick(tr.y, g, cfg("cal_per_cell", smoke), ROUND4["cal_seed"])
    t = pd.DataFrame({"window_id": window_ids(tr, "harth_train")[pick], "subject": tr.subject[pick],
                      "group": g[pick], "label": tr.y[pick]})
    t["pick_order"] = t.groupby(["group", "label"]).cumcount()
    return t


def kd5_table(tr: Windows) -> pd.DataFrame:
    require_harth(tr)
    rng = np.random.default_rng(ROUND4["kd5_seed"])
    idx = []
    for s in sorted(set(tr.subject.tolist())):
        for c in range(7):
            cell = np.flatnonzero((tr.subject == s) & (tr.y == c))
            if len(cell):
                idx += sorted(rng.permutation(cell)[:max(1, int(round(ROUND4["kd5_fraction"] * len(cell))))].tolist())
    idx = np.array(sorted(idx))
    return pd.DataFrame({"window_id": window_ids(tr, "harth_train")[idx], "subject": tr.subject[idx], "label": tr.y[idx]})


def prepare(args) -> None:
    S, out = args.smoke, out_dir(args.smoke)
    pp = prereg_path(out)
    if pp.exists() and not S:
        raise FileExistsError(f"{pp} exists; the round-4 pre-registration is fixed")
    dec, dec_p = r3_decision(S)
    tc = tag_commit()
    if tc is None and not S:
        raise RuntimeError(f"git tag {TAG} not found: tag round 3's final commit first (we tag it ourselves)")
    folds = tuple(args.folds) if args.folds else ROUND4["folds"]
    shas, info = {}, {}
    for k in folds:
        _, fold, tr, _ = P3.load_setup(k, S)
        c, d = calibration_table(tr, S), kd5_table(tr)
        R.write_csv(cal_table_path(out, k), c)
        R.write_csv(kd5_table_path(out, k), d)
        shas[str(k)] = {"calibration": sha(cal_table_path(out, k)), "kd5": sha(kd5_table_path(out, k))}
        per_g = c.groupby("group").size()
        info[str(k)] = {"n_train": len(tr), "n_calibration": len(c), "n_groups": int(per_g.size),
                        "calibration_per_group_min_max": [int(per_g.min()), int(per_g.max())],
                        "calibration_per_class": {int(a): int(b) for a, b in c.groupby("label").size().items()},
                        "n_kd5": len(d), "kd5_fraction_actual": len(d) / len(tr)}
        print(f"fold {k}: {info[str(k)]}")
    R.write_json(pp, {"what": "XGP pilot round 4 (final round, validation only); pre-registered before any run",
                      "written_at": R.now(), **R.git_info(), "code_sha256": R.code_sha256(), "smoke": S,
                      "disclosure": DISCLOSURE, "amendment": AMENDMENT, "rules": RULES, "round4": ROUND4,
                      "groups": "subject (default of har/xgp4.default_groups); no age / HAR70+ / synthetic data",
                      "u_cells": U_CELLS, "s_methods": S_METHODS, "u_dro_controls": U_DRO_CONTROLS,
                      "s_dro_controls": S_DRO_CONTROLS,
                      "recipes": {"baselines": KD, "candidates_and_dro_controls": {**KD_DRO, "dro_groups": "subject",
                                                                                   "dro_eta": ROUND4["dro_eta"]}},
                      "round3_decision_file": str(dec_p.relative_to(REPO_ROOT)),
                      "round3_decision_sha256": sha(dec_p), "round3_tag": TAG, "round3_tag_commit": tc,
                      "xgp_r3_recipe_by_level": {f"{pct(lv)}pct": r3_ft_recipe(dec, lv) for lv in cfg("levels_u", S)},
                      "folds": list(folds), "windows": info, "sha256_tables": shas,
                      "plan": "plan/HAR_IMPLEMENTATION.md, Section 2.5 (b), 'Round 4 pre-registration'"})
    print(f"saved {pp}")


# ------------------------------------------------------------------------------ context

class Context4:
    """Round 4's fold context. Real inner-val windows only: F1 on every inner-val window, LCI on round 3's
    explain windows (the normal windows only). The round-4 calibration set (training windows) with its groups,
    the training subjects as DRO groups, the KD-5% subset, the Taylor batches and round 3's decision. Round 3's
    own context is built lazily, for the xgp_r3 reference only."""

    def __init__(self, args, k: int):
        self.args, self.k, self.smoke, self.dev = args, k, args.smoke, args.device
        S, out = self.smoke, out_dir(self.smoke)
        self.out4 = out
        pp = prereg_path(out)
        if not pp.exists():
            raise FileNotFoundError(f"{pp} missing: run `python -m har.run_xgp_pilot4 --prepare` first")
        self.prereg4 = json.loads(pp.read_text(encoding="utf-8"))
        for kind, q in (("calibration", cal_table_path(out, k)), ("kd5", kd5_table_path(out, k))):
            if sha(q) != self.prereg4["sha256_tables"][str(k)][kind]:
                raise RuntimeError(f"{q.name} changed after the round-4 pre-registration")
        self.dec3, dec_p = r3_decision(S)
        if sha(dec_p) != self.prereg4["round3_decision_sha256"]:
            raise RuntimeError("round 3's decision.json changed after the round-4 pre-registration")
        if not S:
            sel = P3.read_selection(P3.out_dir(S))["90pct"]
            for scen in ("no_finetune", "finetune"):
                if sel[scen] is not None and abs(sel[scen]["beta"] - ROUND4["beta_u"]) > 1e-12:
                    raise RuntimeError(f"round 3's {scen} beta at 90% is {sel[scen]['beta']}, pre-registered "
                                       f"beta_u {ROUND4['beta_u']}")
        self.nsamples = P3.cfg("nsamples", S)
        self.cache = SHAP_CACHE_DIR / ("smoke" if S else "")
        frozen = json.loads(P3.FROZEN_CONFIG.read_text(encoding="utf-8"))
        assert frozen["model"] == "tinyhar", frozen["model"]
        harth, fold, tr, va_full = P3.load_setup(k, S)
        self.fold, self.tr = fold, tr
        # F1 windows: every inner-val window (round 4 calibrates on training windows)
        f1_idx = np.arange(len(va_full))
        if S:
            f1_idx = np.sort(np.random.default_rng(1).permutation(f1_idx)[:P3.SMOKE["n_val"]])
        self.va = va_full.select(f1_idx)
        # explain windows: round 3's (select_explain_windows per subject), subjects sorted, then pick order
        ex = select_explain_windows(va_full, "harth_test", per_subject=P3.cfg("explain_per_subject", S))
        ex = np.concatenate([ex[va_full.subject[ex] == sj] for sj in sorted(set(va_full.subject[ex].tolist()))])
        self.subj = va_full.subject[ex]
        self.ids = window_ids(va_full, "harth_test")[ex].tolist()
        require_harth(tr, self.va, va_full)
        assert set(self.subj.tolist()) == set(fold.val), (set(self.subj.tolist()), fold.val)
        net0, self.ckpt = load_fold_model(k, frozen["architecture"])
        self.net0 = net0.to(self.dev).eval()
        self.bg = fold_background(harth, k).to(self.dev)
        try:
            self.tau, self.tau_src = read_tau(k, S, args.noise_floor_machine)
            self.ceiling, self.ceiling_src = P3.read_ceiling(k, S, args.noise_floor_machine)
        except FileNotFoundError:
            if not S:
                raise
            self.tau, self.tau_src, self.ceiling, self.ceiling_src = 0.6, "smoke default", 0.9, "smoke default"
        self.X_ex = torch.from_numpy(va_full.X[ex]).to(self.dev)
        self.target = predict_proba(self.net0, va_full.X[ex], self.dev).argmax(1)
        t0 = time.perf_counter()
        self.A_ref = self.shap(self.net0, "original")
        t_shap = time.perf_counter() - t0
        self.kd_logits = torch.from_numpy(predict_logits(self.net0, tr.X, self.dev)).to(self.dev)
        # calibration (training windows) with its group labels
        cal = pd.read_csv(cal_table_path(out, k), dtype={"subject": str, "group": str})
        cw = tr.select(find_windows(tr, "harth_train", cal.window_id.tolist()))
        if not np.array_equal(cw.subject.astype(str), cal.subject.to_numpy()):
            raise RuntimeError("calibration windows do not match their table")
        self.cal_X = torch.from_numpy(cw.X).to(self.dev)
        self.cal_t = torch.from_numpy(predict_proba(self.net0, cw.X, self.dev).argmax(1)).to(self.dev)
        self.cal_y = cw.y.copy()
        self.cal_g = default_groups(cw.subject, cal.group.to_numpy())
        # DRO groups: the training subjects (the same group definition)
        self.tr_g, self.group_labels = group_index(default_groups(tr.subject))
        d = pd.read_csv(kd5_table_path(out, k), dtype={"subject": str})
        self.idx5 = find_windows(tr, "harth_train", d.window_id.tolist())
        self.tr5 = tr.select(self.idx5)
        self.z5 = self.kd_logits[torch.from_numpy(self.idx5).to(self.kd_logits.device)]
        self.tr5_g, _ = group_index(default_groups(self.tr5.subject), self.group_labels)
        self.unrecon = unreconstructed_tensors(self.net0, self.cal_X[:4])
        rng = np.random.default_rng(ROUND4["taylor_seed"])
        nb, bs = cfg("taylor_batches", S), ROUND4["taylor_batch_size"]
        idx = rng.permutation(len(tr))[:nb * bs]
        X, y = torch.from_numpy(tr.X), torch.from_numpy(tr.y)
        self.taylor_batches = [(X[idx[i:i + bs]], y[idx[i:i + bs]]) for i in range(0, len(idx), bs)]
        self.taylor_loss = nn.CrossEntropyLoss(weight=class_weights(tr.y).to(self.dev))
        self.shares_cache = {}
        self._r3 = None
        print(f"fold {k} (round 4): train {len(tr)} ({len(self.group_labels)} subjects = DRO groups), F1 windows "
              f"{len(self.va)}, explain {len(self.ids)} ({list(fold.val)}), calibration {len(cw)} training windows in "
              f"{len(set(self.cal_g.tolist()))} groups, KD-5% subset {len(self.tr5)}; tau_k {self.tau:.4f}, ceiling "
              f"{self.ceiling:.4f}; original SHAP {t_shap:.1f} s; not reconstructed (budget-protected): {self.unrecon}")

    def shap(self, net, tag: str) -> np.ndarray:
        return shap_cached(net, self.bg, self.X_ex, self.target, self.ids, ROUND4["shap_seed"], self.nsamples,
                           self.cache, f"{P}-fold{self.k}-{tag}")

    def assess(self, net, tag: str) -> tuple[dict, list[dict]]:
        pred = predict_proba(net, self.va.X, self.dev).argmax(1)
        out = {"f1": pooled_macro_f1(self.va.y, pred), "accuracy": float((pred == self.va.y).mean())}
        t = time.perf_counter()
        A = self.shap(net, tag)
        sub = subject_lci(self.A_ref, A, self.subj)
        rows = [{"window_set": "inner_val", **r} for r in sub]
        lci = float(np.mean([r["lci"] for r in sub]))
        out.update({"lci": lci, "worst_subject_lci": float(min(r["lci"] for r in sub)),
                    "worst_subject": min(sub, key=lambda r: r["lci"])["subject"],
                    "rho": float(np.mean([r["shap_corr"] for r in sub])),
                    "jaccard": float(np.mean([r["jaccard"] for r in sub])),
                    "retention": (lci - self.tau) / (self.ceiling - self.tau),
                    "n_skipped": int(sum(r["n_skipped"] for r in sub)), "shap_seconds": time.perf_counter() - t})
        return out, rows

    def importance4(self, seed: int):
        m = cfg("m_score", self.smoke)
        return lambda net: worst_group_importance(net, self.net0, self.cal_X, self.cal_t, self.cal_y, self.cal_g,
                                                  self.bg, k=ROUND4["top_k"], m=m, eg_seeds=(2 * seed, 2 * seed + 1),
                                                  batch_size=ROUND4["score_batch"])

    def recon4(self):
        return lambda net: layerwise_reconstruct(net, self.net0, self.cal_X, damp=ROUND4["damp"])

    def xgp_shares(self, seed: int) -> dict[str, float]:
        if seed not in self.shares_cache:
            self.shares_cache[seed] = worst_group_shares(self.net0, self.net0, self.cal_X, self.cal_t, self.cal_y,
                                                         self.cal_g, self.bg, k=ROUND4["top_k"],
                                                         m=cfg("m_score", self.smoke), eg_seeds=(2 * seed, 2 * seed + 1))
        return self.shares_cache[seed]

    # ---- xgp_r3 only: round 3's frozen method as it was
    def r3(self) -> P3.Context:
        if self._r3 is None:
            print("  building round 3's context for the xgp_r3 reference")
            self._r3 = P3.Context(self.args, self.k)
        return self._r3

    def r3_aug_set(self, base: Windows, base_logits: torch.Tensor, seed: int, r: float):
        """xgp_r3 only: round 3's aug_training recipe on `base` (identical to it for the full training set)."""
        from .augment import draw_older_like_r3_params, older_like_windows_r3
        N = len(base)
        n = int(round(r * N))
        sd = ROUND3["aug_seed_offset"] + seed
        idx = np.sort(np.random.default_rng(np.random.SeedSequence([sd, 1])).choice(N, n, replace=False))
        q = draw_older_like_r3_params(n, sd)
        aug = older_like_windows_r3(base.select(idx), q["s"], q["a"], q["axes"], q["angles"])
        both = Windows.concat([base, aug], "harth")
        require_harth(both)
        z_aug = torch.from_numpy(predict_logits(self.net0, aug.X, self.dev)).to(self.dev)
        return both, torch.cat([base_logits, z_aug]), np.r_[np.zeros(N, int), np.ones(n, int)]


# ------------------------------------------------------------------------------ store

class Store(P3.Store):
    TABLES = ("runs", "per-subject-lci", "finetune-history", "prune-steps", "budgets")

    def __init__(self, out: Path, k: int, mode: str, machine: str, resume: bool, smoke: bool):
        self.base = f"{P}-fold{k}-{mode}-{machine}"
        self.paths = {t: out / f"{self.base}-{t}.csv" for t in self.TABLES}
        self.config = out / f"{self.base}-config.json"
        exists = self.paths["runs"].exists()
        if exists and not (resume or smoke):
            raise FileExistsError(f"{self.paths['runs']} exists; pass --resume to continue it (never overwritten)")
        keep = resume and exists
        self.rows = {t: (pd.read_csv(p, dtype={"subject": str}).to_dict("records")
                         if keep and p.exists() and p.stat().st_size > 2 else []) for t, p in self.paths.items()}
        self.done = {r["run_id"] for r in self.rows["runs"]}


# ------------------------------------------------------------------------------ run

def run_fold(args, k: int, mode: str) -> None:
    S = args.smoke
    out = out_dir(S)
    store = Store(out, k, mode, args.machine_name, args.resume, S)
    log = out / "logs" / f"{store.base}-{R.stamp()}.log"
    with R.tee_to(log):
        dev = R.device_info(args.device)
        t_fold = time.perf_counter()
        print(f"=== XGP pilot round 4, fold {k}, mode {mode} | machine {args.machine_name} | device {args.device} "
              f"{dev['gpu_name']} | torch {dev['torch_version']} | started {R.now()} | {' '.join(sys.argv)}"
              + (f" | resuming, {len(store.done)} runs done" if store.done else ""))
        ctx = Context4(args, k)
        seeds = tuple(args.seeds) if args.seeds else cfg("seeds", S)
        K, epochs, curve_eps = cfg("K", S), cfg("epochs", S), cfg("curve_epochs", S)
        common = {"fold": k, "mode": mode, "tau_k": ctx.tau, "ceiling": ctx.ceiling, "machine_name": args.machine_name,
                  "device": args.device}
        nnz: dict[float, set] = {}
        starts: dict[tuple, nn.Module] = {}
        cdir = ckpt_dir(S)
        structure_info = {}

        def record(run_id, base, a, rows, net, check_nnz=True):
            rec = size_record(net)
            if check_nnz:
                nnz.setdefault(base["level"], set()).add(rec["n_nonzero"])
                if len(nnz[base["level"]]) != 1:
                    raise RuntimeError(f"n_nonzero differs within {base['level']:.0%}: {nnz[base['level']]}")
            store.rows["per-subject-lci"].extend({"run_id": run_id, **base, **r} for r in rows)
            store.rows["runs"].append({"run_id": run_id, **base, **a, "head_unpruned": head_unpruned(net),
                                       **{x: v for x, v in rec.items() if x != "per_layer_sparsity"}, **common})
            store.done.add(run_id)
            store.save()

        # ---- starts
        def build(kind, level, seed):
            if kind == "l1":
                return prune_by_score(ctx.net0, l1_scores(ctx.net0), level), [], None
            if kind in U_CELLS:
                budget, sel = U_CELLS[kind]
                beta = ROUND4["beta_u"] if sel == "xgp" else 0.0
                imp = ctx.importance4(seed) if sel == "xgp" else None
                if budget == "uniform":
                    net, steps = iterative_prune(ctx.net0, level, K=K, beta=beta, importance_fn=imp, recon_fn=ctx.recon4())
                    return net, steps, None
                shares = ctx.xgp_shares(seed) if budget == "xb" else None
                keep = bounded_budget(ctx.net0, level, ctx.unrecon, budget, shares)
                net, steps = iterative_prune_budget(ctx.net0, keep, K=K, beta=beta, importance_fn=imp,
                                                    recon_fn=ctx.recon4())
                uni = uniform_keep_counts(ctx.net0, level)
                brows = [{"fold": k, "level": level, "seed": seed, "start": kind, "budget": budget, "tensor": n,
                          "n": int(getattr(m, p).numel()), "uniform_keep": uni[n], "keep": keep[n],
                          "share": (shares or {}).get(n), "protected": n in ctx.unrecon}
                         for n, m, p in prunable_tensors(ctx.net0)]
                return net, steps, brows
            if kind in ("xgp_r3_nf", "xgp_r3_ft"):
                scen = "no_finetune" if kind == "xgp_r3_nf" else "finetune"
                c = r3_config(ctx.dec3, scen, level)
                kind3 = c["final"] if scen == "no_finetune" else R3_FT_START[c["final"]][0]
                net, steps = ctx.r3().build_start(kind3, level, seed, c.get("beta"))
                return net, steps, None
            if kind == "l1_filter":
                sc = filter_scores_l1(ctx.net0)
                F_ = {p: len(v) for p, v in sc.items()}
                removed = {p: sorted(int(i) for i in torch.argsort(sc[p], stable=True)[:int(round(level * F_[p]))])
                           for p in sc}
                net = apply_structured(ctx.net0, removed)
                net.prune_sparsity = level
                return net, [], None
            if kind in S_METHODS:
                crit = S_METHODS[kind]
                if crit == "l1":
                    fn = filter_scores_l1
                elif crit == "taylor":
                    fn = lambda net: filter_scores_taylor(net, ctx.taylor_batches, ctx.taylor_loss)  # noqa: E731
                else:
                    imp = ctx.importance4(seed)
                    fn = lambda net: filter_scores_from_weight_scores(net, imp(net))  # noqa: E731
                net, steps = iterative_prune_structured(ctx.net0, level, K=K, score_fn=fn, recon_fn=ctx.recon4())
                return net, steps, None
            raise ValueError(kind)

        deterministic = {"l1", "l1_recon", "lamp_recon", "l1_filter", "l1_filter_recon", "taylor_recon"}

        def get_start(kind, level, seed):
            sk = 0 if kind in deterministic else seed
            key = (kind, level, sk)
            if key in starts:
                return starts[key]
            tag = f"fold{k}-{mode}-{kind}-{pct(level)}pct-seed{sk}"
            p = cdir / f"{P}-{args.machine_name}-{tag}.pt"
            t0 = time.perf_counter()
            if p.exists():
                ck = torch.load(p, map_location="cpu", weights_only=False)
                net = copy.deepcopy(ctx.net0)
                net.load_state_dict(ck["state_dict"])
                net.prune_masks = {n: m.to(ctx.dev) for n, m in ck["prune_masks"].items()}
                net.prune_sparsity = level
                net = net.to(ctx.dev).eval()
                steps, brows = [], None
            else:
                print(f"\n--- start {tag}")
                net, steps, brows = build(kind, level, sk)
                cdir.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": {n: v.cpu() for n, v in net.state_dict().items()},
                            "prune_masks": {n: m.cpu() for n, m in net.prune_masks.items()}}, p)
                if brows:
                    store.rows["budgets"].extend(brows)
            net.start_seconds = time.perf_counter() - t0
            for st in steps:
                store.rows["prune-steps"].append({"start": tag, "step": st["step"], "n_nonzero": st["n_nonzero"],
                                                  "removed": json.dumps(st.get("removed")),
                                                  "recon": json.dumps(st["recon"])})
            starts[key] = net
            return net

        def no_ft(method, kind, level, seed):
            rid = f"no_finetune-{method}-{pct(level)}pct-seed{seed}"
            if rid in store.done:
                return
            net = get_start(kind, level, seed)
            a, rows = ctx.assess(net, f"{mode}-{pct(level)}pct-{rid}")
            base = {"level": level, "regime": "no_finetune", "scenario": "no_finetune", "method": method, "start": kind,
                    "seed": seed, "epoch": 0, "ptq_bits": None, "recipe": None, "r": None, "dro": None,
                    "deterministic": kind in deterministic}
            record(rid, base, {**a, "start_seconds": net.start_seconds}, rows, net)
            print(f"  {rid:<52} LCI {a['lci']:.4f} (worst subject {a['worst_subject_lci']:.4f}); F1 {a['f1']:.4f}")

        def ft(method, kind, level, seed, regime, rec, curve=False, ptq=()):
            rid = f"{regime}-{method}-{pct(level)}pct-seed{seed}"
            if rid in store.done:
                return
            start = get_start(kind, level, seed)
            net = copy.deepcopy(start)
            base_w, base_z, base_g = ((ctx.tr, ctx.kd_logits, ctx.tr_g) if regime == "kd100"
                                      else (ctx.tr5, ctx.z5, ctx.tr5_g))
            if rec["recipe"] == "kdaug":                       # xgp_r3 only: round 3's recipe
                train, z, groups = ctx.r3_aug_set(base_w, base_z, seed, rec["r"])
                gkind = "round3 normal / older-like"
            else:
                train, z, groups = base_w, base_z, base_g
                gkind = "subject"
            dro = bool(rec["dro"])
            base = {"level": level, "regime": regime, "scenario": "finetune", "method": method, "start": kind,
                    "seed": seed, "ptq_bits": None, "recipe": rec["recipe"], "r": rec["r"], "dro": dro,
                    "dro_groups": gkind if dro else None, "deterministic": False}
            print(f"\n=== fold {k} {rid}: start {kind}, {rec['recipe']}" + (f", DRO over {gkind}" if dro else "")
                  + f", {len(train)} windows, E_ft {epochs}")
            t_cb = [0.0]

            def cb(ep, model):
                if not curve or ep not in curve_eps:
                    return
                t0 = time.perf_counter()
                a, rows = ctx.assess(model, f"{mode}-{pct(level)}pct-{rid}-ep{ep}")
                record(f"{rid}-ep{ep}", {**base, "epoch": ep}, a, rows, model)
                t_cb[0] += time.perf_counter() - t0
                print(f"    curve epoch {ep}: LCI {a['lci']:.4f}, F1 {a['f1']:.4f}")
            info = finetune(net, train, ctx.va, epochs=epochs, lr=ROUND4["lr"], seed=seed, device=ctx.dev,
                            kd_teacher_logits=z, groups=groups if dro else None,
                            dro_eta=ROUND4["dro_eta"] if dro else None, on_epoch_end=cb)
            if info["epochs_run"] != epochs:
                raise RuntimeError(f"{rid}: {info['epochs_run']} epochs")
            store.rows["finetune-history"].extend({"run_id": rid, **h} for h in info["history"])
            a, rows = ctx.assess(net, f"{mode}-{pct(level)}pct-{rid}")
            record(rid, {**base, "epoch": epochs}, {**a, "finetune_seconds": info["seconds"] - t_cb[0],
                                                     "n_train_windows": len(train)}, rows, net)
            print(f"    -> LCI {a['lci']:.4f} (worst subject {a['worst_subject_lci']:.4f}); F1 {a['f1']:.4f}")
            for bits in ptq:
                q, qd = ptq_symmetric(net, bits)
                aq, rq = ctx.assess(q, f"{mode}-{pct(level)}pct-{rid}-ptq{bits}")
                recq = size_record(q, qd)
                record(f"{rid}-ptq{bits}", {**base, "epoch": epochs, "ptq_bits": bits},
                       {**aq, "size_kb_nominal_quantized": recq["size_kb_nominal"]}, rq, q, check_nnz=False)
                print(f"    PTQ{bits}: LCI {aq['lci']:.4f}; F1 {aq['f1']:.4f}")

        if "original" not in store.done:
            pred = predict_proba(ctx.net0, ctx.va.X, ctx.dev).argmax(1)
            store.rows["runs"].append({"run_id": "original", "regime": "original", "method": "original",
                                       "f1": pooled_macro_f1(ctx.va.y, pred),
                                       "accuracy": float((pred == ctx.va.y).mean()), "lci": 1.0,
                                       "worst_subject_lci": 1.0, **common})
            store.done.add("original")
            store.save()

        if mode == "unstructured":
            levels = tuple(args.levels) if args.levels else cfg("levels_u", S)
            for lv in levels:
                starts.clear()
                print(f"\n######## fold {k}, unstructured, level {lv:.0%}")
                rec3 = r3_ft_recipe(ctx.dec3, lv)
                has_nf3 = r3_config(ctx.dec3, "no_finetune", lv) is not None
                for m in ("l1", "l1_recon", "lamp_recon"):
                    no_ft(m, m, lv, 0)
                for s in seeds:
                    for m in ("l1_xb_recon", "xgp", "xgp_lamp", "xgp_xb"):
                        no_ft(m, m, lv, s)
                    if has_nf3:
                        no_ft("xgp_r3", "xgp_r3_nf", lv, s)
                for regime in ("kd100", "kd5"):
                    if regime == "kd5" and lv not in cfg("levels_u_kd5", S):
                        continue
                    for s in seeds:
                        for m in ("l1", *U_CELLS):
                            ptq = ()
                            if regime == "kd100" and m in U_PTQ and lv in cfg("ptq_levels_u", S):
                                ptq = (4, 8) if (abs(lv - cfg("ptq8_level_u", S)) < 1e-9 and s == 0) else (4,)
                            curve = regime == "kd100" and m in U_CURVE and lv in cfg("curve_levels_u", S)
                            ft(m, m, lv, s, regime, KD_DRO if m in U_CANDIDATES else KD, curve=curve, ptq=ptq)
                        for c, st in U_DRO_CONTROLS.items():
                            ft(c, st, lv, s, regime, KD_DRO)
                        if rec3 is not None:
                            ft("xgp_r3", "xgp_r3_ft", lv, s, regime, rec3)
        else:
            levels = tuple(args.levels) if args.levels else cfg("levels_s", S)
            for lv in levels:
                starts.clear()
                print(f"\n######## fold {k}, structured, level {lv:.0%} of the filters removed")
                no_ft("l1_filter", "l1_filter", lv, 0)
                no_ft("l1_filter_recon", "l1_filter_recon", lv, 0)
                no_ft("taylor_recon", "taylor_recon", lv, 0)
                for s in seeds:
                    no_ft("xgp_s", "xgp_s", lv, s)
                sl = slice_structured(get_start("l1_filter", lv, 0))
                x1 = torch.zeros(1, 6, 128)
                structure_info[f"{pct(lv)}pct"] = {
                    "n_params_dense": int(sum(p.numel() for p in ctx.net0.parameters())),
                    "n_params_sliced": int(sum(p.numel() for p in sl.parameters())),
                    "macs_dense": count_macs(copy.deepcopy(ctx.net0).cpu(), x1), "macs_sliced": count_macs(sl, x1),
                    "kept_filters_per_layer": {p: int(dict(sl.named_modules())[p].weight.shape[0])
                                               for p, _, _ in structured_layers(sl)}}
                for regime in ("kd100", "kd5"):
                    for s in seeds:
                        for m in ("l1_filter", *S_METHODS):
                            ptq = ()
                            if regime == "kd100" and m in S_PTQ and lv in cfg("ptq_levels_s", S):
                                ptq = (4, 8) if (abs(lv - cfg("ptq8_level_s", S)) < 1e-9 and s == 0) else (4,)
                            ft(m, m, lv, s, regime, KD_DRO if m in S_CANDIDATES else KD,
                               curve=regime == "kd100" and m in S_CURVE, ptq=ptq)
                        for c, st in S_DRO_CONTROLS.items():
                            ft(c, st, lv, s, regime, KD_DRO)
        wall = time.perf_counter() - t_fold
        R.write_json(store.config, {"command_line": sys.argv, "machine_name": args.machine_name, "device": dev,
                                    **R.git_info(), "code_sha256": R.code_sha256(), "fold": k, "mode": mode, "smoke": S,
                                    "smoke_sizes": SMOKE4 if S else None, "round4": ROUND4, "levels": list(levels),
                                    "seeds": list(seeds), "epochs": epochs, "K": K, "nsamples": ctx.nsamples,
                                    "n_calibration_round4": len(ctx.cal_y),
                                    "n_calibration_groups": len(set(ctx.cal_g.tolist())),
                                    "dro_groups": ctx.group_labels, "top_k": ROUND4["top_k"],
                                    "n_kd5": len(ctx.tr5), "n_train": len(ctx.tr), "unreconstructed": ctx.unrecon,
                                    "tau_k": ctx.tau, "tau_source": ctx.tau_src, "ceiling": ctx.ceiling,
                                    "ceiling_source": ctx.ceiling_src, "weights": str(ctx.ckpt),
                                    "preregistration_written_at": ctx.prereg4["written_at"],
                                    "round3_tag_commit": ctx.prereg4.get("round3_tag_commit"),
                                    "structure": structure_info or None,
                                    "n_nonzero": {str(l): sorted(v) for l, v in nnz.items()},
                                    "seconds_wall_this_invocation": wall, "finished_at": R.now()})
        print(f"\nfold {k} mode {mode} done in {wall:.1f} s")
        for p in (*store.paths.values(), store.config, log):
            print(f"saved {p}")


# ------------------------------------------------------------------------------ latency

def latency(args) -> None:
    S = args.smoke
    import json as _j
    frozen = _j.loads(P3.FROZEN_CONFIG.read_text(encoding="utf-8"))
    net0, _ = P3.load_fold_model(0, frozen["architecture"])
    net0 = net0.cpu().eval()
    lat = cfg("latency", S)
    rows = []
    x1 = torch.zeros(1, 6, 128)
    models = {"dense": net0}
    for lv in cfg("levels_s", S):
        sc = filter_scores_l1(net0)
        removed = {p: sorted(int(i) for i in torch.argsort(v, stable=True)[:int(round(lv * len(v)))]) for p, v in sc.items()}
        masked = apply_structured(net0, removed)
        sl = slice_structured(masked)
        X = torch.randn(16, 6, 128, generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            diff = float((masked.eval()(X) - sl(X)).abs().max())
        if diff > 1e-4:
            raise RuntimeError(f"sliced model differs from the masked one by {diff}")
        models[f"structured_{pct(lv)}pct"] = sl
    for name, m in models.items():
        for th in lat["threads"]:
            for b in lat["batches"]:
                r = cpu_latency(m, b, th, lat["warmup"], lat["reps"])
                rows.append({"model": name, "n_params": int(sum(p.numel() for p in m.parameters())),
                             "macs_per_window": count_macs(m, x1), **r, "machine_name": args.machine_name,
                             "torch_version": torch.__version__, "measured_at": R.now()})
                print(rows[-1])
    p = out_dir(S) / f"{P}-structured-latency-cpu-{args.machine_name}.csv"
    R.write_csv(p, pd.DataFrame(rows))
    print(f"saved {p}")


# ------------------------------------------------------------------------------ analysis

def load_runs(out: Path, mode: str, folds, use_machine: dict[int, str]) -> pd.DataFrame:
    parts = []
    for k in folds:
        found = sorted(out.glob(f"{P}-fold{k}-{mode}-*-runs.csv"))
        if k in use_machine:
            found = [f for f in found if f.name == f"{P}-fold{k}-{mode}-{use_machine[k]}-runs.csv"]
        if len(found) != 1:
            raise FileNotFoundError(f"need exactly one fold-{k} {mode} runs file (use --use-machine), found "
                                    f"{[f.name for f in found]}")
        parts.append(pd.read_csv(found[0]))
    df = pd.concat(parts, ignore_index=True)
    return df[df.regime != "original"]


def finals_only(df: pd.DataFrame, epochs: int) -> pd.DataFrame:
    d = df[df.ptq_bits.isna()]
    return d[(d.regime == "no_finetune") | (d.epoch == epochs)]


def expand_deterministic(d: pd.DataFrame, seeds) -> pd.DataFrame:
    det = d[d.deterministic.astype(str).str.lower() == "true"]
    rest = d[d.deterministic.astype(str).str.lower() != "true"]
    reps = [det.assign(seed=s) for s in seeds]
    return pd.concat([rest, *reps], ignore_index=True)


def analyze(args) -> int:
    S, out, mode = args.smoke, out_dir(args.smoke), args.mode
    folds = tuple(args.folds) if args.folds else ROUND4["folds"]
    seeds = cfg("seeds", S)
    um = {int(x.split("=")[0]): x.split("=")[1] for x in (args.use_machine or [])}
    df = load_runs(out, mode, folds, um)
    fin = expand_deterministic(finals_only(df, cfg("epochs", S)), seeds)
    if mode == "unstructured":
        levels_by = {"no_finetune": cfg("levels_u", S), "kd100": cfg("levels_u", S), "kd5": cfg("levels_u_kd5", S)}
        baselines, controls, cands = U_BASELINES, tuple(U_DRO_CONTROLS), U_CANDIDATES
    else:
        levels_by = {r: cfg("levels_s", S) for r in REGIMES}
        baselines, controls, cands = S_BASELINES, tuple(S_DRO_CONTROLS), S_CANDIDATES
    n_pairs = len(folds) * len(seeds)
    min_better = ROUND4["min_better"] if n_pairs == 6 else n_pairs
    decision, table = {}, []
    for regime in REGIMES:
        g = fin[fin.regime == regime]
        lvs = list(levels_by[regime])
        g = g[g.level.isin(lvs)]
        bl = baselines if regime == "no_finetune" else (*baselines, *controls)
        stats, runs = {}, {}
        missing = []
        for m, gm in g.groupby("method"):
            stats[m] = {lv: {"lci": float(x.lci.mean()), "worst_subject_lci": float(x.worst_subject_lci.mean()),
                             "f1": float(x.f1.mean()),
                             "lci_sd_seeds": float(x.lci.std(ddof=1)) if len(x) > 1 else float("nan"), "n": len(x)}
                        for lv, x in gm.groupby("level")}
            pr = gm.groupby(["fold", "seed"]).lci.mean()
            runs[m] = {(int(a), int(b)): float(v) for (a, b), v in pr.items()}
            if any(stats[m].get(lv, {}).get("n", 0) != n_pairs for lv in lvs):
                missing.append(m)
        if missing or not set(bl) <= set(stats):
            decision[regime] = {"final": None, "reason": f"missing runs: {sorted(set(missing) | (set(bl) - set(stats)))}"
                                ", no round-4 decision (round 3 stays final)"}
            continue
        best = {lv: {"f1": max(stats[b][lv]["f1"] for b in bl)} for lv in lvs}
        final, tab = pick_final(stats, best, cands)
        for t in tab:
            table.append({"mode": mode, "regime": regime, **t})
        lci_mean = {m: float(np.mean([stats[m][lv]["lci"] for lv in lvs])) for m in stats}
        wsl_mean = {m: float(np.mean([stats[m][lv]["worst_subject_lci"] for lv in lvs])) for m in stats}
        best_bl = max(bl, key=lambda b: lci_mean[b])
        res = {"final": final, "levels": lvs, "baselines": list(bl),
               "best_baseline_f1": {f"{pct(l)}pct": v for l, v in best.items()},
               "best_baseline_by_lci": best_bl, "lci_mean_levels": lci_mean,
               "worst_subject_lci_mean_levels": wsl_mean, "candidates": tab}
        if final is not None:
            claim = beats(runs[final], runs[best_bl], stats[final], {lv: stats[best_bl][lv] for lv in lvs},
                          min_better=min_better)
            res["vs_best_baseline"] = claim
            res["claim"] = "beats the best baseline" if claim["beats"] else "matches the best baseline at the same size"
        if mode == "unstructured":
            if "xgp_r3" in stats:
                if final is None:
                    res["replacement"] = {"round4_replaces_round3": False, "reason": "no eligible round-4 final"}
                else:
                    rep = beats(runs[final], runs["xgp_r3"], stats[final], {lv: stats["xgp_r3"][lv] for lv in lvs},
                                min_better=min_better)
                    res["replacement"] = {"round4_replaces_round3": rep["beats"], **rep}
            else:
                res["replacement"] = {"round4_replaces_round3": final is not None,
                                      "reason": "round 3 has no final in this regime; round 4 adopted if it has one"}
            res["adopted"] = (final if res["replacement"]["round4_replaces_round3"] else "xgp_r3 (round 3)")
        else:
            res["adopted"] = final
            res["note"] = "structured is an additional track; enters 9B only if finished and decided by 27 Nov morning"
        decision[regime] = res
        print(f"{mode} {regime}: final {final}; adopted {res['adopted']}"
              + (f"; vs best baseline {best_bl}: gain {res['vs_best_baseline']['mean_gain']:+.4f}, "
                 f"{res['vs_best_baseline']['n_better']}/{res['vs_best_baseline']['n_pairs']} better, "
                 f"beats={res['vs_best_baseline']['beats']}" if final else ""))
    # summaries: pooled and per fold, seed SD
    met = list(METRICS)
    def summ(d, by):
        rows = []
        for key, x in d.groupby(by, dropna=False):
            row = dict(zip(by, key))
            row["n_runs"] = len(x)
            for m in met:
                row[f"{m}_mean"] = float(x[m].mean())
                row[f"{m}_sd"] = float(x[m].std(ddof=1)) if len(x) > 1 else float("nan")
            rows.append(row)
        return pd.DataFrame(rows)
    R.write_csv(out / f"{P}-{mode}-summary-mean-sd.csv", summ(fin, ["regime", "level", "method"]))
    R.write_csv(out / f"{P}-{mode}-summary-per-fold-seed-sd.csv", summ(fin, ["regime", "level", "method", "fold"]))
    cur = df[(df.regime == "kd100") & df.ptq_bits.isna()]
    nf = expand_deterministic(df[df.regime == "no_finetune"], seeds)
    ep0 = nf[nf.method.isin(set(cur.method))].assign(epoch=0)
    curve = pd.concat([cur, ep0], ignore_index=True)
    curve = curve[curve.method.isin(U_CURVE if mode == "unstructured" else S_CURVE)]
    R.write_csv(out / f"{P}-{mode}-budget-curve-mean-sd.csv", summ(curve, ["level", "method", "epoch"]))
    q = df[df.ptq_bits.notna()]
    if len(q):
        R.write_csv(out / f"{P}-{mode}-ptq-mean-sd.csv", summ(q, ["level", "method", "ptq_bits"]))
    R.write_csv(out / f"{P}-{mode}-decision-table.csv", pd.DataFrame(table))
    R.write_json(out / f"{P}-{mode}-decision.json", {"decision": decision, "rules": RULES, "disclosure": DISCLOSURE,
                                                     "amendment": AMENDMENT,
                                                     "folds": list(folds), "seeds": list(seeds), "min_better": min_better,
                                                     "analyzed_at": R.now(), "smoke": S, **R.git_info()})
    print(f"saved {out / f'{P}-{mode}-decision.json'}")
    return 0


# ------------------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    act = ap.add_mutually_exclusive_group(required=True)
    act.add_argument("--prepare", action="store_true")
    act.add_argument("--run", action="store_true", help="(default with --mode and --folds)")
    act.add_argument("--analyze", action="store_true")
    act.add_argument("--latency", action="store_true")
    ap.add_argument("--mode", choices=["unstructured", "structured"], default=None)
    ap.add_argument("--folds", type=int, nargs="+", default=None, choices=ROUND4["folds"])
    ap.add_argument("--levels", type=float, nargs="+", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--machine-name", default=R.machine_name())
    ap.add_argument("--noise-floor-machine", default=None)
    ap.add_argument("--use-machine", nargs="*", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    argv = sys.argv[1:]
    if "--mode" in argv and not any(a in argv for a in ("--prepare", "--run", "--analyze", "--latency")):
        argv = ["--run", *argv]
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.prepare:
        return prepare(args)
    if args.latency:
        return latency(args)
    if args.mode is None:
        ap.error("--mode unstructured|structured is required")
    if args.analyze:
        sys.exit(analyze(args))
    if not args.folds:
        ap.error("--folds is required to run a mode")
    t0 = time.perf_counter()
    for k in args.folds:
        run_fold(args, k, args.mode)
    print(f"\nfolds {args.folds}, mode {args.mode}: {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
