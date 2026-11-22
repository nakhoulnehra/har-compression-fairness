"""Step 9: the compression grid (plan Section 3 Step 9; methods Section 2.5, metrics 2.6-2.7, statistics 2.8).

Part A (decided 2026-11-21): the standard methods only, so explanation drift and its subgroup gap are
measured (RQ1, RQ2, RQ0) before XGP's final version is designed. XGP is added later as one more
registered method (`register_method(Method("xgp_kd", score="xgp", recipe="kd"), scorer=...)`) and runs
on the same folds, windows, recipe and E_ft (part B); nothing in the grid itself changes.

Per fold k (folds and machines as in STATUS.md), every model starts from the frozen TinyHAR of the fold
(seed 0, model-selection checkpoint):
    PTQ        ptq8, ptq4 (per-tensor symmetric, primary) and ptq_tulla8, ptq_tulla4 (Tulla-style, secondary)
    pruning    30, 50, 70, 90% uniform per-tensor sparsity, head excluded, identical n_nonzero:
               l1 (CE), l1_kd, snip (CE), snip_kd; park and park_kd at 70% and 90% only (gamma_P = 1, EG m = 4)
               KD = 0.7 KL(T = 3) T^2 + 0.3 class-weighted CE to the original model's logits (har.compression.KD)
               E_ft = 20 for every fine-tuned method (pilot round 2's epochs check), Adam lr 1e-4, seed 0
    scenarios  no_finetune (the pruned start of each score: l1, snip) and finetune (each method)
    references the original model; the noise-floor models (SHAP seeds 1, 2 and retrain seeds 1-3, from the
               noise-floor cache of this machine) scored with the same code, as a per-class noise reference
Explain windows: the fold's fixed explain windows (results/har-setup/har-explain-windows.csv): inner
validation (3 HARTH subjects), general = the fold's HARTH test subjects, older = all 18 HAR70+ subjects;
walking-aid (inferred) = 503, 505, 513, 514, 515. SHAP: GradientExplainer nsamples 200, SHAP seed 0, one
call per group (as in the noise floor, so the original model's SHAP is the noise floor's cached array),
target = the original model's predicted class, background = the fold's 100 HARTH windows.
Metrics per subject: LCI (48 segments, k = 10, omega = 0.5) with rho and Jaccard, LCI on the raw 768
values and with k = 16, retention = (LCI - tau_k) / (ceiling - tau_k) with tau_k and the group's
SHAP-noise ceiling from results/har-noise-floor/ (this machine's file), deletion faithfulness (AOPC,
drop at k = 5, random-deletion baseline), accuracy and non-stair macro-F1 on all of the subject's test
windows, and the confusion matrix (stairs F1 is only ever pooled, in --analyze). Per class: LCI per
(subject, true class) with >= 3 windows, and per (group, class) on the group's pooled windows.

Usage (repo root, inside .venv):
    python -m har.run_compression_grid --folds 0 1 3 --device cuda --machine-name nakhoul-3050
    python -m har.run_compression_grid --folds 2 3 4 --device cuda --machine-name uni-2080
    python -m har.run_compression_grid --analyze          # fold 3 = uni-2080's copy (FOLD_MACHINE_DEFAULT)
    --resume continues an interrupted fold (finished model IDs are skipped); results are saved after every
    model. --smoke: tiny CPU check (results/har-compression-grid-smoke/).

Writes results/har-compression-grid/, per fold k and machine (prefix har-grid-tinyhar-fold<k>-<machine>):
    -per-subject-metrics.csv    one row per (model, subject)
    -per-subject-class-lci.csv  one row per (model, subject, class) with >= 3 windows
    -per-group-class-lci.csv    one row per (model, group, class), the group's windows pooled
    -models.csv                 sizes, sparsities, per-layer sparsity, fine-tuning, timing, tau_k, machine
    -finetune-history.csv       per epoch: losses and inner-val F1
    -shap-48-segments.npz       48-segment SHAP (n, 48) of every model on the explain windows, one array per
                                model ID, plus meta_window_id / meta_subject / meta_group / meta_class_index /
                                meta_target_class / meta_walking_aid_inferred (row order of every array)
    -config.json, logs/
and with --analyze results/har-compression-grid-analysis/har-grid-a-*.csv (see analyze()).
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS; must precede CUDA init

import argparse
import copy
import json
import sys
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch

from . import runlog as R
from .compression import (finetune, head_unpruned, l1_scores, predict_logits, prune_by_score, ptq_symmetric, ptq_tulla, size_record,
                          snip_scores, teacher_segment_attributions, unit_norm)
from .data import CLASS_NAMES, STAIRS, WALKING_AID_INFERRED, require_harth
from .faithfulness import background_mean, deletion_curve, faithfulness_scores, random_deletion_curve
from .folds import load_folds
from .paths import (GRID_ANALYSIS_DIR, GRID_ANALYSIS_SMOKE_DIR, GRID_CKPT_DIR, GRID_DIR, GRID_SMOKE_CKPT_DIR,
                    GRID_SMOKE_DIR, MODEL_SELECTION_DIR, NOISE_FLOOR_CKPT_DIR, NOISE_FLOOR_DIR, NOISE_FLOOR_SMOKE_CKPT_DIR,
                    NOISE_FLOOR_SMOKE_DIR, SHAP_CACHE_DIR)
from .run_model_selection import subsample
from .run_noise_floor import RETRAIN_SEEDS, SHAP_SEEDS, explain_sets, fold_background
from .run_sanity_walking_aid import load_fold_model
from .run_xgp_pilot import read_tau
from .stats import (bootstrap_fa_lci_ci, bootstrap_group_gap_ci, bootstrap_mean_ci, holm, mann_whitney,
                    paired_wilcoxon, pool_over_folds, safe_rate)
from .train import confusion, f1_from_confusion, pooled_macro_f1, predict_proba, subject_metrics
from .attributions import shap_cached
from .compression import KD
from .data import find_windows, load_harth_windows, load_test_windows
from .models import build_model
from .xgp import make_extra_loss

FROZEN_CONFIG = MODEL_SELECTION_DIR / "har-frozen-model-config.json"
GRID = {"levels": (0.3, 0.5, 0.7, 0.9), "park_levels": (0.7, 0.9), "ptq_bits": (8, 4), "epochs": 20, "lr": 1e-4,
        "seed": 0, "park_gamma_p": 1.0, "park_m": 4, "nsamples": 200, "shap_seed": 0, "k": 10, "k_sensitivity": 16,
        "omega": 0.5, "min_class_windows": 3, "random_repeats": 5, "random_seed": 42, "tulla_tau": 0.85}
SMOKE = {"epochs": 1, "n_train": 600, "n_eval_per_subject": 40, "nsamples": 20, "park_m": 2, "min_class_windows": 1}
EXPLAIN_GROUPS = ("inner_val", "general", "older")
TEST_GROUPS = ("general", "older")
SETTING = ["method", "family", "scenario", "level"]
N_BOOT, BOOT_SEED = 10_000, 42
# Fold 3 runs on both machines (plan Step 9b); the analysis uses uni-2080's copy (decided 2026-11-21).
# --use-machine overrides it; --smoke ignores it.
FOLD_MACHINE_DEFAULT = {3: "uni-2080"}


# ------------------------------------------------------------------------------ method registry

@dataclass(frozen=True)
class Method:
    """A fine-tuned pruning method: `score` names the registered scorer that ranks the weights (the
    no_finetune start of that score is assessed once per level), `recipe` the shared supervised term
    ("ce" or "kd"), `park` adds gamma_P L_AM."""
    name: str
    score: str
    recipe: str
    park: bool = False
    levels: tuple = GRID["levels"]


METHODS: dict[str, Method] = {}
SCORERS: dict[str, Callable] = {}


def register_method(m: Method, scorer: Callable | None = None) -> None:
    """Add a method (and, if given, its scorer: ctx -> {prunable tensor name: score}) to the grid."""
    if scorer is not None:
        SCORERS[m.score] = scorer
    if m.score not in SCORERS:
        raise KeyError(f"no scorer registered for score {m.score!r}")
    if m.recipe not in ("ce", "kd"):
        raise ValueError(m.recipe)
    METHODS[m.name] = m


register_method(Method("l1", "l1", "ce"), scorer=lambda ctx: l1_scores(ctx.net0))
register_method(Method("l1_kd", "l1", "kd"))
register_method(Method("snip", "snip", "ce"), scorer=lambda ctx: ctx.snip)
register_method(Method("snip_kd", "snip", "kd"))
register_method(Method("park", "l1", "ce", park=True, levels=GRID["park_levels"]))
register_method(Method("park_kd", "l1", "kd", park=True, levels=GRID["park_levels"]))
PART_A = ("l1", "l1_kd", "snip", "snip_kd", "park", "park_kd")
PTQ = {"ptq8": ("symmetric", 8), "ptq4": ("symmetric", 4), "ptq_tulla8": ("tulla", 8), "ptq_tulla4": ("tulla", 4)}


def pct(level: float) -> int:
    return int(round(level * 100))


def grid_plan(methods: Sequence[str] = PART_A, levels: Sequence[float] = GRID["levels"],
              ptq: Sequence[str] = tuple(PTQ)) -> list[dict]:
    """The models of one fold, in run order: the original, PTQ, then per level (ascending) the
    no_finetune start of each score followed by the fine-tuned methods."""
    base = {"sparsity": np.nan, "bits": np.nan, "score": None, "recipe": None, "park_term": False, "epochs": 0}
    plan = [{**base, "model_id": "original", "method": "original", "family": "original", "scenario": "original",
             "level": "none"}]
    for name in ptq:
        variant, bits = PTQ[name]
        plan.append({**base, "model_id": name, "method": name, "family": "ptq", "scenario": "no_finetune",
                     "level": f"{bits}-bit", "bits": bits, "ptq_variant": variant})
    for lv in sorted(levels):
        ms = [METHODS[m] for m in methods if any(abs(lv - x) < 1e-9 for x in METHODS[m].levels)]
        for s in dict.fromkeys(m.score for m in ms):
            plan.append({**base, "model_id": f"{s}-{pct(lv)}pct-no_finetune", "method": s, "family": "pruning",
                         "scenario": "no_finetune", "level": f"{pct(lv)}%", "sparsity": lv, "score": s})
        for m in ms:
            plan.append({**base, "model_id": f"{m.name}-{pct(lv)}pct-finetune", "method": m.name, "family": "pruning",
                         "scenario": "finetune", "level": f"{pct(lv)}%", "sparsity": lv, "score": m.score,
                         "recipe": m.recipe, "park_term": m.park, "epochs": GRID["epochs"]})
    return plan


def pending(plan: list[dict], done: set) -> list[dict]:
    return [p for p in plan if p["model_id"] not in done]


# ------------------------------------------------------------------------------ metrics (pure)

def retention(lci: float, tau: float, ceiling: float) -> float:
    """(LCI - tau_k) / (ceiling - tau_k): 0 = at the retrain-noise floor, 1 = as consistent as SHAP with itself."""
    return (lci - tau) / (ceiling - tau)


def subgroup(group: str, subject: str) -> str:
    if group != "older":
        return group
    return "walking_aid_inferred" if subject in WALKING_AID_INFERRED else "other_older"


def explanation_metrics(A_ref: np.ndarray, A: np.ndarray, subjects, groups, labels, *, k: int = GRID["k"],
                        omega: float = GRID["omega"], min_class_windows: int = GRID["min_class_windows"]
                        ) -> tuple[list[dict], list[dict], list[dict]]:
    """LCI(original, model) on (n, 6, 128) attributions of the same windows:
    (1) per subject: 48-segment LCI (k) with rho, Jaccard, n_skipped, MPRF; raw-768 LCI; 48-segment LCI at k = 16;
    (2) per (subject, true class) with >= min_class_windows windows; (3) per (group, true class), the group's
    windows pooled (groups: inner_val, general, older, walking_aid_inferred, other_older)."""
    from lci.metric import aggregate_time_segments, compute_lci

    subjects, groups, labels = np.asarray(subjects), np.asarray(groups), np.asarray(labels)
    S_ref, S = aggregate_time_segments(A_ref), aggregate_time_segments(A)
    R_ref, R_ = np.asarray(A_ref).reshape(len(A_ref), -1), np.asarray(A).reshape(len(A), -1)
    keep = ("lci", "shap_corr", "jaccard", "n_skipped")
    subj_rows, class_rows, group_rows = [], [], []
    for s in sorted(set(subjects.tolist())):
        m = subjects == s
        g = str(groups[m][0])
        main = compute_lci(S_ref[m], S[m], k=k, omega=omega)
        raw = compute_lci(R_ref[m], R_[m], k=k, omega=omega)
        k16 = compute_lci(S_ref[m], S[m], k=GRID["k_sensitivity"], omega=omega)
        subj_rows.append({"subject": s, "group": g, **main, "lci768": raw["lci"], "shap_corr768": raw["shap_corr"],
                          "jaccard768": raw["jaccard"], "lci_k16": k16["lci"], "jaccard_k16": k16["jaccard"]})
        for c in sorted(set(labels[m].tolist())):
            mc = m & (labels == c)
            if mc.sum() >= min_class_windows:
                r = compute_lci(S_ref[mc], S[mc], k=k, omega=omega)
                class_rows.append({"subject": s, "group": g, "class_index": int(c), "class_name": CLASS_NAMES[c],
                                   "n_windows": int(mc.sum()), **{x: r[x] for x in keep}})
    sub = np.array([subgroup(g, s) for g, s in zip(groups, subjects)])
    pools = {g: groups == g for g in sorted(set(groups.tolist()))}
    for g in ("walking_aid_inferred", "other_older"):
        if (sub == g).any():
            pools[g] = sub == g
    for g, mg in pools.items():
        for c in sorted(set(labels[mg].tolist())):
            mc = mg & (labels == c)
            if mc.sum() >= min_class_windows:
                r = compute_lci(S_ref[mc], S[mc], k=k, omega=omega)
                group_rows.append({"group": g, "class_index": int(c), "class_name": CLASS_NAMES[c],
                                   "n_windows": int(mc.sum()), "n_subjects": int(len(set(subjects[mc].tolist()))),
                                   **{x: r[x] for x in keep}})
    return subj_rows, class_rows, group_rows


def confusion_str(cm: np.ndarray) -> str:
    return ";".join(str(int(v)) for v in np.asarray(cm).flatten())


def parse_confusion(s: str) -> np.ndarray:
    v = np.array([int(x) for x in str(s).split(";")])
    n = int(round(np.sqrt(len(v))))
    return v.reshape(n, n)


def accuracy_rows(y, pred, subjects) -> list[dict]:
    """Per subject: n_windows, accuracy, non-stair macro-F1 (classes present in the subject) and the
    7 x 7 confusion matrix (rows = true). No per-subject stairs F1 [plan 2.7, Q11]."""
    y, pred, subjects = map(np.asarray, (y, pred, subjects))
    rows = subject_metrics(y, pred, subjects)
    for r in rows:
        m = subjects == r["subject"]
        r["confusion7"] = confusion_str(confusion(y[m], pred[m]))
    return rows


def pooled_class_f1_from_confusion(cm: np.ndarray, classes: Sequence[int] = STAIRS) -> dict:
    f = f1_from_confusion(np.asarray(cm))
    return {c: float(f[c]) for c in classes}


# ------------------------------------------------------------------------------ fold context

def noise_floor_file(k: int, smoke: bool, machine: str, kind: str) -> Path:
    p = (NOISE_FLOOR_SMOKE_DIR if smoke else NOISE_FLOOR_DIR) / f"har-noise-floor-tinyhar-fold{k}-{machine}-{kind}.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: tau_k and the ceilings come from this machine's noise-floor run "
                                f"(python -m har.run_noise_floor --folds {k}), or pass --noise-floor-machine")
    return p


def read_ceilings(k: int, smoke: bool, machine: str) -> dict:
    g = pd.read_csv(noise_floor_file(k, smoke, machine, "group-summary"))
    g = g[g.noise == "shap"].set_index("group")["mean"]
    return {grp: float(g[grp]) for grp in EXPLAIN_GROUPS}


class Context:
    """Everything fixed for one fold: data, original model, explain windows, tau_k, ceilings, SHAP of
    the original model, and (prepare_training) the training-side references."""

    def __init__(self, args, k: int, harth: dict, test: dict, folds, frozen: dict):
        self.args, self.k, self.dev, self.smoke = args, k, args.device, args.smoke
        S = SMOKE if args.smoke else {}
        self.S = S
        self.nsamples = S.get("nsamples", GRID["nsamples"])
        self.cache = SHAP_CACHE_DIR / ("smoke" if args.smoke else "")
        fold = folds[k]
        self.fold = fold
        self.tr = harth["harth_train"].for_subjects(fold.train)
        self.va = harth["harth_test"].for_subjects(fold.val)
        require_harth(self.tr, self.va)
        assert not (set(self.tr.subjects) | set(self.va.subjects)) & (set(fold.test_general) | set(fold.test_older))
        if args.smoke:
            self.tr = subsample(self.tr, S["n_train"], 0)
        nf_machine = args.noise_floor_machine or args.machine_name
        self.nf_machine = nf_machine
        self.tau, self.tau_src = read_tau(k, args.smoke, nf_machine)
        self.ceilings = read_ceilings(k, args.smoke, nf_machine)
        net0, self.ckpt = load_fold_model(k, frozen["architecture"])
        self.net0 = net0.to(self.dev).eval()
        self.bg = fold_background(harth, k).to(self.dev)
        self.bg_mean = background_mean(self.bg)
        # explain windows per group (the noise floor's sets, same order -> its SHAP cache is reused)
        self.sets = explain_sets(fold, harth, test, args.smoke)
        lab = {}
        for g, split, w in (("inner_val", "harth_test", harth["harth_test"]), ("general", "harth_test", test["harth_test"]),
                            ("older", "har70plus_test", test["har70plus_test"])):
            lab[g] = w.y[find_windows(w, split, self.sets[g][1])]
        self.target = {g: predict_proba(self.net0, X, self.dev).argmax(1) for g, (X, _, _) in self.sets.items()}
        self.X_all = np.concatenate([self.sets[g][0] for g in EXPLAIN_GROUPS])
        self.ids_all = sum((list(self.sets[g][1]) for g in EXPLAIN_GROUPS), [])
        self.subj_all = np.concatenate([self.sets[g][2] for g in EXPLAIN_GROUPS])
        self.group_all = np.concatenate([[g] * len(self.sets[g][1]) for g in EXPLAIN_GROUPS])
        self.label_all = np.concatenate([lab[g] for g in EXPLAIN_GROUPS])
        self.target_all = np.concatenate([self.target[g] for g in EXPLAIN_GROUPS])
        # evaluation windows (accuracy, F1): all test windows of the fold's subjects + inner val
        self.eval = {"inner_val": self.va, "general": test["harth_test"].for_subjects(fold.test_general),
                     "older": test["har70plus_test"].for_subjects(fold.test_older)}
        if args.smoke:
            keep = set(self.subj_all.tolist())
            self.eval = {g: w.for_subjects([s for s in w.subjects if s in keep]) for g, w in self.eval.items()}
            self.eval = {g: w.select(np.concatenate([np.flatnonzero(w.subject == s)[:S["n_eval_per_subject"]]
                                                     for s in w.subjects])) for g, w in self.eval.items()}
        self.times, self.scores = {}, {}
        t0 = time.perf_counter()
        self.A_ref = self.shap(self.net0, "original")
        self.times["shap_original"] = time.perf_counter() - t0
        print(f"fold {k}: train {len(self.tr)} windows, inner val {list(fold.val)}, general {list(fold.test_general)}, "
              f"older {len(set(self.sets['older'][2]))} subjects; explain " +
              ", ".join(f"{g} {len(self.sets[g][1])}" for g in EXPLAIN_GROUPS) +
              f"; tau_k {self.tau:.4f} ({self.tau_src}); ceilings " +
              ", ".join(f"{g} {v:.4f}" for g, v in self.ceilings.items()) +
              f"; original SHAP {self.times['shap_original']:.1f} s")

    def shap(self, net, model_id: str, shap_seed: int = GRID["shap_seed"], noise_floor_tag: bool = False) -> np.ndarray:
        """SHAP per explain group (one shap_cached call per group, as in the noise floor), concatenated. The
        original model and the noise-floor models use the noise floor's file tags, so its cache is reused."""
        out = []
        for g in EXPLAIN_GROUPS:
            X, ids, _ = self.sets[g]
            t = (f"har-noise-floor-fold{self.k}-{g}" if noise_floor_tag or model_id == "original" else
                 f"har-grid-fold{self.k}-{g}-{model_id}")
            out.append(shap_cached(net, self.bg, torch.from_numpy(X).to(self.dev), self.target[g], ids, shap_seed,
                                   self.nsamples, self.cache, t))
        return np.concatenate(out)

    def prepare_training(self) -> None:
        """KD teacher logits, Park's fixed targets and SNIP scores, from the original model on the fold's
        HARTH training windows only."""
        require_harth(self.tr)
        t0 = time.perf_counter()
        self.Xtr = torch.from_numpy(self.tr.X).to(self.dev)
        self.kd_logits = torch.from_numpy(predict_logits(self.net0, self.tr.X, self.dev)).to(self.dev)
        self.target_tr = self.kd_logits.argmax(1)
        m = self.S.get("park_m", GRID["park_m"])
        self.park_ref = unit_norm(teacher_segment_attributions(self.net0, self.Xtr, self.target_tr, self.bg, m=m,
                                                               seed=0).to(self.dev))
        self.times["teacher_logits_and_park_targets"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.snip = snip_scores(self.net0, self.tr)
        self.times["snip_scores"] = time.perf_counter() - t0

    def eval_rows(self, net) -> tuple[dict[str, dict], dict[str, float]]:
        """Per subject: accuracy, non-stair macro-F1, confusion; per group: pooled accuracy and 7-class F1."""
        out, pooled = {}, {}
        for g, w in self.eval.items():
            pred = predict_proba(net, w.X, self.dev).argmax(1)
            pooled[f"accuracy_pooled_{g}"] = float((pred == w.y).mean())
            pooled[f"macro_f1_pooled7_{g}"] = pooled_macro_f1(w.y, pred)
            for r in accuracy_rows(w.y, pred, w.subject):
                out[r["subject"]] = {"accuracy": r["accuracy"], "non_stair_macro_f1": r["non_stair_macro_f1"],
                                     "n_windows_eval": r["n_windows"], "confusion7": r["confusion7"]}
        return out, pooled

    def faith_rows(self, net, A: np.ndarray) -> dict[str, dict]:
        from lci.metric import aggregate_time_segments

        seg = aggregate_time_segments(A)
        sc = faithfulness_scores(deletion_curve(net, self.X_all, seg, self.target_all, self.bg_mean, device=self.dev))
        rnd = faithfulness_scores(random_deletion_curve(net, self.X_all, self.target_all, self.bg_mean,
                                                        n_repeats=GRID["random_repeats"], seed=GRID["random_seed"],
                                                        device=self.dev))
        out = {}
        for s in sorted(set(self.subj_all.tolist())):
            m = self.subj_all == s
            out[s] = {"aopc": float(sc["aopc"][m].mean()), "drop_k5": float(sc["drop_k5"][m].mean()),
                      "random_aopc": float(rnd["aopc"][m].mean()), "random_drop_k5": float(rnd["drop_k5"][m].mean())}
        return out


# ------------------------------------------------------------------------------ one fold

class FoldStore:
    """The fold's result tables, saved after every model (atomic writes); --resume reloads them."""

    TABLES = ("per-subject-metrics", "per-subject-class-lci", "per-group-class-lci", "models", "finetune-history")

    def __init__(self, out: Path, k: int, machine: str, resume: bool, smoke: bool):
        self.base = f"har-grid-tinyhar-fold{k}-{machine}"
        self.paths = {t: out / f"{self.base}-{t}.csv" for t in self.TABLES}
        self.npz = out / f"{self.base}-shap-48-segments.npz"
        self.config = out / f"{self.base}-config.json"
        exists = self.paths["models"].exists()
        if exists and not (resume or smoke):
            raise FileExistsError(f"{self.paths['models']} exists; pass --resume to continue it (never overwritten)")
        keep = resume and exists
        self.rows = {t: (self._read(p) if keep else []) for t, p in self.paths.items()}
        self.seg = {}
        if keep and self.npz.exists():
            with np.load(self.npz, allow_pickle=False) as z:
                self.seg = {n: z[n] for n in z.files}
        self.done = {r["model_id"] for r in self.rows["models"]}

    @staticmethod
    def _read(p: Path) -> list[dict]:
        """Rows of a saved table; a table that was still empty (e.g. no fine-tuning yet) reads as []."""
        try:
            return pd.read_csv(p, dtype={"subject": str}).to_dict("records") if p.exists() else []
        except pd.errors.EmptyDataError:
            return []

    def set_window_meta(self, meta: dict[str, np.ndarray]) -> None:
        """Row metadata of the SHAP arrays (window ID, subject, group, true class, target class), saved in
        the npz as meta_*; on --resume it must equal what was saved."""
        for name, v in meta.items():
            if name in self.seg and not np.array_equal(self.seg[name], v):
                raise RuntimeError(f"{self.npz.name}: {name} differs from the saved run; cannot resume")
            self.seg[name] = v

    def add(self, table: str, rows: list[dict]) -> None:
        self.rows[table].extend(rows)

    def save(self) -> None:
        for t, p in self.paths.items():
            R.write_csv(p, pd.DataFrame(self.rows[t]))
        tmp = self.npz.with_name(self.npz.stem + ".tmp.npz")
        np.savez_compressed(tmp, **self.seg)
        tmp.replace(self.npz)


def window_meta(ctx: Context) -> dict[str, np.ndarray]:
    """The explain windows in SHAP-array row order (inner_val, general, older; subjects sorted, pick order)."""
    return {"meta_window_id": np.asarray(ctx.ids_all, dtype=str), "meta_subject": np.asarray(ctx.subj_all, dtype=str),
            "meta_group": np.asarray(ctx.group_all, dtype=str), "meta_class_index": np.asarray(ctx.label_all, np.int64),
            "meta_target_class": np.asarray(ctx.target_all, np.int64),
            "meta_walking_aid_inferred": np.isin(ctx.subj_all, WALKING_AID_INFERRED)}


def assess(ctx: Context, store: FoldStore, net, entry: dict, meta: dict, with_perf: bool = True,
           A: np.ndarray | None = None) -> dict:
    """SHAP + every per-subject metric of one model; appends the rows to the store (not saved yet)."""
    from lci.metric import aggregate_time_segments

    t0 = time.perf_counter()
    if A is None:
        A = ctx.shap(net, entry["model_id"])
    shap_s = time.perf_counter() - t0
    s_rows, c_rows, g_rows = explanation_metrics(ctx.A_ref, A, ctx.subj_all, ctx.group_all, ctx.label_all,
                                                 min_class_windows=ctx.S.get("min_class_windows",
                                                                             GRID["min_class_windows"]))
    perf, pooled_perf = ctx.eval_rows(net) if with_perf else ({}, {})
    faith = ctx.faith_rows(net, A) if with_perf else {}
    key = {k: entry.get(k) for k in ("model_id", "method", "family", "scenario", "level", "sparsity", "bits")}
    common = {"fold": ctx.k, **key, "machine_name": ctx.args.machine_name}
    rows = []
    for r in s_rows:
        ceil = ctx.ceilings[r["group"]]
        rows.append({**common, **r, "walking_aid_inferred": r["subject"] in WALKING_AID_INFERRED,
                     "retention": retention(r["lci"], ctx.tau, ceil), "tau_k": ctx.tau, "ceiling_group": ceil,
                     **perf.get(r["subject"], {}), **faith.get(r["subject"], {}), **meta.get("subject_extra", {})})
    store.add("per-subject-metrics", rows)
    store.add("per-subject-class-lci", [{**common, **r} for r in c_rows])
    store.add("per-group-class-lci", [{**common, **r} for r in g_rows])
    store.seg[entry["model_id"]] = aggregate_time_segments(A).astype(np.float32)
    by_g = pd.DataFrame(rows).groupby("group")
    summary = {f"lci_{g}": float(d.lci.mean()) for g, d in by_g}
    summary.update({f"shap_corr_{g}": float(d.shap_corr.mean()) for g, d in by_g})
    summary.update(pooled_perf)
    summary["n_skipped_total"] = int(sum(r["n_skipped"] for r in s_rows))
    summary["shap_seconds"] = shap_s
    summary["assess_seconds"] = time.perf_counter() - t0
    return summary


def run_fold(args, k: int, harth: dict, test: dict, folds, frozen: dict) -> None:
    out = GRID_SMOKE_DIR if args.smoke else GRID_DIR
    ckpt_dir = GRID_SMOKE_CKPT_DIR if args.smoke else GRID_CKPT_DIR
    machine = args.machine_name
    store = FoldStore(out, k, machine, args.resume, args.smoke)
    log = out / "logs" / f"{store.base}-{R.stamp()}.log"
    with R.tee_to(log):
        dev = R.device_info(args.device)
        t_fold = time.perf_counter()
        print(f"=== compression grid fold {k} | machine {machine} | device {args.device} {dev['gpu_name']} | torch "
              f"{dev['torch_version']} | started {R.now()} | {' '.join(sys.argv)}"
              + (f" | resuming, {len(store.done)} models done" if store.done else ""))
        ctx = Context(args, k, harth, test, folds, frozen)
        store.set_window_meta(window_meta(ctx))
        ctx.prepare_training()
        S = ctx.S
        epochs = S.get("epochs", GRID["epochs"])
        levels = tuple(args.levels) if args.levels else GRID["levels"]
        plan = grid_plan(methods=tuple(args.methods), levels=levels, ptq=() if args.no_ptq else tuple(PTQ))
        model_common = {"fold": k, "tau_k": ctx.tau, "tau_source": ctx.tau_src,
                        **{f"ceiling_{g}": v for g, v in ctx.ceilings.items()}, "machine_name": machine,
                        "device": args.device, "gpu_name": dev["gpu_name"]}
        size0 = size_record(ctx.net0)

        def finish(entry, net, summary, extra):
            rec = size_record(net, extra.pop("quantized", None))
            row = {**{k_: entry.get(k_) for k_ in ("model_id", "method", "family", "scenario", "level", "sparsity",
                                                    "bits", "score", "recipe", "park_term", "ptq_variant")},
                   **extra, **{k_: v for k_, v in rec.items() if k_ != "per_layer_sparsity"},
                   "per_layer_sparsity": json.dumps(rec["per_layer_sparsity"]), **summary, **model_common}
            store.add("models", [row])
            store.done.add(entry["model_id"])
            store.save()
            print(f"  {entry['model_id']:<32} LCI general {summary.get('lci_general', np.nan):.4f}, older "
                  f"{summary.get('lci_older', np.nan):.4f}, inner val {summary.get('lci_inner_val', np.nan):.4f}; "
                  f"acc general {summary.get('accuracy_pooled_general', np.nan):.4f}, older "
                  f"{summary.get('accuracy_pooled_older', np.nan):.4f}; n_nonzero {rec['n_nonzero']}; "
                  f"{summary['assess_seconds']:.0f} s")

        # ---- references: the original model, then the noise-floor models (per-class noise reference)
        if "original" not in store.done:
            s = assess(ctx, store, ctx.net0, plan[0], {}, A=ctx.A_ref)
            finish(plan[0], ctx.net0, s, {"shap_seconds_original": ctx.times["shap_original"]})
        nf_ckpt = NOISE_FLOOR_SMOKE_CKPT_DIR if args.smoke else NOISE_FLOOR_CKPT_DIR
        for kind, seeds in (("shap", SHAP_SEEDS), ("retrain", RETRAIN_SEEDS)):
            for sd in seeds:
                mid = f"noise_{kind}_seed{sd}"
                if mid in store.done:
                    continue
                entry = {"model_id": mid, "method": mid, "family": "noise_ref", "scenario": "reference",
                         "level": "none", "sparsity": np.nan, "bits": np.nan}
                if kind == "shap":
                    net, A = ctx.net0, ctx.shap(ctx.net0, mid, shap_seed=sd, noise_floor_tag=True)
                else:
                    p = nf_ckpt / f"har-noise-floor-tinyhar-fold{k}-seed{sd}.pt"
                    if not p.exists():
                        print(f"  {mid}: {p.name} not on this machine; per-class retrain-noise reference skipped")
                        continue
                    ck = torch.load(p, map_location="cpu", weights_only=False)
                    net = build_model("tinyhar", ck["architecture"], ck["norm_mean"], ck["norm_std"])
                    net.load_state_dict(ck["state_dict"])
                    net = net.to(ctx.dev).eval()
                    A = ctx.shap(net, mid, noise_floor_tag=True)
                s = assess(ctx, store, net, entry, {"subject_extra": {"seed": sd}}, with_perf=False, A=A)
                finish(entry, net, s, {"seed": sd})

        # ---- PTQ and pruning
        starts, nnz, level_now = {}, {}, None
        for entry in plan[1:]:
            mid = entry["model_id"]
            if entry["family"] == "pruning" and entry["sparsity"] != level_now:
                starts, level_now = {}, entry["sparsity"]
            if entry["family"] == "ptq":
                if mid in store.done:
                    continue
                fn = ptq_symmetric if entry["ptq_variant"] == "symmetric" else ptq_tulla
                net, quantized = fn(ctx.net0, entry["bits"])
                net = net.to(ctx.dev).eval()
                s = assess(ctx, store, net, entry, {})
                finish(entry, net, s, {"quantized": quantized})
                continue
            lv, score = entry["sparsity"], entry["score"]
            if score not in starts:
                if score not in ctx.scores:
                    ctx.scores[score] = SCORERS[score](ctx)
                starts[score] = prune_by_score(ctx.net0, ctx.scores[score], lv)
            start = starts[score]
            rec = size_record(start)
            nnz[(lv, score)] = rec["n_nonzero"]
            if len({v for (l_, _), v in nnz.items() if l_ == lv}) != 1:
                raise RuntimeError(f"n_nonzero differs across scores at {lv:.0%}: {nnz}")
            if not head_unpruned(start):
                raise RuntimeError("the classification head was pruned (it has a mask or a zeroed weight)")
            if mid in store.done:
                continue
            if entry["scenario"] == "no_finetune":
                s = assess(ctx, store, start, entry, {})
                finish(entry, start, s, {"epochs_run": 0, "head_unpruned": head_unpruned(start), "head_check": HEAD_CHECK})
                continue
            m = METHODS[entry["method"]]
            print(f"\n=== fold {k} {mid}: start {score}, {m.recipe.upper()}" + (" + gamma_P L_AM" if m.park else "")
                  + f", E_ft {epochs}, lr {GRID['lr']:g}")
            net = copy.deepcopy(start)
            extra = (make_extra_loss("park", X=ctx.Xtr, target=ctx.target_tr, reference=ctx.park_ref, baselines=ctx.bg,
                                     gamma=GRID["park_gamma_p"], m=S.get("park_m", GRID["park_m"]),
                                     seed=1 + GRID["seed"]) if m.park else None)
            info = finetune(net, ctx.tr, ctx.va, epochs=epochs, lr=GRID["lr"], seed=GRID["seed"], device=ctx.dev,
                            extra_loss=extra, kd_teacher_logits=ctx.kd_logits if m.recipe == "kd" else None)
            if info["epochs_run"] != epochs:
                raise RuntimeError(f"{mid}: ran {info['epochs_run']} epochs, expected {epochs}")
            r2 = size_record(net)
            if r2["n_nonzero"] != rec["n_nonzero"]:
                raise RuntimeError(f"{mid}: n_nonzero changed in fine-tuning ({rec['n_nonzero']} -> {r2['n_nonzero']})")
            store.add("finetune-history", [{"fold": k, "model_id": mid, **h} for h in info["history"]])
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            cp = ckpt_dir / f"{store.base}-{mid}.pt"
            torch.save({"state_dict": {k_: v.cpu() for k_, v in net.state_dict().items()},
                        "prune_masks": {k_: v.cpu() for k_, v in net.prune_masks.items()}, "entry": entry}, cp)
            s = assess(ctx, store, net, entry, {})
            finish(entry, net, s, {"epochs": epochs, "epochs_run": info["epochs_run"], "lr": GRID["lr"],
                                   "seed": GRID["seed"], "gamma_p": GRID["park_gamma_p"] if m.park else None,
                                   "park_m": S.get("park_m", GRID["park_m"]) if m.park else None,
                                   "finetune_seconds": info["seconds"],
                                   "val_macro_f1_pooled7_after_ft": info["val_macro_f1_pooled7"],
                                   "head_unpruned": head_unpruned(net), "head_check": HEAD_CHECK,
                                   "checkpoint": str(cp)})

        checks = fold_checks(pd.DataFrame(store.rows["models"]), pd.DataFrame(store.rows["per-subject-metrics"]))
        wall = time.perf_counter() - t_fold
        R.write_json(store.config, {"command_line": sys.argv, "machine_name": machine, "device": dev, **R.git_info(),
                                    "code_sha256": R.code_sha256(), "fold": k, "smoke": args.smoke,
                                    "smoke_sizes": S or None, "grid": GRID, "methods": list(args.methods),
                                    "levels": list(levels), "epochs": epochs, "nsamples": ctx.nsamples,
                                    "kd": KD,
                                    "tau_k": ctx.tau, "tau_source": ctx.tau_src, "ceilings": ctx.ceilings,
                                    "noise_floor_machine": ctx.nf_machine, "weights": str(ctx.ckpt),
                                    "n_explain": {g: len(ctx.sets[g][1]) for g in EXPLAIN_GROUPS},
                                    "n_eval_windows": {g: len(w) for g, w in ctx.eval.items()},
                                    "original_n_params": size0["n_params"], "seconds_fixed": ctx.times,
                                    "acceptance_checks": checks, "tuning": "none in part A (no tuned method, no gate)",
                                    "seconds_wall_this_invocation": wall, "finished_at": R.now()})
        print("\nacceptance checks: " + json.dumps(checks, indent=1, default=str))
        print(f"fold {k} done in {wall:.1f} s")
        for p in (*store.paths.values(), store.npz, store.config, log):
            print(f"saved {p}")


def fold_checks(models: pd.DataFrame, per_subject: pd.DataFrame) -> dict:
    """Step 9 acceptance, per fold: identical n_nonzero per level across pruning models, head unpruned,
    E_ft epochs, PTQ-8 rho > 0.99 per group and accuracy within 0.5 pt of the original, n_skipped."""
    out = {}
    pr = models[models.family == "pruning"]
    out["n_nonzero_identical_per_level"] = {str(lv): bool(g.n_nonzero.nunique() == 1) for lv, g in pr.groupby("level")}
    if "head_unpruned" in pr:
        out["head_unpruned_all"] = bool(pr.head_unpruned.fillna(True).astype(bool).all())
    ft = pr[pr.scenario == "finetune"]
    if len(ft):
        out["epochs_run_equal_e_ft"] = bool((ft.epochs_run == ft.epochs).all())
    ps = per_subject
    orig = ps[ps.model_id == "original"]
    for mid in ("ptq8", "ptq_tulla8"):
        q = ps[ps.model_id == mid]
        if len(q) and len(orig):
            rho = q.groupby("group").shap_corr.mean().to_dict()
            om = models.set_index("model_id")
            acc = {g: float(om.loc[mid, f"accuracy_pooled_{g}"] - om.loc["original", f"accuracy_pooled_{g}"])
                   for g in EXPLAIN_GROUPS if f"accuracy_pooled_{g}" in models}
            out[f"{mid}_rho_by_group"] = rho
            out[f"{mid}_rho_above_0.99_all_groups"] = bool(all(v > 0.99 for v in rho.values()))
            out[f"{mid}_accuracy_delta_pooled_by_group"] = acc
            out[f"{mid}_accuracy_within_0.5pt"] = bool(all(abs(v) <= 0.005 for v in acc.values()))
    out["n_skipped_by_model"] = models.set_index("model_id").n_skipped_total.astype(int).to_dict()
    return out


# ------------------------------------------------------------------------------ analysis (pure)

FAMILY_ORDER = {"original": 0, "noise_ref": 1, "ptq": 2, "pruning": 3}
SCENARIO_ORDER = {"original": 0, "reference": 0, "no_finetune": 1, "finetune": 2}


def sort_settings(df: pd.DataFrame) -> pd.DataFrame:
    """Rows in a fixed order: original, noise references, PTQ, then pruning by level, scenario, method."""
    if df.empty or not set(SETTING) <= set(df.columns):
        return df
    lvl = df.level.astype(str).str.extract(r"(\d+)")[0].astype(float).fillna(-1)
    key = pd.DataFrame({"f": df.family.map(FAMILY_ORDER).fillna(9),
                        "b": np.where(df.family == "ptq", -lvl, lvl),
                        "s": df.scenario.map(SCENARIO_ORDER).fillna(9),
                        "m": [_method_order(list(df.method.unique())).index(m) for m in df.method]}, index=df.index)
    extra = [c for c in ("class_index", "metric", "group", "unit", "threshold", "population", "comparison") if c in df]
    order = pd.concat([key, df[extra]], axis=1).sort_values(["f", "b", "s", "m", *extra], kind="stable").index
    return df.loc[order].reset_index(drop=True)


def _is_test(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.group.isin(TEST_GROUPS)]


def pool_subjects(per_subject: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    """Test subjects only; one row per (setting, subject): a HARTH subject from its single test fold, a
    HAR70+ subject averaged over its fold models [plan 2.8]. Adds `subgroup` (general,
    walking_aid_inferred, other_older)."""
    df = _is_test(per_subject)
    metrics = [m for m in metrics if m in df.columns]
    out = pool_over_folds(df, metrics, keys=SETTING)
    out["subgroup"] = [subgroup(g, s) for g, s in zip(out.group, out.subject)]
    return out


def _settings(df: pd.DataFrame):
    return df.groupby(SETTING, sort=False, dropna=False)


def group_summary(pooled: pd.DataFrame, metrics: Sequence[str], n_boot: int = N_BOOT, seed: int = BOOT_SEED
                  ) -> pd.DataFrame:
    """Per setting and metric: subject means per group with bootstrap CIs, the older - general gap
    (bootstrap CI, Mann-Whitney U) and walking-aid (inferred) - other older (exploratory)."""
    rows = []
    for key, d in _settings(pooled):
        for m in metrics:
            if m not in d:
                continue
            v = {g: d.loc[d.group == g, m].dropna().to_numpy() for g in TEST_GROUPS}
            v.update({g: d.loc[d.subgroup == g, m].dropna().to_numpy() for g in ("walking_aid_inferred", "other_older")})
            r = dict(zip(SETTING, key), metric=m)
            for g, x in v.items():
                ci = bootstrap_mean_ci(x, n_boot, seed)
                r.update({f"mean_{g}": ci["mean"], f"ci_low_{g}": ci["ci_low"], f"ci_high_{g}": ci["ci_high"],
                          f"n_{g}": ci["n"]})
            for name, a, b in (("older_minus_general", "older", "general"),
                               ("walking_aid_minus_other_older", "walking_aid_inferred", "other_older")):
                if len(v[a]) and len(v[b]):
                    gap = bootstrap_group_gap_ci(v[a], v[b], n_boot=n_boot, seed=seed)
                    mw = mann_whitney(v[a], v[b]) if len(v[a]) > 0 and len(v[b]) > 0 else {}
                    pre = "gap" if name == "older_minus_general" else "wa_gap"
                    r.update({f"{pre}_{name}" if pre == "gap" else pre: gap["gap"], f"{pre}_ci_low": gap["ci_low"],
                              f"{pre}_ci_high": gap["ci_high"], f"{pre}_mw_p": mw.get("p"),
                              f"{pre}_mw_rank_biserial_r": mw.get("rank_biserial_r")})
            rows.append(r)
    return pd.DataFrame(rows)


def fa_lci_table(pooled: pd.DataFrame, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> pd.DataFrame:
    """Per setting: LCI_general, LCI_older, LCI_all (group-balanced and subject-weighted), the gap, the worst
    group and FA-LCI = min(LCI_older, LCI_general) with its bootstrap CI [plan 2.7-2.8]."""
    rows = []
    for key, d in _settings(pooled):
        o = d.loc[d.group == "older", "lci"].dropna().to_numpy()
        g = d.loc[d.group == "general", "lci"].dropna().to_numpy()
        if not (len(o) and len(g)):
            continue
        fa = bootstrap_fa_lci_ci(o, g, n_boot=n_boot, seed=seed)
        wa = d.loc[d.subgroup == "walking_aid_inferred", "lci"].dropna()
        rows.append({**dict(zip(SETTING, key)), "lci_general": g.mean(), "lci_older": o.mean(),
                     "lci_all": (g.mean() + o.mean()) / 2, "lci_all_subject_weighted": np.concatenate([o, g]).mean(),
                     "gap_older_minus_general": o.mean() - g.mean(), "worst_group": "older" if o.mean() < g.mean() else "general",
                     "fa_lci": fa["fa_lci"], "fa_lci_ci_low": fa["ci_low"], "fa_lci_ci_high": fa["ci_high"],
                     "lci_walking_aid_inferred": float(wa.mean()) if len(wa) else np.nan,
                     "lci_other_older": float(d.loc[d.subgroup == "other_older", "lci"].mean()),
                     "n_general": len(g), "n_older": len(o)})
    return pd.DataFrame(rows)


def _method_order(methods) -> list[str]:
    reg = list(METHODS) + [m for m in dict.fromkeys(m_.score for m_ in METHODS.values()) if m not in METHODS]
    return sorted(methods, key=lambda m: (reg.index(m) if m in reg else len(reg), m))


def wilcoxon_table(pooled: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    """Paired Wilcoxon per subject [plan 2.8]: (1) every pair of pruning methods within (scenario, level);
    (2) each fine-tuned method vs its no_finetune start at the same level. Populations: all test
    subjects (primary), general and older (secondary). Holm per (metric, comparison, scenario,
    population) over all levels and pairs. PTQ and references are described, not tested."""
    pr = pooled[pooled.family == "pruning"]
    rows = []

    def add(metric, comparison, scenario, level, ma, mb, da, db):
        for pop in ("all", *TEST_GROUPS):
            a = da if pop == "all" else da[da.group == pop]
            b = db if pop == "all" else db[db.group == pop]
            j = a[["subject", metric]].merge(b[["subject", metric]], on="subject", suffixes=("_a", "_b"))
            if len(j) < 2:
                continue
            r = paired_wilcoxon(j[f"{metric}_a"], j[f"{metric}_b"])
            rows.append({"metric": metric, "comparison": comparison, "scenario": scenario, "level": level,
                         "population": pop, "method_a": ma, "method_b": mb, **r})

    for metric in metrics:
        if metric not in pr:
            continue
        for (sc, lv), d in pr.groupby(["scenario", "level"], sort=False):
            for ma, mb in combinations(_method_order(d.method.unique()), 2):
                add(metric, "between_methods", sc, lv, ma, mb, d[d.method == ma], d[d.method == mb])
        ft, nft = pr[pr.scenario == "finetune"], pr[pr.scenario == "no_finetune"]
        for (lv, meth), d in ft.groupby(["level", "method"], sort=False):
            score = METHODS[meth].score if meth in METHODS else None
            s = nft[(nft.level == lv) & (nft.method == score)]
            if len(s):
                add(metric, "finetune_vs_no_finetune", "finetune", lv, meth, f"{score} (no_finetune)", d, s)
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = np.nan
        for _, idx in out.groupby(["metric", "comparison", "scenario", "population"]).groups.items():
            out.loc[idx, "p_holm"] = holm(out.loc[idx, "p"].to_numpy())
    return out


def safe_rate_table(per_subject: pd.DataFrame, tulla_tau: float = GRID["tulla_tau"]) -> pd.DataFrame:
    """Safe-explanation rate [plan 2.8] at tau_k (the fold's), the global tau = min_k tau_k and 0.85:
    unit fold_setting = share of (fold x level) settings with the group-mean LCI (or FA-LCI) >= tau;
    unit subject = share of (fold x level x subject) rows with LCI >= tau. Per (method, scenario), over
    all levels ("all") and per level."""
    df = _is_test(per_subject)
    tau_global = float(df.groupby("fold").tau_k.first().min())
    fs = (df.groupby(["fold", *SETTING], dropna=False)
          .apply(lambda d: pd.Series({"general": d.loc[d.group == "general", "lci"].mean(),
                                      "older": d.loc[d.group == "older", "lci"].mean(), "tau_k": d.tau_k.iloc[0]}),
                 include_groups=False).reset_index())
    fs["fa_lci"] = fs[["general", "older"]].min(axis=1)
    rows = []
    for thr in ("tau_k", "tau_global", "tulla_0.85"):
        t_fs = fs.tau_k if thr == "tau_k" else (tau_global if thr == "tau_global" else tulla_tau)
        t_ps = df.tau_k if thr == "tau_k" else (tau_global if thr == "tau_global" else tulla_tau)
        fs_ok = fs.assign(g=fs.general >= t_fs, o=fs.older >= t_fs, f=fs.fa_lci >= t_fs)
        ps_ok = df.assign(ok=df.lci >= t_ps)
        keys = ["method", "family", "scenario"]
        for lvl_mode in ("all", "level"):
            k2 = keys if lvl_mode == "all" else keys + ["level"]
            for key, d in fs_ok.groupby(k2, dropna=False):
                rows.append({**dict(zip(k2, key)), **({"level": "all"} if lvl_mode == "all" else {}), "unit": "fold_setting",
                             "threshold": thr, "tau_value": "per fold" if thr == "tau_k" else float(t_fs),
                             "safe_general": d.g.mean(), "safe_older": d.o.mean(), "safe_fa_lci": d.f.mean(),
                             "n": len(d)})
            for key, d in ps_ok.groupby(k2, dropna=False):
                rows.append({**dict(zip(k2, key)), **({"level": "all"} if lvl_mode == "all" else {}), "unit": "subject",
                             "threshold": thr, "tau_value": "per fold" if thr == "tau_k" else float(t_ps),
                             "safe_general": d.loc[d.group == "general", "ok"].mean(),
                             "safe_older": d.loc[d.group == "older", "ok"].mean(), "safe_fa_lci": np.nan, "n": len(d)})
    return pd.DataFrame(rows)


def per_class_table(class_rows: pd.DataFrame, group_class_rows: pd.DataFrame | None, n_boot: int = N_BOOT,
                    seed: int = BOOT_SEED) -> pd.DataFrame:
    """Where the drift happens: per setting and true activity class, the subject-mean LCI per group (each
    subject pooled over folds as in plan 2.8) with bootstrap CIs, the older - general gap (bootstrap CI,
    Mann-Whitney U), walking-aid (inferred) and other older; plus the pooled-window LCI per group (mean
    over folds) and, when the noise-floor references are present, their per-class LCI."""
    keys = SETTING + ["class_index", "class_name"]
    df = _is_test(class_rows)
    pooled = pool_over_folds(df, [c for c in ("lci", "shap_corr", "jaccard", "n_windows") if c in df], keys=keys)
    for c in ("shap_corr", "jaccard"):
        if c not in pooled:
            pooled[c] = np.nan
    pooled["subgroup"] = [subgroup(g, s) for g, s in zip(pooled.group, pooled.subject)]
    rows = []
    for key, d in pooled.groupby(keys, sort=False, dropna=False):
        r = dict(zip(keys, key))
        v = {g: d.loc[d.group == g, "lci"].dropna().to_numpy() for g in TEST_GROUPS}
        for g in TEST_GROUPS:
            ci = bootstrap_mean_ci(v[g], n_boot, seed)
            r.update({f"lci_{g}": ci["mean"], f"ci_low_{g}": ci["ci_low"], f"ci_high_{g}": ci["ci_high"], f"n_{g}": ci["n"],
                      f"rho_{g}": d.loc[d.group == g, "shap_corr"].mean(),
                      f"jaccard_{g}": d.loc[d.group == g, "jaccard"].mean()})
        for g in ("walking_aid_inferred", "other_older"):
            x = d.loc[d.subgroup == g, "lci"].dropna()
            r[f"lci_{g}"], r[f"n_{g}"] = (float(x.mean()) if len(x) else np.nan), len(x)
        if len(v["older"]) and len(v["general"]):
            gap = bootstrap_group_gap_ci(v["older"], v["general"], n_boot=n_boot, seed=seed)
            mw = mann_whitney(v["older"], v["general"])
            r.update({"gap_older_minus_general": gap["gap"], "gap_ci_low": gap["ci_low"], "gap_ci_high": gap["ci_high"],
                      "gap_mw_p": mw["p"], "gap_mw_rank_biserial_r": mw["rank_biserial_r"]})
        rows.append(r)
    out = pd.DataFrame(rows)
    if group_class_rows is not None and len(group_class_rows):
        gc = group_class_rows.groupby(keys + ["group"], dropna=False).lci.mean().unstack("group")
        gc.columns = [f"pooled_window_lci_{c}" for c in gc.columns]
        out = out.merge(gc.reset_index(), on=keys, how="left")
    ref = out[out.family == "noise_ref"]
    if len(ref):
        for kind in ("noise_shap", "noise_retrain"):
            rk = ref[ref.method.str.startswith(kind)].groupby("class_index")[[f"lci_{g}" for g in TEST_GROUPS]]
            agg = rk.mean() if kind == "noise_shap" else rk.min()
            agg.columns = [f"{kind}_{'mean' if kind == 'noise_shap' else 'min'}_{c}" for c in agg.columns]
            out = out.merge(agg.reset_index(), on="class_index", how="left")
        for g in TEST_GROUPS:
            col = f"noise_retrain_min_lci_{g}"
            if col in out:
                out[f"below_retrain_noise_{g}"] = out[f"lci_{g}"] < out[col]
    return sort_settings(out)


def stairs_table(per_subject: pd.DataFrame, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> pd.DataFrame:
    """Stairs F1 pooled over the subjects of a group (never per subject) [plan 2.7], with a bootstrap CI
    over subjects (resample subjects, pool their windows), plus pooled accuracy and 7-class macro-F1.
    A HAR70+ subject's windows are pooled over its fold models (confusion matrices summed)."""
    df = _is_test(per_subject)
    df = df[df.confusion7.notna()]
    rows = []
    for key, d in _settings(df):
        for grp, dg in (("general", d[d.group == "general"]), ("older", d[d.group == "older"]),
                        ("other_older", d[(d.group == "older") & ~d.subject.isin(WALKING_AID_INFERRED)])):
            if not len(dg):
                continue
            cms = np.stack([sum(parse_confusion(c) for c in x.confusion7) for _, x in dg.groupby("subject")])
            tot = cms.sum(0)
            f = f1_from_confusion(tot)
            idx = np.random.default_rng(seed).integers(0, len(cms), size=(n_boot, len(cms)))
            boot = cms[idx].sum(1)                                     # (n_boot, 7, 7)
            tp = np.stack([boot[:, c, c] for c in STAIRS], 1).astype(float)
            fp = np.stack([boot[:, :, c].sum(1) for c in STAIRS], 1) - tp
            fn = np.stack([boot[:, c, :].sum(1) for c in STAIRS], 1) - tp
            den = 2 * tp + fp + fn
            bf = np.divide(2 * tp, den, out=np.full_like(tp, np.nan), where=den > 0)
            present = [c for c in range(7) if tot[c].sum() > 0 or tot[:, c].sum() > 0]
            r = {**dict(zip(SETTING, key)), "group": grp, "n_subjects": len(cms), "n_windows": int(tot.sum()),
                 "accuracy_pooled": float(np.trace(tot) / tot.sum()), "macro_f1_pooled7": float(f[present].mean())}
            for j, c in enumerate(STAIRS):
                nm = CLASS_NAMES[c]
                has = tot[c].sum() > 0
                r[f"{nm}_f1_pooled"] = float(f[c]) if has else np.nan
                r[f"{nm}_ci_low"], r[f"{nm}_ci_high"] = ((float(np.nanquantile(bf[:, j], 0.025)),
                                                          float(np.nanquantile(bf[:, j], 0.975))) if has else (np.nan, np.nan))
                r[f"n_{nm}_windows"] = int(tot[c].sum())
            r["stairs_f1_pooled_mean"] = float(np.nanmean([r[f"{CLASS_NAMES[c]}_f1_pooled"] for c in STAIRS]))
            rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ analysis (files)

METRICS = ["lci", "shap_corr", "jaccard", "lci768", "lci_k16", "retention", "n_skipped", "accuracy",
           "non_stair_macro_f1", "aopc", "drop_k5", "random_aopc", "random_drop_k5"]
TESTED = ["lci", "non_stair_macro_f1", "aopc"]


def load_fold_files(src: Path, use_machine: dict[int, str]) -> tuple[dict[str, pd.DataFrame], dict[int, str]]:
    """Concatenate the per-fold tables; a fold present from two machines needs --use-machine k=machine."""
    found = {}
    for p in sorted(src.glob("har-grid-tinyhar-fold*-models.csv")):
        stem = p.name[len("har-grid-tinyhar-fold"):-len("-models.csv")]
        k, machine = stem.split("-", 1)
        found.setdefault(int(k), []).append(machine)
    chosen = {}
    for k, ms in sorted(found.items()):
        if k in use_machine:
            if use_machine[k] not in ms:
                raise FileNotFoundError(f"fold {k}: no files from {use_machine[k]} (found {ms})")
            chosen[k] = use_machine[k]
        elif len(ms) == 1:
            chosen[k] = ms[0]
        else:
            raise RuntimeError(f"fold {k} has files from {ms}; choose one with --use-machine {k}=<machine>")
    tables = {}
    for t in FoldStore.TABLES:
        parts = []
        for k, m in chosen.items():
            p = src / f"har-grid-tinyhar-fold{k}-{m}-{t}.csv"
            if p.exists() and p.stat().st_size > 2:
                parts.append(pd.read_csv(p, dtype={"subject": str}))
        tables[t] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return tables, chosen


def analyze(args) -> None:
    src = GRID_SMOKE_DIR if args.smoke else GRID_DIR
    out = GRID_ANALYSIS_SMOKE_DIR if args.smoke else GRID_ANALYSIS_DIR
    n_boot = args.n_boot
    use = {} if args.smoke else dict(FOLD_MACHINE_DEFAULT)
    use.update({int(a.split("=")[0]): a.split("=", 1)[1] for a in (args.use_machine or [])})
    T, chosen = load_fold_files(src, use)
    ps, models = T["per-subject-metrics"], T["models"]
    if ps.empty:
        raise FileNotFoundError(f"no per-fold grid files in {src}")
    t0 = time.perf_counter()
    pooled = pool_subjects(ps, METRICS)
    files = {
        "per-subject-pooled-over-folds": pooled,
        "group-summary-gaps-bootstrap-ci": group_summary(pooled, METRICS, n_boot),
        "fa-lci-bootstrap-ci": fa_lci_table(pooled, n_boot),
        "method-comparisons-wilcoxon-holm": wilcoxon_table(pooled, TESTED),
        "safe-explanation-rate": safe_rate_table(ps),
        "per-class-lci-drift": per_class_table(T["per-subject-class-lci"], T["per-group-class-lci"], n_boot),
        "stairs-f1-pooled-per-group": stairs_table(ps, n_boot),
    }
    iv = ps[ps.group == "inner_val"]
    files["inner-val-lci-per-fold"] = (iv.groupby(["fold", *SETTING], dropna=False)
                                       [["lci", "shap_corr", "jaccard", "retention", "accuracy", "non_stair_macro_f1"]]
                                       .mean().reset_index())
    checks = {str(k): fold_checks(models[models.fold == k], ps[ps.fold == k]) for k in chosen}
    for name, df in files.items():
        if name != "method-comparisons-wilcoxon-holm":
            df = files[name] = sort_settings(df)
        R.write_csv(out / f"har-grid-a-{name}.csv", df)
    R.write_json(out / "har-grid-a-analysis-config.json",
                 {"folds_and_machines": chosen, "n_boot": n_boot, "boot_seed": BOOT_SEED, "metrics": METRICS,
                  "tested_metrics": TESTED, "tau_k_by_fold": ps.groupby("fold").tau_k.first().to_dict(),
                  "tau_global": float(ps.groupby("fold").tau_k.first().min()), "acceptance_checks": checks,
                  "smoke": args.smoke, **R.git_info(), "code_sha256": R.code_sha256(), "analyzed_at": R.now(),
                  "seconds": time.perf_counter() - t0})
    fa = files["fa-lci-bootstrap-ci"]
    show = ["method", "scenario", "level", "lci_general", "lci_older", "gap_older_minus_general", "fa_lci",
            "fa_lci_ci_low", "fa_lci_ci_high", "lci_walking_aid_inferred"]
    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
        print(fa[[c for c in show if c in fa]].to_string(index=False))
    print(f"\nfolds {chosen}; analysis {time.perf_counter() - t0:.1f} s")
    for name in files:
        print(f"saved {out / f'har-grid-a-{name}.csv'}")


# ------------------------------------------------------------------------------ main

HEAD_CHECK = "no mask and no zeroed weight in the head (compression.head_unpruned, 2026-11-22)"


def recheck_head(args) -> None:
    """--recheck-head (2026-11-22): the models files written before the fix compared the head with the
    original model's, which is always False after fine-tuning (the head trains). Recompute head_unpruned
    for this machine's folds: fine-tuned rows from their saved checkpoints, no-fine-tuning starts by
    re-pruning (the head is never in the prunable set). Only the head_unpruned / head_check columns of
    the models file change; the file is rewritten in place."""
    out = GRID_SMOKE_DIR if args.smoke else GRID_DIR
    frozen = json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))
    for k in args.folds:
        p = out / f"har-grid-tinyhar-fold{k}-{args.machine_name}-models.csv"
        df = pd.read_csv(p)
        net0, _ = load_fold_model(k, frozen["architecture"])
        res = {}
        for i, r in df[df.family == "pruning"].iterrows():
            if r.scenario == "no_finetune":
                net = prune_by_score(net0, l1_scores(net0), float(r.sparsity))   # head excluded by scope
            else:
                ck = torch.load(r.checkpoint, map_location="cpu", weights_only=False)
                net = copy.deepcopy(net0)
                net.load_state_dict(ck["state_dict"])
                net.prune_masks = ck["prune_masks"]
            res[i] = head_unpruned(net)
        for i, v in res.items():
            df.loc[i, "head_unpruned"] = v
            df.loc[i, "head_check"] = HEAD_CHECK
        R.write_csv(p, df)
        print(f"fold {k}: head_unpruned recomputed for {len(res)} pruning rows, all True: {all(res.values())}; "
              f"rewrote {p}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--folds", type=int, nargs="+")
    mode.add_argument("--analyze", action="store_true")
    ap.add_argument("--recheck-head", action="store_true",
                    help="with --folds: recompute head_unpruned in this machine's finished models files")
    ap.add_argument("--methods", nargs="+", default=list(PART_A), help="registered pruning methods to run")
    ap.add_argument("--levels", type=float, nargs="+", default=None, choices=GRID["levels"])
    ap.add_argument("--no-ptq", action="store_true")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--machine-name", default=R.machine_name())
    ap.add_argument("--noise-floor-machine", default=None,
                    help="whose noise-floor files give tau_k and the ceilings (default: --machine-name)")
    ap.add_argument("--use-machine", nargs="*", default=None, help="--analyze: k=machine for folds run on two machines (default: fold 3 = uni-2080)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.analyze:
        return analyze(args)
    if args.recheck_head:
        return recheck_head(args)
    unknown = [m for m in args.methods if m not in METHODS]
    if unknown:
        raise SystemExit(f"unknown methods {unknown}; registered: {list(METHODS)}")
    frozen = json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))
    assert frozen["model"] == "tinyhar", frozen["model"]
    harth, test = load_harth_windows(), load_test_windows()
    counts = pd.Series(harth["harth_test"].subject).value_counts().to_dict()
    folds = load_folds(harth_test_counts={s: int(c) for s, c in counts.items()})
    t0 = time.perf_counter()
    for k in args.folds:
        run_fold(args, k, harth, test, folds, frozen)
    print(f"\nall folds {args.folds}: {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
