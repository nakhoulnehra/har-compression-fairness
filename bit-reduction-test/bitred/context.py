"""One fold's fixed inputs. HARTH only: HAR70+ and the fold's HARTH test subjects are never loaded.

    original model   the frozen TinyHAR of the fold (seed 0, model-selection checkpoint)
    calibration      design data, the fold's HARTH TRAINING windows only: round 4's calibration set (<= 10 windows
                     per (training subject, class), seed 46; har.run_xgp_pilot4.calibration_table) for the KL and
                     Hessian sensitivities, and a class-stratified subset of it (<= 20 per class, seed 47) for the
                     SHAP-based LCI sensitivity
    evaluation       the fold's 3 inner-validation subjects, never used for design: F1 / accuracy on every inner-val
                     window; LCI on round 3/4's explain windows (100 per subject, select_explain_windows)
    SHAP             GradientExplainer, nsamples 200, SHAP seed 0, target = the original model's predicted class,
                     background = the fold's 100 HARTH training windows
    tau_k, ceiling   from the committed noise-floor files (reporting only; no rule uses them)
"""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch

from har.attributions import shap_cached, subject_lci
from har.data import find_windows, load_harth_windows, require_harth, window_ids
from har.folds import load_folds, select_explain_windows
from har.paths import MODEL_SELECTION_DIR, RESULTS_DIR as REPO_RESULTS_DIR, SHAP_CACHE_DIR
from har.run_model_selection import subsample
from har.run_noise_floor import fold_background
from har.run_sanity_walking_aid import load_fold_model
from har.run_xgp_pilot import read_tau
from har.run_xgp_pilot3 import read_ceiling
from har.run_xgp_pilot4 import calibration_table
from har.train import pooled_macro_f1, predict_proba
from har.xgp4 import calibration_pick

FROZEN_CONFIG = MODEL_SELECTION_DIR / "har-frozen-model-config.json"
ROUND4_DIR = REPO_RESULTS_DIR / "har-xgp-pilot" / "round4"

DESIGN = {"explain_per_subject": 100, "lci_cal_per_class": 20, "lci_cal_seed": 47, "nsamples": 200, "shap_seed": 0,
          "k": 10, "omega": 0.5}
SMOKE = {"explain_per_subject": 3, "lci_cal_per_class": 2, "nsamples": 20, "n_train": 600, "n_val": 200}
# whose committed noise-floor files give tau_k and the ceiling (as the 9A analysis: fold 3 = uni-2080)
NF_MACHINE = {0: "nakhoul-3050", 1: "nakhoul-3050", 2: "uni-2080", 3: "uni-2080", 4: "uni-2080"}


def cfg(key: str, smoke: bool):
    return SMOKE[key] if smoke and key in SMOKE else DESIGN[key]


def ids_sha(ids) -> str:
    return hashlib.sha256("|".join(map(str, ids)).encode()).hexdigest()


def lci_calibration_subset(labels: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    """Indices into the calibration set: up to per_class windows per class (one group), class then pick order."""
    return calibration_pick(labels, np.zeros(len(labels), dtype=int), per_class, seed)


def select_windows(k: int, smoke: bool, harth: dict | None = None) -> dict:
    """Every window choice of fold k (no model needed). HARTH training and inner-val windows only."""
    harth = harth if harth is not None else load_harth_windows()
    counts = pd.Series(harth["harth_test"].subject).value_counts().to_dict()
    fold = load_folds(harth_test_counts={s: int(c) for s, c in counts.items()})[k]
    tr = harth["harth_train"].for_subjects(fold.train)
    va = harth["harth_test"].for_subjects(fold.val)
    require_harth(tr, va)
    assert not (set(tr.subjects) | set(va.subjects)) & (set(fold.test_general) | set(fold.test_older))
    if smoke:
        tr = subsample(tr, SMOKE["n_train"], 0)
    # evaluation: explain windows (subjects sorted, then pick order) and the F1 windows
    ex = select_explain_windows(va, "harth_test", per_subject=cfg("explain_per_subject", smoke))
    ex = np.concatenate([ex[va.subject[ex] == s] for s in sorted(set(va.subject[ex].tolist()))])
    f1_idx = np.arange(len(va))
    if smoke:
        f1_idx = np.sort(np.random.default_rng(1).permutation(f1_idx)[:SMOKE["n_val"]])
    # calibration: training windows only
    cal = calibration_table(tr, smoke)
    cidx = find_windows(tr, "harth_train", cal.window_id.tolist())
    sub = lci_calibration_subset(tr.y[cidx], cfg("lci_cal_per_class", smoke), DESIGN["lci_cal_seed"])
    ex_ids = window_ids(va, "harth_test")[ex].tolist()
    cal_ids = cal.window_id.tolist()
    assert set(va.subject[ex].tolist()) == set(fold.val)
    assert not set(cal_ids) & set(ex_ids)
    return {"harth": harth, "fold": fold, "tr": tr, "va": va, "ex": ex, "ex_ids": ex_ids, "f1_idx": f1_idx,
            "cal_idx": cidx, "cal_ids": cal_ids, "lcal_sub": sub}


def window_hashes(w: dict) -> dict:
    return {"explain": ids_sha(w["ex_ids"]), "f1": ids_sha(window_ids(w["va"], "harth_test")[w["f1_idx"]].tolist()),
            "calibration": ids_sha(w["cal_ids"]), "lci_calibration": ids_sha([w["cal_ids"][i] for i in w["lcal_sub"]]),
            "n_explain": len(w["ex_ids"]), "n_f1": int(len(w["f1_idx"])), "n_calibration": len(w["cal_ids"]),
            "n_lci_calibration": int(len(w["lcal_sub"])), "inner_val_subjects": list(w["fold"].val),
            "n_train_subjects": len(w["fold"].train)}


class FoldContext:
    def __init__(self, k: int, device: str, smoke: bool, noise_floor_machine: str | None = None):
        self.k, self.dev, self.smoke = k, device, smoke
        self.nsamples = cfg("nsamples", smoke)
        self.cache = SHAP_CACHE_DIR / "bit-reduction-test" / ("smoke" if smoke else "")
        frozen = json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))
        assert frozen["model"] == "tinyhar", frozen["model"]
        w = select_windows(k, smoke)
        self.windows = w
        self.hashes = window_hashes(w)
        fold, tr, va = w["fold"], w["tr"], w["va"]
        self.fold, self.tr = fold, tr

        # ---- evaluation (inner val)
        ex = w["ex"]
        self.ex_ids, self.ex_subj = w["ex_ids"], va.subject[ex]
        self.X_ex = torch.from_numpy(va.X[ex]).to(device)
        self.va = va.select(w["f1_idx"])

        # ---- calibration (training windows)
        self.cal_matches_round4 = None
        committed = ROUND4_DIR / f"har-xgp-pilot4-tinyhar-fold{k}-calibration-train-windows-by-subject-class.csv"
        if committed.exists() and not smoke:
            saved = pd.read_csv(committed, dtype={"subject": str})
            self.cal_matches_round4 = saved.window_id.tolist() == w["cal_ids"]
            if not self.cal_matches_round4:
                raise RuntimeError(f"fold {k}: calibration windows differ from round 4's committed {committed.name}")
        cw = tr.select(w["cal_idx"])
        require_harth(cw)
        sub = w["lcal_sub"]
        self.cal_ids = w["cal_ids"]
        self.X_cal = torch.from_numpy(cw.X).to(device)
        self.y_cal = torch.from_numpy(cw.y.astype(np.int64)).to(device)
        self.lcal_ids = [self.cal_ids[i] for i in sub]
        self.lcal_y = cw.y[sub]
        self.X_lcal = self.X_cal[torch.from_numpy(sub).to(device)]

        # ---- model, background, targets, noise floor
        net0, self.ckpt = load_fold_model(k, frozen["architecture"])
        self.net0 = net0.to(device).eval()
        self.bg = fold_background(w["harth"], k).to(device)
        self.target_ex = predict_proba(self.net0, va.X[ex], device).argmax(1)
        self.target_lcal = predict_proba(self.net0, cw.X[sub], device).argmax(1)
        nf = noise_floor_machine or NF_MACHINE.get(k)
        try:
            self.tau, self.tau_src = read_tau(k, smoke, nf)
            self.ceiling, self.ceiling_src = read_ceiling(k, smoke, nf)
        except FileNotFoundError:
            if not smoke:
                raise
            self.tau, self.tau_src, self.ceiling, self.ceiling_src = 0.6, "smoke default", 0.9, "smoke default"
        t0 = time.perf_counter()
        self.A_ref_ex = self.shap_eval(self.net0, "original")
        self.A_ref_lcal = self.shap_cal(self.net0, "original")
        self.shap_original_seconds = time.perf_counter() - t0
        print(f"fold {k}: train {len(tr)} windows ({len(fold.train)} subjects); calibration {len(cw)} training windows, "
              f"LCI-calibration subset {len(sub)}; inner val {list(fold.val)}: F1 windows {len(self.va)}, explain "
              f"{len(self.ex_ids)}; tau_k {self.tau:.4f}, ceiling {self.ceiling:.4f}; original SHAP "
              f"{self.shap_original_seconds:.1f} s")

    def shap_eval(self, net, tag: str) -> np.ndarray:
        return shap_cached(net, self.bg, self.X_ex, self.target_ex, self.ex_ids, DESIGN["shap_seed"], self.nsamples,
                           self.cache, f"bitred-fold{self.k}-eval-{tag}")

    def shap_cal(self, net, tag: str) -> np.ndarray:
        return shap_cached(net, self.bg, self.X_lcal, self.target_lcal, self.lcal_ids, DESIGN["shap_seed"],
                           self.nsamples, self.cache, f"bitred-fold{self.k}-cal-{tag}")

    def assess(self, net, tag: str) -> tuple[dict, list[dict]]:
        """F1 / accuracy on every inner-val window; per-subject LCI on the explain windows."""
        pred = predict_proba(net, self.va.X, self.dev).argmax(1)
        out = {"f1": pooled_macro_f1(self.va.y, pred), "accuracy": float((pred == self.va.y).mean())}
        t = time.perf_counter()
        A = self.shap_eval(net, tag)
        sub = subject_lci(self.A_ref_ex, A, self.ex_subj, k=DESIGN["k"], omega=DESIGN["omega"])
        lci = float(np.mean([r["lci"] for r in sub]))
        worst = min(sub, key=lambda r: r["lci"])
        out.update({"lci": lci, "worst_subject_lci": float(worst["lci"]), "worst_subject": worst["subject"],
                    "rho": float(np.mean([r["shap_corr"] for r in sub])),
                    "jaccard": float(np.mean([r["jaccard"] for r in sub])),
                    "retention": (lci - self.tau) / (self.ceiling - self.tau), "below_tau": bool(lci < self.tau),
                    "n_skipped": int(sum(r["n_skipped"] for r in sub)), "shap_seconds": time.perf_counter() - t})
        rows = [{"subject": r["subject"], "lci": r["lci"], "shap_corr": r["shap_corr"], "jaccard": r["jaccard"],
                 "n_skipped": r["n_skipped"]} for r in sub]
        return out, rows
