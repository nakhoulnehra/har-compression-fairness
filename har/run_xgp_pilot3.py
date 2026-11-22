"""XGP pilot round 3: the final, validation-only design of XGP (plan Section 2.5 (b), "Round 3
pre-registration"). Folds 0 and 1, HARTH inner validation only. HAR70+ is never loaded: only the HARTH
splits of the window cache are read, and every window set (older-like copies included) goes through
require_harth.

Disclosure: the round-3 design was informed by the Step 9A test-set analysis (posture-class drift); all
tuning here uses inner-val and older-like windows only.

Design (har/xgp3.py; pre-registered by `--prepare`):
    older-like   time-stretch s ~ U(0.8, 0.9) -> movement scale a ~ U(0.7, 0.9) -> per-sensor 3D rotation
                 (back, thigh independently; random axis, angle ~ U(-15, 15) deg; Um et al. 2017); fixed a priori
    calibration  70 inner-val windows per class (non-explain, seed 44) + their older-like copies (980)
    keep-score   group-aware I: per group (normal, older-like) and class, class-balanced, max over groups;
                 score = (1 - beta)|w|/mean + beta I/mean over the surviving weights
    iterative    K = 5 steps to the target, rescoring on the current model, no training
    recon        ridge least-squares refit of the kept Conv / Linear weights to the original layer outputs
                 on the calibration windows, after every step (masks fixed)
    fine-tuned   shared KD (E_ft 20) + teacher-labelled older-like training copies (mix ratio r) + group DRO
    phase 1      (tuning) no_finetune: l1, snip, l1_recon, xgp(beta); finetune: l1 (CE), l1_kd, snip_kd,
                 l1_kdaug(r), xgp_ft(beta, r); budget curve at 2 / 5 / 20 epochs; levels 50/70/90, seeds 0-2
    phase 2      (ablations at the selected configs) xgp_no_{score,group,iter,recon} (+ _ft), xgp_ft_no_aug,
                 xgp_ft_no_dro, l1_recon_ft
    metrics      per window set (normal, older-like): LCI_val, rho, Jaccard, worst-subject LCI, pooled
                 7-class macro-F1, accuracy; FA-LCI_val = min over the sets; gap = normal - older-like

Usage (repo root, inside .venv):
    python -m har.run_xgp_pilot3 --prepare                                       # once, CPU, both folds
    python -m har.run_xgp_pilot3 --phase tuning --folds 0 --curve-seeds 0 --device cuda --machine-name uni-2080
    python -m har.run_xgp_pilot3 --phase tuning --folds 1 --curve-seeds 0 --device cuda --machine-name nakhoul-3050
    python -m har.run_xgp_pilot3 --analyze --phase tuning                        # -> tuning-selection.json
    python -m har.run_xgp_pilot3 --phase ablation --folds 0 --device cuda --machine-name uni-2080
    python -m har.run_xgp_pilot3 --phase ablation --folds 1 --device cuda --machine-name nakhoul-3050
    python -m har.run_xgp_pilot3 --analyze --phase ablation                      # -> decision.json
    add --smoke for the CPU check (tiny sizes, results/har-xgp-pilot-smoke/round3/); --resume continues a
    run; --curve-seeds 0 restricts the 2 / 5-epoch assessments to seed 0.

Writes results/har-xgp-pilot/round3/:
    har-xgp-pilot3-tinyhar-fold{k}-inner-val-window-roles-older-like-params.csv   --prepare
    har-xgp-pilot3-tinyhar-preregistration.json                                     --prepare
    har-xgp-pilot3-tinyhar-fold{k}-{tuning|ablation}-{machine}-{runs,per-subject-lci,finetune-history,
        prune-steps}.csv, -config.json
    har-xgp-pilot3-tinyhar-tuning-{selection.json,table.csv}, -summary-mean-sd.csv, -budget-curve-mean-sd.csv
    har-xgp-pilot3-tinyhar-decision.json, -decision-table.csv                       --analyze --phase ablation
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
from .augment import OLDER_LIKE_R3, draw_older_like_r3_params, older_like_r3, older_like_windows_r3
from .compression import (KD, finetune, head_unpruned, l1_scores, predict_logits, prune_by_score, size_record,
                          snip_scores)
from .data import Windows, find_windows, load_harth_windows, require_harth, window_ids
from .folds import load_folds, select_explain_windows
from .paths import (MODEL_SELECTION_DIR, NOISE_FLOOR_DIR, NOISE_FLOOR_SMOKE_DIR, SHAP_CACHE_DIR, XGP_PILOT2_DIR,
                    XGP_PILOT3_CKPT_DIR, XGP_PILOT3_DIR, XGP_PILOT3_SMOKE_DIR)
from .run_model_selection import subsample
from .run_noise_floor import fold_background
from .run_sanity_walking_aid import load_fold_model
from .run_xgp_pilot import read_tau
from .train import pooled_macro_f1, predict_proba
from .xgp3 import (COMPONENTS, ROUND3, STARTS, best_baseline, group_class_importance, iterative_prune,
                   layerwise_reconstruct, select_config, select_final)

FROZEN_CONFIG = MODEL_SELECTION_DIR / "har-frozen-model-config.json"
P = "har-xgp-pilot3-tinyhar"
SETS = ("normal", "older_like")
SMOKE = {"levels": (0.7,), "seeds": (0,), "betas": (0.1,), "rs": (0.25,), "K": 2, "epochs": 2, "curve_epochs": (1,),
         "n_train": 600, "n_val": 200, "explain_per_subject": 3, "cal_per_class": 3, "nsamples": 20, "m_score": 2}
DISCLOSURE = ("The round-3 design was informed by the Step 9A test-set analysis (posture-class drift: without "
              "fine-tuning, older adults' LCI fell more in standing at 70%, sitting at 90% and lying at 50%; the "
              "accuracy gap widens at 90%). All round-3 tuning and selection use HARTH inner-validation windows of "
              "folds 0 and 1 and their older-like copies only; HAR70+ and the folds' test subjects are never loaded.")
RULE_TEXT = (
    "Phase 1, per scenario and level: B_set = the highest mean F1 on that window set among the scenario's baselines "
    "(no_finetune: l1, snip, l1_recon; finetune: l1, l1_kd, snip_kd, l1_kdaug at both r). Eligible configuration: "
    "mean F1_normal >= B_normal - 1 pt and mean F1_older_like >= B_older_like - 1 pt. Selected = the eligible "
    "configuration with the highest mean FA-LCI_val (mean over folds 0, 1 and seeds of each run's min(LCI_normal, "
    "LCI_older_like)); ties within 0.005 -> smaller beta, then smaller r. None eligible -> no configuration at that "
    "level (flagged, no fallback), and phase 2 skips it.")
FINAL_RULE_TEXT = (
    "Phase 2, per scenario: candidates = the full XGP and its single-component ablations, each at the selected "
    "per-level configuration; eligible = eligible (same F1 rule) at every level; final = the eligible candidate with "
    "the highest FA-LCI_val averaged over levels; ties within 0.005 -> fewer components, then the higher FA-LCI. No "
    "candidate eligible -> XGP not frozen for that scenario: report and stop. Components: no_finetune full = 4 "
    "(score, group, iter, recon), finetune full = 6 (+ aug, + dro); no_aug removes aug and dro.")
METRICS = ([f"{m}_{s}" for s in SETS for m in ("lci", "rho", "jaccard", "worst_subject_lci", "retention", "f1",
                                                "accuracy")] + ["fa_lci", "gap", "f1_gap"])


ROT_COLS = [f"rot_{sn}_{c}" for sn in ("back", "thigh") for c in ("axis_x", "axis_y", "axis_z", "deg")]
AMENDMENT = ("2026-11-22, before any round-3 run: the older-like transform gains a per-sensor random 3D rotation "
             "(back and thigh triplets independently, axis uniform on the sphere, angle ~ U(-15, 15) deg; Um et al. "
             "2017), applied after the time-stretch and amplitude steps, so that static postures change too. The "
             "first pre-registration (same day, no run used it) was replaced. Phase 1 runs with --curve-seeds 0; "
             "the full budget curve goes to 9B.")


def rot_columns(axes: np.ndarray, angles: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    for j, sn in enumerate(("back", "thigh")):
        for i, c in enumerate(("axis_x", "axis_y", "axis_z")):
            out[f"rot_{sn}_{c}"] = axes[:, j, i]
        out[f"rot_{sn}_deg"] = angles[:, j]
    return out


def rot_from_table(t: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(axes (n, 2, 3), angles (n, 2)) from the window table's rotation columns (back, thigh)."""
    axes = np.stack([np.stack([t[f"rot_{sn}_{c}"].to_numpy() for c in ("axis_x", "axis_y", "axis_z")], 1)
                     for sn in ("back", "thigh")], 1)
    angles = np.stack([t[f"rot_{sn}_deg"].to_numpy() for sn in ("back", "thigh")], 1)
    return axes, angles


def cfg(key: str, smoke: bool):
    return SMOKE[key] if smoke and key in SMOKE else ROUND3[key]


def pct(level: float) -> int:
    return int(round(level * 100))


def file_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def method_id(method: str, beta=None, r=None) -> str:
    parts = ([f"beta={beta:g}"] if beta is not None else []) + ([f"r={r:g}"] if r is not None else [])
    return method + (f"({','.join(parts)})" if parts else "")


def out_dir(smoke: bool) -> Path:
    return XGP_PILOT3_SMOKE_DIR if smoke else XGP_PILOT3_DIR


# ------------------------------------------------------------------------------ data (HARTH only)

def load_setup(k: int, smoke: bool):
    harth = load_harth_windows()
    counts = pd.Series(harth["harth_test"].subject).value_counts().to_dict()
    fold = load_folds(harth_test_counts={s: int(c) for s, c in counts.items()})[k]
    tr = harth["harth_train"].for_subjects(fold.train)
    va = harth["harth_test"].for_subjects(fold.val)
    require_harth(tr, va, harth["harth_train"], harth["harth_test"])
    assert not (set(tr.subjects) | set(va.subjects)) & (set(fold.test_general) | set(fold.test_older))
    if smoke:
        tr = subsample(tr, SMOKE["n_train"], 0)
    return harth, fold, tr, va


def window_table(va: Windows, smoke: bool) -> pd.DataFrame:
    """Every inner-val window with its older-like (s, a) (seed 42, as round 2), its per-sensor rotation
    (axis, angle; independent streams of seed 42) and its role: explain
    (select_explain_windows, per subject), calibration (cal_per_class per class among the non-explain
    windows, seed 44; fewer if a class has fewer), or f1_only. F1 windows = explain + f1_only."""
    require_harth(va)
    q = draw_older_like_r3_params(len(va), OLDER_LIKE_R3["explain_seed"])
    t = pd.DataFrame({"window_id": window_ids(va, "harth_test"), "subject": va.subject, "label": va.y,
                      "speed_s": q["s"], "scale_a": q["a"], **rot_columns(q["axes"], q["angles"]),
                      "role": "f1_only", "pick_order": -1})
    ex = select_explain_windows(va, "harth_test", per_subject=cfg("explain_per_subject", smoke))
    for subj in va.subjects:
        sel = ex[va.subject[ex] == subj]
        t.loc[sel, "role"] = "explain"
        t.loc[sel, "pick_order"] = np.arange(len(sel))
    rest = np.flatnonzero(t.role.to_numpy() == "f1_only")
    rng = np.random.default_rng(ROUND3["cal_seed"])
    n = cfg("cal_per_class", smoke)
    for c in range(7):
        pool = rest[va.y[rest] == c]
        pick = rng.permutation(pool)[:n]
        t.loc[pick, "role"] = "calibration"
        t.loc[pick, "pick_order"] = np.arange(len(pick))
    return t


def table_path(out: Path, k: int) -> Path:
    return out / f"{P}-fold{k}-inner-val-window-roles-older-like-params.csv"


def prereg_path(out: Path) -> Path:
    return out / f"{P}-preregistration.json"


def prepare(args) -> None:
    out = out_dir(args.smoke)
    folds = tuple(args.folds) if args.folds else ROUND3["folds"]
    pp = prereg_path(out)
    if pp.exists() and not args.smoke:
        raise FileExistsError(f"{pp} exists; the pre-registration is fixed (delete it by hand only if no run used it)")
    sha, info = {}, {}
    for k in folds:
        _, fold, _, va = load_setup(k, args.smoke)
        t = window_table(va, args.smoke)
        p = table_path(out, k)
        R.write_csv(p, t)
        sha[str(k)] = file_sha(p)
        n_cal = t[t.role == "calibration"].groupby("label").size().to_dict()
        info[str(k)] = {"inner_val_subjects": list(fold.val), "n_inner_val": len(t),
                        "n_explain": int((t.role == "explain").sum()), "n_calibration": int((t.role == "calibration").sum()),
                        "calibration_per_class": {int(c): int(v) for c, v in n_cal.items()},
                        "n_f1_windows": int((t.role != "calibration").sum())}
        r2 = XGP_PILOT2_DIR / "har-xgp-pilot2-tinyhar-fold0-inner-val-windows-older-like-params.csv"
        if k == 0 and r2.exists() and not args.smoke:
            o = pd.read_csv(r2, dtype={"subject": str})
            info["0"]["explain_windows_equal_round2"] = bool(
                set(o.window_id[o.explain]) == set(t.window_id[t.role == "explain"]))
            info["0"]["older_like_params_equal_round2"] = bool(
                (o.window_id.to_numpy() == t.window_id.to_numpy()).all() and np.allclose(o.speed_s, t.speed_s)
                and np.allclose(o.scale_a, t.scale_a))
        print(f"fold {k}: {info[str(k)]}; saved {p}")
    R.write_json(pp, {"what": "XGP pilot round 3 (final design, validation only); pre-registered before any run",
                      "written_at": R.now(), **R.git_info(), "code_sha256": R.code_sha256(), "smoke": args.smoke,
                      "disclosure": DISCLOSURE, "selection_rule_phase1": RULE_TEXT, "final_rule_phase2": FINAL_RULE_TEXT,
                      "round3": ROUND3, "starts": STARTS, "components": COMPONENTS, "older_like": OLDER_LIKE_R3, "kd": KD,
                      "phase1_curve_seeds": [0], "amended": AMENDMENT,
                      "folds": list(folds), "windows": info, "sha256_window_tables": sha,
                      "plan": "plan/HAR_IMPLEMENTATION.md, Section 2.5 (b), 'Round 3 pre-registration'"})
    print(f"saved {pp}")


def check_table(out: Path, k: int, t: pd.DataFrame) -> dict:
    pp, p = prereg_path(out), table_path(out, k)
    if not pp.exists() or not p.exists():
        raise FileNotFoundError(f"{pp.name} / {p.name} missing: run `python -m har.run_xgp_pilot3 --prepare` first")
    prereg = json.loads(pp.read_text(encoding="utf-8"))
    if file_sha(p) != prereg["sha256_window_tables"][str(k)]:
        raise RuntimeError(f"{p.name} changed after the pre-registration")
    saved = pd.read_csv(p, dtype={"subject": str})
    same = (len(saved) == len(t) and (saved.window_id.to_numpy() == t.window_id.to_numpy()).all()
            and (saved.role.to_numpy() == t.role.to_numpy()).all()
            and np.allclose(saved.speed_s, t.speed_s, rtol=0, atol=1e-12)
            and np.allclose(saved.scale_a, t.scale_a, rtol=0, atol=1e-12)
            and all(np.allclose(saved[c], t[c], rtol=0, atol=1e-12) for c in ROT_COLS))
    if not same:
        raise RuntimeError(f"{p.name} differs from the windows re-derived on this machine")
    return prereg


def read_ceiling(k: int, smoke: bool, machine: str | None) -> tuple[float, str]:
    d = NOISE_FLOOR_SMOKE_DIR if smoke else NOISE_FLOOR_DIR
    found = sorted(d.glob(f"har-noise-floor-tinyhar-fold{k}-*-group-summary.csv"))
    if machine:
        found = [p for p in found if p.name == f"har-noise-floor-tinyhar-fold{k}-{machine}-group-summary.csv"]
    if len(found) != 1:
        raise FileNotFoundError(f"need exactly one fold-{k} noise-floor group summary in {d}; found {found} "
                                "(pass --noise-floor-machine)")
    g = pd.read_csv(found[0])
    row = g[(g.group == "inner_val") & (g.noise == "shap")]
    return float(row["mean"].iloc[0]), str(found[0])


# ------------------------------------------------------------------------------ fold context

class Context:
    def __init__(self, args, k: int):
        self.args, self.k, self.smoke, self.dev = args, k, args.smoke, args.device
        self.out = out_dir(args.smoke)
        self.nsamples = cfg("nsamples", args.smoke)
        self.cache = SHAP_CACHE_DIR / ("smoke" if args.smoke else "")
        frozen = json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))
        assert frozen["model"] == "tinyhar", frozen["model"]
        harth, fold, tr, va_full = load_setup(k, args.smoke)
        t = window_table(va_full, args.smoke)
        self.prereg = check_table(self.out, k, t)
        self.fold, self.tr = fold, tr
        s_all, a_all = t.speed_s.to_numpy(), t.scale_a.to_numpy()
        ax_all, an_all = rot_from_table(t)
        # F1 windows: every inner-val window that is not a calibration window, and its older-like copy
        f1_idx = np.flatnonzero(t.role.to_numpy() != "calibration")
        if args.smoke:
            f1_idx = np.sort(np.random.default_rng(1).permutation(f1_idx)[:SMOKE["n_val"]])
        self.va = va_full.select(f1_idx)
        self.va_o = older_like_windows_r3(self.va, s_all[f1_idx], a_all[f1_idx], ax_all[f1_idx], an_all[f1_idx])
        # explain windows (normal + older-like), subjects sorted, then pick order
        ex = t[t.role == "explain"].sort_values(["subject", "pick_order"])
        X_n = va_full.X[find_windows(va_full, "harth_test", ex.window_id.tolist())]
        X_o = older_like_r3(X_n, ex.speed_s.to_numpy(), ex.scale_a.to_numpy(), *rot_from_table(ex))
        self.n_ex = len(ex)
        self.ids = ex.window_id.tolist() + [f"{i}|older_like" for i in ex.window_id]
        self.subj = np.concatenate([ex.subject.to_numpy(), ex.subject.to_numpy()])
        self.sl = {"normal": slice(0, self.n_ex), "older_like": slice(self.n_ex, 2 * self.n_ex)}
        # calibration windows (normal + older-like), class then pick order
        cal = t[t.role == "calibration"].sort_values(["label", "pick_order"])
        cal_w = va_full.select(find_windows(va_full, "harth_test", cal.window_id.tolist()))
        cal_o = older_like_windows_r3(cal_w, cal.speed_s.to_numpy(), cal.scale_a.to_numpy(), *rot_from_table(cal))
        require_harth(self.tr, self.va, self.va_o, cal_w, cal_o)
        assert set(self.subj) == set(fold.val) and not set(cal.window_id) & set(ex.window_id)
        net0, self.ckpt = load_fold_model(k, frozen["architecture"])
        self.net0 = net0.to(self.dev).eval()
        self.bg = fold_background(harth, k).to(self.dev)
        self.cal = {}
        for g, w in (("normal", cal_w), ("older_like", cal_o)):
            X = torch.from_numpy(w.X).to(self.dev)
            tgt = torch.from_numpy(predict_proba(self.net0, w.X, self.dev).argmax(1)).to(self.dev)
            self.cal[g] = (X, tgt, w.y.copy())
        self.n_cal = {g: len(v[2]) for g, v in self.cal.items()}
        try:
            self.tau, self.tau_src = read_tau(k, args.smoke, args.noise_floor_machine)
            self.ceiling, self.ceiling_src = read_ceiling(k, args.smoke, args.noise_floor_machine)
        except FileNotFoundError:
            if not args.smoke:
                raise
            self.tau, self.tau_src, self.ceiling, self.ceiling_src = 0.6, "smoke default", 0.9, "smoke default"
        X_all = np.concatenate([X_n, X_o])
        self.X_all = torch.from_numpy(X_all).to(self.dev)
        self.target = predict_proba(self.net0, X_all, self.dev).argmax(1)
        self.times = {}
        t0 = time.perf_counter()
        self.A_ref = self.shap(self.net0, "original")
        self.times["shap_original"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.kd_logits = torch.from_numpy(predict_logits(self.net0, tr.X, self.dev)).to(self.dev)
        self.snip = snip_scores(self.net0, tr)
        self.times["teacher_logits_and_snip"] = time.perf_counter() - t0
        print(f"fold {k}: train {len(tr)}, F1 windows {len(self.va)} + {len(self.va_o)} older-like, explain "
              f"{self.n_ex} + {self.n_ex}, calibration {self.n_cal} ({list(fold.val)}); tau_k {self.tau:.4f} "
              f"({self.tau_src}), ceiling {self.ceiling:.4f}; original SHAP {self.times['shap_original']:.1f} s")

    def shap(self, net, tag: str) -> np.ndarray:
        return shap_cached(net, self.bg, self.X_all, self.target, self.ids, ROUND3["shap_seed"], self.nsamples,
                           self.cache, f"har-xgp-pilot3-fold{self.k}-{tag}")

    def assess(self, net, tag: str) -> tuple[dict, list[dict]]:
        out, rows = {}, []
        for name, w in (("normal", self.va), ("older_like", self.va_o)):
            pred = predict_proba(net, w.X, self.dev).argmax(1)
            out[f"f1_{name}"] = pooled_macro_f1(w.y, pred)
            out[f"accuracy_{name}"] = float((pred == w.y).mean())
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
        out["fa_lci"] = min(out["lci_normal"], out["lci_older_like"])
        out["gap"] = out["lci_normal"] - out["lci_older_like"]
        out["f1_gap"] = out["f1_normal"] - out["f1_older_like"]
        out["shap_seconds"] = time.perf_counter() - t
        return out, rows

    # ---- pruned starts
    def importance_fn(self, seed: int, groups):
        sets = {g: self.cal[g] for g in groups}
        m = cfg("m_score", self.smoke)
        return lambda net: group_class_importance(net, self.net0, sets, self.bg, m=m, eg_seeds=(2 * seed, 2 * seed + 1),
                                                  batch_size=ROUND3["score_batch"])

    def recon_fn(self, groups):
        X = torch.cat([self.cal[g][0] for g in groups])
        return lambda net: layerwise_reconstruct(net, self.net0, X, damp=ROUND3["damp"])

    def build_start(self, kind: str, level: float, seed: int, beta: float | None) -> tuple[torch.nn.Module, list[dict]]:
        if kind == "l1":
            return prune_by_score(self.net0, l1_scores(self.net0), level), []
        if kind == "snip":
            return prune_by_score(self.net0, self.snip, level), []
        if kind == "l1_recon":
            net = prune_by_score(self.net0, l1_scores(self.net0), level)
            rec = self.recon_fn(SETS)(net)
            return net, [{"step": 1, "sparsity": level, "n_nonzero": size_record(net)["n_nonzero"], "recon": rec["refit"]}]
        o = STARTS[kind]
        b = beta if o["score"] else 0.0
        return iterative_prune(self.net0, level, K=cfg("K", self.smoke) if o["K"] > 1 else 1, beta=b,
                               importance_fn=self.importance_fn(seed, o["groups"]) if o["score"] else None,
                               recon_fn=self.recon_fn(o["groups"]) if o["recon"] else None)

    def aug_training(self, seed: int, r: float) -> tuple[Windows, torch.Tensor, np.ndarray]:
        """HARTH training windows + round(r N) teacher-labelled older-like copies (indices and (s, a, rotation) seeded
        with 1000 + seed); KD targets = the original model's logits on each copy; groups 0 / 1."""
        tr, N = self.tr, len(self.tr)
        n = int(round(r * N))
        sd = ROUND3["aug_seed_offset"] + seed
        idx = np.sort(np.random.default_rng(np.random.SeedSequence([sd, 1])).choice(N, n, replace=False))
        q = draw_older_like_r3_params(n, sd)
        aug = older_like_windows_r3(tr.select(idx), q["s"], q["a"], q["axes"], q["angles"])
        both = Windows.concat([tr, aug], "harth")
        require_harth(both)
        z_aug = torch.from_numpy(predict_logits(self.net0, aug.X, self.dev)).to(self.dev)
        return both, torch.cat([self.kd_logits, z_aug]), np.r_[np.zeros(N, int), np.ones(n, int)]


# ------------------------------------------------------------------------------ phases

class Store:
    TABLES = ("runs", "per-subject-lci", "finetune-history", "prune-steps")

    def __init__(self, out: Path, k: int, phase: str, machine: str, resume: bool, smoke: bool):
        self.base = f"{P}-fold{k}-{phase}-{machine}"
        self.paths = {t: out / f"{self.base}-{t}.csv" for t in self.TABLES}
        self.config = out / f"{self.base}-config.json"
        exists = self.paths["runs"].exists()
        if exists and not (resume or smoke):
            raise FileExistsError(f"{self.paths['runs']} exists; pass --resume to continue it (never overwritten)")
        keep = resume and exists
        self.rows = {t: (pd.read_csv(p, dtype={"subject": str}).to_dict("records")
                         if keep and p.exists() and p.stat().st_size > 2 else []) for t, p in self.paths.items()}
        self.done = {r["run_id"] for r in self.rows["runs"]}

    def save(self):
        for t, p in self.paths.items():
            R.write_csv(p, pd.DataFrame(self.rows[t]))


def run_fold(args, k: int, phase: str) -> None:
    S = args.smoke
    out = out_dir(S)
    store = Store(out, k, phase, args.machine_name, args.resume, S)
    log = out / "logs" / f"{store.base}-{R.stamp()}.log"
    with R.tee_to(log):
        dev = R.device_info(args.device)
        t_fold = time.perf_counter()
        print(f"=== XGP pilot round 3, fold {k}, phase {phase} | machine {args.machine_name} | device {args.device} "
              f"{dev['gpu_name']} | torch {dev['torch_version']} | started {R.now()} | {' '.join(sys.argv)}"
              + (f" | resuming, {len(store.done)} runs done" if store.done else ""))
        ctx = Context(args, k)
        levels = tuple(args.levels) if args.levels else cfg("levels", S)
        seeds = tuple(args.seeds) if args.seeds else cfg("seeds", S)
        curve_seeds = set(args.curve_seeds) if args.curve_seeds is not None else set(seeds)
        epochs, curve_eps = cfg("epochs", S), cfg("curve_epochs", S)
        selection = read_selection(out) if phase == "ablation" else None
        common = {"fold": k, "phase": phase, "tau_k": ctx.tau, "ceiling": ctx.ceiling, "machine_name": args.machine_name,
                  "device": args.device}
        nnz: dict[float, set] = {}
        starts: dict[tuple, torch.nn.Module] = {}
        ckpt_dir = XGP_PILOT3_CKPT_DIR / ("smoke" if S else "")

        def record(run_id, base, a, rows, net):
            rec = size_record(net)
            nnz.setdefault(base["level"], set()).add(rec["n_nonzero"])
            if len(nnz[base["level"]]) != 1:
                raise RuntimeError(f"n_nonzero differs within {base['level']:.0%}: {nnz[base['level']]}")
            store.rows["per-subject-lci"].extend({"run_id": run_id, **base, **r} for r in rows)
            store.rows["runs"].append({"run_id": run_id, **base, **a, "head_unpruned": head_unpruned(net),
                                       **{x: v for x, v in rec.items() if x != "per_layer_sparsity"}, **common})
            store.done.add(run_id)
            store.save()

        def get_start(kind, level, seed, beta):
            key = (kind, level, seed if kind not in ("l1", "snip", "l1_recon") else 0, beta)
            if key in starts:
                return starts[key]
            tag = f"fold{k}-{kind}-{pct(level)}pct-seed{key[2]}" + (f"-beta{beta:g}" if beta is not None else "")
            p = ckpt_dir / f"{P}-{args.machine_name}-{tag}.pt"
            t0 = time.perf_counter()
            if p.exists():
                ck = torch.load(p, map_location="cpu", weights_only=False)
                net = copy.deepcopy(ctx.net0)
                net.load_state_dict(ck["state_dict"])
                net.prune_masks = {n: m.to(ctx.dev) for n, m in ck["prune_masks"].items()}
                net.prune_sparsity = level
                net = net.to(ctx.dev).eval()
                steps, src = [], "loaded"          # its prune steps were logged when it was built
            else:
                print(f"\n--- start {tag}")
                net, steps = ctx.build_start(kind, level, key[2], beta)
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": {n: v.cpu() for n, v in net.state_dict().items()},
                            "prune_masks": {n: m.cpu() for n, m in net.prune_masks.items()}, "steps": steps}, p)
                src = "built"
            net.start_seconds = time.perf_counter() - t0
            net.start_source = src
            for st in steps:
                store.rows["prune-steps"].append({"start": tag, "step": st["step"], "sparsity": st["sparsity"],
                                                  "n_nonzero": st["n_nonzero"], "recon": json.dumps(st["recon"])})
            starts[key] = net
            return net

        def no_ft(method, kind, level, seed, beta=None):
            mid = method_id(method, beta)
            rid = f"noft-{mid}-{pct(level)}pct-seed{seed}"
            if rid in store.done:
                return
            net = get_start(kind, level, seed, beta)
            a, rows = ctx.assess(net, f"{pct(level)}pct-{rid}")
            base = {"level": level, "scenario": "no_finetune", "method": method, "method_id": mid, "start": kind,
                    "seed": seed, "beta": beta, "r": None, "epoch": 0, "recipe": None}
            record(rid, base, {**a, "start_seconds": net.start_seconds}, rows, net)
            print(f"  {rid:<44} FA-LCI {a['fa_lci']:.4f} (normal {a['lci_normal']:.4f}, older-like "
                  f"{a['lci_older_like']:.4f}); F1 {a['f1_normal']:.4f} / {a['f1_older_like']:.4f}")

        def ft(method, kind, level, seed, *, recipe, beta=None, r=None, dro=False, curve=False):
            mid = method_id(method, beta, r)
            rid = f"ft-{mid}-{pct(level)}pct-seed{seed}"
            if rid in store.done:
                return
            start = get_start(kind, level, seed, beta)
            net = copy.deepcopy(start)
            if recipe == "kdaug":
                train, z, groups = ctx.aug_training(seed, r)
            else:
                train, z, groups = ctx.tr, ctx.kd_logits, None
            base = {"level": level, "scenario": "finetune", "method": method, "method_id": mid, "start": kind,
                    "seed": seed, "beta": beta, "r": r, "recipe": recipe, "dro": dro}
            print(f"\n=== fold {k} {rid}: start {kind}, recipe {recipe}" + (", group DRO" if dro else "")
                  + f", {len(train)} training windows, E_ft {epochs}")
            t_ft = time.perf_counter()
            t_cb = [0.0]

            def cb(ep, model):
                if ep not in curve_eps or not curve or seed not in curve_seeds:
                    return
                t0 = time.perf_counter()
                a, rows = ctx.assess(model, f"{pct(level)}pct-{rid}-ep{ep}")
                record(f"{rid}-ep{ep}", {**base, "epoch": ep}, a, rows, model)
                t_cb[0] += time.perf_counter() - t0
                print(f"    curve epoch {ep}: FA-LCI {a['fa_lci']:.4f}, F1 {a['f1_normal']:.4f} / {a['f1_older_like']:.4f}")
            info = finetune(net, train, ctx.va, epochs=epochs, lr=ROUND3["lr"], seed=seed, device=ctx.dev,
                            kd_teacher_logits=z if recipe in ("kd", "kdaug") else None, groups=groups if dro else None,
                            dro_eta=ROUND3["dro_eta"] if dro else None, on_epoch_end=cb)
            if info["epochs_run"] != epochs:
                raise RuntimeError(f"{rid}: {info['epochs_run']} epochs, expected {epochs}")
            store.rows["finetune-history"].extend({"run_id": rid, **h} for h in info["history"])
            a, rows = ctx.assess(net, f"{pct(level)}pct-{rid}")
            record(rid, {**base, "epoch": epochs}, {**a, "finetune_seconds": info["seconds"] - t_cb[0],
                                                     "n_train_windows": len(train)}, rows, net)
            print(f"    -> FA-LCI {a['fa_lci']:.4f} (normal {a['lci_normal']:.4f}, older-like {a['lci_older_like']:.4f});"
                  f" F1 {a['f1_normal']:.4f} / {a['f1_older_like']:.4f}; {info['seconds'] - t_cb[0]:.1f} s")

        if "original" not in store.done:
            a = {f"f1_{n}": pooled_macro_f1(w.y, predict_proba(ctx.net0, w.X, ctx.dev).argmax(1))
                 for n, w in (("normal", ctx.va), ("older_like", ctx.va_o))}
            store.rows["runs"].append({"run_id": "original", "scenario": "original", "method": "original",
                                       "method_id": "original", **a, "lci_normal": 1.0, "lci_older_like": 1.0,
                                       "fa_lci": 1.0, **common})
            store.done.add("original")
            store.save()

        betas, rs = cfg("betas", S), cfg("rs", S)
        for lv in levels:
            starts.clear()
            print(f"\n######## fold {k}, phase {phase}, level {lv:.0%}")
            if phase == "tuning":
                no_ft("l1", "l1", lv, 0)
                no_ft("snip", "snip", lv, 0)
                no_ft("l1_recon", "l1_recon", lv, 0)
                for s in seeds:
                    for b in betas:
                        no_ft("xgp", "xgp", lv, s, b)
                    ft("l1", "l1", lv, s, recipe="ce", curve=True)
                    ft("l1_kd", "l1", lv, s, recipe="kd", curve=True)
                    ft("snip_kd", "snip", lv, s, recipe="kd", curve=True)
                    for r in rs:
                        ft("l1_kdaug", "l1", lv, s, recipe="kdaug", r=r, dro=True, curve=True)
                    for b in betas:
                        for r in rs:
                            ft("xgp_ft", "xgp", lv, s, recipe="kdaug", beta=b, r=r, dro=True, curve=True)
            else:
                sel = selection[f"{pct(lv)}pct"]
                nf, fsel = sel["no_finetune"], sel["finetune"]
                for s in seeds:
                    if nf is not None:
                        b = nf["beta"]
                        no_ft("xgp_no_score", "xgp_no_score", lv, s, None)
                        for kind in ("xgp_no_group", "xgp_no_iter", "xgp_no_recon"):
                            no_ft(kind, kind, lv, s, b)
                    else:
                        print(f"  {lv:.0%} no_finetune: no selected configuration (phase 1); ablations skipped")
                    if fsel is not None:
                        b, r = fsel["beta"], fsel["r"]
                        ft("xgp_no_score_ft", "xgp_no_score", lv, s, recipe="kdaug", r=r, dro=True)
                        for kind in ("xgp_no_group", "xgp_no_iter", "xgp_no_recon"):
                            ft(f"{kind}_ft", kind, lv, s, recipe="kdaug", beta=b, r=r, dro=True)
                        ft("xgp_ft_no_aug", "xgp", lv, s, recipe="kd", beta=b)
                        ft("xgp_ft_no_dro", "xgp", lv, s, recipe="kdaug", beta=b, r=r, dro=False)
                        ft("l1_recon_ft", "l1_recon", lv, s, recipe="kdaug", r=r, dro=True)
                    else:
                        print(f"  {lv:.0%} finetune: no selected configuration (phase 1); ablations skipped")
        wall = time.perf_counter() - t_fold
        R.write_json(store.config, {"command_line": sys.argv, "machine_name": args.machine_name, "device": dev,
                                    **R.git_info(), "code_sha256": R.code_sha256(), "fold": k, "phase": phase,
                                    "smoke": S, "smoke_sizes": SMOKE if S else None, "round3": ROUND3,
                                    "levels": list(levels), "seeds": list(seeds), "curve_seeds": sorted(curve_seeds),
                                    "epochs": epochs, "curve_epochs": list(curve_eps), "nsamples": ctx.nsamples,
                                    "n_calibration": ctx.n_cal, "n_explain": ctx.n_ex, "n_f1_windows": len(ctx.va),
                                    "tau_k": ctx.tau, "tau_source": ctx.tau_src, "ceiling": ctx.ceiling,
                                    "ceiling_source": ctx.ceiling_src, "weights": str(ctx.ckpt),
                                    "preregistration_written_at": ctx.prereg["written_at"],
                                    "selection_used": selection, "n_nonzero": {str(l): sorted(v) for l, v in nnz.items()},
                                    "seconds_fixed": ctx.times, "seconds_wall_this_invocation": wall,
                                    "finished_at": R.now()})
        print(f"\nfold {k} phase {phase} done in {wall:.1f} s")
        for p in (*store.paths.values(), store.config, log):
            print(f"saved {p}")


# ------------------------------------------------------------------------------ analysis

def load_runs(out: Path, phase: str, folds, use_machine: dict[int, str]) -> pd.DataFrame:
    parts = []
    for k in folds:
        found = sorted(out.glob(f"{P}-fold{k}-{phase}-*-runs.csv"))
        if k in use_machine:
            found = [f for f in found if f.name == f"{P}-fold{k}-{phase}-{use_machine[k]}-runs.csv"]
        if len(found) != 1:
            raise FileNotFoundError(f"need exactly one fold-{k} {phase} runs file in {out} (use --use-machine), "
                                    f"found {[f.name for f in found]}")
        parts.append(pd.read_csv(found[0]))
    df = pd.concat(parts, ignore_index=True)
    return df[df.scenario != "original"]


def finals(df: pd.DataFrame, epochs: int) -> pd.DataFrame:
    """The rows the rules use: no_finetune (epoch 0) and finetune at E_ft."""
    return df[(df.scenario == "no_finetune") | ((df.scenario == "finetune") & (df.epoch == epochs))]


def cell_stats(df: pd.DataFrame) -> dict[tuple, dict]:
    """{(scenario, level, method_id): mean over folds and seeds of fa_lci, f1_normal, f1_older_like}."""
    return {key: {"fa_lci": float(g.fa_lci.mean()), "f1_normal": float(g.f1_normal.mean()),
                  "f1_older_like": float(g.f1_older_like.mean()), "n_runs": len(g)}
            for key, g in df.groupby(["scenario", "level", "method_id"])}


def summary(df: pd.DataFrame, by) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by, dropna=False):
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        row.update({"n_runs": len(g), "folds": ";".join(str(int(f)) for f in sorted(g.fold.unique()))})
        for m in METRICS:
            if m in g:
                row[f"{m}_mean"] = float(g[m].mean())
                row[f"{m}_sd"] = float(g[m].std(ddof=1)) if len(g) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def selection_path(out: Path) -> Path:
    return out / f"{P}-tuning-selection.json"


def read_selection(out: Path) -> dict:
    p = selection_path(out)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run `python -m har.run_xgp_pilot3 --analyze --phase tuning` first")
    return json.loads(p.read_text(encoding="utf-8"))["selected"]


def analyze_tuning(args) -> int:
    S, out = args.smoke, out_dir(args.smoke)
    folds = tuple(args.folds) if args.folds else (ROUND3["folds"] if not S else tuple(
        sorted({int(p.name.split("-fold")[1].split("-")[0]) for p in out.glob(f"{P}-fold*-tuning-*-runs.csv")})))
    df = load_runs(out, "tuning", folds, parse_use_machine(args.use_machine))
    epochs = cfg("epochs", S)
    fin = finals(df, epochs)
    st = cell_stats(fin)
    levels, betas, rs = cfg("levels", S), cfg("betas", S), cfg("rs", S)
    for lv in levels:
        g = fin[fin.level == lv]
        if g.n_nonzero.nunique() != 1:
            raise RuntimeError(f"n_nonzero differs within {lv:.0%}")
    selected, table, best_all = {}, [], {}
    for lv in levels:
        selected[f"{pct(lv)}pct"] = {}
        for scen in ("no_finetune", "finetune"):
            here = {mid: v for (sc, l, mid), v in st.items() if sc == scen and l == lv}
            best = best_baseline(here, ROUND3["baselines"][scen])
            best_all[(scen, lv)] = best
            if scen == "no_finetune":
                cfgs = {(b, None): here[method_id("xgp", b)] for b in betas}
            else:
                cfgs = {(b, r): here[method_id("xgp_ft", b, r)] for b in betas for r in rs}
            pick, tab = select_config(cfgs, best)
            selected[f"{pct(lv)}pct"][scen] = None if pick is None else {"beta": pick[0], "r": pick[1]}
            for t in tab:
                table.append({"scenario": scen, "level": lv, "role": "xgp_config", **t,
                              "best_baseline_f1_normal": best["f1_normal"],
                              "best_baseline_f1_older_like": best["f1_older_like"]})
            for mid, v in here.items():
                if not mid.startswith("xgp"):
                    table.append({"scenario": scen, "level": lv, "role": "baseline", "method_id": mid, **v})
            print(f"{lv:.0%} {scen:<12} best baseline F1 {best['f1_normal']:.4f} / {best['f1_older_like']:.4f} -> "
                  + ("NO ELIGIBLE CONFIGURATION" if pick is None else f"beta {pick[0]:g}"
                     + (f", r {pick[1]:g}" if pick[1] is not None else "")))
    R.write_csv(out / f"{P}-tuning-table.csv", pd.DataFrame(table))
    R.write_csv(out / f"{P}-tuning-summary-mean-sd.csv",
                summary(fin, ["scenario", "level", "method_id"]))
    curve = df[df.scenario == "finetune"]
    curve = pd.concat([curve, start_rows_for_curve(df)], ignore_index=True)
    R.write_csv(out / f"{P}-budget-curve-mean-sd.csv", summary(curve, ["level", "method_id", "epoch"]))
    R.write_json(selection_path(out), {"selected": selected, "rule": RULE_TEXT, "disclosure": DISCLOSURE,
                                       "folds": list(folds), "analyzed_at": R.now(), "smoke": S, **R.git_info()})
    print(f"saved {selection_path(out)}")
    return 0


def start_rows_for_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Epoch 0 of each fine-tuned method = its pruned start (no_finetune rows), relabelled with the
    fine-tuned method's ID (l1 start for l1 / l1_kd / l1_kdaug, snip for snip_kd, xgp(beta) for xgp_ft)."""
    ft = df[df.scenario == "finetune"][["level", "method_id", "start", "beta", "seed", "fold"]].drop_duplicates()
    nf = df[df.scenario == "no_finetune"]
    rows = []
    for _, r in ft.iterrows():
        m = nf[(nf.level == r.level) & (nf.start == r.start) & (nf.fold == r.fold)
               & ((nf.beta == r.beta) | (nf.beta.isna() & pd.isna(r.beta)))]
        if r.start not in ("l1", "snip", "l1_recon"):
            m = m[m.seed == r.seed]
        if len(m):
            x = m.iloc[0].to_dict()
            x.update({"method_id": r.method_id, "epoch": 0, "seed": r.seed})
            rows.append(x)
    return pd.DataFrame(rows).drop_duplicates(subset=["level", "method_id", "seed", "fold"]) if rows else pd.DataFrame()


def analyze_ablation(args) -> int:
    S, out = args.smoke, out_dir(args.smoke)
    folds = tuple(args.folds) if args.folds else (ROUND3["folds"] if not S else tuple(
        sorted({int(p.name.split("-fold")[1].split("-")[0]) for p in out.glob(f"{P}-fold*-ablation-*-runs.csv")})))
    um = parse_use_machine(args.use_machine)
    epochs = cfg("epochs", S)
    d1 = finals(load_runs(out, "tuning", folds, um), epochs)
    d2 = finals(load_runs(out, "ablation", folds, um), epochs)
    st = cell_stats(pd.concat([d1, d2], ignore_index=True))
    sel = read_selection(out)
    levels = cfg("levels", S)
    decision, tables = {}, []
    for scen in ("no_finetune", "finetune"):
        lv_ok = [lv for lv in levels if sel[f"{pct(lv)}pct"][scen] is not None]
        if len(lv_ok) != len(levels):
            decision[scen] = {"final": None, "reason": "no eligible configuration in phase 1 at "
                              + ", ".join(f"{pct(lv)}%" for lv in levels if lv not in lv_ok)}
            continue
        best = {lv: best_baseline({m: v for (sc, l, m), v in st.items() if sc == scen and l == lv},
                                  ROUND3["baselines"][scen]) for lv in levels}
        cands = {}
        for v in COMPONENTS[scen]:
            by = {}
            for lv in levels:
                mid = _cand_mid(v, scen, sel, lv)
                if (scen, lv, mid) not in st:
                    raise KeyError(f"missing {scen} {mid} at {lv:.0%}")
                by[lv] = st[(scen, lv, mid)]
            cands[v] = by
        final, tab = select_final(cands, best, COMPONENTS[scen])
        # descriptive: final - best baseline FA-LCI per level, sign per fold
        vs = {}
        if final is not None:
            allrows = pd.concat([d1, d2], ignore_index=True)
            for lv in levels:
                here = {m: v for (sc, l, m), v in st.items() if sc == scen and l == lv
                        and any(m == b or m.startswith(b + "(") for b in ROUND3["baselines"][scen])}
                bmid = max(here, key=lambda m: here[m]["fa_lci"])
                fmid = _cand_mid(final, scen, sel, lv)
                per_fold = {}
                for f in folds:
                    a = allrows[(allrows.scenario == scen) & (allrows.level == lv) & (allrows.fold == f)]
                    per_fold[str(f)] = float(a[a.method_id == fmid].fa_lci.mean() - a[a.method_id == bmid].fa_lci.mean())
                vs[f"{pct(lv)}pct"] = {"final_method_id": fmid, "best_baseline_by_fa_lci": bmid,
                                       "diff_fa_lci": st[(scen, lv, fmid)]["fa_lci"] - here[bmid]["fa_lci"],
                                       "diff_by_fold": per_fold}
        decision[scen] = {"final": final, "components": list(COMPONENTS[scen][final]) if final else None,
                          "config_by_level": {f"{pct(lv)}pct": sel[f"{pct(lv)}pct"][scen] for lv in levels},
                          "best_baseline_f1_by_level": {f"{pct(lv)}pct": best[lv] for lv in levels},
                          "final_vs_best_baseline_fa_lci (descriptive)": vs}
        for t in tab:
            tables.append({"scenario": scen, **t})
        print(f"{scen}: final = {final}" + ("" if final else "  (NO ELIGIBLE CANDIDATE: XGP not frozen; stop and review)"))
    R.write_csv(out / f"{P}-decision-table.csv", pd.DataFrame(tables))
    R.write_csv(out / f"{P}-ablation-summary-mean-sd.csv", summary(d2, ["scenario", "level", "method_id"]))
    R.write_json(out / f"{P}-decision.json", {"decision": decision, "selection_phase1": sel, "rule_phase1": RULE_TEXT,
                                              "rule_final": FINAL_RULE_TEXT, "disclosure": DISCLOSURE,
                                              "folds": list(folds), "analyzed_at": R.now(), "smoke": S, **R.git_info()})
    print(f"saved {out / f'{P}-decision.json'}")
    return 0 if all(v.get("final") for v in decision.values()) else 2


def _cand_mid(v: str, scen: str, sel: dict, lv: float) -> str:
    c = sel[f"{pct(lv)}pct"][scen]
    special = {"xgp": method_id("xgp", c["beta"]), "xgp_ft": method_id("xgp_ft", c["beta"], c["r"]),
               "xgp_no_score": method_id("xgp_no_score"), "xgp_no_score_ft": method_id("xgp_no_score_ft", None, c["r"]),
               "xgp_ft_no_aug": method_id("xgp_ft_no_aug", c["beta"]),
               "xgp_ft_no_dro": method_id("xgp_ft_no_dro", c["beta"], c["r"])}
    if v in special:
        return special[v]
    return method_id(v, c["beta"]) if scen == "no_finetune" else method_id(v, c["beta"], c["r"])


def parse_use_machine(items) -> dict[int, str]:
    return {int(x.split("=")[0]): x.split("=")[1] for x in (items or [])}


# ------------------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true", help="write the window tables and the pre-registration")
    mode.add_argument("--run", action="store_true", help="(default with --folds) run --phase on --folds")
    mode.add_argument("--analyze", action="store_true", help="merge the folds' files and apply the --phase rule")
    ap.add_argument("--phase", choices=["tuning", "ablation"], default=None)
    ap.add_argument("--folds", type=int, nargs="+", default=None, choices=ROUND3["folds"])
    ap.add_argument("--levels", type=float, nargs="+", default=None, choices=ROUND3["levels"])
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--curve-seeds", type=int, nargs="*", default=None,
                    help="seeds that get the 2 / 5-epoch assessments (default: every seed)")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--machine-name", default=R.machine_name())
    ap.add_argument("--noise-floor-machine", default=None, help="whose noise-floor files give tau_k and the ceiling")
    ap.add_argument("--use-machine", nargs="*", default=None, help="--analyze: k=machine if a fold ran on two machines")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    argv = sys.argv[1:]
    if "--phase" in argv and not any(a in argv for a in ("--prepare", "--run", "--analyze")):
        argv = ["--run", *argv]
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.prepare:
        return prepare(args)
    if args.phase is None:
        ap.error("--phase tuning|ablation is required")
    if args.analyze:
        sys.exit(analyze_tuning(args) if args.phase == "tuning" else analyze_ablation(args))
    if not args.folds:
        ap.error("--folds is required to run a phase")
    t0 = time.perf_counter()
    for k in args.folds:
        run_fold(args, k, args.phase)
    print(f"\nfolds {args.folds}, phase {args.phase}: {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
