"""Step 6: timing benchmark of the frozen model's components on one machine (plan Section 3, Step 6).

Measures, on the frozen TinyHAR fold-k checkpoint and the fold's real windows, the cost of every part
of Steps 8 + 9, then projects the per-fold wall-clock time of Steps 8 + 9 and compares it with the
plan's Section 5 estimate (TinyHAR: 11.6 h per fold on Aya's machine, 5.9 h on Nakhoul's).

Usage (from the repo root, inside .venv):
    python -m har.time_har_components --fold 0 --device cpu  --threads 10 --machine-name aya-laptop
    python -m har.time_har_components --fold 0 --device cpu  --threads 8  --machine-name nakhoul-3050
    python -m har.time_har_components --fold 0 --device cuda              --machine-name nakhoul-3050
    python -m har.time_har_components --fold 0 --smoke                    # tiny sizes, pipeline check only

Components (each a row of the components CSV):
    inference          predict_proba on the fold's test windows (held-out HARTH + all HAR70+)
    ce_epoch           one fine-tuning epoch, weighted CE, Adam lr 1e-4, batch 256 (2 epochs timed);
                       the noise-floor trainings (lr 1e-3) have the same cost per epoch
    val_eval           predict_proba on the 3 inner-validation subjects (once per noise-floor epoch)
    park_targets       the original model's fixed 48-segment EG targets on every training window (m = 4)
    park_epoch         one fine-tuning epoch with CE + gamma_P * L_AM (EG m = 4, double backprop)
    snip_scores        SNIP saliency on 10 batches of 256
    ptq8, ptq4         ptq_symmetric
    shap               single-output GradientExplainer, nsamples 200, fold background (100), on all
                       the fold's explain windows; seconds per window
    lci                per-subject LCI on 48 segments, one model pair
Our method (keep-score, L_cons, gate) is not written yet: in the projection its fine-tuning epochs
and targets are "estimated = Park's cost". Pruning masks are not attached here (the pruning code
comes with Step 9), so the fine-tuning epochs are timed on the unpruned model.
Inside park_epoch, CE uses train mode and the EG term eval mode (BatchNorm running statistics are
not updated by the interpolated inputs); Step 9's finetune must use the same convention.

Writes results/har-component-timing/ (or results/har-component-timing-smoke/ with --smoke):
    har-component-timing-tinyhar-fold<k>-<machine>-<device>.csv              one row per component
    har-component-timing-tinyhar-fold<k>-<machine>-<device>-projection.csv   Steps 8 + 9 per fold
    ...-config.json and logs/...log
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import runlog as R
from .attributions import shap_target_class
from .compression import (park_am_loss, park_teacher_attributions, ptq_symmetric, segment_attributions,
                          snip_scores)
from .data import Windows, find_windows, load_harth_windows, load_test_windows, require_harth
from .folds import load_folds
from .paths import BACKGROUND_CSV, EXPLAIN_WINDOWS_CSV, MODEL_SELECTION_DIR, RESULTS_DIR
from .run_model_selection import subsample
from .run_sanity_walking_aid import load_fold_model
from .train import class_weights, predict_proba, set_seed

FROZEN_CONFIG = MODEL_SELECTION_DIR / "har-frozen-model-config.json"
OUT_DIR, SMOKE_DIR = RESULTS_DIR / "har-component-timing", RESULTS_DIR / "har-component-timing-smoke"

FT = {"lr": 1e-4, "batch_size": 256, "m_park": 4, "gamma_p": 50.0, "snip_batches": 10}   # plan 2.5
FULL = {"ce_epochs": 2, "park_epochs": 1, "nsamples": 200, "n_train": None, "n_val": None,
        "n_explain": None, "n_test": None}
SMOKE = {"ce_epochs": 1, "park_epochs": 1, "nsamples": 20, "n_train": 600, "n_val": 200,
         "n_explain": 8, "n_test": 400}

# Work per fold for Steps 8 + 9 (plan Section 5, "Work per fold", main grid, no KD); n_ex = the fold's
# explain windows, n_val = its inner-validation explain windows.
WORK = {
    "noise_floor_epochs": 4 * 30,       # 4 trainings (seeds 0-3) x <= 30 epochs (upper bound)
    "ce_ft_epochs": 2 * 4 * 10,         # L1 + SNIP-score pruning, 4 levels x E_ft = 10
    "park_ft_epochs": 4 * 10 + 4 * 10,  # grid 4 levels x 10 + tuning 4 configurations x 10 (at 70%)
    "ours_ft_epochs": 4 * 10 + 4 * 10,  # same budget; estimated = Park's cost per epoch
    "snip_calls": 4,
    "ptq_calls": 2,                     # 8 and 4 bit (Tulla-style rows cost the same, included below)
    "shap_ex_models": 4 + 2 + 16 + 4,   # noise floor: 4 models + 2 SHAP seeds; 16 pruned; 4 PTQ
    "shap_val_models": 4 + 4 + 8,       # noise floor (4 models), gate (4 levels), tuning (8 configurations)
    "eval_models": 4 + 16 + 4 + 8,      # models predicted on the fold's test windows
    "lci_pairs": 6 + 16 + 4 + 4 + 8,
}
PLAN_ESTIMATE_H = {"aya": 11.6, "nakhoul": 5.9}    # plan Section 5: TinyHAR, Steps 8 + 9 per fold


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def timed(fn, device: str):
    sync(device)
    t = time.perf_counter()
    out = fn()
    sync(device)
    return out, time.perf_counter() - t


def ce_epoch(model, Xtr, ytr, loss_fn, opt, g, batch_size, device):
    model.train()
    n = len(ytr)
    perm = torch.randperm(n, generator=g).to(device)
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        if len(idx) < 2:
            continue
        opt.zero_grad(set_to_none=True)
        loss_fn(model(Xtr[idx]), ytr[idx]).backward()
        opt.step()
    model.eval()


def park_epoch(model, Xtr, ytr, target, A_T, background, loss_fn, opt, g, gen_eg, batch_size, m, gamma, device):
    n = len(ytr)
    perm = torch.randperm(n, generator=g).to(device)
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        if len(idx) < 2:
            continue
        opt.zero_grad(set_to_none=True)
        model.train()
        ce = loss_fn(model(Xtr[idx]), ytr[idx])
        model.eval()                       # EG term: BN running statistics, no dropout
        am = park_am_loss(model, Xtr[idx], target[idx], A_T[idx], background, m=m, generator=gen_eg)
        (ce + gamma * am).backward()
        opt.step()
    model.eval()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--threads", type=int, default=None, help="torch.set_num_threads (plan: 8 Nakhoul, 10 Aya)")
    ap.add_argument("--machine-name", default=R.machine_name())
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    S, dev_name = (SMOKE if args.smoke else FULL), args.device
    out = SMOKE_DIR if args.smoke else OUT_DIR
    base = f"har-component-timing-tinyhar-fold{args.fold}-{args.machine_name}-{dev_name}"
    csv, proj_csv = out / f"{base}.csv", out / f"{base}-projection.csv"
    cfg_path, log = out / f"{base}-config.json", out / "logs" / f"{base}.log"
    log.unlink(missing_ok=True)
    with R.tee_to(log):
        dev = R.device_info(dev_name)
        started, t_all = R.now(), time.perf_counter()
        print(f"har.time_har_components | {' '.join(sys.argv)} | machine {args.machine_name} | device {dev_name} "
              f"{dev['gpu_name']} | CPU {dev['cpu']} | torch {dev['torch_version']} | threads {dev['torch_threads']}")
        frozen = json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))
        assert frozen["model"] == "tinyhar", frozen["model"]

        # ---- data: fold k's training / inner-val / test windows, explain windows, background
        harth = load_harth_windows()
        test = load_test_windows()
        counts = pd.Series(harth["harth_test"].subject).value_counts().to_dict()
        fold = load_folds(harth_test_counts={s: int(c) for s, c in counts.items()})[args.fold]
        tr = harth["harth_train"].for_subjects(fold.train)
        va = harth["harth_test"].for_subjects(fold.val)
        te_g, te_o = test["harth_test"].for_subjects(fold.test_general), test["har70plus_test"]
        require_harth(tr, va, te_g)
        ex = pd.read_csv(EXPLAIN_WINDOWS_CSV, dtype={"subject": str})
        ex_g = ex[(ex.split == "harth_test") & ex.subject.isin(fold.test_general)]
        ex_o = ex[ex.split == "har70plus_test"]
        ex_v = ex[(ex.split == "harth_test") & ex.subject.isin(fold.val)]
        n_ex, n_val = len(ex_g) + len(ex_o), len(ex_v)
        X_ex = np.concatenate([test["harth_test"].X[find_windows(test["harth_test"], "harth_test", ex_g.window_id.tolist())],
                               te_o.X[find_windows(te_o, "har70plus_test", ex_o.window_id.tolist())]])
        ex_subject = np.concatenate([ex_g.subject.to_numpy(), ex_o.subject.to_numpy()])
        bg_ids = pd.read_csv(BACKGROUND_CSV).query("fold == @args.fold").window_id.tolist()
        background = torch.from_numpy(harth["harth_train"].X[find_windows(harth["harth_train"], "harth_train", bg_ids)])
        X_test = np.concatenate([te_g.X, te_o.X])
        if args.smoke:
            tr, va = subsample(tr, S["n_train"], 0), subsample(va, S["n_val"], 1)
            X_ex, ex_subject = X_ex[:S["n_explain"]], ex_subject[:S["n_explain"]]
            X_test = X_test[:S["n_test"]]
        print(f"fold {args.fold}: train {len(tr)} windows ({len(fold.train)} subjects), inner val {len(va)} "
              f"({list(fold.val)}), test {len(X_test)} (HARTH {list(fold.test_general)} + 18 HAR70+), "
              f"explain {len(X_ex)} of n_ex = {n_ex}, inner-val explain n_val = {n_val}, background {len(background)}")

        net, ckpt = load_fold_model(args.fold, frozen["architecture"])
        net.to(dev_name).eval()
        background_d = background.to(dev_name)
        Xtr, ytr = torch.from_numpy(tr.X).to(dev_name), torch.from_numpy(tr.y).to(dev_name)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights(tr.y).to(dev_name))
        print(f"weights {ckpt}")
        rows = []

        def add(component, seconds, n_items, unit, note=""):
            rows.append({"component": component, "seconds": seconds, "n_items": n_items, "unit": unit,
                         "seconds_per_item": seconds / max(n_items, 1), "note": note})
            print(f"  {component:<14} {seconds:9.2f} s  ({n_items} {unit}; {seconds / max(n_items, 1):.5f} s each) {note}")
            R.write_csv(csv, pd.DataFrame(rows))

        # ---- inference and validation evaluation
        predict_proba(net, X_test[:256], dev_name)                     # warm-up
        _, s = timed(lambda: predict_proba(net, X_test, dev_name), dev_name)
        add("inference", s, len(X_test), "windows", "predict_proba, batch 1024")
        _, s = timed(lambda: predict_proba(net, va.X, dev_name), dev_name)
        add("val_eval", s, len(va), "windows", "inner validation, once per noise-floor epoch")

        # ---- plain CE fine-tuning epochs
        set_seed(0)
        m_ce = copy.deepcopy(net)
        opt = torch.optim.Adam(m_ce.parameters(), lr=FT["lr"])
        g = torch.Generator().manual_seed(0)
        for e in range(S["ce_epochs"]):
            _, s = timed(lambda: ce_epoch(m_ce, Xtr, ytr, loss_fn, opt, g, FT["batch_size"], dev_name), dev_name)
            add(f"ce_epoch_{e + 1}", s, 1, "epoch", f"{len(tr)} windows, batch {FT['batch_size']}")
        del m_ce

        # ---- Park: fixed targets of the original model, then one fine-tuning epoch
        target = torch.from_numpy(predict_proba(net, tr.X, dev_name).argmax(1)).to(dev_name)
        A_T, s = timed(lambda: park_teacher_attributions(net, Xtr, target, background_d, m=FT["m_park"], seed=0,
                                                         batch_size=FT["batch_size"]), dev_name)
        add("park_targets", s, len(tr), "windows", f"EG m = {FT['m_park']}, no graph, once per fold")
        m_p = copy.deepcopy(net)
        opt = torch.optim.Adam(m_p.parameters(), lr=FT["lr"])
        g, gen_eg = torch.Generator().manual_seed(0), torch.Generator().manual_seed(1)
        for e in range(S["park_epochs"]):
            _, s = timed(lambda: park_epoch(m_p, Xtr, ytr, target, A_T, background_d, loss_fn, opt, g, gen_eg,
                                            FT["batch_size"], FT["m_park"], FT["gamma_p"], dev_name), dev_name)
            add(f"park_epoch_{e + 1}", s, 1, "epoch", f"CE + {FT['gamma_p']:g} * L_AM, EG m = {FT['m_park']}, create_graph")
        del m_p, A_T

        # ---- SNIP scores, PTQ
        _, s = timed(lambda: snip_scores(net, tr, n_batches=FT["snip_batches"], seed=42,
                                         batch_size=FT["batch_size"]), dev_name)
        add("snip_scores", s, 1, "call", f"{FT['snip_batches']} batches of {FT['batch_size']}")
        for bits in (8, 4):
            _, s = timed(lambda: ptq_symmetric(net, bits), dev_name)
            add(f"ptq{bits}", s, 1, "call", "ptq_symmetric")

        # ---- SHAP on the fold's explain windows
        Xe = torch.from_numpy(X_ex).to(dev_name)
        with torch.no_grad():
            t_ex = net(Xe).argmax(1).cpu().numpy()
        A, s = timed(lambda: shap_target_class(net, background_d, Xe, t_ex, shap_seed=0, nsamples=S["nsamples"]), dev_name)
        assert A.shape == tuple(Xe.shape)
        add("shap", s, len(Xe), "windows", f"single output, nsamples {S['nsamples']}, background {len(background)}, "
                                           f"{len(np.unique(t_ex))} class groups")

        # ---- LCI per subject on 48 segments (one model pair)
        from lci.metric import compute_lci
        A48 = segment_attributions(torch.from_numpy(A)).numpy()
        A48_s = A48 + np.random.default_rng(0).normal(scale=A48.std() * 0.1, size=A48.shape)
        def lci_all():
            return [compute_lci(A48[ex_subject == su], A48_s[ex_subject == su], k=10, omega=0.5)
                    for su in np.unique(ex_subject)]
        _, s = timed(lci_all, "cpu")
        add("lci", s, 1, "model pair", f"{len(np.unique(ex_subject))} subjects, k = 10")

        # ---- projection of Steps 8 + 9 per fold
        r = {x["component"]: x for x in rows}
        ce_s = float(np.median([x["seconds"] for x in rows if x["component"].startswith("ce_epoch")]))
        park_s = float(np.median([x["seconds"] for x in rows if x["component"].startswith("park_epoch")]))
        scale = 1.0
        if args.smoke:      # scale per-window costs to the real fold sizes so the smoke projection has the right shape
            n_tr_full = len(harth["harth_train"].for_subjects(fold.train))
            scale = n_tr_full / len(tr)
        shap_w = r["shap"]["seconds_per_item"] * (200 / S["nsamples"])
        inf_w, val_s = r["inference"]["seconds_per_item"], r["val_eval"]["seconds"]
        n_test_full = len(te_g) + len(te_o)
        proj = [
            ("noise_floor_training", WORK["noise_floor_epochs"] * (ce_s * scale + val_s), True,
             f"{WORK['noise_floor_epochs']} CE epochs + validation each epoch (30-epoch upper bound)"),
            ("l1_snip_finetune", WORK["ce_ft_epochs"] * ce_s * scale + WORK["snip_calls"] * r["snip_scores"]["seconds"], True,
             f"{WORK['ce_ft_epochs']} CE epochs + {WORK['snip_calls']} SNIP score calls"),
            ("park_finetune_incl_tuning", WORK["park_ft_epochs"] * park_s * scale + r["park_targets"]["seconds"] * scale, True,
             f"{WORK['park_ft_epochs']} Park epochs + targets once"),
            ("ours_finetune_incl_tuning", WORK["ours_ft_epochs"] * park_s * scale + r["park_targets"]["seconds"] * scale, False,
             "estimated = Park's cost (our method not implemented yet)"),
            ("ptq", WORK["ptq_calls"] * 2 * max(r["ptq8"]["seconds"], r["ptq4"]["seconds"]), True, "symmetric + Tulla-style, 8 and 4 bit"),
            ("shap", (WORK["shap_ex_models"] * n_ex + WORK["shap_val_models"] * n_val) * shap_w, True,
             f"({WORK['shap_ex_models']} x {n_ex} + {WORK['shap_val_models']} x {n_val}) windows at nsamples 200"),
            ("evaluation", WORK["eval_models"] * n_test_full * inf_w, True, f"{WORK['eval_models']} models x {n_test_full} test windows"),
            ("lci", WORK["lci_pairs"] * r["lci"]["seconds"], True, f"{WORK['lci_pairs']} model pairs"),
        ]
        pdf = pd.DataFrame([{"part": p, "hours": s / 3600, "measured": m, "note": n} for p, s, m, n in proj])
        total = float(pdf.hours.sum())
        key = next((k for k in PLAN_ESTIMATE_H if k in args.machine_name.lower()), None)
        plan_h = PLAN_ESTIMATE_H.get(key)
        pdf = pd.concat([pdf, pd.DataFrame([{"part": "total_steps_8_9_per_fold", "hours": total, "measured": False,
                                             "note": f"plan Section 5 estimate for this machine: {plan_h} h"
                                                     + (f"; ratio {total / plan_h:.2f}; off by more than 30%: "
                                                        f"{abs(total / plan_h - 1) > 0.3}" if plan_h else "")}])])
        pdf.insert(0, "machine_name", args.machine_name); pdf.insert(1, "device", dev_name)
        pdf["park_over_ce_epoch"] = park_s / ce_s
        pdf["smoke"] = args.smoke
        R.write_csv(proj_csv, pdf)
        print(f"\nPark / CE epoch cost: {park_s / ce_s:.2f}x (plan Section 5 assumed 3x; ours assumed 4x, here = Park)")
        print(pdf[["part", "hours", "measured"]].to_string(index=False))

        df = pd.DataFrame(rows)
        df.insert(0, "machine_name", args.machine_name)
        for k, v in {**dev, **R.git_info(), "fold": args.fold, "smoke": args.smoke, "measured_at": R.now()}.items():
            df[k] = v
        R.write_csv(csv, df)
        R.write_json(cfg_path, {"command_line": sys.argv, "machine_name": args.machine_name, "device": dev,
                                **R.git_info(), "code_sha256": R.code_sha256(), "started_at": started,
                                "finished_at": R.now(), "wall_seconds": time.perf_counter() - t_all,
                                "fold": args.fold, "weights": str(ckpt), "frozen_setting": frozen["setting_id"],
                                "protocol": S, "finetune": FT, "work_per_fold": WORK, "plan_estimate_h": PLAN_ESTIMATE_H,
                                "n_train": len(tr), "n_val": len(va), "n_test": len(X_test), "n_ex": n_ex, "n_val_explain": n_val,
                                "smoke_scale_to_full_train": scale if args.smoke else None})
        print(f"\nwall clock {time.perf_counter() - t_all:.1f} s\nsaved {csv}\nsaved {proj_csv}\nsaved {cfg_path}\nsaved {log}")


if __name__ == "__main__":
    main()
