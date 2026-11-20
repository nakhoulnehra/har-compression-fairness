"""XGP pilot: fold 0, 70% sparsity, HARTH inner validation only (never HAR70+).

Compares, on the fold's 3 inner-validation subjects, L1, SNIP-score pruning, Park (its 4 tuning
configurations: gamma_P in {1, 50} x lr in {1e-4, 3e-4}) and XGP (its 4 tuning configurations:
beta in {0.3, 0.7} x gamma in {0.1, 1.0}) on LCI_val and inner-val macro-F1, and applies the plan's
tuning rule (highest LCI_val among configurations within 1 pt of L1's macro-F1) to Park and XGP.
It is the Step 9 tuning of fold 0, run on its own so the method can be checked before the grid.

    LCI_val   mean over the 3 inner-validation subjects of per-subject LCI(original, model) on 48
              segments (k = 10, omega = 0.5), GradientExplainer nsamples 200, SHAP seed 0, target =
              the original model's predicted class, on the fold's inner-validation explain windows.
    macro-F1  pooled 7-class macro-F1 on all the inner-validation subjects' non-overlapping windows.
    gate      XGP only: LCI_val of the pruned model before fine-tuning vs tau_k of the fold (from
              `python -m har.run_noise_floor`); below tau_k -> CE + gamma L_cons, else CE. When the
              gate picks CE, gamma has no effect, so the two gamma configurations of that beta are one
              run (recorded for both, `shared_with`).
Every pruning method: E_ft = 10 epochs, Adam (lr 1e-4 unless the configuration says otherwise),
batch 256, class-weighted CE, batch-order seed 0, EG seed 1 for the attribution terms; matched
n_nonzero is asserted. The pre-fine-tuning LCI_val and F1 of every pruned start are logged too.

Usage (from the repo root, inside .venv; needs fold 0's noise floor first):
    python -m har.run_noise_floor --folds 0 --device cuda --machine-name nakhoul-3050
    python -m har.run_xgp_pilot --fold 0 --device cuda --machine-name nakhoul-3050
    python -m har.run_xgp_pilot --fold 0 --smoke             # tiny sizes, 1 epoch, nsamples 20 (CPU check)
Options: --tau X overrides tau_k; --noise-floor-machine NAME picks the tau file when several exist.

Writes results/har-xgp-pilot/ (results/har-xgp-pilot-smoke/ with --smoke):
    har-xgp-pilot-tinyhar-fold<k>-70pct-inner-val-<machine>-runs.csv              one row per run
    ...-per-subject-lci.csv      per run, before and after fine-tuning, per inner-val subject
    ...-finetune-history.csv     per run and epoch: CE, attribution term, inner-val F1
    ...-config.json and logs/...log
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS; must precede CUDA init

import argparse
import copy
import json
import sys
import time

import numpy as np
import pandas as pd
import torch

from . import runlog as R
from .attributions import mean_lci, shap_cached, subject_lci
from .compression import (E_FT, FT_LR, finetune, l1_scores, prune_by_score, size_record, snip_scores,
                          teacher_segment_attributions, unit_norm)
from .data import find_windows, load_harth_windows, require_harth
from .folds import load_folds
from .paths import (EXPLAIN_WINDOWS_CSV, MODEL_SELECTION_DIR, NOISE_FLOOR_DIR, NOISE_FLOOR_SMOKE_DIR, SHAP_CACHE_DIR,
                    XGP_PILOT_DIR, XGP_PILOT_SMOKE_DIR)
from .run_model_selection import subsample
from .run_noise_floor import fold_background
from .run_sanity_walking_aid import load_fold_model
from .train import evaluate, predict_proba
from .xgp import (PARK_TUNING, XGP, alignment_importance, gate_branch, make_extra_loss, pruning_scores,
                  reference_windows, select_config)

FROZEN_CONFIG = MODEL_SELECTION_DIR / "har-frozen-model-config.json"
LEVEL = XGP["tune_level"]
NSAMPLES = 200
SMOKE = {"epochs": 1, "n_train": 600, "n_val": 200, "per_subject": 3, "nsamples": 20, "n_ref": 35,
         "m_score": 2, "m_cons": 2, "m_park": 2}


def read_tau(k: int, smoke: bool, machine: str | None) -> tuple[float, str]:
    d = NOISE_FLOOR_SMOKE_DIR if smoke else NOISE_FLOOR_DIR
    found = sorted(d.glob(f"har-noise-floor-tinyhar-fold{k}-*-tau.csv"))
    if machine:
        found = [p for p in found if p.name == f"har-noise-floor-tinyhar-fold{k}-{machine}-tau.csv"]
    if len(found) != 1:
        raise FileNotFoundError(f"need exactly one fold-{k} tau file in {d} (run `python -m har.run_noise_floor "
                                f"--folds {k}` first, or pass --noise-floor-machine / --tau); found {found}")
    row = pd.read_csv(found[0]).iloc[0]
    assert int(row.fold) == k
    return float(row.tau_k), str(found[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--machine-name", default=R.machine_name())
    ap.add_argument("--tau", type=float, default=None, help="override tau_k (default: the noise-floor tau file)")
    ap.add_argument("--noise-floor-machine", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    k, dev_name, S = args.fold, args.device, SMOKE if args.smoke else {}
    out = XGP_PILOT_SMOKE_DIR if args.smoke else XGP_PILOT_DIR
    base = f"har-xgp-pilot-tinyhar-fold{k}-{int(round(LEVEL * 100))}pct-inner-val-{args.machine_name}"
    p_runs, p_subj, p_hist = (out / f"{base}-{n}.csv" for n in ("runs", "per-subject-lci", "finetune-history"))
    cfg_path, log = out / f"{base}-config.json", out / "logs" / f"{base}.log"
    log.unlink(missing_ok=True)
    epochs = S.get("epochs", E_FT)
    nsamples = S.get("nsamples", NSAMPLES)
    m_score, m_cons, m_park = S.get("m_score", XGP["m_score"]), S.get("m_cons", XGP["m_cons"]), S.get("m_park", PARK_TUNING["m"])
    cache = SHAP_CACHE_DIR / ("smoke" if args.smoke else "")
    with R.tee_to(log):
        dev = R.device_info(dev_name)
        t_all, started = time.perf_counter(), R.now()
        print(f"har.run_xgp_pilot | {' '.join(sys.argv)} | machine {args.machine_name} | device {dev_name} "
              f"{dev['gpu_name']} | torch {dev['torch_version']} | started {started}")
        frozen = json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))
        assert frozen["model"] == "tinyhar", frozen["model"]

        # ---- HARTH only: training subjects (fine-tuning, scores) and inner-validation subjects (everything else)
        harth = load_harth_windows()
        counts = pd.Series(harth["harth_test"].subject).value_counts().to_dict()
        fold = load_folds(harth_test_counts={s: int(c) for s, c in counts.items()})[k]
        tr = harth["harth_train"].for_subjects(fold.train)
        va = harth["harth_test"].for_subjects(fold.val)
        require_harth(tr, va, harth["harth_test"])
        assert not (set(tr.subjects) | set(va.subjects)) & (set(fold.test_general) | set(fold.test_older))
        if args.smoke:
            tr, va = subsample(tr, S["n_train"], 0), subsample(va, S["n_val"], 1)
        ex = pd.read_csv(EXPLAIN_WINDOWS_CSV, dtype={"subject": str})
        ex = ex[(ex.split == "harth_test") & ex.subject.isin(fold.val)].sort_values(["subject", "pick_order"])
        if args.smoke:
            ex = ex.groupby("subject").head(S["per_subject"])
        ids = ex.window_id.tolist()
        X_v = harth["harth_test"].X[find_windows(harth["harth_test"], "harth_test", ids)]
        subj_v = ex.subject.to_numpy()
        assert set(subj_v) == set(fold.val), (set(subj_v), fold.val)
        tau, tau_src = (args.tau, "--tau") if args.tau is not None else read_tau(k, args.smoke, args.noise_floor_machine)
        print(f"fold {k}: train {len(tr)} windows ({len(fold.train)} subjects), inner val {len(va)} windows "
              f"({list(fold.val)}), inner-val explain {len(ids)}; tau_{k} = {tau:.4f} ({tau_src}); level {LEVEL:.0%}, "
              f"E_ft {epochs}, nsamples {nsamples}")

        net0, ck = load_fold_model(k, frozen["architecture"])
        net0.to(dev_name).eval()
        bg = fold_background(harth, k).to(dev_name)
        Xv_d = torch.from_numpy(X_v).to(dev_name)
        target_v = predict_proba(net0, X_v, dev_name).argmax(1)
        t0 = time.perf_counter()
        A_ref = shap_cached(net0, bg, Xv_d, target_v, ids, 0, nsamples, cache, f"har-noise-floor-fold{k}-inner_val")
        print(f"original model: SHAP on inner-val explain windows {time.perf_counter() - t0:.1f} s; weights {ck}")

        # ---- fixed references and scores (all from the original model, HARTH training windows)
        times = {}
        Xtr = torch.from_numpy(tr.X).to(dev_name)
        target_tr = torch.from_numpy(predict_proba(net0, tr.X, dev_name).argmax(1)).to(dev_name)
        t0 = time.perf_counter()
        A0_tr = teacher_segment_attributions(net0, Xtr, target_tr, bg, m=m_cons, seed=0).to(dev_name)
        park_ref = unit_norm(A0_tr)
        times["teacher_attributions"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        snip = snip_scores(net0, tr)
        times["snip_scores"] = time.perf_counter() - t0
        ref_idx = reference_windows(tr, n=S.get("n_ref", XGP["n_ref"]))
        t0 = time.perf_counter()
        imp = alignment_importance(net0, Xtr[ref_idx], target_tr[ref_idx], bg, m=m_score, eg_seeds=XGP["eg_seeds"])
        times["xgp_importance"] = time.perf_counter() - t0
        print(f"references: teacher EG (m = {m_cons}) {times['teacher_attributions']:.1f} s, SNIP {times['snip_scores']:.1f} s, "
              f"XGP importance (m = {m_score}, {len(ref_idx)} windows, EG seeds {XGP['eg_seeds']}) {times['xgp_importance']:.1f} s")

        runs, subj_rows, hist_rows = [], [], []

        def assess(net, run_id: str, stage: str) -> dict:
            t = time.perf_counter()
            A = shap_cached(net, bg, Xv_d, target_v, ids, 0, nsamples, cache, f"har-xgp-pilot-fold{k}")
            rows = subject_lci(A_ref, A, subj_v)
            for r in rows:
                subj_rows.append({"run_id": run_id, "stage": stage, **r})
            ev = evaluate(va.y, predict_proba(net, va.X, dev_name).argmax(1), va.subject)
            return {"lci_val": mean_lci(rows), "shap_corr_val": float(np.mean([r["shap_corr"] for r in rows])),
                    "jaccard_val": float(np.mean([r["jaccard"] for r in rows])),
                    "n_skipped": int(sum(r["n_skipped"] for r in rows)), "val_f1": ev["macro_f1_pooled7"],
                    "val_accuracy": ev["accuracy_pooled"], "val_non_stair_f1_subject_mean": ev["non_stair_macro_f1_subject_mean"],
                    "shap_seconds": time.perf_counter() - t}

        def save():
            R.write_csv(p_runs, pd.DataFrame(runs))
            R.write_csv(p_subj, pd.DataFrame(subj_rows))
            R.write_csv(p_hist, pd.DataFrame(hist_rows))

        ev0 = evaluate(va.y, predict_proba(net0, va.X, dev_name).argmax(1), va.subject)
        runs.append({"run_id": "original", "method": "original", "lci_val": 1.0, "val_f1": ev0["macro_f1_pooled7"],
                     "val_accuracy": ev0["accuracy_pooled"], "val_non_stair_f1_subject_mean": ev0["non_stair_macro_f1_subject_mean"],
                     **{k_: v for k_, v in size_record(net0).items() if k_ != "per_layer_sparsity"}})

        # ---- pruned starts (before fine-tuning): L1, SNIP, keep-score per beta; the gate for XGP
        starts = {"l1": prune_by_score(net0, pruning_scores("l1", net0), LEVEL),
                  "snip": prune_by_score(net0, pruning_scores("snip", net0, snip=snip), LEVEL)}
        for b in XGP["betas"]:
            starts[f"keep_beta{b}"] = prune_by_score(net0, pruning_scores("xgp", net0, importance=imp, beta=b), LEVEL)
        pre = {}
        for name, net in starts.items():
            pre[name] = assess(net, f"start_{name}", "pre_ft")
            print(f"  start {name:<14} before fine-tuning: LCI_val {pre[name]['lci_val']:.4f}, F1 {pre[name]['val_f1']:.4f}")
        nnz = {name: size_record(net)["n_nonzero"] for name, net in starts.items()}
        assert len(set(nnz.values())) == 1, f"n_nonzero differs across methods: {nnz}"

        def run(run_id, method, start, *, repair=None, lr=FT_LR, weight=None, extra=None, **meta):
            print(f"\n=== {run_id}: start {start}, repair {repair or 'none'}, lr {lr:g}" + (f", weight {weight:g}" if weight else ""))
            net = copy.deepcopy(starts[start])          # same pruned start, masks included
            info = finetune(net, tr, va, epochs=epochs, lr=lr, seed=0, device=dev_name, extra_loss=extra)
            for h in info["history"]:
                hist_rows.append({"run_id": run_id, **h})
            rec = size_record(net)
            assert rec["n_nonzero"] == nnz[start], (rec["n_nonzero"], nnz[start])
            assert all(torch.all(dict(net.named_parameters())[n][~mk.to(dev_name)] == 0) for n, mk in net.prune_masks.items())
            a = assess(net, run_id, "post_ft")
            row = {"run_id": run_id, "method": method, "start": start, "repair": repair or "none", "lr": lr,
                   "weight": weight, **meta, "tau_k": tau, "pre_ft_lci_val": pre[start]["lci_val"],
                   "pre_ft_val_f1": pre[start]["val_f1"], **a, "epochs_run": info["epochs_run"],
                   "finetune_seconds": info["seconds"], **{k_: v for k_, v in rec.items() if k_ != "per_layer_sparsity"},
                   "level": LEVEL, "machine_name": args.machine_name, "device": dev_name}
            runs.append(row)
            save()
            print(f"    -> LCI_val {a['lci_val']:.4f} (before ft {pre[start]['lci_val']:.4f}), F1 {a['val_f1']:.4f}, "
                  f"{info['seconds']:.1f} s")
            return row

        lcons = lambda gamma: make_extra_loss("lcons", X=Xtr, target=target_tr, reference=A0_tr, baselines=bg,
                                              gamma=gamma, m=m_cons, seed=1)
        park = lambda gamma: make_extra_loss("park", X=Xtr, target=target_tr, reference=park_ref, baselines=bg,
                                             gamma=gamma, m=m_park, seed=1)
        l1_row = run("l1", "l1", "l1")
        run("snip", "snip", "snip")
        for gp in PARK_TUNING["gamma_p"]:
            for lr in PARK_TUNING["lr"]:
                run(f"park_gammap{gp:g}_lr{lr:g}", "park", "l1", repair="park", lr=lr, weight=gp, extra=park(gp),
                    gamma_p=gp)
        for b in XGP["betas"]:
            start = f"keep_beta{b}"
            branch = gate_branch(pre[start]["lci_val"], tau)
            print(f"\nXGP beta {b}: gate LCI_val {pre[start]['lci_val']:.4f} vs tau_{k} {tau:.4f} -> {branch}")
            first = None
            for gm in XGP["gammas"]:
                rid = f"xgp_beta{b:g}_gamma{gm:g}"
                meta = {"beta": b, "gamma": gm, "gate_lci_val": pre[start]["lci_val"], "gate_branch": branch}
                if branch == "ce" and first is not None:      # gamma unused: identical run, recorded for both
                    runs.append({**first, "run_id": rid, "weight": None, **meta, "shared_with": first["run_id"]})
                    save()
                    continue
                if branch == "ce":
                    first = run(rid, "xgp", start, **meta)
                else:
                    run(rid, "xgp", start, repair="lcons", weight=gm, extra=lcons(gm), **meta)

        # ---- the plan's tuning rule, per method family
        df = pd.DataFrame(runs)
        l1_f1 = float(l1_row["val_f1"])
        df["f1_within_1pt_of_l1"] = df.val_f1 >= l1_f1 - XGP["f1_tolerance"] - 1e-12
        df["selected"], df["selection_fallback"] = False, False
        chosen = {}
        for fam in ("park", "xgp"):
            rows = df[df.method == fam].to_dict("records")
            c, fb = select_config(rows, l1_f1)
            df.loc[df.run_id == c["run_id"], ["selected", "selection_fallback"]] = [True, fb]
            chosen[fam] = (c["run_id"], fb)
        runs[:] = df.to_dict("records")
        save()
        times["wall"] = time.perf_counter() - t_all
        R.write_json(cfg_path, {"command_line": sys.argv, "machine_name": args.machine_name, "device": dev, **R.git_info(),
                                "code_sha256": R.code_sha256(), "started_at": started, "finished_at": R.now(),
                                "fold": k, "level": LEVEL, "tau_k": tau, "tau_source": tau_src, "epochs": epochs,
                                "nsamples": nsamples, "xgp": XGP, "park_tuning": PARK_TUNING, "m_score": m_score,
                                "m_cons": m_cons, "m_park": m_park, "smoke": args.smoke, "smoke_sizes": S or None,
                                "weights": str(ck), "inner_val_subjects": list(fold.val), "n_train": len(tr),
                                "n_val": len(va), "n_inner_val_explain": len(ids), "n_ref": len(ref_idx),
                                "n_nonzero": nnz, "chosen": chosen, "l1_val_f1": l1_f1, "seconds": times})
        cols = ["run_id", "gate_branch", "pre_ft_lci_val", "lci_val", "shap_corr_val", "jaccard_val", "val_f1",
                "val_non_stair_f1_subject_mean", "n_nonzero", "f1_within_1pt_of_l1", "selected"]
        show = df[[c for c in cols if c in df.columns]]
        with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.4f}".format):
            print("\n" + show.to_string(index=False))
        print(f"\ntau_{k} = {tau:.4f}; L1 F1 {l1_f1:.4f} -> eligible F1 >= {l1_f1 - XGP['f1_tolerance']:.4f}")
        for fam, (rid, fb) in chosen.items():
            print(f"chosen {fam}: {rid}" + ("  (FALLBACK: no configuration within 1 pt of L1; highest F1)" if fb else ""))
        print(f"wall clock {times['wall']:.1f} s")
        for p in (p_runs, p_subj, p_hist, cfg_path, log):
            print(f"saved {p}")


if __name__ == "__main__":
    main()
