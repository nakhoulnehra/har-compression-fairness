"""CPU inference time and SHAP time per window for the model-choice candidates (plan 2.7, 2.9).

Always runs on CPU (CUDA is hidden before torch is imported), so the numbers are comparable across
candidates; run it on ONE machine (the plan: Aya's laptop) for all models.

Usage (from the repo root):
    python -m har.time_model_selection_cpu                  # all candidates
    python -m har.time_model_selection_cpu --smoke          # a few calls only, to check it works

Per model:
    latency_bs1_ms_median      batch 1, torch.set_num_threads(1), median of 1,000 calls after 50 warm-up
    throughput_ms_per_window   batch 1,024, all threads, median of 5 runs, divided by 1,024
    shap_s_per_window          GradientExplainer, single target output (the model's predicted class),
                               nsamples 200, fold 0's 100-window background, 100 HARTH explain windows
Weights: the fold-0 final checkpoint of the best tuning setting if it exists (copy it from the GPU
machine), otherwise an untrained model of the same architecture (timing depends on the architecture,
not on the weights; recorded in the `weights` column).
Writes <out>/timing/har-cpu-inference-and-shap-time-<machine>.csv, its -config.json and a log.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""   # always CPU

import argparse
import sys
import time

import numpy as np
import pandas as pd
import torch

from . import runlog as R
from .attributions import shap_target_class
from .data import find_windows, load_harth_windows
from .folds import channel_stats, load_folds
from .models import ALL_MODELS, build_model, n_params, resolve_config
from .paths import BACKGROUND_CSV, EXPLAIN_WINDOWS_CSV
from .run_model_selection import DEEP_GRID, TUNING_FOLD, best_setting, dirs, unit_paths
from .train import xgb_predict_proba

FULL = {"warmup": 50, "calls": 1000, "batch": 1024, "batch_runs": 5, "shap_windows": 100, "nsamples": 200}
SMOKE = {"warmup": 3, "calls": 20, "batch": 64, "batch_runs": 2, "shap_windows": 4, "nsamples": 20}


def median_time(fn, warmup: int, calls: int) -> float:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(calls):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", choices=ALL_MODELS, default=list(ALL_MODELS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--machine-name", default=R.machine_name())
    args = ap.parse_args()
    S = SMOKE if args.smoke else FULL
    out, ckpt = dirs(args.smoke)
    csv = out / "timing" / f"har-cpu-inference-and-shap-time-{args.machine_name}.csv"
    cfg_path = csv.with_name(csv.stem + "-config.json")
    log = out / "logs" / f"timing-cpu-{args.machine_name}.log"
    log.unlink(missing_ok=True)
    with R.tee_to(log):
        dev = R.device_info("cpu")
        print(f"har.time_model_selection_cpu | {' '.join(sys.argv)} | machine {args.machine_name} | CPU {dev['cpu']} "
              f"| torch {dev['torch_version']} | threads {dev['torch_threads']}")
        data = load_harth_windows()
        counts = pd.Series(data["harth_test"].subject).value_counts().to_dict()
        fold0 = load_folds(harth_test_counts={s: int(c) for s, c in counts.items()})[TUNING_FOLD]
        bg_ids = pd.read_csv(BACKGROUND_CSV).query("fold == @TUNING_FOLD").window_id.tolist()
        background = torch.from_numpy(data["harth_train"].X[find_windows(data["harth_train"], "harth_train", bg_ids)])
        ex = pd.read_csv(EXPLAIN_WINDOWS_CSV, dtype={"subject": str})
        ex_ids = ex[(ex.split == "harth_test") & ex.subject.isin(fold0.test_general)].window_id.tolist()[:S["shap_windows"]]
        X_ex = data["harth_test"].X[find_windows(data["harth_test"], "harth_test", ex_ids)]
        X_batch = np.resize(data["harth_test"].X[:S["batch"]], (S["batch"], 6, 128)).astype(np.float32)
        mean, std = channel_stats(data["harth_train"], fold0)
        n_threads = torch.get_num_threads()
        rows = []
        for m in args.models:
            try:
                hp, sid, _ = best_setting(out, m)
            except RuntimeError:
                hp = {"max_depth": 3, "learning_rate": 0.1} if m == "xgboost" else DEEP_GRID[0]
                sid = None
            ck = unit_paths(out, ckpt, "final", m, None, TUNING_FOLD, "*")["checkpoint"]
            found = sorted(ck.parent.glob(ck.name))
            print(f"\n[{m}] setting {sid or 'default (no tuning results found)'}; weights: {found[0] if found else 'untrained'}")
            row = {"model": m, "setting_id": sid or "default", "weights": str(found[0]) if found else "untrained"}
            if m == "xgboost":
                import xgboost as xgb
                if not found:
                    print("  no XGBoost checkpoint: skipped (needs a trained booster)")
                    continue
                booster = xgb.Booster()
                booster.load_model(str(found[0]))
                booster.set_param({"nthread": 1})
                one = data["harth_test"].X[:1]
                row["latency_bs1_ms_median"] = 1000 * median_time(lambda: xgb_predict_proba(booster, one), S["warmup"], S["calls"])
                booster.set_param({"nthread": n_threads})
                row["throughput_ms_per_window"] = 1000 * median_time(lambda: xgb_predict_proba(booster, X_batch), 1, S["batch_runs"]) / S["batch"]
                row.update({"n_params": None, "shap_s_per_window": None, "note": "includes feature extraction; SHAP not applicable"})
            else:
                arch = resolve_config(m, hp["width"])
                torch.manual_seed(0)
                net = build_model(m, arch, mean, std)
                if found:
                    net.load_state_dict(torch.load(found[0], map_location="cpu", weights_only=False)["state_dict"])
                net.eval()
                row["n_params"] = n_params(net)
                x1 = torch.from_numpy(data["harth_test"].X[:1])
                torch.set_num_threads(1)
                with torch.no_grad():
                    row["latency_bs1_ms_median"] = 1000 * median_time(lambda: net(x1), S["warmup"], S["calls"])
                torch.set_num_threads(n_threads)
                xb = torch.from_numpy(X_batch)
                with torch.no_grad():
                    row["throughput_ms_per_window"] = 1000 * median_time(lambda: net(xb), 1, S["batch_runs"]) / S["batch"]
                Xe = torch.from_numpy(X_ex)
                with torch.no_grad():
                    target = net(Xe).argmax(1).numpy()
                t = time.perf_counter()
                A = shap_target_class(net, background, Xe, target, shap_seed=0, nsamples=S["nsamples"])
                row["shap_s_per_window"] = (time.perf_counter() - t) / len(Xe)
                assert A.shape == tuple(Xe.shape)
                row["note"] = ""
            row.update({"machine_name": args.machine_name, **dev, **R.git_info(), "measured_at": R.now(),
                        "smoke": args.smoke, **{f"protocol_{k}": v for k, v in S.items()}})
            print(f"  params {row['n_params']}, latency {row['latency_bs1_ms_median']:.3f} ms, throughput "
                  f"{row['throughput_ms_per_window']:.4f} ms/window, SHAP {row['shap_s_per_window']} s/window")
            rows.append(row)
            R.write_csv(csv, pd.DataFrame(rows))       # saved after every model
        R.write_json(cfg_path, {"command_line": sys.argv, "protocol": S, "machine_name": args.machine_name,
                                "device": dev, **R.git_info(), "code_sha256": R.code_sha256(),
                                "background": f"fold {TUNING_FOLD} SHAP background ({len(bg_ids)} windows)",
                                "explain_windows": ex_ids})
        print(f"\nsaved {csv}\nsaved {cfg_path}\nsaved {log}")


if __name__ == "__main__":
    main()
