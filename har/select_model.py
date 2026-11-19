"""Merge the model-choice results of all machines, apply the selection rule (plan 2.9) and freeze the
chosen model.

Usage (from the repo root, after copying every machine's results/har-model-selection/ files here):
    python -m har.select_model --merge-only          # merge + summary, no decision
    python -m har.select_model                       # merge + summary + rule + freeze
    python -m har.select_model --timing-machine NAME # which machine's CPU timing to use (if several)

Rule (cost-based, decided 2026-11-19; F1 = held-out HARTH 7-class macro-F1 pooled over the 22
subjects' windows, in points; candidates = the four deep models):
    1. F* = the best F1.
    2. Eligible = every candidate with F1 >= F* - 2.
    3. Choose the eligible candidate with the lowest CPU SHAP time per window
       (har.time_model_selection_cpu); an exact tie is broken by the smaller parameter count.
XGBoost is the accuracy reference only. The frozen config is written once; --force overwrites it.

Writes to results/har-model-selection/:
    har-model-selection-tuning.csv              every tuning setting tried, with its validation score
    har-model-selection-final-per-fold.csv      one row per model and fold
    har-model-selection-per-subject-metrics.csv per model and held-out HARTH subject
    har-model-selection-summary.csv             per model: pooled F1, per-subject F1, stairs F1, size, timing
    har-frozen-model-config.json                the chosen model, its settings and the rule trace
    logs/select-model-<machine>.log
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from . import runlog as R
from .data import CLASS_NAMES
from .models import ALL_MODELS, DEEP_MODELS, SOURCES, resolve_config
from .paths import MODEL_SELECTION_DIR, MODEL_SELECTION_SMOKE_DIR
from .run_model_selection import COMPANION_SUFFIXES, SEED, grid, setting_id
from .train import TrainConfig, f1_from_confusion

MARGIN = 2.0          # F1 points
N_FOLDS = 5


def read_rows(d, pattern: str) -> pd.DataFrame:
    files = sorted(p for p in d.glob(pattern) if not p.name.endswith(COMPANION_SUFFIXES))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(p, dtype={"machine_name": str, "gpu_name": str}).assign(row_file=str(p)) for p in files],
                   ignore_index=True)
    df["gpu_name"] = df["gpu_name"].fillna("")
    return df[df.status == "done"]


def dedupe(df: pd.DataFrame, keys: list[str], prefer: str | None) -> pd.DataFrame:
    """One row per unit. Identical settings from two machines -> keep --prefer-machine's (or error)."""
    out = []
    for _, g in df.groupby(keys, sort=True):
        if len(g) == 1:
            out.append(g)
            continue
        if g.settings_hash.nunique() > 1:
            raise RuntimeError(f"unit {dict(zip(keys, g.iloc[0][keys]))} has rows with different settings: "
                               f"{g.row_file.tolist()}")
        if prefer and (g.machine_name == prefer).any():
            out.append(g[g.machine_name == prefer].head(1))
        else:
            raise RuntimeError(f"unit {dict(zip(keys, g.iloc[0][keys]))} was run on several machines "
                               f"{g.machine_name.tolist()}; pass --prefer-machine NAME")
    return pd.concat(out, ignore_index=True) if out else df


def summarise(final: pd.DataFrame, per_subject: pd.DataFrame, confusions: dict, timing: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    for m in [x for x in ALL_MODELS if x in set(final.model)]:
        f = final[final.model == m]
        cm = sum(confusions[(m, int(k))] for k in f.fold)
        f1 = f1_from_confusion(cm)
        present = cm.sum(axis=1) > 0
        ps = per_subject[per_subject.model == m]
        r = {"model": m, "setting_id": f.setting_id.iloc[0], "n_folds": len(f), "n_subjects": ps.subject.nunique(),
             "macro_f1_pooled7": float(f1[present].mean()), "accuracy_pooled": float(np.trace(cm) / cm.sum()),
             "non_stair_macro_f1_subject_mean": ps.non_stair_macro_f1.mean(),
             "non_stair_macro_f1_subject_sd": ps.non_stair_macro_f1.std(ddof=1),
             "accuracy_subject_mean": ps.accuracy.mean(), "accuracy_subject_sd": ps.accuracy.std(ddof=1),
             "stairs_up_f1_pooled": float(f1[2]), "stairs_down_f1_pooled": float(f1[3]),
             **{f"f1_pooled_{c}": float(v) for c, v in zip(CLASS_NAMES, f1)},
             "n_params": f.n_params.iloc[0], "machines": ";".join(sorted(set(f.machine_name))),
             "devices": ";".join(sorted(set((f.device + " " + f.gpu_name).str.strip()))),
             "source_check": SOURCES[m][1]}
        if timing is not None and (timing.model == m).any():
            t = timing[timing.model == m].iloc[0]
            r.update({"latency_bs1_ms_median": t.latency_bs1_ms_median, "throughput_ms_per_window": t.throughput_ms_per_window,
                      "shap_s_per_window": t.shap_s_per_window, "timing_machine": t.machine_name, "timing_weights": t.weights})
        rows.append(r)
    return pd.DataFrame(rows)


def apply_rule(s: dict[str, dict]) -> tuple[str, list[str]]:
    """Cost-based rule. s[model] = {"f1": points, "n_params", "shap_s"} for the 4 deep candidates:
    among the candidates within MARGIN points of the best F1, the lowest CPU SHAP time per window;
    an exact tie in SHAP time goes to the smaller parameter count."""
    missing = [m for m in DEEP_MODELS if m not in s]
    if missing:
        raise ValueError(f"the rule needs all deep candidates; missing {missing}")
    bad = [m for m in DEEP_MODELS if s[m].get("shap_s") is None or not np.isfinite(s[m]["shap_s"])]
    if bad:
        raise ValueError(f"no CPU SHAP time for {bad}; run `python -m har.time_model_selection_cpu`")
    F = {m: s[m]["f1"] for m in DEEP_MODELS}
    best = max(F, key=F.get)
    trace = ["F1 (points): " + ", ".join(f"{m} {F[m]:.2f}" for m in DEEP_MODELS),
             f"step 1: best F1 {F[best]:.2f} ({best})"]
    eligible = [m for m in DEEP_MODELS if F[m] >= F[best] - MARGIN]
    trace.append(f"step 2: eligible (F1 >= {F[best] - MARGIN:.2f}): {eligible}")
    choice = min(eligible, key=lambda m: (s[m]["shap_s"], s[m]["n_params"]))
    trace.append("step 3: CPU SHAP s/window " + ", ".join(f"{m} {s[m]['shap_s']:.4f}" for m in eligible)
                 + f" -> lowest: {choice}" + ("" if sum(s[m]["shap_s"] == s[choice]["shap_s"] for m in eligible) == 1
                                              else " (tie on SHAP time, fewer parameters)"))
    return choice, trace


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--prefer-machine", default=None)
    ap.add_argument("--timing-machine", default=None)
    ap.add_argument("--smoke", action="store_true", help="use results/har-model-selection-smoke/")
    ap.add_argument("--force", action="store_true", help="overwrite an existing frozen config")
    args = ap.parse_args()
    out = MODEL_SELECTION_SMOKE_DIR if args.smoke else MODEL_SELECTION_DIR
    log = out / "logs" / f"select-model-{R.machine_name()}.log"
    log.unlink(missing_ok=True)
    with R.tee_to(log):
        print(f"har.select_model | {' '.join(sys.argv)}")
        tune = dedupe(read_rows(out / "tuning", "har-tune-*.csv"), ["model", "setting_id"], args.prefer_machine)
        final = dedupe(read_rows(out / "final", "har-final-*.csv"), ["model", "fold"], args.prefer_machine)
        if final.empty:
            raise SystemExit(f"no final results in {out / 'final'}")
        problems = []
        for m in sorted(set(final.model)):
            t = tune[tune.model == m]
            exp = {setting_id(m, hp) for hp in grid(m)}
            if set(t.setting_id) != exp:
                problems.append(f"{m}: tuning settings {sorted(set(t.setting_id))}, expected {sorted(exp)}")
            f = final[final.model == m]
            if not args.smoke and sorted(f.fold) != list(range(N_FOLDS)):
                problems.append(f"{m}: final folds {sorted(f.fold)}, expected 0-4")
            if f.setting_id.nunique() != 1:
                problems.append(f"{m}: folds trained with different settings {sorted(set(f.setting_id))}")
        per_subject, confusions = [], {}
        for _, r in final.iterrows():
            base = r.row_file[:-4]
            per_subject.append(pd.read_csv(base + "-per-subject.csv", dtype={"subject": str, "machine_name": str}))
            confusions[(r.model, int(r.fold))] = pd.read_csv(base + "-confusion.csv", index_col=0).to_numpy()
        per_subject = pd.concat(per_subject, ignore_index=True)
        timing = None
        tfiles = sorted((out / "timing").glob("har-cpu-inference-and-shap-time-*.csv"))
        tfiles = [p for p in tfiles if not p.name.endswith("-config.json")]
        if args.timing_machine:
            tfiles = [p for p in tfiles if p.stem.endswith(args.timing_machine)]
        if len(tfiles) == 1:
            timing = pd.read_csv(tfiles[0], dtype={"machine_name": str})
        elif len(tfiles) > 1:
            problems.append(f"several timing files {[p.name for p in tfiles]}: pass --timing-machine")
        summary = summarise(final, per_subject, confusions, timing)
        files = {"tuning": out / "har-model-selection-tuning.csv", "final": out / "har-model-selection-final-per-fold.csv",
                 "per_subject": out / "har-model-selection-per-subject-metrics.csv",
                 "summary": out / "har-model-selection-summary.csv"}
        R.write_csv(files["tuning"], tune.sort_values(["model", "setting_id"]))
        R.write_csv(files["final"], final.sort_values(["model", "fold"]))
        R.write_csv(files["per_subject"], per_subject.sort_values(["model", "fold", "subject"]))
        R.write_csv(files["summary"], summary)
        cols = [c for c in ["model", "setting_id", "macro_f1_pooled7", "non_stair_macro_f1_subject_mean", "accuracy_pooled",
                            "stairs_up_f1_pooled", "stairs_down_f1_pooled", "n_params", "latency_bs1_ms_median",
                            "shap_s_per_window"] if c in summary]
        print(summary[cols].to_string(index=False))
        for k, p in files.items():
            print(f"saved {p}")
        if problems:
            print("\nNOT READY TO SELECT:\n  " + "\n  ".join(problems))
        if args.merge_only or problems:
            print(f"saved {log}")
            return
        if timing is None:
            raise SystemExit("no CPU timing file: run `python -m har.time_model_selection_cpu` first")
        s = {r.model: {"f1": 100 * r.macro_f1_pooled7, "n_params": r.n_params, "shap_s": r.shap_s_per_window,
                       "latency_ms": r.latency_bs1_ms_median} for r in summary.itertuples() if r.model in DEEP_MODELS}
        choice, trace = apply_rule(s)
        print("\nselection rule:\n  " + "\n  ".join(trace) + f"\nCHOSEN: {choice}")
        frozen = out / "har-frozen-model-config.json"
        if frozen.exists() and not args.force:
            raise SystemExit(f"{frozen} already exists; the model is frozen once (use --force only to redo it deliberately)")
        row = summary[summary.model == choice].iloc[0]
        t = tune[(tune.model == choice) & (tune.setting_id == row.setting_id)].iloc[0]
        hp = next(h for h in grid(choice) if setting_id(choice, h) == row.setting_id)
        R.write_json(frozen, {
            "model": choice, "setting_id": row.setting_id, "hyperparameters": hp,
            "architecture": resolve_config(choice, hp["width"]), "train_config": TrainConfig(lr=hp["lr"]).to_dict(),
            "seed": SEED, "n_params": int(row.n_params), "source": SOURCES[choice][0], "source_check": SOURCES[choice][1],
            "rule": __doc__.split("Rule")[1].split("XGBoost")[0].strip(), "rule_trace": trace,
            "summary": json.loads(summary.to_json(orient="records")), "tuning_val_f1": float(t.best_val_macro_f1_pooled7),
            "frozen_at": R.now(), "frozen_on": R.machine_name(), **R.git_info(), "code_sha256": R.code_sha256(),
            "inputs": {k: str(p) for k, p in files.items()} | {"timing": str(tfiles[0])}})
        print(f"saved {frozen}\nsaved {log}")


if __name__ == "__main__":
    main()
