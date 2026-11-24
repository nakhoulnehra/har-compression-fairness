"""--analyze: merge the per-fold files and apply the pre-registered rules (rules.py).

Writes to results/ (or results-smoke/):
    bitred-tinyhar-summary-mean-sd-over-folds.csv      per model: mean / SD / min / max over folds
    bitred-tinyhar-probe-headroom.csv                  R1: LCI and F1 drop from uniform 4-bit to 3 / 2 bits
    bitred-tinyhar-allocation-comparisons.csv          R2: XGQ against each comparator per budget
    bitred-tinyhar-sensitivity-mean-over-folds.csv     per tensor x bits: the three sensitivities
    bitred-tinyhar-sensitivity-rank-agreement.csv      per fold: Spearman between the methods' sensitivities
    bitred-tinyhar-allocations-bits-per-tensor.csv     bits per tensor, per method x budget x fold
    bitred-tinyhar-decision.json                       R1-R3
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from har import runlog as R

from .rules import METHODS, decide, spearman

P = "bitred-tinyhar"
METRICS = ["lci", "worst_subject_lci", "rho", "jaccard", "f1", "accuracy", "retention", "size_kb", "mean_bits"]


def load(out, prefix: str, use_machine: dict[int, str]) -> tuple[dict[str, pd.DataFrame], dict[int, str]]:
    found = {}
    for p in sorted(out.glob(f"{prefix}-fold*-models.csv")):
        stem = p.name[len(f"{prefix}-fold"):-len("-models.csv")]
        k, machine = stem.split("-", 1)
        found.setdefault(int(k), []).append(machine)
    chosen = {}
    for k, ms in sorted(found.items()):
        if k in use_machine:
            if use_machine[k] not in ms:
                raise RuntimeError(f"fold {k}: no files from {use_machine[k]} (found {ms})")
            chosen[k] = use_machine[k]
        elif len(ms) == 1:
            chosen[k] = ms[0]
        else:
            raise RuntimeError(f"fold {k} has files from {ms}; choose one with --use-machine {k}=<machine>")
    tables = {}
    for t in ("models", "per-subject-lci", "sensitivity", "hessian-traces", "allocations"):
        parts = [pd.read_csv(out / f"{prefix}-fold{k}-{m}-{t}.csv", dtype={"subject": str})
                 for k, m in chosen.items() if (out / f"{prefix}-fold{k}-{m}-{t}.csv").exists()]
        tables[t] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return tables, chosen


def summary(models: pd.DataFrame) -> pd.DataFrame:
    keys = ["model_id", "family", "granularity"]
    g = models.groupby(keys, dropna=False, sort=False)
    out = g.size().rename("n_folds").reset_index()
    for m in METRICS:
        s = g[m].agg(["mean", "std", "min", "max"]).reset_index(drop=True)
        s.columns = [f"{m}_{c}" for c in ("mean", "sd", "min", "max")]
        out = pd.concat([out, s], axis=1)
    out["share_folds_below_tau"] = g["below_tau"].mean().to_numpy()
    return out


def sensitivity_tables(sens: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [f"omega_{m}" for m in METHODS] + ["lci", "lci_rho", "lci_jaccard"]
    mean = sens.groupby(["tensor", "bits"], sort=False)[cols].agg(["mean", "std"]).reset_index()
    mean.columns = ["_".join(c).rstrip("_") for c in mean.columns]
    rows = []
    for k, d in sens.groupby("fold"):
        for i, a in enumerate(METHODS):
            for b in METHODS[i + 1:]:
                rows.append({"fold": k, "a": a, "b": b, "scope": "all tensor x bits",
                             "spearman": spearman(d[f"omega_{a}"], d[f"omega_{b}"])})
                for bits, db in d.groupby("bits"):
                    rows.append({"fold": k, "a": a, "b": b, "scope": f"tensors at {int(bits)} bits",
                                 "spearman": spearman(db[f"omega_{a}"], db[f"omega_{b}"])})
    return mean, pd.DataFrame(rows)


def analyze(args) -> None:
    from .run import out_dir
    t0 = time.perf_counter()
    out = out_dir(args.smoke)
    use = {int(a.split("=")[0]): a.split("=", 1)[1] for a in (args.use_machine or [])}
    T, chosen = load(out, P, use)
    models = T["models"]
    if models.empty:
        raise FileNotFoundError(f"no per-fold files in {out}")
    per_fold = models[["model_id", "fold", "lci", "f1", "worst_subject_lci"]]
    dec, hr, comp = decide(per_fold)
    summ = summary(models)
    sens_mean, agree = sensitivity_tables(T["sensitivity"]) if not T["sensitivity"].empty else (pd.DataFrame(),) * 2
    alloc = T["allocations"]
    bits_wide = (alloc.pivot_table(index=["budget", "method", "fold"], columns="tensor", values="bits").reset_index()
                 if not alloc.empty else pd.DataFrame())
    files = {"summary-mean-sd-over-folds": summ, "probe-headroom": hr, "allocation-comparisons": comp,
             "sensitivity-mean-over-folds": sens_mean, "sensitivity-rank-agreement": agree,
             "allocations-bits-per-tensor": bits_wide}
    for name, df in files.items():
        R.write_csv(out / f"{P}-{name}.csv", df)
    dec.update({"folds_and_machines": chosen, "n_folds": len(chosen), "smoke": args.smoke, "analyzed_at": R.now(),
                **R.git_info(), "seconds": time.perf_counter() - t0})
    R.write_json(out / f"{P}-decision.json", dec)
    show = ["model_id", "n_folds", "size_kb_mean", "mean_bits_mean", "lci_mean", "lci_sd", "worst_subject_lci_mean",
            "f1_mean", "f1_sd"]
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
        print(summ[show].to_string(index=False))
        print("\nR1 headroom:\n" + (hr.to_string(index=False) if len(hr) else "(no probe rows)"))
        cc = ["budget", "comparator", "d_lci_mean", "n_folds_higher", "n_folds_needed", "d_f1_mean", "d_worst_mean",
              "beats"]
        print("\nR2 comparisons:\n" + (comp[[c for c in cc if c in comp]].to_string(index=False) if len(comp) else "-"))
    print(f"\ncontribution supported: {dec['contribution_supported']}; per budget: " +
          ", ".join(f"{B}: beats all {v['beats_all_comparators']}, headroom {v['headroom_at_bitwidth']}"
                    for B, v in dec["by_budget"].items()))
    print(f"folds {chosen}; saved {out}/{P}-*.csv and -decision.json")
