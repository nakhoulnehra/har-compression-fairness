"""LCI reproduction on Phishing Websites / ResidualMLP (plan/LCI_IMPLEMENTATION.md, Section 3).

Usage (from the repo root):
    python -m lci.run_lci_phishing                              # seeds 0-4, shap_seed 0
    python -m lci.run_lci_phishing --seeds 0 1 2 --shap_seed 0

For each training seed: train the teacher, build the 6 compressed models, compute accuracy,
macro-F1, SHAP (GradientExplainer) and compute_lci(teacher, model). Also measures the LCI noise
floor: SHAP noise (same teacher, shap_seed vs shap_seed+1) and retraining noise (teacher seed s
vs seed s+1).

Writes (all paths defined in lci/paths.py) into results/lci-phishing-reproduction/:
    lci-phishing-runs-per-seed-method.csv        one row per (seed, method)
    lci-phishing-summary-vs-paper.csv            mean +- std over seeds vs the paper
    lci-phishing-noise-floor.csv                 SHAP-noise and retraining-noise LCI
    lci-phishing-shap-attributions.npz           cached SHAP arrays, one per model
    lci-phishing-shap-attributions-configs.json  config of each cached array
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch

from .compression import distill, prune_l1, ptq
from .data import load_phishing_split
from .explain import AttributionCache
from .metric import COLLAPSE_TAU, compute_lci
from .models import ResidualMLP, n_nonzero, n_params
from .paths import LCI_PHISHING_DIR, NOISE_CSV, RUNS_CSV, SHAP_CONFIG_JSON, SHAP_NPZ, SUMMARY_CSV
from .train import evaluate, set_seed, train_teacher

K, OMEGA = 10, 0.5

# Paper Table 2 LCI; accuracy from the authors' results/table2_lci_accuracy.csv.
PAPER = {
    "Teacher": (1.000, 0.964), "PTQ-8bit": (0.999, 0.964), "PTQ-4bit": (0.619, 0.839),
    "Prune-30%": (0.897, 0.937), "Prune-50%": (0.781, 0.936), "Prune-70%": (0.710, 0.925),
    "KD": (0.499, 0.986),
}
METHODS = [  # (name, slug, level, builder)
    ("PTQ-8bit", "ptq8", 8, lambda t, sp, d, seed: ptq(t, 8)),
    ("PTQ-4bit", "ptq4", 4, lambda t, sp, d, seed: ptq(t, 4)),
    ("Prune-30%", "prune30", 0.3, lambda t, sp, d, seed: prune_l1(t, 0.3)),
    ("Prune-50%", "prune50", 0.5, lambda t, sp, d, seed: prune_l1(t, 0.5)),
    ("Prune-70%", "prune70", 0.7, lambda t, sp, d, seed: prune_l1(t, 0.7)),
    ("KD", "kd", 3, lambda t, sp, d, seed: distill(t, sp.X_train, sp.y_train, d, seed=seed)),
]


def lci_row(seed, method, level, ev, lci, size) -> dict:
    return {
        "seed": seed, "method": method, "level": level,
        "test_acc": ev["acc"], "test_f1_macro": ev["f1_macro"],
        "shap_corr": lci["shap_corr"], "jaccard": lci["jaccard"], "lci": lci["lci"],
        "collapsed": lci["lci"] < COLLAPSE_TAU, "n_skipped": lci["n_skipped"],
        "mprf_feature": lci["mprf_feature"], "mprf_shift": lci["mprf_shift"],
        "n_params": size["n_params"], "n_nonzero": size["n_nonzero"], "eff_size_bytes": size["eff_size_bytes"],
    }


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in ["Teacher"] + [m[0] for m in METHODS]:
        g = runs[runs.method == method]
        r = {"method": method, "n_seeds": len(g)}
        for c in ["test_acc", "test_f1_macro", "shap_corr", "jaccard", "lci"]:
            r[f"{c}_mean"] = g[c].mean()
            r[f"{c}_std"] = g[c].std(ddof=1) if len(g) > 1 else 0.0
        r["n_collapsed"] = int(g.collapsed.sum())
        r["paper_lci"], r["paper_acc"] = PAPER[method]
        for c, p in [("lci", r["paper_lci"]), ("test_acc", r["paper_acc"])]:
            gap = p - r[f"{c}_mean"]
            r[f"paper_{'acc' if c == 'test_acc' else c}_gap"] = gap
            r[f"paper_{'acc' if c == 'test_acc' else c}_within_2std"] = abs(gap) <= 2 * r[f"{c}_std"]
        rows.append(r)
    return pd.DataFrame(rows)


def acceptance(runs: pd.DataFrame, summ: pd.DataFrame) -> list[tuple[str, bool, str]]:
    s = summ.set_index("method")
    t, p8 = runs[runs.method == "Teacher"].set_index("seed"), runs[runs.method == "PTQ-8bit"].set_index("seed")
    acc_drop = (t.test_acc - p8.test_acc).abs()
    swaps_ok = (p8.jaccard >= 9 / 11 - 1e-12).all()  # <=1 swap in top-10 -> J >= 9/11
    checks = [
        ("PTQ-8bit: rho close to 1, <=1 top-10 swap every seed, acc within 0.5 pt",
         s.loc["PTQ-8bit", "shap_corr_mean"] > 0.99 and swaps_ok and (acc_drop <= 0.005).all(),
         f"rho={s.loc['PTQ-8bit', 'shap_corr_mean']:.4f}, min J={p8.jaccard.min():.3f}, max |dAcc|={acc_drop.max():.4f}"),
    ]
    for m in ["PTQ-4bit", "Prune-70%", "KD"]:
        checks.append((f"{m}: LCI < {COLLAPSE_TAU}", s.loc[m, "lci_mean"] < COLLAPSE_TAU, f"LCI={s.loc[m, 'lci_mean']:.3f}"))
    l30, l50, l70 = (s.loc[m, "lci_mean"] for m in ["Prune-30%", "Prune-50%", "Prune-70%"])
    checks.append(("LCI monotone Prune-30 > 50 > 70", l30 > l50 > l70, f"{l30:.3f} > {l50:.3f} > {l70:.3f}"))
    off = [m for m in s.index if not s.loc[m, "paper_lci_within_2std"]]
    checks.append(("Every paper LCI within ~2 std of our mean", not off, "outside: " + (", ".join(off) or "none")))
    return checks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--shap_seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()

    split = load_phishing_split()
    d = split.X_train.shape[1]
    background = torch.from_numpy(split.X_train[:50])
    X_explain = torch.from_numpy(split.X_test[:300])
    print(f"data: train {split.X_train.shape}, val {split.X_val.shape}, test {split.X_test.shape}; "
          f"test class counts {np.bincount(split.y_test).tolist()}")

    cache = AttributionCache(SHAP_NPZ, SHAP_CONFIG_JSON)

    def attrs(model, seed, slug, shap_seed):
        cfg = {"dataset": "phishing_uci327", "seed": seed, "method": slug}
        return cache.get(model, background, X_explain, f"seed{seed}_{slug}_shapseed{shap_seed}", cfg, shap_seed)

    runs, noise, teacher_attr = [], [], {}
    for seed in args.seeds:
        ts = time.time()
        set_seed(seed)
        teacher = ResidualMLP(d)
        info = train_teacher(teacher, split.X_train, split.y_train, split.X_val, split.y_val, seed=seed)
        ev = evaluate(teacher, split.X_test, split.y_test)
        print(f"\n[seed {seed}] teacher: {n_params(teacher)} params, best val acc {info['best_val_acc']:.4f} "
              f"@ epoch {info['best_epoch']}, test acc {ev['acc']:.4f}, macro-F1 {ev['f1_macro']:.4f}")

        A_T = attrs(teacher, seed, "teacher", args.shap_seed)
        teacher_attr[seed] = A_T
        n = n_params(teacher)
        runs.append(lci_row(seed, "Teacher", "", ev, compute_lci(A_T, A_T, K, OMEGA),
                            {"n_params": n, "n_nonzero": n_nonzero(teacher), "eff_size_bytes": n * 4}))

        # Noise floor (a): SHAP sampling noise on the same teacher.
        A_T2 = attrs(teacher, seed, "teacher", args.shap_seed + 1)
        r = compute_lci(A_T, A_T2, K, OMEGA)
        noise.append({"kind": "shap_noise", "seed_a": seed, "seed_b": seed,
                      "shap_seed_a": args.shap_seed, "shap_seed_b": args.shap_seed + 1, **r})

        for name, slug, level, build in METHODS:
            model, size = build(teacher, split, d, seed)
            ev = evaluate(model, split.X_test, split.y_test)
            r = compute_lci(A_T, attrs(model, seed, slug, args.shap_seed), K, OMEGA)
            runs.append(lci_row(seed, name, level, ev, r, size))
            print(f"  {name:10s} acc {ev['acc']:.4f} F1 {ev['f1_macro']:.4f} | LCI {r['lci']:.3f} "
                  f"(rho {r['shap_corr']:.3f}, J {r['jaccard']:.3f}) nnz {size['n_nonzero']}/{size['n_params']}")
        print(f"  seed time {time.time() - ts:.1f}s")

    # Noise floor (b): retraining noise, teacher seed s vs seed s+1 (pairs present in --seeds).
    for s in args.seeds:
        if s + 1 in teacher_attr:
            r = compute_lci(teacher_attr[s], teacher_attr[s + 1], K, OMEGA)
            noise.append({"kind": "retrain_noise", "seed_a": s, "seed_b": s + 1,
                          "shap_seed_a": args.shap_seed, "shap_seed_b": args.shap_seed, **r})

    LCI_PHISHING_DIR.mkdir(parents=True, exist_ok=True)
    runs_df = pd.DataFrame(runs)
    summ = summarize(runs_df)
    noise_df = pd.DataFrame(noise)
    runs_df.to_csv(RUNS_CSV, index=False, float_format="%.6f")
    summ.to_csv(SUMMARY_CSV, index=False, float_format="%.6f")
    noise_df.to_csv(NOISE_CSV, index=False, float_format="%.6f")

    pd.set_option("display.width", 200)
    print("\n=== Summary (mean +- std over seeds) ===")
    for _, r in summ.iterrows():
        print(f"{r.method:10s} acc {r.test_acc_mean:.4f}+-{r.test_acc_std:.4f} | LCI {r.lci_mean:.3f}+-{r.lci_std:.3f} "
              f"(rho {r.shap_corr_mean:.3f}, J {r.jaccard_mean:.3f}) | paper LCI {r.paper_lci:.3f} acc {r.paper_acc:.3f}"
              f" | within 2std: LCI {r.paper_lci_within_2std}, acc {r.paper_acc_within_2std}")
    print("\n=== Noise floor ===")
    for kind, g in noise_df.groupby("kind"):
        print(f"{kind:14s} LCI {g.lci.mean():.4f}+-{g.lci.std(ddof=1):.4f} (rho {g.shap_corr.mean():.4f}, "
              f"J {g.jaccard.mean():.4f}), min {g.lci.min():.4f}, n={len(g)}")
    print("\n=== Acceptance criteria ===")
    for desc, ok, detail in acceptance(runs_df, summ):
        print(f"[{'PASS' if ok else 'FAIL'}] {desc}: {detail}")
    print(f"\ntotal time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
