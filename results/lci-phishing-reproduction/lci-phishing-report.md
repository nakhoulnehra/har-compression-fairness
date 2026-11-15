# LCI reproduction on Phishing / ResidualMLP

Reproduction of Tulla et al. (2026), Table 2, Phishing row, following [plan/LCI_IMPLEMENTATION.md](../../plan/LCI_IMPLEMENTATION.md). 5 training seeds (0–4), split seed 42, SHAP seed 0, CPU. Command: `python -m lci.run_lci_phishing --seeds 0 1 2 3 4 --shap_seed 0` (817 s). Unit tests: 12/12 pass (`lci/tests/test_lci.py`).

Data: Phishing Websites, UCI 327, 11,055 × 30, phishing : legitimate = 4,898 : 6,157 (44.3 : 55.7; paper 44 : 56). Split 7,738 / 1,658 / 1,659. Teacher: 71,298 parameters (spec: ≈71.3K; paper: ≈72K).

## Summary vs paper (mean ± sample std over 5 seeds)

| Method | Test acc | Macro-F1 | ρ̄ (shap_corr) | J₁₀ | **LCI** | Collapsed (LCI<0.85) | Paper LCI | Paper acc |
|---|---|---|---|---|---|---|---|---|
| Teacher | 0.969 ± 0.001 | 0.969 ± 0.001 | 1.000 | 1.000 | **1.000** | 0/5 | 1.000 | 0.964 |
| PTQ-8bit | 0.970 ± 0.001 | 0.969 ± 0.001 | 0.999 ± 0.000 | 1.000 ± 0.000 | **0.9995 ± 0.0001** | 0/5 | 0.999 | 0.964 |
| PTQ-4bit | 0.890 ± 0.030 | 0.886 ± 0.033 | 0.828 ± 0.019 | 0.568 ± 0.101 | **0.698 ± 0.053** | 5/5 | 0.619 | 0.839 |
| Prune-30% | 0.964 ± 0.004 | 0.963 ± 0.004 | 0.983 ± 0.005 | 0.861 ± 0.142 | **0.922 ± 0.073** | 1/5 | 0.897 | 0.937 |
| Prune-50% | 0.937 ± 0.007 | 0.936 ± 0.008 | 0.932 ± 0.014 | 0.702 ± 0.119 | **0.817 ± 0.065** | 3/5 | 0.781 | 0.936 |
| Prune-70% | 0.896 ± 0.020 | 0.894 ± 0.019 | 0.837 ± 0.016 | 0.568 ± 0.101 | **0.703 ± 0.051** | 5/5 | 0.710 | 0.925 |
| KD | 0.930 ± 0.004 | 0.929 ± 0.004 | 0.874 ± 0.016 | 0.758 ± 0.083 | **0.816 ± 0.042** | 4/5 | 0.499 | 0.986 |

The Teacher row is LCI(teacher, teacher) on identical attributions, so it is 1 by construction. No samples were skipped in any comparison (`n_skipped` = 0 everywhere). J₁₀ only takes the values 1, 9/11, 2/3, 7/13, 3/7, …, which explains the large LCI std of the pruned models.

## Acceptance criteria

- **PASS**: PTQ-8bit has ρ̄ close to 1, at most one top-10 swap on every seed, and accuracy within 0.5 pt of the teacher. ρ̄ = 0.9991, J = 1.000 on all 5 seeds (zero swaps), max |Δacc| = 0.12 pt.
- **PASS**: PTQ-4bit LCI < 0.85 (0.698).
- **PASS**: Prune-70% LCI < 0.85 (0.703).
- **PASS**: KD LCI < 0.85 (0.816).
- **PASS**: LCI falls monotonically from Prune-30% to 50% to 70% (0.922 > 0.817 > 0.703).
- **FAIL (flagged)**: every paper LCI within ~2 std of our seed mean. PTQ-4bit, Prune-30/50/70% and the teacher are within range. Two are outside:
  - **PTQ-8bit**: 0.9995 vs 0.999. The gap is 0.0005, but the seed std is 0.0001, so the 2-std test degenerates. Both values round to 0.999–1.000; there is no real disagreement.
  - **KD**: 0.816 vs 0.499, a gap of 0.32. The spec predicted this: no KD implementation in the authors' repo reproduces the paper's KD values. Our KD also scores 3.9 pt *below* the teacher (0.930), whereas the paper's 0.986 exceeds its teacher. Not tuned, per the spec.

Accuracy vs the authors' CSV (not an acceptance criterion): our teacher is 0.5 pt higher (0.969 vs 0.964). Our PTQ-4bit is 5.1 pt higher (0.890 vs 0.839), our Prune-70% is 2.9 pt lower (0.896 vs 0.925), and Prune-30% is 2.7 pt higher.

## Noise floor (Step 4)

| | LCI | ρ̄ | J₁₀ | n |
|---|---|---|---|---|
| (a) SHAP noise: same teacher, shap_seed 0 vs 1 | 0.9929 ± 0.0004 (min 0.9924) | 0.986 | 1.000 | 5 seeds |
| (b) Retraining noise: teacher seed s vs s+1 | 0.882 ± 0.048 (min 0.857) | 0.900 | 0.864 | 4 pairs (0-1, 1-2, 2-3, 3-4) |

- The explainer alone costs about 0.007 LCI and never changes the top-10 set. Teacher-vs-compressed comparisons use the same SHAP seed for both models, so part of this sampling noise cancels: PTQ-8bit has ρ̄ = 0.999, above the 0.986 between two SHAP seeds of the same model. Treat (a) as an upper bound on the explainer's share.
- Two teachers that differ only in initialisation reach LCI 0.857–0.954 (three of the four pairs are at 0.857–0.859), barely above the paper's τ = 0.85. So τ sits right at the level of natural run-to-run variation.
- With the data-driven threshold (mean retrain LCI = 0.882), the number of seeds whose explanation is preserved (LCI ≥ 0.882) is:

  | Method | Seeds preserved (of 5) |
  |---|---|
  | PTQ-8bit | 5 |
  | Prune-30% | 4 |
  | Prune-50% | 1 |
  | Prune-70% | 0 |
  | PTQ-4bit | 0 |
  | KD | 0 |

  With τ = 0.85, the number of seeds *not* collapsed is 5, 4, 2, 0, 0 and 1 respectively. The two thresholds agree except for one Prune-50% seed and one KD seed.

## Accuracy vs LCI divergence

Mean drop from the teacher in test accuracy, and the matching LCI:

| Method | Accuracy drop | LCI |
|---|---|---|
| Prune-30% | −0.5 pt | 0.922 (one seed collapses, at 0.820, with −0.7 pt) |
| Prune-50% | −3.2 pt | 0.817 |
| KD | −3.9 pt | 0.816 |
| Prune-70% | −7.4 pt | 0.703 |
| PTQ-4bit | −7.9 pt | 0.698 |

Fixed-point rounding at 4 bits zeroes 79% of the parameters (14,690 of 71,298 remain non-zero) without clamping. Macro-F1 tracks accuracy within 0.5 pt for every method, so there is no hidden class-level degradation.

## Deviations from the spec, with reasons

1. **File names and layout.** The spec names `results/lci_phishing_runs.csv`, `results/lci_phishing_summary.csv` and `results/LCI_PHISHING_REPORT.md`. These became `lci-phishing-runs-per-seed-method.csv`, `lci-phishing-summary-vs-paper.csv` and `lci-phishing-report.md`, to follow the repo's results naming convention (CLAUDE.md), and all outputs of this reproduction are grouped in `results/lci-phishing-reproduction/`. The plan was updated to match.
2. **SHAP cache.** It is one `.npz` holding all 40 arrays plus one `.json` of per-array configs (including a SHA-256 of the model weights), instead of one `.npy` per model. This keeps the results folder tidy. It still lets LCI be recomputed without re-running SHAP.
3. **Extra file.** `lci-phishing-noise-floor.csv` holds the Step 4 values. The spec asks for them in the report but names no file.
4. **Module names.** All code is in `lci/`. The metric is `lci/metric.py` (the spec suggests `lci.py`, which would clash with the package name). The runner is invoked as `python -m lci.run_lci_phishing`.
5. **Retraining noise.** This uses 4 pairs, not 5: with seeds 0–4, seed 4 has no s+1.
6. **Size columns.** `n_nonzero` counts all learnable parameters (weights, biases, BN affine), matching `n_params`. For non-PTQ models, `eff_size_bytes` = `n_params` × 4 (dense float32; pruned models stored dense). For PTQ, it is `n_params` × bits / 8, as specified (nominal).
7. **Data fetch TLS.** ucimlrepo's CSV download failed against this machine's stale Windows CA store. `data/download.py` retries with certifi's CA bundle (the bundle ucimlrepo itself uses for its API call), so verification stays on. The saved files are SHA-256 pinned.
8. **KD seeding.** Student initialisation and batch shuffling use the training seed, which the spec leaves open.

## Files

All files are in `results/lci-phishing-reproduction/`, next to this report. See [results/README.md](../README.md) for their key numbers.
- `lci-phishing-runs-per-seed-method.csv`
- `lci-phishing-summary-vs-paper.csv`
- `lci-phishing-noise-floor.csv`
- `lci-phishing-shap-attributions.npz` and `lci-phishing-shap-attributions-configs.json`
