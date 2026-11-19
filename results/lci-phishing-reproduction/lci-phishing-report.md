# LCI reproduction on Phishing / ResidualMLP

Reproduction of Tulla et al. (2026), Table 2, Phishing row, following [plan/LCI_IMPLEMENTATION.md](../../plan/LCI_IMPLEMENTATION.md).

**Run:** 5 training seeds (0–4), split seed 42, SHAP seed 0, CPU.
- Command: `python -m lci.run_lci_phishing --seeds 0 1 2 3 4 --shap_seed 0`.
- **Regenerated on 2026-11-19 with torch 2.7.1+cpu** (shap 0.51.0) on Aya's machine (Lenovo ThinkPad E14 Gen 5). It took 489 s, 97 ± 9 s per seed. The project moved from torch 2.15 to 2.7.1 for the CUDA 11.1 GPU machine; the earlier run is summarised under [Version history](#version-history).
- Unit tests: 12/12 LCI tests pass (`lci/tests/test_lci.py`, as part of `python -m pytest har/tests lci/tests`, 119/119).

**Data:** Phishing Websites, UCI 327, 11,055 × 30, phishing : legitimate = 4,898 : 6,157 (44.3 : 55.7; paper 44 : 56). Split 7,738 / 1,658 / 1,659. Teacher: 71,298 parameters (spec: ≈ 71.3K; paper: ≈ 72K).

## Summary vs paper (mean ± sample std over 5 seeds)

| Method | Test acc | Macro-F1 | ρ̄ (shap_corr) | J₁₀ | **LCI** | Collapsed (LCI < 0.85) | Paper LCI | Paper acc |
|---|---|---|---|---|---|---|---|---|
| Teacher | 0.970 ± 0.001 | 0.969 ± 0.001 | 1.000 | 1.000 | **1.000** | 0/5 | 1.000 | 0.964 |
| PTQ-8bit | 0.970 ± 0.001 | 0.969 ± 0.001 | 0.999 ± 0.000 | 1.000 ± 0.000 | **0.9995 ± 0.0001** | 0/5 | 0.999 | 0.964 |
| PTQ-4bit | 0.889 ± 0.016 | 0.886 ± 0.017 | 0.820 ± 0.012 | 0.501 ± 0.126 | **0.661 ± 0.066** | 5/5 | 0.619 | 0.839 |
| Prune-30% | 0.965 ± 0.004 | 0.965 ± 0.004 | 0.983 ± 0.004 | 0.824 ± 0.118 | **0.904 ± 0.059** | 1/5 | 0.897 | 0.937 |
| Prune-50% | 0.938 ± 0.010 | 0.937 ± 0.010 | 0.927 ± 0.013 | 0.738 ± 0.177 | **0.833 ± 0.093** | 3/5 | 0.781 | 0.936 |
| Prune-70% | 0.897 ± 0.033 | 0.895 ± 0.032 | 0.851 ± 0.008 | 0.615 ± 0.070 | **0.733 ± 0.033** | 5/5 | 0.710 | 0.925 |
| KD | 0.930 ± 0.004 | 0.929 ± 0.004 | 0.872 ± 0.017 | 0.794 ± 0.138 | **0.833 ± 0.075** | 4/5 | 0.499 | 0.986 |

- The Teacher row is LCI(teacher, teacher) on identical attributions, so it is 1 by construction.
- No samples were skipped in any comparison (`n_skipped` = 0 everywhere).
- J₁₀ only takes the values 1, 9/11, 2/3, 7/13, 3/7, …, which explains the large LCI std of the pruned models.

## Acceptance criteria

- **PASS**: PTQ-8bit has ρ̄ close to 1, at most one top-10 swap on every seed, and accuracy within 0.5 pt of the teacher. ρ̄ = 0.9990, J = 1.000 on all 5 seeds (zero swaps), max |Δacc| = 0.12 pt.
- **PASS**: PTQ-4bit LCI < 0.85 (0.661).
- **PASS**: Prune-70% LCI < 0.85 (0.733).
- **PASS**: KD LCI < 0.85 (0.833).
- **PASS**: LCI falls monotonically from Prune-30% to 50% to 70% (0.904 > 0.833 > 0.733).
- **FAIL (flagged)**: every paper LCI within ~2 std of our seed mean. PTQ-4bit, Prune-30/50/70% and the teacher are within range. Two are outside:
  - **PTQ-8bit**: 0.9995 vs 0.999. The gap is 0.0005, but the seed std is 0.0001, so the 2-std test degenerates. Both values round to 0.999–1.000; there is no real disagreement.
  - **KD**: 0.833 vs 0.499, a gap of 0.33. The spec predicted this: no KD implementation in the authors' repo reproduces the paper's KD values. Our KD also scores 4.0 pt *below* the teacher (0.930 vs 0.970), whereas the paper's 0.986 exceeds its teacher. Not tuned, per the spec.

**Accuracy vs the authors' CSV** (not an acceptance criterion):

| Model | Ours | Authors' CSV | Difference |
|---|---|---|---|
| Teacher | 0.970 | 0.964 | +0.6 pt |
| PTQ-4bit | 0.889 | 0.839 | +5.0 pt |
| Prune-30% | 0.965 | 0.937 | +2.8 pt |
| Prune-70% | 0.897 | 0.925 | −2.8 pt |

## Noise floor (Step 4)

| | LCI | ρ̄ | J₁₀ | n |
|---|---|---|---|---|
| (a) SHAP noise: same teacher, shap_seed 0 vs 1 | 0.9929 ± 0.0003 (min 0.9924) | 0.986 | 1.000 | 5 seeds |
| (b) Retraining noise: teacher seed s vs s+1 | 0.840 ± 0.038 (min 0.783) | 0.899 | 0.780 | 4 pairs (0-1, 1-2, 2-3, 3-4) |

- **The explainer alone costs about 0.007 LCI** and never changes the top-10 set.
  - Teacher-vs-compressed comparisons use the same SHAP seed for both models, so part of this sampling noise cancels: PTQ-8bit has ρ̄ = 0.999, above the 0.986 between two SHAP seeds of the same model.
  - Treat (a) as an upper bound on the explainer's share.
- **Retrained teachers are close to τ = 0.85.** Two teachers that differ only in initialisation reach LCI 0.783–0.862. Three of the four pairs are at 0.855–0.862, just above τ, and one pair (seeds 0 and 1) is *below* it. So a retrained but uncompressed model can count as "logic-collapsed" under the paper's fixed threshold; τ sits at the level of natural run-to-run variation.
- **Seeds preserved under each threshold** (data-driven threshold = mean retrain LCI, 0.840; Tulla's τ = 0.85):

  | Method | Preserved, LCI ≥ 0.840 (of 5) | Not collapsed, LCI ≥ 0.85 (of 5) |
  |---|---|---|
  | PTQ-8bit | 5 | 5 |
  | Prune-30% | 4 | 4 |
  | Prune-50% | 2 | 2 |
  | Prune-70% | 0 | 0 |
  | PTQ-4bit | 0 | 0 |
  | KD | 2 | 1 |

  The two thresholds agree except for one KD seed (LCI 0.843).

## Accuracy vs LCI divergence

Mean drop from the teacher in test accuracy, and the matching LCI:

| Method | Accuracy drop | LCI |
|---|---|---|
| Prune-30% | −0.4 pt | 0.904 (one seed collapses, at 0.826, while keeping the teacher's accuracy: 0.971 vs 0.971) |
| Prune-50% | −3.2 pt | 0.833 |
| KD | −4.0 pt | 0.833 |
| Prune-70% | −7.3 pt | 0.733 |
| PTQ-4bit | −8.1 pt | 0.661 |

Fixed-point rounding at 4 bits zeroes 80% of the parameters (14,601 of 71,298 remain non-zero on average) without clamping. Macro-F1 tracks accuracy within 0.4 pt for every method, so there is no hidden class-level degradation.

## Version history

The first run (Nakhoul's machine, torch 2.15.0, 817 s) gave the same pattern: the same 5/6 criteria passed, and the same two values (PTQ-8bit, KD) were flagged.

| | torch 2.15.0 | torch 2.7.1 |
|---|---|---|
| PTQ-4bit LCI | 0.698 | 0.661 |
| Prune-30 / 50 / 70% LCI | 0.922 / 0.817 / 0.703 | 0.904 / 0.833 / 0.733 |
| KD LCI | 0.816 | 0.833 |
| Retraining-noise LCI | 0.882 (min 0.857) | 0.840 (min 0.783) |
| Teacher accuracy | 0.969 | 0.970 |

The two torch versions train numerically different teachers from the same seeds, so every downstream number moves. Each change is within about one seed std of the respective mean (for example PTQ-4bit: 0.037 against a std of 0.053–0.066). It is run-to-run variation, not a change in the method.

## Deviations from the spec, with reasons

1. **File names and layout.** The spec names `results/lci_phishing_runs.csv`, `results/lci_phishing_summary.csv` and `results/LCI_PHISHING_REPORT.md`. These became `lci-phishing-runs-per-seed-method.csv`, `lci-phishing-summary-vs-paper.csv` and `lci-phishing-report.md`, to follow the repo's results naming convention (CLAUDE.md), and all outputs of this reproduction are grouped in `results/lci-phishing-reproduction/`. The plan was updated to match.
2. **SHAP cache.** It is one `.npz` holding all 40 arrays plus one `.json` of per-array configs (including a SHA-256 of the model weights and the torch version), instead of one `.npy` per model. This keeps the results folder tidy. It still lets LCI be recomputed without re-running SHAP.
3. **Extra file.** `lci-phishing-noise-floor.csv` holds the Step 4 values. The spec asks for them in the report but names no file.
4. **Module names.** All code is in `lci/`. The metric is `lci/metric.py` (the spec suggests `lci.py`, which would clash with the package name). The runner is invoked as `python -m lci.run_lci_phishing`.
5. **Retraining noise.** This uses 4 pairs, not 5: with seeds 0–4, seed 4 has no s+1.
6. **Size columns.** `n_nonzero` counts all learnable parameters (weights, biases, BN affine), matching `n_params`. For non-PTQ models, `eff_size_bytes` = `n_params` × 4 (dense float32; pruned models stored dense). For PTQ, it is `n_params` × bits / 8, as specified (nominal).
7. **Data fetch TLS.** ucimlrepo's CSV download failed against a stale Windows CA store (on both machines). `data/download.py` retries with certifi's CA bundle (the bundle ucimlrepo itself uses for its API call), so verification stays on. The saved files are SHA-256 pinned.
8. **KD seeding.** Student initialisation and batch shuffling use the training seed, which the spec leaves open.

## Files

All files are in `results/lci-phishing-reproduction/`, next to this report. See [results/README.md](../README.md) for their key numbers.
- `lci-phishing-runs-per-seed-method.csv`
- `lci-phishing-summary-vs-paper.csv`
- `lci-phishing-noise-floor.csv`
- `lci-phishing-shap-attributions.npz` and `lci-phishing-shap-attributions-configs.json`
