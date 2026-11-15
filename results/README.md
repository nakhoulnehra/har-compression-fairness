# Results Summary

Key numbers from every file in `results/`, grouped by run folder. One entry per file; update the entry whenever its file is added or changed.

<!-- Entry format:
## <run-folder>/
### <file-name>
- What: <one line>
- Key numbers: <metric = value, ...>
-->

## lci-phishing-reproduction/
LCI reproduction of Tulla et al. (2026) on Phishing / ResidualMLP, seeds 0–4, shap_seed 0. Written by `python -m lci.run_lci_phishing`.

### [lci-phishing-report.md](lci-phishing-reproduction/lci-phishing-report.md)
- What: LCI reproduction report (Phishing / ResidualMLP, seeds 0–4) covering the comparison with the paper, the acceptance criteria, the noise floor and the deviations from the spec.
- Key numbers: 5/6 acceptance criteria pass. The 2-std check flags PTQ-8bit (0.9995 vs 0.999, std ≈ 0) and KD (0.816 vs paper 0.499, gap expected per spec). The retraining-noise LCI (0.882) sits just above τ = 0.85.

### [lci-phishing-summary-vs-paper.csv](lci-phishing-reproduction/lci-phishing-summary-vs-paper.csv)
- What: mean ± std over 5 seeds per method (accuracy, macro-F1, ρ̄, J₁₀, LCI, number collapsed), next to the paper's LCI and accuracy.
- Key numbers (LCI ours vs paper | accuracy ours vs paper):

  | Method | LCI ours | LCI paper | Acc ours | Acc paper |
  |---|---|---|---|---|
  | Teacher | 1.000 | 1.000 | 0.969 | 0.964 |
  | PTQ-8bit | 0.9995 | 0.999 | 0.970 | 0.964 |
  | PTQ-4bit | 0.698 ± 0.053 | 0.619 | 0.890 | 0.839 |
  | Prune-30% | 0.922 ± 0.073 | 0.897 | 0.964 | 0.937 |
  | Prune-50% | 0.817 ± 0.065 | 0.781 | 0.937 | 0.936 |
  | Prune-70% | 0.703 ± 0.051 | 0.710 | 0.896 | 0.925 |
  | KD | 0.816 ± 0.042 | 0.499 | 0.930 | 0.986 |

### [lci-phishing-runs-per-seed-method.csv](lci-phishing-reproduction/lci-phishing-runs-per-seed-method.csv)
- What: 35 rows (5 seeds × Teacher + 6 compressed models). Columns: accuracy, macro-F1, shap_corr, jaccard, LCI, collapsed, n_skipped, MPRF feature and shift, n_params, n_nonzero, eff_size_bytes.
- Key numbers:
  - Seeds collapsed (LCI < 0.85): PTQ-8bit 0/5, PTQ-4bit 5/5, Prune-30% 1/5, Prune-50% 3/5, Prune-70% 5/5, KD 4/5.
  - n_skipped = 0 everywhere.
  - Teacher has 71,298 params. KD student has 1,554. PTQ-4bit leaves 14,690 non-zero.

### [lci-phishing-noise-floor.csv](lci-phishing-reproduction/lci-phishing-noise-floor.csv)
- What: LCI noise floor. 5 SHAP-noise rows (same teacher, shap_seed 0 vs 1) and 4 retraining-noise rows (teacher seed s vs s+1).
- Key numbers:
  - SHAP noise: LCI 0.9929 ± 0.0004 (ρ̄ 0.986, J 1.000).
  - Retraining noise: LCI 0.882 ± 0.048, min 0.857 (ρ̄ 0.900, J 0.864).
  - Seeds preserved (LCI ≥ 0.882): PTQ-8bit 5/5, Prune-30% 4/5, Prune-50% 1/5, Prune-70% 0/5, PTQ-4bit 0/5, KD 0/5.

### [lci-phishing-shap-attributions.npz](lci-phishing-reproduction/lci-phishing-shap-attributions.npz)
- What: cache of the GradientExplainer class-1 attributions, 40 arrays of shape (300, 30) float32. Keys are `seed{s}_{method}_shapseed{k}`, covering 5 seeds × (teacher at shap_seed 0 and 1, plus 6 compressed models).
- Key numbers: 300 explained test rows, 50 background rows, nsamples = 200. Used to recompute LCI without re-running SHAP.

### [lci-phishing-shap-attributions-configs.json](lci-phishing-reproduction/lci-phishing-shap-attributions-configs.json)
- What: config of each array in the `.npz`: seed, method, SHAP settings, shap 0.51.0 / torch 2.15.0, and a SHA-256 of the model weights. A cached array is reused only if its config matches exactly.
- Key numbers: 40 entries.
