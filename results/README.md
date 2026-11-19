# Results Summary

Key numbers from every file in `results/`, grouped by run folder. One entry per file; update the entry whenever its file is added or changed.

<!-- Entry format:
## <run-folder>/
### <file-name>
- What: <one line>
- Key numbers: <metric = value, ...>
-->

## lci-phishing-reproduction/
LCI reproduction of Tulla et al. (2026) on Phishing / ResidualMLP, seeds 0–4, shap_seed 0. Written by `python -m lci.run_lci_phishing`. All files were **regenerated on 2026-11-19 with torch 2.7.1+cpu** on Aya's machine (489 s); the torch 2.15.0 numbers are in the report's version history.

### [lci-phishing-report.md](lci-phishing-reproduction/lci-phishing-report.md)
- What: LCI reproduction report (Phishing / ResidualMLP, seeds 0–4) covering the comparison with the paper, the acceptance criteria, the noise floor, the version history and the deviations from the spec. Written by hand from the CSVs.
- Key numbers:
  - 5/6 acceptance criteria pass. The 2-std check flags PTQ-8bit (0.9995 vs 0.999, std ≈ 0) and KD (0.833 vs paper 0.499, gap expected per spec).
  - The retraining-noise LCI (0.840, min 0.783) sits around τ = 0.85; one retrained teacher pair falls below it.

### [lci-phishing-summary-vs-paper.csv](lci-phishing-reproduction/lci-phishing-summary-vs-paper.csv)
- What: mean ± std over 5 seeds per method (accuracy, macro-F1, ρ̄, J₁₀, LCI, number collapsed), next to the paper's LCI and accuracy.
- Key numbers (LCI ours vs paper | accuracy ours vs paper):

  | Method | LCI ours | LCI paper | Acc ours | Acc paper |
  |---|---|---|---|---|
  | Teacher | 1.000 | 1.000 | 0.970 | 0.964 |
  | PTQ-8bit | 0.9995 | 0.999 | 0.970 | 0.964 |
  | PTQ-4bit | 0.661 ± 0.066 | 0.619 | 0.889 | 0.839 |
  | Prune-30% | 0.904 ± 0.059 | 0.897 | 0.965 | 0.937 |
  | Prune-50% | 0.833 ± 0.093 | 0.781 | 0.938 | 0.936 |
  | Prune-70% | 0.733 ± 0.033 | 0.710 | 0.897 | 0.925 |
  | KD | 0.833 ± 0.075 | 0.499 | 0.930 | 0.986 |

### [lci-phishing-runs-per-seed-method.csv](lci-phishing-reproduction/lci-phishing-runs-per-seed-method.csv)
- What: 35 rows (5 seeds × Teacher + 6 compressed models). Columns: accuracy, macro-F1, shap_corr, jaccard, LCI, collapsed, n_skipped, MPRF feature and shift, n_params, n_nonzero, eff_size_bytes.
- Key numbers:
  - Seeds collapsed (LCI < 0.85): PTQ-8bit 0/5, PTQ-4bit 5/5, Prune-30% 1/5, Prune-50% 3/5, Prune-70% 5/5, KD 4/5.
  - n_skipped = 0 everywhere.
  - Teacher has 71,298 params. KD student has 1,554. PTQ-4bit leaves 14,601 non-zero (mean over seeds).

### [lci-phishing-noise-floor.csv](lci-phishing-reproduction/lci-phishing-noise-floor.csv)
- What: LCI noise floor. 5 SHAP-noise rows (same teacher, shap_seed 0 vs 1) and 4 retraining-noise rows (teacher seed s vs s+1).
- Key numbers:
  - SHAP noise: LCI 0.9929 ± 0.0003 (ρ̄ 0.986, J 1.000).
  - Retraining noise: LCI 0.840 ± 0.038, min 0.783, max 0.862 (ρ̄ 0.899, J 0.780).
  - Seeds preserved (LCI ≥ 0.840): PTQ-8bit 5/5, Prune-30% 4/5, Prune-50% 2/5, Prune-70% 0/5, PTQ-4bit 0/5, KD 2/5.

### [lci-phishing-shap-attributions.npz](lci-phishing-reproduction/lci-phishing-shap-attributions.npz)
- What: cache of the GradientExplainer class-1 attributions, 40 arrays of shape (300, 30) float32. Keys are `seed{s}_{method}_shapseed{k}`, covering 5 seeds × (teacher at shap_seed 0 and 1, plus 6 compressed models).
- Key numbers: 300 explained test rows, 50 background rows, nsamples = 200. Used to recompute LCI without re-running SHAP.

### [lci-phishing-shap-attributions-configs.json](lci-phishing-reproduction/lci-phishing-shap-attributions-configs.json)
- What: config of each array in the `.npz`: seed, method, SHAP settings, shap 0.51.0 / torch 2.7.1+cpu, and a SHA-256 of the model weights. A cached array is reused only if its config matches exactly.
- Key numbers: 40 entries.

## har-setup/
HAR data pipeline (plan/HAR_IMPLEMENTATION.md Steps 1–2): HARTH and HAR70+ cut into 128-sample (2.56 s) windows of 6 axes (back + thigh), 7 shared classes, plus subject-wise folds, normalisation statistics, the fixed explain windows and the SHAP backgrounds. Written by `python -m har.build_windows`; all choices are seeded (42) and made from the data alone, before any model exists.
- Signals: resampled to 50 Hz where needed (S006), then HARTH's 4th-order 20 Hz Butterworth low-pass per gap-free stretch, on both datasets.
- Rules: windows lie inside gap-free runs of shared-label rows; 50% overlap for HARTH training, none for testing; a window is kept only if one label covers > 50% of it (≥ 65 of 128 rows).
- Walking-aid status is **inferred**: 503, 505, 513, 514, 515 are the only HAR70+ subjects with no stairs rows.

### [har-window-counts-per-subject.csv](har-setup/har-window-counts-per-subject.csv)
- What: one row per subject and split (`harth_train` step 64, `harth_test` step 128, `har70plus_test` step 128). Columns: group, walking-aid status, hours, windows, single-label / mixed-kept / mixed-dropped, windows per class.
- Key numbers:
  - Windows per subject: HARTH training 191 (S009) to 6,300 (S008); HARTH test 97 (S009) to 3,161 (S008); HAR70+ 674 (505) to 1,184 (515).
  - Stairs per HAR70+ subject: 0–11 stairs-up and 0–8 stairs-down windows. Every one of the 13 other older subjects has at least one stairs window.

### [har-window-counts-per-group-and-class.csv](har-setup/har-window-counts-per-group-and-class.csv)
- What: windows per split and group (general; older; older walking-aid / no walking-aid, both inferred) and per class.
- Key numbers (walking / shuffling / stairs up / stairs down / standing / sitting / lying):

  | Split, group | Subjects | Windows | Per class | Mixed kept |
  |---|---|---|---|---|
  | HARTH train, general | 22 | 84,169 | 18,213 / 3,093 / 1,243 / 1,125 / 10,855 / 43,187 / 6,453 | 8,079 (9.6%) |
  | HARTH test, general | 22 | 42,248 | 9,148 / 1,563 / 623 / 562 / 5,472 / 21,629 / 3,251 | 4,061 (9.6%) |
  | HAR70+ test, older | 18 | 17,403 | 8,401 / 381 / 38 / 39 / 3,228 / 3,739 / 1,577 | 2,083 (12.0%) |
  | older, walking-aid (inferred) | 5 | 4,479 | 1,895 / 61 / 0 / 0 / 852 / 1,145 / 526 | 515 |
  | older, no walking-aid (inferred) | 13 | 12,924 | 6,506 / 320 / 38 / 39 / 2,376 / 2,594 / 1,051 | 1,568 |

  - Sitting is 51% of HARTH windows. Stairs are 2.8% of HARTH training windows and 0.4% of HAR70+ (77 windows).

### [har-dropped-data-summary.csv](har-setup/har-dropped-data-summary.csv)
- What: every transformation or drop, per subject, with stage, reason, rows / windows and detail.
- Key numbers:
  - Columns: `index` dropped in S015 and S021, `Unnamed: 0` in S023.
  - S006: 408,709 rows at 10 ms decimated to 204,360 rows at 20 ms (FIR anti-alias).
  - Non-shared HARTH labels dropped: 771,053 rows (12.3% of HARTH at 50 Hz). Running 291,356; cycling sit 381,444; cycling stand 54,249; cycling sit inactive 36,139; cycling stand inactive 7,865.
  - Gaps > 40 ms: 644 in HARTH, 255 in HAR70+; no window crosses one.
  - Rows in runs shorter than 128 samples: 4,976 (HARTH), 816 (HAR70+).
  - Run tails shorter than one step: 20,086 (HARTH train), 39,926 (HARTH test), 16,349 (HAR70+) rows.
  - Windows without a > 50% label: 498 (HARTH train), 260 (HARTH test), 116 (HAR70+).

### [har-fold-assignment.csv](har-setup/har-fold-assignment.csv)
- What: one row per (fold, subject) with role train / validation / test, group, walking-aid status and the windows used.
- Key numbers:
  - 5 folds; HARTH test sets of 4, 4, 5, 4, 5 subjects (8,209–8,702 windows each), balanced greedily by window count.
  - Each fold: 14–15 training subjects (51,451–55,552 windows) and 3 validation subjects (6,163–7,709 windows).
  - All 18 HAR70+ subjects (17,403 windows) are test in every fold and in no train or validation set.
  - Every HARTH subject is tested exactly once, and each fold's HARTH test set has all 7 classes.

### [har-inner-validation-subjects.csv](har-setup/har-inner-validation-subjects.csv)
- What: the 3 validation subjects of each fold, held out from training, with their non-overlapping window counts. Drawn with seed 42 + fold among subjects with ≥ 1,000 windows.
- Key numbers:
  - fold 0: S010, S020, S021
  - fold 1: S014, S018, S024
  - fold 2: S013, S019, S022
  - fold 3: S017, S022, S024
  - fold 4: S012, S016, S024
  - That is 13 distinct subjects (S022 twice, S024 three times), with 1,104–2,947 windows each.

### [har-normalisation-stats-per-fold.csv](har-setup/har-normalisation-stats-per-fold.csv)
- What: per fold and channel, the mean and std (in g) over every sample of the fold's HARTH training-subject windows. Validation, test and HAR70+ are never used; `channel_stats` raises if they are passed in.
- Key numbers:
  - Mean ranges over the folds: back_x −0.90 to −0.87, thigh_x −0.58 to −0.51, thigh_z 0.39 to 0.48.
  - Std ranges over the folds: 0.21–0.25 (back_y) to 0.62–0.69 (thigh_z).
  - Regenerated 2026-11-19 after the 20 Hz low-pass was added to the windows: the means are unchanged to 6 decimals and the stds are 0.1–1.1% lower. All other `har-setup/` files are byte-identical.

### [har-explain-windows.csv](har-setup/har-explain-windows.csv)
- What: the fixed explain windows, the same for every model and method. One row per window: ID (`<split>:<subject>:<start_row>`), subject, group, walking-aid status (inferred), class, pick order, start row and time, majority share, and the HARTH subject's test fold.
- How they were chosen: from the non-overlapping test windows only, before any model existed. Up to 30 per subject, stratified by class: round-robin over the classes present, random within class. The generator for each subject is seeded with (42, crc32(subject)).
- Key numbers:
  - 1,200 windows: 30 for each of the 40 subjects (22 HARTH = 660, 18 HAR70+ = 540). IDs are unique.
  - 349 (29%) have mixed labels, against 10% of all windows, because transition-heavy classes are oversampled.

### [har-explain-windows-per-subject-class-and-group.csv](har-setup/har-explain-windows-per-subject-class-and-group.csv)
- What: explain-window counts per subject (with available windows and classes present) and per group, by class.
- Key numbers (walking / shuffling / stairs up / stairs down / standing / sitting / lying):

  | Group | Subjects | Windows | Per class |
  |---|---|---|---|
  | general | 22 | 660 | 117 / 106 / 84 / 90 / 101 / 89 / 73 |
  | older | 18 | 540 | 107 / 92 / 28 / 33 / 98 / 94 / 88 |
  | older, walking-aid (inferred) | 5 | 150 | 32 / 24 / 0 / 0 / 32 / 31 / 31 |
  | older, no walking-aid (inferred) | 13 | 390 | 75 / 68 / 28 / 33 / 66 / 63 / 57 |

  - Stairs per subject: 0–5 up and 0–5 down. HAR70+ subjects other than the walking-aid five have 1–9 stairs windows in total, except 502 (stairs down only, 2).

### [har-shap-background-windows.csv](har-setup/har-shap-background-windows.csv)
- What: the SHAP background of each fold: 100 `harth_train` windows from that fold's training subjects only (never validation, test or HAR70+). Stratified by class like the explain windows, generator seeded with (42, fold). Same columns as the explain file, plus fold.
- Key numbers:
  - 5 × 100 windows, 14 or 15 per class (walking and shuffling 15, the others 14).
  - 13–14 training subjects per fold, at most 18 windows from any one subject.
  - 22–34 mixed-label windows per fold, mostly stairs and shuffling.

### [har-explain-and-background-windows-per-fold.csv](har-setup/har-explain-and-background-windows-per-fold.csv)
- What: per fold, the explain windows of its test subjects (general and older), of its validation subjects, and the background's class mix and subjects.
- Key numbers: per fold, 120 or 150 general + 540 older = 660 or 690 test explain windows, plus 90 validation explain windows (for the gate and τ) and 100 background windows.

## har-model-selection-smoke/
**Pipeline check only, not results.** `python -m har.run_model_selection --stage all --device cpu --smoke` (fold 0, 1 epoch, 300 / 200 / 200 windows), then `python -m har.time_model_selection_cpu --smoke` and `python -m har.select_model --smoke`, on Aya's laptop (CPU), 2026-11-19. Its accuracies and its "frozen" model say nothing about the candidates. The real model choice goes to `results/har-model-selection/` (same layout). There are 120 files, so the entries below are by file pattern.

### tuning/har-tune-&lt;model&gt;-&lt;setting&gt;-fold0-Aya.csv (+ `-config.json`, `-history.csv`)
- What: one row per tried setting (4 per model, 20 in total): fold-0 validation scores, settings hash, seed, device, GPU, machine, git commit, code hash and run time. The `-config.json` holds the exact settings and command line; `-history.csv` the per-epoch curve.
- Key numbers: 20/20 settings done; the validation subjects are exactly fold 0's S010, S020, S021; no test subjects (the `test_subjects` column is empty).

### final/har-final-&lt;model&gt;-fold0-Aya.csv (+ `-config.json`, `-history.csv`, `-per-subject.csv`, `-confusion.csv`)
- What: the best tuning setting per model, trained on fold 0 and evaluated on its held-out HARTH test subjects (S008, S019, S022, S025): one row of metrics, per-subject metrics and the 7×7 confusion matrix.
- Key numbers (smoke, 1 epoch; regenerated 2026-11-19 with torch 2.7.1, the source-checked models and the 20 Hz low-passed windows): pooled 7-class macro-F1 CNN1D 0.322, TinyHAR 0.193, DeepConvLSTM 0.108, HART 0.301, XGBoost 0.792. The earlier version gave reproducible values on two identical runs.

### timing/har-cpu-inference-and-shap-time-Aya.csv (+ `-config.json`)
- What: CPU latency (batch 1, 1 thread), batched throughput and single-output SHAP time per window, smoke protocol (20 calls, 4 SHAP windows, nsamples 20).
- Key numbers: not meaningful at this protocol; the full protocol is 1,000 calls, 100 windows and nsamples 200.

### har-model-selection-tuning.csv, -final-per-fold.csv, -per-subject-metrics.csv, -summary.csv, har-frozen-model-config.json
- What: the merged tables, the per-model summary and the rule's decision written by `har.select_model --smoke`.
- Key numbers: the cost-based rule ran end to end. Only CNN1D was within 2 pt of the best 1-epoch F1, so the smoke "choice" is CNN1D; it means nothing.

### logs/
- What: the full terminal output of every unit (`tune-…`, `final-…`), every session (`session-all-Aya-<time>.log`), the timing run and the selection.
