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

## har-model-selection/
Real model choice (plan Step 4), HARTH only. **Done: TinyHAR is chosen and frozen** (lr 1e-3, width 1.0, 54,273 parameters; 2026-11-20 on Aya's laptop, git eb964e4). Training ran in three sessions, each `python -m har.run_model_selection --stage all` (tuning + folds 0–4) for its own models, torch 2.7.1, clean trees:
- CNN1D + TinyHAR: Nakhoul's RTX 3050 Ti (`--device cuda`, git 473b75d), 2026-11-19, 21.6 min.
- XGBoost: Aya's laptop (`--device cpu`, git 473b75d), 2026-11-19, 8.7 min.
- HART + DeepConvLSTM: Nakhoul's RTX 3050 Ti (`--device cuda`, git 2c64f45), 2026-11-19/20, 83.2 min.

There are 233 files (60 tuning, 125 final, 48 logs), listed by pattern. File names end in the machine name (`nakhoul-3050` or `aya-laptop`).

### tuning/har-tune-&lt;model&gt;-&lt;setting&gt;-fold0-&lt;machine&gt;.csv (+ `-config.json`, `-history.csv`)
- What: one row per setting, trained on fold 0's 15 training subjects (51,451 windows) and scored on validation subjects S010, S020, S021 (7,709 windows). The neural nets try lr {1e-3, 3e-4} × width {1.0, 0.5}, capped at 30 epochs with early stopping (patience 5). XGBoost tries depth {3, 6} × eta {0.1, 0.5}.
- Key numbers (validation pooled 7-class macro-F1, best first):
  - XGBoost: **0.832** (depth 3, eta 0.1), then 0.814, 0.808, 0.805.
  - TinyHAR: **0.818** (lr 1e-3, width 1.0, 54,273 parameters), then 0.751 (1e-3, 0.5), 0.743 (3e-4, 1.0), 0.740 (3e-4, 0.5).
  - DeepConvLSTM: **0.768** (lr 3e-4, width 1.0, 458,183 parameters), then 0.766 (3e-4, 0.5; 115,431 parameters), 0.736 (1e-3, 0.5), 0.690 (1e-3, 1.0).
  - CNN1D: **0.761** (lr 3e-4, width 0.5, 316,167 parameters), then 0.736 (1e-3, 1.0), 0.724 (1e-3, 0.5), 0.707 (3e-4, 1.0).
  - HART: **0.743** (lr 1e-3, width 0.5, 355,379 parameters), then 0.743 (3e-4, 1.0), 0.739 (3e-4, 0.5), 0.738 (1e-3, 1.0). All four are within 0.5 pt.
- Validation accuracy is 0.84–0.93 for all 20 settings; stairs F1 separates them (0.41–0.82).

### final/har-final-&lt;model&gt;-fold&lt;k&gt;-&lt;machine&gt;.csv (+ `-config.json`, `-history.csv`, `-per-subject.csv`, `-confusion.csv`), k = 0–4
- What: the best tuning setting per model, trained on each of the 5 HARTH folds and tested on that fold's held-out subjects (8,209–8,702 test windows per fold). Includes per-subject metrics and the 7×7 confusion matrix.
- Key numbers (test, mean ± SD over the 5 folds):

| Model | Parameters | Macro-F1 (7 classes, pooled) | Accuracy (pooled) | Stairs F1 (mean of up and down) | Non-stair macro-F1 (subject mean) | Per fold, macro-F1 for folds 0–4 |
|---|---|---|---|---|---|---|
| XGBoost (reference, not compressed) | 253–569 trees | **0.826 ± 0.029** | 0.933 ± 0.020 | 0.745 ± 0.046 | 0.840 ± 0.028 | 0.857, 0.849, 0.805, 0.832, 0.788 |
| TinyHAR | 54,273 | **0.824 ± 0.042** | 0.919 ± 0.025 | 0.765 ± 0.126 | 0.824 ± 0.036 | 0.855, 0.878, 0.805, 0.776, 0.803 |
| CNN1D | 316,167 | 0.797 ± 0.052 | 0.908 ± 0.032 | 0.745 ± 0.110 | 0.789 ± 0.052 | 0.840, 0.859, 0.789, 0.736, 0.763 |
| HART | 355,379 | 0.796 ± 0.045 | 0.911 ± 0.023 | 0.702 ± 0.123 | 0.798 ± 0.049 | 0.845, 0.808, 0.814, 0.723, 0.791 |
| DeepConvLSTM | 458,183 | 0.770 ± 0.053 | 0.903 ± 0.034 | 0.637 ± 0.152 | 0.793 ± 0.043 | 0.761, 0.859, 0.725, 0.735, 0.769 |

- The selection rule's F1 is pooled over all 22 HARTH test subjects (the 5 folds' confusion matrices summed, 42,248 windows): XGBoost 0.829, TinyHAR **0.820**, CNN1D 0.795, HART 0.792, DeepConvLSTM 0.772. Pooled accuracy: 0.933, 0.919, 0.908, 0.911, 0.903.
- TinyHAR is the best neural net, 0.9 pt below XGBoost on the pooled F1. It is 2.5–2.8 pt ahead of CNN1D and HART and 4.8 pt ahead of DeepConvLSTM, and it is also the smallest model. Only TinyHAR is within the 2-pt band of the selection rule (plan Section 2.9), so `select_model` chose it.
- TinyHAR beats every other neural net in every fold except fold 1, where DeepConvLSTM (0.859) and CNN1D (0.859) are within 2 pt of it (0.878).
- Fold 3 (S013, S020, S021, S023) is the hardest fold for every neural net, mainly through stairs F1 (0.47–0.57). DeepConvLSTM has the weakest and most variable stairs F1 (0.47–0.81 per fold).

### [har-frozen-model-config.json](har-model-selection/har-frozen-model-config.json)
- What: the frozen model (architecture, hyperparameters, training recipe, seed), the selection rule and its trace, the per-candidate summary with CPU timing, git commit and code hash. Written by `python -m har.select_model`.
- Key numbers: **TinyHAR**, setting lr1e-3-width1, 54,273 parameters, pooled 7-class macro-F1 0.820 (best F1 overall among the deep models; eligible band ≥ 0.800 holds only TinyHAR), CPU SHAP 0.175 s/window. Frozen 2026-11-20 13:07 on Aya's laptop, clean tree.

### har-model-selection-tuning.csv, har-model-selection-final-per-fold.csv, har-model-selection-per-subject-metrics.csv, har-model-selection-summary.csv
- What: merged tables of the per-unit tuning/final files above (all machines): tuning rows, final rows per fold, per-subject test metrics per fold, and one summary row per candidate (pooled over the 22 HARTH test subjects).
- Key numbers: as in the tuning and final entries above (summary pooled macro-F1: XGBoost 0.829, TinyHAR 0.820, CNN1D 0.795, HART 0.792, DeepConvLSTM 0.772).

### [timing/har-cpu-inference-and-shap-time-Aya.csv](har-model-selection/timing/har-cpu-inference-and-shap-time-Aya.csv) (+ `-config.json`)
- What: CPU latency (batch 1, 1 thread, median of 1,000), throughput (batch 1,024) and GradientExplainer SHAP time per window (nsamples 200, 100 windows) of each candidate's fold-0 final checkpoint, on Aya's laptop (`python -m har.time_model_selection_cpu`).
- Key numbers (SHAP s/window | latency ms): TinyHAR 0.175 | 2.05; CNN1D 0.295 | 1.74; HART 0.335 | 6.88; DeepConvLSTM 1.196 | 10.79; XGBoost no SHAP | 3.91 (incl. features).

### logs/
- What: the terminal output of every unit (20 tuning + 25 final) and of the three sessions (`session-all-nakhoul-3050-20261119-215942.log`, `session-all-aya-laptop-20261119-215550.log`, `session-all-nakhoul-3050-20261119-224004.log`).

## har-model-selection-smoke/
**Git-ignored, local only: a pipeline check, not results.** The `--smoke` runs of `har.run_model_selection`, `har.time_model_selection_cpu` and `har.select_model` write here (fold 0, 1 epoch, 300 / 200 / 200 windows), with the same layout as `har-model-selection/`. The folder is rebuilt in under a minute, and its numbers say nothing about the candidates. The smoke runs on each machine passed (2026-11-19: Aya CPU 36.4 s, Nakhoul CUDA 36.7 s; see `environment/time.md`).

## har-component-timing/
Step 6 timing of the frozen TinyHAR on fold 0 (51,451 training windows, 26,105 test windows, 660 explain windows), `python -m har.time_har_components --fold 0 --device cuda --machine-name <m>`. Each run writes `har-component-timing-tinyhar-fold0-<m>-<device>.csv` (seconds per component), `-projection.csv` (hours for Steps 8–9 per fold), `-config.json` and `logs/<same>.log`. "Ours" is charged at Park's cost (`measured = False`, our method is not implemented yet).

| | nakhoul-3050 (RTX 3050 Ti Laptop), 2026-11-20, git 8b770af | uni-2080 (RTX 2080), 2026-11-20, git 52d9f98 |
|---|---|---|
| CE epoch | 2.5 s | 1.5 s |
| Park epoch (CE + 50·L_AM, EG m = 4) | 52.0 s (**20.8× CE**; plan assumed 3×) | 65.9 s (**44.7× CE**) |
| SHAP, 660 windows, nsamples 200 | 92.2 s | 87.4 s |
| Projection: Park fine-tuning (incl. tuning) | 1.16 h | 1.48 h |
| Projection: ours (= Park) | 1.16 h | 1.48 h |
| Projection: SHAP | 0.72 h | 0.68 h |
| Projection: noise floor / L1+SNIP | 0.09 h / 0.06 h | 0.05 h / 0.03 h |
| **Projected total, Steps 8–9 per fold** | **3.20 h** | **3.73 h** |
| Wall clock of the timing run | 186.6 s | 209.1 s |

Park's and our fine-tuning together take 73% (3050 Ti) and 79% (2080) of the projected time; PTQ, evaluation and LCI are all under 15 s.

## har-component-timing-smoke/
**Git-ignored, local only: a pipeline check, not results.** `python -m har.time_har_components --fold 0 --smoke` (Step 6) writes here: 600 training windows, 8 explain windows, nsamples 20, files named like the real ones in `har-component-timing/`. Its projection is dominated by per-batch overhead and says nothing about run time. Passed on Aya CPU on 2026-11-20 (28.5 s wall clock) and on the nakhoul-3050 GPU (7.8 s).

## har-noise-floor/
Step 8 noise floor, fold 0, frozen TinyHAR: seed 0 (the model-selection checkpoint) against retrained seeds 1–3 (retrain noise) and against SHAP seeds 1–2 on seed 0 (SHAP noise), nsamples 200. `python -m har.run_noise_floor --folds 0 --device cuda --machine-name nakhoul-3050`, 2026-11-20, git c7787e2 (clean tree), 709 s.

### [har-noise-floor-tinyhar-fold0-nakhoul-3050-tau.csv](har-noise-floor/har-noise-floor-tinyhar-fold0-nakhoul-3050-tau.csv)
- What: τ_0 = min over retrain seeds of the inner-val LCI (seed 0 vs seed s), and checks.
- Key numbers: **τ_0 = 0.6593** (seed 1; seeds 2 / 3: 0.7564 / 0.7724). Below Tulla's τ = 0.85. SHAP noise above retrain noise in every group.

### [har-noise-floor-tinyhar-fold0-nakhoul-3050-group-summary.csv](har-noise-floor/har-noise-floor-tinyhar-fold0-nakhoul-3050-group-summary.csv)
- What: mean / min / max subject-mean LCI per group (inner val 3, general 4, older 18 subjects) and noise type.
- Key numbers (retrain mean [min, max] | SHAP mean): inner val 0.729 [0.659, 0.772] | 0.946; general 0.759 [0.707, 0.811] | 0.941; older 0.769 [0.736, 0.786] | 0.927.

### [har-noise-floor-tinyhar-fold0-nakhoul-3050-per-subject-lci.csv](har-noise-floor/har-noise-floor-tinyhar-fold0-nakhoul-3050-per-subject-lci.csv)
- What: 125 rows (noise × seed × subject): LCI, SHAP correlation, Jaccard, MPRF feature / shift.
- Key numbers: per-subject retrain LCI 0.602–0.932 (older), 0.642–0.870 (general), 0.631–0.873 (inner val); SHAP LCI 0.809–0.980.

### [har-noise-floor-tinyhar-fold0-nakhoul-3050-retrained-models.csv](har-noise-floor/har-noise-floor-tinyhar-fold0-nakhoul-3050-retrained-models.csv)
- What: inner-val macro-F1 and training time of seed 0 and retrained seeds 1–3.
- Key numbers: inner-val macro-F1 seed 0 0.818, seeds 1–3 0.832 / 0.785 / 0.817; training 44 / 30 / 26 s (16 / 11 / 9 epochs, early stopping).

### har-noise-floor-tinyhar-fold0-nakhoul-3050-config.json, logs/har-noise-floor-tinyhar-fold0-nakhoul-3050.log
- What: command line, device, git, code hash, seeds, explain-window counts, seconds; terminal output.
- Key numbers: training 104 s, SHAP 603 s, wall 709 s.

## har-xgp-pilot/
XGP pilot (plan 2.5 (b)), fold 0, 70% pruning, inner val (S010, S020, S021; 90 explain windows, nsamples 200), τ_0 = 0.6593 from `har-noise-floor/`. `python -m har.run_xgp_pilot --fold 0 --device cuda --machine-name nakhoul-3050`, 2026-11-20/21, git c7787e2 (clean tree). All pruned models: 17,365 of 54,273 weights non-zero. Fine-tuning times are not usable as timings (see `environment/time.md`).

### [har-xgp-pilot-tinyhar-fold0-70pct-inner-val-nakhoul-3050-runs.csv](har-xgp-pilot/har-xgp-pilot-tinyhar-fold0-70pct-inner-val-nakhoul-3050-runs.csv)
- What: one row per configuration: LCI_val, SHAP corr, Jaccard, inner-val macro-F1, before-fine-tuning LCI/F1, gate branch, tuning selection.
- Key numbers (LCI_val / F1; original F1 0.818; eligible F1 ≥ L1 − 1 pt = 0.721):

| Run | Before ft LCI / F1 | LCI_val | F1 |
|---|---|---|---|
| L1 | 0.402 / 0.374 | 0.884 | 0.731 |
| SNIP | 0.074 / 0.239 | 0.771 | 0.678 |
| Park γp 1, lr 1e-4 (**selected**) | 0.402 / 0.374 | 0.810 | 0.743 |
| Park γp 1, lr 3e-4 / γp 50, lr 1e-4 / γp 50, lr 3e-4 | 0.402 / 0.374 | 0.721 / 0.789 / 0.797 | 0.704 / 0.607 / 0.676 |
| XGP β 0.3, γ 0.1 (**selected, fallback**) | 0.486 / 0.485 | **0.887** | 0.711 |
| XGP β 0.3, γ 1 | 0.486 / 0.485 | 0.874 | 0.671 |
| XGP β 0.7, γ 0.1 / γ 1 | 0.444 / 0.403 | 0.799 / 0.815 | 0.674 / 0.634 |

- Gate: both β start below τ_0 (0.486, 0.444 < 0.659), so all 4 XGP runs used CE + γ·L_cons.
- No XGP configuration is within 1 pt of L1's F1 (best 0.711, 2.0 pt below), so XGP is picked by the fallback (highest F1). Its LCI_val is 0.003 above L1. Only L1 and Park γp 1 lr 1e-4 meet the F1 rule.
- The keep-score raises LCI before fine-tuning (β 0.3: 0.486 vs L1 0.402) and F1 before fine-tuning (0.485 vs 0.374). Larger γ and β lower F1.

### [har-xgp-pilot-tinyhar-fold0-70pct-inner-val-nakhoul-3050-per-subject-lci.csv](har-xgp-pilot/har-xgp-pilot-tinyhar-fold0-70pct-inner-val-nakhoul-3050-per-subject-lci.csv)
- What: 42 rows (14 runs incl. 4 pre-fine-tuning starts × 3 inner-val subjects): LCI, SHAP corr, Jaccard.
- Key numbers (S010 / S020 / S021): L1 0.853 / 0.853 / 0.945; XGP β 0.3 γ 0.1 0.851 / 0.862 / 0.946; Park selected 0.826 / 0.768 / 0.836; SNIP 0.722 / 0.792 / 0.799.

### [har-xgp-pilot-tinyhar-fold0-70pct-inner-val-nakhoul-3050-finetune-history.csv](har-xgp-pilot/har-xgp-pilot-tinyhar-fold0-70pct-inner-val-nakhoul-3050-finetune-history.csv)
- What: per epoch (10 per run): train CE, extra loss, inner-val F1 / accuracy, seconds.
- Key numbers: median s/epoch: CE (L1 / SNIP) 9–12, Park 158–207, L_cons 105. Epoch 1 of `xgp_beta0.7_gamma1` is 16,134 s (laptop hibernated).

### har-xgp-pilot-tinyhar-fold0-70pct-inner-val-nakhoul-3050-config.json, logs/har-xgp-pilot-tinyhar-fold0-70pct-inner-val-nakhoul-3050.log
- What: command line, device, git, code hash, XGP / Park settings, n_nonzero, chosen configurations, seconds; terminal output.
- Key numbers: teacher EG 40.0 s, SNIP 0.9 s, I(w) 10.1 s; wall 28,005 s (includes ≈ 4.5 h hibernation).

## har-xgp-pilot/round2/
XGP pilot round 2 (plan 2.5 (b), round 2): the last validation-only design iteration before XGP is frozen. Fold 0, inner val only (S010, S020, S021), HAR70+ never loaded. `python -m har.run_xgp_pilot2`. Pre-registration written first (`--prepare`, Aya CPU, 2026-11-21, before any GPU run). 70% + EG-noise diagnostic on nakhoul-3050, 90% on uni-2080, both 2026-11-21 at git 7dd23b6 (clean), then `--analyze`. **Decision: final XGP = xgp_kd, β* 0.1 at both levels; Step 9 uses E_ft = 20.**

### [har-xgp-pilot2-tinyhar-fold0-decision.json](har-xgp-pilot/round2/har-xgp-pilot2-tinyhar-fold0-decision.json)
- What: the pre-registered decision. It holds the final XGP, β per level, the best baseline F1 per level, the epochs check and the Step 9 E_ft.
- Key numbers: **final_xgp = xgp_kd**; β* 0.1 at 70% and at 90%, neither flagged. Best baseline F1 (l1_kd) 0.7828 at 70% and 0.6770 at 90%. Epochs check (l1_kd_e20 − l1_kd, seed 0): +0.97 pt at 70%, **+7.69 pt at 90%** (0.7292 against 0.6523). That is over the 1-pt bar, so **E_ft = 20 for every method in Step 9**.

### [har-xgp-pilot2-tinyhar-fold0-decision-table.csv](har-xgp-pilot/round2/har-xgp-pilot2-tinyhar-fold0-decision-table.csv)
- What: one row per XGP variant and baseline: mean LCI over the two window sets and F1 at each level, the F1 margin against the best baseline, eligibility, and whether it was selected.
- Key numbers (mean LCI 70 / 90 / over both levels; F1 70 / 90): **xgp_kd 0.875 / 0.753 / 0.814; 0.776 / 0.718, the only eligible variant, selected**. xgp_worstclass_kd 0.875 / 0.770 / 0.823; 0.761 / 0.728, misses the bar at 70% by 2.1 pt. xgp_aug_kd 0.875 / 0.770 / 0.823; 0.764 / 0.713, misses at 70%. xgp_worstclass_aug_kd 0.876 / 0.766 / 0.821; 0.768 / 0.688, misses at 70%. Baselines: l1 0.852 / 0.781 / 0.816; **l1_kd 0.884 / 0.774 / 0.829**; snip_kd 0.791 / 0.120 / 0.456; park 0.866 / 0.746 / 0.806; park_kd 0.888 / 0.729 / 0.809.
- Reading: the rule picked xgp_kd because it is the only variant eligible at 70%, not because of its LCI. Its mean LCI over both levels (0.814) is below l1_kd (0.829) and l1 (0.816). At 90%, every XGP variant beats every baseline on F1 (0.69–0.73 against l1_kd 0.677).

### [har-xgp-pilot2-tinyhar-fold0-summary-mean-sd-over-seeds.csv](har-xgp-pilot/round2/har-xgp-pilot2-tinyhar-fold0-summary-mean-sd-over-seeds.csv)
- What: mean and SD over seeds for every method and start at both levels. Covers LCI, Spearman, Jaccard, worst-subject LCI, retention, F1 and accuracy on the normal and older-like windows, plus fine-tuning seconds.
- Key numbers: the per-level values in the runs entries here come from this file.

### [har-xgp-pilot2-tinyhar-fold0-90pct-inner-val-uni-2080-runs.csv](har-xgp-pilot/round2/har-xgp-pilot2-tinyhar-fold0-90pct-inner-val-uni-2080-runs.csv)
- What: the same layout as the 70% runs file, at 90% sparsity: 39 assessments. Starts have n_nonzero 6,821 (original 54,273). τ_0 and ceiling_0 come from nakhoul-3050's fold-0 noise floor.
- Key numbers (mean ± SD over seeds; F1 normal / LCI normal / LCI older-like / mean LCI): l1 0.639 / 0.760 / 0.801 / 0.781; **l1_kd 0.677 ± 0.022** / 0.774 / 0.774 / 0.774; snip_kd 0.055 ± 0.061 / 0.124 / 0.116 / 0.120 (collapsed); park 0.604 / 0.713 / 0.778 / 0.746; park_kd 0.631 / 0.736 / 0.723 / 0.729; l1_kd_e20 0.729 / 0.776 / 0.836 / 0.806. xgp_kd β 0.05: 0.683 / 0.712 / 0.766 / 0.739; **β 0.1: 0.718 ± 0.009 / 0.769 / 0.737 / 0.753**; β 0.3: 0.552 ± 0.214 / 0.649 / 0.645 / 0.647 (one seed collapsed). With β* 0.1: xgp_worstclass_kd 0.728 / 0.759 / 0.781 / 0.770; xgp_aug_kd 0.713 / 0.749 / 0.791 / 0.770; xgp_worstclass_aug_kd 0.688 / 0.734 / 0.798 / 0.766.
- Reading: the eligibility bar is 0.667, and all four XGP variants clear it at 90%. Without fine-tuning, every start is near chance (F1 0.04–0.23).

### [har-xgp-pilot2-tinyhar-fold0-90pct-inner-val-uni-2080-per-subject-lci.csv](har-xgp-pilot/round2/har-xgp-pilot2-tinyhar-fold0-90pct-inner-val-uni-2080-per-subject-lci.csv)
- What: per-subject LCI (S010, S020, S021) for each 90% assessment on the normal and older-like windows.

### [har-xgp-pilot2-tinyhar-fold0-90pct-inner-val-uni-2080-finetune-history.csv](har-xgp-pilot/round2/har-xgp-pilot2-tinyhar-fold0-90pct-inner-val-uni-2080-finetune-history.csv)
- What: per-epoch losses, val F1 and accuracy for every 90% fine-tuning run. Fine-tuning takes ≈ 2 s per CE/KD epoch and ≈ 65 s per Park epoch on the RTX 2080.

### [har-xgp-pilot2-tinyhar-fold0-90pct-inner-val-uni-2080-config.json](har-xgp-pilot/round2/har-xgp-pilot2-tinyhar-fold0-90pct-inner-val-uni-2080-config.json)
- What: the command, device (RTX 2080, torch 2.7.1+cu118), git 7dd23b6 (clean), the pre-registration hashes, τ_0 0.659 / ceiling_0 0.946 (nakhoul-3050 fold 0), n_nonzero per start, and the rules. Level loop 4,928 s; wall clock 5,114 s (85 min).

### logs/har-xgp-pilot2-tinyhar-fold0-90pct-uni-2080-20261121-152237.log
- What: the full console log of the 90% run (uni-2080).

### [har-xgp-pilot2-tinyhar-fold0-70pct-inner-val-nakhoul-3050-runs.csv](har-xgp-pilot/round2/har-xgp-pilot2-tinyhar-fold0-70pct-inner-val-nakhoul-3050-runs.csv)
- What: one row per assessment at 70% (original, 8 no-finetune starts, 30 fine-tuned runs: seeds 0–2, plus seed-0-only park, park_kd, l1_kd_e20). Pooled 7-class macro-F1 and LCI on the normal and older-like inner-val windows, sparsity, timings, β*. Every start has n_nonzero 17,365. τ_0 0.659, ceiling_0 0.946.
- Key numbers (mean ± SD over seeds; F1 normal / LCI normal / LCI older-like / mean LCI): original F1 0.818 (older-like 0.696). l1 0.738 ± 0.012 / 0.831 / 0.872 / 0.852; **l1_kd 0.783 ± 0.008** / 0.867 / 0.901 / 0.884; snip_kd 0.749 / 0.781 / 0.801 / 0.791; park 0.742 / 0.833 / 0.899 / 0.866; **park_kd 0.768 / 0.871 / 0.904 / 0.888**; l1_kd_e20 0.786 / 0.846 / 0.875 / 0.861. xgp_kd β 0.05: 0.770 / 0.872 / 0.885 / 0.879; **β 0.1: 0.776 ± 0.007 / 0.873 / 0.876 / 0.875**; β 0.3: 0.758 / 0.839 / 0.899 / 0.869. With β* 0.1: xgp_worstclass_kd 0.761 / 0.874 / 0.877 / 0.875; xgp_aug_kd 0.764 / 0.874 / 0.876 / 0.875; xgp_worstclass_aug_kd 0.768 / 0.865 / 0.886 / 0.876.
- Reading: best baseline F1 = l1_kd 0.7828, so the eligibility bar is 0.7728. At 70% only xgp_kd β 0.1 clears it; β* = 0.1, not flagged. XGP variants have the highest normal-window LCI (≈ 0.873 against 0.867 for l1_kd) but lower older-like LCI (≈ 0.876 against 0.901 / 0.904 for l1_kd / park_kd), so their mean LCI (≈ 0.875) is below l1_kd (0.884) and park_kd (0.888). Epochs check, seed 0: l1_kd_e20 0.7859 against l1_kd 0.7762 = +0.97 pt, just under the 1-pt bar for E_ft 20 at this level.

### [har-xgp-pilot2-tinyhar-fold0-70pct-inner-val-nakhoul-3050-per-subject-lci.csv](har-xgp-pilot/round2/har-xgp-pilot2-tinyhar-fold0-70pct-inner-val-nakhoul-3050-per-subject-lci.csv)
- What: per-subject LCI (S010, S020, S021) for each 70% assessment on the normal and older-like windows. The worst-subject LCI is also in the runs file.

### [har-xgp-pilot2-tinyhar-fold0-70pct-inner-val-nakhoul-3050-finetune-history.csv](har-xgp-pilot/round2/har-xgp-pilot2-tinyhar-fold0-70pct-inner-val-nakhoul-3050-finetune-history.csv)
- What: per-epoch CE, KL and L_AM losses, val F1 and accuracy for every 70% fine-tuning run. Fine-tuning takes ≈ 3.3 s per CE/KD epoch and ≈ 60 s per Park epoch.

### [har-xgp-pilot2-tinyhar-fold0-70pct-inner-val-nakhoul-3050-config.json](har-xgp-pilot/round2/har-xgp-pilot2-tinyhar-fold0-70pct-inner-val-nakhoul-3050-config.json)
- What: the command, device (RTX 3050 Ti, torch 2.7.1+cu118), git 7dd23b6 (clean), code sha256, the pre-registration hashes, τ_0 / ceiling_0 and their sources, n_nonzero per start, and the selection and Step 9 rules. Level wall clock 5,603 s (93 min).

### [har-xgp-pilot2-tinyhar-fold0-eg-noise-diagnostic-nakhoul-3050.csv](har-xgp-pilot/round2/har-xgp-pilot2-tinyhar-fold0-eg-noise-diagnostic-nakhoul-3050.csv)
- What: EG-noise diagnostic on the original model, 300 normal explain windows (100 per subject). Each row is a per-subject LCI, Spearman correlation and top-10 Jaccard between two explainers.
- Key numbers (mean LCI / Spearman): SHAP vs EG m = 4: 0.684 / 0.549; m = 16: 0.770 / 0.772; m = 200: 0.946 / 0.952. EG m = 4 seed 0 vs seed 1: 0.515 / 0.305. SHAP seed 0 vs seed 1: 0.915 / 0.952. So EG at small m is noisy: two m = 4 seeds agree less with each other than with SHAP.

### logs/har-xgp-pilot2-tinyhar-fold0-70pct-nakhoul-3050-20261121-152712.log
- What: the full console log of the 70% run. One-off setup costs: SHAP on the original 85 s, teacher logits + Park targets 38 s, the 4 importances 7–16 s each.

### [har-xgp-pilot2-tinyhar-fold0-inner-val-windows-older-like-params.csv](har-xgp-pilot/round2/har-xgp-pilot2-tinyhar-fold0-inner-val-windows-older-like-params.csv)
- What: every fold-0 inner-val window (non-overlapping split) by window ID: label, older-like speed s and scale a (seed 42), explain flag and pick order.
- Key numbers: 7,709 windows; 300 explain windows (100 per subject; per class 47 / 47 / 45 / 45 / 44 / 44 / 28); the first 30 per subject are exactly round 1's 90 windows. s in [0.800, 0.900] (mean 0.850), a in [0.700, 0.900] (mean 0.801).

### [har-xgp-pilot2-tinyhar-fold0-reference-windows-older-like-params.csv](har-xgp-pilot/round2/har-xgp-pilot2-tinyhar-fold0-reference-windows-older-like-params.csv)
- What: the 1,000 class-stratified HARTH training reference windows of the keep-score (seed 42, as in round 1), with their older-like s and a (seed 43) for `xgp_aug_kd`.
- Key numbers: 1,000 windows.

### [har-xgp-pilot2-tinyhar-fold0-preregistration.json](har-xgp-pilot/round2/har-xgp-pilot2-tinyhar-fold0-preregistration.json)
- What: the pre-registered selection rule, the Step 9 fairness rule, the design (variants, baselines, β grid {0.05, 0.1, 0.3}, seeds 0–2, KD T = 3, α = 0.7, E_ft 10, lr 1e-4), and the sha256 of the two window tables (line endings normalised). Every level run checks the tables against it and refuses to run if they changed.
- Key numbers: written 2026-11-21 at git 182b168 (uncommitted round-2 code).

## har-noise-floor-smoke/
**Git-ignored, local only: a pipeline check, not results.** `python -m har.run_noise_floor --folds 0 --smoke` (Step 8) writes here: seeds 1–3 retrained for 1 epoch on 600 windows, 3 explain windows per subject (3 HAR70+ subjects), nsamples 20. The file names match the real ones (`har-noise-floor-tinyhar-fold0-<machine>-{per-subject-lci, group-summary, tau, retrained-models}.csv`, `-config.json`). Its τ_0 (0.07) means nothing. Passed on Aya CPU on 2026-11-20 (24.8 s wall clock; a rerun reuses the cached checkpoints and SHAP).

## har-xgp-pilot-smoke/
**Git-ignored, local only: a pipeline check, not results.** `python -m har.run_xgp_pilot --fold 0 --smoke` writes here: 600 training windows, 1 fine-tuning epoch, 9 inner-val explain windows, nsamples 20, 35 reference windows, EG m = 2. The file names match the real ones (`har-xgp-pilot-tinyhar-fold0-70pct-inner-val-<machine>-{runs, per-subject-lci, finetune-history}.csv`, `-config.json`). All 10 pruned runs had the same n_nonzero (17,365 of 54,273). Passed on Aya CPU on 2026-11-20: 33 s with the smoke τ_0 (both β gated to CE), 49 s with `--tau 0.99` (both β gated to CE + γ·L_cons).

## har-xgp-pilot-smoke/round2/
**Git-ignored, local only: a pipeline check, not results.** `python -m har.run_xgp_pilot2 --smoke` (`--prepare`, `--levels 0.7 0.9 --diagnostic`, `--analyze`) writes here: 600 training windows, 200 inner-val windows for F1, 3 explain windows per subject, nsamples 20, 35 reference windows, EG m = 2, 1 epoch (2 for `l1_kd_e20`), seeds 0, 1. The file names match the real ones. Every level had one n_nonzero across all starts and runs. Passed on Aya CPU on 2026-11-21: both levels + diagnostic in 78 s, analyze 8 s (its decision means nothing at smoke sizes); `--resume` skipped every finished run.

## har-baseline-gap/
Step 5, baseline gap check on HAR70+ (**first HAR70+ use**), with the frozen TinyHAR (`har-model-selection/har-frozen-model-config.json`, lr 1e-3, width 1.0, 54,273 parameters). Evaluation only, nothing trained or changed: the 5 final fold checkpoints (seed 0, trained on Nakhoul's GPU) run on CPU with their stored normalisation. General = each HARTH subject's test windows from its one held-out fold (22 subjects, 42,248 windows); older = all 18 HAR70+ subjects with every fold model (17,403 windows), averaged over the 5 folds. Written by `python -m har.run_sanity_walking_aid` on Aya's laptop, 2026-11-20, 29 s (rerun with the merged Ustad scoring).
- Groups: walking aid (inferred) = 503, 505, 513, 514, 515; other older = the other 13; walker (inferred) = the walking-aid group without 503 (the 4 walker users; 503 taken as the walking-sticks user, project decision, not stated in Ustad).
- Class sets:
  - `all`: 7 classes; accuracy on all windows, macro-F1 over the non-stair classes.
  - `non_stair`: windows labelled walking, shuffling, standing, sitting, lying.
  - **`ustad4` (primary Ustad scoring)**: all windows; labels **and** predictions merged to 4 classes, stairs up/down → walking, shuffling → standing [Ustad Sec 2.4].
  - `ustad4_drop_argmax7` (secondary): only windows labelled walking/standing/sitting/lying; a shuffling or stairs prediction counts as wrong.
  - `ustad4_drop_argmax4` (secondary): the same windows, argmax over the 4 classes only.
- Statistics: per-subject values; bootstrap 95% CI of the gap in group means (10,000 resamples, seed 42, subjects within groups) + two-sided Mann-Whitney U with rank-biserial r. Walking-aid tests are exploratory (n = 5 / 4).
- **Acceptance (plan Step 5): passed.** Other older − walking aid is positive in every class set and in each of the 5 fold models. On Ustad's merged 4 classes it is **+13.7 pt accuracy (CI +0.2 to +30.8, p = 0.075)**, vs Ustad et al.'s +5 pt (87% vs 92%, HARTH-trained XGBoost); with 503 excluded **+18.9 pt (CI +7.6 to +36.7, p = 0.003)**.

### [har-baseline-gap-tinyhar-group-summary.csv](har-baseline-gap/har-baseline-gap-tinyhar-group-summary.csv)
- What: 25 rows (5 class sets × 5 groups): n subjects, subject IDs, windows per model, subject mean / SD / median / min / max of accuracy and macro-F1; for `all` also window-pooled accuracy and pooled stairs F1 (older groups: mean over the 5 fold models).
- Key numbers (subject mean ± SD, %):

| Group | n | Acc `all` | Macro-F1 `all` (non-stair) | **Acc `ustad4`** | **Macro-F1 `ustad4`** | Acc `drop_argmax7` | Acc `drop_argmax4` |
|---|---|---|---|---|---|---|---|
| General | 22 | 90.8 ± 6.0 | 82.2 ± 6.4 | 95.5 ± 3.1 | 93.3 ± 3.9 | 91.1 ± 6.7 | 97.3 ± 2.4 |
| Older | 18 | 76.3 ± 10.9 | 72.4 ± 11.8 | 84.6 ± 12.1 | 84.3 ± 14.6 | 76.6 ± 11.0 | 91.6 ± 9.9 |
| Walking aid (inferred) | 5 | 69.9 ± 18.2 | 63.9 ± 19.2 | 74.7 ± 19.3 | 72.3 ± 23.6 | 70.0 ± 18.4 | 84.0 ± 17.2 |
| Other older | 13 | 78.7 ± 5.8 | 75.7 ± 5.6 | 88.4 ± 5.1 | 88.9 ± 5.8 | 79.1 ± 5.9 | 94.6 ± 2.9 |
| Walker (inferred, no 503) | 4 | 64.8 ± 16.4 | 58.2 ± 16.5 | 69.5 ± 17.9 | 66.3 ± 22.4 | 64.7 ± 16.5 | 80.4 ± 17.5 |

- Secondary macro-F1: `drop_argmax7` general 91.2, older 86.0, walking aid 76.5, other 89.6, walker 71.4; `drop_argmax4` 95.6, 89.8, 80.3, 93.5, 75.7.
- Window-pooled accuracy (`all`): general 91.9, older 76.7, other older 78.8, walking aid 70.4, walker 65.4. Pooled stairs F1 up / down: general 0.791 / 0.699 (623 / 562 windows); older 0.046 / 0.115 (38 / 39 windows per model).
- Older accuracy on Ustad's classes depends on the scoring: 84.6 merged, 76.6 with `drop_argmax7`, 91.6 with `drop_argmax4` (general: 95.5, 91.1, 97.3). The merged set also scores the shuffling and stairs windows, so it is not on the same windows as the two `drop` sets.

### [har-baseline-gap-tinyhar-group-tests.csv](har-baseline-gap/har-baseline-gap-tinyhar-group-tests.csv)
- What: 30 rows (3 comparisons × 5 class sets × 2 metrics): group means, gap (pt), bootstrap 95% CI, Mann-Whitney U / p, rank-biserial r; for the walking-aid comparisons also the gap's min / max over the 5 fold models; Ustad's reference gaps (+5 pt HARTH model, +2 pt HAR70+ model) on the `ustad4` accuracy rows.
- Key numbers (gap in pt [95% CI], MW p, rank-biserial r):

| Comparison | **Acc `ustad4`** | **Macro-F1 `ustad4`** | Acc `all` | Macro-F1 `all` | Acc `drop_argmax7` | Acc `drop_argmax4` |
|---|---|---|---|---|---|---|
| Older − general (18 vs 22) | −10.9 [−17.2, −6.1], p < 0.001, r −0.81 | −9.0 [−16.8, −3.2], p 0.004, r −0.54 | −14.5 [−20.6, −9.4], p < 0.001 | −9.8 [−16.2, −4.2], p 0.001 | −14.6 [−20.7, −9.2], p < 0.001 | −5.6 [−11.0, −2.0], p 0.001 |
| Other older − walking aid (13 vs 5) | +13.7 [+0.2, +30.8], p 0.075, r 0.57 | +16.6 [+0.4, +38.1], p 0.075, r 0.57 | +8.8 [−4.2, +24.9], p 0.21 | +11.8 [−2.7, +28.4], p 0.12 | +9.1 [−4.1, +25.4], p 0.21 | +10.6 [+0.2, +26.2], p 0.035 |
| Other older − walker, 503 excluded (13 vs 4) | +18.9 [+7.6, +36.7], p 0.003, r 0.92 | +22.7 [+8.3, +45.0], p 0.003, r 0.92 | +13.9 [+2.9, +30.3], p 0.023 | +17.5 [+5.3, +33.3], p 0.006 | +14.3 [+3.3, +30.8], p 0.023 | +14.2 [+4.3, +31.7], p 0.001 |

- Per fold model (`ustad4` accuracy): other older − walking aid +11.3 to +17.3 pt; other older − walker +16.0 to +23.6 pt.
- The `non_stair` rows are within 0.4 pt of the `all` rows.

### [har-baseline-gap-tinyhar-per-subject.csv](har-baseline-gap/har-baseline-gap-tinyhar-per-subject.csv)
- What: 200 rows (40 subjects × 5 class sets): accuracy and macro-F1 pooled over folds (older: mean of 5 fold models, with SD over folds), windows, subgroup.
- Key numbers (older, `ustad4` accuracy): walking-aid subjects 503 95.5, 505 79.1, 513 75.8, 514 **42.9**, 515 80.3; other older 76.9 (512) to 96.1 (504). 514 is the outlier in every class set; 503 (the assumed walking-sticks user) scores like the other older subjects. `all` accuracy: 514 40.5, 503 90.3; SD over the 5 fold models per older subject 1.3–16.5 pt.

### [har-baseline-gap-tinyhar-per-subject-per-fold.csv](har-baseline-gap/har-baseline-gap-tinyhar-per-subject-per-fold.csv)
- What: 560 rows (fold × subject × class set; general subjects only in their test fold, older subjects in all 5): windows, correct, accuracy, macro-F1, classes present.
- Key numbers: the input to the pooled table above; 22 general + 90 older rows per class set.

### [har-baseline-gap-tinyhar-pooled-accuracy-and-stairs-f1-per-fold.csv](har-baseline-gap/har-baseline-gap-tinyhar-pooled-accuracy-and-stairs-f1-per-fold.csv)
- What: window-pooled 7-class accuracy and pooled stairs up / down F1 per fold model and older subgroup, plus one general row over all 22 subjects (fold `all`).
- Key numbers: see the group summary (means over folds); walking-aid groups have no true stairs windows; their stairs F1 is 0 because the model predicts stairs on a few of their windows.

### [har-baseline-gap-tinyhar-general-recheck-vs-model-selection.csv](har-baseline-gap/har-baseline-gap-tinyhar-general-recheck-vs-model-selection.csv)
- What: the CPU-recomputed per-subject accuracy and non-stair macro-F1 of the 22 HARTH test subjects minus the stored GPU values in `har-model-selection/har-model-selection-per-subject-metrics.csv`.
- Key numbers: max |difference| 0.00000 for both: the checkpoints load and predict identically on CPU.

### har-baseline-gap-tinyhar-config.json, logs/har-baseline-gap-run.log
- What: command line, checkpoint paths, each class set's kept classes / F1 classes / argmax restriction / merge map, group IDs, bootstrap settings, Ustad reference values, machine, torch, git commit, code hash, elapsed time; the terminal output.
- Key numbers: git 2981eeb, dirty tree (the Step 5 code is uncommitted), 28.6 s.
