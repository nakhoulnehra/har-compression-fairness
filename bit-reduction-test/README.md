# Bit-reduction test: explanation-guided mixed-precision quantization (XGQ)

A self-contained experiment, kept separate from `har/` and `results/`. It reuses the repo's data, folds, frozen TinyHAR,
SHAP and LCI code read-only, and writes only inside this folder. It does not use the low-bit diagnostic of commit
a18e156 (`har/run_diagnostic_low_bit.py`, `ptq_per_channel`, `low_bit_size`).

## Question

Below 4 bits, does weight-only quantization of the frozen TinyHAR lose explanation consistency (LCI) faster than
accuracy? Can a mixed-precision bit allocation driven by measured explanation sensitivity (XGQ, ours) keep LCI higher
than accuracy-driven allocations at the same storage size, without losing F1? The accuracy-driven baselines are
ZeroQ-style output KL and HAWQ-style Hessian trace.

Why below 4 bits: at 30 KB and above, standard 4-bit PTQ already keeps LCI ≈ 0.90 (9A: FA-LCI 0.904, inner-val LCI
0.895 on folds 0–1). Every explanation-preserving pruning method, XGP included, is dominated there. A new method can
only count as a contribution where existing methods measurably fail.

## Design (pre-registered: fixed in code before any real run; `--prepare` records it with the window hashes)

**Data: HARTH only.** HAR70+ and the folds' HARTH test subjects are never loaded (tested). Folds 0–4. Each fold uses:
- the frozen TinyHAR of the fold (seed 0, model-selection checkpoint);
- no training and no fine-tuning anywhere.

**Window sets per fold:**

| Window set | Content | Used for |
|---|---|---|
| Calibration (design) | Round 4's set: ≤ 10 HARTH **training** windows per (subject, class), seed 46, ≈ 980 windows | KL and Hessian sensitivities |
| LCI-calibration subset | ≤ 20 per class from the calibration set, seed 47, ≈ 140 windows | SHAP-based LCI sensitivity |
| Evaluation (never used for design) | The fold's 3 inner-val subjects: F1 / accuracy on every window; LCI on round 3/4's explain windows (100 per subject) | All reported numbers |

**SHAP:** GradientExplainer, nsamples 200, SHAP seed 0, target = the original model's predicted class, background =
the fold's 100 training windows. **LCI:** 48 segments, k = 10, ω = 0.5. On evaluation it is the subject mean (plus the
worst subject); on calibration it is pooled over the windows.

**Quantizer:** symmetric round-to-nearest, weight-only. It covers the 13 weight tensors (convs, attention, FF,
fusion, LSTM ih/hh, attention score, head; 53,116 weights). The other 1,157 parameters stay fp32.
- per-channel = one fp32 scale per output row (820 scales);
- per-tensor = one scale per tensor.

**Size:** honest storage = weights × bits + fp32 scales + fp32 for the non-quantized parameters.

**Models per fold:**
1. **Probe (uniform):** per-tensor 4 / 3 / 2-bit (30.5 / 24.0 / 17.5 KB) and per-channel 8 / 4 / 3 / 2-bit
   (59.6 / 33.7 / 27.2 / 20.7 KB).
2. **Sensitivities**, one tensor quantized at a time (per-channel, 2 / 3 / 4 / 8 bits, the rest fp32), all on
   calibration windows only:
   - **xgq (ours):** Ω = 1 − LCI(original, model) on the LCI-calibration subset.
   - **zeroq_kl:** Ω = mean KL(p_original ‖ p_model) on the calibration set. This is ZeroQ, Cai et al. 2020, Sec 3.3,
     with real windows instead of distilled data.
   - **hawq:** Ω = max(Tr H, 0) / n · ‖Q_b(W) − W‖², with Tr H of the CE loss by Hutchinson (100 iterations,
     seed 49). This is HAWQ-V2 / V3 (Dong et al. 2020; Yao et al. 2021). A negative trace estimate (CE Hessian not
     PSD) is clipped to 0 and the raw value is reported.
   - The ZeroQ / HAWQ PDFs are not in `papers/`: verify the cited sections before submission.
3. **Allocation:** every method uses the same exact multiple-choice knapsack. It picks bits ∈ {2, 3, 4, 8} per
   tensor, minimising Σ Ω under the size budget (additive assumption, as in ZeroQ / HAWQ). The methods differ only in
   Ω. Budgets = size(uniform per-channel 2-bit) + f × (size(3-bit) − size(2-bit)):

   | Budget | f | Size |
   |---|---|---|
   | 3bit | 1 | 27.2 KB |
   | 2.5bit | 0.5 | 23.9 KB |
   | 2.25bit | 0.25 | 22.3 KB |

   A budget equal to uniform 2-bit would force every method to all-2-bit, since 2 bits is the minimum.

## Decision rules (`bitred/rules.py`, checked against the pre-registration at run time)

- **R1, headroom:** explanation headroom exists at b ∈ {3, 2} bits iff the mean over folds of
  LCI_val(uniform per-channel 4-bit) − LCI_val(uniform per-channel b-bit) ≥ 0.05.
- **R2, beats:** XGQ beats comparator M at budget B iff all of these hold:
  - (a) the mean LCI_val gain is ≥ 0.02;
  - (b) LCI_val is strictly higher in ≥ 4 of 5 folds (an identical allocation counts as not higher);
  - (c) mean F1_val ≥ M's − 0.01;
  - (d) mean worst-subject LCI_val ≥ M's − 0.01.
- **R3, claim:** XGQ beats existing allocation at B iff it beats every comparator at B. The comparators are
  zeroq_kl, hawq, and at 3bit also uniform per-channel 3-bit. The contribution is supported iff R3 holds at a budget
  whose bit width has headroom (3bit → 3; 2.5bit → 3 or 2; 2.25bit → 2). Otherwise it is reported as not supported,
  and nothing is retuned.
- **R4, no tuning:** no method has a hyperparameter chosen on inner-val; every fold is reported.

If R1 shows no headroom, there is no gap to fill, and the test ends there with that finding.

**Fold-0 screen (added 2026-11-24, before any real result existed).** Fold 0 runs first, and folds 1–4 run only if
fold 0 passes both checks:
- **S1:** LCI_val(uniform per-channel 4-bit) − LCI_val(3-bit) ≥ 0.03, or the same for 2-bit.
- **S2:** at ≥ 1 budget, XGQ's LCI_val ≥ max(zeroq_kl, hawq) + 0.01.

Failing either stops the test, and it is reported as not supported. Passing is not evidence: the claim still needs
R1–R3 over all 5 folds.

## 2-bit gap test (added 2026-11-24 after fold 0, before running it)

Fold 0 showed that uniform 2-bit collapses accuracy and LCI together (per-channel: F1 0.40, LCI 0.54). Before any
method is built for 2 bits, this test asks whether the standard fix for low-bit accuracy also restores explanations.
- **Recipe:** quantization-aware KD fine-tuning (straight-through estimator). All 13 tensors at 2 bits per-channel
  (20.7 KB), with the shared KD recipe, 20 epochs, seed 0.
- **Learning rates,** both fixed in advance: 1e-4 (the project's fine-tuning lr) and 1e-3 (TinyHAR's training lr).
- **Data and scoring:** fold 0, HARTH only, assessed exactly like the other models (`bitred/run_qat.py`).

**Rule (fixed before the run):**
- A run "recovers" if F1_val ≥ F1_val(original) − 0.03.
- **Gap:** the best LCI_val among recovering runs is ≤ 0.85, i.e. at least 0.05 below uniform 4-bit's 0.90. A new
  method then has something explanation-specific to fix.
- **No gap:** that LCI_val is > 0.85. A standard recipe already fixes 2 bits.
- **Inconclusive:** no run recovers F1. Then 2 bits is an accuracy problem first.

`python -m bitred.run_qat --fold 0 --device cuda --machine-name nakhoul-3050` writes
`bitred-qat2bit-tinyhar-fold0-<m>-{models, per-subject-lci, finetune-history}.csv`, `-decision.json` and
`-config.json`.

## Ultra-small gap test (added 2026-11-24 after the 2-bit gap test, before running it)

At 20.7 KB, per-channel 2-bit QAT + KD closed the gap. This test pushes the same question to 15–18 KB, where the
next size reduction needs a coarser quantizer or pruning on top of quantization (`bitred/run_qat_small.py`).

**Settings.** All sizes are honest storage with the bitmask:
- kept weights × bits;
- fp32 scales;
- 1 bit per position for every pruned tensor;
- fp32 for the 1,157 non-quantized parameters.

| Setting | Start | Quantizer | Size |
|---|---|---|---|
| `tensor-2bit` | the frozen TinyHAR | 2-bit, one scale per tensor | 17.5 KB |
| `lamp_recon90-channel-4bit` | lamp_recon at 90 % sparsity | 4-bit per channel | 16.9 KB |
| `lamp_recon95-channel-4bit` | lamp_recon at 95 % sparsity | 4-bit per channel | 15.6 KB |

- **lamp_recon** is round 4's baseline, built with its own functions (`har/xgp4.py`, `har/xgp3.py`) and values:
  - LAMP keep counts (Lee et al. 2021) through `bounded_budget`: floor 2 %, uniform counts for the tensors the
    reconstruction cannot refit, total = the uniform total;
  - K = 5 nested magnitude steps (`iterative_prune_budget`, β 0);
  - after each step, the ridge layer-wise reconstruction (damp 0.01) on round 4's calibration training windows
    (checked against round 4's committed table);
  - the head is not pruned, and nothing is trained by gradient before QAT.
- **QAT:** the 2-bit gap test's recipe unchanged (`bitred/qat.py`: STE, shared KD recipe, Adam, batch 256).
  - 20 epochs, seed 0, lr 1e-4 and 1e-3, last-epoch model, fold 0 first.
  - The pruning masks are fixed: masked gradients are zeroed before each step, the masks are re-applied after it,
    and the runner asserts that every pruned weight is still 0 after quantization.
- **References** (assessed, not used by the rule): the same model without QAT (`ptq-tensor-2bit`,
  `lamp_recon9x-ptq-channel-4bit`) and the fp32 pruned starts (`lamp_recon9x-fp32`, 33.1 / 22.8 KB).

**Rule:** the 2-bit gap test's rule (`run_qat.decide`), applied to each setting over its two lrs. It was fixed
before the run.
- A run recovers if F1_val ≥ F1_val(original) − 0.03.
- **Gap** at that size if the best LCI_val among recovering runs is ≤ 0.85.
- **No gap** if it is > 0.85.
- **Inconclusive** if no run recovers F1: that size is an accuracy problem first.

`decision.json` lists the verdict per setting and the settings with a gap.

`python -m bitred.run_qat_small --fold 0 --device cuda --machine-name <m>` writes
`bitred-qat-ultrasmall-tinyhar-fold0-<m>-{models, per-subject-lci, finetune-history, lamp-keep-counts}.csv`,
`-decision.json` and `-config.json`. Each `-models.csv` row holds:
- size in KB (with the bitmask, plus its parts);
- F1, accuracy;
- LCI, worst-subject LCI, ρ, J.

Estimated time: 6 QAT runs at ≈ 65–70 s of fine-tuning plus ≈ 45 s of SHAP each, and 5 reference and 1 original
assessments at ≈ 45 s each, so ≈ 15–20 min on a GPU. Smoke: 14 s CPU.

## Data-regime gap test (added 2026-11-25 after the ultra-small gap test, before running it)

The 2-bit and ultra-small gap tests fine-tune on all ≈ 51,450 training windows of the fold. This test asks whether the
verdicts change when QAT gets only a small share of the training data (`bitred/run_qat_regimes.py`).

**Regimes.** These are the QAT training windows, all HARTH training windows of fold 0. The teacher is the original
model's logits on the same windows.

| Regime | Windows | Source |
|---|---|---|
| `kd5` | 2,583 (5.0 %) | Round 4's 5 % subset: max(1, round(0.05 n)) per (training subject, class), seed 48 (`kd5_table`). Checked against round 4's committed id table. |
| `calibration` | 978 (1.9 %) | Round 4's calibration set: ≤ 10 per (subject, class), seed 46. These are the same windows lamp_recon's reconstruction uses. Checked against round 4's committed table. |

**Settings:** the ultra-small test's three, plus per-channel 2-bit from the 2-bit test.

| Setting | Size |
|---|---|
| `channel-2bit` | 20.7 KB |
| `tensor-2bit` | 17.5 KB |
| `lamp_recon90-channel-4bit` | 16.9 KB |
| `lamp_recon95-channel-4bit` | 15.6 KB |

**Optimisation: equal Adam steps, not equal epochs.** Every QAT run takes the full-data run's number of Adam
steps: 20 epochs × 201 steps on fold 0's 51,451 training windows = **4,020 steps** at batch 256. The small sets
therefore get many more epochs:

| Regime | Steps per epoch | Epochs for 4,020 steps |
|---|---|---|
| `kd5` | 11 | ≈ 365.5 (the last one stops mid-epoch) |
| `calibration` | 4 | 1,005 |

- Same lr pair (1e-4, 1e-3), seed 0, and a fresh seeded batch order every epoch.
- The model after the last step is scored.
- Validation F1 is logged every 201 steps, 20 points as in the full-data runs. It is only logged, never used to
  select a model.
- The step mode (`qat_kd_finetune(steps=…)`) reproduces the epoch mode exactly when the step counts match (tested),
  so the full-data runs are the same recipe at the same step count.

*Changed 2026-11-25, before any real run of this test.* The first version kept 20 epochs, which means 220 steps on
`kd5` and 80 on `calibration`. That mixes less data with less optimisation. Equal steps separate data size from
step count, so a difference from the full-data verdicts can be put down to data size alone.

**Everything else is unchanged:**
- the lamp_recon starts, built as in the ultra-small test;
- the QAT recipe (STE, shared KD, Adam, batch 256);
- masks fixed;
- scoring on inner val.

The no-QAT references (PTQ models and fp32 pruned starts) are assessed once, since they do not depend on the regime.

**Rule:** the 2-bit gap test's rule (`run_qat.decide`), fixed before the run, applied to each setting × regime
over its two lrs.
- A run recovers if F1_val ≥ F1_val(original) − 0.03.
- **Gap** if the best LCI_val among recovering runs is ≤ 0.85.
- **No gap** if it is > 0.85.
- **Inconclusive** if no run recovers F1.

`decision.json` gives the verdict per setting × regime and lists the cells with a gap. The full-data verdicts are the
earlier files: the 2-bit gap test for `channel-2bit`, and the ultra-small test for the other three.

`python -m bitred.run_qat_regimes --fold 0 --device cuda --machine-name <m>` writes
`bitred-qat-regimes-tinyhar-fold0-<m>-{models, per-subject-lci, finetune-history, lamp-keep-counts,
regime-train-window-ids}.csv`, `-decision.json` and `-config.json`.

Estimated time: 16 QAT runs of 4,020 steps each (≈ 50 s each, like a 20-epoch full-data run), plus 24 SHAP
assessments at ≈ 40 s each, so ≈ 30–35 min on a GPU. Smoke: 14 s CPU.

## Final comparison: XGP vs the best existing at ultra-small size (added 2026-11-25, before any run)

The gap tests found no explanation-specific gap below 20.7 KB: there, accuracy fails first. This last pre-registered
test asks the method question directly in the ultra-small pipeline (unstructured pruning, then 4-bit per-channel
weights, then QAT + KD). At the same size, does round 4's XGP keep explanations more consistent than the best
existing pruning baseline? Runner: `bitred/run_xgp_compare.py`.

**Methods,** per fold, built from the frozen TinyHAR with round 4's functions and values (`har/xgp4.py`,
`har/xgp3.py`, unchanged):

| Method | Round 4 cell | Budget | Selection |
|---|---|---|---|
| `lamp_recon` (best existing) | `U_CELLS["lamp_recon"]` = (lamp, l1) | LAMP keep counts (Lee et al. 2021) through `bounded_budget` | magnitude, β 0 |
| `xgp_xb` (ours) | `U_CELLS["xgp_xb"]` = (xb, xgp) | XGP shares (`worst_group_shares` of the original model) through the same `bounded_budget` | β 0.1 with the worst-group EG importance, recomputed every step |

Both methods share:
- K = 5 nested steps (`iterative_prune_budget`);
- after every step, the ridge reconstruction (`layerwise_reconstruct`, damp 0.01);
- round 4's calibration set (≤ 10 training windows per (subject, class), seed 46, groups = subjects) and the fold's
  100 SHAP background windows as EG baselines;
- top-k 3, m = 8, floor 2 %;
- EG seeds (0, 1), which is round 4's seed 0.

At each level both keep the same total of weights (the uniform total), so they have the same size. The head is never
pruned. On folds 0–1 the fp32 starts are checked against round 4's own no-fine-tune scores (`round4_f1`, `round4_lci`
in `-models.csv`). For lamp_recon at 90 % on fold 0, the ultra-small test already reproduces round 4 exactly
(F1 0.660375, LCI 0.599676).

**Pipeline and sizes** (KB with the bitmask):

| Level | Method | QAT | Size |
|---|---|---|---|
| 90 % | `lamp_recon`, `xgp_xb` | 4-bit per channel, QAT + KD | 16.92 KB |
| 95 % | `lamp_recon`, `xgp_xb` | 4-bit per channel, QAT + KD | 15.64 KB |
| — | `tensor-2bit`, the quantization-only reference | 2-bit per tensor, QAT + KD | 17.54 KB |

- **QAT** (`bitred/qat.py`): STE, shared KD recipe, Adam, batch 256, masks fixed.
  - **4,020 Adam steps in every regime and fold** (20 epochs of fold 0's full set), with validation logged every
    201 steps.
  - lr 1e-4 and 1e-3, seed 0, the model after the last step.
- **Regimes** (QAT training windows, HARTH training subjects of the fold):

  | Regime | Windows |
  |---|---|
  | `full` | all |
  | `kd5` | round 4's 5 % subset, seed 48 |
  | `calibration` | round 4's calibration set |

  Round 4 committed the kd5 and calibration tables for folds 0–1 only. Those are checked; folds 2–4 rebuild the
  same deterministic selections.
- **No-QAT references,** assessed once per start: the fp32 start and its 4-bit PTQ model.
- **Folds 0–4,** seed 0, HARTH only. Every number is on the fold's 3 inner-val subjects, as in every bit-reduction
  test:
  - F1 on all inner-val windows;
  - LCI and worst-subject LCI on the 100 explain windows per subject.

**Primary lr (fixed and confirmed 2026-11-25, before the run):** per regime, the lr with the higher mean F1_val of `lamp_recon` over the
5 folds and both levels. A tie goes to 1e-4. It is chosen on the comparator only, never on XGP or on LCI. That gives
the baseline its best-accuracy lr. Both lrs are reported in `-comparisons.csv`; only the primary one decides.

**Rule R2 (fixed before the run; `bitred.rules.compare`).** At each level × regime, at the primary lr, XGP beats
LAMP iff, over the 5 folds, all of these hold:
- (a) the mean LCI_val gain is ≥ 0.02;
- (b) LCI_val is strictly higher in ≥ 4 of 5 folds;
- (c) mean F1_val ≥ LAMP's − 0.01;
- (d) mean worst-subject LCI_val ≥ LAMP's − 0.01.

With fewer than 5 folds the verdict is "incomplete". `-decision.json` gives the verdict for each of the 6
level × regime cells, and `xgp_beats_lamp_anywhere`. Nothing is retuned after the run.

**Outputs:**
- per fold: `bitred-xgp-compare-tinyhar-fold<k>-<m>-{models, per-subject-lci, finetune-history, budgets}.csv`
  and `-config.json`;
- `--analyze`: `-summary-mean-sd-over-folds.csv`, `-per-fold.csv`, `-primary-lr.csv`, `-comparisons.csv` and
  `-decision.json`.

**Time per fold** (estimate from the measured gap tests on uni-2080):
- 30 QAT runs at ≈ 50 s;
- 40 SHAP assessments at ≈ 40 s;
- the XGP shares and 2 xgp_xb builds at ≈ 1.5–2 min each (round 4: 91–110 s);
- so ≈ 55–60 min on uni-2080, and ≈ 65–75 min on nakhoul-3050.

`--resume` continues a fold: it skips finished models and reuses the saved starts. Smoke: ≈ 25 s per fold on CPU.

## Final test on HAR70+ (pre-registered 2026-11-25, before any code or run; run once)

The final comparison's one winning cell (90 %|kd5 at lr 1e-3; see Results) is tested on held-out subjects, older
adults included. Runner: `bitred/run_final_test.py`. Nothing in the method, the baselines, the QAT + KD recipe
(4,020 Adam steps, batch 256, lr 1e-3 primary), the metrics or the 9A test protocol changes.

**Models per fold (0–4)**, rebuilt exactly as in `run_xgp_compare` (same functions, values, regimes and code path):

| Models | Levels × regimes | lr | Seeds |
|---|---|---|---|
| `lamp_recon`, `xgp_xb` (+ 4-bit per channel, QAT + KD) | 90 / 95 % × full / kd5 / calibration | 1e-3 (primary), 1e-4 | 0 |
| the same two, 90 %\|kd5 only | 90 % × kd5 | 1e-3, 1e-4 | 1, 2 |
| ablations `l1_xb_recon` (XGP budgets + magnitude selection) and `xgp_lamp` (LAMP budgets + XGP selection) | all 6 cells | 1e-3, 1e-4 | 0 |
| `tensor-2bit` (per-tensor 2-bit QAT + KD, 17.54 KB) | each regime | 1e-3, 1e-4 | 0 |
| `original` | — | — | — |

That makes 62 QAT models plus the original per fold. Sizes include the bitmask: 16.92 KB at 90 %, 15.64 KB at 95 %.

- **Seed s** follows round 4's convention: QAT batch-order seed s, and EG seeds (2s, 2s + 1) for the XGP importance
  and shares. lamp_recon's start is deterministic, so for seeds 1–2 only its QAT seed changes.
- **Seed-0 starts:** where `run_xgp_compare`'s checkpoint of the same start is on the machine, it is reused.
  Otherwise the start is rebuilt with the same code. `start_source` records which.
- **Retraining:** the comparison's QAT models were not saved, and GPU QAT is not bit-reproducible (the run-to-run
  F1 drift is ≈ 0.01). So every QAT model is retrained here. Their validation numbers are reported next to the
  comparison's as a reproduction check.

**Order (enforced in code):**
1. `--phase validation`, per fold. It trains every model, saves its checkpoint and scores it on inner val with the
   final comparison's protocol: F1_val, LCI_val and worst-subject LCI_val. It never loads HAR70+ or the fold's HARTH
   test subjects.
2. `--analyze-validation`. It needs all 5 folds, writes the validation tables and records each fold's sha256 in
   `validation-complete.json`.
3. `--phase test`, per fold. It refuses until that file exists and the fold's validation file still matches its
   sha256. It assesses the saved checkpoints (no training) and checks that each one's inner-val F1 equals its
   validation F1.
4. `--analyze`. It needs all 5 test folds; otherwise there is no verdict. Test files are never overwritten
   (`--resume` only continues).

**Test protocol (9A, `har.run_compression_grid.Context` and `assess`, unchanged):**
- **Explain windows:** the fold's fixed 9A windows (inner val 90, general 120, older 540).
- **SHAP:** GradientExplainer, nsamples 200, SHAP seed 0, target = the original model's predicted class,
  background = the fold's 100 HARTH windows.
- **LCI:** 48 segments, k = 10, ω = 0.5. τ_k and the ceilings come from the committed noise-floor files: folds 0–1
  nakhoul-3050, folds 2–4 uni-2080, as in the 9A analysis.
- **AOPC:** deletion faithfulness with the random-deletion baseline.
- **Accuracy and non-stair macro-F1:** on all of the subject's test windows.
- **Unit = test subject (40):** the 22 HARTH general subjects, each from its single test fold, and the 18 HAR70+
  subjects, each averaged over the 5 fold models (`har.stats.pool_over_folds`). Walking-aid (inferred) = 503, 505,
  513, 514, 515 (5); other older = 13.
- **Per group** (older vs general; walking-aid vs other older): accuracy, non-stair F1, LCI, FA-LCI, AOPC. Also the
  gaps with bootstrap CIs (10,000 draws, seed 42), the safe-explanation rate (τ_k, global τ, 0.85) and the size in KB
  with the bitmask.
- **d_accuracy:** per (fold, subject), the model's accuracy minus the original model's.

**Hypotheses:**
- **H1 (single primary, confirmatory, no multiplicity correction).** At 90 %|kd5, lr 1e-3:
  - Each test subject's LCI is the mean over seeds 0, 1, 2 (each seed pooled over folds first).
  - Test: paired one-sided Wilcoxon signed-rank, xgp_xb > lamp_recon, over the 40 subjects (zero_method "wilcox").
  - **Supported iff p < 0.05.**
  - Also reported: the median and mean difference, rank-biserial r and d_z; non-stair F1 (per subject, two-sided
    Wilcoxon, descriptive) and pooled 7-class F1 per group; worst-group LCI = min(LCI_older, LCI_general) = FA-LCI
    at λ 0.5, with the paired bootstrap CI of its difference.
- **H2.** Same cell, seed mean.
  - gap = mean_older(AOPC) − mean_general(AOPC) per method.
  - D2 = |gap_lamp_recon| − |gap_xgp_xb|.
  - Paired stratified bootstrap: subjects are resampled within each group, with the same draws for both methods.
  - **Supported iff the 95 % CI of D2 is > 0.** The signed gaps and their CIs are also reported.
- **H3 (exploratory, 5 vs 13 subjects).** Same cell, seed mean.
  - wa_gap = mean_walking_aid(d_accuracy) − mean_other_older(d_accuracy) (`d_accuracy_wa_gap`; < 0 = walking-aid
    users lose more).
  - D3 = wa_gap_xgp_xb − wa_gap_lamp_recon.
  - Paired bootstrap within the 5 and the 13.
  - **"Supported (exploratory)" iff the 95 % CI of D3 is > 0.**
- **Secondary cells.** The other 5 level × regime cells, lr 1e-3, seed 0: the same one-sided Wilcoxon on per-subject
  LCI, Holm over the 5. The H1 cell at seed 0 alone is descriptive.

**Secondary analyses** (pre-registered as secondary; none of them changes H1–H3):
1. **Ablations,** in every cell at both lrs (seed 0).
   - Validation (folds 0–4) first: means, and R2 vs lamp_recon, descriptive.
   - Then test: one-sided Wilcoxon on LCI vs lamp_recon, Holm over the 12 (2 ablations × 6 cells, lr 1e-3).
   - Also xgp_xb vs each ablation, two-sided, Holm over the 12.
2. **Seeds 1 and 2 at 90 %|kd5 on validation:** per-seed means and R2, descriptive.
3. **Both lrs:** every H1–H3 and secondary-cell statistic is also computed at lr 1e-4, labelled secondary.
4. **LCI sensitivity:** per-subject LCI is recomputed from the saved 48-segment SHAP with k ∈ {5, 10, 20} and
   ω ∈ {0.25, 0.5, 0.75} at 90 %|kd5 (seed mean). H1's Wilcoxon is reported for each of the 9 settings.

**Outputs** (repo-root `results/har-final-test-ultrasmall/`, prefix `har-final-test-ultrasmall-tinyhar`; the smoke
run writes to `results/har-final-test-ultrasmall-smoke/`):
- **`validation/`, per fold:** `-fold<k>-<m>-{models, per-subject-lci, finetune-history, budgets}.csv` and
  `-config.json`.
- **`validation/`, from `--analyze-validation`:** `-validation-summary-mean-sd-over-folds.csv`,
  `-validation-comparisons-vs-lamp-recon.csv`, `-validation-seeds-90pct-kd5.csv`,
  `-validation-reproduction-vs-xgp-compare.csv` and `-validation-complete.json`.
- **Test, per fold:** `-fold<k>-<m>-{per-subject-metrics, per-subject-class-lci, per-group-class-lci, models}.csv`,
  `-shap-48-segments.npz` and `-config.json`.
- **Test, from `--analyze`:** `-decision.json` (H1–H3), `-hypotheses-both-lrs.csv`, `-secondary-cells-holm.csv`,
  `-ablations-test-holm.csv`, `-group-summary-gaps-bootstrap-ci.csv`, `-fa-lci-bootstrap-ci.csv`,
  `-safe-explanation-rate.csv`, `-per-subject-pooled-over-folds.csv`, `-lci-sensitivity-k-omega.csv` and
  `-models-summary.csv`.

**Time per fold** (estimate from the final comparison's and 9A's measured times):

| Machine | Validation phase | Test phase | Fold total |
|---|---|---|---|
| uni-2080 | 62 QAT × ≈ 51 s + 63 inner-val SHAP × ≈ 39 s + ≈ 6 min of starts ≈ 1.7 h | 63 × ≈ 130 s (9A assess) ≈ 2.3 h | ≈ 4 h |
| nakhoul-3050 | 62 × ≈ 65 s + 63 × ≈ 43 s + ≈ 12 min of starts ≈ 2.1 h | 63 × ≈ 100 s ≈ 1.8 h | ≈ 3.9 h (≈ 5 h if it throttles) |

- On uni-2080 the seed-0 xgp_xb starts are reused from the comparison's checkpoints; there, the comparison ran all
  5 folds.
- **Split across the two machines:** folds 0 1 on nakhoul-3050 and 2 3 4 on uni-2080.
  - Validation ≈ 5 h; then merge and run `--analyze-validation`; then test ≈ 7 h.
  - **≈ 12 h wall**, plus the merges.
- **Two terminals on uni-2080:** each process is ≈ 1.5× slower (as measured in the comparison), so ≈ 16–17 h wall.
- **Smoke** (CPU, folds 0–1): validation 1.8 min, test 1.8 min, analysis ≈ 10 s.

## Commands (repo `.venv`, run from this folder)

```
cd bit-reduction-test
python -m pytest tests                                             # 38 tests
python -m bitred.run --prepare                                     # pre-registration (CPU, seconds); commit it before the runs
python -m bitred.run --folds 0 1 2 3 4 --device cuda --machine-name nakhoul-3050
#   or split: nakhoul-3050 --folds 0 1 2, uni-2080 --folds 3 4 (same flags, own --machine-name)
python -m bitred.run --analyze                                     # after merging; --use-machine k=<m> if a fold ran twice
# --resume continues an interrupted fold; --smoke writes a pipeline check to results-smoke/ (git-ignored)
python -m bitred.run_qat --fold 0 --device cuda --machine-name nakhoul-3050         # 2-bit gap test (done)
python -m bitred.run_qat_small --fold 0 --smoke --device cpu --machine-name smoke-cpu
python -m bitred.run_qat_small --fold 0 --device cuda --machine-name nakhoul-3050   # ultra-small gap test (done)
python -m bitred.run_qat_regimes --fold 0 --smoke --device cpu --machine-name smoke-cpu
python -m bitred.run_qat_regimes --fold 0 --device cuda --machine-name <m>         # data-regime gap test (done)
python -m bitred.run_xgp_compare --folds 0 --smoke --device cpu --machine-name smoke-cpu
python -m bitred.run_xgp_compare --folds 0 1 --device cuda --machine-name nakhoul-3050    # final comparison
python -m bitred.run_xgp_compare --folds 2 3 4 --device cuda --machine-name uni-2080
python -m bitred.run_xgp_compare --analyze
# final test on HAR70+ (order enforced: validation on all folds -> --analyze-validation -> test -> --analyze)
python -m bitred.run_final_test --phase validation --folds 0 1 --smoke --device cpu --machine-name smoke-cpu
python -m bitred.run_final_test --analyze-validation --smoke
python -m bitred.run_final_test --phase test --folds 0 1 --smoke --device cpu --machine-name smoke-cpu
python -m bitred.run_final_test --analyze --smoke
python -m bitred.run_final_test --phase validation --folds 0 1 --device cuda --machine-name nakhoul-3050
python -m bitred.run_final_test --phase validation --folds 2 3 4 --device cuda --machine-name uni-2080
python -m bitred.run_final_test --analyze-validation              # after merging both machines' validation/ files
python -m bitred.run_final_test --phase test --folds 0 1 --device cuda --machine-name nakhoul-3050
python -m bitred.run_final_test --phase test --folds 2 3 4 --device cuda --machine-name uni-2080
python -m bitred.run_final_test --analyze                         # after merging; writes -decision.json (H1-H3)
```

**Time:** about 12,500 SHAP windows per fold (probe 7 × 300, sensitivity 52 × 140, allocation 9 × 300, original). At
≈ 0.14–0.15 s per window that is ≈ 30–40 min per fold on a GPU, ≈ 2.5–3.4 h for 5 folds on one machine, or
≈ 1.5–2 h split across both. Hutchinson and KL take about 2 min per fold. Smoke: 26 s CPU, 18 s CUDA (fold 0).

## Outputs (`results/`)

Per fold and machine (`bitred-tinyhar-fold<k>-<machine>-…`):

| File | Content |
|---|---|
| `-models.csv` | One row per assessed model: size, mean bits, bits per tensor, LCI, worst-subject LCI, ρ, J, F1, accuracy, retention, below τ_k |
| `-per-subject-lci.csv` | Per-subject LCI |
| `-sensitivity.csv` | Tensor × bits: the three Ω, calibration LCI / ρ / J, trace |
| `-hessian-traces.csv` | Hutchinson traces with standard error |
| `-allocations.csv` | Bits per tensor, per method × budget |
| `-config.json`, `logs/` | Run configuration and console logs |

`--analyze` writes:
- `bitred-tinyhar-summary-mean-sd-over-folds.csv`;
- `-probe-headroom.csv` (R1);
- `-allocation-comparisons.csv` (R2);
- `-sensitivity-mean-over-folds.csv`;
- `-sensitivity-rank-agreement.csv`: Spearman between the methods' Ω per fold. If XGQ's ranking matches KL's,
  the explanation objective adds nothing;
- `-allocations-bits-per-tensor.csv`;
- `-decision.json` (R1–R3).

## Results (fold 0 only, nakhoul-3050, 2026-11-24)

### `bitred-tinyhar-fold0-nakhoul-3050-*` (main test, 31 min)
- **Original:** F1 0.818. τ_0 = 0.659; the SHAP-noise ceiling is 0.946.
- **Uniform probe** (LCI / F1):

  | Model | Size | LCI | F1 |
  |---|---|---|---|
  | Per-tensor 4-bit | 30.5 KB | 0.895 | 0.817 |
  | Per-tensor 3-bit | 24.0 KB | 0.888 | 0.808 |
  | Per-tensor 2-bit | 17.5 KB | 0.310 | 0.223 |
  | Per-channel 8-bit | 59.6 KB | 0.970 | 0.819 |
  | Per-channel 4-bit | 33.7 KB | 0.900 | 0.809 |
  | Per-channel 3-bit | 27.2 KB | 0.935 | 0.827 |
  | Per-channel 2-bit | 20.7 KB | 0.535 | 0.398 |

  3 bits costs nothing (within SHAP noise). 2 bits collapses LCI and F1 together.
- **Allocations** (LCI / F1):

  | Budget | XGQ | ZeroQ-KL | HAWQ |
  |---|---|---|---|
  | 3bit (27.2 KB) | 0.895 / 0.813 | 0.925 / 0.801 | 0.956 / 0.811 |
  | 2.5bit (23.9 KB) | **0.845 / 0.808** | 0.817 / 0.791 | 0.808 / 0.787 |
  | 2.25bit (22.3 KB) | **0.685 / 0.685** | 0.839 / 0.800 | 0.806 / 0.784 |

  XGQ put convs.3/6/9 at 2 bits: each is harmless alone (calibration LCI 0.96–0.97), but together they compound.
  The additive per-tensor LCI sensitivity fails.
- **Fold-0 screen:** S1 passes (via 2 bits only) and S2 passes (2.5bit, +0.028). Folds 1–4 were not run (see below).

### `bitred-qat2bit-tinyhar-fold0-nakhoul-3050-*` (2-bit gap test, 3.7 min)
- **QAT 2-bit per-channel (20.7 KB), lr 1e-4:** F1 **0.806**, LCI **0.886** (worst subject 0.852).
- **lr 1e-3:** F1 0.717, LCI 0.777. Its F1 swings 0.65–0.83 between epochs.
- **Verdict (pre-registered rule): no gap.** Standard QAT + KD recovers both F1 (−1.2 pt) and LCI (≈ uniform
  4-bit's 0.90) at 2 bits, 10× smaller than fp32.
- It also beats every allocation above while being smaller (20.7 KB vs 23.9 KB for XGQ's 0.845 / 0.808). So the
  XGQ claim is moot, and folds 1–4 are not worth running.
- **Caveat:** one fold, last-epoch model; the lr 1e-4 run's per-epoch F1 varies 0.74–0.83.

### `bitred-qat-ultrasmall-tinyhar-fold0-uni-2080-*` (ultra-small gap test, 13.7 min)
- **Original:** F1 0.818, so a run recovers at F1 ≥ 0.788.
- **Results** (QAT runs at lr 1e-4 / 1e-3; LCI with the worst subject in brackets):

  | Setting | Size | No QAT (F1 / LCI) | QAT lr 1e-4 (F1 / LCI) | QAT lr 1e-3 (F1 / LCI) |
  |---|---|---|---|---|
  | `tensor-2bit` | 17.5 KB | 0.223 / 0.310 | 0.783 / 0.746 (0.652) | 0.733 / 0.658 (0.649) |
  | `lamp_recon90-channel-4bit` | 16.9 KB | 0.648 / 0.597 | 0.739 / 0.817 (0.812) | 0.780 / 0.889 (0.827) |
  | `lamp_recon95-channel-4bit` | 15.6 KB | 0.481 / 0.258 | 0.674 / 0.753 (0.718) | 0.705 / 0.740 (0.685) |

  The fp32 lamp_recon starts score F1 / LCI 0.660 / 0.600 at 90 % and 0.490 / 0.257 at 95 %.
- **Verdict (pre-registered rule): inconclusive at every size.** No run recovers F1, so below 20.7 KB the problem is
  accuracy first.
  - The near misses: `tensor-2bit` at lr 1e-4 (F1 −3.5 pt, LCI 0.746) and lamp_recon 90 % at lr 1e-3 (F1 −3.8 pt,
    LCI 0.889).
  - The second keeps LCI above 0.85 while F1 is short, so explanations are not what breaks first.
- **Sizes:** they include the 6.44 KB bitmask of the pruned tensors. Without it, 10.49 / 9.20 KB.

### `bitred-qat-regimes-tinyhar-fold0-uni-2080-*` (data-regime gap test, 24.5 min)
- **Setup:** equal optimisation, 4,020 Adam steps per QAT run (≈ 49 s each in every regime). Original F1 0.818, so a
  run recovers at F1 ≥ 0.788.
- **Results** (F1 / LCI per lr):

  | Setting | Size | kd5, lr 1e-4 | kd5, lr 1e-3 | calibration, lr 1e-4 | calibration, lr 1e-3 |
  |---|---|---|---|---|---|
  | `channel-2bit` | 20.7 KB | **0.795 / 0.856** | 0.710 / 0.748 | 0.743 / 0.836 | 0.756 / 0.666 |
  | `tensor-2bit` | 17.5 KB | 0.695 / 0.726 | 0.778 / 0.635 | 0.737 / 0.715 | 0.727 / 0.599 |
  | `lamp_recon90-channel-4bit` | 16.9 KB | 0.761 / 0.829 | 0.762 / 0.794 | 0.666 / 0.797 | 0.713 / 0.779 |
  | `lamp_recon95-channel-4bit` | 15.6 KB | 0.674 / 0.771 | 0.715 / 0.793 | 0.619 / 0.697 | 0.658 / 0.678 |

- **Verdicts (pre-registered rule):**
  - `channel-2bit` with kd5: **no gap** (F1 0.795, LCI 0.856, worst subject 0.822).
  - Every other cell: **inconclusive**. No run recovers F1.
  - No cell has a gap.
- **Reading:**
  - With 5 % of the data, 2-bit QAT still recovers both F1 and LCI at 20.7 KB, as with full data.
  - On the calibration set (978 windows) it does not recover F1.
  - Below 20.7 KB, F1 fails first in every regime, as in the full-data run.

### `bitred-xgp-compare-tinyhar-*` (final comparison, folds 0–4, uni-2080, 2026-11-25)
- **Runs:** both commands ran at the same time on uni-2080 (folds 0 1 in 2.7 h, folds 2 3 4 in 3.6 h). Mean over
  the 5 folds: original F1 0.835.
- **Primary lr = 1e-3 in every regime.** lamp_recon's mean F1 at 1e-4 / 1e-3: full 0.763 / 0.799, kd5 0.767 / 0.782,
  calibration 0.757 / 0.764.
- **R2 at the primary lr** (`-decision.json`):

  | Cell | LCI xgp_xb vs lamp_recon | Folds higher | ΔF1 | Δworst | Verdict |
  |---|---|---|---|---|---|
  | 90 %\|full | 0.822 vs 0.814 (+0.008) | 3/5 | +0.003 | +0.001 | does not beat |
  | **90 %\|kd5** | **0.790 vs 0.750 (+0.040)** | **4/5** | **+0.016** | **+0.065** | **XGP beats LAMP** |
  | 90 %\|calibration | 0.793 vs 0.773 (+0.019) | 3/5 | +0.016 | +0.030 | does not beat (gain < 0.02, 3/5 folds) |
  | 95 %\|full | 0.736 vs 0.726 (+0.010) | 3/5 | +0.021 | +0.026 | does not beat |
  | 95 %\|kd5 | 0.693 vs 0.693 (−0.000) | 4/5 | +0.014 | −0.029 | does not beat |
  | 95 %\|calibration | 0.716 vs 0.695 (+0.021) | 3/5 | +0.014 | +0.040 | does not beat |

- **At lr 1e-4** (secondary): XGP beats LAMP in 5 of 6 cells (all but 95 %|full).
  There the gains are larger (+0.03 to +0.05) but LAMP is weaker.
- **F1:** xgp_xb's mean F1 is ≥ lamp_recon's in every cell (+0.3 to +2.4 pt).
- **Reference, per-tensor 2-bit QAT** (17.5 KB) at lr 1e-3, F1 / LCI: full 0.801 / 0.594, kd5 0.783 / 0.563,
  calibration 0.768 / 0.599. At lr 1e-4: 0.808 / 0.771 with full data. Both pruned + 4-bit models are smaller and
  have higher LCI.
- **Reading:** XGP's explanation gain is real but small. It passes R2 only where LAMP is weakest (5 % data, 90 %).
  This single cell is the H1 cell of the final test on HAR70+.
