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

## Commands (repo `.venv`, run from this folder)

```
cd bit-reduction-test
python -m pytest tests                                             # 19 tests
python -m bitred.run --prepare                                     # pre-registration (CPU, seconds); commit it before the runs
python -m bitred.run --folds 0 1 2 3 4 --device cuda --machine-name nakhoul-3050
#   or split: nakhoul-3050 --folds 0 1 2, uni-2080 --folds 3 4 (same flags, own --machine-name)
python -m bitred.run --analyze                                     # after merging; --use-machine k=<m> if a fold ran twice
# --resume continues an interrupted fold; --smoke writes a pipeline check to results-smoke/ (git-ignored)
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
