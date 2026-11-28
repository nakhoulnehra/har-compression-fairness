# Gap study v6: run log

2026-11-28, Ansel (RTX 3050 Ti 4 GB, 16 GB RAM). Goal: find a **new** explanation gap after compression, not
re-validate an old one. No new method. HARTH inner validation only for every choice. Protocol, metric, windows and
the frozen TinyHAR are unchanged from v3–v5 (`lci_indep`, ceiling = the original's own SHAP-noise LCI, 0.932).

## 12:35 Theory, before any run

What v3–v5 established: within a fold, explanation fidelity tracks the label-free output KL on the calibration
windows (Spearman −0.57 at 80 %, −0.80 at 90 %), not accuracy. Every gap closed once a baseline minimised output
divergence well enough (OBC masks, KL-chosen layer budgets, fmkd). So a real gap needs a setting where:
(a) output fidelity does not pin what the explainer reads; or
(b) the output-fidelity tools cannot be applied.

Candidates, ranked by theory:

| # | Candidate | Why the existing methods would fail | Data check (existing files) | Rank |
|---|---|---|---|---|
| A | **Compression that leaves the original's weight basin**: dense narrow student (structured width reduction, the MCU form: no sparse kernels) | GradientExplainer gives φ_i = (x_i − b_i) E[∂f/∂z_i] along the path z = b + a(x − b). Output matching on mixup blends (fmkd, Beyer 2022) constrains f along these paths, so only the **sum** Σφ_i (completeness), not the split over 48 features. In-place pruning inherits the split through the shared weights, which is why output KL predicted LCI in v5. A fresh student has no such inheritance: the split is a Rashomon choice. | The retrain-seed floor τ = 0.63–0.66 (same data, same architecture, same accuracy, independent seeds) vs the ceiling 0.93: the data does not determine the split. Nothing in v3–v5 trained a model away from the original's weights. | 1 |
| B | Data-scarce one-shot (≪ 1k calibration windows) | rank-deficient Hessian; noisy KL choice | theory predicts function fidelity drops too, i.e. an accuracy gap, not an explanation gap | 3 |
| C | Calibration / deployment shift (older adults, sensor noise or rotation) | calibration does not cover the deployment region | v5 shift probe: no larger gap; v3: HAR70+ gap (0.117) smaller than HARTH (0.144) | rejected |
| D | Low-bit PTQ (3-bit, ≈ 27 KB) | — | GPTQ / OBQ (the OBC analogue) never run, so by v5's lesson it most likely closes; 2-bit QAT headroom was 0.08 | 4 |
| E | Deterministic explainers (occlusion, ceiling 1.0; AKL 0.814) | — | occlusion LCI also tracks output fidelity; changing the explainer is metric shopping | 5 |
| F | Other architectures (HART, DeepConvLSTM) | same in-basin argument → expected to behave like TinyHAR | — | 6 |

Candidate A goes to experiments first.

## 12:55 Candidate A: rules written before any result

Code: `gap_study_v6/run_v6.py`, stage `stu:F`. Student = TinyHAR with F filters (same structure, same input
normalisation). Primary F = 12: 10,401 parameters, 19.2 % of the original (the parameter count of 80 % sparsity,
dense). All runs seed 0, 30 epochs, batch 256, last state kept.

Existing baselines (strongest recipes in the repo plus the existing explanation-transfer method):
- `kd`: Hinton KD (T 3, α 0.7), lr 1e-3, from scratch;
- `fmkd`: KD + function matching on mixup blends (Beyer et al. 2022; the recipe that closed v4 / v5), lr 1e-3, from scratch;
- `fmjm`: fmkd + Jacobian matching of the explained logit's input gradient on SHAP-style path points of training
  windows (Srinivas & Fleuret 2018; v5's batch-normalised term, λ 10), lr 1e-3, from scratch;
- `slice_fmkd`: structured L1 filter pruning of the original to width 12 (Li et al. 2017; every dependent tensor
  sliced; slicing to width 28 reproduces the original exactly, checked), then fmkd, lr 3e-4. This keeps part of
  the original's weights.

Reference: `ce`, label-only training from scratch (the floor).

Oracles (not methods): `capfit` / `slice_capfit` = fmkd + gradient-field matching on the 150 **evaluated** windows'
path points (v5's capacity oracle, λ 10).

Decision rules:
1. y = the best existing baseline (mean `lci_indep` over the folds run). x = the best oracle. u = ceiling.
2. **Gap** if x − y ≥ 0.05, the fold-paired 95 % CI excludes 0, positive on every fold run (≥ 3 folds), and y's
   accuracy is within 2 pt of the original's.
3. **Fail** if y ≥ 0.85 (as good as v5's in-place compression at the same parameter count; then leaving the basin
   costs nothing extra), or if x − y < 0.05.
4. Validation only; nothing is read on test.

## 13:15 First readings, width 12 (folds 0 and 2)

| Model | Fold 0 | Fold 2 | Accuracy Δ (f0 / f2) | Occlusion LCI (f0 / f2) | cal KL (f0 / f2) |
|---|---|---|---|---|---|
| fmkd from scratch | 0.823 | 0.749 | −0.7 / −0.2 pt | 0.591 / 0.603 | 0.019 / 0.016 |
| slice_fmkd | 0.888 | 0.725 | +0.4 / −0.3 pt | 0.764 / 0.631 | 0.021 / 0.023 |
| capfit (oracle, from scratch) | 0.873 | 0.773 | −0.5 / −0.2 pt | 0.640 / 0.641 | 0.057 / 0.093 |
| ceiling | 0.945 | 0.915 | | | |

- **Theory check passes:** the KL law breaks for students.
  - fmkd from scratch has a lower LCI than v5's in-place models at the same output KL:
    - fold 0: KL 0.019 gives 0.823, against 0.84–0.92 for in-place models at KL 0.016–0.019;
    - fold 2: KL 0.016 gives 0.749, against 0.79–0.89.
  - Occlusion LCI is ≈ 0.60, against ≈ 0.80 for in-place models at that KL.
  - So output fidelity no longer pins the explanation once the original's weights are left.
- **Rule 2 does not pass yet at width 12.** The oracle from scratch is only +0.017 above the best baseline on
  average, so the narrow model looks capacity-limited, as in v5.
- Decision (before any width-20 result):
  - stop the width-12 kd / fmjm / ce runs and run `slice_capfit` at width 12 (the oracle from the strongest start);
  - add **width 20** (28,049 parameters, 52 %, the 50 % compression level): `slice_fmkd`, `fmkd`, `slice_capfit`,
    `fmjm`. There a capacity limit is unlikely, so a gap would come from leaving the basin.
  - Same rules as at 12:55.

## 13:32 Width 12 decided: fails rule 2

`slice_capfit` (oracle from the sliced start): 0.873 (fold 0) and 0.743 (fold 2), no higher than `slice_fmkd`
(0.888 / 0.725). Even a direct fit to the evaluated windows' gradient field does not lift a width-12 dense model,
so the deficit at width 12 is a capacity limit of that architecture, not a missing objective. Width 20 is still
running.

## 13:45 Candidate B: rules written before any result (CPU data check next to the GPU runs)

Theory: the deployer has only N **unlabelled** windows (random from the training stream, no class balancing; v3–v5
used ≤ 10 windows per labelled (subject, class) cell).
- The refit of the fusion layer (168 inputs, 8 time steps per window) becomes rank-deficient below ≈ 21 windows.
- The KL choice among allocations becomes noisy.
- Prediction: weak; the refit is dampened and well determined above ≈ 21 windows.

Models (`scarce:N`, 80 %):
- `akl`: v5's strongest one-shot combination (OBC + full refit, allocation with the lowest KL on the same N windows);
- `akl_mix`: the same with v4's blended calibration;
- `obcu`: uniform OBC.

Achievable reference: v5's `akl` with 978 balanced windows (0.864; capacity oracle 0.875).
**Gap** if the best of the three at N is ≥ 0.05 below the same fold's 978-window `akl` (seed 0) with accuracy within
2 pt (the gap must not be an accuracy collapse), on ≥ 4/5 folds. Data check first: fold 0, N ∈ {16, 64}, seed 0.

## 13:45 Candidate B fails (fold 0, seed 0; worker stopped)

| N unlabelled windows | akl | akl_mix | obcu | F1 (akl / mix / obcu; original 0.818) | accuracy (original 0.891) |
|---|---|---|---|---|---|
| 16 | 0.700 | 0.752 | 0.722 | 0.629 / 0.683 / 0.645 | 0.855 / 0.875 / 0.883 |
| 64 | **0.949** | — | — | 0.754 | 0.879 |
| 978 balanced (v5, KL choice a5) | 0.925 | | | | |

- Above the predicted rank threshold (≈ 21 windows), 64 random unlabelled windows already reach the ceiling (0.945).
- Below it, F1 collapses (−13 to −19 pt): an accuracy failure, not an explanation gap.
- B is rejected; theory and data agree.

## 13:45 Candidate A at width 20: the sliced start closes it

| Width 20 (28,049 parameters) | Fold 0 | Fold 2 |
|---|---|---|
| slice_fmkd | **0.920** (accuracy +0.2 pt, occlusion 0.817) | **0.849** (+0.1 pt, occlusion 0.655) |
| ceiling | 0.945 | 0.915 |

- When the student is a sub-network of the original and has the capacity, inheriting the weights (Li et al. 2017
  slicing + fmkd) keeps the explanation. So for the TinyHAR family, structured compression is closed by an
  existing route.
- What is left to test is distillation **without** weight inheritance, the only option when the student's
  architecture differs from the teacher's. Width 20 from scratch gives a same-architecture case where a same-size
  model with high LCI is known to exist (`slice_fmkd`, 0.885 mean), so the maximum is measured.
- Queue (restart 13:45): folds 0 / 2 `fmkd`, `fmjm`, `capfit` at width 20; fold 4 `slice_fmkd`, `fmkd`, `fmjm`,
  `capfit`.
- Rule for this scoped question, written now: the from-scratch gap counts if the best from-scratch baseline (kd /
  fmkd / fmjm) is ≥ 0.05 below the best same-size model (sliced or oracle) on every fold run, with accuracy within
  2 pt. It is reported **as scoped to students that cannot inherit the teacher's weights**, because slicing closes
  it otherwise.

## 14:00 Candidate A fails at width 20 too (from scratch); workers stopped

| Width 20 | Fold 0 | Fold 2 | Fold 4 |
|---|---|---|---|
| fmkd from scratch | **0.935** (accuracy −2.0 pt) | 0.825 (0.0 pt) | — |
| slice_fmkd | 0.920 | 0.849 | 0.868 |
| ceiling | 0.945 | 0.915 | — |

- A student with enough capacity keeps the explanation without inheriting any weight: from-scratch fmkd is within
  0.024 of the sliced model and above it on fold 0. The from-scratch gap fails (< 0.05).
- The width-12 decoupling (lower LCI than in-place models at the same output KL) is therefore a capacity effect of
  the narrow dense architecture; at width 12 even the oracles do not lift it.
- Candidate A is rejected; `fmjm` / `capfit` at width 20 were stopped, since they cannot change the verdict.
- Candidate C, checked on existing test data (v5): AKL older adults 0.860 vs general 0.856. No group gap; stays
  rejected.

## 14:10 Candidate D (one-shot low-bit weight quantization): rules before any result

Theory: quantization is in-basin, so by v5's law LCI should track the output KL. The strongest backprop-free
quantizer, GPTQ (the OBQ / OBC analogue; never run in this repo), should then keep the explanation wherever it keeps
the function.

Models (`quant:b`, b ∈ {3, 2}; 27.2 / 20.7 KB with fp32 scales, low-bit diagnostic sizes):
- `rtn`: per-channel round-to-nearest;
- `gptq`: GPTQ, per-channel grid, calibration = the 978 windows, head RTN.

Rules:
- **Fail** if `gptq` at 3 bits is ≥ 0.85 (as good as the strongest 80 % pruning) and the 2-bit loss is an accuracy
  collapse (> 2 pt).
- Otherwise, if `gptq` is ≥ 0.05 below the ceiling with accuracy within 2 pt, measure the same-size maximum (QAT +
  fmkd) before calling it a gap.

## 14:10 Candidate D readings (folds 0 / 2 / 4)

| Model | Fold 0 | Fold 2 | Fold 4 | Accuracy Δ (f0 / f2 / f4) |
|---|---|---|---|---|
| rtn 3-bit | 0.863 | 0.894 | 0.869 | +1.4 / −0.8 / −0.2 pt |
| **gptq 3-bit** | **0.940** | **0.907** | **0.940** | +0.5 / −0.9 / −0.2 pt |
| rtn 2-bit | 0.596 | 0.634 | 0.676 | −29 / −11 / −15 pt |
| **gptq 2-bit** | **0.860** | **0.799** | **0.837** | −3.3 / −2.2 / −2.4 pt |
| ceiling | 0.945 | 0.915 | 0.962 | |

- 3 bits: GPTQ reaches the ceiling, so 3 bits is closed (theory confirmed: an in-basin method that minimises
  layer-output error keeps the explanation).
- 2 bits (ternary, 20.7 KB): GPTQ keeps 0.832 mean against a 0.941 ceiling (0.109 below), at −2.6 pt accuracy. It
  is the first candidate today with a large residual after the strongest one-shot method.

**Rules for 2 bits, written before the next runs:**
- y = the best existing backprop-free model:
  - `gptq` (20.7 KB);
  - `gptq_p8`: GPTQ with conv0, attn_score and the head at 8 bits, the standard first / last-layer protection
    (DoReFa, PACT) and the analogue of v5's allocation lever that closed the pruning gap; +0.47 KB (≈ 2 %), disclosed.
- x = the same-size maximum: `qat_fmkd` / `qat_fmkd_p8` (STE QAT + fmkd from the GPTQ start, 20 epochs, lr 1e-4;
  training, outside the one-shot scenario).
- **Gap** if x − y ≥ 0.05, positive on every fold run (≥ 3), x's accuracy within 2 pt of the original's.
- It is scoped like v4: backprop-free one-shot 2-bit quantization. A full KL-chosen mixed-precision search at equal
  size needs sub-2-bit (pruned) tensors and is not run (time); it is listed as the open closer to check first.

## 14:28 Candidate D at 2 bits: no achievable model above the protected one-shot

| 2-bit model | Fold 0 | Fold 2 | Fold 4 | Mean | Accuracy Δ (f0 / f2 / f4) |
|---|---|---|---|---|---|
| gptq (20.7 KB) | 0.860 | 0.799 | 0.837 | 0.832 | −3.3 / −2.2 / −2.4 pt |
| **gptq_p8** (21.2 KB) | **0.902** | **0.856** | 0.832 | **0.863** | −1.3 / −0.3 / −1.1 pt |
| qat_fmkd_p8 (same size, training) | 0.866 | 0.774 | **0.872** | 0.837 | +0.3 / −0.2 / −0.4 pt |
| ceiling | 0.945 | 0.915 | 0.945 | 0.935 | |

- x − y = −0.026: the training maximum is **below** the one-shot baseline on 2/3 folds. Rule 2 fails.
- Keeping the three tiny tensors at 8 bits (+2 % size) does at 2 bits what the layer budget did for 80 % pruning
  (+0.031).
- Residual of the strongest model to the ceiling: 0.072, the same size as v5's 80 % / 90 % residuals (0.068 / 0.081).
- D is rejected. `qat_fmkd` without protection (folds 0 / 2 / 4) keeps running for the table, but cannot change the
  verdict, because y = gptq_p8.

## 14:32 Last data check: one-shot 90 % (existing v5 data, no run)

- v5's strongest one-shot model, AKL90 (KL-chosen allocation), scores 0.771; fmkd at the same masks scores 0.851.
- Per-fold differences: +0.129 / +0.185 / +0.074 / +0.019 / −0.006. Not positive on every fold.
- The one-shot model loses 2.2 pt accuracy (fold 1: −4.4 pt), so part of the gap is accuracy.
- Even the LCI-best allocation (an oracle choice) gives only 0.781, so the allocation lever cannot fix it.
- Rejected at the data stage: inconsistent across folds and not explanation-specific.

## 14:35 Decision

**No candidate passes theory, data and the strongest-baseline test today.** Experiments stop here so the deliverables
are done by 16:00.

| Candidate | Best existing | Best achievable measured | Ceiling | Why it fails |
|---|---|---|---|---|
| A, W12 dense student | 0.806 (slice_fmkd, 2 folds) | 0.823 (capfit oracle) | 0.930 | capacity limit of the dense narrow model |
| A, W20 dense student | 0.885 (slice_fmkd, folds 0 / 2 / 4); from scratch 0.880 (folds 0 / 2) | — | 0.934 | no gap; output matching suffices |
| B, 64 unlabelled windows | 0.949 (fold 0) | — | 0.945 | at the ceiling; at 16 windows accuracy collapses |
| C, older adults (test) | 0.860 older vs 0.856 general | — | 0.915 | no group gap |
| D, 3-bit | 0.929 (gptq, 5 folds) | — | 0.932 | at the ceiling |
| D, 2-bit | 0.863 (gptq_p8, 3 folds) | 0.837 (QAT + fmkd) | 0.935 | training does not beat one-shot |
| 90 % one shot | 0.771 (AKL90) | 0.851 (fmkd, same masks) | 0.932 | not on every fold; −2.2 pt accuracy |

What the day adds to v5's law:
- In-basin compressors (pruning, quantization) keep the explanation exactly as well as they keep the function.
- Out-of-basin students break the KL → LCI link only when they are capacity-limited, and then no objective
  recovers it.
- At ≈ 20–28 KB every route (80 % / 90 % pruning, 2-bit quantization, a 50 % dense student) stops 0.06–0.08 below
  the ceiling, and no oracle or training maximum exceeds the strongest existing one-shot model by ≥ 0.05.

## 14:30 One more check while writing: is the 2-bit maximum higher with another QAT lr?

`qat_fmkd` without protection was stopped (it cannot change y). The oracle's lr is varied on validation (an
upper-bound probe, disclosed, as in v5): `qat_fmkd_p8` at lr 3e-5 and 3e-4, 20 epochs, folds 0 / 2 / 4.
- Rule: D 2-bit is reopened only if the best lr beats gptq_p8 by ≥ 0.05 on the mean and on every fold.

## 14:55 QAT lr check and folds 1 / 3

| 2-bit model | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean (0 / 2 / 4) | Mean (5 folds) |
|---|---|---|---|---|---|---|---|
| gptq_p8 | 0.902 | 0.732 | 0.856 | 0.879 | 0.832 | 0.863 | 0.840 |
| qat_fmkd_p8 lr 3e-5 | 0.875 | 0.734 | 0.877 | 0.818 | 0.831 | 0.861 | 0.827 |
| qat_fmkd_p8 lr 1e-4 | 0.866 | | 0.774 | | 0.872 | 0.837 | |
| qat_fmkd_p8 lr 3e-4 | 0.876 | | 0.843 | | 0.802 | 0.840 | |
| per-fold best QAT (oracle choice) | 0.876 | | 0.877 | | 0.872 | 0.875 | |

- The best lr (3e-5) is the same as the one-shot model (−0.002). Even the per-fold best over the three lrs is only
  +0.012 above it. D 2-bit is not reopened.
- Folds 1 / 3 (15:04): QAT lr 3e-5 gives 0.734 / 0.818, against gptq_p8 0.732 / 0.879. On 5 folds, QAT gives 0.827
  and gptq_p8 0.840. Fold 1 stays at 0.73 for both (ceiling 0.914): v5's fold-1 representability limit. D is final.

## 15:05 Experiments end; deliverables. Final: no new gap passes (table at 14:35, D 2-bit numbers as above).
