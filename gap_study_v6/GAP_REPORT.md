# Gap report v6: searching for a new explanation gap after compression

2026-11-28, Ansel. **No new method.** Goal: a new gap, not a re-validation of v3–v5. Every model is an existing
technique or an oracle labelled as such. Every choice uses HARTH inner validation; nothing is read on test.

Protocol (unchanged from v3–v5):
- frozen TinyHAR (54,273 parameters), 5 subject folds, 50 explain windows per inner-val subject;
- GradientExplainer (nsamples 200) of the original's predicted class on 48 features;
- `lci_indep`: model SHAP seed 1 vs the original's seed 0;
- ceiling = the original's own SHAP-noise LCI: 0.932 (per fold 0.914–0.962).

Run log with every rule written before its result: `gap_study_v6/RUNS.md`.

**Result:** no new fixable gap. Four candidates and one data-only check failed. Each failed because an existing
method already closes it, or because nothing measured at that size reaches higher. Section 6 has the decision.

## 1. Theory: where can a real gap still exist?

**What v3–v5 established.**
- Within a fold, LCI tracks the label-free output KL on calibration windows (Spearman −0.57 at 80 %, −0.80 at 90 %),
  not accuracy.
- Each earlier gap closed once a baseline minimised output divergence well enough: OBC masks, KL-chosen layer
  budgets, fmkd.

So a real gap needs one of two conditions:
- **(a) the output does not pin what the explainer reads.** GradientExplainer gives φ_i = (x_i − b_i) E[∂f/∂z_i]
  along z = b + a (x − b). Matching outputs on such paths (fmkd's mixup blends) fixes only the sum Σφ_i
  (completeness), not the split over the 48 features. In-place compression inherits the split through the shared
  weights. A model that leaves the original's weight basin does not.
- **(b) the output-fidelity tools cannot reach the function at that size or data budget.**

Candidates, ranked before any run:

| # | Candidate | Mechanism | Prior | Outcome |
|---|---|---|---|---|
| A | dense narrow student (structured width reduction / distillation; the MCU form, no sparse kernels) | (a): leaves the basin; retrain-seed floor τ ≈ 0.63 shows the data does not fix the split | high | **rejected** (section 2) |
| B | one-shot with very few unlabelled calibration windows | (b): fusion refit rank-deficient below ≈ 21 windows; noisy KL choice | low | **rejected** (section 3) |
| C | calibration / deployment shift, older adults | (b) off-distribution | rejected by existing data | **rejected** (section 3) |
| D | one-shot low-bit weight quantization (GPTQ never run here) | in-basin, so it should track KL; open below 3 bits | medium-low | **rejected** (section 4) |
| E | deterministic explainers (occlusion) | metric change | — | not run (metric shopping) |
| F | other architectures | same in-basin argument as TinyHAR | — | not run (time) |

## 2. Candidate A: compression that leaves the weight basin

Students are TinyHAR with fewer filters and the original's input normalisation:
- width 12: 10,401 parameters, the parameter count of 80 % sparsity, dense;
- width 20: 28,049 parameters, ≈ 50 %.

All runs are seed 0, 30 epochs. Code: `run_v6.py`, stage `stu`. Structured slicing of the original
(`slice_tinyhar`: L1 filter / row norms, every dependent tensor sliced) reproduces the original exactly at width 28
(checked).

| Model (validation `lci_indep`) | W12 fold 0 | W12 fold 2 | W20 fold 0 | W20 fold 2 | W20 fold 4 |
|---|---|---|---|---|---|
| fmkd from scratch (Beyer 2022 function matching + KD) | 0.823 | 0.749 | **0.935** | 0.825 | — |
| slice_fmkd (Li et al. 2017 slicing + fmkd) | **0.888** | 0.725 | 0.920 | **0.849** | 0.868 |
| capfit oracle from scratch (gradient field of the evaluated windows) | 0.873 | **0.773** | — | — | — |
| slice_capfit oracle | 0.873 | 0.743 | — | — | — |
| ceiling | 0.945 | 0.915 | 0.945 | 0.915 | 0.962 |

Accuracy stays within 0.7 pt of the original everywhere except W20 fmkd fold 0 (−2.0 pt).

- **The theory's signature is there.** At width 12, fmkd students explain worse than v5's in-place models at the
  same output KL:
  - fold 0: KL 0.019 gives 0.823, against 0.84–0.92 for in-place models;
  - fold 2: KL 0.016 gives 0.749, against 0.79–0.89;
  - occlusion LCI ≈ 0.60 against ≈ 0.80.

  In-place models' LCI on log KL predicts 0.851 for them, a residual of −0.065 (`kl-decoupling.csv`).
- **But it is not fixable at width 12.** Fitting the evaluated windows' gradient field directly (capfit, from
  scratch or from the sliced start) adds +0.017 on average over the best baseline. The dense width-12 architecture
  cannot represent the explanation: a capacity limit, not a missing objective.
- **At width 20 there is no gap.** fmkd from scratch is within 0.024 of the sliced model (above it on fold 0). The
  sliced model is at 0.92 / 0.85 / 0.87, which is 0.03–0.09 below the ceiling and in v5's in-place range. With
  enough capacity, output matching does pin the split.
- Verdict: rule 2 fails at both widths. A is rejected.

## 3. Candidates B and C

**B, scarce unlabelled calibration** (fold 0, seed 0):

| N random unlabelled windows | akl (OBC + refit + KL-chosen allocation) | akl_mix | obcu | F1 akl (original 0.818) |
|---|---|---|---|---|
| 16 | 0.700 | 0.752 | 0.722 | 0.629 |
| 64 | **0.949** | 0.885 | — | 0.754 |
| 978 balanced (v5) | 0.925 | | | |

- Above the predicted rank threshold, 64 random unlabelled windows already reach the ceiling.
- Below it, F1 collapses (−13 to −19 pt): an accuracy failure, not an explanation gap.

**C, deployment population** (existing v5 test, no new run): the strongest one-shot model AKL scores 0.860 on older
adults and 0.856 on general adults. The older group is not worse. Rejected.

## 4. Candidate D: one-shot low-bit weight quantization

Per-channel symmetric weight-only quantization of every weight tensor; sizes as in the low-bit diagnostic (fp32
scales and biases): 3-bit 27.2 KB, 2-bit (ternary) 20.7 KB. Code: `quant_v6.py`, stages `quant` and `qat`.
- `gptq`: GPTQ (Frantar et al. 2023), the quantization form of OBQ / OBC, never run in this repo. Per-row grid, the
  978 calibration windows, units in forward order, head RTN.
- `gptq_p8`: conv0 (140 weights), attn_score (56) and the head (448) kept at 8 bits (+0.47 KB, ≈ 2 %). This is the
  standard first / last-layer protection (DoReFa, PACT) and the analogue of v5's allocation lever.

| Model (validation `lci_indep`) | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean | Accuracy Δ |
|---|---|---|---|---|---|---|---|
| rtn 3-bit | 0.863 | 0.711 | 0.894 | 0.920 | 0.869 | 0.851 | −2.0 pt |
| **gptq 3-bit** | 0.940 | 0.875 | 0.907 | 0.965 | 0.940 | **0.926** | −0.2 pt |
| rtn 2-bit | 0.596 | 0.431 | 0.634 | 0.663 | 0.676 | 0.600 | −24.9 pt |
| gptq 2-bit | 0.860 | 0.592 | 0.799 | 0.800 | 0.837 | 0.778 | −2.2 pt |
| **gptq_p8 2-bit** | 0.902 | 0.732 | 0.856 | 0.879 | 0.832 | **0.840** (folds 0 / 2 / 4: 0.863) | −0.8 pt |
| qat_fmkd_p8 2-bit (STE QAT + fmkd, 20 ep, lr 1e-4; same-size maximum) | 0.866 | — | 0.774 | — | 0.872 | 0.837 | −0.1 pt |
| qat_fmkd_p8, lr 3e-5 (best lr on folds 0 / 2 / 4) | 0.875 | 0.734 | 0.877 | 0.818 | 0.831 | 0.827 (0 / 2 / 4: 0.861) | −0.2 pt |
| qat_fmkd_p8, lr 3e-4 | 0.876 | — | 0.843 | — | 0.802 | 0.840 | −0.3 pt |
| ceiling | 0.945 | 0.914 | 0.915 | 0.940 | 0.945 | 0.932 | |

- **3 bits is closed by GPTQ** (0.926 vs ceiling 0.932; RTN 0.851, so GPTQ − RTN = +0.074 [0.004, 0.144], 5/5 folds).
  The theory held: an in-basin method that minimises the layer-output error keeps the explanation.
- **2 bits.** GPTQ − RTN = +0.178, 5/5 folds. With the 8-bit protection, GPTQ reaches 0.840 on 5 folds (0.863 on folds 0 / 2 / 4) at
  −0.8 pt accuracy, 0.092 below the ceiling. Quantization-aware training with the strongest training recipe (fmkd) does **not** go
  higher at any of three lrs (folds 0 / 2 / 4: 0.861 / 0.837 / 0.840; 5 folds at lr 3e-5: 0.827 vs 0.840). Fold 1
  stays at 0.73 for both (ceiling 0.914), the representability limit v5 found on that fold. The rule
  (x − y ≥ 0.05) fails.
- Per byte, quantization dominates pruning: 3-bit GPTQ (27.2 KB) keeps 0.926, while 90 % pruning (26.6 KB
  + 6.5 KB bitmask) keeps 0.771 one-shot and 0.851 with fmkd.

**One-shot 90 % pruning** (existing v5 data, no run):
- AKL90 scores 0.771; fmkd at its masks scores 0.851.
- Per-fold differences: +0.129 / +0.185 / +0.074 / +0.019 / −0.006, so the gap is not positive on every fold.
- The one-shot model loses 2.2 pt accuracy (fold 1: −4.4 pt).
- Even the LCI-best allocation is 0.781.
- Rejected at the data stage.

## 5. How high can LCI go (oracles / same-size maxima)

| Setting (size) | Strongest existing | Highest measured at the same size | Ceiling |
|---|---|---|---|
| dense width 12 (10.4 k parameters) | 0.806 (slice_fmkd) | 0.823 (capfit oracle) | 0.930 |
| dense width 20 (28.0 k) | 0.885 (slice_fmkd, folds 0 / 2 / 4) | — (fmkd from scratch 0.880) | 0.934 |
| 3-bit (27.2 KB) | 0.926 (gptq) | — | 0.932 |
| 2-bit (20.7–21.2 KB) | 0.840 (gptq_p8, 5 folds) | 0.827 (QAT + fmkd, best lr, 5 folds); per-fold best of 3 lrs on folds 0 / 2 / 4: 0.875 vs 0.863 | 0.932 |
| 80 % sparse (v5) | 0.864 (AKL) | 0.875 (capfit) | 0.932 |
| 90 % sparse (v5) | 0.851 (fmkd from AKL) | 0.798 (capfit) | 0.932 |

In every setting at ≈ 20–28 KB the strongest existing compressor stops 0.06–0.08 below the ceiling. No oracle or
training maximum measured in v5 or v6 exceeds it by ≥ 0.05. The only bound above is the ceiling itself.

## 6. Decision

**No new fixable gap passes the bar today** (theory → data → strongest baselines → maximum, rules in `RUNS.md`).
- A (leave the basin): the theory's signature is real (LCI decoupled from output KL), but it only appears when
  the dense student is capacity-limited, and then no objective lifts it (oracle +0.017).
- B (scarce calibration): no gap above ≈ 21 windows; below it, an accuracy collapse.
- C (older adults): no group gap.
- D (low-bit quantization): GPTQ closes 3 bits; at 2 bits, protected GPTQ (0.863) is above the training maximum.
- 90 % one shot: inconsistent across folds, partly accuracy.

**The closest candidate** is one-shot 2-bit quantization: best existing 0.840 (gptq_p8, 5 folds), ceiling 0.932,
best measured maximum 0.827 (QAT + fmkd; per-fold oracle choice on 3 folds +0.012). Not promising: nothing shows
the 0.09 residual is reachable at that size.

**What the day adds to v5's law.**
- In-basin compression (pruning or quantization) keeps the explanation exactly as well as it keeps the function.
- A model outside the basin breaks that link only when it lacks capacity.
- So at these sizes the remaining explanation loss is a capacity residual (≈ 0.07), not a missing objective.
- A future method would have to beat 0.86–0.93 at the same size. The data so far gives no evidence that this is
  possible.

## 7. Scope and limits

- One architecture (TinyHAR, 54 k parameters), one training dataset (HARTH), inner val of 3 subjects per fold; seed 0.
- Several candidates ran on 2–3 folds because of time (see each table). The rules required ≥ 3 folds before a gap
  counts.
- The GPTQ (Frantar et al. 2023), OBQ, DoReFa (Zhou et al. 2016), PACT (Choi et al. 2018), Li et al. 2017, Beyer et al.
  2022, Srinivas & Fleuret 2018 PDFs are not in `papers/`. Verify the cited sections before submission.

## 8. Files

- Code: `gap_study_v6/`
  - `run_v6.py`: stages `stu`, `scarce`, `quant`, `qat`;
  - `quant_v6.py`: RTN, GPTQ, 8-bit protection, QAT + fmkd;
  - `analyze_v6.py`;
  - `test_v6.py` (3 checks, pass).
- Numbers: `results/gap-study-v6/validation/` (summarised in `results/README.md`).
