# Gap report v5: is the one-shot 80 % explanation gap real, and where is the gap?

2026-11-27 → 28, Ansel. **No new method.** Every model is an existing technique, a combination of existing
pieces, or an oracle that is labelled as such. Selection uses HARTH inner validation only. The held-out test (22 HARTH
test subjects + 18 HAR70+ older adults) was run once for a candidate set fixed beforehand. Run log and the rules
written before each read: `gap_study_v5/RUNS.md`.

Protocol (unchanged from v3 / v4):
- Frozen TinyHAR (54,273 parameters), 5 subject folds.
- SHAP GradientExplainer of the original's predicted class, 48 features.
- `lci_indep`: model SHAP seed 1 vs the original's seed 0.
- Ceiling u = the original's own SHAP-noise LCI: 0.932 on inner val, 0.917 on test.

**Result.**
- **v4's gap (one-shot 80 %) does not survive two existing ingredients it had not used:**
  - Hessian-aware masks (OBC + full refit): 0.817, vs v4's best 0.773;
  - output-driven layer budgets: KL-chosen allocation, label-free, backprop-free: **0.864** (3 calibration seeds), equal to full-data fmkd (0.865).
- On held-out test it is **0.858** vs v4's 0.767.
- **v3's second gap (90 % + KD)** closes the same way: fmkd from the KL-chosen allocation reaches 0.851 (test
  0.832) vs 0.781 (test 0.767).
- What remains is a residual of 0.06–0.08 below the uncompressed ceiling (0.932 / 0.917). No oracle exceeded the
  strongest existing combination by ≥ 0.05 (capacity oracle 80 %: 0.875).
- **No fixable between-method gap passes the bar.**

## 1. Theory check of the v4 gap (before any run)

v4's claim: one-shot pruning without retraining at 80 %. Best baseline 0.773 (blended calibration), the same masks
reach 0.865 (fmkd), ceiling 0.932. From the documents alone, three weak points:
1. **Baseline coverage.** v3 listed OBC (exact OBS, Frantar et al. 2022) and DSnoT (Zhang et al. 2024) as "not run".
   OBC is the backprop-free method that accounts for how removals interact (input Hessian). That is v3's own
   requirement 2 for a fix. SparseGPT was dismissed as "collapsing", but it was never given the full refit that every
   other baseline gets. No baseline changed the per-layer budget except LAMP.
2. **Upper bound.** x = fmkd at magnitude masks is neither the highest LCI possible at 80 % nor the reference for a
   baseline with other masks.
3. **Other conditions** (sensor shift, other sparsities, layer allocation) had not been checked for a larger gap.

## 2. The v4 gap does not survive the missing existing baselines

Validation, 80 %, 5 folds (seeds 0–2 where stated); accuracy Δ vs the original (0.927):

| Method (existing) | LCI (indep) | Gap to u | Accuracy Δ | Occlusion LCI |
|---|---|---|---|---|
| v4 best: l1_fullrecon_mixcal (3 seeds) | 0.773 | 0.159 | −1.1 pt | 0.648 |
| **OBC + full refit** (obc_fullrecon, 3 seeds) | **0.817** | 0.115 | −0.2 pt | 0.714 |
| OBC + full refit, seed 0 only | 0.831 | 0.101 | −0.3 pt | — |
| OBC + full refit + 5× calibration (seeds 0–2) | 0.839 | 0.093 | −0.1 pt | 0.719 | | | |
| OBC + full refit + blended calibration (seeds 0–2) | 0.830 | 0.102 | −0.2 pt | 0.750 | | | |
| OBC, dampening 0.1 / 0.001 | 0.830 / 0.830 | | −0.3 / −0.5 pt | |
| OBC, 5 nested steps + refit | 0.817 | | −0.3 pt | |
| **OBC + refit, layer budget chosen by output KL** (label-free; SPDY / OWL-style criterion over 7 group allocations; 3 seeds, seed SD 0.013) | **0.864** | 0.068 | −0.5 pt | 0.814 |
| the same + fmkd 20 ep | 0.864 | 0.068 | −0.2 pt | 0.796 |
| l1 + refit, K = 20 steps | 0.790 | | −0.8 pt | 0.700 |
| SparseGPT masks + full refit | 0.723 | | −2.0 pt | 0.560 |
| DSnoT (from Wanda) + full refit | 0.742 | | −3.4 pt | 0.565 |
| BatchNorm re-estimation after l1_fullrecon | 0.651 | | −7.9 pt | 0.404 |
| plain OBC (own OBS weights, no refit) | 0.748 | | −14.2 pt | 0.531 |
| *training references:* fmkd at l1's masks (v4, 40 ep) | 0.865 | 0.067 | −0.1 pt | 0.758 |
| fmkd at OBC's masks (40 ep) | 0.864 | 0.068 | −0.1 pt | 0.752 |
| label-free calfm (v4) | 0.835 | 0.097 | −0.3 pt | 0.692 |

- **OBC closes 63 % of v4's x − y on seed 0.** With the output-KL allocation it reaches 0.864 (vs mixcal +0.091, CI [0.048, 0.134], 5/5 folds), which equals full-data
  fmkd. By v4's rule 4, v4's gap is dropped: its baseline was not the strongest existing one-shot method.
- **Held-out test** (40 subjects, ceiling 0.917):

  | Model | LCI | Subjects with gap ≥ 0.10 | Accuracy |
  |---|---|---|---|
  | OBC | 0.834 | 17 / 40 | 0.836 |
  | FMKD | 0.837 | 16 / 40 | 0.848 |
  | MIXCAL (v4's y) | 0.767 | 33 / 40 | 0.827 |
  | L1FR (v3's y) | 0.762 | 32 / 40 | 0.820 |
  | original | — | — | 0.843 |

  FMKD − MIXCAL = +0.069 (36/40 subjects, p < 1e-4), so v4's difference is real against its own baseline. But
  OBC − MIXCAL = +0.067 (39/40), and FMKD − OBC = +0.003 (p = 0.26): the existing OBC closes it on unseen subjects
  too. 
- The strongest one-shot combination on test, **AKL (KL-chosen allocation)**, reaches **0.858** (13/40 subjects
  with a gap ≥ 0.10). It beats FMKD by +0.021 (31/40, p = 0.002) and MIXCAL by +0.091 (39/40). It is not above the
  capacity oracle: CAP 0.851, AKL − CAP = +0.007. Its cost is accuracy on older adults: 0.740 vs the original's 0.763
  (FMKD 0.775).

## 3. Where the problem lives (diagnostics for method design; validation only, no tuning on test)

- **Unit-only pruning** (one unit at 80 %, the rest dense, full refit), LCI cost vs the ceiling:
  - LSTM 0.068 (48 % of the weights), conv0 0.046 (only 140 weights), fusion 0.027, conv6 0.025, conv9 0.017;
  - attention / feed-forward ≤ 0.007.

  The full model loses 0.17 (l1) / 0.115 (OBC), so the costs interact (super-additive).
- **Layer budget is a lever.** Same 80 % total, OBC + refit:
  - protecting the convs (a2) gives 0.889 / 0.892 / 0.900 on folds 0 / 2 / 4, vs uniform 0.846 / 0.856 / 0.835;
  - protecting the LSTM (a1, a6) is *worse*.

  The output-KL criterion picks a good allocation on every fold, so this lever is already available to existing
  methods.
- **Calibration distribution is not the problem.** Calibrating on the explainer's own probe points (blends with the
  SHAP background and occluded windows) gives 0.756, the same as plain calibration.
- **Same-mask weight swaps** (one unit's weights from fmkd into l1_fullrecon and back, BatchNorm swapped with its
  conv): conv swaps destroy accuracy, and single attention / fusion swaps move LCI by < 0.03. The training gain is
  joint and co-adapted, not located in one unit.
- **Per class** (folds 0 / 2 / 4): walking stays the weakest for every model (best 0.78 vs ceiling 0.82). The
  allocation gains are largest on stairs and standing.
- **Shift probe** (existing checkpoints, no training):
  - sensor noise σ 0.05 g raises every gap by ≈ 0.02–0.03 (80 % l1: 0.198 vs clean 0.172; fmkd 0.089 vs 0.067);
  - ±15° sensor rotation leaves the gaps as clean;
  - no condition opens a larger gap where clean explanations are intact.
- **Fold dependence.** Fold 1 is low for every method, from 70 % on (fmkd 0.78, OBC 0.705, capfit 0.760; ceiling
  0.914). Its retrain-seed floor is also the lowest (0.603 vs 0.72–0.73): where the original's explanation is weakly
  determined by the data, compression moves it most.

## 4. How high can LCI go at a given size (oracles)

| Oracle (not a method) | What it uses | 80 % | 90 % |
|---|---|---|---|
| jfm: fmkd + per-point gradient-field matching on training-window SHAP path points | training windows | 0.783 (2 folds; *below* fmkd) | — |
| capfit: fmkd + gradient-field matching on the **evaluated** windows' path points (λ 10, 20 ep) | 150 inner-val windows | 0.875 (5 folds) | 0.798 (5 folds; ≤ the strongest existing 90 % combination on 4/5 folds) |
| capS: as capfit, λ 100, 40 ep, from the KL allocation's masks | same | 0.800 on fold 1 (fmo 0.782, ceiling 0.914): a hard fit to the evaluated windows stays at ≈ 0.80 | — |

- On test, CAP generalises (it never saw a test subject): 0.851, above FMKD (+0.014, p = 0.011) but not above AKL.
- **Reading.** At 80 % every route lands at 0.864–0.875 on validation and 0.85–0.86 on test, 0.06–0.07 below the
  ceiling, and no oracle exceeds the strongest existing combination by ≥ 0.05. Fold 1 sets the floor: even a hard
  fit to the evaluated windows stays at 0.80 there (ceiling 0.914). That residual is a representability limit of the
  80 % model, not a missing objective.

### 90 % (≈ 26.6 KB fp32: MCU size)

| Model | Validation LCI | Accuracy Δ | Test LCI (40 subjects) | Test accuracy |
|---|---|---|---|---|
| uniform OBC + refit (one shot) | 0.700 | −6.3 pt | — | — |
| KL-chosen allocation, OBC + refit (one shot) | 0.771 | −2.2 pt | 0.777 | 0.812 |
| v3's tuned lamp_recon_kd (lr 3e-4) | 0.781 | −0.9 pt | 0.767 | 0.839 |
| v4's fmkd from LAMP (4 folds) | 0.787 | −0.9 pt | — | — |
| **fmkd 20 ep from the KL-chosen allocation** | **0.851** | −0.1 pt | **0.832** | 0.836 |
| capfit 90 % (oracle) | 0.798 | −0.4 pt | — | — |
| original | 0.932 (ceiling) | — | 0.917 (ceiling) | 0.843 |

- fmkd from the KL-chosen allocation − lamp_recon_kd = +0.070 on validation (4/5 folds) and **+0.065 on test**
  (36/40 subjects, p < 1e-4), at −0.3 pt accuracy. **v3's secondary gap (90 % + KD) closes with existing pieces.**
- The capacity oracle (from LAMP's masks) is below it on 4/5 folds: no evidence of headroom above 0.851 at 90 %.

## 5. Final gap

Decided by the rules written before each read (RUNS.md, decision points 4 and the 01:50 90 % rules):
1. **No between-method gap passes.**
   - At 80 %, one shot: y = KL-chosen allocation 0.864 (3 seeds) vs x = capacity oracle 0.875 (+0.011, CI [−0.011, 0.033]) and fmkd 0.865; fmkd on top of the allocation adds nothing (0.864).
   - At 90 %: y = fmkd from the KL allocation 0.851 vs capacity oracle 0.798.
   - Both earlier gaps came from baselines that used uniform per-layer budgets (or LAMP's) and magnitude masks.
2. **Residual explanation gap of the strongest existing compressors** (the only gap left):

   | | Validation | Test |
   |---|---|---|
   | 80 % one shot | 0.864 vs 0.932 (0.068) | 0.858 vs 0.917 (0.059; 13/40 subjects ≥ 0.10) |
   | 90 % | 0.851 vs 0.932 (0.081) | 0.832 vs 0.917 (0.085; 16/40) |

   It concentrates on fold 1, on walking / standing, and on older adults' accuracy for the one-shot model.
3. **Best achievable at the same size.** Measured: 0.875 (80 %, capacity oracle; per-fold best allocations also
   0.875). In theory: only the 0.932 ceiling bounds it. Nothing measured tonight shows that a model of this size can
   go above ≈ 0.875 on these folds. Fold 1 caps at ≈ 0.80 even when the evaluated windows are fitted directly.
4. **Why (theory, backed by data).** Within a fold, LCI tracks the label-free output KL on the calibration windows
   (Spearman −0.57 at 80 %, −0.80 at 90 %), and not accuracy (−0.06 at 80 %). Explanation fidelity is a property of
   function fidelity; accuracy alone is not. Methods that minimise output divergence with good masks and budgets
   therefore keep explanations without any explanation term, and the explanation-supervised oracles (jfm, jfm2,
   capfit) add ≤ 0.01.

   **What a new method would need:** beat 0.864 (80 %) / 0.851 (90 %) at the same size and accuracy, i.e. close part
   of a ≤ 0.07 residual that, on the hardest fold, looks like a capacity limit.

## 6. Scope and limits

- One architecture (TinyHAR, 54 k parameters), one training dataset (HARTH), inner val of 3 subjects per fold.
- Seed 0 for most references.
- The allocation grid is coarse (7 group allocations), so a continuous search could move both the KL choice and the
  LCI-best choice.
- Oracles fit validation windows and are reported as bounds only.
- Test: one run for a fixed candidate set.
- The OBC (Frantar et al. 2022), DSnoT (Zhang et al. 2024), SPDY (Frantar & Alistarh 2022), OWL (Yin et al. 2024)
  and EagleEye (Li et al. 2020) PDFs are not in `papers/`. Verify the cited sections before submission.
- The "KL-chosen allocation" is an SPDY / OWL-*style* criterion (lowest output KL over a fixed grid), not either
  paper's exact search.

## 7. Files

- Code: `gap_study_v5/`
  - `baselines_v5.py`: OBC, DSnoT, SparseGPT + refit, BatchNorm re-estimation, K20, allocations;
  - `run_v5.py`: all validation stages;
  - `shift_probe.py`, `analyze_v5.py`, `analyze_test_v5.py`, `diagnose_v5.py`.
- Numbers: `results/gap-study-v5/` (summarised in `results/README.md`).
