# Gap summary: explanations break under one-shot pruning while accuracy holds

Final decision, 2026-11-27. It draws on every study in this repository (9A grid, XGP rounds 1–4, low-bit / bit-reduction
tests, PAIR, GMR, gap studies v3 and v4). **No new method is proposed here.** All numbers are HARTH inner validation
(5 subject folds, 3 subjects × 50 explain windows per fold) unless marked as test.

## The gap

> **One-shot post-training pruning without retraining (≈ 1k unlabeled calibration windows, no gradient training) at
> 80 % unstructured sparsity of TinyHAR:** the best existing methods score at most **77.3 % LCI** (l1_fullrecon_mixcal;
> l1_fullrecon 76.0 %), against **93.2 %** for the uncompressed model and **86.5 %** that the *same masks* can reach.
> Accuracy stays at **91.5 %** (original 92.7 %, −1.1 pt). Gradient-free occlusion LCI drops in the same way (**64.8 %**,
> vs 72.6 % at the same masks).

- **Gap to the uncompressed model:** 0.159 LCI (17 %). The relative LCI drop is about 14× the relative accuracy drop.
- **Gap to what is achievable at the same masks:** 0.092 LCI, which is the part we target.

Definitions:
- LCI = 0.5 · per-window Pearson + 0.5 · Jaccard of each subject's global top-10 features.
- Explanations: SHAP GradientExplainer of the original model's predicted class, on 48 features (6 axes × 8 time segments).
- Compressed models are scored with SHAP seed 1 against the original's seed 0 (`lci_indep`), so the ceiling 0.932 is the
  original's own SHAP-noise level.

## Proof

| Evidence | Number | File |
|---|---|---|
| Ceiling (original, SHAP seed 0 vs 1) | 0.932 (0.914–0.946 per fold) | `results/gap-study-v4/validation/...-summary-mean-over-folds-seeds.csv` |
| Best one-shot baselines at 80 % (5 folds × 3 calibration seeds) | mixcal 0.773, l1_fullrecon 0.760, wanda_fullrecon 0.760, cal5x 0.761, ria_fullrecon 0.752 | same |
| Same masks, explanation reachable (function-matching KD, 40 ep) | 0.865, accuracy 0.926 | same |
| Same-mask reference − best baseline (mixcal) | +0.092, fold-paired 95 % CI [0.027, 0.157], 5/5 folds positive; occlusion +0.110 [0.053, 0.166] | `...-paired-gaps.csv` |
| Same-mask reference − l1_fullrecon | +0.105 [0.043, 0.168], 5/5 folds ≥ 0.05; occlusion +0.119 | same |
| Gap to the ceiling vs noise | 0.172 for l1_fullrecon, CI [0.069, 0.276]; ≈ 10× the seed SD (0.017–0.020) and the SHAP-noise SD (0.016) | v3 `phase2/...-gate-per-cell.csv` |
| Held-out test, one shot, nothing selected on test (l1_fullrecon) | HARTH gap 0.144 (15/22 subjects ≥ 0.10); HAR70+ older adults 0.117 (12/18), FA-LCI 0.775 vs 0.915 | v3 `test-confirmation/...csv` |
| Third explainer (gradient × input, seed 0) | l1_fullrecon 0.650 vs same-mask fmkd 0.704 (4/5 folds higher) | `...-gradient-x-input-lci-per-fold.csv` |
| Artefacts ruled out | explainer self-consistency on the pruned model 0.933 (= original); top-1 agreement 0.92, and LCI on agreeing windows is still 0.778; no attribution flattening; sign flips 1.7 % | v3 `phase3/` |

## Where the best baselines fail

1. **They match outputs, not the behaviour the explanation reads.** Layer-wise reconstruction makes the logits match on
   the calibration windows (≈ 3 % error). SHAP and occlusion, however, query the model *between and around* real windows:
   on blends with the background, and on windows with one segment occluded. Nothing constrains that region. At 50–70 %
   the weights stay close to the original, so this behaviour survives implicitly (0.905 / 0.871). At 80 % the refit moves
   the weights far, and the behaviour drifts while accuracy holds.
2. **The new explanation is deterministic, not noise.** Pruned models from different calibration draws agree with each
   other at 0.917, but with the original at only 0.786. The mask and refit fix a *different* time pattern of evidence.
3. **What breaks.** The global top-10 Jaccard falls to 0.70 (ceiling 0.92, same masks 0.84) and the window correlation to
   0.82 (0.95 / 0.88). Dynamic classes suffer most: walking 0.61 vs ceiling 0.84 (same masks 0.72), stairs down 0.70 vs
   0.90 (0.77), standing 0.69 vs 0.84 (0.75). Lying is almost intact (0.92 vs 0.95).
4. **Each obvious fix inside the scenario was tried and failed:**
   - more calibration data (5×): 0.761;
   - blended (mixup) calibration windows: only +0.013;
   - activation-aware selection (Wanda, RIA): 0.760 / 0.752;
   - explanation-guided Taylor selection: worse, 0.716;
   - LAMP budgets: 0.720, trading LCI for F1;
   - SparseGPT / OBS: collapses at every dampening value (F1 ≤ 0.39).
5. **Earlier explanation-aware attempts also failed.** XGP rounds 1–4, PAIR (attribution-matching QAT loss) and GMR
   (gradient-matched reconstruction) either lost to tuned baselines or moved along the LCI–F1 trade-off. The lessons:
   noisy expected-gradient targets, per-weight saliency that ignores how removals interact, and gains that did not
   transfer to HAR70+.

## Why it is fixable in theory

- **The target exists at this size.** The *same* 80 % masks hold 0.865 LCI at 92.6 % accuracy once the weights are
  constrained on the input region the explainer reads. So the problem is the weight values the one-shot methods choose,
  not the sparsity budget.
- **The missing information is available to a deployer.** Function matching only on the ≈ 1k unlabeled calibration
  windows (no labels, gradient training) already reaches 0.835. The signal that restores the explanation is in the
  calibration data plus the original model. One-shot reconstruction simply does not use it.
- **The deficit is systematic** (cross-seed 0.917), concentrated on temporally structured classes, and grows smoothly
  with sparsity (gap 0.03 / 0.06 / 0.17 / 0.30 at 50 / 70 / 80 / 90 %). It is a structured error, so it can be targeted.

## Scope and limits (state these in the paper)

- **Scenario boundary.** The gap holds for *backprop-free* one-shot compression, the SparseGPT / Wanda setting. If
  gradient steps on the calibration windows are allowed (calfm, AdaRound / BRECQ style), the 80 % gap to full-data
  training shrinks to 0.022 and does not pass the bar.
- **Other cells checked and rejected:**
  - 90 % + KD: every method ≈ 0.78, and there is no same-size evidence of headroom.
  - 2-bit QAT: only 0.08 headroom.
  - ≥ 3-bit or ≤ 70 %: ceiling regime.
  - Older vs general adults: an accuracy gap of the base model, not a compression gap.
- **Generality is limited:**
  - one architecture (TinyHAR, 54k parameters) and one training dataset (HARTH);
  - inner val has 3 subjects per fold;
  - the same-mask references are seed 0 only;
  - the 80 % reference and mixcal have no held-out test run yet (`gap_study_v4.test_confirm`, commands in
    `environment/requirements.md`).

## Where the details are

- `gap_study_v3/GAP_REPORT.md`: full baseline sweep, artefact checks and mechanism. Code: `gap_study_v3/`
  (`baselines.py`, `confirm.py`, `analyze.py`, `diagnose.py`, `test_confirmation.py`).
- `gap_study_v4/RUNS.md`: pre-registered selection rules and decision points. Code: `gap_study_v4/` (`run_v4.py`,
  `analyze_v4.py`, `diagnose_v4.py`, `explainer_check.py`, `test_confirm.py`).
- Numbers: `results/gap-study-v3/`, `results/gap-study-v4/` (summarised in `results/README.md`).
