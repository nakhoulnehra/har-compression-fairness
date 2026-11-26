# Gap report: explanation consistency after compressing a HAR model

2026-11-26, gap study v3 (third attempt). **No new method was built.** Every model is an existing technique, or
an earlier run's checkpoint re-scored. Selection and tuning used HARTH inner validation only. Test numbers come
from the existing one-shot test runs (9A and agent B's); nothing was selected on test.

**Result: one confirmed gap and one weaker second gap.**
- **Primary:** post-training pruning **without retraining** at **80 %** sparsity. Every fairly tuned existing method
  stays ≥ 0.10 LCI below the uncompressed model in 5/5 folds and on both test populations, at −1 pt accuracy.
- **Secondary:** **90 % with KD fine-tuning**. It passes the same validation gate (4/5 folds, −0.9 pt accuracy)
  but is weaker: two folds are at 0.08 / 0.10, and the test so far covers only the weaker standard baseline.

Model and metric:
- Frozen TinyHAR (54,273 parameters), HARTH-trained, 5 subject folds.
- Explanations: SHAP GradientExplainer of the original model's predicted class, 48 features (6 axes × 8 time
  segments).
- LCI = 0.5 · per-window Pearson + 0.5 · Jaccard of the subject's global top-10 features.
- Inner val: 50 explain windows per subject, 3 subjects per fold.

## 1. Audit summary (details: `PREVIOUS_ATTEMPTS.md`)

| Agent | What it tried | What it found | What went wrong |
|---|---|---|---|
| A, "PAIR" (11-25 21:00 → 11-26 01:14, uncommitted) | Calibration-only 2-bit QAT + KD; a new attribution-matching loss (PAIR), pre-registered twice, with a test phase | PAIR lost to KD at lr 1e-4 both times (+0.012 / −0.031 LCI at −2 / −4 pt F1). By-product: KD at lr 1e-3 vs 1e-4 gives the same F1 but −0.11 LCI | **Method before gap.** The cell had only 0.08 headroom to the full-data reference, the early "gap" came partly from a mis-set baseline lr, and a second setting was left half-run |
| B, "gap study + GMR" (11-26 06:18 → 09:23, uncommitted, `gap_study/`) | Master table, 5 baselines × 5 folds × 3 seeds without retraining at 50–90 %; **then a new method (GMR)** with μ tuning, a grid extension, and a test phase | Gap at 80 % (l1_fullrecon 0.787 vs 0.931). GMR: +0.037 on validation, +0.023 on test, no gain on HAR70+ | **Built a method.** SparseGPT ran at one dampening value. No explanation-guided baseline. Same-seed SHAP compared with an other-seed ceiling. Its gradient-error mechanism contradicts its own data (the best-LCI method has the worst gradient error). It dismissed the fine-tuned 90 % cell with a self-referential headroom rule |

## 2. Phase 1: ranked candidates (existing files only, before any new run)

| # | Cell | u | Best existing baseline (LCI / accuracy) | Drop | Accuracy | Hypothesised cause | Real-world case | Outcome |
|---|---|---|---|---|---|---|---|---|
| 1 | LCI, HARTH (+ HAR70+), unstructured 90 %, KD fine-tuning (full data) | 0.931 | 9A l1_kd 0.744 / 0.911 (Park KD 0.716 / 0.798) | 0.187 (20 %) | −1.6 pt | KD constrains only the outputs; at 90 % the evidence gets re-routed; Park's targets are noisy | Standard compress-then-fine-tune for MCU deployment | **Passes, weaker** (§4) |
| 2 | LCI, HARTH (+ HAR70+), unstructured 80 %, no retraining (≈ 980 unlabeled calibration windows) | 0.931 | agent B l1_fullrecon 0.787 / 0.916 | 0.145 (15.5 %) | −1.0 pt | Layer-wise reconstruction matches outputs, not the class evidence over time | Deployer without labels, training pipeline or GPU | **Confirmed** (§3) |
| 3 | LCI, 2-bit per channel, calibration-only QAT + KD | 0.931 | agent A kd lr 1e-4 0.808 / 0.924 | 0.123 (13 %) | −0.3 pt | Capacity trade-off | As 2, at MCU size | Not confirmed: 3/5 folds ≥ 0.10 on existing data, 1 seed, least headroom; not re-run |
| 4 | LCI, unstructured 90 %, no retraining | 0.931 | wanda_fullrecon 0.625 / 0.859 | 0.306 (33 %) | −6.8 pt (F1 −17 pt) | As 2, plus accuracy loss | — | Passes the numeric gate, but accuracy is not usable (§3 sweep) |
| 5 | Structured filters 75 %, no retraining | 0.931 | taylor_recon 0.736 (F1 0.752) | 0.19 | F1 −8 pt | Coarse removal | Latency | Excluded: accuracy, 2 folds |
| — | ≤ 70 % or ≥ 3-bit with KD; 2-bit / 90 % + 4-bit PTQ without training | — | 0.88–0.96 / 0.26–0.60 | — | — | — | — | Excluded: ceiling regime / accuracy collapse |

## 3. Confirmed gap: post-training pruning without retraining, 80 %

### Gap statement
> In **post-training compression without retraining** (unstructured pruning, ≈ 980 unlabeled HARTH training windows
> for calibration, no gradient training) at **80 %** sparsity on **HARTH** (5 subject folds), the uncompressed
> TinyHAR scores **0.932 LCI** (its own SHAP-noise ceiling, independent SHAP seeds). Every existing method, fairly
> tuned, scores at most **0.760** (best: magnitude selection with a full layer-wise ridge refit of every layer,
> l1_fullrecon; 0.786 under the same-seed protocol). The gap of **0.172** (**18.5 %** relative; same-seed 0.146 /
> 15.6 %):
> - is outside seed variance over **3 calibration seeds × 5 folds** (fold-paired 95 % CI [0.069, 0.276]; ≈ 10×
>   the within-fold seed SD 0.017 and the SHAP-noise SD 0.016);
> - holds in **5 of 5 folds** (0.104–0.310);
> - holds on **HARTH test subjects** (gap 0.144, 15/22 subjects ≥ 0.10) and on **HAR70+ older adults** (0.117,
>   12/18; FA-LCI 0.775 vs 0.915);
> - comes at only **−1.0 pt accuracy** (inner val; −2.2 pt on test), so the LCI drop is 14× the relative accuracy
>   drop.
>
> It is most likely caused by **a deterministic, mask-induced re-routing of the temporal evidence inside the
> window**: the pruned network reaches the same decisions through a different time pattern. Supporting evidence:
> - pruned models from different calibration draws agree with each other at 0.917 but with the original at 0.786;
> - the gradient-free occlusion LCI drops too (0.639 vs 0.94 at 50 %);
> - the per-window correlation falls from 0.95 to 0.85;
> - the losses concentrate on walking and stairs;
> - prediction flips, explainer noise, sensor re-weighting, attribution flattening and calibration-set size are
>   all ruled out.

### Full sweep (validation, mean over 5 folds × 3 seeds; LCI same-seed / independent-seed, accuracy)

Original accuracy 0.927, F1 0.836; u = 0.932.

| Level | Best baseline | LCI same / indep | Gap same / indep | Folds ≥ 0.10 (same / indep) | Accuracy (Δ) | F1 (Δ) | Gate |
|---|---|---|---|---|---|---|---|
| 50 % | wanda / l1_fullrecon | 0.953 / 0.905 (l1_fullrecon) | −0.02 / 0.03 | 0 / – | 0.929 (+0.3) | 0.836 | no gap |
| 70 % | l1_fullrecon | 0.887 / 0.871 | 0.045 / 0.061, CI [−0.014, 0.136] | 1 / 1 | 0.925 (−0.1) | 0.823 (−1.2) | fails (size, CI) |
| **80 %** | **l1_fullrecon** | **0.786 / 0.760** | **0.146 / 0.172**, CI [0.069, 0.276] | **3 / 5** | **0.916 (−1.0)** | 0.791 (−4.4) | **passes** (G3 on the independent value) |
| 90 % | wanda_fullrecon | 0.625 / – (l1_fullrecon indep 0.637) | 0.307 / 0.295 | 5 / – | 0.859 (−6.8) | 0.667 (−17) | numeric pass, **accuracy not usable** |

The gap grows smoothly with sparsity (−0.02 → 0.06 → 0.17 → 0.30), with no single-level spike. Test, from the
existing one-shot runs (`results/gap-study-v3/test-confirmation/`):
- 70 %: gap 0.004 (no gap, consistent with validation).
- 80 %: 0.144 HARTH / 0.117 HAR70+.
- 90 %: 0.26 / 0.24 at −11 / −14 pt accuracy.

### Every baseline at 80 % (same tuning budget: 5 folds × 3 calibration seeds unless stated)

| Method (existing technique) | Tuning | LCI same / indep | Accuracy | F1 | Notes |
|---|---|---|---|---|---|
| l1_fullrecon (magnitude, K = 5 steps, ridge refit of every layer incl. LSTM / attention) | ridge damp {0.001, 0.01, 0.03, 0.1} (agent B), 0.01 best | **0.786 / 0.760** | 0.916 | 0.791 | best LCI |
| wanda_fullrecon (Sun et al. 2024 + full refit) | damp 0.01 | 0.779 / 0.760 | 0.913 | 0.784 | |
| l1_fullrecon with 5× calibration windows (≤ 50 per subject-class, ≈ 4.8 k) | seed 0 | 0.778 / 0.761 | 0.919 | 0.796 | more data does not help |
| taylor_fullrecon (explanation-target first-order Taylor selection, Molchanov 2019 / Yeom 2021 style, + full refit) | — | 0.728 / 0.716 | 0.913 | 0.783 | explanation-guided selection is *worse* |
| lamp_recon (round 4, LAMP budgets, partial refit) | — | 0.726 / 0.720 | 0.913 | **0.807** | most accurate by F1, low LCI |
| l1_recon (round 4) | — | 0.723 / 0.706 | 0.910 | 0.787 | |
| SparseGPT (Frantar & Alistarh 2023) | percdamp {0.01, 0.1, 1.0} | best 0.519 (0.01) / – | 0.59–0.63 | 0.39 | collapses at every dampening value: one-shot OBS on this 54 k recurrent model |
| GMR (agent B's prototype, not an existing method, reported for completeness) | μ chosen after a grid extension | 0.824 / – | ≈ 0.916 | 0.791 | closes 26 % on validation, 16 % on test, none on HAR70+ |
| *Retraining is outside the scenario:* KD fine-tuning at 80 % (l1_kd / l1_fullrecon_kd, seed 0) | lr 1e-4 | 0.846 / 0.813, 0.848 / 0.818 | 0.920–0.922 | 0.810–0.814 | shows the gap is closable with training (gap to u ≈ 0.08–0.12) |

Trade-offs:
- No baseline gets both accuracy and explanations. lamp_recon has the best F1 (+1.6 pt over l1_fullrecon) but
  −0.06 LCI.
- More calibration data costs 5× the build time for no LCI gain.
- Explanation-guided selection costs a gradient pass per step and loses 0.06 LCI.
- Retraining recovers +0.06 LCI but needs the labelled training set and a GPU, which the scenario lacks.

### Metric-artefact checks (Phase 2 rule 6)
| Check | Result at 80 % (l1_fullrecon) | Reading |
|---|---|---|
| Noise floor of the uncompressed model | SHAP seed 0 vs 1: 0.932 (0.914–0.946 per fold, SD 0.016). Retrained seed: 0.666 (folds 0, 1, 3) | The gap is ≈ 10× the SHAP noise. The pruned model sits halfway between the ceiling and the retrain floor |
| Same-seed vs independent-seed SHAP | 0.786 vs 0.760. At 50 % 0.953 vs 0.905 | Agent B's same-seed comparison inflated compressed LCI (common random numbers). The gap is *larger* than reported |
| Explainer stability on the compressed model | SHAP self-consistency 0.933 (original 0.932) | The explainer is as stable on the pruned model as on the original |
| Prediction flips | top-1 agreement 0.919; LCI on agreeing windows 0.778 | Not explained by changed predictions |
| Accuracy collapse | −1.0 pt accuracy, −4.4 pt F1; relative LCI drop / relative accuracy drop = 14 | Disproportionate |
| Explainer family | Occlusion (gradient-free) LCI 0.639, vs 0.936 at 50 % and 0.537 for a retrained model | Not a gradient-SHAP artefact: the model's *function* changes. This refutes agent B's gradient-only mechanism |

### Phase 3: mechanism (diagnostics only; `results/gap-study-v3/phase3/`)
1. **Systematic, not noisy.** Pruned models from different calibration draws agree with each other at LCI 0.917
   (≈ the SHAP ceiling) but with the original at 0.786. The removal pattern determines a new, stable explanation.
   More calibration windows do not move it (5×: 0.778).
2. **Which part of LCI breaks.**
   - Per-window correlation: 0.848 (noise 0.949).
   - Global top-10 Jaccard: 0.725 (noise 0.915); 1.1 of the 10 top features change sensor axis (noise 0.13).
3. **Where.**
   - By class (LCI vs SHAP noise): walking 0.61 vs 0.89, stairs down / up 0.73 / 0.74 vs 0.95 / 0.89,
     standing 0.75 vs 0.86, sitting 0.81 vs 0.89, lying 0.90 vs 0.97. Dynamic, temporally structured
     activities break most.
   - By sensor: the error shares per axis match the SHAP-noise pattern (back-x 0.22, thigh-x 0.20, …), so no
     sensor is singled out.
4. **How.**
   - Not flattening: top-10 mass 0.56 vs 0.53, magnitude ratio 1.00.
   - Not sign flips: 1.7 % of the top-10 features.
   - The attribution *pattern over time segments* changes. lamp_recon instead also shrinks magnitudes (ratio 0.68).
5. **Smooth, not a threshold.** Window correlation 0.988 / 0.960 / 0.848 / 0.681 at 50 / 70 / 80 / 90 %; occlusion
   LCI 0.94 / 0.84 / 0.64 / 0.49.
6. **Correlates.** Within a model, windows where the pruned model is less confident on the explained class have
   lower explanation agreement (Spearman 0.38 for l1_fullrecon). This is absent for lamp_recon (−0.05), whose
   change is a global rescaling.
7. **Closability (oracle-style check, no tuning).** The same masks with KD fine-tuning on the full data reach 0.848
   at 80 %, so a model with this sparsity *can* keep the explanation. What is missing is a no-training way to get
   there.

### Requirements for a future method (requirements only, no design)
1. **Preserve the sensitivity, not just the outputs.** Matching layer outputs on the calibration windows (logit
   error ≈ 3 %) still loses 0.17 LCI. A method must keep how the explained logit responds to *localised
   perturbations in time* (occlusion and path points), which lie off the calibration inputs. It breaks the
   reconstruction assumption that "calibration inputs span the directions that matter".
2. **Account for the joint effect of removals on the temporal evidence.** Per-weight saliency, even with the
   explanation target (Taylor), is worse than magnitude. Since the mask fixes the new explanation (cross-seed
   0.917), selection must see interactions, not per-weight scores.
3. **Stay in the scenario.** Label-free, ≤ 1 k windows, no gradient training, CPU minutes. More unlabeled windows
   alone do not help.
4. **Not trade accuracy away.** lamp_recon / GMR μ ≥ 0.3 show that moving along the LCI–F1 trade-off is easy. A
   method must beat l1_fullrecon on LCI at equal accuracy *and* beat lamp_recon on F1.
5. **Generalise beyond the training population** (GMR's gain vanished on HAR70+). Evaluate with independent SHAP
   seeds and a gradient-free explainer.

## 4. Secondary gap: 90 % sparsity with KD fine-tuning (passes on validation, weaker)

Re-run under one protocol (5 folds; KD fine-tuning on the full training windows, 20 epochs, 9A recipe):

| Method (existing technique), 90 % | lr (tuning: {3e-5, 1e-4, 3e-4}, rule in RUNS.md) | Seeds × folds | LCI same / indep | Accuracy (Δ) | F1 (Δ) |
|---|---|---|---|---|---|
| **lamp_recon_kd** (LAMP budgets + reconstruction start + KD) | **3e-4 (selected)** | 3 × 5 | **0.797 / 0.781** | 0.918 (−0.9) | 0.810 (−2.5) |
| lamp_recon_kd | 1e-4 | 3 × 5 | 0.766 / 0.749 | 0.914 (−1.2) | 0.797 |
| l1_fullrecon_kd (full refit start + KD) | 3e-4 (selected) | 3 × 5 | 0.771 / 0.764 | 0.917 (−0.9) | 0.799 |
| l1_fullrecon_kd | 1e-4 / 3e-5 | 3 × 5 / 1 × 5 | 0.751 / 0.694 | 0.915 / 0.908 | 0.787 / 0.756 |
| l1_kd (9A recipe) | 1e-4 / 3e-4 / 3e-5 | 3 × 5 / 1 × 5 / 1 × 5 | 0.750 / 0.750 / 0.679 | 0.910 / 0.909 / 0.904 | 0.777 / 0.789 / 0.759 |
| park_kd / park (Park et al. 2020, 9A checkpoints re-scored) | 1e-4 (γ 1) | 1 × 3 (folds 0, 1, 3) | 0.743 / 0.718 | 0.859 / 0.830 | 0.667 / 0.635 |

Gate (u = 0.932, same as §3):
- Gap 0.135 same-seed / **0.151 independent-seed (16.2 %)**, CI [0.061, 0.241].
- Folds ≥ 0.10: 3/5 same-seed, **4/5 independent-seed** (0.144 / 0.268 / 0.081 / 0.157 / 0.104).
- Within-fold seed SD 0.027, SHAP-noise SD 0.016.
- Accuracy −0.9 pt; LCI drop / accuracy drop = 15.
- Explainer stable on the compressed model (self 0.958); top-1 agreement 0.94; LCI on agreeing windows 0.806.
- Occlusion LCI 0.651.

Neighbours (same seed-0 protocol):
- 80 %: l1_fullrecon_kd 0.848 / 0.818, gap 0.084 / 0.115, folds 1/5 / 3/5 → **no gap**.
- 70 %: 9A l1_kd 0.882 → no gap.
- So the fine-tuned gap appears between 80 % and 90 %, past the regime where KD recovers explanations.

- **Why it is weaker than §3.** Fold 2's gap is 0.081 and fold 4's is 0.104. On the same-seed protocol only 3/5 folds pass. Seed SD (0.027) is higher than in §3, and the tuned baseline has no test run yet.
- **What a stronger baseline buys.** The 9A standard l1_kd (0.750) understated the best achievable value.
  LAMP budgets + a reconstruction start + a tuned lr (3e-4) add ≈ +0.04–0.05 LCI at +1–2 pt F1. That is the same
  lesson as XGP rounds 3–4: the "gap" shrank once baselines got LAMP and reconstruction.
- **Mechanism (diagnostic).** KD fine-tunes of the same mask with different batch orders agree with each other at
  0.959 but with the original at 0.750 (tuned lamp_recon_kd, whose masks differ per calibration seed: 0.850 vs 0.797). Their window correlation (0.77) is near an independently retrained model's
  (0.73). Walking (0.58; tuned lamp 0.66) and standing (0.65; 0.68) break most. Occlusion also drops (0.59; 0.69). The fine-tuned sparse model
  converges to its *own* explanation, as different from the original as a retrained model's.
- **Attribution-preserving distillation** (Park et al. 2020; 9A checkpoints re-scored, folds 0, 1, 3): park_kd
  0.743 at −5.4 pt accuracy, park 0.718 at −8.2 pt. No better than KD and less accurate (cf. agent A's diagnosis:
  noisy EG targets).
- **Test** (existing 9A run, l1_kd only, not the tuned baseline): HARTH gap 0.169 (14/22 subjects), HAR70+ 0.182
  (18/18) at −0.7 / −3.4 pt accuracy, FA-LCI 0.733. **Open step:** one test run of lamp_recon_kd lr 3e-4, seed 0
  (checkpoints are in `data/processed/har/checkpoints/gap-study-v3/`).

## 5. Baselines considered but not run
| Method | Why not run |
|---|---|
| RIA (Zhang et al., ICLR 2024), DSnoT (training-free mask refinement, 2024) | Wanda + full refit covers activation-aware selection. RIA / DSnoT gains are reported on LLMs with outlier channels, which this 54 k model lacks. Cheap: the first thing to add if the jury asks |
| OBC (Frantar et al. 2022) | SparseGPT is its scalable approximation, and it collapses here at every dampening value. Exact OBC on all layers is feasible but was not run |
| ECQx (LRP-guided quantization), Park et al. WCAM as a fine-tuning loss in the no-retraining cell | Need gradient training (outside scenario 3). Park was evaluated in the fine-tuned cell (§4) |
| C-SWAP (causal explainability-aware structured pruning, arXiv 2510.18636), Malik et al. 2025 (attribution-guided neuron pruning) | Structured / neuron-level and image-focused. The attribution-guided idea is represented by taylor_fullrecon (and the team's XGP rounds) |
| Saliency-assisted / task-circuit quantization (2024–2025) | Quantization cells are in the ceiling regime (≥ 3-bit) or collapse (2-bit PTQ) |

## 6. Confidence and limits
- **Moderate-high** that the 80 % no-retraining gap is real for this model: 5/5 folds, 3 seeds, 2 test populations,
  ≈ 10× the noise, artefacts ruled out.
- **Low** for generality:
  - one architecture (TinyHAR);
  - one explainer family plus occlusion;
  - one training dataset (HARTH);
  - inner val has only 3 subjects per fold (S022 / S024 recur across folds).
- On agent B's same-seed protocol only 3 of 5 folds reach 0.10, so the scope claim relies on the independent-seed
  comparison, which this report argues is the correct one.
- Next check before claiming generality: repeat §3 on the frozen CNN1D (`results/har-model-selection/`, all 5 folds
  exist) with the same runner.
