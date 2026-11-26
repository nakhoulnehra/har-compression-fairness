# Gap report v4: explanation consistency of a compressed TinyHAR

2026-11-26, gap study v4. **No new method was built.** Existing techniques only; every choice on HARTH inner
validation (5 subject folds, 3 subjects × 50 explain windows per fold); HAR70+ and HARTH test subjects touched once,
after selection. Run log and pre-registered rules: `gap_study_v4/RUNS.md`. Numbers: `results/gap-study-v4/`.

DRAFT — decision section pending.

## 1. What the earlier experiments established (audit)

| Study | Question | Finding | What it means for explanations under compression |
|---|---|---|---|
| Noise floor (step 8) | How much do explanations vary without compression? | SHAP seed vs seed: 0.91–0.95 LCI; an independently retrained TinyHAR (same data, same accuracy): 0.67 | Accuracy does not pin the explanation. Equally accurate models disagree at 0.67 |
| 9A grid (5 folds, test) | Drift and group gaps with standard methods | KD fine-tuning keeps LCI ≥ 0.88 up to 70 %; at 90 % l1_kd falls to 0.74 at −1.6 pt accuracy; no older-vs-general LCI gap after fine-tuning | The compression gap is in sparsity, not in age groups |
| XGP rounds 1–4, PAIR, GMR | Explanation-aware pruning / losses | All lost to tuned baselines (LAMP + reconstruction + KD) or moved along the LCI–F1 trade-off | Attribution-matching losses with noisy EG targets do not help; baselines must be strong |
| Low-bit / ultra-small / final test | Quantization below 30 KB | ≥ 3-bit with KD ≈ ceiling; 2-bit collapses accuracy | Quantization cells are ceiling or broken |
| v3 (this morning) | Gap with fairly tuned baselines | No-retrain 80 %: best 0.760 vs ceiling 0.932; 90 % + KD: 0.781 | Gap measured against the *uncompressed* model; same-size evidence only KD (0.818) |

## 2. Mechanism (theory, then the test of it)

- **The explainer reads the model off the data.** GradientExplainer (expected gradients) evaluates gradients at blends
  x' + a(x − x') of a background (training) window and the explained window. Occlusion evaluates windows with one
  segment replaced by the background mean. LCI compares these attributions.
- **Accuracy only constrains the model on the data.** Two equally accurate models agree at 0.67, so the off-data
  behaviour is left free by the training signal.
- **Every existing compressor matches the original only on real windows.** Layer-wise reconstruction fits
  calibration windows; KD fits training windows. At low sparsity the weights stay near the original, so the off-data
  behaviour survives implicitly. At 80–90 % the refit moves the weights far, and only the data-constrained part is
  kept. The explanation drifts while accuracy stays.
- **Prediction tested in v4:** matching the original on blends of windows (function matching, Beyer et al. CVPR
  2022) at the same masks should restore SHAP and occlusion LCI. It does (§4).
