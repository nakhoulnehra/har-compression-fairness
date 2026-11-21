# Status
Current step: 9A — standard-compression grid (baselines only) on real test data, then subgroup analysis
New order (decided 2026-11-21): measure drift and unfairness with existing methods first (RQ1, RQ2, RQ0), then design XGP's final version for the observed drift (validation-only tuning; disclosed in the paper), then 9B adds XGP to the same grid.
Done:
- 1–5 setup, plan, data, model choice (TinyHAR frozen), baseline gap (older −10.9 pts, p < 0.001)
- 6 timing; 8 noise floor all folds done (τ: f0 0.66, f1 0.58, f2 0.63, f3 0.61/0.62 on the two machines, f4 0.61)
- XGP pilots (fold 0, validation only): round 1 (repair loss hurts); round 2 pre-registered → xgp_kd β 0.1, E_ft 20 for all. XGP wins without fine-tuning at 70% (LCI 0.62 vs 0.38, F1 0.60 vs 0.37); ties L1+KD after fine-tuning; worse on older-like proxy windows. EG m=4 self-agreement 0.51 → explains repair-loss failure.
- Round 3 (shift-aware + worst-case KD) designed, ON HOLD until 9A results.
Next: 9A runs on both GPUs → --analyze → look at where older adults drift more → finalise XGP → 9B.

## Decisions since the plan (short)
- XGP pilot round 2 (2026-11-21; plan 2.5 (b), round 2): KD is a shared fine-tuning recipe for every method, not part of XGP; XGP = the keep-score variants only; L_cons and the gate are dropped (EG m = 4 self-agreement ≈ 0.44 in an unsaved check; the gate never fired), and the EG-noise diagnostic is reported as a finding. Pre-registered rule: final XGP = the eligible variant with the highest mean LCI_val over {70, 90%} × {normal, older-like}; eligible = mean F1 ≥ best baseline − 1 pt at both levels; ties within 0.005 → fewer components; no eligible variant → stop, no fallback. Our resolution of β: the variants use the β of xgp_kd picked at that level by the same criterion (ties → smaller β). E_ft = 20 in Step 9 if l1_kd_e20 beats l1_kd (seed 0) by > 1 pt F1 at either level. Older-like windows: s ~ U(0.8, 0.9) time-stretch + a ~ U(0.7, 0.9) movement scaling around the per-channel mean (`har/augment.py`); worst-class I uses the reference windows' labels; seeds = the fine-tuning batch-order seed.
- Selection rule: best macro-F1; among models within 2 pt, lowest CPU SHAP time; params break ties
- 20 Hz low-pass on both datasets
- FA-LCI = min(LCI_older, LCI_general) (λ = 0.5)
- Machine split for 4f: each machine ran `--stage all` (tuning + all 5 folds) for its own models (Nakhoul GPU: CNN1D, TinyHAR, HART, DeepConvLSTM; Aya CPU: XGBoost), not the tune/final fold split in plan Step 4
- Step 5: walker (inferred) = walking-aid group without 503 (503 taken as the walking-sticks user; Ustad gives no IDs). Ustad 4-class scoring (primary, `ustad4`) merges labels and predictions: stairs → walking, shuffling → standing [Ustad Sec 2.4]. Secondary: `ustad4_drop_argmax7` (only 4-class windows, 7-class argmax) and `ustad4_drop_argmax4` (same windows, argmax over the 4).
- Step 6: Nakhoul's machine is timed on both CPU and GPU, because whether Steps 8–9 run on its GPU is still open (it changes the fold split, and Steps 7/9b then compare CPU with GPU).
- Step 7 (XGP, 2026-11-20; details in plan 2.5 (b)):
  - I(w) uses the dot-product alignment at θ0. The Pearson/cosine gradient there is zero (tested).
  - I(w) uses 1,000 class-stratified training windows, EG m = 8, and is averaged over EG seeds 0 and 1. It is computed once per fold.
  - L_cons = Pearson term + (1 − cos) of the 48-dim mean-|A| profile, EG m = 4.
  - Gate: LCI_val (real SHAP, inner val) < τ_k → CE + γ·L_cons, else CE. 10 epochs either way.
  - Tuning: highest LCI_val with F1 ≥ L1 − 1 pt. If no configuration qualifies: highest F1, flagged `selection_fallback`. When the gate picks CE, the two γ configurations are one run.
  - Ablations at 70/90%: xgp_score_only, l1_plus_lcons, xgp_no_gate, xgp.
- Step 7: pruning masks are in-place keep-masks (`model.prune_masks`) re-applied after every optimizer step, not `torch.nn.utils.prune`. Its forward-pre-hook never refreshes MHA's `out_proj.weight`.
- Step 8: τ_k = min over retrain seeds 1–3 of LCI_val(seed 0, seed s), the inner-val subject mean, the same quantity the gate compares. Output files are per fold and machine.
- Step 6: `eg_attributions` runs attention on PyTorch's math SDPA kernel (the fused kernels have no double backward; Park's loss failed on TinyHAR without it). In Park's fine-tuning step, CE uses train mode and the EG term eval mode (BN running stats not updated by interpolated inputs); Step 9's finetune must use the same convention.
- Step 6 GPU fixes: (1) `har/run_sanity_walking_aid.py` no longer sets `CUDA_VISIBLE_DEVICES=""` at import (it hid the GPU from `time_har_components`); `har/tests/test_time_har_components.py` checks the import leaves it unchanged. (2) `eg_attributions`, `shap_target_class` and `snip_scores_from_batches` (every gradient taken with the model in eval mode) run with `torch.backends.cudnn.flags(enabled=False)`: cuDNN's RNN backward only runs in training mode, so gradients through TinyHAR's LSTM in eval mode failed on CUDA. **Our method's loss (keep-score I(w), L_cons) must do the same** (go through `eg_attributions`, or wrap its own gradients in both `sdpa_kernel(SDPBackend.MATH)` and `cudnn.flags(enabled=False)`). No effect on CPU.

## Open issues
- On Windows, pandas writes CRLF line endings, so after `har.build_windows` git shows results/har-setup/ as modified even though the content is identical (`git diff --ignore-cr-at-eol` is empty). Restore with `git restore results/har-setup/`, or fix it with `lineterminator="\n"` or a `.gitattributes`.
- HART on CUDA warns that the backward pass of memory-efficient attention is non-deterministic: repeat runs of HART on GPU may not be bit-identical.
- Run inside `.venv`: the system Python on Nakhoul's laptop has no numpy.
- HAR70+: at least 15 pt of older subjects' walking/standing/sitting/lying windows are predicted as shuffling or stairs (older 4-class accuracy 76.6% with the 7-class argmax vs 91.6% restricted to the 4 classes; general 91.1 vs 97.3; merged Ustad scoring 84.6 vs 95.5). Subject 514 is an outlier (42.9% merged 4-class accuracy).
- `run_noise_floor` and `run_xgp_pilot` ran on GPU; `run_xgp_pilot2` is smoke-tested on CPU only. If either fails on GPU, rerun it there with `--smoke --device cuda` first.
- L_cons epoch cost is assumed equal to Park's (52 s on the 3050 Ti); the pilot's `finetune_seconds` will measure it.
- Step 6 projection uses 8 × n_val SHAP windows for tuning (Park 4 + ours 4 configurations); plan Section 5 counts 360 for both, so the projection is slightly higher there.
