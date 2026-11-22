# Status
Current step: XGP pilot round 3 (final design, validation only), code done 2026-11-22; GPU runs next, then 9B.
Order (decided 2026-11-21): measure drift and unfairness with existing methods first (9A: RQ1, RQ2, RQ0) ✔, then design XGP's final version for the observed drift (round 3; validation-only tuning; disclosed in the paper), then 9B adds XGP to the same grid.
Done:
- 1–5 setup, plan, data, model choice (TinyHAR frozen), baseline gap (older −10.9 pts, p < 0.001)
- 6 timing; 8 noise floor all folds done (τ: f0 0.66, f1 0.58, f2 0.63, f3 0.61/0.62 on the two machines, f4 0.61)
- XGP pilots (fold 0, validation only): round 1 (repair loss hurts); round 2 pre-registered → xgp_kd β 0.1, E_ft 20 for all. XGP wins without fine-tuning at 70% (LCI 0.62 vs 0.38, F1 0.60 vs 0.37); ties L1+KD after fine-tuning; worse on older-like proxy windows. EG m=4 self-agreement 0.51 → explains repair-loss failure.
- 9A grid (all 5 folds) + `--analyze` (results/har-compression-grid-analysis/). Without fine-tuning, older adults drift more in posture classes (l1: standing 70% gap −0.10, sitting 90% −0.11, lying 50% −0.13); the accuracy gap widens at 90% (l1 −0.21, l1_kd −0.17 vs −0.11 to −0.14 at 30–70%). With KD fine-tuning the LCI gap is ≈ 0 up to 70%.
- **Round 3 (2026-11-22):** pre-registration written into the plan first (2.5 (b), "Round 3 pre-registration", with the disclosure line, gait-ageing citations for the fixed augmentation, SparseGPT / OBC / group-DRO citations), then code. `har/xgp3.py` (group-aware class-balanced keep-score, K = 5 iterative pruning, ridge layer-wise reconstruction, rules), `har/run_xgp_pilot3.py` (two phases: tuning, ablation; `--folds`, `--machine-name`; budget curve at 0 / 2 / 5 / 20 epochs), DRO + epoch callback in `compression.finetune`.
  - **Amended the same day, before any run:** the older-like transform gains a per-sensor random 3D rotation (back and thigh independently, random axis, angle ~ U(−15°, 15°), after the stretch and amplitude steps; Um et al. 2017; fixed a priori), so static postures change too. Phase 1 uses `--curve-seeds 0`.
  - 248/248 tests pass; the full smoke chain passes on CPU.
  - **`--prepare` rerun for real** → `results/har-xgp-pilot/round3/` pre-registration (with the `amended` note) + window tables with the rotation parameters (fold 0's explain windows and s, a = round 2's).
- 9A head check fixed: `head_unpruned` = no mask and no zeroed head weight (the old "head unchanged" test was False for every fine-tuned row, because the head trains). `--recheck-head` rewrites the column of finished folds from the checkpoints (needs each GPU machine's checkpoints).

Next (exact commands in environment/requirements.md, round 3 block):
1. Pull. Smoke on each GPU: uni-2080 `python -m har.run_xgp_pilot3 --phase tuning --folds 0 --smoke --device cuda --machine-name uni-2080`; nakhoul-3050 the same with `--folds 1 --machine-name nakhoul-3050`.
2. Phase 1: uni-2080 `python -m har.run_xgp_pilot3 --phase tuning --folds 0 --curve-seeds 0 --device cuda --machine-name uni-2080 --noise-floor-machine nakhoul-3050`; nakhoul-3050 `python -m har.run_xgp_pilot3 --phase tuning --folds 1 --curve-seeds 0 --device cuda --machine-name nakhoul-3050 --noise-floor-machine nakhoul-3050` (≈ 6.6 h each). The 2 / 5-epoch curve points are taken for seed 0 only; the full budget curve goes to 9B. Commit both folds' files; the pruned-start checkpoints stay local.
3. `python -m har.run_xgp_pilot3 --analyze --phase tuning` → `har-xgp-pilot3-tinyhar-tuning-selection.json`; commit it, pull it on both GPUs.
4. Phase 2: the same two commands with `--phase ablation` (≈ 3.5 h each).
5. `python -m har.run_xgp_pilot3 --analyze --phase ablation` → `har-xgp-pilot3-tinyhar-decision.json`. Exit 2 = no eligible candidate for a scenario → XGP not frozen there; stop and review (no fallback).
6. 9A head column: `python -m har.run_compression_grid --folds 0 1 3 --recheck-head --machine-name nakhoul-3050` and `--folds 2 3 4 --recheck-head --machine-name uni-2080`.
7. Register the final XGP in `run_compression_grid` (one scorer + method entry) → 9B.

## Decisions since the plan (short)
- RQ0–RQ3 written into the plan (top of the file), 2026-11-21.
- Fold 3 in the analysis = uni-2080's copy; nakhoul-3050's fold 3 is for the Step 9b machine comparison only (2026-11-21).
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
- Round 3: the older-like proxy now rotates each sensor by up to 15° (amendment), so posture windows change. It is still a generic orientation perturbation, not a model of older adults' postures. If 9A's posture drift has another cause, the proxy misses it (limitation note in the plan).
- Round 3: calibration windows come from the same 3 inner-val subjects as the evaluation windows (disjoint windows), so XGP's inner-val numbers may be slightly optimistic against baselines that never see inner-val. 9B's test numbers are not affected. Stairs F1 in round 3 rests on 42–45 explain windows per stairs class (calibration took the rest).
- Round 3: SparseGPT / OBC / Sagawa / Um et al. / gait-ageing PDFs are not in `papers/`; verify the cited sections and tables before submission.
- The 9A analysis config reports `head_unpruned_all: false` (the old check's bug). After `--recheck-head` on both machines, rerun `python -m har.run_compression_grid --analyze` to refresh it; nothing else changes.
- 9A: `run_compression_grid` is smoke-tested on CPU only. Run `--smoke --device cuda` on each GPU first (needs that machine's smoke noise floor: `python -m har.run_noise_floor --folds 0 --smoke --device cuda --machine-name <m>`).
- CPU inference time per model (plan 2.7, Aya's machine only) is not part of the grid runner yet.
- On Windows, pandas writes CRLF line endings, so after `har.build_windows` git shows results/har-setup/ as modified even though the content is identical (`git diff --ignore-cr-at-eol` is empty). Restore with `git restore results/har-setup/`, or fix it with `lineterminator="\n"` or a `.gitattributes`.
- HART on CUDA warns that the backward pass of memory-efficient attention is non-deterministic: repeat runs of HART on GPU may not be bit-identical.
- Run inside `.venv`: the system Python on Nakhoul's laptop has no numpy.
- HAR70+: at least 15 pt of older subjects' walking/standing/sitting/lying windows are predicted as shuffling or stairs (older 4-class accuracy 76.6% with the 7-class argmax vs 91.6% restricted to the 4 classes; general 91.1 vs 97.3; merged Ustad scoring 84.6 vs 95.5). Subject 514 is an outlier (42.9% merged 4-class accuracy).
- `run_noise_floor` and `run_xgp_pilot` ran on GPU; `run_xgp_pilot2` is smoke-tested on CPU only. If either fails on GPU, rerun it there with `--smoke --device cuda` first.
- L_cons epoch cost is assumed equal to Park's (52 s on the 3050 Ti); the pilot's `finetune_seconds` will measure it.
- Step 6 projection uses 8 × n_val SHAP windows for tuning (Park 4 + ours 4 configurations); plan Section 5 counts 360 for both, so the projection is slightly higher there.
