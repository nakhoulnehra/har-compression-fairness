# Status
Current step: 8 (noise floor, fold 0) then the XGP pilot (fold 0, inner validation). Both are run on GPU by us.
Done: 7 (code): XGP + noise floor + pilot, implemented and tested on Aya CPU (2026-11-20). 176/176 tests pass (96 s). Smoke runs pass: noise floor 24.8 s, pilot 33 s (also run with `--tau 0.99` to exercise the L_cons branch, 49 s). Not committed.
- `har/xgp.py`: keep-score, I(w), L_cons, gate, tuning rule, method and ablation registry.
- `har/compression.py`: `l1_scores`, `prune_by_score` (in-place masks), `size_record`, `finetune` (exactly E_ft = 10 epochs), `teacher_segment_attributions`, `attribution_grad_context`.
- `har/attributions.py`: SHAP cache, per-subject LCI. `lci/metric.py`: `aggregate_time_segments`, `compute_lci_by_subject`.
- Runners: `har/run_noise_floor.py`, `har/run_xgp_pilot.py`.
- Tests: `har/tests/test_xgp.py`, `test_noise_floor.py`, `lci/tests/test_lci_windows.py`, plus additions to `test_compression.py` and `test_attributions.py`.
Fold split for Steps 8–9: nakhoul-3050 folds 0, 1, 3; uni-2080 folds 2, 3, 4 (fold 3 on both = cross-machine check)
GPU notes: attribution/SHAP/SNIP gradients run with cuDNN off and math attention (LSTM backward in eval mode)

Next, on nakhoul-3050 (GPU), from the repo root in `.venv`, one job at a time. Needs `data/processed/har/checkpoints/har-model-selection/har-final-tinyhar-fold0-nakhoul-3050.pt`.
1. `python -m har.run_noise_floor --folds 0 --device cuda --machine-name nakhoul-3050` (≈ 15 min) -> results/har-noise-floor/
2. `python -m har.run_xgp_pilot --fold 0 --device cuda --machine-name nakhoul-3050` (≈ 75 min upper bound) -> results/har-xgp-pilot/
Then: check the pilot table (gate branch per β, LCI_val vs L1/SNIP/Park, F1 within 1 pt of L1), commit results/, and put the times in environment/time.md and the numbers in results/README.md. Step 9's grid runner (`har/run_compression_grid.py`) is still to be written. The XGP parts it needs (`METHODS`, `ABLATIONS`, `pruning_scores`, `make_extra_loss`, `select_config`) are in `har/xgp.py`.

Done: 1 setup, 2 plan, 3 data pipeline + explain windows + backgrounds; 4e Nakhoul GPU smoke run; 4f all 5 candidates trained (tuning + folds 0–4) in results/har-model-selection/: CNN1D + TinyHAR (Nakhoul GPU, 21.6 min), XGBoost (Aya CPU, 8.7 min), HART + DeepConvLSTM (Nakhoul GPU, 83.2 min)
4: TinyHAR frozen (81.98 macro-F1, 54K params, 0.17 s SHAP/window).
5: baseline gap on HAR70+ passed (results/har-baseline-gap/, Aya CPU, 30 s; `python -m har.run_sanity_walking_aid`). Other older − walking aid (inferred) > 0 in every class set and every fold model: Ustad's merged 4 classes +13.7 pt accuracy (CI +0.2, +30.8; Ustad's HARTH model: +5 pt); 503 excluded (4 walker users) +18.9 pt (CI +7.6, +36.7). Older − general: −14.5 pt accuracy, −9.8 pt non-stair macro-F1 (CIs exclude 0).
6: component timing on both GPUs (fold 0): nakhoul-3050 3.2 h/fold, uni-2080 3.7 h/fold with ours charged at Park's cost. With the XGP ablations: ≈ 3.95 h and 4.6 h per fold (plan Section 5).
Last commit: e07eabf.

## Decisions since the plan (short)
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
- `run_noise_floor` and `run_xgp_pilot` are smoke-tested on CPU only. If either fails on GPU, rerun it there with `--smoke --device cuda` first.
- L_cons epoch cost is assumed equal to Park's (52 s on the 3050 Ti); the pilot's `finetune_seconds` will measure it.
- Step 6 projection uses 8 × n_val SHAP windows for tuning (Park 4 + ours 4 configurations); plan Section 5 counts 360 for both, so the projection is slightly higher there.
