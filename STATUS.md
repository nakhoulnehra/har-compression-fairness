# Status
Current step: 7: implement XGP + noise floor (Step 8) + fold-0 pilot
Done: 6: component timing on both GPUs (fold 0): nakhoul-3050 3.2 h/fold, uni-2080 3.7 h/fold (ours estimated = Park's cost)
Fold split for Steps 8–9: nakhoul-3050 folds 0, 1, 3; uni-2080 folds 2, 3, 4 (fold 3 on both = cross-machine check)
GPU notes: attribution/SHAP/SNIP gradients run with cuDNN off and math attention (LSTM backward in eval mode)
Next: after the pilot passes on validation, run noise floor + compression grid on both GPUs

Done: 1 setup, 2 plan, 3 data pipeline + explain windows + backgrounds; 4e Nakhoul GPU smoke run; 4f all 5 candidates trained (tuning + folds 0–4) in results/har-model-selection/: CNN1D + TinyHAR (Nakhoul GPU, 21.6 min), XGBoost (Aya CPU, 8.7 min), HART + DeepConvLSTM (Nakhoul GPU, 83.2 min)
4: TinyHAR frozen (81.98 macro-F1, 54K params, 0.17 s SHAP/window).
5: baseline gap on HAR70+ passed (results/har-baseline-gap/, Aya CPU, 30 s; `python -m har.run_sanity_walking_aid`). Other older − walking aid (inferred) > 0 in every class set and every fold model: Ustad's merged 4 classes +13.7 pt accuracy (CI +0.2, +30.8; Ustad's HARTH model: +5 pt); 503 excluded (4 walker users) +18.9 pt (CI +7.6, +36.7). Older − general: −14.5 pt accuracy, −9.8 pt non-stair macro-F1 (CIs exclude 0).
6 (code): baseline parts of `har/compression.py` (prunable_tensors, ptq_symmetric, snip_scores, eg_attributions, segment_attributions, Park's targets + L_AM) with tests (`har/tests/test_compression.py`, `har/tests/test_attributions.py`); `har/time_har_components.py`. 139/139 tests pass; `--smoke` on Aya CPU 28.5 s.
Last commit: 8d7428b (Step 5). Step 6 code not committed.

Next: the Step 6 real runs, one job per machine at a time, from the repo root in `.venv`. Each needs `data/processed/har/checkpoints/har-model-selection/har-final-tinyhar-fold0-nakhoul-3050.pt`.
- Aya, CPU: `python -m har.time_har_components --fold 0 --device cpu --threads 10 --machine-name aya-laptop`
- Nakhoul, CPU: `python -m har.time_har_components --fold 0 --device cpu --threads 8 --machine-name nakhoul-3050`
- Nakhoul, GPU: `python -m har.time_har_components --fold 0 --device cuda --machine-name nakhoul-3050`
Then: times into environment/time.md and results/README.md. If the projected Steps 8 + 9 per fold are off by more than 30% from Section 5 (TinyHAR 11.6 h Aya, 5.9 h Nakhoul), update Section 5 and the fold split, and decide CPU or GPU for Nakhoul's folds.

## Decisions since the plan (short)
- Selection rule: best macro-F1; among models within 2 pt, lowest CPU SHAP time; params break ties
- 20 Hz low-pass on both datasets
- FA-LCI = min(LCI_older, LCI_general) (λ = 0.5)
- Machine split for 4f: each machine ran `--stage all` (tuning + all 5 folds) for its own models (Nakhoul GPU: CNN1D, TinyHAR, HART, DeepConvLSTM; Aya CPU: XGBoost), not the tune/final fold split in plan Step 4
- Step 5: walker (inferred) = walking-aid group without 503 (503 taken as the walking-sticks user; Ustad gives no IDs). Ustad 4-class scoring (primary, `ustad4`) merges labels and predictions: stairs → walking, shuffling → standing [Ustad Sec 2.4]. Secondary: `ustad4_drop_argmax7` (only 4-class windows, 7-class argmax) and `ustad4_drop_argmax4` (same windows, argmax over the 4).
- Step 6: Nakhoul's machine is timed on both CPU and GPU, because whether Steps 8–9 run on its GPU is still open (it changes the fold split, and Steps 7/9b then compare CPU with GPU).
- Step 6: our method (keep-score, L_cons, gate) is NOT implemented yet; it will be designed and reviewed separately before Step 9. In the timing, its cost is "estimated = Park's cost".
- Step 6: `eg_attributions` runs attention on PyTorch's math SDPA kernel (the fused kernels have no double backward; Park's loss failed on TinyHAR without it). In Park's fine-tuning step, CE uses train mode and the EG term eval mode (BN running stats not updated by interpolated inputs); Step 9's finetune must use the same convention.
- Step 6 GPU fixes: (1) `har/run_sanity_walking_aid.py` no longer sets `CUDA_VISIBLE_DEVICES=""` at import (it hid the GPU from `time_har_components`); `har/tests/test_time_har_components.py` checks the import leaves it unchanged. (2) `eg_attributions`, `shap_target_class` and `snip_scores_from_batches` (every gradient taken with the model in eval mode) run with `torch.backends.cudnn.flags(enabled=False)`: cuDNN's RNN backward only runs in training mode, so gradients through TinyHAR's LSTM in eval mode failed on CUDA. **Our method's loss (keep-score I(w), L_cons) must do the same** (go through `eg_attributions`, or wrap its own gradients in both `sdpa_kernel(SDPBackend.MATH)` and `cudnn.flags(enabled=False)`). No effect on CPU.

## Open issues
- On Windows, pandas writes CRLF line endings, so after `har.build_windows` git shows results/har-setup/ as modified even though the content is identical (`git diff --ignore-cr-at-eol` is empty). Restore with `git restore results/har-setup/`, or fix it with `lineterminator="\n"` or a `.gitattributes`.
- HART on CUDA warns that the backward pass of memory-efficient attention is non-deterministic: repeat runs of HART on GPU may not be bit-identical.
- Run inside `.venv`: the system Python on Nakhoul's laptop has no numpy.
- HAR70+: at least 15 pt of older subjects' walking/standing/sitting/lying windows are predicted as shuffling or stairs (older 4-class accuracy 76.6% with the 7-class argmax vs 91.6% restricted to the 4 classes; general 91.1 vs 97.3; merged Ustad scoring 84.6 vs 95.5). Subject 514 is an outlier (42.9% merged 4-class accuracy).
- Step 6 `--device cuda` path is untested (no GPU on Aya's laptop); the smoke run covered CPU only. If it fails on Nakhoul's machine, run with `--smoke` there first.
- Step 6 projection uses 8 × n_val SHAP windows for tuning (Park 4 + ours 4 configurations); plan Section 5 counts 360 for both, so the projection is slightly higher there.
