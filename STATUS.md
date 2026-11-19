# Status
Current step: 4 (model choice): sub-step 4f, full runs
Done: 1 setup, 2 plan, 3 data pipeline + explain windows + backgrounds; 4e Nakhoul GPU smoke run; 4f part 1: CNN1D + TinyHAR (tuning + folds 0–4) on Nakhoul's RTX 3050 Ti, 21.6 min, in results/har-model-selection/
Last commit: 473b75d
Next: 4f Nakhoul XGBoost on CPU (`--stage all --device cpu --models xgboost --machine-name nakhoul-3050`); uni GPU: HART + DeepConvLSTM; then 4g CPU timing + select_model

## Decisions since the plan (short)
- Selection rule: best macro-F1; among models within 2 pt, lowest CPU SHAP time; params break ties
- 20 Hz low-pass on both datasets
- FA-LCI = min(LCI_older, LCI_general) (λ = 0.5)
- Machine split for 4f: each machine runs `--stage all` (tuning + all 5 folds) for its own models (Nakhoul: CNN1D, TinyHAR, XGBoost), not the tune/final fold split in plan Step 4

## Open issues
- On Windows, pandas writes CRLF line endings, so after `har.build_windows` git shows results/har-setup/ as modified even though the content is identical (`git diff --ignore-cr-at-eol` is empty). Restore with `git restore results/har-setup/`, or fix it with `lineterminator="\n"` or a `.gitattributes`.
- HART on CUDA warns that the backward pass of memory-efficient attention is non-deterministic: repeat runs of HART on GPU may not be bit-identical.
