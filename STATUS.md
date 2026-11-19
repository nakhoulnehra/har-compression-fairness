# Status
Current step: 4 (model choice): sub-step 4g, CPU timing + selection
Done: 1 setup, 2 plan, 3 data pipeline + explain windows + backgrounds; 4e Nakhoul GPU smoke run; 4f all 5 candidates trained (tuning + folds 0–4) in results/har-model-selection/: CNN1D + TinyHAR (Nakhoul GPU, 21.6 min), XGBoost (Aya CPU, 8.7 min), HART + DeepConvLSTM (Nakhoul GPU, 83.2 min)
Last commit: 2c64f45 (HART + DeepConvLSTM results not yet committed)
Next: 4g. Commit + pull results/har-model-selection/, copy fold-0 checkpoints (data/processed/har/checkpoints/har-model-selection/har-final-*-fold0-*) to Aya's laptop; there run `python -m har.time_model_selection_cpu`, `python -m har.select_model --merge-only`, `python -m har.select_model`. Then plan Step 5 (walking-aid sanity check, first HAR70+ use)

## Decisions since the plan (short)
- Selection rule: best macro-F1; among models within 2 pt, lowest CPU SHAP time; params break ties
- 20 Hz low-pass on both datasets
- FA-LCI = min(LCI_older, LCI_general) (λ = 0.5)
- Machine split for 4f: each machine ran `--stage all` (tuning + all 5 folds) for its own models (Nakhoul GPU: CNN1D, TinyHAR, HART, DeepConvLSTM; Aya CPU: XGBoost), not the tune/final fold split in plan Step 4

## Open issues
- On Windows, pandas writes CRLF line endings, so after `har.build_windows` git shows results/har-setup/ as modified even though the content is identical (`git diff --ignore-cr-at-eol` is empty). Restore with `git restore results/har-setup/`, or fix it with `lineterminator="\n"` or a `.gitattributes`.
- HART on CUDA warns that the backward pass of memory-efficient attention is non-deterministic: repeat runs of HART on GPU may not be bit-identical.
- Run inside `.venv`: the system Python on Nakhoul's laptop has no numpy.
