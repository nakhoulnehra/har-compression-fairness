# Status
Current step: 4 (model choice): sub-step 4b, torch 2.7.1 switch
Done: 1 setup, 2 plan, 3 data pipeline + explain windows + backgrounds
Last commit: <hash>
Next: 4c commit, 4d uni GPU smoke run, 4e Nakhoul GPU smoke run, 4f full runs, 4g CPU timing + select_model

## Decisions since the plan (short)
- Selection rule: best macro-F1; among models within 2 pt, lowest CPU SHAP time; params break ties
- 20 Hz low-pass on both datasets
- FA-LCI = min(LCI_older, LCI_general) (λ = 0.5)

## Open issues
- (none)