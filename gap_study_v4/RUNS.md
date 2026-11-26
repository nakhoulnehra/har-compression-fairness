# Gap study v4: run log

**Setup.** Ansel (RTX 3050 Ti Laptop 4 GB, Ryzen 7 5800U), torch 2.7.1+cu118, `.venv`, repo root,
`$env:PYTHONPATH = "bit-reduction-test"`. Real runs are started by `gap_study_v4/power_watch.ps1` (v3's watcher,
copied unchanged: waits for AC, stops workers after 3 min on battery, resumes with `--resume`, keeps Windows awake).
Protocol = v3's (`gap_study_v3.confirm`, imported unchanged): HARTH inner val, 50 explain windows per subject, 3
subjects per fold, GradientExplainer nsamples 200, 48 features; primary metric `lci_indep` (model SHAP seed 1 vs the
original's seed 0).

## Selection rules (written 2026-11-26 13:20, after the launch and before any v4 fine-tuned result was read; two
v4 no-retraining rows, RIA folds 0 / 3 seed 0, were visible)
1. **y (best existing method in the scenario)** = the highest mean `lci_indep` (5 folds x all seeds) among the
   scenario's existing methods (v3 + v4 rows).
2. **x (achievable at the same size)** = a training-based reference at the same masks as y. For each reference, the
   lr is chosen by v3's rule: the highest mean `lci_indep` (seed 0, 5 folds) among the lrs whose mean F1 is within
   1 pt of that reference's best-F1 lr. x must keep accuracy: mean accuracy >= y's - 0.5 pt.
3. **Gap kept** only if x - y >= 0.05, the fold-paired 95 % CI of x - y excludes 0, x - y > 0 in >= 4/5 folds, and
   occlusion LCI moves the same way (not one explainer only).
4. **Baseline checks.** If any existing method in the scenario (RIA, blended calibration, ...) closes >= half of
   x - y, the gap is dropped (the baseline was not the strongest).
5. **Fine-tuned 90 % cell (v3's second gap).** If function-matching KD (existing, Beyer et al. 2022) beats v3's
   lamp_recon_kd lr 3e-4 by >= 0.05, that cell's baseline was weak and its gap is re-measured with fmkd as B.
6. One held-out test run of the selected cell's y and x models (HARTH test subjects + HAR70+), after selection.

| # | Date / time | Command | Config | Status | Result path |
|---|---|---|---|---|---|
| S1 | 2026-11-26 13:08 | `python -m gap_study_v4.run_v4 --folds 0 --stages ref nr4 ft4 --smoke --threads 4 --out <scratchpad>/smoke-v4` | smoke sizes (600 training windows, 3 explain windows / subject, nsamples 20, 1 FT epoch) | pass, 59 s | scratchpad (not kept) |
| S2 | 2026-11-26 13:13 | `python -m gap_study_v4.test_confirm --folds 0 --smoke --out <scratchpad>/smoke-v4/test --models B=... X=...` + `--analyze` | smoke checkpoints, 9A smoke explain windows (3 / subject), nsamples 20 | pass (pipeline only; numbers not read) | scratchpad (not kept) |
| V1 | 2026-11-26 13:11 | watcher `-Jobs gap_study_v4/jobs-validation.json`: 5 workers `python -u -m gap_study_v4.run_v4 --folds <k> --stages ref ft4 --device cpu --ft-device cuda --threads 2 --worker ft --machine-name Ansel --resume`, 2 CPU workers `--folds 0 1 2` / `--folds 3 4 --stages ref nr4 --threads 3 --worker nr` | ft4: fmkd 80 % (from agent B's l1_fullrecon seed-0 checkpoint) lr 3e-4 / 1e-4; kd 80 % lr 3e-4; calkd 80 % lr 1e-4 / 3e-4; fmkd 90 % (from lamp_recon seed 0) lr 3e-4 / 1e-4; all seed 0. nr4: ria_fullrecon, l1_fullrecon_mixcal at 80 %, seeds 0-2. **13:52:** fmkd 90 % lr 1e-4 dropped from the plan for time (function-matching fine-tunes take ≈ 36 min each with 5 on the GPU, not the ≈ 11 min estimated). **≈ 13:30–14:00:** the watcher process died silently (no log line); the workers are independent processes and kept running. Replaced its keep-awake role with `gap_study_v4/keep_awake.ps1` at 14:06 (no battery gating from then on) | nr4 done 13:25 (both workers); ft4 running | `results/gap-study-v4/validation/gap-study-v4-validation-tinyhar-fold<k>-Ansel-{ft,nr}-{models,per-subject-lci,config}.{csv,json}`; checkpoints `data/processed/har/checkpoints/gap-study-v4/`; attributions `data/processed/har/gap-study-v4/attributions/` |
| S3 | 2026-11-26 14:08 | `python -m gap_study_v4.run_v4 --folds 0 --stages cal4 --smoke --threads 2 --out <scratchpad>/smoke-v4b` | smoke | pass, 30 s | scratchpad (not kept) |
| V2 | 2026-11-26 14:09 | `python -u -m gap_study_v4.run_v4 --device cpu --ft-device cuda --machine-name Ansel --resume --threads 2 --worker cal --stages ref cal4 --folds 0 1 2 3 4` (Start-Process, hidden; no watcher) | calfm 80 % (label-free KD + function matching on the ≈ 980 calibration windows and their blends, 200 epochs) lr 1e-4 / 3e-4, seed 0. Added after reading fmkd's first results, as a scenario-boundary baseline: can a deployer with only the unlabeled calibration windows get x in minutes? | running | `...-fold<k>-Ansel-cal-{models,per-subject-lci,config}.*` |
| E1 | 2026-11-26 14:26 | — | **fold 2 ft worker crashed**: `RuntimeError: CUDA error: an illegal memory access` inside `finetune` during fmkd 80 % lr 1e-4 (6 processes on the 4 GB GPU after V2 started; 3.9 GB in use). Its finished model (fmkd lr 3e-4) is kept | restart queued | — |
| V3 | 2026-11-26 14:30 (armed; starts when fold 0's ft worker exits) | `gap_study_v4/chain_fold2.ps1` → `python -u -m gap_study_v4.run_v4 ... --worker ft --stages ref ft4b --folds 2 --resume` | ft4b = ft4 without fmkd 80 % lr 1e-4 (that run only feeds the lr rule; fold 2's lr comparison is then missing and is stated). A first launch at 14:28 fired early (process match bug) and was killed after loading data, before any model | queued | as V1 (`-ft-` files) |

## Decision point 2 (written 2026-11-26 15:22, after reading V1/V2 validation results; before V4)
Read so far (validation, 80 %, lci_indep, 5 folds): best no-retrain baseline l1_fullrecon_mixcal 0.773 / l1_fullrecon
0.760; calfm (label-free KD + function matching on the ≈ 980 calibration windows, lr 3e-4) **0.821** at −0.2 pt accuracy
vs l1_fullrecon; fmkd (full labelled data, same masks) 0.857 at +0.7 pt. At 90 %, fmkd 20 ep (3 folds) 0.782 ≈
lamp_recon_kd 0.781.
- calfm uses only the deployer's unlabeled calibration windows. Post-training methods that run gradient steps on
  calibration data (AdaRound, BRECQ) are standard, so calfm counts as an existing method of the calibration-only
  scenario. It closes 63 % of x − y, so by rule 4 **v3's strictly backprop-free framing is not kept as the main gap**.
- **Scenario now evaluated:** label-free, ≈ 1k unlabeled calibration windows, any compute (the deployer has the trained
  model and a small unlabeled sample, not the vendor's labelled training set). y = best existing method there
  (no-retrain methods, calkd, calfm with lr {1e-4, 3e-4, 1e-3}); x = best full-data same-mask reference (fmkd 20 or
  40 epochs, lr 3e-4) with accuracy >= y's − 0.5 pt. Rules 3 (gate) and 6 (one test run) unchanged.
- If this scenario fails rule 3, the report states that no gap passes the bar, with the backprop-free 80 % cell as
  the closest candidate and its calfm objection.
- The 90 % fine-tuned cell is not re-run further (fmkd does not beat lamp_recon_kd; no same-size evidence of headroom).

| # | Date / time | Command | Config | Status | Result path |
|---|---|---|---|---|---|
| K1 | 2026-11-26 15:22 | — | stopped folds 1 / 3 / 4 ft workers (they had started fmkd 90 % lr 1e-4, dropped at 13:52; nothing of it written), fold 0's ft worker (stalled ≈ 60 min in kd 80 % lr 3e-4 at low GPU use; nothing written) and the fold 2 restart chain (never fired) | done | — |
| V4 | 2026-11-26 15:24 | 5 processes `python -u -m gap_study_v4.run_v4 --device cpu --ft-device cuda --machine-name Ansel --resume --threads 2 --worker ft --stages ref x4 --folds <k>` (Start-Process, hidden) | x4: calfm lr 1e-3; calkd lr 1e-4 / 3e-4 (folds 0, 2 only; done elsewhere); **fmkd 80 % 40 epochs lr 3e-4**; kd 80 % lr 3e-4 (folds 0, 2); fmkd 90 % 20 ep lr 3e-4 (folds 0, 2); seed 0 | running | `...-fold<k>-Ansel-ft-*`; logs `v4-x4-f<k>-*.log` |
| V5 | 2026-11-26 15:31 | `python -u -m gap_study_v4.run_v4 --device cpu --ft-device cpu --machine-name Ansel --resume --threads 4 --worker calcpu --stages ref cal5 --folds 0 1 2 3 4` | lr-grid edge checks of the label-free baselines: calfm lr 3e-3 (lr 1e-3 was best, at the edge), calkd lr 1e-3 (3e-4 was best). **Trained on CPU** (GPU memory near the 4 GB limit with V4): numerically not bit-identical to GPU runs | running | `...-fold<k>-Ansel-calcpu-*` |

## Decision point 3 (written 2026-11-26 16:30, before any 90 % label-free run)
Read: 80 % label-free scenario fails rule 3. calfm lr {1e-4, 3e-4, 1e-3, 3e-3} = 0.801 / 0.821 / **0.835** / 0.819
(optimum inside the grid), calkd best 0.798 (lr 1e-3); x = fmkd 0.857 (20 ep; 40 ep fold 0 0.910 vs 0.908). x − y
= 0.022 < 0.05. **No gap at 80 % in the label-free scenario.**
New scenario (motivated by that result, disclosed): **90 % unstructured sparsity (6,821 weights, ≈ 26.6 KB fp32: fits
a ≤ 30 KB MCU budget), label-free, ≈ 1k unlabeled calibration windows, any compute.**
- y = best existing method there: no-retrain 90 % (v3: l1_fullrecon 0.637 / wanda / lamp_recon / taylor, accuracy
  −6 to −7 pt), calfm and calkd fine-tuned from agent B's 90 % seed-0 starts (lamp_recon, l1_fullrecon), lr grid
  {3e-4, 1e-3, 3e-3} for calfm from lamp_recon, lr 1e-3 for the others. Same rule: highest mean lci_indep.
- x = full-data same-size reference: v3's lamp_recon_kd lr 3e-4 (3 seeds × 5 folds, 0.781, accuracy 0.918) or
  fmkd 90 % lr 3e-4 (folds where it ran). Accuracy rule: x's accuracy >= y's − 0.5 pt.
- Accuracy clause: if y's accuracy drops > 1 pt vs the original, the cell counts only because x shows accuracy and
  LCI are recoverable together at that size (x's accuracy within 1 pt of the original).
- Rule 3 gate unchanged (x − y >= 0.05, fold-paired CI excludes 0, >= 4/5 folds positive, occlusion same direction),
  plus gradient × input checked as the third explainer. Then one test run (rule 6).

| # | Date / time | Command | Config | Status | Result path |
|---|---|---|---|---|---|
| K2 | 2026-11-26 16:26 | — | stopped fold 0's x4 worker after its fmkd 80 % 40-ep model (it had started kd 80 % lr 3e-4; nothing written). Fold 0 has no kd 80 % lr 3e-4 row (v3's lr 1e-4 row exists) | done | — |
| V6 | 2026-11-26 16:26 | 5 processes `python -u -m gap_study_v4.run_v4 --device cpu --ft-device cpu --machine-name Ansel --resume --threads 3 --worker lf90 --stages ref lf90 --folds <k>` | lf90 (decision point 3): calfm from lamp_recon 90 % lr 1e-3 / 3e-4 / 3e-3, calfm from l1_fullrecon 90 % lr 1e-3, calkd from lamp_recon 90 % lr 1e-3; seed 0; **trained on CPU** | running | `...-fold<k>-Ansel-lf90-*` |
