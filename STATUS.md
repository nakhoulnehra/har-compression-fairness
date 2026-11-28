# Status
Current step: **gap study v5 done (2026-11-28, `gap_study_v5/GAP_REPORT.md`, `GAP_SUMMARY.md` update)**: the 2026-11-27 gap (one-shot 80 %) does not survive the existing OBC + KL-chosen layer budgets (0.864 val / 0.858 test, = full-data fmkd; v4 baseline 0.773). v3's 90 % gap closes too (0.851 vs 0.781). No fixable between-method gap passes; the residual to the ceiling is 0.068 (80 %) / 0.081 (90 %), and no oracle exceeds the strongest existing combination by ≥ 0.05. Presentation 3 Dec. Earlier XGP / 9B plan below is history.
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

- **Round 3 progress:** phase 1 for both folds, the tuning selection and phase 2 for fold 0 are done (committed). Phase 2 for fold 1 and `decision.json` are still to do.
- **Round 4 (2026-11-23):**
  - Design review: `plan/round4-design-review.md`. Decisions taken on it:
    - B adopted (3 × 2: uniform / LAMP / XGP budgets × L1 / XGP selection, all K = 5 + reconstruction);
    - C adopted, conv filters only (`convs.0/3/6`; no LSTM units, head input unchanged), with baselines filter-L1 + recon and Taylor (Molchanov 2019) + recon, and CPU latency;
    - A dropped, except calibration on training windows;
    - no LSTM / attention reconstruction; XGP and LAMP budgets are capped so those tensors are never pruned more than under uniform;
    - D and E as proposed;
    - 9A-ext merged into 9B;
    - round-3 reference row `xgp_r3`.
  - Pre-registration written into the plan first (2.5 (b), "Round 4 pre-registration"), then code: `har/xgp4.py`, `har/run_xgp_pilot4.py`, tests `har/tests/test_xgp4.py`. Round 3's files are unchanged and only imported.
  - **Amendment 1 (2026-11-23, before `--prepare`)**: all synthetic older-like data removed from round 4. The reason: the proxy added nothing in round 3, and the method must not depend on one dataset's groups.
    - XGP is group-agnostic: an optional group label per window, with the subject as the default (15 training subjects per fold).
    - Calibration: ≤ 10 training windows per (subject, class) cell, ≈ 980 windows.
    - Importance: the mean of the k = 3 worst groups' layer-normalised I_g. I_g comes from one class-balanced alignment objective per group.
    - Fine-tuning: XGP candidates use KD + group DRO over the training subjects (Sagawa 2020, η 0.01); baselines use plain KD, and each baseline also gets a KD + DRO control (`l1_kd_dro`, `l1_recon_kd_dro`, `lamp_recon_kd_dro`; structured `*_kd_dro`).
    - Validation uses real inner-val windows only: F1_val on all of them; LCI_val and worst-subject LCI_val on round 3's explain windows.
    - Older vs general and walking-aid are evaluation-only, in 9B.
    - `xgp_r3` stays round 3's method as it was, including its own older-like data, and is scored with the round-4 metrics.
  - Rule: round 4 replaces round 3 per regime only if it **beats** `xgp_r3`. "Beats" means all of these: mean LCI_val gain ≥ 0.02; higher in ≥ 5 of 6 paired (fold, seed) runs; F1_val within 1 pt at every level; mean worst-subject LCI_val ≥ the other method's − 0.01. The same rule decides the "beats the best baseline" claim, and the DRO controls count as baselines.
  - 270/270 tests pass (22 in `test_xgp4.py`). The CPU smoke chain passes (both modes, both analyses, latency).

- **Low-bit diagnostic (2026-11-24; baselines only, no new method, no retraining):** asks whether, below ≈ 30 KB, the best existing compression silently breaks explanations or hurts one group more.
  - Models on the frozen TinyHAR: per-channel PTQ at 4 / 3 / 2 bits; round 4's `lamp_recon` at 90 / 95% in fp32 and + 4-bit; per-layer 2-bit sensitivity (13 tensors).
  - Validation on HARTH inner val, folds 0–4. Test as 9A: older vs general, walking-aid split, bootstrap CIs.
  - Code: `har/run_diagnostic_low_bit.py`. 281/281 tests pass; the CPU smoke and the analysis pass. Output: `results/har-diagnostic-low-bit/`.
  - Sizes: ptq_pc4 33.7 KB (just above 30 KB because of the fp32 scales and biases), ptq_pc3 27.2, ptq_pc2 20.7; lamp_recon 90 / 95% + 4-bit 16.9 / 15.6 KB with the bitmask.
  - GPU / CPU runs to do (commands in requirements.md, "Low-bit diagnostic"; ≈ 40 min per fold on a GPU, ≈ 25–30 min on Aya's CPU).

- **Bit-reduction test (2026-11-24, separate folder `bit-reduction-test/`):** XGQ, explanation-guided mixed-precision weight quantization (bits per tensor from measured LCI sensitivity) against ZeroQ-style KL and HAWQ-style Hessian allocation with the same knapsack, at 27.2 / 23.9 / 22.3 KB, plus a uniform 4/3/2-bit probe; HARTH inner val only, folds 0-4, no training. Pre-registered (`results/bitred-tinyhar-preregistration.json`), 19/19 tests and CPU + CUDA smoke pass. GPU run to do (≈ 2.5-3.4 h on one GPU; commands in requirements.md). Independent of commit a18e156's low-bit diagnostic. **Ultra-small gap test** (`bitred/run_qat_small.py`, pre-registered in the README): the 2-bit gap test's QAT + KD and rule at 17.5 KB (per-tensor 2-bit), 16.9 KB (lamp_recon 90 % + 4-bit per channel) and 15.6 KB (lamp_recon 95 % + 4-bit), sizes with bitmask, masks fixed, fold 0. Done on uni-2080: inconclusive at every size (no QAT run gets F1 within 3 pt of 0.818; the best, lamp_recon 90 % at lr 1e-3, F1 0.780 with LCI 0.889). **Data-regime gap test** (`bitred/run_qat_regimes.py`, pre-registered in the README 2026-11-25): the same QAT and rule on round 4's kd5 (2,583) and calibration (978) training windows, four settings (adds per-channel 2-bit), verdict per setting × regime; equal Adam steps across regimes (4,020, as the full-data runs; changed from equal epochs before any real run). Done on uni-2080: no gap at 20.7 KB with kd5 (channel 2-bit F1 0.795, LCI 0.856); every other cell inconclusive (F1 does not recover). **Final comparison** (`bitred/run_xgp_compare.py`, pre-registered in the README 2026-11-25): xgp_xb vs lamp_recon (round 4's functions and values) at 90 / 95 % + 4-bit per channel + QAT (4,020 steps, lr 1e-4 / 1e-3; primary lr = lamp_recon's better mean F1 per regime), regimes full / kd5 / calibration, folds 0-4, R2 verdict per level x regime; per-tensor 2-bit QAT as reference. Done on uni-2080 (all 5 folds, 2026-11-25): primary lr 1e-3 in every regime; **XGP beats LAMP only at 90 %|kd5** (LCI 0.790 vs 0.750, 4/5 folds, F1 +1.6 pt, worst subject +0.065); the other 5 cells do not pass R2. **Final test on HAR70+** (`bitred/run_final_test.py`, pre-registered in the README 2026-11-25 before any code; run once): H1 (primary) = one-sided paired Wilcoxon of per-subject test LCI (seed mean 0-2), xgp_xb > lamp_recon at 90 %|kd5, lr 1e-3, 40 test subjects, alpha 0.05; H2 = older-minus-general AOPC gap reduction (bootstrap CI); H3 (exploratory) = walking-aid d_accuracy gap (5 vs 13); other 5 cells Holm, seed 0; ablations l1_xb_recon / xgp_lamp, seeds 1-2, both lrs, LCI k x omega sensitivity as secondary. 9A test protocol unchanged; order enforced (validation all folds -> --analyze-validation -> test -> --analyze). 38/38 bitred tests, 254/254 har tests, CPU smoke chain pass. GPU runs to do (≈ 12 h wall split; commands in requirements.md).

- **Gap study v3 (2026-11-26, Ansel; `gap_study_v3/GAP_REPORT.md`; no new method):** confirmed gap = post-training pruning **without retraining at 80 %**: uncompressed LCI 0.932 vs best fairly tuned baseline 0.760 (l1_fullrecon, independent SHAP seeds), gap 0.172 (18.5 %), CI [0.07, 0.28], 5/5 folds × 3 seeds, HARTH test 15/22 and HAR70+ 12/18 subjects ≥ 0.10, accuracy −1 pt; cause: mask-induced re-routing of temporal evidence (cross-seed 0.917 vs 0.786; occlusion also drops). Second, weaker gap: 90 % + KD fine-tuning (tuned lamp_recon_kd 0.781, gap 0.151, 4/5 folds; test of the tuned baseline still open). Two earlier agent attempts (PAIR, GMR) did not beat the baselines; their files were removed, summary in `GAP_SUMMARY.md`.
- **Gap study v4 (2026-11-26, Ansel; `gap_study_v4/RUNS.md`):** same-mask references and more baselines. At 80 % one-shot, the best baseline (blended calibration) is 0.773 and fmkd at the same masks is 0.865 (+0.092, CI [0.027, 0.157], 5/5 folds; occlusion agrees). Label-free gradient training on the calibration windows (calfm) reaches 0.835, so the gap is scoped to backprop-free compression. 90 %: every method ≈ 0.78, no gap between methods.
- **Gap study v5 (2026-11-27 → 28, Ansel; `gap_study_v5/RUNS.md`, `GAP_REPORT.md`; no new method):** ran the backprop-free baselines v3 had listed as not run.
  - OBC + full refit reaches 0.817 (seed 0: 0.831), closing 63 % of v4's x − y.
  - A layer budget chosen by output KL on the calibration windows reaches 0.864 (3 seeds), the same as full-data fmkd. On the held-out test it is 0.858 vs fmkd 0.837 vs v4's mixcal 0.767.
  - At 90 %, fmkd from the KL-chosen allocation reaches 0.851 (v3's baseline 0.781).
  - Oracles: capacity 0.875 at 80 %, 0.798 at 90 %; gradient-field oracles fall below fmkd.
  - Ablations: the LSTM and conv0 carry most of the explanation cost, and the costs are super-additive; probe calibration does not help; sensor shifts open no larger gap. LCI tracks the calibration KL (Spearman −0.57 / −0.80), not accuracy.
  - Decision: no fixable between-method gap; residual to the ceiling 0.068 / 0.081.
- **Final decision + cleanup (2026-11-27, superseded by v5):** `GAP_SUMMARY.md` (gap, proof, where baselines fail, why fixable). Watcher scripts, jobs files, console logs and the v3 audit / v4 draft reports were deleted.

Next for the bit-reduction test: the final test on HAR70+ (validation on nakhoul-3050 folds 0 1 and uni-2080 folds 2 3 4, merge, `--analyze-validation`, then the test phase on the same machines, merge, `--analyze`; commands in requirements.md).

Next (exact commands in environment/requirements.md, round 4 block; times in environment/time.md):
1. nakhoul-3050: round-3 phase 2 for fold 1, `python -m har.run_xgp_pilot3 --phase ablation --folds 1 --device cuda --machine-name nakhoul-3050 --noise-floor-machine nakhoul-3050` (≈ 3.5 h; ≈ 4.5 h if throttled).
2. Merge, then `python -m har.run_xgp_pilot3 --analyze --phase ablation` → round-3 `decision.json`; commit, then `git tag xgp-round3-final` (we tag).
3. `python -m har.run_xgp_pilot4 --prepare` (CPU, seconds; refuses before the decision and the tag) → commit, pull on both GPUs.
4. uni-2080: `python -m har.run_xgp_pilot4 --mode unstructured --folds 0 --device cuda --machine-name uni-2080 --noise-floor-machine nakhoul-3050`, then `--mode structured --folds 1` (same flags). Estimate ≈ 6–6.5 h + ≈ 2.5–2.7 h ≈ 8.5–9.2 h.
5. nakhoul-3050: `python -m har.run_xgp_pilot4 --mode unstructured --folds 1 --device cuda --machine-name nakhoul-3050 --noise-floor-machine nakhoul-3050`, then `--mode structured --folds 0` (same flags). Estimate ≈ 8.5–9.2 h; ≈ 11–12 h if throttled. A GPU smoke first needs round 3's smoke chain on that machine (requirements.md).
6. Aya's laptop: `python -m har.run_xgp_pilot4 --latency --machine-name Aya` (about a minute).
7. `python -m har.run_xgp_pilot4 --analyze --mode unstructured` and `--mode structured` → decisions. Structured enters 9B only if finished and decided by 27 Nov morning.
8. 9A head column: `python -m har.run_compression_grid --folds 0 1 3 --recheck-head --machine-name nakhoul-3050` and `--folds 2 3 4 --recheck-head --machine-name uni-2080`.
9. Register the adopted XGP (round 3 or 4 per regime), plus the 9A-ext baselines (l1_recon, LAMP + recon, 95%, KD 5%, PTQ4 on pruned models, structured baselines if structured is in), in `run_compression_grid` → 9B.

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
- Gap study v5: the OBC, DSnoT, SPDY, OWL and EagleEye PDFs are not in `papers/`; verify the cited sections. The KL-chosen allocation is a coarse 7-point grid (SPDY / OWL-style criterion, not either exact method).
- Round 4: the LAMP, Li et al. 2017 and Molchanov et al. 2019 PDFs are not in `papers/` (the folder is absent on LenovoP330-01); verify the cited sections before submission. Structured pruning at 50% removes only ≈ 15% of the parameters (the convs are 22.6% of the weights) but ≈ 49% of the MACs; batch-1 CPU latency may not drop (smoke: ≈ 1.0 ms dense and sliced).
- Round 4's `xgp_r3` at 80% / 95% uses round 3's 90% configuration (nearest level, pre-registered); round 3 never ran there.
- Round 4 (amendment 1):
  - β = 0.1 comes from round 3's tuning with the older-like group score; it is kept fixed (not retuned) for the subject-group score.
  - Worst-subject LCI_val is a minimum over only 3 inner-val subjects per fold, which is noisy, so it is a non-inferiority guard (−0.01), not the primary metric.
  - `xgp_r3` still uses round 3's older-like copies internally (it is round 3's frozen method).
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
