# Time

Wall-clock time of the important commands, run from the repo root. Each entry names the machine it ran on; the machines are described in [specs.md](specs.md). All runs are CPU only, except the model-choice training marked "GPU" below.

## Nakhoul's machine (Acer Swift X, Ryzen 7 5800U)

machine model: Acer Swift X (SFX14-41G)
commands ran: python -m lci.run_lci_phishing --seeds 0 1 2 3 4 --shap_seed 0
time taken to finish: 817.0 s (13 min 37 s); per seed 162.3 ± 7.8 s (152.9, 170.0, 170.8, 157.9, 159.9)

machine model: Acer Swift X (SFX14-41G)
commands ran: python -m lci.time_lci_phishing_components --seed 0
time taken to finish: 141.6 s

Per-stage breakdown from that command (one seed, CPU only, SHAP cache bypassed):

| Stage | Time | Share |
|---|---|---|
| Data load + preprocess + split | 0.04 s | 0.0% |
| Teacher training (30 epochs, 7,738 rows) | 8.1 s | 5.7% |
| Compression, all 6 methods (KD training is 1.8 s; PTQ and pruning are under 0.02 s each) | 1.9 s | 1.3% |
| SHAP GradientExplainer, 7 models (300 samples × nsamples 200; 17.0–19.9 s each) | 131.4 s | 92.8% |
| LCI computation, 6 comparisons | 0.2 s | 0.1% |

The full run makes 8 SHAP calls per seed (7 here, plus the SHAP-noise call), so SHAP accounts for about 150 of the 162 s per seed.

## Aya's machine (Lenovo ThinkPad E14 Gen 5, Core i7-1355U)

Measured 2026-11-18, fresh setup (empty `data/raw/`).

machine model: Lenovo ThinkPad E14 Gen 5 (21JLS20R00)
commands ran: python data/download.py
time taken to finish: 158.1 s (2 min 38 s), from scratch: HARTH zip 296 MB in about 126 s and HAR70+ zip 42 MB in about 19 s (both through the curl fallback), plus the Phishing fetch, extraction and verification. Download time depends mostly on the network, not on the machine.

machine model: Lenovo ThinkPad E14 Gen 5 (21JLS20R00)
commands ran: python data/verify_checksums.py
time taken to finish: 1.4 s (44/44 files OK)

machine model: Lenovo ThinkPad E14 Gen 5 (21JLS20R00)
commands ran: python -m pytest lci/tests/test_lci.py
time taken to finish: 0.9 s (12/12 passed; pytest reports 0.19 s for the tests themselves)

machine model: Lenovo ThinkPad E14 Gen 5 (21JLS20R00)
commands ran: python -m lci.time_lci_phishing_components --seed 0 (2026-11-19, run to compare the two machines)
time taken to finish: 194.6 s summed over stages (214.8 s wall clock incl. imports)

| Stage | Aya's machine | Nakhoul's machine | Ratio |
|---|---|---|---|
| Teacher training (30 epochs) | 17.1 s | 8.1 s | 2.1× |
| Compression, all 6 methods | 3.1 s | 1.9 s | 1.6× |
| SHAP GradientExplainer, 7 models | 174.1 s (10.1–30.5 s each) | 131.4 s | 1.32× |
| Total | 194.6 s | 141.6 s | 1.37× |

Nakhoul's machine is about 2× faster at training and 1.3× faster at SHAP on this workload. [plan/HAR_IMPLEMENTATION.md](../plan/HAR_IMPLEMENTATION.md) Section 5 uses these ratios to estimate HAR run times and to split folds between the machines.

machine model: Lenovo ThinkPad E14 Gen 5 (21JLS20R00)
commands ran: python -m har.build_windows --force (2026-11-19, HAR data pipeline, plan Steps 1–2)
time taken to finish: 28.6 s wall clock. Of that, 20.7 s is loading 40 CSVs (8.7M rows), resampling S006 and cutting 143,820 windows, and about 4 s is folds, normalisation statistics and the summary CSVs; the rest is imports.

machine model: Lenovo ThinkPad E14 Gen 5 (21JLS20R00)
commands ran: python -m har.build_windows (from the window cache)
time taken to finish: 4.3 s

machine model: Lenovo ThinkPad E14 Gen 5 (21JLS20R00)
commands ran: python -m pytest har/tests lci/tests
time taken to finish: 17.6 s wall clock (49/49 passed; pytest reports 16.6 s, mostly the real-data tests that re-derive every window of 6 subjects from the raw CSVs)

machine model: Lenovo ThinkPad E14 Gen 5 (21JLS20R00)
commands ran: python -m har.build_windows (2026-11-19, from the window cache, now including explain windows and SHAP backgrounds)
time taken to finish: 4.2 s wall clock (selecting the 1,200 explain windows and 5 × 100 background windows takes well under 1 s)

machine model: Lenovo ThinkPad E14 Gen 5 (21JLS20R00)
commands ran: python -m pytest har/tests lci/tests
time taken to finish: 8.7 s wall clock (61/61 passed; pytest reports 8.1 s, with the raw CSVs already in the OS file cache)

machine model: Lenovo ThinkPad E14 Gen 5 (21JLS20R00)
commands ran: python -m pytest har/tests lci/tests (2026-11-19, with the model-choice tests)
time taken to finish: 17.1 s wall clock (114/114 passed; pytest reports 15.8 s)

machine model: Lenovo ThinkPad E14 Gen 5 (21JLS20R00)
commands ran: python -m har.run_model_selection --stage all --device cpu --smoke
time taken to finish: 31.1 s wall clock for 25 units (20 tuning + 5 final; 1 epoch on 300 / 200 / 200 windows each). This is a pipeline check only, not a timing of the real model choice.

machine model: Lenovo ThinkPad E14 Gen 5 (21JLS20R00)
commands ran: python -m har.time_model_selection_cpu --smoke, then python -m har.select_model --smoke
time taken to finish: 8.4 s and 3.9 s wall clock (smoke protocol: 20 latency calls, 4 SHAP windows at nsamples 20)

machine model: Lenovo ThinkPad E14 Gen 5 (21JLS20R00)
commands ran: python -m pytest har/tests lci/tests, then the three smoke commands again (2026-11-19, after the models were checked against the papers)
time taken to finish: tests 30.4 s (117/117 passed); run_model_selection --smoke 42.0 s, time_model_selection_cpu --smoke 10.3 s, select_model --smoke 4.8 s

Per-window CPU cost of the source-checked candidates: paper sizes, width 1.0; scratch benchmark on random windows, 10 threads, 2026-11-19. Measured with torch 2.15.0 and again after the switch to torch 2.7.1, which is 1.5–1.8× faster on this CPU. The cost order is the same in both versions:

| Model | Parameters | Training, ms per window (2.7.1 / 2.15.0) | SHAP (1 output, nsamples 200, background 100), s per window (2.7.1 / 2.15.0) |
|---|---|---|---|
| DeepConvLSTM | 458,183 | 4.00 / 5.94 | 0.70 / 1.27 |
| CNN1D (HARTH) | 1,254,919 | 3.44 / 5.26 | 0.45 / 0.81 |
| HART | 1,389,555 | 1.16 / 2.04 | 0.22 / 0.39 |
| TinyHAR | 54,273 | 0.34 / 0.61 | 0.08 / 0.15 |

machine model: Lenovo ThinkPad E14 Gen 5 (21JLS20R00), torch 2.7.1+cpu (2026-11-19, after the switch from torch 2.15.0)
commands ran / time taken to finish:
- `python -m har.build_windows --force` (with the 20 Hz low-pass): 35.5 s wall clock (27.5 s loading, resampling, filtering and windowing)
- `python -m pytest har/tests lci/tests`: 30.5 s (119/119 passed)
- `python -m har.run_model_selection --stage all --device cpu --smoke`: 36.4 s; `python -m har.time_model_selection_cpu --smoke`: 12.1 s; `python -m har.select_model --smoke`: 6.3 s
- `python -m lci.run_lci_phishing --seeds 0 1 2 3 4 --shap_seed 0`: **489.4 s** (8 min 9 s); per seed 97.0 ± 8.9 s (110.9, 100.3, 89.8, 92.3, 91.6). It took 817 s with torch 2.15.0 on Nakhoul's machine (above), so the two numbers differ in both machine and torch version.
- `python -m har.run_model_selection --stage all --device cpu --models xgboost --machine-name aya-laptop` (git 473b75d, 10 threads): **8.7 min wall clock** (21:55:52–22:04:33; 9 units). Tuning, fold 0 (4 settings): 227 s (40–76 s per setting). Final folds 0–4: 292 s (50–68 s per fold, mean 58 s; 253–569 trees after early stopping).

machine model: Lenovo ThinkPad E14 Gen 5 (21JLS20R00), torch 2.7.1+cpu (2026-11-20)
commands ran / time taken to finish:
- `python -m pytest har/tests`: 96.1 s (113/113 passed)
- `python -m har.run_sanity_walking_aid --smoke`: 3.5 s
- `python -m har.run_sanity_walking_aid` (Step 5, evaluation only): **30.0 s** wall clock; 28.6 s for the rerun with the merged Ustad 4-class scoring (5 class sets) and `har/tests/test_stats.py` 7/7 in 12.3 s. The 5 frozen TinyHAR fold models predict 42,248 HARTH test windows (each subject once) and 5 × 17,403 HAR70+ windows on CPU; the rest is metrics and 24 bootstraps of 10,000 resamples.
- Step 6 code (2026-11-20, baseline parts of `har/compression.py` + `har/time_har_components.py`): `python -m pytest har/tests lci/tests` **76.7 s** wall clock (139/139 passed; 13 new in `test_compression.py` and `test_attributions.py`); `python -m har.time_har_components --fold 0 --smoke --machine-name Aya`: **28.5 s** wall clock (19.6 s inside the script). Smoke sizes only, so no timing numbers from it.
- After the two GPU fixes (cuDNN off in the attribution gradients, no `CUDA_VISIBLE_DEVICES` change at import): `python -m pytest har/tests lci/tests` **101.9 s** wall clock (141/141 passed; 2 new in `test_time_har_components.py`, each a fresh interpreter); `--smoke` again **51.8 s** wall clock (31.0 s inside the script), run straight after the test suite. The smoke components were 1.4–2.6× slower than in the first smoke run; tiny sizes and machine load, not a code effect (cuDNN flags do nothing on CPU).
- After the third GPU fix (cuDNN off in `snip_scores_from_batches` too): `python -m pytest har/tests lci/tests` **169.9 s** wall clock (141/141 passed); `--smoke` **42.1 s** wall clock (25.7 s inside the script).

### Step 6 timing benchmark (GPU runs done: nakhoul-3050, uni-2080; CPU runs pending)

Commands in [../STATUS.md](../STATUS.md). The per-component CSVs and projections go to `results/har-component-timing/`. One rule for every run: **our method's fine-tuning cost = "estimated = Park's cost"**. Our method (keep-score, L_cons, gate) is not implemented yet, so the projection charges its 80 fine-tuning epochs and its targets at Park's measured cost per epoch and marks the row `measured = False`. Its keep-score (EG m = 8 on 1,000 windows, once per level) is not in the projection.

## Nakhoul's machine, GPU (RTX 3050 Ti Laptop, 4 GB), torch 2.7.1+cu118

Measured 2026-11-19, machine name `nakhoul-3050`, git 473b75d.

machine model: Acer Swift X (SFX14-41G), NVIDIA GeForce RTX 3050 Ti Laptop GPU
commands ran / time taken to finish:
- `pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118`: about 3 min (2.8 GB wheel at about 30 MB/s, then install); the other packages under 2 min (xgboost 102 MB at 1.3 MB/s)
- `python data/verify_checksums.py`: under 2 s (44/44 files OK)
- `python -m har.build_windows` (from the window cache): 30.9 s wall clock (the script reports 20.2 s)
- `python -m pytest har/tests lci/tests`: 54.1 s (119/119 passed)
- `python -m har.run_model_selection --stage all --device cuda --smoke --machine-name nakhoul-3050`: 36.7 s
- `python -m har.run_model_selection --stage all --device cuda --models cnn1d tinyhar --machine-name nakhoul-3050`: **21.6 min wall clock** (21:59:37–22:21:15; 18 units, 30-epoch cap, early stopping with patience 5)

| Part | CNN1D | TinyHAR |
|---|---|---|
| Tuning, fold 0 (4 settings each) | 400 s (59–145 s per setting) | 187 s (28–59 s) |
| Final folds 0–4 (chosen setting) | 474 s (63–141 s per fold, mean 95 s) | 226 s (32–63 s, mean 45 s) |
| Seconds per epoch (51k training windows) | width 1.0: 9.3–11 s; width 0.5: 4–5.5 s | width 1.0: about 4.2 s; width 0.5: about 3 s |
| Total | 14.6 min | 6.9 min |

On this GPU, TinyHAR at width 1.0 takes about 2.5× less time per epoch than CNN1D at width 1.0 (CPU benchmark above: 10×). Plan Section 5's CPU upper bounds for the model choice on Nakhoul's machine (CNN1D 0.4 h, TinyHAR 2.1 h) were not needed.

- `python -m har.run_model_selection --stage all --device cuda --models hart deepconvlstm --machine-name nakhoul-3050` (2026-11-19/20, git 2c64f45, clean tree, NVIDIA driver 591.59): **83.2 min wall clock** (22:40:04–00:03:14; 18 units, 30-epoch cap, early stopping with patience 5). Exit 0. The first try with the system Python (not `.venv`) failed at once with `No module named 'numpy'`.

| Part | HART | DeepConvLSTM |
|---|---|---|
| Tuning, fold 0 (4 settings each) | 582 s (77–246 s per setting) | 952 s (93–308 s) |
| Final folds 0–4 (chosen setting) | 789 s (103–261 s per fold, mean 158 s) | 2,660 s (306–737 s, mean 532 s) |
| Seconds per epoch (51–56k training windows) | width 1.0: 9–11.7 s; width 0.5: 7.7–10.5 s | width 1.0: 25–32 s; width 0.5: 11.5–12.7 s |
| Epochs run per final fold | 11–28 | 11–25 |
| Total | 22.9 min | 60.2 min |

DeepConvLSTM at width 1.0 is the slowest candidate on this GPU: about 3× HART and 3× CNN1D per epoch. Its LSTM runs step by step over the 128 time steps, so the GPU helps it least. Plan Section 5's CPU upper bounds (DeepConvLSTM 5.4 h, HART 2.0 h on Nakhoul's machine) were not needed.

Step 6 timing benchmark on this GPU (2026-11-20, git 8b770af, clean tree):
- `python -m har.time_har_components --fold 0 --device cuda --smoke --machine-name nakhoul-3050`: 7.8 s wall clock
- `python -m har.time_har_components --fold 0 --device cuda --machine-name nakhoul-3050`: **186.6 s wall clock**. Measured: CE epoch 2.5 s, Park epoch 52.0 s (20.8× CE; plan Section 5 assumed 3×), Park targets 31.8 s, SNIP scores 0.7 s, SHAP 92.2 s for 660 windows (0.14 s each), inference 0.4 s for 26,105 windows. **Projected Steps 8–9 per fold: 3.20 h** (Park 1.16 h, ours = Park 1.16 h, SHAP 0.72 h, noise floor 0.09 h, L1+SNIP 0.06 h).

For comparison, uni-2080 (RTX 2080, 2026-11-20, git 52d9f98): 209.1 s wall clock; CE epoch 1.5 s, Park epoch 65.9 s (44.7× CE), SHAP 87.4 s; projected 3.73 h per fold. The 3050 Ti is slower on plain CE but faster on Park's double-backward epochs, so its projection comes out lower.

All five candidates' training is done (GPU, Nakhoul: 105 min for CNN1D, TinyHAR, HART, DeepConvLSTM; CPU, Aya: 8.7 min for XGBoost). The result rows hold `run_seconds`, `device` and `gpu_name`. CPU estimates are in [plan/HAR_IMPLEMENTATION.md](../plan/HAR_IMPLEMENTATION.md) Section 5. The CPU timing command (`python -m har.time_model_selection_cpu`, full protocol) is meant to run on Aya's machine and has not been run yet.

## Step 7 code: XGP + noise floor + pilot (Aya's machine, CPU, 2026-11-20)

machine model: Lenovo ThinkPad E14 Gen 5 (21JLS20R00), torch 2.7.1+cpu
commands ran / time taken to finish:
- `python -m pytest har/tests lci/tests`: **96 s** wall clock (176/176 passed; 35 new).
- `python -m har.run_noise_floor --folds 0 --smoke --machine-name Aya`: **24.8 s** wall clock (3 one-epoch retrainings 7.0 s, 18 SHAP calls on 30 windows at nsamples 20 8.2 s). Rerun with cached checkpoints and SHAP: under 10 s inside the script.
- `python -m har.run_xgp_pilot --fold 0 --smoke --machine-name Aya`: **33 s** wall clock (26.0 s inside the script; 8 one-epoch fine-tunings, both β gated to CE). With `--tau 0.99` (every XGP configuration takes CE + γ·L_cons): 49 s. In smoke, one L_cons epoch takes ≈ 3.7 s and one Park epoch ≈ 3.5 s on 600 windows, against ≈ 0.8 s for CE. That is consistent with L_cons costing about as much as Park.

Smoke sizes only, so these are no estimate of the real runs. GPU estimates for the real runs (from Step 6), as in plan Section 5:
- Noise floor, fold 0: ≈ 15 min on nakhoul-3050.
- Pilot, fold 0: ≈ 75 min upper bound on nakhoul-3050 (≈ 90 min on uni-2080).
- Steps 8 + 9 with the XGP ablations: ≈ 3.95 h per fold on nakhoul-3050 and ≈ 4.6 h on uni-2080.

## Step 8 noise floor + XGP pilot, fold 0 (nakhoul-3050 GPU, 2026-11-20/21, git c7787e2, clean tree)

- `python -m har.run_noise_floor --folds 0 --device cuda --machine-name nakhoul-3050`: **709 s** (23:36–23:48; estimate ≈ 15 min). Retraining seeds 1–3 104 s (26–44 s each, early stopping), 18 SHAP calls 603 s (90 / 120 / 540 windows, nsamples 200). On AC power.
- `python -m har.run_xgp_pilot --fold 0 --device cuda --machine-name nakhoul-3050`: 23:48–07:35, wall 28,005 s, of which ≈ 4.5 h hibernation (the laptop was unplugged at 01:58 and hibernated on critical battery at 02:56; resumed at 07:23). **Without the hibernation ≈ 3.3 h** (estimate was ≈ 75 min).

| Part (10 epochs each) | s / epoch (median) | s / run | Power |
|---|---|---|---|
| L1, SNIP (CE) | 9–12 | 85, 113 | AC |
| Park, 4 configurations (CE + γp·L_AM) | 158–207 | 1,546–2,065 | AC (last epochs of the 4th on battery) |
| XGP L_cons, β 0.3 γ 0.1 / γ 1, β 0.7 γ 0.1 | 105 | 1,047–1,060 | battery |
| XGP L_cons, β 0.7 γ 1 | 53–105 (rising) | not usable (epoch 1 = 16,134 s, hibernation) | AC |
| SHAP per assessment (90 windows) | | 14–32 | |
| References: teacher EG / SNIP / I(w) | | 40.0 / 0.9 / 10.1 | AC |

- Park's epoch here (≈ 206 s, AC) is ≈ 4× the Step 6 benchmark (52.0 s) on the same GPU, and epoch times varied up to 2× within one run (Park 112–207 s, L_cons 53–105 s). Laptop power/thermal state is not controlled, so these are not clean timings. The Step 6 projection (3.20 h/fold; 3.95 h with ablations) is likely too low; re-time Park and L_cons on a plugged-in, idle machine before planning Step 9.
- L_cons epochs (53–105 s, even on battery) were about half of Park's (158–207 s, on AC), so L_cons is cheaper than Park, not equal as Step 6 assumed.

## Step 8 noise floor, folds 1 and 3 (nakhoul-3050 GPU, 2026-11-21, git 75b5229, clean tree)

- `python -m har.run_noise_floor --folds 1 3 --device cuda --machine-name nakhoul-3050`: 21:11–21:46, **≈ 35 min** for both folds.
- Fold 1: **890 s** (training 160 s: seeds 1–3 40 / 44 / 73 s, 13–17 epochs; SHAP 728 s).
- Fold 3: **1,204 s** (training 440 s: seeds 1–3 86 / 270 / 82 s, 10 / 23 / 7 epochs, ≈ 9–12 s per epoch against ≈ 3 s in fold 1; SHAP 763 s). For comparison, fold 3 on uni-2080 took 690 s (training 93 s, SHAP 596 s).
- Per fold, the noise floor costs ≈ 12–20 min on nakhoul-3050, depending on early stopping and epoch speed.

## XGP pilot round 2 code (Aya's machine, CPU, 2026-11-21)

machine model: Lenovo ThinkPad E14 Gen 5 (21JLS20R00), torch 2.7.1+cpu
commands ran / time taken to finish:
- `python -m pytest har/tests lci/tests`: ≈ 90 s (195/195 passed; 19 new).
- `python -m har.run_xgp_pilot2 --prepare` (real, writes the pre-registration): ≈ 8 s.
- `python -m har.run_xgp_pilot2 --levels 0.7 0.9 --diagnostic --smoke`: **78 s** wall clock (60 smoke fine-tunings of 1–2 epochs, 16 starts, diagnostic). `--analyze --smoke`: 8 s.

Estimate of the real runs (one level per machine, from round 1's nakhoul-3050 timings: CE epoch ≈ 9 s, Park epoch 158–207 s on AC, SHAP 0.155–0.36 s per window; KD costs about as much as CE because the teacher logits are precomputed):

| Part, per level | Count | nakhoul-3050 |
|---|---|---|
| CE / KD fine-tunings, 10 epochs | 27 + `l1_kd_e20` (20 epochs) | ≈ 0.7 h |
| Park, park_kd (10 epochs, CE/KD + L_AM) | 2 | ≈ 0.9–1.15 h |
| SHAP, 600 windows per assessment (300 normal + 300 older-like) | 38 (8 starts + 30 runs) + original | ≈ 1.0 h (up to 2.3 h at round 1's slowest SHAP speed) |
| Teacher logits, Park targets, SNIP, 4 importances (≈ 6,000 window-passes at m = 8, 2 EG seeds) | once | ≈ 2 min |
| EG-noise diagnostic (EG m = 4, 16, 200 + SHAP seed 1 on 300 windows) | 70% machine only | ≈ 2 min |
| **Total** | | **≈ 2.8 h (up to ≈ 4.3 h)** |

uni-2080 (90%): ≈ 1.16× nakhoul-3050 (Step 6: 3.7 vs 3.2 h per fold) → **≈ 3.2 h (up to ≈ 5 h)**. Both on AC power, laptop sleep/hibernation off; `--resume` continues an interrupted run.

## XGP pilot round 2, 70% level (nakhoul-3050, GPU, 2026-11-21)

machine: nakhoul-3050, NVIDIA GeForce RTX 3050 Ti Laptop GPU, torch 2.7.1+cu118
command: `python -m har.run_xgp_pilot2 --levels 0.7 --diagnostic --device cuda --machine-name nakhoul-3050 --noise-floor-machine nakhoul-3050`
- **Wall clock 5,834 s (1.62 h)**, against the 2.8 h estimate. Level loop 5,603 s; started 15:27, finished 17:04.
- EG-noise diagnostic: ≈ 2 min. Setup: SHAP on the original 85 s, teacher logits + Park targets 38 s, SNIP 0.8 s, importances 7 / 9 / 15 / 16 s (plain / worstclass / aug / worstclass_aug).
- Fine-tuning total 0.60 h: CE/KD 10 epochs ≈ 30–36 s per run (≈ 3.3 s per epoch, faster than round 1's 9 s); l1_kd_e20 73 s; **Park ≈ 595 s, park_kd ≈ 606 s** (≈ 60 s per epoch, against 158–207 s in round 1).
- SHAP total 0.95 h over 38 assessments: ≈ 80–95 s each (600 windows, ≈ 0.14 s per window, round 1's fastest speed).
- The estimate was too high because CE and Park epochs both ran about 3× faster than in round 1.

## XGP pilot round 2, 90% level (uni-2080, GPU, 2026-11-21)

machine: uni-2080, GeForce RTX 2080, torch 2.7.1+cu118
command: `python -m har.run_xgp_pilot2 --levels 0.9 --device cuda --machine-name uni-2080 --noise-floor-machine nakhoul-3050`
- **Wall clock 5,114 s (1.42 h)**, against the 3.2 h estimate. Level loop 4,928 s; started 15:22, finished 16:47.
- Fine-tuning total 0.52 h: CE/KD 10 epochs ≈ 19–20 s per run (≈ 2 s per epoch); l1_kd_e20 40 s; Park ≈ 652 s, park_kd ≈ 655 s (≈ 65 s per epoch).
- SHAP total 0.84 h over 38 assessments (≈ 80 s each).
- `python -m har.run_xgp_pilot2 --analyze` (nakhoul-3050, after both levels): a few seconds.
- Round 2 in total: both levels ran in parallel on two machines, so ≈ 1.6 h of wall clock.

## Step 9A compression grid code (Aya's machine, CPU, 2026-11-21)

machine model: Lenovo ThinkPad E14 Gen 5 (21JLS20R00), torch 2.7.1+cpu
commands ran / time taken to finish:
- `python -m pytest har/tests lci/tests`: **61 s** (223/223 passed; 28 new).
- `python -m har.run_compression_grid --folds 0 --smoke --machine-name Aya`: **62 s** (38 models; 20 one-epoch fine-tunings on 600 windows, SHAP nsamples 20 on 30 windows). `--resume` with everything done: 1.5 s. `--analyze --smoke --n-boot 500`: 5 s.

Estimate of the real runs (from pilot round 2's GPU timings: CE/KD epoch ≈ 3.3 s on nakhoul-3050 and ≈ 2 s on uni-2080, Park epoch ≈ 60 / 65 s, SHAP ≈ 0.14 / 0.13 s per window):

| Part, per fold | Count | nakhoul-3050 | uni-2080 |
|---|---|---|---|
| CE / KD fine-tunings, 20 epochs | 16 | ≈ 18 min | ≈ 11 min |
| park, park_kd at 70 / 90%, 20 epochs | 4 | ≈ 80 min | ≈ 87 min |
| SHAP, ≈ 765 windows (90 inner val + 120–150 general + 540 older) | 32 models (original and noise references are cache hits) | ≈ 57 min | ≈ 54 min |
| Teacher logits + Park targets, faithfulness (66 forward passes per window), accuracy on ≈ 25K test windows | once / per model | ≈ 5 min | ≈ 5 min |
| **Total per fold** | | **≈ 2.7 h** | **≈ 2.6 h** |
| **Assigned folds** | 0, 1, 3 / 2, 3, 4 | **≈ 8 h** | **≈ 8 h** |

Upper bound ≈ 2× (round 1's slow epochs on a laptop on battery). `--analyze` with 10,000 bootstrap draws: minutes, not measured on real data yet.

## Step 9A compression grid, folds 0, 1, 3 (nakhoul-3050 GPU, 2026-11-21/22, git 9b228cd)

machine: nakhoul-3050, NVIDIA GeForce RTX 3050 Ti Laptop GPU, torch 2.7.1+cu118
command: `python -m har.run_compression_grid --folds 0 1 3 --device cuda --machine-name nakhoul-3050` (22:45), stopped on purpose at 01:50 during fold 1, resumed with `--resume` at 06:58, finished 11:50.
- Smoke runs first: `run_noise_floor --folds 0 --smoke` 23 s; `run_compression_grid --folds 0 --smoke` 59 s.
- **Fold 0: 2 h 34 min** (22:45–01:19).
- **Fold 1: ≈ 2 h 39 min** (01:19–01:50, then 06:59–09:08). The model in progress at the stop (`l1_kd-50pct-finetune`, ≈ 2 min) was redone.
- **Fold 3: 2 h 42 min** (09:08–11:50).
- **Total ≈ 7 h 55 min of compute** for 3 folds (estimate ≈ 8 h), i.e. ≈ 2.6 h per fold. The resumed run reported 17,480 s (4.86 h).
- Park / park_kd at 20 epochs: ≈ 63–65 s per epoch, ≈ 22 min per run, 4 runs per fold ≈ 1.5 h of each fold.

## XGP pilot round 3 code (LenovoP330-01, CPU, 2026-11-22)

machine: LenovoP330-01 (Intel Core, family 6 model 158), torch 2.7.1+cu118, all runs on CPU
commands ran / time taken to finish:
- `python -m pytest har/tests lci/tests`: **48 s** (244/244 passed; 19 new). After the rotation amendment: **58 s** (248/248).
- Smoke chain (`--smoke`): phase 1 ≈ 12 s per fold, phase 2 ≈ 5 s per fold, both analyses < 5 s. `--prepare` (real): ≈ 10 s.

Estimate of the real runs per fold, from pilot round 2's GPU timings (SHAP ≈ 80 s per assessed model on 300 + 300 windows; 20-epoch KD ≈ 40 s on uni-2080 and ≈ 70 s on nakhoul-3050; one importance on ≈ 1,000 windows ≈ 8–17 s):

| Part, per fold | Count | Estimate |
|---|---|---|
| Phase 1, no-fine-tuning starts (l1, snip, l1_recon once per level; xgp × 3 β × 3 seeds), 5 importance + reconstruction steps each | 30 assessed | ≈ 1 h |
| Phase 1, fine-tuned runs (l1, l1_kd, snip_kd, l1_kdaug × 2 r, xgp_ft × 6) × 3 seeds × 3 levels, curve at 2 / 5 / 20 epochs (aug runs have 1.25–1.5 × the steps) | 99 runs, 297 assessments | ≈ 8.5 h (≈ 5.6 h with `--curve-seeds 0`) |
| **Phase 1 total** | | **≈ 9.5 h** (≈ 6.6 h) |
| Phase 2: 4 starts × 3 seeds × 3 levels + 7 fine-tuned ablations × 3 seeds × 3 levels (epoch 20 only) | 36 + 63 | **≈ 3.5 h** |

Phase 1 runs with `--curve-seeds 0` (decided 2026-11-22), so ≈ 6.6 h per fold. The rotation adds well under a second per window set. Folds run in parallel (fold 0 on uni-2080, fold 1 on nakhoul-3050), so wall clock ≈ 6.6 h + 3.5 h on the slower machine. Not measured yet; replace with the real times after the runs.

### Fold 1 phase 1 on nakhoul-3050 (RTX 3050 Ti Laptop, cuda, 2026-11-22/23), complete
- Complete in 3 sessions, **≈ 10 h 08 min wall clock** in total: 15:20–20:07 (4 h 47 min, throttled, stopped by us), 21:11–01:50 (4 h 38 min, stopped by the overnight 01:50 cutoff), 09:38–10:21 (43 min, `phase tuning done in 2607.2 s`). All three used the same command; the second and third added `--resume`.
- Summed component time over the 135 runs: SHAP 313 min, fine-tuning 255 min, starts (importance + reconstruction steps) 34 min.
- Not throttled (sessions 2 and 3: GPU at 75–79 °C, about 1,600 MHz): 20-epoch fine-tuning 66–78 s (CE / KD) and 119–151 s (kdaug / xgp_ft); SHAP ≈ 92–97 s per assessment (300 + 300 windows); xgp start ≈ 57 s. The wall clock matched the summed per-run components (ratio 1.00–1.08).
- With a cool GPU throughout, the whole phase would take ≈ 7.5 h.

### Fold 1 phase 2 (ablation) on nakhoul-3050 (RTX 3050 Ti Laptop, cuda, 2026-11-23), complete
- `python -m har.run_xgp_pilot3 --phase ablation --folds 1 --device cuda --machine-name nakhoul-3050 --noise-floor-machine nakhoul-3050`, then the same with `--resume`.
- 2 sessions, **≈ 7 h 22 min wall clock**: 12:50–18:27 (5 h 37 min, slow; 40 / 63 fine-tuned runs; stop requested at 17:37, the log kept running to 18:27) and 19:15–21:00 (`phase ablation done in 6283.4 s`, 1 h 45 min; 23 fine-tuned runs + 8 starts).
- Summed component time over the 99 runs: fine-tuning 256 min, SHAP 153 min, starts (importance + reconstruction) 21 min.
- Session 1 (slow, likely throttled): 20-epoch fine-tuning 247–257 s (KD), 305–391 s (kdaug / DRO), 15.6–17.5 s per epoch; ≈ 8.5 min wall per fine-tuned run.
- Session 2 (fast, GPU ≈ 70 °C, 13–17 W): 4.1 s per epoch for 70% kdaug without DRO (82 s per run, vs 15.6 s per epoch in session 1, ≈ 4× faster); at 90% (r 0.5, 83,328 windows): 67–69 s (KD), 101–141 s (kdaug); ≈ 3 min wall per fine-tuned run.
- Planned ≈ 3.5 h; at session-2 speed the whole phase would take ≈ 3–3.5 h. Use uni-2080's times for the paper's cost comparison.

Session 1 details (written at the stop):
- `python -m har.run_xgp_pilot3 --prepare --smoke` + `--phase tuning --folds 1 --smoke --device cuda --machine-name nakhoul-3050`: **37 s** for the tuning smoke.
- `python -m har.run_xgp_pilot3 --phase tuning --folds 1 --curve-seeds 0 --device cuda --machine-name nakhoul-3050 --noise-floor-machine nakhoul-3050`: started 15:20:42, **stopped by us at 20:07** (GPU heat), 4 h 47 min. Done: level 50% and part of 70%; 18 / 30 no-fine-tuning starts and 34 / 99 fine-tuned runs (≈ 40% of phase 1). Continue with `--resume`.
- Original-model SHAP (300 + 300 windows): 82 s, the same as on uni-2080.
- Thermal throttling for the whole run: GPU at 88–92 °C (driver target 87 °C), SM clock 645–735 MHz of 2,100 MHz. Fine-tuning epochs took 10–13 s (≈ 200–260 s per 20-epoch run), about 3× the round-2 KD rate on this machine.
- Projected phase 1 on this laptop at this rate: ≈ 12 h (≈ 7–8 h left at the stop). Reported per-model times from this machine are throttled; use uni-2080's numbers for the paper's cost comparison.

## XGP pilot round 4 code (LenovoP330-01, CPU, 2026-11-23; amendment 1 the same day)

machine: LenovoP330-01, torch 2.7.1+cu118, CPU only
commands ran / time taken to finish:
- `python -m pytest har/tests lci/tests`: **45 s** (270/270 passed; 22 in `test_xgp4.py`).
- Smoke chain (`--smoke`, folds 0 and 1, amended design): `--prepare` 5 s; unstructured 66 s (≈ 33 s per fold); structured 34 s (≈ 17 s per fold); both analyses < 5 s; `--latency` 5 s.
- (First design, superseded by amendment 1: unstructured smoke 54 s, structured 29 s.)
- Read-only per-layer importance diagnostic for the design review (EG m = 8, one EG seed, 426 / 459 calibration windows per group): ≈ 11 s per fold.

Estimate of the real runs (amended design), from round 3's uni-2080 timings:
- SHAP: ≈ 90 s per assessment on 300 + 300 windows, so ≈ 45 s on the 300 real explain windows. Round 4 has no older-like copies.
- XGP start (K = 5 importance + reconstruction): ≈ 57 s. The importance has the same cost as round 3: ≈ 980 calibration windows in 15 per-group backward passes, against round 3's 14 per-class calls.
- L1 / LAMP iterative + reconstruction: a few s.
- 20-epoch KD: ≈ 40–50 s (≈ 55 s with DRO over 15 subjects; `xgp_r3` with round 3's aug ≈ 80 s).
- KD 5%: ≈ 5 s.

| Mode, per fold | Assessments | Estimate (uni-2080) |
|---|---|---|
| Unstructured: no fine-tuning (3 deterministic + 4 seeded cells + xgp_r3) × 3 levels | 54 | 0.9 h incl. starts |
| Unstructured: KD 100% (l1 + 6 cells + 3 DRO controls + xgp_r3) × 3 seeds × 3 levels | 99 | 2.7 h |
| Unstructured: curve 2 / 5 epochs (4 methods × 3 seeds × 90 / 95%) | 48 | 0.6 h |
| Unstructured: PTQ4 (4 methods × 3 seeds × 90 / 95%) + PTQ8 (90%, seed 0) | 28 | 0.35 h |
| Unstructured: KD 5% (11 methods × 3 seeds × 90 / 95%) | 66 | 0.9 h |
| **Unstructured total** | **295** | **≈ 6–6.5 h per fold** |
| Structured: no fine-tuning (3 deterministic + xgp_s × 3 seeds) × 2 levels | 12 | 0.2 h |
| Structured: KD 100% + KD 5% (7 methods incl. 3 DRO controls × 3 seeds × 2 levels each) | 84 | 1.7 h |
| Structured: curve (3 methods × 3 seeds × 2 levels × 2) + PTQ4 / PTQ8 at 75% | 36 + 12 | 0.6 h |
| **Structured total** | **144** | **≈ 2.5–2.7 h per fold** |
| **Per GPU (one unstructured fold + one structured fold)** | 439 | **≈ 8.5–9.2 h; nakhoul-3050 ≈ 11–12 h if it throttles (× 1.3)** |

SHAP is ≈ 60% of the time. Not measured yet; replace with the real times after the runs.
