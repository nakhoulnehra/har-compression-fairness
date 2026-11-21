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
