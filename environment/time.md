# Time

Wall-clock time of the important commands, run from the repo root. Each entry names the machine it ran on; the machines are described in [specs.md](specs.md). All runs are CPU only.

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

The full model choice is meant to run on GPU machines; record its times here per machine and GPU (the result rows hold `run_seconds`, `device` and `gpu_name`). CPU estimates for it are in [plan/HAR_IMPLEMENTATION.md](../plan/HAR_IMPLEMENTATION.md) Section 5. The CPU timing command (`python -m har.time_model_selection_cpu`, full protocol) is meant to run on Aya's machine and has not been run yet.
