# Requirements

Technical requirements for the project so far: data download and verification, the LCI reproduction on Phishing ([../plan/LCI_IMPLEMENTATION.md](../plan/LCI_IMPLEMENTATION.md)), and the HAR data pipeline (windows, folds, normalisation, explain windows, SHAP backgrounds; [../plan/HAR_IMPLEMENTATION.md](../plan/HAR_IMPLEMENTATION.md) Steps 1–2), and the HAR model choice (Step 4: candidates, tuning, final folds, CPU timing, selection). The model choice added xgboost; its training can also run on GPU machines (see [GPU setup](#gpu-setup-model-choice-on-gpu-machines)).

## Python and packages
- Python 3.11 (runs used 3.11.9, on both machines in [specs.md](specs.md)).
- Packages, with the exact versions the runs used. PyTorch is the CPU-only build on the laptops; GPU machines use the CUDA 11.8 build of the same torch version (below).
- **torch 2.7.1 since 2026-11-19** (was 2.15.0): the university GPU machine has CUDA 11.1 and its driver cannot be updated, and 2.7.1 is the newest torch with a CUDA 11.8 build. Nothing else had to change: after the reinstall `pip check` reports no broken requirements, and `pip freeze` differs only in torch. Every output made with torch 2.15 was regenerated with 2.7.1 (the Phishing LCI reproduction and the model-choice smoke run).
- The full pinned environment, including every transitive dependency, is in [requirements.txt](requirements.txt) (`pip freeze` of `.venv` on Aya's machine).

| Package | Version | Used by |
|---|---|---|
| torch | 2.7.1 (CPU; CUDA build `2.7.1+cu118` on GPU machines) | `lci/`, `har/` (models, training, compression) |
| shap | 0.51.0 | `lci/explain.py`, `har/attributions.py` (GradientExplainer) |
| xgboost | 3.2.0 | `har/train.py` (XGBoost accuracy reference in the model choice) |
| numpy | 2.4.6 | everything |
| pandas | 3.1.1 | `lci/`, `har/` (CSV loading, summaries), Phishing fetch in `data/download.py` |
| scikit-learn | 1.9.1 | `lci/data.py` (scaling, split), metrics |
| scipy | 1.17.1 | `har/data.py` (`decimate` for S006's 100 → 50 Hz, Butterworth 20 Hz low-pass), `har/features.py`; dependency of scikit-learn / shap |
| ucimlrepo | 0.0.7 | Phishing fetch in `data/download.py` |
| certifi | 2026.10.23 | CA-bundle fallback for the Phishing fetch in `data/download.py` (installed as a dependency of ucimlrepo) |
| pytest | 9.2.0 | `lci/tests/`, `har/tests/` |

The HARTH / HAR70+ part of `data/download.py` and `data/verify_checksums.py` use the standard library only.

If Python 3.11 is not installed (Windows), install it per user, alongside any other Python version:

```bash
winget install --id Python.Python.3.11 --version 3.11.9 --scope user --source winget
```

Install into a virtualenv at the repo root (`.venv/` is git-ignored). Either install the exact pinned environment:

```bash
py -3.11 -m venv .venv        # Windows; use `python3.11 -m venv .venv` on Linux/macOS
.venv/Scripts/activate        # Windows (Git Bash); use `source .venv/bin/activate` on Linux/macOS
pip install -r environment/requirements.txt
```

or the top-level packages only (pip resolves the rest):

```bash
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
pip install numpy==2.4.6 pandas==3.1.1 scikit-learn==1.9.1 scipy==1.17.1 shap==0.51.0 ucimlrepo==0.0.7 certifi==2026.10.23 pytest==9.2.0 xgboost==3.2.0
```

## GPU setup (model choice on GPU machines)

The model choice (`python -m har.run_model_selection`) can train on a CUDA GPU with `--device cuda`; everything else stays the same as above. Use the **same torch version (2.7.1)**, only the CUDA 11.8 build:

1. **Driver.** The university GPU machine reports CUDA 11.1 in `nvidia-smi`, and its driver cannot be updated. torch 2.7.1 is available as `+cu118` (checked on 2026-11-19 with `pip index versions torch --index-url https://download.pytorch.org/whl/cu118`; listed versions 2.0.0 to 2.7.1).
   - CUDA 11.x builds rely on CUDA's minor-version compatibility to run on a driver that reports an older 11.x, as long as it is ≥ 450.80.02 on Linux (≥ 452.39 on Windows). A CUDA 11.1 driver meets that.
   - Features that need PTX JIT compilation for newer GPU architectures may not work under that compatibility. So **verify with step 2's check and the GPU smoke run before the real run**.
2. Create the venv as above, then install the CUDA torch **first**, then the rest without torch:

   ```bash
   python3.11 -m venv .venv && source .venv/bin/activate          # Windows: py -3.11 -m venv .venv; .venv/Scripts/activate
   pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
   pip install numpy==2.4.6 pandas==3.1.1 scikit-learn==1.9.1 scipy==1.17.1 shap==0.51.0 ucimlrepo==0.0.7 certifi==2026.10.23 pytest==9.2.0 xgboost==3.2.0
   python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0)); print((torch.randn(256, 256, device='cuda') @ torch.randn(256, 256, device='cuda')).sum().item())"
   ```

   - **Expect** `2.7.1+cu118 11.8 True <GPU name>` and a number on the second line.
     - If `is_available()` is `False`, the driver is too old even for minor-version compatibility.
     - If the matrix product fails with "no kernel image is available", the GPU's architecture is not in the build.
     - In either case train on CPU instead (`--device cpu`).
   - Do not use `pip install -r environment/requirements.txt` here: it pins `torch==2.7.1+cpu`.
3. Data and window cache (identical on every machine, keyed by the pinned checksums):

   ```bash
   python data/download.py && python data/verify_checksums.py     # 44/44 files OK
   python -m har.build_windows                                    # data/processed/har/ cache; rewrites results/har-setup/ identically
   git status results/har-setup                                   # must show no changes
   python -m pytest har/tests lci/tests                           # 119 passed
   python -m har.run_model_selection --stage all --device cuda --smoke   # 1-minute GPU check -> results/har-model-selection-smoke/
   ```

- **XGBoost on GPU:** `--device cuda` is passed to XGBoost too. The pip wheel's CUDA support depends on the platform (on Linux it also pulls in an NCCL package), so if XGBoost fails on the GPU, run it with `--device cpu --models xgboost`: it trains on CPU in minutes, and the device is not part of the settings hash, so the results merge normally.
- **Determinism on GPU:** the runner sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `cudnn.deterministic=True` and `torch.use_deterministic_algorithms(True, warn_only=True)`. GPU results are repeatable on the same GPU, but not bit-identical to CPU or to another GPU model; every result row records `device`, `gpu_name` and `machine_name`.
- **Disk:** model-choice checkpoints go to `data/processed/har/checkpoints/har-model-selection/` (git-ignored): roughly 0.2 MB (TinyHAR), 1.8 MB (DeepConvLSTM), 5 MB (CNN1D) and 5.6 MB (HART) per model and run at the paper width, about 110 MB for the 36 deep runs.

## Network
- HTTPS access to `archive.ics.uci.edu` (one-time download, about 356 MB):
  - HARTH v1.2: `https://archive.ics.uci.edu/static/public/779/harth.zip` (311 MB)
  - HAR70+: `https://archive.ics.uci.edu/static/public/780/har70.zip` (44 MB)
  - Phishing Websites (UCI ID 327): fetched through the `ucimlrepo` API (about 0.8 MB)
- If Python's default CA store is stale (seen on both Windows machines), `data/download.py` retries with curl (HAR archives) or certifi's CA bundle (Phishing). TLS verification stays on, and SHA-256 checks guard the content.

## Disk
- About 1.4 GB free for `data/raw/` (355 MB of zip archives plus the extracted CSVs, plus 0.8 MB for Phishing).
- About 1.1 GB for `.venv/` (CPU PyTorch and the packages above).
- About 0.45 GB for `data/processed/har/` (git-ignored window cache written by `python -m har.build_windows`: 143,820 windows × 6 × 128 float32), and about 1 GB of RAM while it is built or loaded.

## Data integrity
- All 44 files must match the SHA-256 hashes pinned in `data/checksums.sha256`: 2 archives, 40 subject CSVs (22 HARTH, 18 HAR70+) and the 2 Phishing CSVs (`phishing/features.csv`, `phishing/targets.csv`).
- Check with `python data/verify_checksums.py`.

## Commands run
Run from the repo root, in this order, inside the `.venv` above. Wall-clock times, per machine, are in [time.md](time.md).

Nakhoul's machine:

```bash
python data/download.py                    # HARTH, HAR70+, Phishing -> data/raw/, verify SHA-256
python data/verify_checksums.py            # 44/44 files OK
python -m pytest lci/tests/test_lci.py     # 12 LCI unit tests
python -m lci.run_lci_phishing --seeds 0 1 2 3 4 --shap_seed 0   # LCI grid -> results/lci-phishing-reproduction/
python -m lci.time_lci_phishing_components --seed 0              # per-stage timing (prints only)
```

Nakhoul's machine, GPU setup and model choice (2026-11-19, RTX 3050 Ti Laptop, machine name `nakhoul-3050`). The venv had `torch 2.15.0+cpu` before this; it was replaced by the CUDA build of 2.7.1:

```bash
git pull                                   # at 473b75d
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118   # replaces 2.15.0+cpu
pip install numpy==2.4.6 pandas==3.1.1 scikit-learn==1.9.1 scipy==1.17.1 shap==0.51.0 ucimlrepo==0.0.7 certifi==2026.10.23 pytest==9.2.0 xgboost==3.2.0
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"   # 2.7.1+cu118 True NVIDIA GeForce RTX 3050 Ti Laptop GPU
python data/verify_checksums.py            # 44/44 files OK
python -m har.build_windows                # results/har-setup/ identical in content; git shows it as modified only because pandas writes CRLF on Windows (git diff --ignore-cr-at-eol is empty), so it was restored with git restore
python -m pytest har/tests lci/tests       # 119/119 passed
python -m har.run_model_selection --stage all --device cuda --smoke --machine-name nakhoul-3050   # GPU check -> results/har-model-selection-smoke/
python -m har.run_model_selection --stage all --device cuda --models cnn1d tinyhar --machine-name nakhoul-3050   # real run -> results/har-model-selection/
git pull                                   # at 2c64f45 (XGBoost results from Aya's machine merged)
.venv/Scripts/activate                     # needed: the system Python has no numpy, so a run outside the venv fails at once
python -m har.run_model_selection --stage all --device cuda --models hart deepconvlstm --machine-name nakhoul-3050   # 2026-11-19/20, real run, 83.2 min -> results/har-model-selection/
```

Aya's machine (setup, 2026-11-18):

```bash
winget install --id Python.Python.3.11 --version 3.11.9 --scope user --source winget
py -3.11 -m venv .venv
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install numpy==2.4.6 pandas==3.1.1 scikit-learn==1.9.1 scipy==1.17.1 shap==0.51.0 ucimlrepo==0.0.7 pytest==9.2.0
pip freeze                                 # -> environment/requirements.txt
python data/download.py                    # 44/44 files OK (curl and certifi TLS fallbacks both used)
python data/verify_checksums.py            # 44/44 files OK
python -m pytest lci/tests/test_lci.py     # 12/12 passed
python -m lci.time_lci_phishing_components --seed 0   # 2026-11-19, machine speed comparison (prints only)
python -m har.build_windows --force        # 2026-11-19, HAR windows, folds, normalisation -> data/processed/har/ (cache) + results/har-setup/
python -m har.build_windows                # same, from the cache
python -m pytest har/tests lci/tests       # 49/49 passed (37 HAR data-pipeline tests + 12 LCI tests)
python -m har.build_windows                # 2026-11-19, + explain windows and SHAP backgrounds -> results/har-setup/
python -m pytest har/tests lci/tests       # 61/61 passed (49 HAR tests incl. 12 explain/background tests + 12 LCI tests)
pip install xgboost                        # 2026-11-19, -> xgboost 3.2.0 (added to requirements.txt)
python -m pytest har/tests lci/tests       # 114/114 passed (102 HAR tests incl. 53 model-choice tests + 12 LCI tests)
python -m har.run_model_selection --stage all --device cpu --smoke   # model-choice pipeline check -> results/har-model-selection-smoke/
python -m har.time_model_selection_cpu --smoke                       # CPU timing check -> same folder
python -m har.select_model --smoke                                   # merge + rule + freeze check -> same folder
```

Source check (2026-11-19): the papers were copied into `papers/` (git-ignored, copyrighted PDFs). Their text was extracted for checking with `pypdf` in a separate scratch environment, so nothing was added to `.venv` or `requirements.txt`. After the model fixes:

```bash
python -m pytest har/tests lci/tests       # 117/117 passed
python -m har.run_model_selection --stage all --device cpu --smoke   # rerun with the source-checked models
python -m har.time_model_selection_cpu --smoke
python -m har.select_model --smoke
```

The commands above ran with torch 2.15.0. On 2026-11-19 the project switched to torch 2.7.1 (see Python and packages), with the 20 Hz low-pass and the cost-based selection rule added:

```bash
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu   # replaces 2.15.0+cpu; pip check: no broken requirements
pip freeze                                 # -> environment/requirements.txt (only torch changed)
python -m har.build_windows --force        # cache rebuilt with the 20 Hz low-pass; 9/10 results/har-setup/ files byte-identical
python -m pytest har/tests lci/tests       # 119/119 passed
python -m har.run_model_selection --stage all --device cpu --smoke   # smoke outputs regenerated with torch 2.7.1
python -m har.time_model_selection_cpu --smoke
python -m har.select_model --smoke
python -m lci.run_lci_phishing --seeds 0 1 2 3 4 --shap_seed 0       # Phishing LCI reproduction regenerated with torch 2.7.1
python -m har.run_model_selection --stage all --device cpu --models xgboost --machine-name aya-laptop   # real run, 8.7 min -> results/har-model-selection/
```

The full model choice (tuning + 5 folds) is run on GPU machines; its exact commands are in [../plan/HAR_IMPLEMENTATION.md](../plan/HAR_IMPLEMENTATION.md) Step 4.

Step 5, baseline gap check on HAR70+ (2026-11-20, Aya's laptop, torch 2.7.1+cpu; first use of HAR70+). Needs the 5 TinyHAR final checkpoints `data/processed/har/checkpoints/har-model-selection/har-final-tinyhar-fold{0..4}-nakhoul-3050.pt` (copied from Nakhoul's GPU machine). No new packages: `har/stats.py` uses scipy's `mannwhitneyu`.

```bash
python -m pytest har/tests                  # 113/113 passed (incl. 6 new tests in har/tests/test_stats.py)
python -m har.run_sanity_walking_aid --smoke   # 300 windows per split, 200 resamples (output folder deleted afterwards)
python -m har.run_sanity_walking_aid           # real run, CPU, 30 s -> results/har-baseline-gap/
# rerun the same day after making Ustad's merged 4-class scoring primary [Ustad Sec 2.4]:
python -m pytest har/tests/test_stats.py      # 7/7 passed
python -m har.run_sanity_walking_aid --smoke  # output folder deleted afterwards
python -m har.run_sanity_walking_aid          # CPU, 28.6 s -> results/har-baseline-gap/ (all files rewritten)
```

Step 6, timing benchmark of the frozen model (2026-11-20, Aya's laptop, torch 2.7.1+cpu). New code: the baseline parts of `har/compression.py` (`prunable_tensors`, `ptq_symmetric`, `snip_scores`, `eg_attributions`, `segment_attributions`, Park's `park_teacher_attributions` / `park_am_loss`) and `har/time_har_components.py`. Our method (keep-score, L_cons, gate) is not written yet. No new packages. `eg_attributions` forces PyTorch's math SDPA kernel (`torch.nn.attention.sdpa_kernel(SDPBackend.MATH)`): the fused attention kernels have no double backward, so Park's loss failed on TinyHAR's attention without it. Needs `data/processed/har/checkpoints/har-model-selection/har-final-tinyhar-fold0-*.pt`.

```bash
python -m pytest har/tests lci/tests                                    # 139/139 passed
python -m har.time_har_components --fold 0 --smoke --machine-name Aya   # -> results/har-component-timing-smoke/ (git-ignored)
# real runs (not run yet), one job per machine at a time:
python -m har.time_har_components --fold 0 --device cpu  --threads 10 --machine-name aya-laptop     # Aya
python -m har.time_har_components --fold 0 --device cpu  --threads 8  --machine-name nakhoul-3050   # Nakhoul
python -m har.time_har_components --fold 0 --device cuda --smoke      --machine-name nakhoul-3050   # Nakhoul GPU smoke, 7.8 s (2026-11-20)
python -m har.time_har_components --fold 0 --device cuda              --machine-name nakhoul-3050   # Nakhoul GPU, run 2026-11-20, 186.6 s, git 8b770af
```

Step 7 code, XGP + noise floor (Step 8) + fold-0 pilot (2026-11-20, Aya's laptop, torch 2.7.1+cpu). No new packages. New code:
- `har/xgp.py`: keep-score, I(w), L_cons, gate, tuning rule, ablation registry.
- `har/compression.py`: `l1_scores`, `prune_by_score`, `size_record`, `finetune`, `teacher_segment_attributions`, `attribution_grad_context`.
- `har/attributions.py`: SHAP cache in `data/processed/har/shap-cache/`, per-subject LCI.
- `lci/metric.py`: `aggregate_time_segments`, `compute_lci_by_subject`.
- `har/run_noise_floor.py`, `har/run_xgp_pilot.py`.

Every attribution gradient runs with the math SDPA kernel and cuDNN off (`compression.attribution_grad_context`). Both runners need `data/processed/har/checkpoints/har-model-selection/har-final-tinyhar-fold0-*.pt`.

```bash
python -m pytest har/tests lci/tests                                        # 176/176 passed (96 s)
python -m har.run_noise_floor --folds 0 --smoke --machine-name Aya           # -> results/har-noise-floor-smoke/ (git-ignored)
python -m har.run_xgp_pilot --fold 0 --smoke --machine-name Aya              # -> results/har-xgp-pilot-smoke/ (git-ignored)
python -m har.run_xgp_pilot --fold 0 --smoke --machine-name Aya --tau 0.99   # same, forces the CE + gamma L_cons branch
# real runs, nakhoul-3050 GPU, one after the other (2026-11-20/21, git c7787e2, exit 0 both):
python -m har.run_noise_floor --folds 0 --device cuda --machine-name nakhoul-3050   # -> results/har-noise-floor/ (709 s)
python -m har.run_xgp_pilot --fold 0 --device cuda --machine-name nakhoul-3050      # -> results/har-xgp-pilot/ (reads fold 0's tau; 7.8 h incl. 4.5 h hibernation)
python -m har.run_noise_floor --folds 1 3 --device cuda --machine-name nakhoul-3050   # 2026-11-21, git 75b5229, nakhoul-3050 (torch 2.7.1+cu118); fold 1 890 s, fold 3 1,204 s
```

XGP pilot round 2 code (2026-11-21, Aya's laptop, torch 2.7.1+cpu). No new packages. New code:
- `har/augment.py`: older-like copies of HARTH windows (time-stretch s ~ U(0.8, 0.9), movement scaling a ~ U(0.7, 0.9) around the per-channel mean).
- `har/compression.py`: the shared KD fine-tuning recipe (`KD`, `kd_kl_term`, `kd_loss`, `predict_logits`, `finetune(kd_teacher_logits=...)`).
- `har/xgp.py`: `XGP2`, `classwise_alignment_importance`, `worst_class_importance`, `choose_beta`, `select_final_xgp` (the pre-registered rule).
- `har/run_xgp_pilot2.py` (`--prepare`, `--levels`, `--diagnostic`, `--analyze`, `--resume`, `--smoke`); tests in `har/tests/test_xgp_round2.py`.

```bash
python -m pytest har/tests lci/tests                                   # 195/195 passed
python -m har.run_xgp_pilot2 --prepare --smoke                         # -> results/har-xgp-pilot-smoke/round2/ (git-ignored)
python -m har.run_xgp_pilot2 --levels 0.7 0.9 --diagnostic --smoke     # 78 s
python -m har.run_xgp_pilot2 --analyze --smoke                         # 8 s
python -m har.run_xgp_pilot2 --prepare                                 # -> results/har-xgp-pilot/round2/ pre-registration (run 2026-11-21, before any GPU run)
# real runs, one per machine; uni-2080 reads tau_0 and ceiling_0 from nakhoul-3050's fold-0 noise floor:
python -m har.run_xgp_pilot2 --levels 0.7 --diagnostic --device cuda --machine-name nakhoul-3050 --noise-floor-machine nakhoul-3050   # ran 2026-11-21 on nakhoul-3050 (torch 2.7.1+cu118, git 7dd23b6), 1.62 h
python -m har.run_xgp_pilot2 --levels 0.9 --device cuda --machine-name uni-2080 --noise-floor-machine nakhoul-3050   # ran 2026-11-21 on uni-2080 (torch 2.7.1+cu118, git 7dd23b6), 1.42 h
python -m har.run_xgp_pilot2 --analyze                                 # ran 2026-11-21 on nakhoul-3050 after both levels -> final XGP = xgp_kd, beta* 0.1, E_ft 20 in Step 9
```

Step 9A code, compression grid with the standard methods only (2026-11-21, Aya's laptop, torch 2.7.1+cpu). No new packages. New code:
- `har/run_compression_grid.py`: the grid (method registry, `--folds`, `--resume`, `--smoke`, `--analyze`).
- `har/faithfulness.py`: deletion curve, random-deletion baseline, AOPC and drop_k5 (plan 2.7).
- `har/compression.py`: `ptq_tulla` (= `lci.compression.ptq`).
- `har/stats.py`: `paired_wilcoxon`, `holm`, `cohens_dz`, `bootstrap_mean_ci`, `bootstrap_fa_lci_ci`, `safe_rate`; `bootstrap_group_gap_ci` vectorised for the mean (same draws, same result).
- Tests: `har/tests/test_compression_grid.py`, `har/tests/test_faithfulness.py`, additions to `har/tests/test_stats.py`.

Needs on each machine: `data/processed/har/checkpoints/har-model-selection/har-final-tinyhar-fold{k}-*.pt`, the fold's noise-floor files from the same machine in `results/har-noise-floor/` (τ_k and the SHAP-noise ceilings), and, for the per-class retrain-noise reference and SHAP cache hits, that machine's `data/processed/har/checkpoints/har-noise-floor/` and `data/processed/har/shap-cache/`.

```bash
python -m pytest har/tests lci/tests                                        # 223/223 passed (61 s)
python -m har.run_compression_grid --folds 0 --smoke --machine-name Aya      # -> results/har-compression-grid-smoke/ (git-ignored), 62 s
python -m har.run_compression_grid --folds 0 --smoke --machine-name Aya --resume   # skips all 38 models
python -m har.run_compression_grid --analyze --smoke --n-boot 500            # -> results/har-compression-grid-analysis-smoke/
# GPU runs (to do), one job per machine:
python -m har.run_noise_floor --folds 0 --smoke --device cuda --machine-name nakhoul-3050      # smoke noise floor for this machine name
python -m har.run_compression_grid --folds 0 --smoke --device cuda --machine-name nakhoul-3050
python -m har.run_compression_grid --folds 0 1 3 --device cuda --machine-name nakhoul-3050   # ran 2026-11-21/22 (git 9b228cd, torch 2.7.1+cu118); stopped at 01:50 during fold 1, then:
python -m har.run_compression_grid --folds 0 1 3 --device cuda --machine-name nakhoul-3050 --resume   # 06:58-11:50, exit 0
python -m har.run_noise_floor --folds 0 --smoke --device cuda --machine-name uni-2080
python -m har.run_compression_grid --folds 0 --smoke --device cuda --machine-name uni-2080
python -m har.run_compression_grid --folds 2 3 4 --device cuda --machine-name uni-2080
python -m har.run_compression_grid --analyze                                                   # after both machines' files are merged; fold 3 = uni-2080's copy
```

XGP pilot round 3 code (2026-11-22, LenovoP330-01, torch 2.7.1+cu118, CPU only). No new packages. New and changed code:
- `har/xgp3.py`: `ROUND3`, `STARTS`, `COMPONENTS`, `alignment_importance_ref`, `group_class_importance`, `masked_keep_scores`, `layerwise_reconstruct` / `ridge_refit_rows` / `refit_modules`, `iterative_prune`, `select_config`, `select_final`, `best_baseline`.
- `har/run_xgp_pilot3.py` (`--prepare`, `--phase tuning|ablation`, `--folds`, `--machine-name`, `--curve-seeds`, `--analyze`, `--resume`, `--smoke`, `--use-machine`).
- `har/augment.py` (amendment): `OLDER_LIKE_R3`, `draw_rotation_params`, `draw_older_like_r3_params`, `rotation_matrices`, `rotate_sensors`, `older_like_r3`, `older_like_windows_r3`; round 2's functions unchanged.
- `har/compression.py`: `head_unpruned`; in `finetune`, group DRO (`groups`, `dro_eta`) and `on_epoch_end` (RNG states saved and restored around it).
- `har/run_compression_grid.py`: the head check is now `head_unpruned` (no mask, no zeroed weight) and a `head_check` column is added; `--recheck-head` recomputes the column of finished folds from their checkpoints.
- Tests: `har/tests/test_xgp_round3.py` (23).

```bash
python -m pytest har/tests lci/tests                                   # 244/244 passed (48 s); after the rotation amendment 248/248 (58 s)
python -m har.run_xgp_pilot3 --prepare --smoke                         # -> results/har-xgp-pilot-smoke/round3/ (git-ignored)
python -m har.run_xgp_pilot3 --phase tuning --folds 0 --curve-seeds 0 --smoke --machine-name smoke-cpu      # 12 s (fold 1 the same)
python -m har.run_xgp_pilot3 --analyze --phase tuning --smoke
python -m har.run_xgp_pilot3 --phase ablation --folds 0 --smoke --machine-name smoke-cpu    # 5 s (fold 1 the same)
python -m har.run_xgp_pilot3 --analyze --phase ablation --smoke
python -m har.run_compression_grid --folds 0 --smoke --machine-name smoke-cpu --noise-floor-machine nakhoul-3050 --levels 0.7 --no-ptq   # 16 s, head_unpruned_all true (outputs deleted afterwards)
python -m har.run_compression_grid --folds 0 --recheck-head --smoke --machine-name uni-2080   # 28 smoke rows, all True
python -m har.run_xgp_pilot3 --prepare                                 # -> results/har-xgp-pilot/round3/ pre-registration (run 2026-11-22, before any GPU run; deleted and rerun the same day after the rotation amendment, and the smoke chain was rerun too)
# GPU runs (to do): fold 0 on uni-2080, fold 1 on nakhoul-3050; tau_k and the ceiling come from nakhoul-3050's noise floor (in git)
python -m har.run_xgp_pilot3 --phase tuning --folds 0 --smoke --device cuda --machine-name uni-2080
python -m har.run_xgp_pilot3 --phase tuning --folds 0 --curve-seeds 0 --device cuda --machine-name uni-2080 --noise-floor-machine nakhoul-3050
python -m har.run_xgp_pilot3 --prepare --smoke                         # on nakhoul-3050 first (the smoke pre-registration is local), 2026-11-22
python -m har.run_xgp_pilot3 --phase tuning --folds 1 --smoke --device cuda --machine-name nakhoul-3050                                       # DONE 2026-11-22, 37 s
python -m har.run_xgp_pilot3 --phase tuning --folds 1 --curve-seeds 0 --device cuda --machine-name nakhoul-3050 --noise-floor-machine nakhoul-3050   # DONE 2026-11-23 (3 sessions, the 2nd and 3rd with --resume)
python -m har.run_xgp_pilot3 --analyze --phase tuning                  # after both folds' files are merged
python -m har.run_xgp_pilot3 --phase ablation --folds 0 --device cuda --machine-name uni-2080 --noise-floor-machine nakhoul-3050
python -m har.run_xgp_pilot3 --phase ablation --folds 1 --device cuda --machine-name nakhoul-3050 --noise-floor-machine nakhoul-3050
python -m har.run_xgp_pilot3 --analyze --phase ablation                # -> decision.json
python -m har.run_compression_grid --folds 0 1 3 --recheck-head --machine-name nakhoul-3050   # 9A head column fix, on nakhoul-3050
python -m har.run_compression_grid --folds 2 3 4 --recheck-head --machine-name uni-2080       # on uni-2080
```

### XGP pilot round 4 (final round; plan 2.5 (b), "Round 4 pre-registration" with amendment 1; code 2026-11-23, LenovoP330-01 CPU)
- Code: `har/xgp4.py`, `har/run_xgp_pilot4.py`; tests `har/tests/test_xgp4.py` (22). Round 3's files are only imported, never changed.
- Amendment 1 (same day, before `--prepare`):
  - no synthetic older-like data;
  - XGP groups = subjects (default of `xgp4.default_groups`);
  - worst-group importance (k = 3);
  - group DRO for the XGP candidates, plus `*_kd_dro` controls for every baseline;
  - validation on real inner-val windows only (LCI_val, worst-subject LCI_val, F1_val).
- No new packages (torch 2.7.1, numpy, pandas, scipy as before).

```bash
python -m pytest har/tests lci/tests                                   # 270/270 passed (45 s)
# CPU smoke (LenovoP330-01; old-design smoke outputs and smoke checkpoints deleted first):
python -m har.run_xgp_pilot4 --prepare --smoke                         # -> results/har-xgp-pilot-smoke/round4/ (git-ignored), 5 s
python -m har.run_xgp_pilot4 --mode unstructured --folds 0 1 --smoke   # 66 s
python -m har.run_xgp_pilot4 --mode structured --folds 0 1 --smoke     # 34 s
python -m har.run_xgp_pilot4 --analyze --mode unstructured --smoke
python -m har.run_xgp_pilot4 --analyze --mode structured --smoke
python -m har.run_xgp_pilot4 --latency --smoke                         # 5 s
# --- GPU runs (to do), in this order ---
# 1. round 3 must be decided first (nakhoul-3050):
python -m har.run_xgp_pilot3 --phase ablation --folds 1 --device cuda --machine-name nakhoul-3050 --noise-floor-machine nakhoul-3050
python -m har.run_xgp_pilot3 --analyze --phase ablation                # -> round3 decision.json; commit it, then: git tag xgp-round3-final
# 2. round-4 pre-registration (CPU, any machine, after the tag; refuses without it):
python -m har.run_xgp_pilot4 --prepare                                 # -> results/har-xgp-pilot/round4/ prereg + tables; commit, pull on both GPUs
# 3. optional GPU smoke (needs round 3's smoke decision on that machine: the round-3 smoke chain above, then):
python -m har.run_xgp_pilot4 --prepare --smoke
python -m har.run_xgp_pilot4 --mode unstructured --folds 0 --smoke --device cuda --machine-name uni-2080
# 4. real runs, one unstructured fold + one structured fold per GPU (each GPU runs the unstructured fold it ran in round 3):
#    uni-2080
python -m har.run_xgp_pilot4 --mode unstructured --folds 0 --device cuda --machine-name uni-2080 --noise-floor-machine nakhoul-3050
python -m har.run_xgp_pilot4 --mode structured --folds 1 --device cuda --machine-name uni-2080 --noise-floor-machine nakhoul-3050
#    nakhoul-3050
python -m har.run_xgp_pilot4 --mode unstructured --folds 1 --device cuda --machine-name nakhoul-3050 --noise-floor-machine nakhoul-3050
python -m har.run_xgp_pilot4 --mode structured --folds 0 --device cuda --machine-name nakhoul-3050 --noise-floor-machine nakhoul-3050
# 5. CPU latency (Aya's laptop; optionally each GPU machine's CPU), any time after step 2:
python -m har.run_xgp_pilot4 --latency --machine-name Aya
# 6. after both machines' files are merged:
python -m har.run_xgp_pilot4 --analyze --mode unstructured             # -> har-xgp-pilot4-tinyhar-unstructured-decision.json
python -m har.run_xgp_pilot4 --analyze --mode structured               # -> har-xgp-pilot4-tinyhar-structured-decision.json
```

## Bit-reduction test (XGQ, explanation-guided mixed-precision quantization; separate folder `bit-reduction-test/`)
Self-contained (own package `bitred`, results in `bit-reduction-test/results/`); reuses har/ and lci/ read-only; HARTH only. Design and rules: `bit-reduction-test/README.md`.
```
cd bit-reduction-test
python -m pytest tests                                              # 19/19 passed (7 s, nakhoul-3050 laptop CPU)
python -m bitred.run --prepare --smoke                              # -> results-smoke/ (git-ignored)
python -m bitred.run --folds 0 --smoke --device cpu --machine-name smoke-cpu     # 26 s
python -m bitred.run --folds 0 --smoke --device cuda --machine-name smoke-cuda   # 18 s
python -m bitred.run --analyze --smoke
python -m bitred.run --prepare                                      # done 2026-11-24 -> results/bitred-tinyhar-preregistration.json (commit before the runs)
# --- GPU runs (to do) ---
python -m bitred.run --folds 0 1 2 3 4 --device cuda --machine-name nakhoul-3050
#   or split: --folds 0 1 2 on nakhoul-3050, --folds 3 4 on uni-2080 (--machine-name uni-2080)
python -m bitred.run --analyze                                      # after merging
# --- 2-bit gap test (done, fold 0) and ultra-small gap test (QAT + KD at 15.6-17.5 KB; README "Ultra-small gap test") ---
python -m bitred.run_qat --fold 0 --device cuda --machine-name nakhoul-3050
python -m pytest tests                                              # 24/24 passed (8 s, LenovoP330-01 CPU)
python -m bitred.run_qat_small --fold 0 --smoke --device cpu --machine-name smoke-cpu     # 14 s
python -m bitred.run_qat_small --fold 0 --device cuda --machine-name uni-2080             # done 2026-11-24 (13.7 min)
python -m pytest tests                                              # 27/27 passed (9 s, LenovoP330-01 CPU)
python -m bitred.run_qat_regimes --fold 0 --smoke --device cpu --machine-name smoke-cpu  # 14 s
python -m bitred.run_qat_regimes --fold 0 --device cuda --machine-name <m>               # done 2026-11-25 on uni-2080 (24.5 min)
# --- final comparison: XGP (xgp_xb) vs lamp_recon at 90 / 95 % + 4-bit QAT, full / kd5 / calibration, folds 0-4 (README) ---
python -m pytest tests                                              # 31/31 passed (11 s, LenovoP330-01 CPU)
python -m bitred.run_xgp_compare --folds 0 1 2 3 4 --smoke --device cpu --machine-name smoke-cpu   # ≈ 25 s per fold
python -m bitred.run_xgp_compare --folds 0 --smoke --resume --device cpu --machine-name smoke-cpu # resume check
python -m bitred.run_xgp_compare --analyze --smoke
python -m bitred.run_xgp_compare --folds 0 1 --device cuda --machine-name uni-2080         # done 2026-11-25 (2.7 h, both commands at once on uni-2080)
python -m bitred.run_xgp_compare --folds 2 3 4 --device cuda --machine-name uni-2080       # done 2026-11-25 (3.6 h)
python -m bitred.run_xgp_compare --analyze                                                 # done: 90%|kd5 XGP beats LAMP (only cell)
# --- final test on HAR70+ (README "Final test on HAR70+"; pre-registered 2026-11-25; run once) ---
python -m pytest tests                                              # 38/38 passed (12 s, LenovoP330-01 CPU); har/tests 254/254
python -m bitred.run_final_test --phase validation --folds 0 1 --smoke --device cpu --machine-name smoke-cpu   # 1.8 min
python -m bitred.run_final_test --analyze-validation --smoke
python -m bitred.run_final_test --phase test --folds 0 1 --smoke --device cpu --machine-name smoke-cpu         # 1.8 min
python -m bitred.run_final_test --analyze --smoke                                                              # ≈ 10 s
# GPU (to do). Optional on nakhoul-3050 first: copy uni-2080's data/processed/checkpoints/bit-reduction-test/xgp-compare/fold{0,1}-*.pt
# so the seed-0 lamp_recon / xgp_xb starts are the comparison's own (otherwise rebuilt with the same code; start_source says which)
python -m bitred.run_final_test --phase validation --folds 0 1 --device cuda --machine-name nakhoul-3050      # done 2026-11-25 (4.0 h; starts rebuilt, not copied)
python -m bitred.run_final_test --phase validation --folds 2 3 4 --device cuda --machine-name uni-2080        # ≈ 5 h
#   merge results/har-final-test-ultrasmall/validation/ from both machines, then on one machine:
python -m bitred.run_final_test --analyze-validation                # writes validation-complete.json; commit / pull it on both machines
python -m bitred.run_final_test --phase test --folds 0 1 --device cuda --machine-name nakhoul-3050            # ≈ 3.6 h
python -m bitred.run_final_test --phase test --folds 2 3 4 --device cuda --machine-name uni-2080              # ≈ 6.9 h
python -m bitred.run_final_test --analyze                           # after merging; -decision.json (H1-H3)
# --resume continues either phase; each fold's test must run on the machine (and --machine-name) of its validation
```
