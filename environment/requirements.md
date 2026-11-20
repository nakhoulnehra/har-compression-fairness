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
