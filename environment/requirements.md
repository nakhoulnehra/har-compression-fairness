# Requirements

Technical requirements for the project so far: data download and verification, the LCI reproduction on Phishing ([../plan/LCI_IMPLEMENTATION.md](../plan/LCI_IMPLEMENTATION.md)), and the HAR data pipeline (windows, folds, normalisation, explain windows, SHAP backgrounds; [../plan/HAR_IMPLEMENTATION.md](../plan/HAR_IMPLEMENTATION.md) Steps 1–2). No new packages were needed for the HAR pipeline.

## Python and packages
- Python 3.11 (runs used 3.11.9, on both machines in [specs.md](specs.md)).
- Packages, with the exact versions the runs used. PyTorch is the CPU-only build; no GPU is needed.
- The full pinned environment, including every transitive dependency, is in [requirements.txt](requirements.txt) (`pip freeze` of `.venv` on Aya's machine).

| Package | Version | Used by |
|---|---|---|
| torch | 2.15.0 (CPU) | `lci/` (model, training, compression) |
| shap | 0.51.0 | `lci/explain.py` (GradientExplainer) |
| numpy | 2.4.6 | everything |
| pandas | 3.1.1 | `lci/`, `har/` (CSV loading, summaries), Phishing fetch in `data/download.py` |
| scikit-learn | 1.9.1 | `lci/data.py` (scaling, split), metrics |
| scipy | 1.17.1 | `har/data.py` (`scipy.signal.decimate` for S006's 100 → 50 Hz); dependency of scikit-learn / shap |
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
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install numpy==2.4.6 pandas==3.1.1 scikit-learn==1.9.1 scipy==1.17.1 shap==0.51.0 ucimlrepo==0.0.7 certifi==2026.10.23 pytest==9.2.0
```

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
```
