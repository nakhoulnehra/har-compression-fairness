# Requirements

Technical requirements for the project so far: data download and verification, and the LCI reproduction on Phishing ([../plan/LCI_IMPLEMENTATION.md](../plan/LCI_IMPLEMENTATION.md)).

## Python and packages
- Python 3.11 (runs used 3.11.9).
- Packages, with the exact versions the runs used. PyTorch is the CPU-only build; no GPU is needed.

| Package | Version | Used by |
|---|---|---|
| torch | 2.15.0 (CPU) | `lci/` (model, training, compression) |
| shap | 0.51.0 | `lci/explain.py` (GradientExplainer) |
| numpy | 2.4.6 | everything |
| pandas | 3.1.1 | `lci/`, Phishing fetch in `data/download.py` |
| scikit-learn | 1.9.1 | `lci/data.py` (scaling, split), metrics |
| scipy | 1.17.1 | dependency of scikit-learn / shap |
| ucimlrepo | 0.0.7 | Phishing fetch in `data/download.py` |
| pytest | 9.2.0 | `lci/tests/` |

The HARTH / HAR70+ part of `data/download.py` and `data/verify_checksums.py` use the standard library only.

Install into a virtualenv at the repo root (`.venv/` is git-ignored):

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows (Git Bash); use `source .venv/bin/activate` on Linux/macOS
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install numpy==2.4.6 pandas==3.1.1 scikit-learn==1.9.1 scipy==1.17.1 shap==0.51.0 ucimlrepo==0.0.7 pytest==9.2.0
```

## Network
- HTTPS access to `archive.ics.uci.edu` (one-time download, about 356 MB):
  - HARTH v1.2: `https://archive.ics.uci.edu/static/public/779/harth.zip` (311 MB)
  - HAR70+: `https://archive.ics.uci.edu/static/public/780/har70.zip` (44 MB)
  - Phishing Websites (UCI ID 327): fetched through the `ucimlrepo` API (about 0.8 MB)
- If Python's default CA store is stale (seen on this Windows machine), `data/download.py` retries with curl (HAR archives) or certifi's CA bundle (Phishing). TLS verification stays on, and SHA-256 checks guard the content.

## Disk
- About 1.4 GB free for `data/raw/` (355 MB of zip archives plus the extracted CSVs, plus 0.8 MB for Phishing).
- About 1.1 GB for `.venv/` (CPU PyTorch and the packages above).

## Data integrity
- All 44 files must match the SHA-256 hashes pinned in `data/checksums.sha256`: 2 archives, 40 subject CSVs (22 HARTH, 18 HAR70+) and the 2 Phishing CSVs (`phishing/features.csv`, `phishing/targets.csv`).
- Check with `python data/verify_checksums.py`.

## Commands run
Run from the repo root, in this order, inside the `.venv` above. Wall-clock times are in [time.md](time.md).

```bash
python data/download.py                    # HARTH, HAR70+, Phishing -> data/raw/, verify SHA-256
python data/verify_checksums.py            # 44/44 files OK
python -m pytest lci/tests/test_lci.py     # 12 LCI unit tests
python -m lci.run_lci_phishing --seeds 0 1 2 3 4 --shap_seed 0   # LCI grid -> results/lci-phishing-reproduction/
python -m lci.time_lci_phishing_components --seed 0              # per-stage timing (prints only)
```
