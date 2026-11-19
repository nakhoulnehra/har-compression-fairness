# har-compression-fairness

## Layout

| Folder | What |
|---|---|
| [data/](data/) | Download and checksum scripts; raw data lands in `data/raw/` (git-ignored). |
| [lci/](lci/) | Everything LCI: code, experiment runner and tests. |
| [results/](results/) | Every result file, one subfolder per group of related runs. [results/README.md](results/README.md) summarizes the key numbers of each file. |
| [environment/](environment/) | [specs.md](environment/specs.md) (machines), [requirements.md](environment/requirements.md) (setup, packages, commands run), [requirements.txt](environment/requirements.txt) (full pinned `pip freeze`), [time.md](environment/time.md) (per-command timings, per machine). |
| [plan/](plan/) | [LCI_IMPLEMENTATION.md](plan/LCI_IMPLEMENTATION.md): spec for the LCI implementation and validation. [HAR_IMPLEMENTATION.md](plan/HAR_IMPLEMENTATION.md): plan for the HAR part (HARTH → HAR70+, compression grid, FA-LCI). |

Setup (Python 3.11, CPU-only PyTorch, pinned packages): see [environment/requirements.md](environment/requirements.md).

## Data

Datasets (UCI ML Repository):

- [HARTH v1.2](https://archive.ics.uci.edu/dataset/779/harth) (22 subjects) and [HAR70+](https://archive.ics.uci.edu/dataset/780/har70) (18 subjects aged 70+).
- [Phishing Websites](https://archive.ics.uci.edu/dataset/327/phishing+websites) (UCI ID 327), used to validate the LCI metric. DOI [10.24432/C51W2X](https://doi.org/10.24432/C51W2X), license CC BY 4.0. Mohammad, R. & McCluskey, L. (2012). 11,055 rows, 30 integer features in {-1, 0, 1}, binary target `result` (-1 = phishing, 1 = legitimate; 4,898 : 6,157 = 44 : 56). Fetched with `ucimlrepo` (`fetch_ucirepo(id=327)`), exactly as the LCI paper's authors do, and saved raw and unmodified.

```bash
python data/download.py          # download, extract to data/raw/, verify SHA-256
python data/verify_checksums.py  # re-check files at any time
```

Files land in `data/raw/harth/` and `data/raw/har70plus/` (one CSV per subject) and `data/raw/phishing/` (`features.csv`, `targets.csv`). `data/raw/` is git-ignored; checksums are pinned in `data/checksums.sha256`.

## LCI reproduction (Phishing / ResidualMLP)

Implements the Logic Consistency Index of Tulla et al. (2026) per [plan/LCI_IMPLEMENTATION.md](plan/LCI_IMPLEMENTATION.md). All code is in [lci/](lci/): `metric.py` is the pure-numpy LCI, `run_lci_phishing.py` the experiment, `paths.py` defines every output file, and the tests are in `lci/tests/`.

```bash
python -m pytest lci/tests/test_lci.py                           # LCI unit tests
python -m lci.run_lci_phishing --seeds 0 1 2 3 4 --shap_seed 0   # full grid -> results/lci-phishing-reproduction/
python -m lci.time_lci_phishing_components --seed 0              # per-stage timing for environment/time.md
```

Results and the comparison with the paper: [results/lci-phishing-reproduction/lci-phishing-report.md](results/lci-phishing-reproduction/lci-phishing-report.md).
