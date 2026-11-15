"""Every path the LCI code reads or writes. All outputs go under the repo-root results/ folder,
one subfolder per group of related runs, with descriptive kebab-case names (see CLAUDE.md)."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
PHISHING_DIR = REPO_ROOT / "data" / "raw" / "phishing"

# Outputs of lci/run_lci_phishing.py: the LCI reproduction on Phishing / ResidualMLP
LCI_PHISHING_DIR = RESULTS_DIR / "lci-phishing-reproduction"
RUNS_CSV = LCI_PHISHING_DIR / "lci-phishing-runs-per-seed-method.csv"
SUMMARY_CSV = LCI_PHISHING_DIR / "lci-phishing-summary-vs-paper.csv"
NOISE_CSV = LCI_PHISHING_DIR / "lci-phishing-noise-floor.csv"
SHAP_NPZ = LCI_PHISHING_DIR / "lci-phishing-shap-attributions.npz"
SHAP_CONFIG_JSON = LCI_PHISHING_DIR / "lci-phishing-shap-attributions-configs.json"
REPORT_MD = LCI_PHISHING_DIR / "lci-phishing-report.md"
