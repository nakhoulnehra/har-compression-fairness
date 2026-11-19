"""Every path the HAR code reads or writes. Results go under results/<run-group>/ with descriptive
kebab-case names (CLAUDE.md); rebuildable caches go under data/processed/ (git-ignored)."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
MANIFEST = REPO_ROOT / "data" / "checksums.sha256"

# Rebuildable cache (git-ignored)
PROCESSED_DIR = REPO_ROOT / "data" / "processed" / "har"
WINDOW_CACHE_NPZ = PROCESSED_DIR / "har-windows.npz"
WINDOW_CACHE_META = PROCESSED_DIR / "har-windows-meta.json"
WINDOW_STATS_CSV = PROCESSED_DIR / "har-window-stats.csv"
RECORDING_STATS_CSV = PROCESSED_DIR / "har-recording-stats.csv"

# Outputs of har/build_windows.py (plan Steps 1-2)
RESULTS_DIR = REPO_ROOT / "results"
HAR_SETUP_DIR = RESULTS_DIR / "har-setup"
WINDOW_COUNTS_PER_SUBJECT_CSV = HAR_SETUP_DIR / "har-window-counts-per-subject.csv"
WINDOW_COUNTS_PER_GROUP_CLASS_CSV = HAR_SETUP_DIR / "har-window-counts-per-group-and-class.csv"
DROPPED_DATA_CSV = HAR_SETUP_DIR / "har-dropped-data-summary.csv"
FOLD_ASSIGNMENT_CSV = HAR_SETUP_DIR / "har-fold-assignment.csv"
INNER_VALIDATION_CSV = HAR_SETUP_DIR / "har-inner-validation-subjects.csv"
NORMALISATION_STATS_CSV = HAR_SETUP_DIR / "har-normalisation-stats-per-fold.csv"
EXPLAIN_WINDOWS_CSV = HAR_SETUP_DIR / "har-explain-windows.csv"
EXPLAIN_COUNTS_CSV = HAR_SETUP_DIR / "har-explain-windows-per-subject-class-and-group.csv"
BACKGROUND_CSV = HAR_SETUP_DIR / "har-shap-background-windows.csv"
EXPLAIN_BACKGROUND_PER_FOLD_CSV = HAR_SETUP_DIR / "har-explain-and-background-windows-per-fold.csv"
