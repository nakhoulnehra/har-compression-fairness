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

# Model choice (plan Step 4): har/run_model_selection.py, har/time_model_selection_cpu.py, har/select_model.py
MODEL_SELECTION_DIR = RESULTS_DIR / "har-model-selection"
MODEL_SELECTION_SMOKE_DIR = RESULTS_DIR / "har-model-selection-smoke"
CHECKPOINT_DIR = PROCESSED_DIR / "checkpoints"            # git-ignored (data/processed/)
MODEL_SELECTION_CKPT_DIR = CHECKPOINT_DIR / "har-model-selection"
MODEL_SELECTION_SMOKE_CKPT_DIR = CHECKPOINT_DIR / "har-model-selection-smoke"

# SHAP cache (plan 2.6; git-ignored): raw (n, 6, 128) arrays keyed by model weights, windows, background,
# targets, SHAP seed and nsamples
SHAP_CACHE_DIR = PROCESSED_DIR / "shap-cache"

# Noise floor (plan Step 8, Section 2.7) and the XGP pilot (fold 0, inner validation only)
NOISE_FLOOR_DIR = RESULTS_DIR / "har-noise-floor"
NOISE_FLOOR_SMOKE_DIR = RESULTS_DIR / "har-noise-floor-smoke"
NOISE_FLOOR_CKPT_DIR = CHECKPOINT_DIR / "har-noise-floor"
NOISE_FLOOR_SMOKE_CKPT_DIR = CHECKPOINT_DIR / "har-noise-floor-smoke"
XGP_PILOT_DIR = RESULTS_DIR / "har-xgp-pilot"
XGP_PILOT_SMOKE_DIR = RESULTS_DIR / "har-xgp-pilot-smoke"
# XGP pilot round 2 (fold 0, inner validation only; a validation-only design iteration before XGP is frozen)
XGP_PILOT2_DIR = XGP_PILOT_DIR / "round2"
XGP_PILOT2_SMOKE_DIR = XGP_PILOT_SMOKE_DIR / "round2"

# Compression grid (plan Step 9): per-fold files, and the --analyze outputs (part A = baselines only)
GRID_DIR = RESULTS_DIR / "har-compression-grid"
GRID_SMOKE_DIR = RESULTS_DIR / "har-compression-grid-smoke"
GRID_ANALYSIS_DIR = RESULTS_DIR / "har-compression-grid-analysis"
GRID_ANALYSIS_SMOKE_DIR = RESULTS_DIR / "har-compression-grid-analysis-smoke"
GRID_CKPT_DIR = CHECKPOINT_DIR / "har-compression-grid"
GRID_SMOKE_CKPT_DIR = CHECKPOINT_DIR / "har-compression-grid-smoke"
