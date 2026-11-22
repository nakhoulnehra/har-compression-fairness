# Task: Explanation consistency of compressed HAR models across age groups (HARTH → HAR70+)

This document is the complete specification for the HAR part of the study. It covers the data and windowing, the subject-wise folds, the choice of the model (made once, then frozen), the compression grid, SHAP, the window version of LCI, FA-LCI, deletion faithfulness, the noise floor, the statistics, the outputs and the run plan for the two machines in [environment/specs.md](../environment/specs.md). It builds on the LCI reproduction ([LCI_IMPLEMENTATION.md](LCI_IMPLEMENTATION.md)); reuse `lci/` wherever this document says so.

**Research question.** When a HAR model trained on adults (HARTH) is compressed, do its SHAP explanations stay consistent with those of the original model, and does that consistency drop more for older adults (HAR70+) than for the general population the model was trained on? Can an explanation-guided pruning rule, built from HARTH data only, keep explanations consistent for both groups at the same size?

**Research questions (for the paper, decided 2026-11-21).**
- **RQ0:** Is FA-LCI a valid measure of subgroup explanation consistency? It must equal the worse group's LCI, detect injected gaps and agree with deletion faithfulness (the three FA-LCI validation checks, Section 2.8).
- **RQ1:** Does compression (pruning, quantization) change SHAP explanations beyond retraining noise, even when accuracy is kept?
- **RQ2:** Is this explanation drift larger for older adults (and walking-aid users, inferred) than for general adults?
- **RQ3:** Can XGP reduce the drift and the subgroup gap at the same model size, without losing accuracy?

Step 9A answers RQ1 and RQ2 with the standard methods and gives RQ0 its grid data; Step 9B (XGP) answers RQ3; the RQ0 validation checks run in Step 11.

**Ground rules**
- **HAR70+ is test-only.** It is never used for training, early stopping, hyperparameter tuning, model choice, choosing or tuning the pruning rule, or setting any threshold that a method uses during training. Enforce this in code (Section 2.4), not just by convention.
- **No person appears in both train and test.** All splits are by subject.
- **The subject is the statistical unit.** Average each subject's windows first, then compare subjects.
- Never fall back to synthetic data. If a file is missing or fails its checksum, raise an error and stop.
- Every random choice is controlled by an explicit seed: numpy, torch, Python `random`, fold assignment, window sampling, SHAP sampling and bootstrap.
- Always report LCI's two components (`shap_corr`, `jaccard`) next to LCI, and FA-LCI's parts (`lci_all`, `lci_older`, `lci_general`, `gap`) next to FA-LCI.
- Label the walking-aid subgroup **"inferred"** in every file, table, figure and sentence.
- Do not tune anything to make a result look better after HAR70+ numbers have been seen. If a check fails, report it.

---

## 1. Data

Both datasets are already downloaded and SHA-256 verified by `python data/download.py` (`data/raw/harth/`, `data/raw/har70plus/`). Facts below were measured on the raw files on 2026-11-19 with a throw-away profiling script; the implementation must re-derive and assert them.

| | HARTH v1.2 | HAR70+ |
|---|---|---|
| Subjects | 22 (S006, S008–S010, S012–S029) | 18 (501–518) |
| Role | train + "general" test | "older" test only |
| Sensors | back + thigh 3-axis accelerometer, in g | same |
| Sampling rate | 50 Hz, **except S006 at 100 Hz** (10 ms steps) | 50 Hz |
| Recorded time | ≈ 34.8 h | ≈ 12.6 h |
| Columns | `timestamp, back_x..z, thigh_x..z, label`; S015 and S021 add `index`, S023 adds an unnamed index column | `timestamp, back_x..z, thigh_x..z, label` |
| Gaps | 5–69 breaks > 100 ms per subject, longest 302 s (S010) | 7–20 per subject, longest 112 s |
| Labels | 1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 130, 140 | 1, 3, 4, 5, 6, 7, 8 |

**Shared classes (the only ones kept)**, with model class indices:

| HARTH/HAR70+ label | 1 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| Class | walking | shuffling | stairs up | stairs down | standing | sitting | lying |
| Index | 0 | 1 | 2 | 3 | 4 | 5 | 6 |

Points the implementation must handle:
- **S006** must be resampled to 50 Hz (Section 2.1) or every window would cover 1.28 s instead of 2.56 s.
- **Three HARTH subjects contribute few shared-class windows**, because most of their recording is cycling or running: S009 (≈ 190 training windows), S027 (≈ 400), S026 (≈ 820). The others have 1,800–6,300. Fold assignment balances window counts (Section 2.2).
- **Stairs are rare in HAR70+**: 4,560 + 4,978 rows in total, about 75 non-overlapping windows over 13 subjects, and several subjects have fewer than one full stairs window. Stairs F1 is therefore reported **pooled across subjects, never per subject**, and per-subject macro-F1 covers the five non-stair classes (Section 2.7).
- **Walking-aid subjects (inferred).** 503, 505, 513, 514 and 515 are exactly the five HAR70+ subjects with no stairs rows at all (502 has stairs down only, so it is not in this group). This matches the decision; the group is still "inferred" because no metadata file confirms it.
- **Sitting dominates HARTH** (2.9 M of 6.6 M shared-class rows); stairs are 2% of rows. Training uses class weights (Section 2.4).
- Approximate window counts (from the profiling script, which split segments at gaps > 100 ms; the real numbers come from Step 1): HARTH ≈ 84.7K windows at 50% overlap after S006 resampling (≈ 90% with a single label), HAR70+ ≈ 17.5K non-overlapping windows (87% single-label).

---

## 2. Implementation specification

All new code lives in a new package `har/`, laid out like `lci/`. Pure-numpy metric code goes in `lci/metric.py` (Section 2.6) so the Phishing reproduction and the HAR study share one LCI. Output paths are defined in `har/paths.py` only.

```
har/
  __init__.py
  paths.py              every input/output path (Section 4)
  data.py               loading, 50 Hz resampling, segmentation, windowing, window cache
  folds.py              subject folds, inner validation, explain windows, SHAP background
  features.py           hand-crafted window features for the XGBoost reference
  models.py             CNN1D, DeepConvLSTM, TinyHAR, HART; build_model()
  train.py              class weights, training with early stopping, fine-tuning, per-subject metrics
  compression.py        PTQ (per-tensor symmetric; Tulla-style), L1, SNIP-score pruning,
                        L1 + attribution matching, explanation-guided pruning, optional KD
  attributions.py       SHAP (single target output) with cache; differentiable EG surrogate for losses
  faithfulness.py       deletion curves against the background mean
  stats.py              Wilcoxon + Holm, Cohen's d_z, Mann-Whitney U, bootstrap CIs, safe rate
  build_windows.py      Step 1 runner
  run_model_selection.py, select_model.py, run_sanity_walking_aid.py
  run_noise_floor.py, run_compression_grid.py, analyze.py, time_har_components.py
  tests/                Section 2.10
```

### 2.1 Loading, resampling and windowing (`har/data.py`)

```python
SHARED_LABELS = (1, 3, 4, 5, 6, 7, 8)
LABEL_TO_INDEX = {1: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6}
CLASS_NAMES = ("walking", "shuffling", "stairs_up", "stairs_down", "standing", "sitting", "lying")
CHANNELS = ("back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z")   # tensor channel order
FS, WIN, SEG = 50, 128, 16             # 2.56 s windows, 8 segments of 0.32 s
NON_STAIR = (0, 1, 4, 5, 6)            # class indices used for the walking-aid comparison
WALKING_AID_INFERRED = ("503", "505", "513", "514", "515")

@dataclass
class Windows:
    X: np.ndarray               # (n, 6, 128) float32, raw acceleration in g (not normalized)
    y: np.ndarray               # (n,) int64 class index 0..6
    subject: np.ndarray         # (n,) str, e.g. "S006" or "501"
    dataset: str                # "harth" or "har70plus"
    group: np.ndarray           # (n,) str, "general" (HARTH) or "older" (HAR70+)
    walking_aid: np.ndarray     # (n,) str, inferred_walking_aid / inferred_no_walking_aid / not_applicable
    majority_share: np.ndarray  # (n,) float32, share of the 128 rows with the window label
    start_time: np.ndarray      # (n,) datetime64[ms], first timestamp of the window
    start_row: np.ndarray       # (n,) int64, first row in the subject's 50 Hz recording (for audits)

def load_recording(dataset: str, subject: str) -> pd.DataFrame
    # Reads one CSV; keeps timestamp, the 6 channels and label; drops `index` / unnamed columns.
    # Asserts: no NaN, timestamps strictly increasing, labels within the dataset's known set.
def to_50hz(df: pd.DataFrame) -> pd.DataFrame
    # Median step 20 ms -> unchanged. 10 ms -> per contiguous segment, scipy.signal.decimate(q=2,
    # ftype="fir", zero_phase=True) on each channel, labels and timestamps take every 2nd row.
    # Any other rate -> raise.
def lowpass_20hz(df: pd.DataFrame) -> pd.DataFrame
    # HARTH's 4th-order 20 Hz Butterworth low-pass [HARTH Sec 3.3] on the 6 channels, zero-phase
    # (sosfiltfilt), per gap-free segment, both datasets (decided 2026-11-19). Labels, timestamps,
    # rows unchanged. preprocess(raw) = lowpass_20hz(to_50hz(raw)) is what build_all() windows.
def contiguous_segments(df: pd.DataFrame, max_gap_s: float = 0.04) -> list[slice]
    # Splits where the step exceeds 2x the 20 ms period.
def make_windows(df: pd.DataFrame, dataset: str, subject: str, step: int) -> Windows
def build_all(force: bool = False) -> dict[str, Windows]
    # {"harth_train": step 64, "harth_test": step 128, "har70plus_test": step 128};
    # cached in data/processed/har/ (git-ignored), keyed by a hash of the raw files and this config.
```

Windowing rules:
1. Windows never cross a gap and never contain a row whose label is not shared (2, 13, 14, 130, 140). *As implemented:* windows are cut inside **runs**, the maximal stretches of rows that are gap-free and carry only shared labels. Non-shared rows are dropped before windowing (never relabelled), and the window grid restarts after them.
2. **Training (HARTH):** 50% overlap (step 64). **Testing (HARTH held-out and HAR70+):** no overlap (step 128). Both start at the first row of each run; rows in runs shorter than 128 and run tails shorter than one step are dropped and counted.
3. **Window label = majority label** (the mode over the 128 rows).
4. **Mixed-label windows:** a window is kept if its majority label covers **more than 50%** of its rows (at least 65 of 128) and dropped otherwise (ties such as 64/64, or no label above 50%). Kept mixed windows carry their `majority_share`. Step 1 reports, per subject and dataset, the number of single-label windows, kept mixed windows and dropped mixed windows. *Why:* transitions are part of real use, so test windows at transitions are kept; windows with no clear label have no ground truth and are dropped. A robustness check in Step 11 repeats the main LCI result on single-label windows only.
5. **Normalization (decided, Q10).** No normalization is stored in the cache. Models normalize per channel with mean and std computed on the `harth_train` windows of the **current fold's HARTH training subjects only**: the 3 inner-validation subjects, the fold's test subjects and all of HAR70+ are excluded. These statistics are stored with the model and applied **unchanged** to inner-validation, HARTH test and HAR70+ windows; they are never recomputed on those sets. Because the normalization is part of the model, SHAP, the background and deletion all work in raw g. Class weights (Section 2.4) come from the same training windows.

```python
# in har/folds.py (it needs the fold definition):
def channel_stats(w: Windows, fold: FoldSplit, subjects: Sequence[str] | None = None) -> tuple[np.ndarray, np.ndarray]
    # (mean (6,), std (6,)), population std over every sample of the windows of `subjects`
    # (default: fold.train). Calls require_harth(w); raises if any requested subject is not one of
    # the fold's training subjects (validation, test, HAR70+ or unknown IDs).
```

### 2.2 Folds, inner validation, explain windows and background (`har/folds.py`)

```python
def assign_folds(window_counts: dict[str, int], n_folds: int = 5) -> dict[str, int]
    # Deterministic greedy balancing: sort HARTH subjects by test-window count (desc, ties by ID),
    # put each in the fold with the fewest windows that has < 5 subjects (ties -> lowest fold).
    # Gives folds of 5, 5, 4, 4, 4 subjects with similar window counts.
def inner_val_subjects(train_subjects: list[str], window_counts: dict[str, int],
                       n: int = 3, seed: int = 42, fold: int = 0) -> list[str]
    # 3 of the fold's non-test subjects with >= 1,000 non-overlapping windows, drawn with seed + fold
    # (a single seed for all folds made the same few subjects the validation set of most folds).
    # Used for early stopping, tuning, the pruning-rule choice, the gate and tau. Never for testing.
def make_folds(harth_test_counts: dict[str, int], older_subjects) -> list[FoldSplit]
    # FoldSplit(fold, train, val, test_general, test_older); FoldSplit.check() raises on any overlap
    # or if a HAR70+ subject is in train/val. Implemented in har/folds.py.
def select_explain_windows(w: Windows, split: str, per_subject: int = 30, seed: int = 42) -> np.ndarray
    # Indices into a test split ("harth_test" or "har70plus_test"; raises for any other split).
    # Per subject: up to 30 windows, stratified by class: round-robin over the classes present
    # (one window per class per round, in class-index order; random within class) until 30 are
    # chosen or the subject runs out. Generator per subject: default_rng([seed, crc32(subject)]).
def select_background(w: Windows, split: str, fold: FoldSplit, n: int = 100, seed: int = 42) -> np.ndarray
    # 100 harth_train windows from fold.train only (raises for HAR70+ windows or another split),
    # stratified by class the same way over the pooled training windows; default_rng([seed, fold]).
def window_ids(w: Windows, split: str) -> np.ndarray      # in har/data.py: "<split>:<subject>:<start_row>"
def find_windows(w: Windows, split: str, ids) -> np.ndarray  # IDs -> indices; raises if one is missing
```

*As built (2026-11-19):* every subject has ≥ 30 test windows, so there are 1,200 explain windows (660 general, 540 older). Every fold's background has 100 windows, 14 or 15 per class, from 13–14 training subjects. Stratifying by class oversamples transition-heavy classes, so 29% of explain windows and 22–34% of background windows have mixed labels (10% of all windows do). The single-label robustness check in Step 11 covers this. When a subject is a fold's validation subject, the gate and τ use that subject's explain windows.

- The fold assignment, inner-validation subjects, explain windows and backgrounds are written once in Step 2 to `results/har-setup/` and read from there by every later step. Nothing re-samples them.
- **Explain windows are the same for every model.** Each HARTH subject is tested in exactly one fold, so its 30 windows are fixed. The 18 × ≤30 HAR70+ windows are the same in every fold. About 670 windows are explained per model per fold (≈ 130 general + ≈ 540 older).
- **Background:** "100 fixed HARTH training windows" means fixed per fold. A fold's test subjects cannot be in its background, so the background cannot be identical across folds. Within a fold, every model uses the same background.

### 2.3 Candidate models (`har/models.py`, `har/features.py`)

Every model takes `(B, 6, 128)` raw-g input, applies its stored per-channel normalization, and returns `(B, 7)` logits. Channels 0–2 are the back sensor, 3–5 the thigh sensor.

| Candidate | Source | Notes |
|---|---|---|
| `cnn1d` | Logacjov et al. (2021) [HARTH Sec 4.1] | 4 conv layers × 128 kernels, kernel sizes [6, 12, 12, 32], ReLU, dropout 0.4 after each layer, two 512-unit ReLU dense layers, output layer. Our choices: axes as input channels, 'same' padding, global average pooling. 1,254,919 parameters. |
| `deepconvlstm` | Ordóñez & Roggen (2016) [DCL Sec 3.1, 3.3, Table 1] | C(64)×4 with kernel 5 along time per channel, no padding, R(128)×2, dropout 0.5 on dense inputs, orthogonal initialisation, last time step. 458,183 parameters. |
| `tinyhar` | Zhou et al. (2022) [TINY Sec 3.2] | 4 individual conv layers (5×1, stride 2, BN, ReLU), 1 transformer block across channels, FC fusion to 2F, 1 LSTM, temporal attention + γ. F = 28 [TINY Sec 4.2]. 54,273 parameters. Written from the paper; no code ported. |
| `hart` | Ek, Portet & Lalanda [HART Sec 3.2, 4.1–4.2, Table 7] | Re-implemented in PyTorch from the paper only (no code ported): d = 192, 6 blocks, 3 heads, sensor-wise MSA on d/4 + d/4 and LightConv on d/2, Swish, dropout 0.3, GAP. Its two sensor branches (accelerometer, gyroscope in the original) become **back** (ch. 0–2) and **thigh** (ch. 3–5). 1,389,555 parameters (paper: 1,445,918). |

Every value is cited, or marked "our choice", in `har/models.py` and in Appendix A.
| `xgboost` | reference only | `features.window_features(X) -> (n, F)`: per channel and per sensor magnitude: mean, std, min, max, median, IQR, skewness, kurtosis, energy, dominant frequency, spectral entropy, plus the 3 axis correlations per sensor. Never compressed or explained; never eligible as the frozen model. |

```python
def build_model(name: str, cfg: dict, norm_mean: np.ndarray, norm_std: np.ndarray) -> nn.Module
```

Each candidate has a unit test for output shape, parameter count, a forward/backward pass and a 2-window SHAP call (Section 2.10).

### 2.4 Training (`har/train.py`)

```python
@dataclass
class TrainConfig:
    lr: float = 1e-3; weight_decay: float = 1e-4; batch_size: int = 256
    max_epochs: int = 30; patience: int = 5          # early stopping on inner-val macro-F1

def require_harth(*ws: Windows) -> None             # raises if any w.dataset != "harth"
def class_weights(y: np.ndarray, n_classes: int = 7) -> torch.Tensor
    # Inverse class frequency on the fold's training windows (HARTH training subjects only),
    # normalized to mean 1 (Q10: sitting is 44% of HARTH rows, stairs 2%).
def train_model(model, train: Windows, val: Windows, cfg: TrainConfig, seed: int) -> dict
    # Adam, weighted CrossEntropy, CosineAnnealingLR(T_max=max_epochs); keeps the state dict with
    # the best inner-val macro-F1. Returns {"best_epoch", "best_val_f1", "history"}.
def finetune(model, train: Windows, val: Windows, epochs: int, cfg: TrainConfig, seed: int,
             extra_loss: Callable | None = None) -> dict
    # Fixed number of epochs (no early stopping), so every pruning method gets the same budget;
    # pruning masks are kept during fine-tuning and made permanent at the end.
def predict_proba(model, X: np.ndarray, batch_size: int = 1024) -> np.ndarray    # (n, 7)
def subject_metrics(y, pred, subject, f1_classes: Sequence[int] = NON_STAIR) -> pd.DataFrame
    # One row per subject: n_windows, accuracy (all windows), macro-F1 over the classes in
    # `f1_classes` that are present in that subject's true labels (default: the 5 non-stair classes).
def pooled_class_f1(y, pred, classes: Sequence[int] = (2, 3)) -> dict
    # F1 per class over all windows of a group pooled across subjects (stairs up, stairs down,
    # and their mean). Never computed per subject.
def pooled_macro_f1(y, pred, classes: Sequence[int] = tuple(range(7))) -> float
    # Macro-F1 over all windows of a set of subjects pooled together.
def train_xgboost(train: Windows, val: Windows, params: dict, seed: int) -> object
```

**One shared training recipe (deliberate deviation, decided 2026-11-19).** Every candidate uses the `TrainConfig` above and the same tuning budget, not the recipe of its own paper (SGD for the HARTH CNN, RMSProp for DeepConvLSTM, Adam with each paper's lr, epochs and batch size for TinyHAR and HART). The comparison then measures architectures, not training effort. This is stated in the report (Appendix A.7).

**Leakage guard.** `train_model`, `finetune`, every tuning loop, every pruning-score function and the gate call `require_harth()` on all data they receive. A test checks that passing HAR70+ windows raises.

### 2.5 Compression grid (`har/compression.py`)

Each method starts from a `copy.deepcopy` of the frozen model trained on the fold (seed 0).

**Pruning scope (decided, Q3).**
- **Prunable tensors:** the weight tensors of every `Conv1d`, `Conv2d` and `Linear` module, the LSTM `weight_ih_l*` / `weight_hh_l*` and the attention in/out projection weights, whichever the frozen architecture has.
- **The classification head is excluded**, because pruning up to 90% of it can wipe out whole classes. Biases, normalization parameters and embeddings are not pruned either.
- **Uniform per-layer sparsity:** the same fraction is removed from every prunable tensor. That gives every pruning method exactly the same non-zero count at each level, so sizes are matched by construction.
- The nominal level (30/50/70/90%) is the sparsity of the prunable tensors. Every model also reports the **actual total sparsity including the head**, and per-layer sparsity.

```python
def prunable_tensors(model) -> list[tuple[nn.Module, str]]      # excludes the head
def prune_by_score(model, scores: dict[str, torch.Tensor], sparsity: float) -> nn.Module
    # Per tensor, zero the `sparsity` fraction with the lowest score (ties -> lowest flat index),
    # via torch.nn.utils.prune.custom_from_mask; masks stay attached until finetune() ends.
def size_record(model, quantized: dict[str, int] | None = None) -> dict
    # {"n_params", "n_nonzero",
    #  "sparsity_prunable"        zeros / size over the prunable tensors (= the nominal level),
    #  "sparsity_weights_incl_head" zeros / size over all weight tensors including the head,
    #  "sparsity_all_params"      1 - n_nonzero / n_params (weights, biases, norm parameters),
    #  "per_layer_sparsity"       {tensor name: sparsity},
    #  "size_kb_dense"            n_params x 4 B,
    #  "size_kb_nominal"}         PTQ: quantized tensors at `bits` per value + the rest at 4 B;
    #                             pruned: n_nonzero x 4 B (values only, index overhead not counted;
    #                             say so in the report).
```

| Method | Levels | Score / procedure | Fine-tune (HARTH train, inner val) |
|---|---|---|---|
| **PTQ (primary)** | 8, 4 bit | `ptq_symmetric(model, bits)`: standard per-tensor symmetric weight quantization (below). | none |
| **PTQ, Tulla-style (secondary)** | 8, 4 bit | `ptq_tulla(model, bits)` = `lci.compression.ptq` unchanged: every learnable parameter rounded to a fixed 1/(2^(b−1)−1) grid, no scale, no clamping. Only for comparison with Tulla et al. (2026); always labelled "Tulla-style". | none |
| **L1** | 30, 50, 70, 90% | `l1_scores(model)`: \|w\| | `E_ft` epochs, CE |
| **SNIP-score pruning** | same | `snip_scores(model, train, n_batches=10, seed=42)`: the SNIP saliency \|w · ∂L/∂w\| (Lee et al., 2019) on the **trained** model, gradients summed over 10 fixed batches of 256 HARTH training windows (Q2). | `E_ft` epochs, CE |
| **L1 + attribution matching** (Park et al., 2020, adapted) | same | L1 scores | `E_ft` epochs, CE + γ_P · L_AM |
| **XGP, explanation-guided pruning (ours)** | same | keep-score variant fixed by pilot round 2 (below) | the shared fine-tuning recipe (KD if round 2 confirms it, the same for every method); round 1's L_cons and gate are dropped |
| **XGP ablations** | 70, 90% only | `xgp_score_only` (keep-score, CE), `l1_plus_lcons` (L1, CE + γ · L_cons always), `xgp_no_gate` (keep-score, CE + γ · L_cons always) | `E_ft` epochs, matched `n_nonzero` |

**Pruning masks (implemented 2026-11-20; deviation from `custom_from_mask` above).** `prune_by_score` stores plain keep-masks in `model.prune_masks` and zeroes the weights in place. `finetune` zeroes the masked gradients before each optimizer step and re-applies the masks after it. *Why:* `torch.nn.utils.prune` recomputes `weight` in a forward pre-hook of the owning module, but `nn.MultiheadAttention` reads `out_proj.weight` without calling `out_proj`'s forward, so that hook never runs and TinyHAR's attention output projection would not be masked. `finetune` runs exactly `E_ft` epochs and keeps the last state (no early stopping). It logs inner-val macro-F1 after every epoch.

**PTQ, primary (decided, Q1).** `ptq_symmetric` quantizes the same weight tensors that pruning touches, **plus the head**. Per tensor:
- `s = max|W| / (2^(b−1) − 1)`;
- `W_q = clamp(round(W / s), −(2^(b−1) − 1), 2^(b−1) − 1) · s`, so 255 levels at 8 bits and 15 at 4 bits, with zero exactly representable;
- no calibration and no fine-tuning.

Biases and normalization parameters stay float32, and activations are not quantized (weight-only, the usual setting for PTQ of small models). The result is stored back as float (fake quantization), so gradients exist for SHAP. Both PTQ variants appear in every table, labelled "PTQ" and "PTQ (Tulla-style)"; the Tulla-style rows are excluded from the statistical comparisons (Section 2.8).

**SNIP-score pruning (decided, Q2).** This name is used everywhere. Original SNIP prunes at initialization and trains from scratch; here only its saliency score is used, on the trained model, followed by the same fine-tuning as every other method. The report states this difference.

**KD as a compression method (decided, Q14): not part of the main grid** (not to be confused with the shared KD *fine-tuning recipe* of pilot round 2, which every pruning method uses and which changes no model size). Its size is not matched to the pruning levels, so it cannot enter the paired comparisons. It runs only if time is left after Step 9 (Step 10): one student of the frozen model's family, width scaled to ≈ 70% of the parameters, 30 epochs, `α·KL(T)·T² + (1−α)·CE` with T = 3 and α = 0.7 (as in `lci.compression.distill`). It is reported descriptively.

`E_ft = 10` epochs for every pruning method, with the same optimizer (Adam, lr 1e-4), batch size, class weights and seed. L1 without fine-tuning is not part of the grid.

**Attribution surrogate used inside losses.** SHAP itself is too slow inside training. Losses use a differentiable expected-gradients surrogate (the quantity GradientExplainer estimates):

```python
def eg_attributions(model, X: torch.Tensor, target: torch.Tensor, baselines: torch.Tensor,
                    m: int, create_graph: bool) -> torch.Tensor          # (B, 6, 128)
    # mean over m (baseline b, alpha ~ U(0,1)) draws of (x - b) * d f_target(b + alpha (x - b)) / dx,
    # baselines drawn from the fold's 100-window background with a seeded generator.
```

Both attribution losses compare 48-segment attributions (Section 2.6), with target = the original model's predicted class (Section 2.6).

**(a) Park et al. (2020), adapted** (checked against papers/2010.15054v1.pdf; Appendix A).
- **The paper's loss:** supervised loss plus β times the **unsquared** ℓ2 distance between the L2-normalised attribution maps of the compressed and the original network, summed over the matched layers [Park Sec 3.2, Eq. 2, p5].
  - The paper matches at the **last convolutional layer** with sensitivity-weighted Grad-CAM maps (SWA, Eq. 4–5).
  - It matches only the map of the **full network's top-1 prediction** [Park Sec 4, p6].
  - It uses β = 50 for SWA [Park App. C.2, p15].
- **Here:** `L_AM = mean_i ‖ â_S(x_i) − â_T(x_i) ‖₂`, unsquared as in Eq. 2. `â` is the 48-segment EG attribution (m = 4) of the original model's top-1 class, scaled to unit L2 norm per window; the original model's attributions are precomputed and fixed.
- **Deviation (our choice):** matching on the input attribution of a time series, instead of a Grad-CAM map at the last convolutional layer. It is the only level that is comparable across the four architectures. Record this in the report.

**(b) XGP, explanation-guided pruning (ours)** (`har/xgp.py`, implemented 2026-11-20). **Uses HARTH data only:** every function that takes windows calls `require_harth`, and the pilot never loads HAR70+.

**Pilot round 2 (decided 2026-11-21): the last validation-only design iteration before XGP is frozen.** Fold 0, HARTH inner validation only; HAR70+ is never loaded (`require_harth` on every window set, older-like copies included). Code: `har/run_xgp_pilot2.py`, `har/augment.py`, `XGP2` in `har/xgp.py`; output `results/har-xgp-pilot/round2/`.
- **What changed after round 1.** XGP is now **only the weight selection** (the keep-score variants below). Fine-tuning is a **shared recipe, not part of XGP**: every method gets the same one. The recipe tested is KD: `0.7 · KL(T = 3) · T² + 0.3 · class-weighted CE`, KL to the original model's logits, CE on the labels (as in `lci.compression.distill`); `E_ft = 10`, Adam lr 1e-4. **L_cons and the gate are dropped**: EG with m = 4 agreed with itself only ≈ 0.44 in an unsaved check (so the repair loss is mostly noise), and after 5 epochs LCI was above τ_0, so the gate never fired. The EG-noise diagnostic is reported as a finding. The L_cons / gate / round-1 tuning text below is kept as the record of round 1.
- **Variants (all + KD).** `xgp_kd`: keep-score, β ∈ {0.05, 0.1, 0.3}. `xgp_worstclass_kd`: I(w) = max over classes of I_c(w) / mean_layer I_c, with I_c computed on that class's reference windows (by label). `xgp_aug_kd`: I on the 1,000 real + 1,000 older-like reference windows. `xgp_worstclass_aug_kd`: both. The last three use the β of `xgp_kd` chosen at that level.
- **Baselines.** `l1` (CE), `l1_kd`, `snip_kd`, `park` (CE + γ_P · L_AM, γ_P = 1, lr 1e-4 = round 1's selection), `park_kd` (KD + γ_P · L_AM). Epochs check: `l1_kd_e20` (L1 start, KD, E_ft = 20).
- **Runs.** 70% (nakhoul-3050) and 90% (uni-2080), seeds 0, 1, 2 = the fine-tuning batch-order seed (Park's EG seed = 1 + seed). `park`, `park_kd`, `l1_kd_e20`: seed 0 only. Every pruned start is also assessed before fine-tuning (scenario `no_finetune`). n_nonzero is asserted identical for every method within a level.
- **Windows (fixed, seeded, saved by window ID before any run, `--prepare`).** 100 inner-val explain windows per subject (`select_explain_windows`, per_subject = 100, non-overlapping split; the first 30 per subject are the round-1 windows). Older-like copies (`har/augment.py`): time-stretch with speed s ~ U(0.8, 0.9) (the central round(128 s) samples per channel, linearly resampled to 128), then movement scaling around each window's per-channel mean, x' = mean + a(x − mean), a ~ U(0.7, 0.9), so the gravity component is unchanged. s and a are drawn once per window (seed 42 for the inner-val windows, 43 for the reference windows) and saved. F1 uses all inner-val windows and their older-like copies. SHAP target = the original model's predicted class on each (normal or older-like) window; background = the fold's 100 HARTH windows; nsamples 200, SHAP seed 0.
- **Reported per run, on normal and older-like windows:** LCI_val with its ρ and Jaccard parts, worst-subject LCI (min over the 3 subjects), pooled 7-class macro-F1, retention = (LCI − τ_0) / (ceiling_0 − τ_0) with ceiling_0 = the fold-0 inner-val SHAP-noise LCI mean of the noise floor (0.9455; τ_0 = 0.6593), n_nonzero, finetune_seconds; summaries as mean ± sd over seeds.
- **Diagnostic (finding).** On the original fold-0 model and the normal inner-val windows: LCI(real SHAP, EG) for m = 4, 16, 200, EG m = 4 vs EG m = 4 with another seed, and SHAP seed 0 vs seed 1 for scale.
- **Pre-registered selection rule (written 2026-11-21, before any GPU run; also in `har-xgp-pilot2-tinyhar-fold0-preregistration.json`).** Final XGP = the eligible XGP variant with the highest mean LCI_val over {70%, 90%} × {normal, older-like} windows (mean over seeds). Eligible = mean pooled 7-class macro-F1 on the normal inner-val windows ≥ the best baseline's mean F1 − 1 pt, **at both levels**. Baselines: l1, l1_kd, snip_kd, park, park_kd. Ties within 0.005 → the variant with fewer components (keep-score 1, + worst-class 1, + augmentation 1). **If no variant is eligible, there is no fallback: report and stop.**
  - Our resolution of the β detail (fixed before the runs): the variants' β at a level = the β of `xgp_kd` that is eligible at that level alone and has the highest mean LCI there (ties within 0.005 → the smaller β; if no β is eligible there, the highest F1, flagged); `xgp_kd` enters the rule with that β.
- **Step 9A (2026-11-21):** round 2's epochs check gave E_ft = 20, so every fine-tuned method in the grid runs 20 epochs; the KD recipe is shared by l1_kd, snip_kd and park_kd. XGP's final version is designed after 9A (Section 3, Step 9): round 3, pre-registered below (2026-11-22).
- **Fairness rule for Step 9.** Whatever recipe and E_ft the final XGP uses, L1 and SNIP-score pruning get the same in the grid; Park runs as published and as park_kd. E_ft = 20 for every method in Step 9 if `l1_kd_e20` beats `l1_kd` (seed 0, same level) by more than 1 pt pooled F1 (normal windows) at either level; otherwise 10.
- `python -m har.run_xgp_pilot2 --analyze` merges both machines' files, applies the rule and writes the decision table.

#### Round 3 pre-registration

Written 2026-11-22, before any round-3 code existed and before any round-3 run.

**Amendment (2026-11-22, still before any round-3 run).** The older-like transform gains a per-sensor random 3D rotation (below), to address the posture limitation that the first version flagged. Phase 1 also runs with `--curve-seeds 0`; the full budget curve goes to 9B. The first `--prepare` output (same day, used by no run) was deleted and `--prepare` rerun, so the JSON and the window tables reflect the amended design. Its `amended` field records this.

`--prepare` copies the rule into `results/har-xgp-pilot/round3/har-xgp-pilot3-tinyhar-preregistration.json`, together with the SHA-256 of the window tables. This is the final design iteration of XGP and uses validation data only. Code: `har/xgp3.py`, `har/run_xgp_pilot3.py`. Output: `results/har-xgp-pilot/round3/`.

- **Disclosure (goes into the paper verbatim in substance).** The round-3 design was informed by the Step 9A test-set analysis (posture-class drift). All tuning uses inner-val and older-like windows only.
  - Without fine-tuning, the older adults' LCI fell more than the general adults' in the posture classes (`har-grid-a-per-class-lci-drift.csv`, l1): standing at 70% (gap −0.103, CI −0.185 to −0.022), sitting at 90% (−0.105, CI −0.188 to −0.018) and lying at 50% (−0.131, CI −0.222 to −0.040).
  - The accuracy gap widens at 90% (`har-grid-a-group-summary-gaps-bootstrap-ci.csv`): fine-tuned l1 −0.209 and l1_kd −0.172, against −0.11 to −0.14 at 30–70%.
  - Round-3 tuning and selection use only the HARTH inner-validation windows of folds 0 and 1 and their older-like copies. HAR70+ and the folds' test subjects are never loaded (`require_harth` on every window set). The 9A rows are not rerun or changed.
- **Why each component (the link to 9A).**
  - Older adults drift most when there is no fine-tuning, so the pruning step itself must keep the explanation. That motivates the keep-score, iterative re-scoring and the layer-wise reconstruction, which repairs the function without any training.
  - The drift is concentrated in some classes for one group, so the keep-score protects each class and each group separately (class-balanced within a group, max over groups).
  - The accuracy gap widens at 90%, so the fine-tuned variant adds older-like training windows and weights the worse group (group DRO).
- **Older-like augmentation: round 2's two steps plus a per-sensor rotation (amended); every parameter fixed a priori, none tuned.** `har/augment.py`, `older_like_r3`, applies three steps in order:
  1. time-stretch with speed s ~ U(0.8, 0.9);
  2. movement scaling a ~ U(0.7, 0.9) around the per-channel mean;
  3. **a per-sensor random 3D rotation.** The back triplet (channels 0–2) and the thigh triplet (3–5) are rotated independently. The axis is uniform on the unit sphere, the angle ~ U(−15°, 15°), and x'(t) = R x(t) at every time step, gravity included.

  s and a are drawn exactly as in round 2 (same seed and streams, so fold 0's values are identical to round 2's). The axes and angles come from two further, independent streams of the same seed. Round 2's `older_like` is unchanged.
  - Basis in the gait-ageing literature (PDFs not in `papers/` yet; check the exact tables before submission):
    - comfortable walking speed falls with age, by roughly 10–25% from the 20–40-year decades to the 70–80-year decades [Bohannon & Andrews 2011, *Physiotherapy* 97:182–189, meta-analysis tables by decade; Hollman et al. 2011, *Gait & Posture* 34:111–118];
    - healthy older adults walk with reduced joint excursion and push-off independent of speed (reduced hip extension and ankle plantar-flexion) [Kerrigan et al. 1998, *Arch Phys Med Rehabil* 79:317–322; Winter et al. 1990, *Phys Ther* 70:340–347];
    - their trunk accelerations are smaller and less rhythmic [Menz, Lord & Fitzpatrick 2003, *Age Ageing* 32:137–142; review: Kavanagh & Menz 2008, *Gait & Posture* 28:1–15].
    - rotation: Um et al. (2017, ICMI, "Data augmentation of wearable sensor data for Parkinson's disease monitoring using convolutional neural networks") use random rotation of the accelerometer signal to simulate different sensor placements and orientations. Here it stands for the differences in posture (e.g. trunk inclination) and in sensor fit between older and younger wearers. The ±15° bound is our a-priori choice for a small orientation change; it is not taken from the paper (PDF not in `papers/`; check the section on rotation before submission).
  - The ranges are our a-priori reading of these sources. They are not fitted to them, and not fitted to HAR70+.
  - **Known limitation (amended).** Steps 1–2 keep each window's per-channel mean (the gravity component), so on their own they left static postures (lying, sitting, standing) almost unchanged. That is where 9A saw the drift. The rotation (step 3) changes the gravity direction per sensor, so posture windows now change too. What remains a limitation:
    - the rotation is a generic orientation perturbation (random axis, at most 15°), not a model of how older adults actually sit, stand or lie;
    - if the 9A posture drift comes from something else (e.g. different movement within a posture, or label differences between the datasets), the proxy still misses it;
    - the ±15° bound is not validated against HAR70+, and must not be (HAR70+ is test-only).
- **Windows per fold (folds 0 and 1; `--prepare`, saved by window ID, seeded, CPU).**
  - Each inner-val window of the fold gets an older-like (s, a) drawn as in round 2 (seed 42), plus a back and a thigh rotation (axis, angle) from independent streams of seed 42 (`draw_older_like_r3_params`). All are saved in the window table.
  - Roles:
    - **explain**: 100 per subject, `select_explain_windows` as in round 2; fold 0's windows are identical to round 2's.
    - **calibration**: 70 per class, so 490 normal windows (`stratified_pick`, seed 44), drawn from the non-explain windows; a class with fewer than 70 gives all it has. Their older-like copies make 980 calibration windows.
    - **F1**: every inner-val window that is not a calibration window, plus its older-like copy.
  - Calibration windows are disjoint from the explain and F1 windows.
  - *Counts written by `--prepare` (2026-11-22).*
    - Calibration per class: fold 0 70 / 70 / 56 / 20 / 70 / 70 / 70 (426 + 426 older-like); fold 1 70 / 70 / 64 / 45 / 70 / 70 / 70 (459 + 459).
    - The stairs classes run short, so calibration takes every non-explain stairs window. In the F1 sets, stairs then come from the explain windows only (fold 0: 45 up / 45 down; fold 1: 42 / 42).
    - Fold 0's explain windows and older-like (s, a) equal round 2's (checked).
  - *Caveat.* They come from the same 3 subjects, so XGP's inner-val numbers can be slightly optimistic relative to baselines that never see inner-val windows. 9B's test numbers are not affected (calibration = inner val, test = other subjects).
- **XGP, round 3 (all on a deepcopy of the fold's frozen model; fine-tuning only in the fine-tuned variant).**
  1. **Calibration set**: the 980 calibration windows above, class-balanced in each group.
  2. **Group-aware keep-score.**
     - `I_{g,c}(w) = | w · ∂/∂w Σ_i ⟨A_θ(x_i), sg(A_θ0(x_i))⟩ |` on the calibration windows of group g ∈ {normal, older-like} and class c (true label).
     - A = 48-segment EG attribution of the original model's predicted class (m = 8; EG seeds (2s, 2s + 1) for run seed s, so seed 0 uses round 2's (0, 1)). The same EG draws are used for θ and θ0.
     - `I_g = mean_c I_{g,c} / mean_layer I_{g,c}` (class-balanced), and `I = max(I_normal, I_older-like)`.
     - `score = (1 − β)|w| / mean + β · I / mean`, both means over the tensor's surviving weights.
     - At θ0 with one group and one class, I equals round 2's `alignment_importance` (tested).
  3. **Iterative pruning.**
     - K = 5 steps to the target level s, with step k at sparsity s·k/5.
     - At each step the scores (|w| and I) are recomputed on the current model, after its reconstruction. Pruned weights stay pruned (nested masks).
     - No gradient training. Every tensor ends at exactly the uniform target, so n_nonzero equals L1's.
  4. **Layer-wise reconstruction (no gradient training), after every pruning step.**
     - Layer by layer in forward order, the kept weights and the bias of each refitted layer are refit by ridge least squares. Per output row i: `min ||w_iᵀ X̃ + b_i − y_i||² + λ ||w_i||²` over the row's kept weights.
     - X̃ = the layer's inputs in the current pruned model (earlier layers already refit). y = the original layer's outputs on the original model's inputs. Both are taken on the 980 calibration windows in eval mode.
     - λ = 0.01 · mean diag(X̃ X̃ᵀ), the 1% dampening of SparseGPT / GPTQ.
     - Refitted layers: TinyHAR's 4 Conv2d, the transformer feed-forward linear1 / linear2, the fusion Linear and the attention-score Linear.
     - Not refitted (masks only): the attention in/out projections (`out_proj` is applied without calling its forward, so its input cannot be hooked) and the LSTM. The head is unpruned.
     - Masks are fixed and n_nonzero is unchanged (asserted).
     - *Sources* (PDFs not in `papers/` yet; verify section and equation numbers): the layer-wise compression problem `argmin ||WX − ŴX||²` with a fixed sparsity pattern [OBC: Frantar, Singh & Alistarh 2022, NeurIPS, Sec 3, Eq. 1]; the row-wise exact least-squares reconstruction for a fixed mask, and the 1% Hessian dampening [SparseGPT: Frantar & Alistarh 2023, ICML, Sec 3 (Eq. 2) and the experimental setup].
     - *Deviation:* OBC and SparseGPT choose the mask and update the remaining weights jointly. Here the mask comes from the keep-score, and the reconstruction only refits.
  5. **Fine-tuned variant.**
     - The shared KD recipe (E_ft = 20, Adam lr 1e-4, batch 256, 0.7 KL(T = 3) T² + 0.3 class-weighted CE).
     - Training data: the HARTH training windows plus round(r · N) older-like copies. The indices are uniform without replacement; the indices and (s, a, rotation) are seeded with 1000 + seed.
     - The copies are **teacher-labelled**: their KD target is the original model's logits on the copy, and their CE label is the true activity label, which the transform does not change.
     - Group DRO over {normal, older-like} [Sagawa et al. 2020, ICLR, Algorithm 1 (online group DRO)]: per batch, `q_g ← q_g · exp(η ℓ_g)`, normalised, and loss = Σ q_g ℓ_g over the groups present. ℓ_g = the recipe's loss on group g's windows. η = 0.01; q starts uniform.
     - An epoch covers all N(1 + r) windows, so it has (1 + r) times l1_kd's steps. The control `l1_kdaug` gets the identical recipe.
     - The DRO weights q per epoch are logged.
- **Phase 1 (tuning)**: folds 0 and 1, levels 50 / 70 / 90%, seeds 0 / 1 / 2.
  - No fine-tuning:
    - `l1`, `snip` and `l1_recon` (one-shot L1 + the reconstruction on the same calibration set). These are deterministic, so they run once, as seed 0.
    - `xgp(β)` with β ∈ {0.1, 0.3, 0.5}.
  - Fine-tuned (E_ft = 20):
    - `l1` (CE), `l1_kd` and `snip_kd` (9A recipes);
    - `l1_kdaug(r)` = L1 start + the round-3 fine-tuning recipe;
    - `xgp_ft(β, r)` = the xgp(β) start of the same seed + that recipe.
    - r ∈ {0.25, 0.5}; seed = the batch-order and augmentation seed.
  - **Fine-tuning budget curve:** every phase-1 fine-tuned run is also assessed after 2 and 5 epochs, inside the same run (0 = its start, 20 = the final model). The RNG states are saved and restored around each assessment, so training is identical with or without the curve (tested). **Phase 1 runs with `--curve-seeds 0` (amended):** the 2- and 5-epoch points are taken for seed 0 only, and the full budget curve over all seeds goes to 9B. The rules use epoch 20 of every seed, so they are unaffected.
- **Phase 2 (ablations, the same budget: the same calibration set, K, E_ft, seeds, folds and levels, each at the per-level configuration that phase 1 selected).**
  - No fine-tuning:
    - `xgp_no_score` (β = 0: |w| only, still iterative and reconstructed);
    - `xgp_no_group` (I and the reconstruction on the 490 normal calibration windows only);
    - `xgp_no_iter` (K = 1);
    - `xgp_no_recon`.
  - Fine-tuned:
    - each of those four starts + the round-3 recipe(r*);
    - `xgp_ft_no_aug` (the XGP start + the plain shared KD, so aug and DRO removed);
    - `xgp_ft_no_dro` (KD + aug(r*), unweighted);
    - `l1_recon_ft` (l1_recon + the recipe(r*)).
  - These runs are assessed at epoch 20 only.
- **Metrics per run**, on each window set (normal, older-like):
  - LCI_val (mean over the 3 subjects) with ρ and Jaccard, and the worst-subject LCI;
  - pooled 7-class macro-F1 and accuracy on the F1 windows;
  - **FA-LCI_val = min(LCI_normal, LCI_older-like)** and **gap = LCI_normal − LCI_older-like**;
  - the F1 gap, n_nonzero and seconds.
  - SHAP as in round 2: GradientExplainer, nsamples 200, SHAP seed 0, target = the original model's predicted class on each window, background = the fold's 100 HARTH windows.
- **Pre-registered selection rule (phase 1; per scenario and level).**
  - B_set = the highest mean F1 on that window set among the scenario's baselines. The no-fine-tuning baselines are l1, snip and l1_recon; the fine-tuned baselines are l1, l1_kd, snip_kd and l1_kdaug (both r).
  - A configuration is **eligible** if its mean F1_normal ≥ B_normal − 1 pt **and** its mean F1_older-like ≥ B_older-like − 1 pt.
  - **Selected** = the eligible configuration with the highest mean FA-LCI_val (the mean over folds 0, 1 and seeds of each run's FA-LCI). Ties within 0.005 → the smaller β, then the smaller r.
  - If nothing is eligible at a level, there is no configuration there (flagged, no fallback), and phase 2 skips that level and scenario.
- **Pre-registered final rule (phase 2; per scenario).**
  - Candidates are the full XGP and its single-component ablations, each at the selected per-level configuration. A candidate must be eligible (the same F1 rule) at **every** level.
  - **Final** = the eligible candidate with the highest FA-LCI_val, averaged over the levels. Ties within 0.005 → fewer components, then the higher FA-LCI.
  - Components:
    - without fine-tuning, the full method has 4 (the I term, group-aware calibration, iterative, reconstruction);
    - fine-tuned, it has 6 (those 4 plus aug and DRO); `no_aug` removes both aug and DRO.
  - If no candidate is eligible, XGP is not frozen for that scenario: report and stop. Only measured variants can be chosen, never an untested combination.
  - Reported but not part of the rule: final − best baseline FA-LCI per level, with the sign per fold; the budget curve.
- **Machines:** `--folds` and `--machine-name` split the work. Fold 0 runs on uni-2080 and fold 1 on nakhoul-3050; both have the checkpoints. τ_k and the inner-val ceiling come from nakhoul-3050's fold-0 and fold-1 noise-floor files, which are in git.
- **Cost estimate** (from round 2: ≈ 80 s of SHAP per assessed model and 40–70 s per 20-epoch KD run): phase 1 ≈ 6.5 h per fold with `--curve-seeds 0` (≈ 9.5 h with the curve on every seed); phase 2 ≈ 3.5 h per fold. The rotation costs well under a second per window set.

*Round 1 (2026-11-20/21, superseded by round 2 for the method design):*
- **Keep-score** per weight: `score(w) = (1 − β) · |w| / mean_layer|w| + β · I(w) / mean_layer I(w)`. A tensor whose I is all zero keeps only the |w| term. With β = 0 it ranks exactly like L1 (tested).
- **Importance:** `I(w) = | w · ∂/∂w Σ_i ⟨ A_θ(x_i), sg(A_θ0(x_i)) ⟩ |` at θ0. This is a first-order (Taylor) estimate of how much removing `w` changes the alignment of the model's explanation with the original explanation.
  - `A` is the 48-segment EG attribution of the original predicted class (m = 8, `create_graph=True`), on **1,000 fixed class-stratified HARTH training windows** (`stratified_pick`, seed 42), in batches of 125. `sg` = stop-gradient.
  - At θ0, `sg(A_θ0)` is `A` itself with the same EG draws, so one EG pass per batch gives the alignment `Σ_i ⟨A_i, A_i.detach()⟩`.
  - I is **averaged over 2 EG seeds** (0, 1): mean of the per-seed |w · grad|.
  - I is computed **once per fold**, because it does not depend on the level or on β.
  - **Dot product on purpose:** the Pearson and cosine alignments are at their maximum (1) at θ0, so their gradient there is zero, and every weight would get I = 0. A test shows this (`test_pearson_and_cosine_alignment_have_zero_gradient_at_theta0`).
- **Repair loss:** `L_cons = (1 − mean_i Pearson(A_θ(x_i), sg(A_θ0(x_i)))) + (1 − cos(g_θ, g_θ0))`, EG m = 4.
  - `g` = mean over the batch of |A| per segment, the 48-dim global importance profile. It targets LCI's top-k / Jaccard term, which is not differentiable itself. The first term mirrors ρ̄.
  - eps = 1e-12 inside the square roots of Pearson and cosine, so an all-zero row gives r = 0 with finite gradients.
  - `A_θ0` on every training window (m = 4, EG seed 0) is precomputed once per fold and shared with Park: Park's targets are the same rows scaled to unit norm. During fine-tuning the EG draws come from a generator seeded with 1.
- **Gate.** After pruning and before fine-tuning, compute LCI_val with real SHAP (GradientExplainer, nsamples 200, SHAP seed 0) on the fold's inner-validation explain windows. If `LCI_val < τ_k`, fine-tune with `CE + γ · L_cons`; otherwise fine-tune with CE only. Both branches use `E_ft = 10` epochs. Log the gate value and the branch taken for every (fold, level), and in tuning for each β.
- **Gate metric = LCI_val** (decided 2026-11-19): per-subject LCI(original, pruned) averaged over the fold's 3 inner-validation HARTH subjects, compared with `τ_k` from the same subjects (Section 2.7). FA-LCI is not the gate because it needs HAR70+, which would put test data into a training decision.
- **Every attribution gradient** (I(w), L_cons, Park's L_AM) runs through `eg_attributions` / `attribution_grad_context`: math SDPA kernel and cuDNN off. Following Park's convention, CE is computed in train mode and the attribution term in eval mode.
- **Ablations** (70% and 90% only; same `E_ft`, same seeds, matched `n_nonzero`): `xgp_score_only` (keep-score, CE), `l1_plus_lcons` (L1 + always-on repair), `xgp_no_gate` (keep-score + always-on repair), `xgp` (full). Each uses the fold's chosen (β, γ). When the gate picks CE, `xgp` equals `xgp_score_only`; when it picks the penalty, `xgp` equals `xgp_no_gate`. Both rows are still reported.

**Tuning (round 1; replaced by the round-2 rule above for XGP; HARTH inner validation only, equal budget).** Park's γ_P ∈ {1, 50} (50 = the paper's SWA strength [Park App. C.2]; 1 = our choice, because unit-norm input attributions put the matching term on a different scale from the paper's), and ours (β, γ) ∈ {0.3, 0.7} × {0.1, 1.0}. Ours gets 4 configurations, Park 2; to equalise the budget, Park also gets 2 learning-rate variants {1e-4, 3e-4}. Tuning runs at 70% sparsity per fold.
- The chosen configuration maximizes inner-val LCI_val among those whose inner-val macro-F1 (pooled 7-class) is ≥ the L1 baseline's − 1 pt. Being better than L1 by more than 1 pt does not disqualify. It is then used at every sparsity in that fold.
- If no configuration qualifies (our choice; the plan gave no rule), take the highest macro-F1 and flag it as `selection_fallback`.
- When the gate of a β picks CE, γ has no effect, so its two configurations are one run, recorded for both (`shared_with`).
- Results are logged to the grid's per-fold model file. The fold-0 pilot (`python -m har.run_xgp_pilot`, `results/har-xgp-pilot/`) runs exactly this tuning for L1, SNIP-score pruning, Park and XGP before Step 9.

### 2.6 SHAP and the window version of LCI (`har/attributions.py`, `lci/metric.py`)

- Explainer: `shap.GradientExplainer(wrapper, background)` with `nsamples=200`. The background is the fold's 100 windows (Section 2.2), as a float32 tensor in raw g.
- **Target output (decided, Q4).** For window `x`, target class `c(x)` = **the original (uncompressed) model's predicted class** on `x` in that fold. It is the same for every compressed model, so LCI always compares attributions of the same output. Outputs are logits, as in the Phishing reproduction.
- **Single-output explanation.** Group the explain windows by `c(x)`, wrap the model as `x ↦ logits[:, c]`, and run GradientExplainer once per class group. Set `np.random.seed(shap_seed + c)` and `torch.manual_seed(shap_seed + c)` before each call, then reassemble in the original order. *Why:* explaining all 7 outputs costs 5.5–8.5× more (Section 5) and LCI uses only one output.
- Output per model: an `(n, 6, 128)` float32 array. Raw arrays are cached in `data/processed/har/shap-cache/` (git-ignored; ≈ 2 MB per model per fold). The 48-segment arrays (≈ 130 KB per model per fold) go in `results/` (Section 4). Reuse `lci.explain.AttributionCache`, extended so the config also holds `target_rule`, `nsamples`, `fold`, the background-window hash and the explain-window hash.

```python
def target_classes(original_model, X: np.ndarray) -> np.ndarray                         # (n,)
def shap_target_class(model, background: torch.Tensor, X: torch.Tensor, target: np.ndarray,
                      shap_seed: int, nsamples: int = 200) -> np.ndarray                 # (n, 6, 128)
```

**Features for LCI.** The main result uses **48 features: 6 axes × 8 time segments of 16 samples**, each the **signed sum** of the SHAP values in that segment. Summing keeps SHAP's additivity, and the sign keeps the meaning of the per-sample correlation. Feature index = `axis * 8 + segment`. The main result is repeated on the **768 raw values** (6 × 128). This reuses the cached arrays and needs no extra SHAP.

**LCI per subject, then per group.** For each subject, `compute_lci(A_T[subject windows], A_S[subject windows], k=10, omega=0.5)`, so top-10 uses the subject's own global importance over its ≤30 windows. `LCI_group` = mean over the group's subjects. The general group in a fold has ≈ 4–5 subjects; the older group always has 18.

**What changes in `lci/metric.py`.** `compute_lci` itself needs **no change**. It already flattens `(N, 6, 128)` to `(N, 768)`, and `k = 10 ≤ F` holds for both 48 and 768. The 12 existing tests must keep passing unchanged. Add three pure-numpy functions (no torch or shap import, so the existing import test still holds):

```python
def aggregate_time_segments(A, seg_len: int = 16) -> np.ndarray
    # (N, C, T) -> (N, C * T // seg_len), signed sum per (channel, segment), channel-major order.
    # Raises if T % seg_len != 0.
def compute_lci_by_subject(A_T, A_S, subjects, k: int = 10, omega: float = 0.5) -> dict[str, dict]
    # compute_lci on each subject's rows; same keys as compute_lci plus "subject".
def fa_lci(lci_older: float, lci_general: float, lam: float = 0.5) -> dict
    # {"fa_lci", "lci_all", "lci_older", "lci_general", "gap"}, with
    # lci_all = (lci_older + lci_general) / 2, gap = lci_older - lci_general (signed) and
    # fa_lci = lci_all - lam * |gap|. At the default lam = 0.5 this equals min(lci_older, lci_general).
```

Things to watch with windows:
- A heavily pruned model can return an all-zero attribution for a window. `compute_lci` skips it (std < 1e-8) and counts it in `n_skipped`; report `n_skipped` per group.
- **k (decided, Q5).** The main setting is k = 10, as in Tulla et al. and the Phishing run. Top-10 of 48 is 21% of the features (Phishing: 10 of 30 = 33%), so J₁₀ is stricter here. As a sensitivity check, the main LCI table is repeated with k = 16 (Phishing's 33% share) from the cached SHAP, with no extra SHAP runs.

### 2.7 FA-LCI, faithfulness, other metrics and the noise floor

**FA-LCI** = `LCI_all − λ · |LCI_older − LCI_general|`, with **λ = 0.5** (decided 2026-11-19).
- `LCI_all` = the **group-balanced** mean `(LCI_older + LCI_general) / 2`. With subject weighting, 18 older vs ≈ 4–5 general subjects per fold would make `LCI_all` mostly an older-group number.
- With these two choices, **FA-LCI = `min(LCI_older, LCI_general)`, the worst-group LCI.** Use that identity as a unit test.
- **Main value**, per (method, level), pooled over folds: `LCI_general` = mean over the 22 HARTH subjects, each from the fold where it is tested; `LCI_older` = mean over the 18 HAR70+ subjects, each first averaged over its 5 fold models. CIs come from the bootstrap in Section 2.8.
- **Per-fold value**, per (fold, method, level), from that fold's subjects: used for the safe explanation rate and the FA-LCI validation.
- Always report `lci_all`, `lci_older`, `lci_general`, the signed gap and which group is the worst next to FA-LCI. Report the subject-weighted `LCI_all` as well.

**Deletion faithfulness** (`har/faithfulness.py`). For each model, on its own SHAP:

```python
def background_mean(background: np.ndarray) -> np.ndarray                    # (6, 128)
def deletion_curve(model, X, A_seg, target, bg_mean, ks=range(0, 11)) -> np.ndarray   # (n, len(ks))
    # For each k: replace the k segments with the largest SHAP values for the target class
    # (descending, stable ties) by the background mean of the same (axis, time) positions;
    # record softmax probability of the target class.
def faithfulness_scores(curve: np.ndarray) -> dict
    # "drop_k5" = p(k=0) - p(k=5); "aopc" = mean over k = 1..10 of p(k=0) - p(k).
def random_deletion_curve(model, X, target, bg_mean, ks, n_repeats=5, seed=42) -> np.ndarray
```

"Predicted class" here is `c(x)`, the original model's prediction (same target as LCI). Report per subject, then per group, together with the random-deletion baseline.

**Other metrics, per subject then per group** (F1 definitions decided, Q11):
- accuracy per subject, on all of the subject's windows;
- **macro-F1 per subject over the 5 non-stair classes** (walking, shuffling, standing, sitting, lying) present in that subject, with the same definition in both groups so they can be compared;
- **stairs F1 pooled across subjects** (decided 2026-11-19): F1 for stairs up, stairs down and their mean, computed on all windows of a group pooled together, **never per subject**. Its CI comes from a bootstrap over subjects (resample subjects, pool their windows). Report it for both groups.
- pooled 7-class macro-F1 per group, as a secondary number;
- `n_nonzero`, the nominal sparsity, the actual total sparsity including the head (`sparsity_weights_incl_head`, `sparsity_all_params`), and `size_kb_dense` / `size_kb_nominal`;
- CPU inference time per window: batch 1, `torch.set_num_threads(1)`, median of 1,000 calls after 50 warm-up calls, plus batched throughput (batch 1,024, all threads).
  - Measure all models on **one machine** (Aya's) so the times are comparable.
  - Unstructured sparsity and fake quantization do not speed up dense PyTorch CPU kernels, so these times will be close to the original model's. Report them, and say so.

**Noise floor** (`har/run_noise_floor.py`), per fold:
- **Retrain noise:** frozen architecture retrained with seeds 1, 2, 3 (seed 0 is the main model) → LCI(seed 0, seed s).
- **SHAP noise:** seed-0 model with SHAP seeds 1 and 2 vs SHAP seed 0 → LCI(shap 0, shap s).
- Compute both on the inner-validation, general and older explain windows. Report per group (5 folds × 3 values for retrain, 5 × 2 for SHAP).
- **τ (data-driven, decided 2026-11-19)** = the lower end (minimum) of the retrain-noise LCI on the **inner-validation HARTH subjects**, per fold (`τ_k`), used by that fold's gate. The reporting threshold is `τ = min_k τ_k`. It is computed without HAR70+, so the gate stays leak-free.
  - The retrain-noise ranges on the general and older test windows are reported for comparison, not used as thresholds.
- Also report everything against **Tulla's τ = 0.85**.
- **Retention** (added 2026-11-21, pilot round 2): `(LCI − τ_k) / (ceiling_k − τ_k)`, with `ceiling_k` = the fold's inner-val SHAP-noise LCI mean. 0 = at the retrain-noise floor, 1 = as consistent as SHAP is with itself.
- **Implemented 2026-11-20** (`har/run_noise_floor.py --folds k`):
  - Seeds 1–3 retrain with the frozen setting (Adam lr 1e-3, weight decay 1e-4, ≤ 30 epochs, patience 5) and the fold's normalisation statistics. Seed 0 is the model-selection checkpoint.
  - `τ_k` = min over the 3 retrain seeds of LCI_val(seed 0, seed s), the same subject mean the gate uses.
  - Files are per fold and machine (fold 3 runs on both): `har-noise-floor-tinyhar-fold{k}-{machine}-{per-subject-lci, group-summary, tau, retrained-models}.csv`. Step 11 merges them into the two files of Section 4.
  - Retrained weights and SHAP arrays are cached, so a rerun reuses them.

### 2.8 Statistics (`har/stats.py`)

Unit = subject. **Pooling over folds** (decided 2026-11-19): a general (HARTH) subject has one value per (method, level), from the single fold in which it is tested. An older (HAR70+) subject has five, one per fold model, averaged into one value. That gives **40 subjects per (method, level): 22 general + 18 older**.

```python
def pool_over_folds(per_fold: pd.DataFrame, value: str) -> pd.DataFrame
    # (fold, method, level, subject, group, value) -> one row per (method, level, subject, group);
    # HARTH: its single test fold; HAR70+: mean over the 5 folds.
def paired_wilcoxon(a, b) -> dict
    # scipy.stats.wilcoxon, two-sided, zero_method="wilcox"; returns stat, p, n, median diff and
    # the matched-pairs rank-biserial correlation.
def holm(pvals: np.ndarray) -> np.ndarray   # Holm-Bonferroni adjusted p-values (numpy, no extra package)
def cohens_dz(a, b) -> float                # mean(a - b) / std(a - b, ddof=1)
def mann_whitney(x, y) -> dict              # scipy.stats.mannwhitneyu, two-sided; + rank-biserial r
def bootstrap_group_gap_ci(older, general, stat=np.mean, n_boot=10_000, seed=42) -> dict
    # Point estimate and percentile 95% CI of stat(older) - stat(general), resampling subjects
    # with replacement within each group.
def bootstrap_fa_lci_ci(older_A, general_A, older_B=None, general_B=None, lam=0.5,
                        n_boot=10_000, seed=42) -> dict
    # FA-LCI of method A with its 95% CI; if B is given, also the paired difference FA_A - FA_B,
    # using the SAME resampled subjects for both methods in each bootstrap draw.
def safe_rate(lci_values, tau) -> float     # share of settings with LCI >= tau
```

- **Methods: paired Wilcoxon per subject (primary).** For each metric, ours vs each of L1, SNIP-score pruning and Park at each sparsity. The pairs are the **40 subjects** (22 HARTH test + 18 HAR70+), each subject's value under method A vs method B, pooled over folds as above.
  - Metrics: LCI on the 48 segments, non-stair macro-F1 and AOPC.
  - Holm correction over the 12 comparisons (3 baselines × 4 levels) per metric.
  - Effect sizes with every test: the matched-pairs rank-biserial correlation (the Wilcoxon's own effect size) and Cohen's d_z.
  - As a secondary breakdown (decided, Q16), the same tests within each group (22 general, 18 older), Holm-corrected separately and labelled secondary.
  - PTQ (both variants) and the optional KD are described, not tested.
- **FA-LCI and the group gaps: bootstrap 95% CIs over subjects** (10,000 resamples, seed 42, resampling subjects within each group):
  - the group gap `older − general` per (method, level) for LCI, accuracy, non-stair macro-F1 and AOPC;
  - FA-LCI per (method, level);
  - the FA-LCI difference between ours and each baseline, as a paired bootstrap (the same resampled subjects for both methods);
  - pooled stairs F1 per group, with subjects resampled and their windows pooled.
  - Mann-Whitney U for older vs general is reported next to each gap CI as a secondary p-value; the CI is the result.
- **Walking-aid (inferred), exploratory:** 5 vs 13 older subjects, on windows of classes 1, 3, 6, 7, 8 only (indices `NON_STAIR`), with LCI, accuracy and F1 recomputed on those windows. Report the bootstrap CI of the gap and Mann-Whitney U. With n = 5 the test has very little power, so the effect size and CI are the result; make no confirmatory claim.
- **Safe explanation rate** per method = share of its settings (fold × level) with LCI ≥ τ. Report it for `LCI_general`, `LCI_older` and FA-LCI (per-fold values), at both τ (data-driven) and 0.85.
- **FA-LCI validation:**
  1. **Equality.** When `LCI_older == LCI_general`, FA-LCI equals LCI_all (unit test, and checked on the grid rows where the gap is 0).
  2. **Injected gap.** Per fold, take the original model's own SHAP as both teacher and student. Add Gaussian noise `σ · std(A)` to the student attributions of the **older** windows only, with σ ∈ {0, 0.1, 0.25, 0.5, 1, 2} and seed 42. Expected: `LCI_older` falls, `LCI_general` stays 1, and FA-LCI equals `LCI_older` and falls strictly, twice as fast as `LCI_all` (if `LCI_older = 1 − δ`, then `LCI_all = 1 − δ/2` and FA-LCI = `1 − δ`). Repeat with the noise on the general windows to check FA-LCI then follows `LCI_general`.
  3. **Faithfulness agreement.** Spearman ρ across all settings (fold × method × level) between FA-LCI and group-balanced AOPC, next to ρ for plain `LCI_all`. Also ρ between |LCI gap| and |AOPC gap|. Report the values; there is no pass threshold.

### 2.9 Model choice (once, then frozen)

- Same windows, classes, folds, class weights, normalization and `TrainConfig` for all candidates.
- **Equal tuning budget (decided, Q6):** 4 configurations per candidate (lr ∈ {1e-3, 3e-4} × width ∈ {paper default, half}; for XGBoost: depth ∈ {3, 6} × learning rate ∈ {0.1, 0.5}, where depth 3 / 0.1 is HARTH's [HARTH Sec 4.1] and depth 3 / 0.5 Ustad's [Ustad Sec 2.3.3], both with 1,024 trees, λ = 1, α = 0). The best one is then trained on all 5 folds (seed 0), so there are 4 + 5 = 9 trainings per candidate.
  - Tuning uses fold 0 only. Each configuration is trained on fold 0's training subjects minus its 3 inner-validation subjects, and scored on **those 3 inner-validation subjects only**.
  - **Fold 0's test subjects are never used in tuning**: not for training, early stopping, scoring or selection.
  - `run_model_selection` enforces this. It builds the tuning sets from `results/har-setup/har-inner-validation-subjects.csv` and `har-fold-assignment.csv`, asserts that the tuning-train, tuning-validation and fold-0 test subject sets are pairwise disjoint and that no HAR70+ subject is present, and logs the subject IDs of each set in `har-model-selection-tuning.csv`.
  - Fold 0's inner-validation subjects are test subjects in other folds. That is a small optimistic bias in held-out HARTH F1, the same for every candidate; the report states it.
- Per candidate, report:
  - held-out HARTH 7-class macro-F1 pooled over the 22 subjects' windows (the ranking metric, so stairs count without noisy per-subject stairs F1), per-subject non-stair macro-F1 and accuracy (mean ± sd over the 22 subjects), and pooled stairs F1;
  - parameter count;
  - CPU inference time per window (Section 2.7);
  - SHAP time per window (single target output, nsamples 200, background 100, measured on 100 windows on Aya's machine).
  - HAR70+ is not evaluated at this step.
- **Rule: cost-based** (decided 2026-11-19 after the source check, A.6.1; it replaces the earlier HART/DeepConvLSTM conditions and the cheap/costly grouping, whose costs came from stand-in models). F1 is the ranking metric above; the candidates are the four deep models.
  1. F\* = the best F1.
  2. Eligible = every candidate with F1 ≥ F\* − 2 pt.
  3. Choose the eligible candidate with the **lowest CPU SHAP time per window** (`har.time_model_selection_cpu`, one machine). An exact tie is broken by the smaller parameter count.
  - Implemented in `select_model.apply_rule`; the full trace is written into the frozen config.
- XGBoost is reported as the accuracy reference only.
- **Freezing:** write the architecture, configuration and training recipe to `results/har-model-selection/har-frozen-model-config.json`. Nothing about the model changes after this, whatever later steps show.
- **Sanity check after freezing** (uses HAR70+, so it comes after the freeze and cannot change it). Frozen model, 5 folds, walking-aid (inferred) vs other older subjects, reporting accuracy on two class sets:
  - Ustad's four classes (walking, standing, sitting, lying), for direct comparison;
  - the plan's non-stair classes (adding shuffling).
- **The Ustad figures** (corrected after reading the paper): Ustad et al. (2023) report 87% (walking aids, n = 5) vs 92% (no walking aids, n = 13) overall accuracy for the **HARTH model**. That is an XGBoost trained on HARTH only and tested on HAR70+, the same direction of transfer as here [Ustad Abstract; Sec 3.2, p7–8].
  - It was scored on those four classes, with 5-s windows and 161 features [Ustad Sec 2.3.2].
  - Their HAR70+-trained model gives 93% vs 95%.
- Compare the **direction and rough size** of the gap. The absolute values differ because the windows, features and models differ.

### 2.10 Tests (must pass before the step that uses the code)

`lci/tests/test_lci.py` (12 existing tests) must keep passing. New test files:
- **`lci/tests/test_lci_windows.py`** (written 2026-11-20 for `aggregate_time_segments` and `compute_lci_by_subject`; the `fa_lci` tests come with Step 11):
  - `aggregate_time_segments` sums correctly and keeps the sign;
  - channel-major index order;
  - raises when T is not divisible by `seg_len`;
  - LCI on `(N, 6, 128)` equals LCI on the same data reshaped to `(N, 768)`;
  - `compute_lci_by_subject` equals `compute_lci` run on each subject's slice;
  - `fa_lci`: equal groups give FA-LCI = LCI_all; at the default λ = 0.5, FA-LCI = min(LCI_older, LCI_general) on 1,000 random pairs (including negative LCI values); λ = 0 gives LCI_all; the signed gap and the worst group are reported correctly;
  - the injected-gap behaviour on synthetic attributions: with noise on the older group, FA-LCI equals LCI_older, is strictly decreasing in σ, and falls twice as fast as LCI_all; with noise on the general group, FA-LCI follows LCI_general.
- **`har/tests/test_data.py`:**
  - extra index columns are dropped;
  - a synthetic 100 Hz recording comes back from `to_50hz` with half the rows and an unchanged 2 Hz sine;
  - no window crosses a gap;
  - windows containing a non-shared label are dropped;
  - the majority rule: 65/128 kept, 64/64 dropped;
  - training step 64 and test step 128;
  - on the real data: 22 + 18 subjects, 7 classes; the HAR70+ subjects with no stairs rows are exactly the 5 walking-aid (inferred) subjects, and they have no stairs windows;
  - on the real data, every window of S006, S010, S015, S023, S026 and 507 re-derived from the raw file: identical values, no step > 40 ms inside, only shared labels, majority label > 50%.
- **`har/tests/test_folds.py`:**
  - every HARTH subject is in exactly one test fold;
  - fold sizes are 5, 5, 4, 4, 4;
  - no subject is in both train and test, and the inner-val subjects are disjoint from test;
  - no HAR70+ subject appears in any training, validation or background set;
  - the model-selection tuning sets for fold 0 (tuning-train = fold 0's training subjects minus its inner-val subjects; tuning-validation = its 3 inner-val subjects) are pairwise disjoint from each other and from fold 0's test subjects;
  - explain windows are ≤ 30 per subject and identical across two calls;
  - background windows come from training subjects only.
- **`har/tests/test_models.py`:** per candidate: output `(B, 7)`, parameter count logged, forward + backward pass, eval-mode determinism, and a SHAP call on 2 windows giving `(2, 6, 128)`.
- **`har/tests/test_train.py`:**
  - `require_harth` raises for HAR70+;
  - per-subject macro-F1 covers only the non-stair classes present in the subject;
  - `pooled_class_f1` equals sklearn's per-class F1 on the concatenated windows and never runs per subject;
  - class weights have mean 1 and are computed from training-subject windows only;
  - `channel_stats` uses only the given training subjects: adding, removing or changing inner-val, test or HAR70+ windows leaves the stored mean and std unchanged, and passing an inner-val or test subject as a training subject raises;
  - a model applies its stored statistics unchanged to HAR70+ input (no re-normalization);
  - a 1-epoch run on a tiny subset is reproducible with the same seed.
- **`har/tests/test_compression.py`:**
  - at each sparsity, L1, SNIP-score pruning, Park and ours give identical `n_nonzero`;
  - the head is never pruned; `sparsity_prunable` equals the nominal level exactly, and `sparsity_weights_incl_head` and `sparsity_all_params` match a hand count on a toy network;
  - pruned weights stay zero after `finetune`;
  - `ptq_symmetric`: every quantized tensor has at most 2^b − 1 distinct values, max |W_q| = max |W|, zero maps to zero, biases and normalization parameters are unchanged, and the head is quantized;
  - `ptq_tulla` matches `lci.compression.ptq` exactly;
  - SNIP-score pruning scores equal |w·g| on a 2-layer toy network;
  - `prune_by_score` removes exactly round(s · numel) per tensor, lowest score first, ties at the lowest flat index; it rejects scores that do not cover exactly the prunable tensors;
  - `finetune` runs exactly `E_ft` epochs, raises on HAR70+ train or val, and keeps pruned weights at zero on TinyHAR (Conv, LSTM, attention) with CE alone and with an attribution term.
- **`har/tests/test_xgp.py`** (implemented 2026-11-20):
  - with β = 0 the keep-score ranks like L1 at every level (identical masks); β = 1 ranks like I(w); the formula and the all-zero-I guard on a toy network;
  - I(w) equals |w · ∂/∂w Σ⟨A, sg(A)⟩| recomputed by hand; it is the mean over EG seeds, reproducible, leaves the model untouched, and covers every prunable tensor of TinyHAR;
  - the Pearson and cosine alignments have zero gradient at θ0 while the dot product does not (why I(w) uses the dot product);
  - the reference windows are fixed, class-stratified and HARTH only;
  - `pearson_rows` matches numpy and stays finite on all-zero rows; the parts of L_cons (affine invariance, sign flip, profile term); L_cons is 0 for the original model and differentiable after a perturbation;
  - the gate picks the penalty branch exactly when gate < τ (equality → CE);
  - the tuning selection rule, including the fallback;
  - at each sparsity, every method and ablation (L1, SNIP-score pruning, Park, `xgp_score_only`, `l1_plus_lcons`, `xgp_no_gate`, `xgp`, both β) gives identical `n_nonzero`;
  - the pilot's source loads HARTH windows only.
- **`har/tests/test_noise_floor.py`:** `τ_k` = min over retrain seeds of the inner-val subject mean, unchanged by HAR70+, general or SHAP-noise rows; the per-group summary and the SHAP-above-retrain check.
- `har/tests/test_attributions.py` also tests the SHAP cache (hit on the same config; new entry for another SHAP seed or other weights).
- **`har/tests/test_attributions.py`:**
  - single-output SHAP for a linear model satisfies completeness (Σ SHAP ≈ f_c(x) − mean f_c(background), tolerance 1e-3);
  - per-class grouping returns rows in the original order;
  - `eg_attributions` is differentiable (`create_graph=True` gives non-zero second-order gradients).
- **`har/tests/test_faithfulness.py`:** k = 0 leaves the input unchanged; replaced segments equal the background mean; on a model whose output depends on one segment only, deletion removes the probability mass at k = 1.
- **`har/tests/test_stats.py`:**
  - Holm matches a hand-worked example;
  - d_z and the matched-pairs rank-biserial correlation on known data;
  - `pool_over_folds` takes HARTH values from the single test fold and averages HAR70+ over 5 folds;
  - the bootstrap CIs cover the true gap on a synthetic example;
  - the paired FA-LCI bootstrap uses the same subject draws for both methods (identical methods give a difference of exactly 0);
  - `safe_rate` boundaries.

---

## 3. Next steps (run in this order)

Each step ends with an acceptance check. If a check fails, stop, fix the code (never the data or a threshold), and rerun from that step.

0. **Environment.** Add `xgboost` to the environment (pin the installed version in `environment/requirements.md` and `environment/requirements.txt`) and `data/processed/` to `.gitignore`. Use the same `requirements.txt` on both machines.
   *Acceptance:* `python data/verify_checksums.py` prints 44/44 on both machines; `pip freeze` is identical on both.

1. **Build the windows** (`python -m har.build_windows`). *Status 2026-11-19: done, together with all of Step 2 (folds, validation subjects, normalisation statistics, explain windows and SHAP backgrounds).*
   *Why:* every later step reads the same cached windows, so windowing cannot differ between machines or runs.
   *Acceptance:* `har/tests/test_data.py` passes; S006 resampled to 50 Hz; 22 + 18 subjects; only class indices 0–6; window counts per subject written to `results/har-setup/har-window-counts-per-subject.csv` and within ±5% of Section 1's estimates (larger differences are explained in the report); the walking-aid (inferred) subjects have zero stairs windows.

2. **Folds, inner validation, explain windows, backgrounds** (part of `build_windows`, written to `results/har-setup/`).
   *Why:* fixed once, so every method and both machines use identical subjects and windows.
   *Acceptance:* `har/tests/test_folds.py` passes; no subject overlap; each fold's test set has all 7 classes in HARTH; ≈ 670 explain windows per fold.

3. **Metric extensions and their tests** (`lci/metric.py`, `lci/tests/test_lci_windows.py`).
   *Why:* as in the LCI plan, the metric is pinned down before any model exists, so later surprises point at models, not the metric.
   *Acceptance:* the 12 old tests and all new metric tests pass.

4. **Model selection** (`python -m har.run_model_selection`, `python -m har.time_model_selection_cpu`, `python -m har.select_model`).
   *Why:* one fixed architecture for the whole grid, chosen on HARTH alone, cheap enough to explain on CPU.
   *Status 2026-11-19: implemented and smoke-tested on CPU; the full run is to be done on GPU machines. Training of all 5 candidates is done (`--stage all`, tuning + folds 0–4, in `results/har-model-selection/`): CNN1D + TinyHAR on Nakhoul's RTX 3050 Ti (21.6 min), XGBoost on Aya's CPU (8.7 min), HART + DeepConvLSTM on Nakhoul's RTX 3050 Ti (83.2 min). Still to do: commands 3–4 below (CPU timing on Aya's laptop, then `select_model`). Each machine runs `--stage all` for its own models, instead of the fold split in Commands below.*
   - **Code:** `har/models.py`, `har/features.py`, `har/train.py`, `har/attributions.py` (single-output SHAP), `har/runlog.py`, plus the three runners.
   - **HAR70+ is never loaded:** `data.load_harth_windows` reads only the HARTH arrays of the cache, and the folds come from `har-fold-assignment.csv`.
   - **Training** may run on GPU (`--device cuda`). CPU inference and SHAP time are measured separately, always on CPU (`time_model_selection_cpu`, on Aya's machine for all models).
   - **Early stopping and tuning** use the pooled 7-class macro-F1 on the validation subjects' non-overlapping windows.
   - **Sources:**
     - All four candidates and the XGBoost reference were checked against the PDFs in `papers/` on 2026-11-19 (Appendix A). Every value in `har/models.py` and `har/features.py` is cited or marked "our choice".
     - This changed the CNN (now HARTH's 4 × 128-kernel CNN, 1.25M parameters), TinyHAR (stride 2 in every layer, F = 28, 1 head), DeepConvLSTM (orthogonal initialisation) and the XGBoost features and grid (HARTH's feature set and settings).
   - **Commands** (each machine first follows the GPU setup in `environment/requirements.md`):
     1. Tuning (fold 0, 20 units), split by model: A `--stage tune --device cuda --models hart deepconvlstm`; B `--stage tune --device cuda --models cnn1d tinyhar` and `--stage tune --device cpu --models xgboost`. Exchange `results/har-model-selection/tuning/` (commit and pull).
     2. Final folds: A `--stage final --device cuda --folds 0 1 2 --models cnn1d tinyhar deepconvlstm hart` and B the same with `--folds 3 4`. XGBoost runs on CPU with `--device cpu --models xgboost --folds …`.
     3. Merge: commit and pull both machines' `results/har-model-selection/` (file names carry the machine name, so they never collide). Copy the fold-0 checkpoints `data/processed/har/checkpoints/har-model-selection/har-final-*-fold0-*` to Aya's laptop.
     4. On Aya's laptop: `python -m har.time_model_selection_cpu`, `python -m har.select_model --merge-only` (completeness check), then `python -m har.select_model` (rule + freeze).
   *Acceptance:*
   - all candidates trained with the same budget;
   - **tuning check:** `har-model-selection-tuning.csv` lists, per configuration, the tuning-train and tuning-validation subject IDs. The tuning-validation IDs are exactly fold 0's 3 inner-validation subjects. Neither set contains any of fold 0's test subjects or any HAR70+ subject. The run aborts otherwise;
   - the summary table is complete;
   - the cost-based rule in Section 2.9 is applied (lowest CPU SHAP time per window among the candidates within 2 pt of the best pooled macro-F1; parameter count breaks exact ties), and its trace is logged;
   - the frozen config is written;
   - the chosen model's held-out HARTH macro-F1 is within 2 pt of XGBoost's, or the gap is reported.

5. **Sanity check: walking-aid (inferred) vs other older** (`python -m har.run_sanity_walking_aid`).
   *Why:* if the frozen model does not show the known gap direction, the subgroup or the data handling is suspect.
   *Acceptance:* the gap (other older − walking-aid) is positive. Report its size and bootstrap CI, on Ustad's four classes, next to Ustad et al.'s 5 pt for their HARTH-trained model (87% vs 92%). If it is not positive, re-check the subject IDs and the class filter. The model is not changed.

6. **Timing benchmark of the frozen model on both machines** (`python -m har.time_har_components --fold 0`).
   *Why:* the run-time table in Section 5 uses stand-in models; the real per-epoch and per-window SHAP costs decide the fold split.
   *Acceptance:* times written to `environment/time.md` for both machines; Section 5's estimates and the fold split updated if they are off by more than 30%.

7. **Cross-machine reproducibility check.** Train fold 0, seed 0 of the frozen model on both machines.
   *Why:* folds will be split across the two machines, so results must not depend on the machine.
   *Acceptance:* |Δ macro-F1| ≤ 0.5 pt, and LCI(model on Nakhoul's, model on Aya's) ≥ the SHAP-noise floor of Step 8. If bitwise identical, say so. Otherwise record per fold which machine produced it.

8. **Noise floor** (`python -m har.run_noise_floor --folds ...`).
   *Why:* defines what counts as a real explanation change (τ) before any compression result is seen, and gives the gate its threshold.
   *Acceptance:* per group, the SHAP-noise LCI is above the retrain-noise LCI; `τ_k` for each fold and the global τ are written; τ is reported next to 0.85.

9. **Compression grid** (`python -m har.run_compression_grid --folds ...`, folds split across machines, Section 5).
   **New order (decided 2026-11-21).** Step 9 is split. **9A** runs the standard methods only (PTQ, L1, SNIP-score pruning, Park, each with CE and the shared KD recipe) on the real test windows and measures explanation drift and its subgroup gap (RQ1, RQ2; grid data for RQ0; RQs defined at the top). **Then XGP's final version is designed** for the drift 9A shows (which classes, which levels, which groups), and **9B** adds it to the same grid as one more registered method. The earlier round-3 idea (shift-aware + worst-case KD, designed before any test-set drift was seen) is replaced by the round-3 pre-registration in Section 2.5 (b) (2026-11-22), designed after 9A.
   *Why:* round 2 showed XGP's advantage disappears after fine-tuning and reverses on older-like proxy windows; the proxies are a guess at the shift. Measuring where real drift happens first tells us what XGP must fix, instead of tuning XGP against a proxy. *Cost to disclose:* XGP is then designed after seeing test-set behaviour of the baselines. XGP's own tuning still uses HARTH inner validation only (never HAR70+ or test subjects), its design choice is pre-registered before 9B runs, and the paper states the order. The baselines' numbers do not change, because 9B reuses 9A's rows.
   **9A specification** (`har/run_compression_grid.py`, implemented and smoke-tested 2026-11-21; per fold, folds split as in STATUS.md):
   - Models: original; `ptq8`, `ptq4` (per-tensor symmetric, primary) and `ptq_tulla8`, `ptq_tulla4` (secondary); pruning at 30/50/70/90% with identical `n_nonzero`: `l1` (CE), `l1_kd`, `snip` (CE), `snip_kd` at all levels, `park` and `park_kd` at 70% and 90% only. KD = 0.7 · KL(T = 3) · T² + 0.3 · class-weighted CE vs the original model's logits. **E_ft = 20** for every fine-tuned method (round 2's epochs check), Adam lr 1e-4, seed 0, γ_P = 1, EG m = 4.
   - Scenarios: `no_finetune` = each score's pruned start (`l1`, `snip`; Park and the KD variants start from the L1 start, so their no_finetune model is the `l1` row) and `finetune`.
   - Explain windows: the fold's fixed windows; inner val (kept for XGP's validation-only design), general = the fold's HARTH test subjects, older = all 18 HAR70+; walking-aid (inferred) = 503, 505, 513, 514, 515. SHAP is called per group exactly as in the noise floor, so the original model's SHAP is the noise floor's cached array.
   - Per subject: LCI (48 segments, k = 10) with ρ and Jaccard, raw-768 LCI, k = 16 LCI, retention = (LCI − τ_k) / (ceiling_g − τ_k) with τ_k and the **group's** SHAP-noise mean from the noise-floor file of the machine running the fold, deletion faithfulness (AOPC, drop_k5, random baseline), accuracy, non-stair macro-F1 and the confusion matrix (stairs F1 only pooled). Per class (true label): LCI per (subject, class) with ≥ 3 windows and per (group, class) on pooled windows. The noise-floor models (SHAP seeds 1–2, retrain seeds 1–3) are scored with the same code, giving a per-class noise reference.
   - Saved after every model; `--resume` skips finished model IDs. Fine-tuned weights go to `data/processed/har/checkpoints/har-compression-grid/` (git-ignored).
   - `--analyze`: per method × level × scenario, group LCI with bootstrap CIs over subjects, FA-LCI with CI, the older − general gap (bootstrap CI, Mann-Whitney U), walking-aid (inferred) − other older (exploratory), paired Wilcoxon per subject between every pair of pruning methods within (scenario, level) and between each fine-tuned method and its start (40 subjects primary, per group secondary; Holm per metric × comparison × scenario × population), safe-explanation rate (fold settings and subjects; τ_k, min τ_k, 0.85), the per-class drift table and pooled stairs F1. The XGP-vs-baseline tests of Section 2.8 come with 9B.
   - Fold 3 runs on both machines; **the analysis uses uni-2080's copy** (decided 2026-11-21; `FOLD_MACHINE_DEFAULT`, overridable with `--use-machine`). nakhoul-3050's copy is for the Step 9b comparison only.
   *Why:* the core experiment: every method at matched size, same folds, same explain windows.
   *Acceptance, per fold:*
   - identical `n_nonzero` across L1, SNIP-score pruning, Park and ours at each level;
   - the head is unpruned, and the nominal and actual total sparsity are both recorded. *Check fixed 2026-11-22:* `head_unpruned` = the head has no mask and no zeroed weight (`compression.head_unpruned`). The old check (head unchanged) was always False after fine-tuning, because the head trains. `--recheck-head` recomputes the column of finished folds from the saved checkpoints;
   - every pruning method ran exactly `E_ft` epochs;
   - the tuning and gate logs show HARTH subjects only;
   - the primary PTQ-8 (per-tensor symmetric) has ρ̄ > 0.99 per group and accuracy within 0.5 pt of the original; the Tulla-style rows are present and labelled;
   - `n_skipped` is reported;
   - the per-fold files are written and their SHAP configs match the cache.

9b. **Cross-machine fold check (decided, Q15).** Run one complete fold, fold 3, on **both** machines with identical code, `requirements.txt` and inputs. Fold 3 is normally Aya's; Nakhoul's machine runs it as an extra fold (Section 5). The whole fold is rerun: noise floor, tuning, gate and grid. Compare the two runs with `python -m har.analyze --compare-machines --fold 3`.
   *Why:* Step 7 compares one trained model; this checks that a whole fold's results, including tuning and gate decisions, do not depend on the machine.
   *Acceptance:*
   - the same tuning choice and gate branch for every level;
   - per (method, level, group): |Δ group-mean LCI| ≤ the fold's SHAP-noise spread (max − min of the SHAP-noise LCI), and |Δ non-stair macro-F1| ≤ 0.5 pt;
   - the same method ranking by LCI at every level;
   - `τ_3` agrees within the SHAP-noise spread.
   - If the results are bitwise identical, say so.
   - The comparison goes to `results/har-compression-grid/har-grid-fold3-cross-machine-comparison.csv`, and the analysis uses the run from Aya's machine (the assigned one).
   - If the check passes, recording the machine per fold is enough. If it fails, the folds are rerun on one machine before Step 11.

10. **KD (optional, not in the main grid, Q14).** Only if time is left after Steps 9 and 9b (≈ 5% of a fold's cost per fold). Reported descriptively; not in any statistical comparison.
    *Acceptance:* student params within 70 ± 5% of the original.

11. **Analysis** (`python -m har.analyze`). Merge folds, pool each subject over folds, run Section 2.8 (Wilcoxon over the 40 subjects, Holm, effect sizes, bootstrap CIs for FA-LCI and the group gaps), the FA-LCI validation, faithfulness summaries, pooled stairs F1, the raw-768 repeat of the main LCI table, the k = 16 sensitivity check, the single-label-window robustness check and the walking-aid (inferred) comparison.
    *Acceptance:*
    - every Wilcoxon test has n = 40 pairs (22 + 18), with a Holm-adjusted p-value and both effect sizes; no Tulla-style PTQ or KD row enters a test;
    - every pruned model reports both the nominal sparsity and the actual total sparsity including the head;
    - all three FA-LCI validation checks are reported; (1) and (2) behave as expected, and FA-LCI equals the worst-group LCI in every row;
    - no per-subject stairs F1 appears in any output;
    - the raw-768 and 48-segment LCI rank the methods in the same order at each level, or the differences are reported;
    - every number in the report traces back to a CSV in `results/`.

12. **Write-up.** Write `results/har-analysis/har-report.md` and update `results/README.md`, `environment/requirements.md` and `environment/time.md` (CLAUDE.md).

---

## 4. Deliverables and output files

Code: the `har/` package, the `lci/metric.py` additions, and all tests from Section 2.10, passing.

Results (names follow CLAUDE.md; every file gets an entry in `results/README.md` in the same step it is written):

| Folder | Files |
|---|---|
| `results/har-setup/` | `har-window-counts-per-subject.csv` (single-label / mixed-kept / mixed-dropped per subject and split), `har-window-counts-per-group-and-class.csv`, `har-dropped-data-summary.csv`, `har-fold-assignment.csv`, `har-inner-validation-subjects.csv`, `har-normalisation-stats-per-fold.csv`, `har-explain-windows.csv`, `har-explain-windows-per-subject-class-and-group.csv`, `har-shap-background-windows.csv`, `har-explain-and-background-windows-per-fold.csv` |
| `results/har-model-selection/` | `har-model-selection-tuning.csv`, `har-model-selection-per-subject-metrics.csv`, `har-model-selection-summary.csv` (F1, params, inference ms, SHAP s/window), `har-frozen-model-config.json`, `har-walking-aid-inferred-sanity-check.csv` |
| `results/har-noise-floor/` | per fold k and machine: `har-noise-floor-tinyhar-fold{k}-{machine}-per-subject-lci.csv`, `-group-summary.csv`, `-tau.csv`, `-retrained-models.csv`, `-config.json`; merged in Step 11 into `har-noise-floor-per-subject-lci.csv`, `har-noise-floor-tau-per-fold.csv` |
| `results/har-xgp-pilot/` | `har-xgp-pilot-tinyhar-fold0-70pct-inner-val-{machine}-runs.csv` (one row per run: gate value and branch, pre/post-fine-tuning LCI_val, inner-val F1, n_nonzero, selected), `-per-subject-lci.csv`, `-finetune-history.csv`, `-config.json` |
| `results/har-compression-grid/` | per fold k: `har-grid-fold{k}-per-subject-metrics.csv` (fold, method, level, subject, group, accuracy, macro-F1, LCI48, LCI768, shap_corr, jaccard, n_skipped, mprf, AOPC, drop_k5, random AOPC), `har-grid-fold{k}-models.csv` (n_params, n_nonzero, nominal sparsity, actual total sparsity incl. head, per-layer sparsity, sizes, inference time, tuning choice, gate value and branch, machine), `har-grid-fold{k}-shap-48-segments.npz` + `-configs.json`; `har-grid-fold3-cross-machine-comparison.csv` (Step 9b) |
| `results/har-analysis/` | `har-per-subject-pooled-over-folds.csv` (one row per method, level, subject), `har-summary-per-method-level-group.csv`, `har-fa-lci-pooled-bootstrap-ci.csv` (FA-LCI, its parts, worst group, CI and paired differences vs ours), `har-fa-lci-per-fold-method-level.csv`, `har-method-comparisons-wilcoxon-holm.csv` (40-subject primary + per-group secondary), `har-group-gaps-older-vs-general-bootstrap-ci.csv`, `har-stairs-f1-pooled-per-group.csv`, `har-subgroup-walking-aid-inferred.csv`, `har-safe-explanation-rate.csv`, `har-fa-lci-validation.csv`, `har-raw-768-vs-48-segment-lci.csv`, `har-lci-k16-sensitivity.csv`, `har-single-label-windows-robustness.csv`, `har-report.md` |

Git-ignored, rebuildable: `data/processed/har/` (window cache, raw 768-value SHAP cache, model state dicts).

`environment/requirements.md` gets xgboost and every command run, with the machine it ran on; `environment/time.md` gets Step 6's benchmark and the wall-clock time of Steps 4, 8, 9, 9b and 11 per machine.

---

## 5. Estimated CPU run time and the machine split

**How the estimates were made.**
- **Per-window costs** were measured on Aya's machine on 2026-11-19 with rough stand-ins of the four candidates (not the final architectures) on random `(6, 128)` windows, 10 threads, shap 0.51.0, torch 2.15.0+cpu. *These planning estimates predate the source check and the switch to torch 2.7.1; measured costs of the real models are in Appendix A.6 and `environment/time.md`, and Step 6 re-measures before the grid.*
- **Nakhoul's machine** is scaled from the same Phishing pipeline (`python -m lci.time_lci_phishing_components --seed 0`) run on both machines: teacher training 17.1 s vs 8.1 s (**2.1×**), SHAP 174.1 s vs 131.4 s (**1.32×**), total 194.6 s vs 141.6 s.
- **Uncertainty is about ±50%.** The final architectures will differ in size, the CPU ratio was measured on an MLP and not on convolutions or LSTMs, and early stopping will cut the model-selection times below these upper bounds. Step 6 replaces these numbers with measured ones.

| Stand-in (Aya's machine) | Params | Train, ms / window | Inference bs 1, ms | SHAP, 1 output, s / window | SHAP, all 7 outputs, s / window |
|---|---|---|---|---|---|
| CNN1D | 43,911 | 0.20 | 0.36 | 0.083 | 0.457 |
| DeepConvLSTM | 295,943 | 2.71 | 5.05 | 0.494 | 4.213 |
| TinyHAR | 45,511 | 1.05 | 2.27 | 0.227 | 1.351 |
| HART | 305,927 | 1.00 | 3.27 | 0.147 | 1.043 |

**Work per fold** (≈ 56K training windows per fold after removing test and inner-val subjects; ≈ 670 explained windows):

| Part | Training (epoch-equivalents) | SHAP (windows) |
|---|---|---|
| Noise floor: 4 trainings (seeds 0–3) × 30 epochs; SHAP: 4 models + 2 SHAP seeds, plus inner-val windows | 120 | 4,380 |
| L1 4 × 10, SNIP-score pruning 4 × 10 | 80 | 8 × 670 |
| Park: 4 × 10 epochs at ≈ 3× cost (double backprop) | 120 | 4 × 670 |
| Ours: 4 × 10 epochs at ≈ 4× cost if every level is gated (upper bound), plus the gate SHAP on inner val | 160 | 4 × 670 + 450 |
| Tuning (Park + ours, 4 configurations each at 70%, the same costs) | 160 | 360 |
| PTQ 8 / 4, per-tensor symmetric (primary) and Tulla-style (secondary) | 0 | 4 × 670 |
| **Total per fold (main grid)** | **640** | **≈ 18,600** |
| KD (optional, not in the main grid) | +40 | +670 |

Deletion faithfulness (≈ 11 forward passes per window per model), statistics and the bootstrap take minutes and are left out of the table.

**Estimated time per step** (hours unless marked; A = Aya's machine, N = Nakhoul's machine):

| Step | CNN1D A | CNN1D N | HART A | HART N | Notes |
|---|---|---|---|---|---|
| 1–2 Windows, folds | ≈ 5 min | ≈ 3 min | same | same | Parsing ≈ 10.6M CSV rows; not measured |
| 3 Tests | < 1 min | < 1 min | | | |
| 4 Model selection, all candidates | see below | | | | 9 trainings × ≤ 30 epochs each |
| 6 Timing benchmark | ≈ 15 min | ≈ 10 min | ≈ 30 min | ≈ 20 min | |
| 7 Cross-machine check | 0.1 | 0.05 | 0.5 | 0.25 | 1 training each |
| 8 Noise floor, per fold | 0.5 | 0.25 | 2.0 | 1.0 | |
| 9 Grid incl. tuning, per fold (main grid, both PTQ variants, no KD) | 2.0 | 1.0 | 8.7 | 4.3 | |
| 8 + 9, all 5 folds on one machine | 12.2 | 6.4 | 54 | 27 | |
| 9b Cross-machine fold check (fold 3 again, on Nakhoul's) | – | 1.3 | – | 5.3 | One extra fold on the faster machine |
| 11 Analysis | ≈ 15 min | ≈ 10 min | ≈ 20 min | ≈ 15 min | 10,000-sample bootstraps |

Model selection (Step 4), upper bound at 30 epochs, A / N: CNN1D 0.85 / 0.4 h, DeepConvLSTM 11.4 / 5.4 h, TinyHAR 4.4 / 2.1 h, HART 4.2 / 2.0 h, XGBoost < 0.5 h (not measured). If DeepConvLSTM or TinyHAR were frozen, Steps 8 + 9 would take 29.5 / 14.8 h (DeepConvLSTM) or 11.6 / 5.9 h (TinyHAR) per fold.

**GPU estimates with XGP and its ablations (2026-11-20).** These are built from the Step 6 measurements on fold 0 (`results/har-component-timing/`):

| Cost | nakhoul-3050 | uni-2080 |
|---|---|---|
| CE epoch | 2.5 s | 1.5 s |
| Attribution epoch (Park's L_AM or L_cons, EG m = 4, double backprop) | 52 s | 66 s |
| SHAP, per window | 0.14 s | 0.13 s |

- **L_cons cost per epoch = Park's** (same EG pass; Pearson and cosine are negligible). This is still an estimate until the pilot measures it.
- **I(w):** 2 seeds × 1,000 windows × m = 8, about 8% of one Park epoch's EG work, so ≈ 5 s, once per fold.
- **Gate:** 90 inner-val windows per gate.
- **Steps 8 + 9 per fold:** the Step 6 projection (3.20 h / 3.73 h, where ours is charged at Park's cost) plus the ablations (an upper bound, every ablation run separately):

  | Ablation work per fold | nakhoul-3050 | uni-2080 |
  |---|---|---|
  | `l1_plus_lcons` + `xgp_no_gate`: 2 × 2 levels × 10 attribution epochs | 35 min | 44 min |
  | `xgp_score_only`: 2 × 10 CE epochs | 1 min | 0.5 min |
  | SHAP for the 6 extra models: 6 × 660 windows | 9 min | 9 min |
  | **Total per fold** | **≈ 3.95 h** | **≈ 4.6 h** |

- **Fold split** (STATUS.md): nakhoul-3050 runs folds 0, 1, 3 (≈ 11.9 h) and uni-2080 runs folds 2, 3, 4 (≈ 13.9 h).
- **Before Step 9:**
  - Noise floor, fold 0: ≤ 90 CE epochs + 6 × 750 SHAP windows, ≈ 15 min on nakhoul-3050.
  - Pilot, fold 0: 4 Park + up to 4 XGP fine-tunings at ≈ 9 min each, + L1, SNIP and ≈ 15 × 90 SHAP windows. About 75 min on nakhoul-3050 (≈ 90 min on uni-2080). It is shorter if the gate picks CE for a β (2 CE runs instead of 4 attribution runs).

**Step 9A estimate (2026-11-21, from pilot round 2's measured GPU timings).** Per fold: 16 CE/KD fine-tunings × 20 epochs (nakhoul-3050 ≈ 3.3 s, uni-2080 ≈ 2 s per epoch), 4 Park runs × 20 epochs (≈ 60 / 65 s per epoch), SHAP for 32 models × ≈ 765 windows (≈ 0.14 / 0.13 s per window; the original and noise-floor models are cache hits on the machine that ran the fold's noise floor), deletion faithfulness and accuracy (minutes):

| Part, per fold | nakhoul-3050 | uni-2080 |
|---|---|---|
| CE / KD fine-tuning | ≈ 18 min | ≈ 11 min |
| Park, park_kd | ≈ 80 min | ≈ 87 min |
| SHAP | ≈ 57 min | ≈ 54 min |
| Setup, faithfulness, accuracy | ≈ 5 min | ≈ 5 min |
| **Total per fold** | **≈ 2.7 h** | **≈ 2.6 h** |
| **3 folds (0, 1, 3 / 2, 3, 4)** | **≈ 8 h** | **≈ 8 h** |

Upper bound ≈ 2× (round 1 saw 2–4× slower epochs on a laptop in a poor power state).

**Machine split.** Nakhoul's machine is ≈ 1.9–2× faster on this workload, so it gets about two thirds of the work.
- **Step 4:** Nakhoul's runs DeepConvLSTM and TinyHAR (≈ 7.5 h); Aya's runs CNN1D, HART and XGBoost (≈ 5.5 h). After training, all candidates' inference and SHAP times are measured on **Aya's machine only**, so the model choice compares like with like.
- **Steps 8 + 9 + 9b:** **folds 0, 1, 2 on Nakhoul's machine, folds 3, 4 on Aya's, and Nakhoul's also reruns fold 3** for the cross-machine check (Step 9b). Without the check, Nakhoul's machine would finish hours earlier, so the extra fold costs no wall-clock time:

  | Frozen model | Nakhoul's (folds 0, 1, 2 + 3 again) | Aya's (folds 3, 4) |
  |---|---|---|
  | CNN1D | ≈ 5.1 h | ≈ 4.9 h |
  | HART | ≈ 21 h | ≈ 21 h |

  - Each fold is self-contained (its own noise floor, τ_k, tuning and grid), so folds never wait on each other. Only Step 11 needs all 5 folds; Step 9b needs both copies of fold 3.
- **Merging.** Each machine commits its per-fold CSVs and `.npz` files. Models and raw SHAP stay in the git-ignored cache; copy them only if Step 11 needs to recompute something. The per-fold model file records the machine.
- **Run one job per machine at a time** (parallel jobs distort the timing and the thread counts), and keep `torch.set_num_threads` at the machine's physical core count: 8 on Nakhoul's, 10 on Aya's.

---

## 6. Decisions

There are no open questions. Every point below is decided and already applied in the sections above.

**First review, 2026-11-19:**
- **Gate:** LCI on the 3 inner-validation HARTH subjects per fold (Section 2.5).
- **τ:** minimum retrain-seed LCI on those subjects; noise ranges are still reported for both groups (Section 2.7).
- **λ = 0.5,** so FA-LCI = the worst-group LCI (Sections 2.6–2.7).
- **Statistics:** paired Wilcoxon over 40 subjects pooled over folds, Holm-corrected, with effect sizes; bootstrap CIs over subjects for FA-LCI and the group gaps (Section 2.8).
- **Stairs F1:** pooled across subjects, never per subject (Section 2.7).
- **HART:** chosen only if it beats the next-best model by more than 2 macro-F1 points (Section 2.9).

**Second review, 2026-11-19 (questions Q1–Q17):**

| Q | Topic | Decision | Where |
|---|---|---|---|
| Q1 | PTQ quantizer | **Primary PTQ = standard per-tensor symmetric weight quantization** at 8 and 4 bits. The authors' fixed grid (`lci.compression.ptq`) stays as a secondary variant labelled **"Tulla-style"**, for comparison with their paper; it is excluded from the statistical tests. | 2.5, 2.8, 2.10, Step 9 |
| Q2 | SNIP | SNIP saliency applied to the trained model, then the same `E_ft` fine-tuning as every other method. Named **"SNIP-score pruning"** everywhere; the report states the difference from original SNIP. | 2.5 |
| Q3 | Pruning scope | Uniform per-layer sparsity over conv, linear, LSTM and attention weights; **classification head excluded**. Report the nominal level, the **actual total sparsity including the head**, and per-layer sparsity. | 2.5, 2.7, 2.10, Steps 9 and 11 |
| Q4 | SHAP target class | The original model's predicted class, fixed for every compressed model. | 2.6 |
| Q5 | k for 48 features | k = 10 main; k = 16 as a sensitivity check from the cached SHAP. | 2.6, Step 11 |
| Q6 | Model-selection tuning | Fold 0 only, scored on **fold 0's 3 inner-validation subjects only, never its test subjects**. An acceptance check verifies the subject sets. | 2.9, 2.10, Step 4 |
| Q7 | F1 margin | 2 pt; no 1 pt cross-check. | 2.9 |
| Q8 | DeepConvLSTM | *Superseded on 2026-11-19 (A.6.1)* by the cost-based rule: no model-specific conditions. | 2.9 |
| Q9 | Architecture details | Layer sizes and defaults from the papers. Done 2026-11-19: all re-implemented from the PDFs in `papers/`, no code ported, so licences do not apply. Every value is cited in Appendix A. | 2.3, App. A |
| Q10 | Preprocessing | Per-channel z-score and inverse-frequency class weights. **Normalization statistics come from the fold's HARTH training subjects only** and are applied unchanged to validation, test and HAR70+. | 2.1, 2.4, 2.10 |
| Q11 | F1 definitions | Per-subject macro-F1 over the 5 non-stair classes in both groups; stairs F1 pooled per group; model choice ranks by 7-class macro-F1 pooled over the 22 held-out HARTH subjects. | 2.7, 2.9 |
| Q12 | Walking-aid IDs | Checked 2026-11-19: Ustad et al. (2023) give no subject IDs. They state 5 walking-aid users (4 walker, 1 walking sticks outdoors) [Sec 2.1, p2] and 13 participants with at least one stair bout [Sec 3.1, p7]. That is consistent with our inference, but not proof, so "inferred" stays everywhere. | 1, ground rules |
| Q13 | Mixed-label windows | Strict majority (> 50%), plus the single-label-window robustness check. | 2.1, Step 11 |
| Q14 | KD | Not in the main grid; optional, only if time is left, reported descriptively. | 2.5, Step 10 |
| Q15 | Cross-machine | Record the machine per fold, **plus Step 9b: fold 3 is run on both machines and the results are compared**. If it fails, the folds are rerun on one machine. | Steps 7, 9b; 5 |
| Q16 | Secondary tests | Per-group Wilcoxon tests and Mann-Whitney U next to each gap CI, labelled secondary. | 2.8 |
| Q17 | Gate timing | Checked once per (fold, level), after pruning and before fine-tuning. | 2.5 |

Action items that are not decisions:
- ~~confirm the walking-aid IDs in Ustad et al. (2023) (Q12)~~: done, the paper lists none;
- ~~check the HART and TinyHAR code licences (Q9)~~: not needed, nothing was ported;
- decide the open points raised by the source check (Appendix A.6);
- ~~pin the xgboost version (Step 0)~~: done, xgboost 3.2.0.

---

## Appendix A. Source check (2026-11-19)

Every architecture and method value in `har/models.py`, `har/features.py`, `har/train.py` and this plan was checked against the PDFs in `papers/`. That folder is git-ignored because it holds copyrighted PDFs; copy it from the shared folder.

**Status labels:**
- **paper**: as stated in the source.
- **fixed**: was wrong in the code or plan, corrected on 2026-11-19.
- **our choice**: the source does not state it.
- **plan**: a decision that deliberately differs from the source, made so all candidates share the same windows, classes, normalisation and training budget.

### A.0 Sources

| Key | Paper | File in `papers/` |
|---|---|---|
| HARTH | Logacjov, Bach, Kongsvold, Bårdstu, Mork (2021), HARTH: A Human Activity Recognition Dataset for Machine Learning, *Sensors* 21:7853 | `sensors-21-07853 (1).pdf` |
| Ustad | Ustad et al. (2023), Validation of an Activity Type Recognition Model Classifying Daily Physical Behavior in Older Adults: The HAR70+ Model, *Sensors* 23:2368 | `sensors-23-02368-v2 (1).pdf` |
| DCL | Ordóñez & Roggen (2016), Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition, *Sensors* 16:115 | `sensors-16-00115.pdf` |
| TINY | Zhou et al. (2022), TinyHAR: A Lightweight Deep Learning Model Designed for HAR, ISWC '22 | `2022_ISWC____TinyHAR__…(2).pdf` |
| HART | Ek, Portet & Lalanda, Transformer-based Models to Deal with Heterogeneous Environments in HAR, arXiv 2209.11750v2 | `2209.11750v2.pdf` |
| SNIP | Lee, Ajanthan & Torr (2019), SNIP: Single-shot Network Pruning based on Connection Sensitivity, ICLR | `1810.02340v2.pdf` |
| Park | Park et al. (2020), Attribution Preservation in Network Compression for Reliable Network Interpretation, NeurIPS | `2010.15054v1.pdf` |
| Tulla | Tulla et al. (2026), Silent corruption: Logic collapse and attribution fidelity failure in compressed IDS, *Inf. Sci.* 757 | `5- Silent-Corruption-IDS.pdf` |
| RP | CSC689 research plan (ours) | `CSC689 Research Plan …pdf` |

### A.1 Data and windows

| Item | Ours | Source | Status |
|---|---|---|---|
| HARTH sensors, range | back + thigh, 3 axes, ±8 g | [HARTH Sec 3.1, p6] | paper |
| HARTH sampling | 50 Hz; S006 at 100 Hz decimated to 50 Hz | first session recorded at 100 Hz "which we later downsampled to 50 Hz" [HARTH Sec 3.1, p6] | paper (S006 is shipped at 100 Hz in v1.2) |
| HAR70+ sampling, sensors | 50 Hz, back + thigh AX3, ±8 g | [Ustad Sec 2.2.1, p3] | paper |
| 20 Hz low-pass | 4th-order 20 Hz Butterworth low-pass on **both datasets**, per gap-free segment after resampling, before windowing (`data.lowpass_20hz`) | HARTH: "20 Hz low-pass fourth-order Butterworth filter" [HARTH Sec 3.3, p10]. Ustad uses a 4th-order Butterworth band-pass [Ustad Sec 2.3.2, p5] (band not stated; not used) | paper (decided 2026-11-19, A.6.2); zero-phase (sosfiltfilt) = our choice |
| Window length | 128 samples = 2.56 s, 50% overlap train / none test | HART uses 128 (2.56 s) [HART Sec 2.2, p4]; HARTH uses 1 s non-overlapping [HARTH Sec 3.3]; Ustad 5 s non-overlapping [Ustad Sec 2.3.2] | plan (RP) |
| Window label | majority, kept if > 50% | majority voting [HARTH Sec 3.3, Fig. 5]; "dominant activity type" [Ustad Sec 2.3.2]; last sample [DCL Sec 4.2] | majority = paper (HARTH/Ustad); > 50% cut = our choice |
| Normalisation | per-channel z-score from training subjects | channel-wise z-normalisation [HART Sec 2.2, p4]; HARTH min-max-scales features only [HARTH Sec 3.3] | paper (HART) / plan |
| Shared classes | walking, shuffling, stairs up/down, standing, sitting, lying | HAR70+ labels [Ustad Sec 2.3.1, p4]; HARTH's 12 labels [HARTH Sec 3.3] | paper |
| Evaluation pooling | summed confusion matrices over held-out subjects → macro-F1 | "We sum up the resulting 22 matrices … to compute the recall, precision, and F1-score" [HARTH Sec 4.2, p12] | paper (our ranking metric follows it) |
| Walking-aid subgroup | 503, 505, 513, 514, 515 (inferred) | 5 users (4 walker, 1 sticks) [Ustad Sec 2.1, p2]; 13 participants with ≥ 1 stair bout [Ustad Sec 3.1, p7]; no subject IDs given | inferred (consistent, not confirmed) |

### A.2 Candidate models

**CNN1D: HARTH's CNN** [HARTH Sec 3.2.6 p9; Sec 4.1 p11–12]

| Value | Ours | Source | Status |
|---|---|---|---|
| Conv layers / kernels per layer | 4 / 128 | "number of layers (best: 4)", "number of kernels … (best: 128)" [Sec 4.1] | **fixed** (was 3 × 64) |
| Kernel sizes | 6, 12, 12, 32 | "kernel shape/size in each layer (best: [6, 12, 12, 32])" [Sec 4.1] | **fixed** (was 5) |
| Conv type | Conv1d over time, axes as input channels | "1D convolutional kernels … to enable a single kernel to learn to extract useful information of each axis" [Sec 4.1] | our choice (reading; per-axis shared kernels are the alternative, A.6.4) |
| Activation | ReLU | [Sec 3.2.6] (general description) | paper |
| Dropout | 0.4 after each layer | [Sec 4.1, p12] | **fixed** (was 0.5 once) |
| Dense layers | 2 × 512, ReLU, then output | "last two hidden layers … fully connected 512-dimensional layers with ReLU" [Sec 4.1, p12] | **fixed** (was none) |
| Padding | 'same' | not stated (implied: valid convs 6+12+12+32 cannot fit their 50-sample windows) | our choice |
| Normalisation layers | none | none stated | our choice (was BatchNorm) |
| Conv → dense | global average pooling over time | not stated (flattening 128 × 128 → 512 would add 8.4M parameters) | our choice |
| Training | plan recipe | SGD, lr 0.001, 80 epochs, no early stopping [Sec 4.1] | plan |
| Parameters | 1,254,919 | not reported | — |

**DeepConvLSTM** [DCL]

| Value | Ours | Source | Status |
|---|---|---|---|
| Layers | C(64)–C(64)–C(64)–C(64)–R(128)–R(128)–Sm | [Sec 3.1, p8] | paper |
| Convolution | along time only, channels separate, kernel 5, no padding, no pooling | [Sec 3.1]; P = 5 and "only computed where the input and the kernel fully overlap" [Sec 3, Eq. 10, p7–8]; no pooling [Sec 3] | paper; parameters per conv layer 384 / 20,544 as [Table 1] |
| Activations | ReLU (conv), tanh (LSTM, PyTorch default) | [Sec 3.1] | paper |
| Dropout | 0.5 on the inputs of every dense layer | [Sec 3.3, p9] | paper |
| Initialisation | orthogonal | "Weights are randomly orthogonally initialized" [Sec 3.3] | **fixed** (was PyTorch default) |
| Output | last time step | [Sec 4.2, p12] | paper |
| Training | plan recipe | RMSProp, lr 1e-3, ρ 0.9, batch 100 [Sec 3.3] | plan |
| Width 0.5 | half filters and LSTM cells | width multiplier α for DeepConvLSTM [TINY Sec 4.1, p3] | paper (as a method) |

**TinyHAR** [TINY Sec 3.2 p2–3, Sec 4 p3–4]

| Value | Ours | Source | Status |
|---|---|---|---|
| Individual conv subnet | 4 layers, 5 × 1 kernels along time, per channel, BN + ReLU, same F | [3.2.1] | paper |
| Stride | 2 in **every** layer (T* = 8) | "the stride in each layer is set to 2" [3.2.1] | **fixed** (was every second layer) |
| Filters F | 28 | F is set per dataset; the only stated value is 28 (Opportunity) [Sec 4.2, p4] | **fixed** (was 20); paper value for another dataset |
| Cross-channel interaction | 1 transformer encoder block across channels at each time step | [3.2.2] | paper |
| Attention heads | 1 | "a scaled dot-product self-attention layer" [3.2.2]; no head count | our choice (was 2) |
| FF size, dropout | 2F, 0.2 | not stated | our choice |
| Cross-channel fusion | FC on flattened C × F → F* = 2F | [3.2.3] | paper; ReLU after it = our choice |
| Temporal | 1 LSTM, hidden F* | [3.2.4] | paper |
| Temporal attention | weighted sum of hidden states added to the last one, trainable γ | [3.2.5] | paper; linear scores + softmax, γ init 0 = our choice |
| Training | plan recipe (batch 256 matches) | Adam lr 1e-4, decay to 10% on 5-epoch plateau, ≤ 150 epochs, batch 256 [Sec 4.1, p4] | plan |
| Parameters | 54,273 | not comparable (depends on channels) | — |

**HART** [HART]

| Value | Ours | Source | Status |
|---|---|---|---|
| Input | W = 128, two 3-axis sensors | [Sec 3.2, p7] | paper |
| Embedding | one conv per sensor, d/S = 96 filters, non-overlapping frames, concatenated, + learnable position embedding, no class token | [Sec 3.2, p7] | paper |
| Frame length / stride | 16 / 16 (8 tokens) | not stated | our choice |
| d | 192 | [Sec 3.2, p7; Sec 4.2, p10] (one sentence says 196 [Sec 4.2, p11]) | paper |
| Blocks / heads | 6 / 3 per block | [Sec 4.2, p10] | paper; 3 heads in each sensor-wise MSA = our reading |
| Block | LN; MSA on half of each sensor's embedding (d/4 each), LightConv on the remaining d/2; concat in original positions + residual; LN; FF + residual | [Sec 3.2, p8] | paper |
| LightConv internals, kernel sizes | Wu et al. (2019) LightConv (GLU, softmax-normalised shared depthwise kernels); kernels 3, 5, 7, 7, 7, 7 | cited to Wu et al., not described [Sec 3.2, p8] | our choice |
| Activation, dropout | Swish, 0.3 | [Sec 4.1, p10] | paper |
| DropPath | before every residual, rate 0.1 (linear in depth) | DropPath stated, rate not [Sec 4.1] | paper / rate = our choice |
| FF hidden, head | 2d; GAP → FF 1,024 (Swish) → classifier | GAP → "a feed-forward layer" → head [Sec 3.2, p8]; sizes not stated | our choice |
| Parameters | 1,389,555 | 1,445,918 [Table 7, p13] | −3.9% |
| Training | plan recipe (class weights match) | Adam 5e-4, batch 32, 200 epochs, label smoothing 0.1, class-weighted [Sec 4.1] | plan |

### A.3 XGBoost reference and features [HARTH Sec 3.3, 4.1; Ustad Sec 2.3.2–2.3.3]

| Value | Ours | Source | Status |
|---|---|---|---|
| Features | 155: 8 signals (6 axes + 2 magnitudes) × (8 gravity + 9 movement) + 15 axis correlations + 1 magnitude correlation + 3 two-sensor gravity means | the 161-feature set [HARTH Sec 3.3, p10–11]; Ustad uses "the same 161 features" [Ustad Sec 2.3.2] | **fixed** (was 94 generic features); 6 of 161 not identifiable = our choice to omit |
| Pre-filter, gravity split | 4th-order 20 Hz low-pass; 4th-order 1 Hz low-pass = gravity, rest = movement | [HARTH Sec 3.3] | paper; filtering per window = our choice |
| Trees, depth, learning rate | 1,024; grid depth {3, 6} × η {0.1, 0.5} | 1,024 / 3 / 0.1 [HARTH Sec 4.1, p12]; 1,024 / 3 / 0.5 [Ustad Sec 2.3.3, p6] | **fixed** (grid now contains both paper settings); depth 6 = our choice |
| λ, α, subsampling | 1, 0, none | [HARTH Sec 4.1] | **fixed** (α added) |
| Loss / stopping | multi-class error rate; early stopping 30 rounds | error rate [HARTH Sec 4.1]; early stopping not used there | paper / our choice |
| Class weights | inverse frequency | not stated | plan |

### A.4 Compression methods

| Method / value | Ours | Source | Status |
|---|---|---|---|
| PTQ bit widths | 8, 4 | "8-bit and 4-bit integer quantization" [Tulla Sec 4, p6] | paper |
| PTQ quantizer | per-tensor symmetric (primary); fixed grid "Tulla-style" (secondary) | Tulla gives no quantizer details; the fixed grid is from the authors' code (see LCI_IMPLEMENTATION.md) | plan (Q1) |
| L1 unstructured levels | 30, 50, 70, 90% | 30, 50, 70% [Tulla Sec 4, p6] | paper + 90% = plan |
| SNIP saliency | \|w · ∂L/∂w\| | ∂L/∂c_j at c = 1 [SNIP Eq. 5, p4]; normalised \|g_j\| [Eq. 6] | paper (normalisation does not change a per-layer ranking) |
| SNIP when / scope | trained model; per-layer sparsity; 10 mini-batches | at initialisation, single shot, global top-κ [SNIP Alg. 1, Eq. 7, Sec 4.2]; "can accumulate the saliency measure over multiple batches" [Sec 4.2, p5] | plan (Q2, Q3); batches = paper option |
| Park loss | CE + γ_P · mean ‖â_S − â_T‖₂, unit-norm maps, unsquared | Eq. 2 [Park Sec 3.2, p5] | **fixed** (plan had the square) |
| Park map and target | 48-segment input EG attribution of the original model's top-1 class | Grad-CAM SWA at the last conv layer, top-1 of the full network [Park Sec 4, p6] | our choice (level) / paper (target) |
| Park strength | γ_P ∈ {1, 50} | β = 50 for SWA [Park App. C.2, p15] | **fixed** (was {0.1, 1}); 1 = our choice |
| Park pruning schedule | one-shot L1, then E_ft = 10 epochs | iterative magnitude pruning, ρ_w = 0.2 × 16 iterations, 30 epochs each, SGD lr 1e-3 [Park Sec 4.2, App. C.2] | plan (same budget for all methods) |
| KD | optional, T = 3, α = 0.7 | T = 3, 15 epochs [Tulla Sec 4, p6]; α not stated (authors' code) | paper / our choice |
| Explanation-guided pruning, gate | ours | not in any source (RP) | ours |

### A.5 Explanations, metrics, statistics

| Item | Ours | Source | Status |
|---|---|---|---|
| Explainer | shap GradientExplainer | [Tulla Sec 4, p6] | paper |
| Explained samples / background | ≤ 30 windows per subject; 100 background windows | 300 samples, 50 background passes [Tulla Sec 4, p6] | plan (RP) |
| LCI | ω · mean per-sample Pearson + (1 − ω) · Jaccard of top-k | Def. 3.1 [Tulla p4] | paper |
| k, ω | 10, 0.5 | [Tulla Sec 4, p6] | paper |
| τ | data-driven (retrain noise) and 0.85 | 0.85, from the bimodal LCI distribution [Tulla Def. 3.2, p4] | plan + paper |
| Safe explanation rate | share of settings with LCI ≥ τ | Fig. 1(b) [Tulla p4] | paper (as a concept) |
| 48 segments, FA-LCI (λ = 0.5), deletion faithfulness, statistics | as in Sections 2.6–2.8 | RP. The RP PDF still writes λ = 1 [RP p4]; λ = 0.5 was decided on 2026-11-19 | ours (update the RP) |

### A.6 Points the source check raised (all decided 2026-11-19)

1. **Selection rule → cost-based** (Section 2.9).
   - The old cheap/costly grouping came from stand-ins, where CNN1D had 44K parameters. With the papers' sizes (Aya's laptop, CPU, 10 threads, torch 2.7.1; torch 2.15.0 values in `environment/time.md`), CNN1D is the second most expensive candidate:

     | Model | Parameters | Training, ms per window | SHAP (1 output, nsamples 200), s per window |
     |---|---|---|---|
     | DeepConvLSTM | 458K | 4.00 | 0.70 |
     | CNN1D (HARTH) | 1.25M | 3.44 | 0.45 |
     | HART | 1.39M | 1.16 | 0.22 |
     | TinyHAR | 54K | 0.34 | 0.08 |

   - **Decision:** take the best pooled 7-class macro-F1; among all candidates within 2 pt of it, choose the lowest CPU SHAP time per window; parameter count breaks ties. Implemented in `har/select_model.py` and tested in `har/tests/test_model_selection.py`.
   - Section 5's CPU estimates are from stand-ins with torch 2.15. The real models differ (HART SHAP 0.22 s vs the stand-in's 0.15 s; DeepConvLSTM 0.70 s vs 0.49 s), so Step 6 re-measures before the grid.
2. **Pre-filtering → HARTH's 20 Hz low-pass on both datasets** (Section 2.1, `data.lowpass_20hz`).
   - 4th-order Butterworth [HARTH Sec 3.3], zero-phase, per gap-free segment after resampling, before windowing. Our choice: segments ≤ 15 rows stay unfiltered; they are far shorter than a window.
   - The cache was rebuilt on 2026-11-19. Window counts, window IDs, explain windows, backgrounds, folds and validation subjects are **unchanged**: 9 of the 10 `results/har-setup/` files are byte-identical, because filtering changes values but never rows, labels or timestamps.
   - Only `har-normalisation-stats-per-fold.csv` changed: the means are identical to 6 decimals, and the standard deviations are 0.1–1.1% lower (power above 20 Hz removed).
   - The XGBoost features no longer low-pass again per window.
3. **Training recipes → one shared recipe for all candidates, kept as a deliberate deviation** (Section 2.4 and A.7).
4. **CNN1D convolution type → axes as input channels**, marked "our choice" (`har/models.py`, A.2).
5. **The research plan PDF** still says λ = 1 for FA-LCI; the decided value is 0.5 (update the PDF; nothing to change in the code).

### A.7 Report notes: deviations to state in the final report

These go into `results/har-analysis/har-report.md` (Step 12), under "Deviations from the source papers".

1. **One shared training recipe** (deliberate). Every candidate is trained with the same recipe, so the comparison measures architectures, not training effort:
   - Adam, lr ∈ {1e-3, 3e-4}, weight decay 1e-4, batch 256, cosine schedule, ≤ 30 epochs with early stopping (patience 5) on validation pooled macro-F1;
   - class-weighted cross-entropy without label smoothing;
   - the same 4-setting tuning budget.

   The papers' own recipes are not used:

   | Model | Paper's recipe |
   |---|---|
   | HARTH CNN | SGD, lr 0.001, 80 epochs, no early stopping [HARTH Sec 4.1] |
   | DeepConvLSTM | RMSProp, lr 1e-3, ρ 0.9, batch 100 [DCL Sec 3.3] |
   | TinyHAR | Adam, lr 1e-4, decay to 10% on plateau, ≤ 150 epochs, batch 256 [TINY Sec 4.1] |
   | HART | Adam, lr 5e-4, batch 32, 200 epochs, label smoothing 0.1, class-weighted [HART Sec 4.1] |

   Absolute accuracies can therefore differ from the papers'.
2. **Windows:** 2.56 s (as HART), not HARTH's 1 s or Ustad's 5 s; majority label with a > 50% cut.
3. **CNN1D:** axes as input channels, 'same' padding and global average pooling are our choices (A.2).
4. **HART, TinyHAR:** the sizes the papers do not state are our choices (A.2).
5. **XGBoost:** 155 of HARTH's 161 features (6 not identifiable from the text); gravity filter per window.
6. **Compression:** SNIP's score used on the trained model with per-layer sparsity; Park's matching on input attributions with one-shot pruning; per-tensor symmetric PTQ as primary (A.4).
7. **Filtering:** HARTH's 20 Hz low-pass applied to HAR70+ too. Ustad's band-pass is not used.
