# Task: Explanation consistency of compressed HAR models across age groups (HARTH → HAR70+)

This document is the complete specification for the HAR part of the study. It covers the data and windowing, the subject-wise folds, the choice of the model (made once, then frozen), the compression grid, SHAP, the window version of LCI, FA-LCI, deletion faithfulness, the noise floor, the statistics, the outputs and the run plan for the two machines in [environment/specs.md](../environment/specs.md). It builds on the LCI reproduction ([LCI_IMPLEMENTATION.md](LCI_IMPLEMENTATION.md)); reuse `lci/` wherever this document says so.

**Research question.** When a HAR model trained on adults (HARTH) is compressed, do its SHAP explanations stay consistent with those of the original model, and does that consistency drop more for older adults (HAR70+) than for the general population the model was trained on? Can an explanation-guided pruning rule, built from HARTH data only, keep explanations consistent for both groups at the same size?

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
| `cnn1d` | Logacjov et al. (2021) | 1D convolutions over time, all 6 channels as input channels. Layer sizes as published in the paper (Q9). |
| `deepconvlstm` | Ordóñez & Roggen (2016) | 4 conv layers (64 filters, kernel 5) + 2-layer LSTM (128) + linear head. |
| `tinyhar` | Zhou et al. (2022) | Port the authors' PyTorch reference implementation if its licence allows; otherwise re-implement from the paper (Q9). |
| `hart` | Ek et al. (2022/2023) | Re-implemented in PyTorch with the defaults (patch size, depth, width) from the paper and the authors' TensorFlow code; check the code's licence before porting, otherwise re-implement from the paper (Q9). Its two sensor branches (accelerometer, gyroscope in the original) become **back** (ch. 0–2) and **thigh** (ch. 3–5). |
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
| **Explanation-guided (ours)** | same | keep-score (below) | `E_ft` epochs, CE, or CE + γ · L_cons if the gate fails |

**PTQ, primary (decided, Q1).** `ptq_symmetric` quantizes the same weight tensors that pruning touches, **plus the head**. Per tensor:
- `s = max|W| / (2^(b−1) − 1)`;
- `W_q = clamp(round(W / s), −(2^(b−1) − 1), 2^(b−1) − 1) · s`, so 255 levels at 8 bits and 15 at 4 bits, with zero exactly representable;
- no calibration and no fine-tuning.

Biases and normalization parameters stay float32, and activations are not quantized (weight-only, the usual setting for PTQ of small models). The result is stored back as float (fake quantization), so gradients exist for SHAP. Both PTQ variants appear in every table, labelled "PTQ" and "PTQ (Tulla-style)"; the Tulla-style rows are excluded from the statistical comparisons (Section 2.8).

**SNIP-score pruning (decided, Q2).** This name is used everywhere. Original SNIP prunes at initialization and trains from scratch; here only its saliency score is used, on the trained model, followed by the same fine-tuning as every other method. The report states this difference.

**KD (decided, Q14): not part of the main grid.** Its size is not matched to the pruning levels, so it cannot enter the paired comparisons. It runs only if time is left after Step 9 (Step 10): one student of the frozen model's family, width scaled to ≈ 70% of the parameters, 30 epochs, `α·KL(T)·T² + (1−α)·CE` with T = 3 and α = 0.7 (as in `lci.compression.distill`). It is reported descriptively.

`E_ft = 10` epochs for every pruning method, with the same optimizer (Adam, lr 1e-4), batch size, class weights and seed. L1 without fine-tuning is not part of the grid.

**Attribution surrogate used inside losses.** SHAP itself is too slow inside training. Losses use a differentiable expected-gradients surrogate (the quantity GradientExplainer estimates):

```python
def eg_attributions(model, X: torch.Tensor, target: torch.Tensor, baselines: torch.Tensor,
                    m: int, create_graph: bool) -> torch.Tensor          # (B, 6, 128)
    # mean over m (baseline b, alpha ~ U(0,1)) draws of (x - b) * d f_target(b + alpha (x - b)) / dx,
    # baselines drawn from the fold's 100-window background with a seeded generator.
```

Both attribution losses compare 48-segment attributions (Section 2.6), with target = the original model's predicted class (Section 2.6).

**(a) Park et al. (2020), adapted.** They match normalized attribution maps between the compressed and the original network during fine-tuning. Here: `L_AM = mean_i ‖ â_S(x_i) − â_T(x_i) ‖²₂`, where `â` is the 48-segment EG attribution (m = 4) scaled to unit L2 norm per window, and the original model's attributions are precomputed and fixed. The original work matches maps at intermediate layers of image CNNs; here the matching is on the input attribution of a time series, the only level that is comparable across architectures. Record this deviation in the report.

**(b) Explanation-guided pruning (ours). Uses HARTH data only.**
- Keep-score per weight: `score(w) = (1 − β) · |w| / mean_layer|w| + β · I(w) / mean_layer I(w)`.
- `I(w) = | w · ∂/∂w Σ_i ⟨ A_θ(x_i), sg(A_θ0(x_i)) ⟩ |`. This is a first-order (Taylor) estimate of how much removing `w` changes the alignment of the model's explanation with the original explanation. `A` is the 48-segment EG attribution (m = 8, `create_graph=True`), on 1,000 fixed HARTH training windows stratified by class. `sg` = stop-gradient.
- Fine-tune penalty: `L_cons = 1 − mean_i PearsonR( A_θ(x_i), sg(A_θ0(x_i)) )` (m = 4). It mirrors the ρ̄ term of LCI; the Jaccard term is not differentiable.
- **Gate.** After pruning and before fine-tuning, compute the gate metric with real SHAP (GradientExplainer, nsamples 200) on the inner-validation explain windows. If gate < τ, fine-tune with `CE + γ · L_cons`; otherwise fine-tune with CE only. Both branches use `E_ft` epochs. Record the gate value and the branch taken for every (fold, level).
- **Gate metric = LCI_val** (decided 2026-11-19): per-subject LCI(original, pruned) averaged over the fold's 3 inner-validation HARTH subjects, compared with `τ_k` from the same subjects (Section 2.7). FA-LCI is not the gate because it needs HAR70+, which would put test data into a training decision. The gate is evaluated once per (fold, level), after pruning and before fine-tuning.

**Tuning (HARTH inner validation only, equal budget).** Park's γ_P ∈ {0.1, 1.0}, and ours (β, γ) ∈ {0.3, 0.7} × {0.1, 1.0}. Ours gets 4 configurations, Park 2; to equalise the budget, Park also gets 2 learning-rate variants {1e-4, 3e-4}. Tuning runs at 70% sparsity per fold. The chosen configuration maximizes inner-val LCI_val among those whose inner-val macro-F1 is within 1 pt of the L1 baseline's, and is then used at every sparsity in that fold. Results are logged to the grid's per-fold model file.

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
- **Equal tuning budget (decided, Q6):** 4 configurations per candidate (lr ∈ {1e-3, 3e-4} × width ∈ {paper default, half}; for XGBoost: depth ∈ {4, 6} × learning rate ∈ {0.1, 0.3}). The best one is then trained on all 5 folds (seed 0), so there are 4 + 5 = 9 trainings per candidate.
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
- **Rule** (F1 = the ranking metric above; HART condition decided 2026-11-19, DeepConvLSTM condition and the 2 pt margin decided as Q7 and Q8):
  - **Costly candidates** are HART and DeepConvLSTM: on the stand-ins, about 11 h and 30 h per fold on Aya's machine (Section 5). **Cheap candidates** are CNN1D and TinyHAR.
  1. **HART** is chosen only if F_HART > F_next + 2 pt, where F_next is the best F1 among CNN1D, DeepConvLSTM and TinyHAR.
  2. Otherwise, **DeepConvLSTM** is chosen only if F_DCL > F_cheap + 2 pt, where F_cheap is the best F1 among CNN1D and TinyHAR.
  3. Otherwise, choose among the cheap candidates: eligible = F1 ≥ F_cheap − 2 pt. Choose the eligible candidate with the fewest parameters. If two are within 10% in parameters, choose the one with the lower SHAP time per window, then the lower inference time.
  - **Edge case:** HART and DeepConvLSTM both beat the cheap candidates by more than 2 pt but are within 2 pt of each other. Step 1 then fails for HART and step 2 picks DeepConvLSTM, the costlier of the two. In that case only, choose HART instead (similar F1, lower cost), and note it in the report.
  - There is no 1 pt cross-check (Q7).
- XGBoost is reported as the accuracy reference only.
- **Freezing:** write the architecture, configuration and training recipe to `results/har-model-selection/har-frozen-model-config.json`. Nothing about the model changes after this, whatever later steps show.
- **Sanity check after freezing** (uses HAR70+, so it comes after the freeze and cannot change it): frozen model, 5 folds, accuracy on non-stair classes for walking-aid (inferred) vs other older subjects. Ustad et al. (2023) report 87% vs 92%. That was a model trained on HAR70+ itself, so compare the **direction and rough size** of the gap, not the absolute values.

### 2.10 Tests (must pass before the step that uses the code)

`lci/tests/test_lci.py` (12 existing tests) must keep passing. New test files:
- **`lci/tests/test_lci_windows.py`:**
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
  - with β = 0 the keep-score ranks like L1;
  - the gate picks the penalty branch exactly when gate < τ.
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

4. **Model selection** (`python -m har.run_model_selection --candidates ... --folds 0 1 2 3 4`, then `python -m har.select_model`).
   *Why:* one fixed architecture for the whole grid, chosen on HARTH alone, cheap enough to explain on CPU.
   *Acceptance:*
   - all candidates trained with the same budget;
   - **tuning check:** `har-model-selection-tuning.csv` lists, per configuration, the tuning-train and tuning-validation subject IDs. The tuning-validation IDs are exactly fold 0's 3 inner-validation subjects. Neither set contains any of fold 0's test subjects or any HAR70+ subject. The run aborts otherwise;
   - the summary table is complete;
   - the rule in Section 2.9 is applied (HART only if > 2 pt over the next-best, DeepConvLSTM only if > 2 pt over the best cheap candidate, otherwise the smallest cheap candidate within 2 pt), and each branch taken is logged;
   - the frozen config is written;
   - the chosen model's held-out HARTH macro-F1 is within 2 pt of XGBoost's, or the gap is reported.

5. **Sanity check: walking-aid (inferred) vs other older** (`python -m har.run_sanity_walking_aid`).
   *Why:* if the frozen model does not show the known gap direction, the subgroup or the data handling is suspect.
   *Acceptance:* the gap (other older − walking-aid) is positive. Report its size and bootstrap CI next to Ustad et al.'s 5 pt. If it is not positive, re-check the subject IDs and the class filter. The model is not changed.

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
   *Why:* the core experiment: every method at matched size, same folds, same explain windows.
   *Acceptance, per fold:*
   - identical `n_nonzero` across L1, SNIP-score pruning, Park and ours at each level;
   - the head is unpruned, and the nominal and actual total sparsity are both recorded;
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
| `results/har-noise-floor/` | `har-noise-floor-per-subject-lci.csv`, `har-noise-floor-tau-per-fold.csv` |
| `results/har-compression-grid/` | per fold k: `har-grid-fold{k}-per-subject-metrics.csv` (fold, method, level, subject, group, accuracy, macro-F1, LCI48, LCI768, shap_corr, jaccard, n_skipped, mprf, AOPC, drop_k5, random AOPC), `har-grid-fold{k}-models.csv` (n_params, n_nonzero, nominal sparsity, actual total sparsity incl. head, per-layer sparsity, sizes, inference time, tuning choice, gate value and branch, machine), `har-grid-fold{k}-shap-48-segments.npz` + `-configs.json`; `har-grid-fold3-cross-machine-comparison.csv` (Step 9b) |
| `results/har-analysis/` | `har-per-subject-pooled-over-folds.csv` (one row per method, level, subject), `har-summary-per-method-level-group.csv`, `har-fa-lci-pooled-bootstrap-ci.csv` (FA-LCI, its parts, worst group, CI and paired differences vs ours), `har-fa-lci-per-fold-method-level.csv`, `har-method-comparisons-wilcoxon-holm.csv` (40-subject primary + per-group secondary), `har-group-gaps-older-vs-general-bootstrap-ci.csv`, `har-stairs-f1-pooled-per-group.csv`, `har-subgroup-walking-aid-inferred.csv`, `har-safe-explanation-rate.csv`, `har-fa-lci-validation.csv`, `har-raw-768-vs-48-segment-lci.csv`, `har-lci-k16-sensitivity.csv`, `har-single-label-windows-robustness.csv`, `har-report.md` |

Git-ignored, rebuildable: `data/processed/har/` (window cache, raw 768-value SHAP cache, model state dicts).

`environment/requirements.md` gets xgboost and every command run, with the machine it ran on; `environment/time.md` gets Step 6's benchmark and the wall-clock time of Steps 4, 8, 9, 9b and 11 per machine.

---

## 5. Estimated CPU run time and the machine split

**How the estimates were made.**
- **Per-window costs** were measured on Aya's machine on 2026-11-19 with rough stand-ins of the four candidates (not the final architectures) on random `(6, 128)` windows, 10 threads, shap 0.51.0, torch 2.15.0+cpu.
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
| Q8 | DeepConvLSTM | Chosen only if it beats the best of CNN1D and TinyHAR by more than 2 pt (same principle as HART). | 2.9 |
| Q9 | Architecture details | Layer sizes and defaults from the papers (Logacjov et al. 2021; Ek et al.) and authors' code; check licences before porting HART or TinyHAR code, otherwise re-implement from the paper. | 2.3 |
| Q10 | Preprocessing | Per-channel z-score and inverse-frequency class weights. **Normalization statistics come from the fold's HARTH training subjects only** and are applied unchanged to validation, test and HAR70+. | 2.1, 2.4, 2.10 |
| Q11 | F1 definitions | Per-subject macro-F1 over the 5 non-stair classes in both groups; stairs F1 pooled per group; model choice ranks by 7-class macro-F1 pooled over the 22 held-out HARTH subjects. | 2.7, 2.9 |
| Q12 | Walking-aid IDs | Check Ustad et al. (2023) and its supplement for subject IDs; until confirmed, "inferred" everywhere. | 1, ground rules |
| Q13 | Mixed-label windows | Strict majority (> 50%), plus the single-label-window robustness check. | 2.1, Step 11 |
| Q14 | KD | Not in the main grid; optional, only if time is left, reported descriptively. | 2.5, Step 10 |
| Q15 | Cross-machine | Record the machine per fold, **plus Step 9b: fold 3 is run on both machines and the results are compared**. If it fails, the folds are rerun on one machine. | Steps 7, 9b; 5 |
| Q16 | Secondary tests | Per-group Wilcoxon tests and Mann-Whitney U next to each gap CI, labelled secondary. | 2.8 |
| Q17 | Gate timing | Checked once per (fold, level), after pruning and before fine-tuning. | 2.5 |

Action items that are not decisions:
- confirm the walking-aid IDs in Ustad et al. (2023) (Q12);
- check the HART and TinyHAR code licences (Q9);
- pin the xgboost version (Step 0).
