# Task: Implement and validate the Logic Consistency Index (LCI)

This document is the complete specification. It covers the dataset, preprocessing, model, training, the three compression methods, SHAP settings and the LCI metric. You do not need the source paper or the authors' code. Where the paper was silent, the choice below comes from the authors' public code and says so.

Source paper: Tulla et al., "Silent corruption: Logic collapse and attribution fidelity failure in compressed intrusion detection systems", *Information Sciences* 757 (2026) 123908, https://doi.org/10.1016/j.ins.2026.123908. Authors' code: https://github.com/hamidborkot/logic-collapse-horizon. **Do not copy from that repo.** It contains at least three mutually inconsistent LCI formulas. Only the definition in Section 2.6 below matches the paper.

**Ground rules**
- Never fall back to synthetic data. The authors' loader, printing only a one-line notice, substitutes `make_classification` data if the download fails. Do not do that. If the download fails, raise an error and stop.
- Every random choice is controlled by an explicit seed: numpy, torch, Python `random`, the data split and SHAP sampling.
- Always report LCI's two components (`shap_corr`, `jaccard`) next to LCI itself.

---

## 1. Data fetching (add to our existing data-fetching scripts)

First read our existing data-fetching scripts and their README section, and follow the same structure, naming, output directory and documentation style. Add one new dataset entry. Fetch it exactly the way the authors' code does, nothing more and nothing less, and make sure you add it to our script and read me:

**Dataset: Phishing Websites, UCI Machine Learning Repository, ID 327**
- Page: https://archive.ics.uci.edu/dataset/327/phishing+websites
- Direct zip (reference only; the fetch method is `ucimlrepo` below): https://archive.ics.uci.edu/static/public/327/phishing+websites.zip
- DOI: https://doi.org/10.24432/C51W2X — license CC BY 4.0
- Citation: Mohammad, R. & McCluskey, L. (2012). Phishing Websites [Dataset]. UCI Machine Learning Repository.
- 11,055 instances, 30 integer features (values in {-1, 0, 1}), one binary target `Result` (values {-1, 1}; -1 = phishing, 1 = legitimate).

⚠️ Use **ID 327**. One file in the authors' repo uses UCI ID 967, which is a different, much larger phishing-URL dataset. That is wrong for this task.

**Fetch method (identical to the authors' code):**
```python
# pip install ucimlrepo
from ucimlrepo import fetch_ucirepo
ds = fetch_ucirepo(id=327)
features = ds.data.features      # pandas DataFrame, 11055 x 30
targets  = ds.data.targets       # pandas DataFrame, 11055 x 1
```
Save `features` and `targets` unmodified (raw, no scaling or encoding) in the same format and location our other datasets use. On load, assert `features.shape == (11055, 30)`, `targets.shape[0] == 11055`, and that the target has exactly 2 unique values. Print the class counts: the paper reports a 44:56 class ratio (phishing : legitimate). Add the dataset to the README the same way the existing ones are documented: source link, DOI, license, size and how to run the fetch.

---

## 2. Implementation specification

Suggested modules (adapt names and locations to our repo's conventions):
`models.py` · `train.py` · `compression.py` · `explain.py` (SHAP) · `lci.py` (metric) · `run_lci_phishing.py` (experiment) · `tests/test_lci.py`.
In this repo all of them live in [`lci/`](../lci/) (the metric is `lci/metric.py`, the tests `lci/tests/test_lci.py`); output paths are defined in `lci/paths.py`.

### 2.1 Preprocessing and split (exactly the authors' pipeline)
Applied to the raw features and targets from Section 1:
1. Label-encode any object/category column (`sklearn.preprocessing.LabelEncoder` on `.astype(str)`). Phishing has none, but keep this step.
2. Keep numeric columns only.
3. Drop constant columns (`df.std() > 0`).
4. Replace ±inf with NaN, then fill NaN with 0.
5. `StandardScaler().fit_transform` on the **whole** dataset (fit before the split, as the authors did), then cast to `float32`.
6. Labels: `LabelEncoder().fit_transform(targets.values.ravel())`, then cast to `int64`. This maps -1→0 (phishing) and 1→1 (legitimate).
7. Split: stratified 70/15/15 with `random_state=42`. Use `train_test_split(test_size=0.30, stratify=y)`, then split the remaining 30% with `test_size=0.50, stratify=y_tmp`. **The split seed is always 42**, even when the training seed changes.

### 2.2 Model: ResidualMLP (about 72K parameters at d=30)
```
proj : Linear(d, 128)
r1   : ResBlock(128)
r2   : ResBlock(128)
head : Linear(128, 2)
forward(x) = head( r2( r1( relu(proj(x)) ) ) )

ResBlock(h):
  net = Sequential(Linear(h,h), BatchNorm1d(h), ReLU(), Dropout(0.2), Linear(h,h), BatchNorm1d(h))
  forward(x) = relu(net(x) + x)
```
Sanity check: the parameter count should be about 71.3K for d=30. The paper reports "approximately 72K".

### 2.3 Teacher training
- Loss: CrossEntropy. Optimizer: Adam with lr=1e-3 and weight_decay=1e-4. Schedule: `CosineAnnealingLR(T_max=30)`, stepped once per epoch.
- 30 epochs, batch size 512, shuffle the training set.
- After each epoch, evaluate validation accuracy. **Keep the state dict with the best validation accuracy** and load it at the end.
- Report test accuracy and macro-F1. The authors' results CSV reports teacher test accuracy 0.964 on this dataset.

### 2.4 Compression methods
Each method starts from a `copy.deepcopy` of the trained teacher. Run all evaluation and SHAP in `model.eval()` mode.

**(a) Post-training quantization (PTQ), simulated, 8-bit and 4-bit.** This follows the authors' code; the paper names only the bit widths.
Round **every learnable parameter** (Linear weights and biases, BatchNorm weights and biases) to a fixed grid with step `1/(2**(bits-1) - 1)`. BatchNorm buffers (running mean/var) are not parameters and stay unchanged. There is no clamping, so this is fixed-point rounding rather than a true b-bit code, and `eff_size_bytes` is nominal. Store the values back as float so gradients still work for SHAP:
```python
scale = float(2 ** (bits - 1) - 1)   # 127 for 8-bit, 7 for 4-bit
with torch.no_grad():
    for p in model.parameters():
        p.data = torch.round(p.data * scale) / scale
```
No calibration and no fine-tuning. This is the authors' `fake_ptq` (`run_wilcoxon.py`, `run_mprf.py`); it reproduces the paper's PTQ-4bit result. The authors' notebook min-max weight-only version does not (it leaves LCI and accuracy almost unchanged); it matches the paper's separate per-bit sweep (Table 3), not Table 2.

**(b) Pruning: L1 unstructured at 30%, 50% and 70% sparsity.**
For every `nn.Linear` module, including `head`, call `torch.nn.utils.prune.l1_unstructured(module, name="weight", amount=p)`, then `prune.remove(module, "weight")` to make it permanent. Pruning is per layer (local), applies to weights only, and has **no fine-tuning afterwards**. This reproduces the paper's pruning LCI values; the fine-tuned variant in the authors' `src/compression.py` does not.

**(c) Knowledge distillation (KD).**
- Student (the authors' notebook and README; the paper does not specify it): `Sequential(Linear(d,32), ReLU(), Linear(32,16), ReLU(), Linear(16,2))`, about 1.5K parameters.
- Loss (paper: soft cross-entropy, temperature T=3, 15 epochs; α=0.7 comes from the authors' only T=3 implementation, which uses a full-size ResidualMLP student, and from the README's 70% KD / 30% CE split):
  `L = α · KL( log_softmax(z_s/T) || softmax(z_t/T) ) · T²  +  (1-α) · CE(z_s, y)` with `T=3`, `α=0.7`, `KLDivLoss(reduction="batchmean")`.
- Use the teacher in eval mode under `torch.no_grad()`. Optimizer: Adam with lr=1e-3. 15 epochs, batch size 512, shuffled. Return the student from the final epoch.
- No KD implementation in the authors' repo reproduces the paper's Phishing KD values (LCI 0.499, accuracy 0.986, which exceeds the teacher). Expect a gap. Report it; do not tune to close it.

Each compressed model gets a size record: parameter count. For PTQ, also record the effective size, `n_params × bits / 8` bytes. For pruning, also record the count of nonzero weights.

### 2.5 SHAP attributions (paper: GradientExplainer, 300 explained samples, 50 background)
- Explainer: `shap.GradientExplainer(model, background)`, called as `explainer.shap_values(X_explain, nsamples=200)`.
- `background` = the **first 50 rows of the training split**. `X_explain` = the **first 300 rows of the test split**. Use exactly the same two tensors for the teacher and every compressed model.
- Output class: **class index 1** (legitimate), as in the authors' code. Different shap versions return a list of per-class arrays, an `(N, d, C)` array or an `(N, d)` array. Normalize all of them to `(N, d)` for class 1 (`sv[1]` if it is a list, `sv[..., 1]` if it is 3-D).
- GradientExplainer is stochastic. Before every `shap_values` call, set `np.random.seed(shap_seed)` and `torch.manual_seed(shap_seed)` (default `shap_seed=0`) so runs are reproducible.
- Cache each model's attribution array to disk (`.npy`) together with its config, so LCI can be recomputed without re-running SHAP.
- Use `float32` throughout. The PTQ models are float models with quantized values, so gradients are available.

### 2.6 The Logic Consistency Index (paper Eq. 1, Definitions 3.1–3.2, 3.4)
Inputs: teacher attributions `A_T` and student attributions `A_S`, each with shape `(N, ...)`. Any trailing shape is allowed: flatten the trailing dims to `(N, F)` first. Both must have the same shape and correspond to the same N inputs in the same order.

```
LCI(T, S) = ω · ρ̄ + (1 − ω) · J_k            with ω = 0.5, k = 10

ρ̄   = mean over samples i of PearsonR( A_T[i, :], A_S[i, :] )
       Correlate PER SAMPLE across the F features, then average over samples.
       Skip sample i if either vector has std < 1e-8 or the result is NaN.
       Return the number of skipped samples. If all are skipped, ρ̄ = 0.

J_k  = |Top_k(T) ∩ Top_k(S)| / |Top_k(T) ∪ Top_k(S)|      (Jaccard, NOT overlap/k)
       Top_k(M) = indices of the k largest values of mean_i |A_M[i, :]|
       (global importance = mean absolute attribution over the N samples).
       Order features with np.argsort(-importance, kind="stable") so ties break by lowest index.
       If k > F, raise an error. Do not silently clip k.

Logic Collapse:  LCI < τ,  τ = 0.85 (paper's empirical threshold)
MPRF (diagnostic): argmax_j | rank_T(j) − rank_S(j) |, ranks by mean |A| (descending, same stable order as Top_k).
       Return the feature index and its rank shift.
```

Function signature:
```python
def compute_lci(A_T, A_S, k=10, omega=0.5) -> dict:
    # returns {"lci", "shap_corr", "jaccard", "n_skipped", "n_samples", "k",
    #          "mprf_feature", "mprf_shift"}
```
Implementation notes:
- **Do not clip LCI to [0, 1].** The paper states the range is [0, 1], but ρ̄ can be negative (the paper reports ρ̄ = −0.39 for one distilled model). Report the raw value.
- `compute_lci` must be a pure function of two arrays. It must not import torch or shap, so it can be reused with attributions from any source.
- Keep `k` and `omega` as parameters. Use `k=10, omega=0.5` everywhere in this task.

### 2.7 Unit tests for `compute_lci` (must pass before any experiment)
1. Identical arrays → LCI = 1.0, ρ̄ = 1.0, J = 1.0.
2. `A_S = 3 · A_T` (positive scaling) → LCI = 1.0.
3. `A_S = −A_T` → ρ̄ = −1.0 and J = 1.0, so LCI = 0.0.
4. Same global ranking but per-sample noise → J = 1.0 and ρ̄ < 1.
5. Hand-built case with disjoint top-k sets → J = 0.
6. Hand-built Jaccard: top-10 sets sharing 5 features → J = 5/15 = 0.333…, not 0.5.
7. A sample with a constant vector is skipped and counted in `n_skipped`.
8. Shape mismatch raises an error. `k > F` raises an error.
9. Input with shape `(N, a, b)` gives the same result as the same data reshaped to `(N, a·b)`.

---

## 3. Next steps (run in this order)

1. **Fetch the Phishing dataset and verify its shape and class counts (Section 1).**
   *Why:* the paper reports Phishing / ResidualMLP numbers we can compare against. It is the smallest benchmark in the paper (11K rows, 30 features), so the whole grid runs on CPU in minutes.

2. **Implement `compute_lci` and pass all unit tests in Section 2.7.**
   *Why:* the authors' code mixes per-feature and per-sample correlation and overlap/k with Jaccard. The tests pin down the paper's exact definition before any model is involved. That keeps any mismatch with the paper attributable to training or compression, not the metric.

3. **Train the ResidualMLP teacher (Sections 2.1–2.3) for seeds 0–4. The data split stays at seed 42.**
   *Why:* this is the paper's architecture and recipe. The paper does not say how many runs Table 2 comes from, so five seeds tell us whether a gap between our number and theirs is real or just run-to-run noise.

4. **Measure the LCI noise floor for each seed. Report:**
   - (a) **SHAP noise:** LCI(teacher, teacher), computing the second attribution with `shap_seed=1`.
   - (b) **Retraining noise:** LCI(teacher seed s, teacher seed s+1). These are two uncompressed models with identical setup and different initialization.

   *Why:* our study defines "explanation preserved" as an LCI at or above the LCI between the baseline and a copy retrained with a different seed. That way only drops beyond natural run-to-run variation count as a loss. (a) shows how much of any drop comes from the explainer alone. (b) is our data-driven threshold, to compare against the paper's fixed τ = 0.85.

5. **For each seed, apply the 6 compressed configurations from Section 2.4: PTQ-8bit, PTQ-4bit, Prune-30%, Prune-50%, Prune-70% and KD.**
   *Why:* these are exactly the paper's settings for the three compression families our study covers. PTQ-8 is the near-lossless control. PTQ-4, Prune-70 and KD are where the paper reports collapse. Prune-30/50 test the gradual-then-sharp degradation the paper describes. We leave out the paper's LoRA and LoRA-SHAP: they are not size-reducing compression and are outside our study's scope.

6. **For every model: compute test accuracy and macro-F1, SHAP (Section 2.5) and `compute_lci(teacher, model)` (Section 2.6). Record everything.**
   *Why:* the paper's central claim is that accuracy and LCI diverge, so both must come from the same model, the same samples and the same run. Macro-F1 is included because accuracy alone can hide class-level degradation.

7. **Write the results and compare them with the paper.** Write `results/lci-phishing-reproduction/lci-phishing-runs-per-seed-method.csv`, with one row per (seed, method) and these columns:
   `seed, method, level, test_acc, test_f1_macro, shap_corr, jaccard, lci, collapsed(lci<0.85), n_skipped, mprf_feature, mprf_shift, n_params, n_nonzero, eff_size_bytes`.
   Also write `results/lci-phishing-reproduction/lci-phishing-summary-vs-paper.csv` with the mean ± std over seeds per method, next to the paper's values below.
   *Why:* the comparison tells us whether our LCI and compression code is correct before we build on it.

**Paper's reference values (Phishing, ResidualMLP; LCI from Table 2, accuracy from the authors' `results/table2_lci_accuracy.csv`, since the paper omits it):**

| Method | Paper LCI | Authors' CSV Acc |
|---|---|---|
| Teacher | 1.000 | 0.964 |
| PTQ-8bit | 0.999 | 0.964 |
| PTQ-4bit | 0.619 | 0.839 |
| Prune-30% | 0.897 | 0.937 |
| Prune-50% | 0.781 | 0.936 |
| Prune-70% | 0.710 | 0.925 |
| KD | 0.499 | 0.986 |

**Acceptance criteria.** Exact numbers will not match, because the paper reports an unspecified number of runs on unknown hardware. The implementation is considered correct if the qualitative pattern holds on the seed mean:
- PTQ-8bit has ρ̄ close to 1, a top-10 set that differs from the teacher's by at most one feature on every seed (SHAP sampling noise alone can cause one swap), and accuracy within 0.5 pt of the teacher.
- PTQ-4bit, Prune-70% and KD each have LCI < 0.85.
- KD accuracy is reported but not tested against the paper (see Section 2.4c).
- LCI falls monotonically from Prune-30% to Prune-50% to Prune-70%.
- Every paper value lies within roughly 2 std of our seed mean. Flag any that do not. Do not tune hyperparameters to force agreement.

If a criterion fails, first re-check Sections 2.5–2.6: the SHAP class index, per-sample vs per-feature correlation, and Jaccard vs overlap/k. Only after that, investigate training.

## 4. Deliverables
- The new fetch entry in our data scripts, plus the README update.
- The modules from Section 2, with `lci/tests/test_lci.py` passing.
- `lci/run_lci_phishing.py` (`python -m lci.run_lci_phishing`), runnable with one command and with `--seeds` and `--shap_seed` flags.
- The two CSVs from Step 7 and the cached attributions (`results/lci-phishing-reproduction/lci-phishing-shap-attributions.npz` + `-configs.json`).
- A short `results/lci-phishing-reproduction/lci-phishing-report.md` containing: the summary table vs the paper, a pass/fail line for each acceptance criterion, the noise-floor values from Step 4, and any deviation from this spec with its reason.