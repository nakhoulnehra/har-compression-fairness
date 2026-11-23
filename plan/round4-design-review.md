# XGP round 4 (final): design review

Written 2026-11-23. This is a review only: no code, no runs.

**Status (2026-11-23, later):** the decisions taken on this review are pre-registered in `plan/HAR_IMPLEMENTATION.md` 2.5 (b), "Round 4 pre-registration". That section is binding where it differs from this file:
- C is in, conv filters only;
- A is dropped except calibration on training windows;
- there is no LSTM / attention reconstruction, and XGP's budgets are capped instead;
- 9A-ext is merged into 9B;
- the "beats" rule is gain ≥ 0.02 and ≥ 5 of 6 paired runs;
- the decision day is 27 Nov.

**Amendment 1 (2026-11-23, before `--prepare`):** the pre-registration now removes all synthetic older-like data from round 4. Where this review uses older-like copies, FA-LCI_val or "both window sets", the plan's amended text applies:
- XGP's groups default to the subject;
- the importance is the mean of the 3 worst groups;
- fine-tuning uses group DRO over the subjects, with `*_kd_dro` controls for the baselines;
- validation uses LCI_val, worst-subject LCI_val and F1_val on real inner-val windows;
- older vs general and walking-aid are evaluation-only groups, used in 9B.

Every round is disclosed in the paper.

**Inputs:**
- `STATUS.md`;
- plan 2.5 (b), "Round 3 pre-registration";
- `har/xgp3.py` and `har/run_xgp_pilot3.py`;
- `results/har-compression-grid-analysis/` (9A);
- `results/har-xgp-pilot/round3/`: phase 1 for folds 0 and 1, the tuning selection, and phase 2 for **fold 0 only**. Phase 2 for fold 1 has not run yet, so the round-3 decision is still open.

**New diagnostic (read-only, CPU, 11 s per fold, not saved in `results/`).** The raw, un-normalised round-3 importance I at θ0, per tensor and per output row:
- the round-3 calibration windows of each fold (426 / 459 per group);
- EG m = 8, EG seed 0;
- the script is `r4_importance_diag.py` in the session scratchpad.

If any number from it goes into the paper, rerun it as a proper `results/` file.

All validation numbers below are means over folds 0–1 and seeds 0–2 (n = 6 runs), unless they are marked "fold 0".

## Constraints (added 2026-11-23)

**Round 3 stays reproducible.**
- Round 4 goes in **new files only**: `har/xgp4.py`, `har/run_xgp_pilot4.py`, `har/tests/test_xgp4.py`, and the output folder `results/har-xgp-pilot/round4/` (prefix `har-xgp-pilot4-tinyhar-`).
- Round 4 imports from `har/xgp3.py` and `har/augment.py`. It never edits them, and never edits `har/run_xgp_pilot3.py` or `results/har-xgp-pilot/round3/`.
- The commit holding round 3's final decision gets the tag `xgp-round3-final`. We tag it ourselves.
- A shared helper that round 4 needs (e.g. LAMP or a PTQ wrapper in `compression.py`) is **added**, never changed in behaviour. Round 3's tests and its smoke chain must still pass unchanged.
- The round-4 pre-registration JSON records the tag's commit hash.

**Pre-registered replacement rule (round 4 against round 3).**
- **Round-3 reference.** Round 3's *final* XGP (from `har-xgp-pilot3-tinyhar-decision.json`) is rerun inside the round-4 pilot as the row `xgp_r3`:
  - with round 3's own pre-registered settings (calibration on the round-3 inner-val calibration windows, its β and r per scenario);
  - at the round-4 focus levels (80 / 90 / 95%) and in every round-4 regime;
  - on the same folds, seeds and evaluation windows.

  This is needed because round 3 never ran at 80 or 95%, and its 90% rows came from a different regime set. For a level that round 3 did not select (80%, 95%), `xgp_r3` uses the configuration round 3 selected at the nearest level: 90% for both. This is fixed now.
- **Evaluation windows.** Both rounds are scored on round 3's F1 and explain windows (`har-xgp-pilot3-tinyhar-fold{k}-inner-val-window-roles-older-like-params.csv`, with round 3's calibration windows excluded), whatever calibration set round 4 uses. The comparison then uses identical windows.
- **Rule, per regime (no fine-tuning; KD 100%; KD 5%).** Round 4's final replaces round 3's final only if both hold:
  - (i) mean FA-LCI_val over the focus levels (folds 0–1 × seeds 0–2) **> round 3's + 0.005** (the round-3 tie tolerance);
  - (ii) **at every focus level**, mean F1_normal ≥ round 3's − 1 pt **and** mean F1_older-like ≥ round 3's − 1 pt.

  Otherwise **round 3 is final** for that regime, and round 4 is reported as a disclosed, unsuccessful round.
- Round 4's own eligibility against the baselines (below) must also hold. If round 3 itself has no frozen final in a regime (round-3 exit code 2), round 4 can be adopted there only through its own rule.
- Mixed outcomes across regimes are allowed and reported per regime. This is never decided after seeing 9B.

---

## 0. What the data already say (this drives every recommendation)

1. **Reconstruction, not the explanation score, gives round 3 its gains.** Without fine-tuning:

   | FA-LCI_val (F1 normal) | 70% | 90% |
   |---|---|---|
   | `l1` | 0.39 (0.34) | 0.21 (0.05) |
   | `l1_recon` | 0.79 (0.74) | 0.48 (0.61) |
   | `xgp` (β 0.1) | 0.80 (0.77) | 0.49 (0.68) |

   - In the fold-0 ablation, `xgp_no_recon` collapses (70%: 0.34; 90%: 0.16).
   - `xgp_no_score` matches the full method at 70% (0.869 vs 0.870) and beats it in LCI at 90% (0.552 vs 0.516). It loses in F1 at 90% (0.645 vs 0.715).
   - So the I term buys F1 at 90%, not explanation consistency.
   - Reconstruction is a published technique (OBC / SparseGPT). A fair round 4 must give it to the baselines, and then XGP's own margin is small.
2. **Fine-tuned, XGP's margin grows with sparsity.**
   - Selected `xgp_ft` minus the best baseline, in FA-LCI: about 0 at 50%, +0.02 at 70% (0.864 vs 0.845), +0.07 at 90% (0.746 vs 0.673). F1 at 90%: 0.725 vs 0.690.
   - The fold-0 control `l1_recon_ft` already reaches 0.793 at 90% (`l1_kdaug` 0.769, `xgp_ft` 0.828). Again, a large part of the gap is reconstruction.
3. **Fewer epochs favour pruning-time repair.** At 90% (budget curve, seed 0, n = 2), XGP after 2 epochs ≈ `l1_kd` after 20:

   | 90% | 2 epochs | 5 epochs | 20 epochs |
   |---|---|---|---|
   | `xgp_ft(0.1, 0.5)` FA-LCI | 0.60 | 0.71 | 0.75 |
   | `l1_kd` FA-LCI | 0.49 | 0.63 | 0.67 |
   | `xgp_ft(0.1, 0.5)` F1 | 0.71 | 0.72 | 0.73 |
   | `l1_kd` F1 | 0.59 | 0.65 | 0.69 |

   - `l1_recon` starts were not on the curve. Whether this is an XGP effect or a reconstruction effect is untested.
4. **The group parts did nothing measurable.**
   - LCI_older-like ≥ LCI_normal in most rows, so FA-LCI_val = LCI_normal and the "max over groups" never binds.
   - In fold 0, `xgp_no_group` ≈ `xgp` without fine-tuning.
   - DRO hurt at 90% (`xgp_ft_no_dro` 0.848 vs `xgp_ft` 0.828, fold 0). The augmentation was mixed (`xgp_ft_no_aug` 0.906 vs 0.891 at 70%).
   - The older-like copies do reproduce the **accuracy** gap: F1 normal − older-like ≈ 0.09–0.12, against 9A's real non-stair F1 gap of −0.098. They do not reproduce an **explanation** gap.
5. **9A: after fine-tuning there is no explanation gap left to close.**
   - With KD the LCI gap is between −0.02 and +0.01 at every level.
   - The accuracy gap is mostly **pre-existing**: the original model's is −0.145 (CI −0.21 to −0.09).
   - Compression adds to it only at 90%: `l1` −0.209 (+6.4 pt), `l1_kd` −0.172 (+2.7 pt).
   - What XGP can realistically improve:
     - (i) the explanation gap without fine-tuning or with a small budget (`l1` without fine-tuning at 70%: −0.088);
     - (ii) the *compression-induced* part of the accuracy gap at ≥ 90%.
   - The domain-shift part of the gap (−14.5 pt) cannot be removed by HARTH-only pruning. Claims must be framed as Δgap = gap(compressed) − gap(original).
6. **Per-layer importance (new diagnostic).** Raw I share against parameter share, both folds:

   | tensor group | params | I share f0 / f1 (normal) | \|w\| share | residual error after recon at 90%, last step (fold 0, xgp) |
   |---|---|---|---|---|
   | convs.0 (140 weights) | 0.27% | 7.7% / 7.1% | 0.9% | **0.46** |
   | convs.3/6/9 | 22.3% | 48.7% / 42.7% | 17.6% | 0.08–0.10 |
   | attention in/out proj | 5.9% | 8.0% / 7.3% | 9.2% | not refit |
   | FFN linear1/2 | 5.9% | 3.6% / 3.6% | 7.3% | 0.33–0.46 |
   | fusion | 17.8% | 12.0% / 15.3% | 14.8% | 0.15 |
   | LSTM ih + hh | 47.6% | 20.0% / 23.9% | 50.0% | **not refit** |

   - The convolutions are under-budgeted by uniform sparsity, and the LSTM is over-budgeted.
   - At 90% uniform, `convs.0` keeps 14 of its 140 weights (5 per filter), so at least half of the first-layer filters die. At 95% it keeps 7.
   - The LSTM, 48% of the weights, is never reconstructed.
   - Within a tensor, I and |w| correlate at Spearman 0.40–0.75. At the filter level in the convs, I and filter L1 norm are **uncorrelated** (ρ −0.09 to 0.27), and the lower-I half of the filters holds only 10–31% of I.
   - The I term therefore changes *which weights* survive only moderately. It would change *which filters* survive a lot.
   - I_normal against I_older-like: ρ 0.55–0.89, so the groups differ somewhat, but neither dominates any layer.

---

## A. Slow-mover proxy (slowest 25% of the fold's HARTH training subjects by walking cadence)

**(1) Likely to help? Low.**
- The round-3 group machinery (group-aware score, DRO) had no measurable effect (§0.4), and A feeds the same machinery.
- 25% of 15 training subjects is about 4 people. HARTH's participants are adults, so cadence varies within a normal adult range; check the age range in HARTH Sec 3.1.
- 9A's drift is in the **posture** classes (standing, sitting, lying), where walking cadence carries no information.
- Two parts of A do not work as proposed:
  - **The F1 check cannot use these subjects.** The frozen model was trained on them, so their F1 is in-sample. The F1 check has to stay on inner-val, which has 3 subjects per fold and often 0–1 slow movers.
  - **Scoring and calibration on training subjects is fine.** It is actually better than round 3, see the next point.

**(2) Better alternatives.**
- **Validate the proxy on CPU before spending GPU time (a gate).**
  - For each fold k, compute per-subject cadence on fold k's 3 inner-val subjects: 15 (fold, subject) pairs, about 13 distinct people, all HARTH, none HAR70+.
  - Use the dominant thigh-acceleration frequency over each subject's pooled walking windows. Single 2.56-s windows have a 0.39 Hz FFT resolution, which is too coarse.
  - Correlate cadence with the original model's accuracy and with the 9A inner-val LCI drop of `l1` without fine-tuning and `l1_kd` at 90%.
  - Pre-register a threshold, e.g. Spearman ρ ≤ −0.3 with the drop in the expected direction. A enters round 4 only if the gate passes.
  - n ≈ 15 is small, so a pass is weak evidence. A fail is still informative.
- **No-proxy alternative: subjects as DRO groups** (worst-subject robustness; Sagawa et al. 2020 with each training subject a group). It needs no cadence assumption, but the round-3 DRO result makes even this low priority.
- **Keep from A in any case:** move the calibration set (score + reconstruction) **from inner-val to training windows**, class-balanced, with their older-like copies. This removes round 3's open caveat ("calibration windows come from the same 3 inner-val subjects"), frees all inner-val windows for F1 and SHAP, and costs nothing. The recon baselines use the same calibration set.

**(3) Fairness and validity risks.**
- Cadence is chosen by us, a priori. Its link to HAR70+ is an assumption we cannot test without the test set, and we must not test it there.
- The 4-subject group means high variance in DRO weights.
- If A is used, it must be disclosed as an untested proxy, like the older-like transform.

**(4) Cost.**
- Gate: 0.5 day of code, minutes of CPU.
- A as a training signal: +0.5 day of code. GPU cost about equal to the existing older-like copies, i.e. a negligible change per run.

**Recommendation:** run the gate. Otherwise drop A as a component and keep only "calibration from training windows".

## B. Explanation-guided layer budgets

**(1) Likely to help? Yes. This is the best-supported new part,** above all at 90–95% (§0.6).
- Uniform sparsity starves the convolutions, which hold half to 56% of the importance on 22.6% of the weights, and `convs.0` in particular.
- It leaves the LSTM over-provisioned, at 48% of the weights against 20–24% of the importance.
- The reconstruction residual is largest exactly in `convs.0`. That residual cannot be fixed, because the row has only 0–1 kept weights.
- Prediction: B helps most at 95%, helps at 90%, and is neutral at 80%.

**(2) Better alternatives and controls.**
- **The generic control should be LAMP** (Lee et al., ICLR 2021, layer-adaptive magnitude pruning), not plain global magnitude.
  - Global magnitude is a known strawman: it can empty whole layers, and its scale depends on each layer's parametrisation.
  - ERK (Evci et al., ICML 2020) is the other standard; LAMP is the stronger, data-free choice.
  - Reviewers would ask for one of them. Use LAMP as the generic control, and report global magnitude descriptively only.
- **Allocation rule (pre-register).**
  - keep_l ∝ I_l (raw, un-normalised, summed over the tensor) with **water-filling**: cap at the tensor size and redistribute the excess, so the total n_nonzero equals the uniform total exactly.
  - Use the same group combination as the score: the max of the group-normalised sums.
  - Floor: ≥ 1 kept weight per output row and ≥ 2% per tensor, so no filter is dead by construction.
  - The budgets are computed once at θ0, before step 1 of the iterative pruning. Recomputing them per step makes the budgets drift and blurs the ablation.
- **Factorial ablation (the clean proof), all with the same reconstruction:**

  | budget ↓ / selection → | L1 | XGP score |
  |---|---|---|
  | uniform | `l1_recon` | `xgp` (round 3) |
  | LAMP (generic) | `lamp_recon` | `xgp_lamp` |
  | XGP budgets | `l1_xb_recon` | `xgp_xb` |

  - The budget effect is the XGP-budget row against the LAMP row, in both columns.
  - The selection effect is the XGP column against the L1 column, in each row.
- **Extend the reconstruction to the LSTM and the attention in-proj (all recon methods, baselines too).**
  - The gate pre-activations are linear in [x_t, h_{t−1}], so `weight_ih` / `weight_hh` refit row-wise exactly like a Linear on hooked LSTM inputs, with the pruned model's x_t and h_{t−1} against the original's gates.
  - q, k, v are linear in the token input.
  - This covers 52% of the prunable weights that round 3 never refits.
  - It helps the baselines equally, so it does not widen XGP's margin, but it raises the bar honestly. Include it only if it can be done before the pilot (est. 1 day). Otherwise state it as a limitation.

**(3) Fairness and validity risks.**
- Different layer budgets give the same n_nonzero but **different FLOPs**: the convs are 70% of the multiply-adds (our count: ≈ 1.37 M of ≈ 1.96 M MACs per window).
  - Keeping more conv weights raises the sparse FLOPs.
  - Report both. The claim is "same size (n_nonzero)", not "same compute".
  - Unstructured sparsity gives no real speed-up on CPU anyway, so compute claims belong to C.
- LAMP and the XGP budgets must use the same floor.
- The per-layer sparsity table goes in the paper.

**(4) Cost.**
- Code: about 0.5 day for budgets, LAMP and water-filling, plus tests (n_nonzero equality, the floor, β = 0 + uniform = round 3).
- GPU: negligible beyond the runs themselves; I at θ0 is already computed.

## C. Structured mode: XGP-S against L1 filter pruning

**(1) Likely to help? Yes, for XGP's own score this is the most promising part.**
- At the filter level, I is uncorrelated with filter L1 (§0.6). So XGP-S and Li et al. (2017) remove genuinely different filters, whereas in unstructured pruning the two rankings largely overlap.
- It is also the only mode with real latency and memory gains on a wearable CPU.
- Expected latency gain: modest. TinyHAR is ≈ 54 k parameters and ≈ 2 M MACs per window, and at batch 1 the framework overhead, the LSTM's 8 sequential steps and the attention dominate. Measure it; do not promise it.

**(2) Better alternatives and controls.**
- **Baselines, all followed by the shared KD recipe:**
  - `l1_filter`: Li et al., ICLR 2017, filter L1 norm.
  - `l1_filter_recon`: the same filters plus least-squares refit of the next layer. This is the channel-pruning-with-reconstruction family: He, Zhang & Sun, ICCV 2017; ThiNet, Luo et al., ICCV 2017.
  - `taylor_filter_recon`: Molchanov et al., CVPR 2019, |Σ w·∂L/∂w| per filter on the **loss**, plus reconstruction.
- The last one matters most. XGP-S's summed |w·∂(alignment)/∂w| is Molchanov's Taylor criterion with an explanation objective in place of the loss. `taylor_filter_recon` against `xgp_s` therefore isolates exactly "explanation objective vs loss objective". Without it, a reviewer can attribute the gain to Taylor scoring in general.
- Optional: network slimming (BN γ; Liu et al., ICCV 2017), because every conv has BN. Run it descriptively, only if time is left.
- **Structural groups in TinyHAR (decide before coding):**

  | group | removes | clean? |
  |---|---|---|
  | convs.0/3/6 output filters | conv row + BN channel + next conv's input slice | yes. The BN-bias constant of a removed channel is not exactly foldable because of zero padding at the edges; the next layer's refit absorbs it. |
  | FFN hidden units | linear1 row + linear2 column | yes |
  | fusion output units | fusion row + LSTM `weight_ih` column | yes |
  | LSTM hidden units | 4 gate rows (ih, hh) + hh column + `attn_score` column + **head column** | shrinks the head's input. The head is not scored, but its columns go. Allow it and state it (recommended: the LSTM is 48% of the weights). |
  | convs.9 filters = transformer d_model | — | **exclude**: LayerNorm statistics change, not function-preserving |

  - Use the same per-group removal fraction for every method. Params and FLOPs are then identical across methods by construction.
  - Levels: remove 50% and 75% of the units in each group. Report the resulting parameter and FLOP reductions (about 60–75% of the parameters at the 75% level).
- **Latency protocol:**
  - Physically slice into a smaller TinyHAR; test that its output equals the masked model's to 1e-5.
  - torch 2.7.1 CPU on Aya's laptop (plan 2.7): 1 thread and the default thread count, batch 1 and batch 256, warm-up, median of 200 runs.
  - Same procedure for the dense model and for every method.

**(3) Fairness and validity risks.**
- The baselines get the same reconstruction, the same KD, the same calibration windows and the same widths.
- Structured pruning is the new headline track. Its levels must be fixed before any run, not chosen after seeing curves.
- 9A has no structured rows, so 9A-ext (below) is needed for the test grid.

**(4) Cost.**
- Largest code item: about 1.5–2 days for slicing, the group definitions, the equivalence tests, the latency harness and structured reconstruction (refitting the next layer reuses `ridge_refit_rows`).
- GPU per fold: see the pilot table.

## D. Settings for all methods

- **Levels 80 / 90 / 95%: yes.**
  - The margin grows with sparsity (§0.2), and the deployment story is at ≥ 90%.
  - At 95%, uniform sparsity is degenerate (`convs.0` keeps 7 weights), so baselines must get LAMP budgets there too. Otherwise the comparison is a strawman.
  - Drop 50% (a tie in round 3) and 70% from the pilot. 9B still reports 30–70% from 9A.
- **KD with 5% of the training windows: yes.**
  - It models on-device or personalised repair with little data, and §0.3 suggests pruning-time repair matters most when fine-tuning is weak.
  - Pre-register:
    - the subset: stratified by subject × class, one seeded index set per fold, shared by all methods;
    - the same 20 epochs (so 1/20 of the steps), not the same number of steps;
    - the teacher logits on the subset only.
- **Pruning + PTQ.**
  - 9A: `ptq8` is lossless (FA-LCI 0.994, accuracy gap unchanged), so prune + 8-bit adds nothing. Run it at one level as a sanity row only.
  - Prune + **4-bit** (`ptq4` alone: FA-LCI 0.904) can compound, and is the realistic microcontroller setting.
  - Apply `ptq_symmetric` to every method's final model (zeros stay exactly zero).
  - Risk: the ridge refit can widen the weight range, which hurts per-tensor 4-bit. That is a real interaction to report, not to fix after seeing it.
  - Report the true size: 4-bit values + index overhead. Say whether the sparse index is counted (plan 2.5 currently does not count it).
- **The rest of the recipe:**
  - The fine-tuning recipe is the one round 3's final decision selects (aug and DRO in or out), applied to every method.
  - Round 3's fold-0 data lean towards **plain KD**: `no_aug` / `no_dro` were as good or better. If fold 1 confirms this, use plain KD for everyone. It is simpler, cheaper, and removes a confound.

## E. Budget curve 2 / 5 / 20 epochs, all seeds

- **Yes, but only where it decides a claim:** the 90 and 95% levels, KD 100%, and the four key methods (`l1_recon_kd`, `lamp_recon_kd`, `xgp_kd`, `xgp_xb_kd`).
- Round 3 had n = 2 per point (seed 0 × 2 folds), too thin for the low-budget claim in §0.3. All seeds gives n = 6.
- The `*_recon` baselines **must** be on the curve. That is the missing comparison.
- Cost: 48 extra SHAP assessments per fold ≈ 1.2 h (uni-2080).

---

## 9A-ext: baseline-only settings on the 9A test grid first

**Purpose.** Check that the problem exists in the new settings (95%, 5% data, pruning + 4-bit, structured) for the *standard* methods before XGP is finalised. Then round 4 targets observed drift, not assumed drift.

**Validity: this is the second test-informed design step.** HAR70+ stays test-only for every *training or tuning* decision, but looking at baseline test results before finalising XGP is what round 3's disclosure already covers. To keep it defensible:
1. **Freeze the round-4 design and both rules (the pre-registration) before 9A-ext runs.** 9A-ext may decide only *which settings are reported as headline settings*, by the pre-registered criterion below. It never changes an XGP hyperparameter, component or rule.
2. **Pre-registered "the problem exists" criterion, per setting** (method × level × fine-tuning regime). The setting is a headline setting if, for at least one standard method, any of these holds:
   - (a) compression-induced older accuracy gap: Δgap = gap(compressed) − gap(original) < 0, with the bootstrap CI excluding 0;
   - (b) an older-minus-general LCI gap with the CI excluding 0;
   - (c) FA-LCI below τ_global (0.583) or more than 0.05 below the best 9A fine-tuned row at the same level.

   All settings are reported either way.
3. 9A-ext rows become part of the 9B grid and are never rerun. Each model still sees the test set once, which keeps the "test used once" rule per model.
4. Disclose it in the paper next to round 3's disclosure.

**Runs per fold (seed 0, as in 9A; same SHAP sets, about 765 windows; uni-2080 costs: about 125 s SHAP per model, about 70 s per 20-epoch KD run):**

| block | models | est. |
|---|---|---|
| `l1_recon` without fine-tuning at 30/50/70/80/90/95% (uniform; LAMP at 95%) | 6 | 13 min |
| `l1_recon_kd` at 70/80/90/95% | 4 | 13 min |
| `l1` without fine-tuning + `l1_kd` at 80 and 95% (LAMP at 95%), `lamp_recon(_kd)` at 90/95% | 8 | 23 min |
| KD on 5% data: `l1_kd_5pct`, `l1_recon_kd_5pct` at 80/90/95% | 6 | 14 min |
| Structured: `l1_filter`, `l1_filter_recon`, `taylor_filter_recon` without fine-tuning and + KD, at 50 and 75% units | 12 | 35 min |
| Pruning + PTQ4 on `l1_kd`, `l1_recon_kd`, `l1_filter_recon_kd` at 90/95% (structured at 75%); PTQ8 sanity at 90% | 7 | 15 min |
| **Total** | **43** | **≈ 1.9 h per fold (uni-2080)** |

- All 5 folds take about 9.5 h of GPU in total: about 5.7 h on nakhoul-3050 (folds 0, 1, 3; ×1.3 for throttling, see `environment/time.md`) and about 3.8 h on uni-2080 (folds 2, 4).
- Plus the CPU latency rows (Aya's laptop, under 30 min).
- Code needed first: everything in the "baseline" half of B, C and D (LAMP, the floor, structured slicing and reconstruction, the 5% subset, PTQ composition, the 95% level, `l1_recon` in `run_compression_grid`).
- 9A-ext is therefore **not cheaper in code** than the round-4 pilot. It only moves the baseline GPU work earlier.

**Recommended order:** finish round-3 phase 2 for fold 1 → round-3 decision → write the round-4 pre-registration, including the 9A-ext criterion → code and tests → 9A-ext and the round-4 pilot. The last two can run in parallel on different machines, because 9A-ext cannot change the pilot.

---

## Proposed round-4 pilot (folds 0 and 1, HARTH inner val only, ≤ 12 h per GPU)

**For the 26 Nov decision, only the U track runs, plus the `xgp_r3` reference rows** (see Constraints and Schedule). The S track below is kept as the design for a later round; it is not part of round 4's rule.

**Base:** round 3's final XGP (for the score, iterative K = 5 and reconstruction), with calibration from training windows (A), LSTM/MHA reconstruction for every recon method if it is ready (B), and the fine-tuning recipe from round 3's decision, applied to every method. Seeds 0–2.

**Tracks and methods:**

- **U (unstructured), levels 80 / 90 / 95%.**
  - The 3 × 2 factorial of B: `l1_recon`, `lamp_recon`, `l1_xb_recon`, `xgp`, `xgp_lamp`, `xgp_xb`.
  - `snip_recon`: the loss-Taylor control at the weight level.
  - `l1` without fine-tuning as a reference.
- **S (structured), levels 50 / 75% of units:** `l1_filter`, `l1_filter_recon`, `taylor_filter_recon`, `xgp_s`.
- **Regimes:** without fine-tuning; KD 100% (E 20); KD 5% (E 20). The U-track 5% regime runs only at 90 and 95%.
- **Extras:** the 2/5-epoch curve (E); PTQ4 on the KD-100% finals of `l1_recon`, `lamp_recon`, `xgp`, `xgp_xb`, and the S methods at 75%, at 90/95%; PTQ8 sanity at 90%, seed 0.

**Cost per fold (uni-2080: SHAP ≈ 90 s per assessment on 300 + 300 windows; XGP start ≈ 60 s; KD run ≈ 60 s, 5% ≈ 10 s):**

| block | assessments | est. |
|---|---|---|
| U without fine-tuning: 3 levels × (l1, l1_recon, lamp_recon, snip_recon once; l1_xb_recon, xgp, xgp_lamp, xgp_xb × 3 seeds) | 48 | 1.6 h |
| U KD 100%: 7 starts (all but `l1`; `l1_kd` added) × 3 seeds × 3 levels | 63 | 2.6 h |
| U KD 5%: the same 7 × 3 seeds × 2 levels | 42 | 1.2 h |
| S without fine-tuning: 2 levels × (3 deterministic + xgp_s × 3) | 12 | 0.4 h |
| S KD 100% and KD 5%: 4 × 3 seeds × 2 levels × 2 regimes | 48 | 1.6 h |
| Budget curve (E): 4 methods × 3 seeds × 2 levels × 2 points | 48 | 1.2 h |
| PTQ4 + PTQ8 sanity | 30 | 0.8 h |
| Setup (original SHAP, teacher logits, I at θ0, budgets) | — | 0.2 h |
| **Total** | **291** | **≈ 9.6 h uni-2080; ≈ 12.5 h nakhoul-3050 if throttled** |

- To stay ≤ 12 h on the 3050, run it cool or in two `--resume` sessions.
- If that is not possible, drop these in this order: PTQ8 sanity → `snip_recon` with KD 5% → curve point 5 → the U-track KD 5% at 95%.
- The SHAP sets stay fixed (300 + 300 explain windows); do not shrink them to save time.

**Ablations that prove where the gains come from (all at matched n_nonzero, or matched widths for S):**

| claim | comparison |
|---|---|
| XGP budgets beat generic budgets | `xgp_xb` vs `xgp_lamp` and `l1_xb_recon` vs `lamp_recon` |
| The explanation score beats magnitude | `xgp_*` vs `l1_*` (same budget row) |
| The explanation objective beats the loss objective | `xgp` vs `snip_recon`; `xgp_s` vs `taylor_filter_recon` |
| The gain is not just reconstruction | every comparison above has reconstruction on both sides |
| Iterative pruning and reconstruction are necessary | round-3 phase 2, cited, not rerun |
| The low-budget advantage | the curve, `xgp_xb` vs `lamp_recon` at 2/5 epochs |
| Robust to deployment quantization | PTQ4 rows, the same comparisons |

**Pre-registered rules (proposal):**

- **Eligibility, per track × regime × level.**
  - B_set = the best mean F1 on each window set among that cell's baselines. Baselines now **include the recon, LAMP, SNIP/Taylor-recon and filter-recon methods**.
  - Eligible = F1_normal ≥ B_normal − 1 pt and F1_older-like ≥ B_older-like − 1 pt.
- **Final per track × regime.**
  - Candidates: `xgp`, `xgp_lamp`, `xgp_xb` (U); `xgp_s` (S).
  - The candidate must be eligible at every level of the track.
  - Final = the highest FA-LCI_val averaged over the levels.
  - Ties within 0.005 → fewer components (uniform < LAMP < XGP budgets), then higher FA-LCI.
  - No eligible candidate → that track and regime is not frozen; report it, no fallback.
- **"Clear win" (decides what the paper may claim, not what is frozen).**
  - Final − best baseline ≥ +0.03 FA-LCI_val, averaged over the levels, **and** the same sign in both folds **and** at each headline level (90 and 95% for U; 75% for S).
  - Otherwise the claim is "matches the best baseline at the same size".
- **Secondary, reported and pre-declared:** Δ(F1 normal − F1 older-like) against the original model, i.e. the compression-induced part of the accuracy gap on the proxy.
  - FA-LCI_val on the proxy is in effect LCI_normal (§0.4), so this F1 Δ is the only proxy measure of the group goal. It must be named as such.
- 9B then runs the frozen finals beside all 9A and 9A-ext baselines on all 5 folds, with all seeds for the curve, using the test set once.

## Schedule: what fits a decision on 26 Nov (presentation 3 Dec)

Today is 23 Nov, so there are about 3 days.

| day | work | fits? |
|---|---|---|
| 23 Nov | Round-3 phase 2, fold 1 (≈ 3.5 h, nakhoul-3050) → `--analyze --phase ablation` → round-3 decision; we tag `xgp-round3-final`. Code for round 4 starts (U track only). | yes |
| 24 Nov | `xgp4.py` (LAMP, XGP budgets, water-filling, floor, 95%, 5% KD subset, PTQ4 on finals, the `xgp_r3` row, curve), `run_xgp_pilot4.py`, tests, CPU smoke; the pre-registration (design + both rules + the 9A-ext criterion) written into the plan **before any GPU run**; the A gate on CPU | tight but yes |
| 24 Nov night → 25 Nov | Round-4 pilot, U track: fold 0 on uni-2080, fold 1 on nakhoul-3050 | yes, if the 3050 runs cool (see the cost below) |
| 25 Nov evening → 26 Nov | `--analyze` → round-4 decision and replacement decision | yes |
| 26 Nov → 2 Dec | 9B (frozen final + all baselines on the test grid), analysis, figures | 9B ≈ 2.6 h per fold (9A rate) plus the new rows; fits, on both GPUs |

**Does not fit a 26 Nov decision (flagged):**
- **C, structured mode.** About 1.5–2 days of code (slicing, structured reconstruction, equivalence tests, latency harness) on top of the U track, with no room left for its GPU runs. Move it to the paper's future work, or to a post-presentation round 5 with its own pre-registration. It must not be added to the round-4 rule after the fact.
- **The LSTM / MHA reconstruction extension.** About 1 day of code. Out of scope for round 4; state it as a limitation.
- **9A-ext before the decision.** The GPUs are busy with the round-3 fold-1 ablation and the round-4 pilot until about 25 Nov evening.
  - The U-track part of 9A-ext (no structured rows): ≈ 1.2 h per fold, ≈ 6 h over 5 folds.
  - It can run 26–27 Nov, after the decision, as part of the 9B grid.
  - That is still valid, because 9A-ext was never allowed to change XGP (see 9A-ext, point 1). It then documents rather than gates the settings.
  - The structured 9A-ext rows drop together with C.
- **A as a training component.** Only the CPU gate fits. The verdict is drop or gate, and with the deadline, drop.
- **The round-4 pilot with every block.** Without the S track and with the added `xgp_r3` rows (≈ 24 assessments, ≈ 0.9 h), it comes to ≈ 8.5 h on uni-2080 and ≈ 11 h on nakhoul-3050 when throttled.
  - If the 3050 throttles past 12 h, apply the drop order above.
  - A run that is not finished by 25 Nov 22:00 is analysed only on finished blocks *if the pre-registration says so*. Write that clause in: blocks are analysed in the order no fine-tuning → KD 100% → KD 5% → curve → PTQ, and an unfinished regime has no round-4 decision, so round 3 stays final there.

## Summary of recommendations

| part | verdict | why |
|---|---|---|
| A slow-mover proxy | CPU gate first; otherwise drop. Keep "calibration from training windows" | the group parts were inert in round 3; posture drift; the F1 check cannot use training subjects |
| B layer budgets | **adopt**; LAMP as the control, water-filling, floor, budgets fixed at θ0; add LSTM/MHA reconstruction for all if ready | convs: half to 56% of the importance on 22.6% of the weights; `convs.0` starved; LSTM unreconstructed |
| C structured | best design candidate (`l1_filter_recon` and `taylor_filter_recon` baselines; exclude convs.9 / d_model), but **does not fit 26 Nov**: a later round or future work | filter I ⟂ filter L1, so XGP's score matters most here; the only real latency gain |
| D settings | 80/90/95 (LAMP for baselines at 95), KD 5%, PTQ4 (PTQ8 sanity only), the round-3 recipe for all | margin grows with sparsity and with less fine-tuning; PTQ8 is lossless in 9A |
| E curve | all seeds, 90/95 only, 4 methods, recon baselines included | the low-budget claim is n = 2 now and has no recon control |
| 9A-ext | run it **after** the round-4 pre-registration is frozen, with a pre-registered "problem exists" criterion; ≈ 9.5 GPU-h over 5 folds | otherwise it is a second, undisclosed test-informed tuning step |

**Honest risk for the "clear win" goal.** Once the baselines get reconstruction and LAMP, XGP's unstructured margin may shrink to within noise. Round 3 already shows the reconstruction carrying most of the effect. The parts most likely to give a margin of XGP's own are the **structured score (C)** and the **explanation budgets at 95% (B)**. If neither clears the bar, the paper's contribution is the measurement (RQ1, RQ2, RQ0) plus a method that matches the strongest fair baseline. It should then be written that way, not tuned further.

**Open decisions for us:**
1. Allow structured removal of LSTM hidden units (it shrinks the head's input)?
2. Is the LSTM/MHA reconstruction extension in scope before the pilot?
3. Run the A gate or drop A outright?
4. Run 9A-ext before or in parallel with the pilot? Both are valid once the pre-registration is frozen.
