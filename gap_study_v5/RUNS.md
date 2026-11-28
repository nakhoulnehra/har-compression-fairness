# Gap study v5: run log (2026-11-27 → 28, Ansel)

**Setup.** Ansel (RTX 3050 Ti Laptop 4 GB, Ryzen 7 5800U, 16 GB), torch 2.7.1+cu118, `.venv`, repo root,
`$env:PYTHONPATH = "bit-reduction-test"`. Workers are started detached by `gap_study_v5/launch.ps1 <name> <args>`
(logs in `gap_study_v5/logs/`, deleted at cleanup); `gap_study_v5/keep_awake.ps1` keeps Windows awake;
`gap_study_v5/watch.sh <pattern>` wakes the agent on a Traceback or a result line. Protocol = v3 / v4
(`gap_study_v3.confirm.score`, unchanged): HARTH inner val, 50 explain windows per subject, 3 subjects per fold,
GradientExplainer nsamples 200, 48 features, primary metric `lci_indep`.

## Step 0: theory check of the v4 gap (21:50, before any v5 run)
v4's gap: one-shot pruning without retraining at 80 %, best baseline 0.773 (mixcal), same-mask reference 0.865 (fmkd,
full data), ceiling 0.932. Weak points found from the documents alone:
1. **Baseline coverage.** v3 Sec. 5 lists OBC (exact OBS, Frantar et al. 2022) and DSnoT (Zhang et al. 2024) as
   "considered but not run". OBC is *the* backprop-free method that accounts for how removals interact (input
   Hessian), which is exactly v3's own requirement 2 for a fix. SparseGPT "collapsed", but it was run without the
   full refit every other baseline gets, so its masks were never tested. BatchNorm re-estimation (standard
   training-free post-pruning step) and a slower magnitude schedule (K = 20) were not run either.
   **If any of these closes ≥ half of x − y, v4's rule 4 drops the gap.**
2. **Upper bound.** x = fmkd at l1's masks. That is neither the highest LCI reachable at 80 % nor the reference for a
   baseline with different masks. An explanation-supervised oracle (gradient field on SHAP path points) could show
   more headroom.
3. **Other conditions** (the theme allows any scenario): deployment shift (sensor noise, sensor rotation) might open
   a larger or earlier gap; HAR70+ (population shift) did not (v3 test: 80 % gap 0.117 < HARTH 0.144; 70 % 0.004).
Order: theory → existing data → cheap CPU experiments → GPU references.

| # | Time | Command | Config | Status | Result |
|---|---|---|---|---|---|
| S1 | 21:58 | `python -m gap_study_v5.run_v5 --folds 0 --stages ref nr5 abl swap orc --smoke --threads 8 --ft-device cuda --out <scratch>/smoke-v5` | smoke | pass, 2 min | scratchpad |
| V1 | 22:00 | 5 workers `launch.ps1 cpu-f<k> --folds <k> --stages ref nr5 abl swap --device cpu --threads 2 --worker nr --machine-name Ansel --resume` | nr5 (6 baselines, 80 %, seed 0), abl (unit-only pruning, probe calibration), swap (24 same-mask weight swaps) | running | `results/gap-study-v5/validation/...-fold<k>-Ansel-nr-*` |
| V2 | 22:00 | 2 workers `launch.ps1 orc-f<k> --folds <k> --stages ref orc --device cpu --ft-device cuda --threads 2 --worker orc ...` (k = 0, 1) | jfm oracle, 80 % / 90 %, lam 1 / 10 | stopped after lam 1 at 80 % (decision point 1) | `...-orc-*` |
| V3 | 22:02 | `python -u -m gap_study_v5.shift_probe --folds 0 1 2 3 4 --threads 2 --resume` | noise σ 0.05 g, rotation ±15° | running | `...-Ansel-shift-models.csv` |

## Decision point 1 (22:10–22:35, after reading the first nr5 and orc rows)
- **obc_fullrecon (OBC masks + v3's full refit) = 0.846 / 0.705 / 0.856 / 0.911 / 0.835 (seed 0, folds 0–4), mean
  0.831**, accuracy within 0.5 pt of the original. v4's best (mixcal) was 0.773, fmkd 0.865: OBC closes
  (0.831 − 0.773) / (0.865 − 0.773) = 63 % of v4's x − y. **By v4's rule 4, v4's gap as stated is dropped**: its
  baseline was not the strongest existing one-shot method.
- Plain OBC (its own OBS weights, no refit) collapses in accuracy (0.71–0.84), as SparseGPT did: the refit to the
  original's outputs is what makes Hessian-aware masks usable here.
- jfm oracle at 80 %, lam 1: fold 0 0.856 (fmkd e20 0.908), fold 1 0.710 (fmkd e40 0.769). Matching the
  explainer's gradient field *lowers* LCI and agreement: no higher ceiling from explanation supervision. The other
  jfm settings (lam 10, 90 %) were stopped to free the GPU.
- New questions, in order: (a) is OBC still below what its *own* masks can reach (fmkd from obc_fullrecon, 40 ep)?
  (b) does iterative OBC (K = 5 nested steps + refit) close the rest? (c) OBC seeds 1–2, 70 % and 90 %.

| # | Time | Command | Config | Status | Result |
|---|---|---|---|---|---|
| V4 | 22:12 | `launch.ps1 obcx --folds 0 1 2 3 4 --stages ref nr5x:obc_fullrecon:0.9:0 nr5s:obc_fullrecon nr5x:obc_fullrecon:0.7:0 --device cpu --threads 2 --worker obcx ...` | OBC 90 % / 70 % seed 0, 80 % seeds 1–2 | running | `...-obcx-*` |
| S2 | 22:31 | smoke of `nr5x:obc_fullrecon_k5:0.8:0` and `fmo:obc_fullrecon-80pct-seed0:0.8` | smoke | pass | scratchpad |
| V5 | 22:32 | `launch.ps1 fmo-a --folds 0 2 4 --stages ref fmo:obc_fullrecon-80pct-seed0:0.8 --device cpu --ft-device cuda --threads 2 --worker fmo ...`; `fmo-b` folds 1 3 at 22:33 (after orc-f1's lam-1 row) | fmkd (v4 recipe, lr 3e-4, 40 ep) from obc_fullrecon seed 0 | running | `...-fmo-*` |
| V6 | 22:32 | `launch.ps1 obck --folds 0 1 2 3 4 --stages ref nr5x:obc_fullrecon_k5:0.8:0 --device cpu --threads 2 --worker obck ...` | iterative OBC + refit, K = 5 | running | `...-obck-*` |
| K1 | 22:40 | — | **all workers stopped on the user's request** (models in progress lost; finished rows kept) | done | — |
| R1 | 23:23 | same commands as V1, V3–V6 relaunched with `--resume` (user: proceed; find the gap if the earlier work did not) | as before | running | as before |

## Decision point 2 (23:30): what x is, and the "highest LCI in theory"
- The v4 gap needs a reference x that a method could reach. With OBC as the best baseline, x must be measured at
  **OBC's masks** (fmo, running). Two oracles bound what is possible at a given size:
  - **jfm2** (generalisable oracle, training windows only): fmkd + gradient-field matching on SHAP path points,
    normalised over the batch (v5's jfm divided per path point, so near-zero teacher gradients dominated: a design
    flaw, a likely reason for its lower LCI).
  - **capfit** (capacity oracle): fmkd + the same term on the path points of the 150 *evaluated* inner-val windows,
    with the explainer's own background and targets. **Not a method and not a fair comparison**: it bounds what a
    model with this mask can represent (the "best achievable in theory" of the task), not what can be learned.
- Fold-level data check first (no run): per-subject LCI shows fold 1 low for *every* method (fmkd 0.78 on all three
  subjects), so the deficit is a property of fold 1's model, not of one subject.

| # | Time | Command | Config | Status | Result |
|---|---|---|---|---|---|
| S3 | 23:31 | smoke `orc2:obc_fullrecon-80pct-seed0:0.8:1:40 cap:obc_fullrecon-80pct-seed0:0.8:10:20` | smoke | pass | scratchpad |
| V7 | 23:33 | `launch.ps1 cap --folds 0 1 --stages ref cap:obc_fullrecon-80pct-seed0:0.8:10:20 --device cpu --ft-device cuda --threads 2 --worker cap ...` | capfit 80 %, lam 10, 20 ep, from OBC | running | `...-cap-*` |
| V8 | 23:36 | `launch.ps1 cap90 --folds 0 --stages ref cap:lamp_recon-90pct-seed0:0.9:10:20 ... --worker cap90` | capfit 90 % from agent B's lamp_recon 90 % seed 0 | running | `...-cap90-*` |

## Decision point 3 (00:00, after fmo fold 0, obc 3 seeds × 5 folds, abl, shift probe)
- obc_fullrecon 80 %: **0.817** (3 seeds × 5 folds, seed SD 0.018), accuracy −0.2 pt; obc_fullrecon_k5 0.817 (no
  gain from the nested schedule). fmkd from OBC's own masks, fold 0: **0.907** (fmkd from l1's masks 0.910) vs OBC
  fold 0 seed mean 0.820: the headroom at OBC's masks is ≈ 0.09 on fold 0.
- abl (unit-only pruning, 5 folds): pruning the LSTM alone costs 0.068 LCI, conv0 0.046, fusion 0.027, conv6 0.025,
  the attention / feed-forward units ≤ 0.007. The full model loses 0.17 (l1) / 0.115 (OBC): the losses interact
  (super-additive). probe calibration (the explainer's own blends and occlusions as calibration inputs) 0.756 = no
  gain: the calibration distribution is not where the problem lives.
- swap (same masks, exchange one unit's weights between l1_fullrecon and fmkd) is **uninformative**: swapping a conv
  without its BatchNorm statistics destroys accuracy (−30 to −63 pt). Stopped; replaced by swap2 (conv + its
  BatchNorm swapped together).
- shift probe: sensor noise σ 0.05 g raises every gap by ≈ 0.02–0.03 (l1_fullrecon 80 %: 0.198 vs clean 0.172;
  fmkd 0.089 vs 0.067; 70 %: 0.095 vs 0.061 but F1 −5.6 pt); ±15° rotation leaves the gaps as clean. No condition
  opens a new, larger gap at a sparsity where clean explanations are intact.
- Rule-4 risk: x − y at OBC's masks is small (≈ 0.05–0.09), so OBC's own settings must be checked: dampening
  {0.001, 0.1}, blended calibration (mixcal), 5× calibration.
- GPU at 3.9 / 4.1 GB with 4 processes: no more GPU workers until one ends.

| # | Time | Command | Config | Status | Result |
|---|---|---|---|---|---|
| K2 | 00:02 | — | stopped the 5 `nr` workers in the swap stage (their nr5 and abl rows are complete) | done | — |
| S4 | 00:04 | smoke of `nr5x:obc_fullrecon_mixcal / _damp0.1 / _cal5x` and `swap2` | smoke | pass | scratchpad |
| V9 | 00:06 | 5 workers `launch.ps1 obcv-f<k> --folds <k> --stages ref nr5x:obc_fullrecon_mixcal:0.8:0 nr5x:obc_fullrecon_damp0.1:0.8:0 nr5x:obc_fullrecon_damp0.001:0.8:0 nr5x:obc_fullrecon_cal5x:0.8:0 swap2 --device cpu --threads 2 --worker obcv ...` | OBC checks + swap2 | running | `...-obcv-*` |

## Decision point 4 (00:25; written before reading jfm2, capfit folds 1–4, fmo folds 3–4)
Read: OBC checks, seed 0, 5 folds: obc_fullrecon 0.831, _mixcal 0.842, _cal5x **0.846**, _damp0.1 0.830,
_damp0.001 0.830. fmo (fmkd at OBC's masks) folds 0–2: 0.907 / 0.782 / 0.845 (mean 0.845) vs obc_fullrecon_cal5x
on the same folds 0.819. capfit fold 0: 80 % **0.922** (ceiling 0.945, accuracy 0.909 vs 0.891), 90 % 0.862
(existing on fold 0: 0.80–0.83).
- The one-shot vs. training distinction is closing: the best one-shot (OBC + 5× calibration) is within ≈ 0.03 of
  fmkd at the same masks. What remains is a residual that **every** existing pipeline leaves (one-shot OBC ≈ 0.84,
  label-free calfm 0.835, full-data fmkd 0.865) below the uncompressed ceiling 0.932, while capfit shows the masks can
  represent ≈ 0.92.
- **Rules for the final gap (fixed now):**
  1. y = the highest mean lci_indep at 80 % over ALL existing methods (one-shot, label-free, full-data), any seed
     count, 5 folds.
  2. x_gen = a reference that never sees the evaluated windows: jfm2 (training windows only) on validation, and
     capfit / jfm2 on the **held-out test** (capfit fits 150 inner-val windows; HARTH test subjects and HAR70+ are
     never seen by it, so on test it is a fair model). x_cap = capfit on validation = the capacity bound ("best
     achievable in theory"), reported as such.
  3. Gap kept if x_gen − y ≥ 0.05 with the fold-paired CI excluding 0, ≥ 4/5 folds positive, occlusion in the same
     direction, accuracy ≥ y's − 0.5 pt.
  4. Otherwise the one-shot cell is re-evaluated with y = best one-shot (OBC family) and x = fmo / capfit, same gate;
     and if that fails too, the report states that no between-method gap passes, and gives the ceiling gap with
     the capacity bound.
- One held-out test run of the selected y / x models after selection (rule 6 of v4).

| # | Time | Command | Config | Status | Result |
|---|---|---|---|---|---|
| V10 | 00:21 | `launch.ps1 jfm2 --folds 1 2 --stages ref orc2:obc_fullrecon-80pct-seed0:0.8:1:40 --device cpu --ft-device cuda --threads 2 --worker jfm2 ...` | jfm2, lam 1, 40 ep, from OBC seed 0 (the folds with headroom) | running | `...-jfm2-*` |
| E1 | 00:22 | — | a chain meant to start capfit folds 2–4 after the fmo workers matched an earlier fold's "done" line and fired at once (6 GPU processes); both stopped within a minute, before any model; re-chained on the fmo processes' exit | done | — |
| V11 | chained | `launch.ps1 cap-b --folds 3 ...` after fmo-b exits, `cap-c --folds 2 4 ...` after fmo-a exits; `--stages ref cap:obc_fullrecon-80pct-seed0:0.8:10:20 --worker cap` | capfit 80 % folds 2–4 | queued | `...-cap-*` |
| R2 | 00:24 | — | read: fmo folds 0–3 0.907 / 0.782 / 0.845 / 0.858 (≤ obc_fullrecon_cal5x on fold 3: 0.909); capfit 80 % fold 1 **0.760** (same-seed 0.802; ceiling 0.914): even fitting the evaluated windows does not lift fold 1 at these masks | — | — |
| T1 | 00:27 | 5 workers `launch_mod.ps1 test-f<k> gap_study_v4.test_confirm --folds <k> --threads 2 --resume --out results/gap-study-v5/test --models OBC=<v5>/fold{k}-Ansel-obc_fullrecon-80pct-seed0.pt OBC5X=<v5>/...obc_fullrecon_cal5x-80pct-seed0.pt FMKD=<v4>/...fmkd_from_l1_fullrecon-80pct-seed0-lr0.0003-e40.pt MIXCAL=<v4>/...l1_fullrecon_mixcal-80pct-seed0.pt L1FR=<B>/...l1_fullrecon-80pct-seed0-damp0.01.pt` | held-out test (HARTH test subjects + HAR70+), candidate set fixed by decision point 4 before any test number; fmo / capfit / jfm2 are added with `--resume` when all their folds exist | running | `results/gap-study-v5/test/` |
| V12 | 00:31 (relaunched 00:35: PowerShell had split the unquoted `1,2`, ValueError before any model) | `launch.ps1 obcs --folds 0 1 2 3 4 --stages ref "nr5x:obc_fullrecon_cal5x:0.8:1,2" "nr5x:obc_fullrecon_mixcal:0.8:1,2" --device cpu --threads 2 --worker obcs ...` | seeds 1–2 of the two best one-shot variants (rule 1 uses every seed) | running | `...-obcs-*` |

Data note (00:30, no run): per fold, fold 1 is fragile from 70 % on (l1_fullrecon 0.747, obc 0.771 vs ceiling 0.914;
other folds ≥ 0.86) and has the lowest retrain-seed floor (0.603 vs 0.72–0.73 on folds 0 / 3): where the original's
explanation is least determined by the data, compression moves it most. Five folds are too few to test this.
| E2 | 00:28 | — | obcs (V12) failed at load: `_ArrayMemoryError` (16 GB RAM, ≈ 1.6 GB free with 13 workers); no model lost elsewhere; requeued to start when the 5 obcv workers exit | requeued | — |

## Decision point 5 (00:45–00:58): the 80 % plateau, and a new candidate (layer budget allocation)
Read (validation, 5 folds unless stated):
- fmo (fmkd at OBC's masks, 40 ep) = 0.907 / 0.782 / 0.845 / 0.858 / 0.925, **mean 0.863**, vs the best one-shot
  obc_fullrecon_cal5x (seed 0) 0.846: x − y = 0.017. **The one-shot vs training gap does not pass (rule 4 → 4).**
- capfit 80 % folds 0 / 1 / 3 = 0.922 / 0.760 / 0.892 (fold 3 below obc_fullrecon_cal5x 0.909): even fitting the
  evaluated windows stays near the same plateau on 2 of 3 folds. The capacity oracle (masks fixed, 20 ep, lam 10) is
  not a tight bound.
- swap2 (conv + BatchNorm swapped together): still −10 to −90 pt accuracy for conv swaps; single-unit swaps of the
  attention / fusion weights change LCI by < 0.03. fmkd's gain is joint and co-adapted, not located in one unit.
  Stopped before lstm / attn_score / other (diagnostic enough).
- Held-out test, fold 0–4, OBC (seed 0): HARTH test 0.842 / 0.798 / 0.831 / 0.886 / 0.823 (ceilings 0.951 / 0.906 /
  0.940 / 0.951 / 0.857), HAR70+ 0.835 / 0.768 / 0.834 / 0.864 / 0.861 (ceilings 0.927 / 0.882 / 0.924 / 0.923 /
  0.917).
**Theory for the new candidate.** Every baseline so far keeps the same per-tensor budget (uniform), or LAMP's
(magnitude-driven, worse: 0.720). Unit-only pruning showed the explanation cost is very unequal per unit (LSTM 0.068,
conv0 0.046 with 140 weights, attention block ≤ 0.007), and the LSTM holds 48 % of the weights. Where the budget goes
is a degree of freedom that output reconstruction does not see (reconstruction error and explanation drift need not
agree). Test: OBC + full refit with fixed group allocations at the same 80 % total (a0 uniform … a6 LSTM strongly
protected; conv0 / attn_score dense in a1–a6), and the label-free, backprop-free existing criterion for choosing
an allocation (SPDY / OWL style: the lowest output KL to the original on the calibration windows, `cal_kl`). Gap if
the KL-chosen allocation (existing) is ≥ 0.05 LCI below the LCI-best one with the usual gate, then confirmed on test
(selection on inner val only).

| # | Time | Command | Config | Status | Result |
|---|---|---|---|---|---|
| K3 | 00:57 | — | stopped the obcv workers in swap2 (V9's OBC checks are complete) | done | — |
| S5 | 00:55 | smoke `--stages ref alloc` | smoke | pass | scratchpad |
| V13 | 00:58 | `launch.ps1 alloc-a --folds 0 1`, `alloc-b --folds 2 3`, `alloc-c --folds 4`, all `--stages ref alloc --device cpu --threads 2 --worker alloc ...` | 7 allocations, OBC + full refit, seed 0, with cal_kl / val_kl | running | `...-alloc-*` |

## Decision point 6 (01:15): allocation is a strong lever, but the existing KL criterion finds it → move to 90 %
Read (80 %, seed 0, folds 0 / 2 / 4 so far): a0 uniform 0.846 / 0.856 / 0.835; **a2 (convs protected) 0.889 /
0.892 / 0.900** (occlusion +0.08–0.16, accuracy −0.8 / +0.5 / −1.1 pt); a3 0.868 / 0.871 / 0.902; a1 (LSTM protected)
0.844 / 0.795 / 0.863. The lowest calibration KL so far is a3 on every fold (0.015 / 0.006 / 0.010 vs uniform 0.052 /
0.030 / 0.025): **the existing label-free allocation criterion already moves most of the way** (≈ +0.03–0.07), so
at 80 % allocation is a stronger baseline, not a gap. With OBC + KL allocation ≈ 0.88 and fmkd 0.863, the 80 %
deficit to the ceiling (0.932) is ≈ 0.05 for the strongest existing combination.
Theory: the tighter the budget, the more the allocation and the refit matter, and the more an output criterion
and the explanation can disagree. At 90 % (5,272 weights kept, ≈ 26.6 KB fp32: an MCU budget) every existing method
so far is ≤ 0.79 (fine-tuned) / 0.70 (one shot, −6 pt), and capfit 90 % on fold 0 reaches 0.862. The 90 % cell is
re-opened with the strongest existing pieces found tonight (OBC, KL-chosen allocation, fmkd from those starts), before
any claim (lesson of decision point 1).

| # | Time | Command | Config | Status | Result |
|---|---|---|---|---|---|
| V14 | chained (after V13 per worker) | `after.sh "<V13 worker>" alloc90-{a,b,c} --folds {0 1, 2 3, 4} --stages ref alloc90 --device cpu --threads 2 --worker alloc90 ...` | 7 allocations at 90 % (same proportions, halved), OBC + refit, cal_kl | queued | `...-alloc90-*` |
| V15 | chained (after T1) | `after.sh "gap_study_v4.test_confirm" cap90-{b,c} --folds {1 2, 3 4} --stages ref cap:lamp_recon-90pct-seed0:0.9:10:20 --ft-device cuda --worker cap90 ...` | capacity oracle 90 %, folds 1–4 | queued | `...-cap90-*` |
| K4 | 01:17 | — | stopped jfm2 in fold 2 (fold 1 had finished just before: **0.764**, vs fmkd at the same masks 0.782: batch-normalised gradient matching does not help either); the 80 % question is answered by the baselines; GPU and RAM go to 90 % | done | `...-fold1-Ansel-jfm2-*` |
| V16 | 01:17 | `launch.ps1 fmo90-a --folds 0 1 2 --stages ref fmo:obc_fullrecon-90pct-seed0:0.9 --device cpu --ft-device cuda --threads 2 --worker fmo90 ...` | fmkd 40 ep from OBC uniform 90 % | running | `...-fmo90-*` |
| T1 done | 01:42 | `python -m gap_study_v5.analyze_test_v5 --models OBC OBC5X FMKD MIXCAL L1FR --pairs ...` (v4's `--analyze` fails on a one-character-short pooled-F1 label string; the v5 summary drops pooled F1, all else as v4) | 40 test subjects (22 HARTH + 18 HAR70+) | done | `results/gap-study-v5/test/gap-study-v5-test-tinyhar-*` |

## Decision point 7 (01:45): held-out test of the 80 % cell
- Test, subject level (40 subjects): OBC 0.834, OBC5X 0.827, FMKD (v4's x) 0.837, MIXCAL (v4's y) 0.767, L1FR 0.762;
  ceiling 0.917. FMKD − MIXCAL = +0.069 (36/40 subjects, p < 1e-4), so v4's gap replicates on test **against its
  own baseline**; but OBC − MIXCAL = +0.067 (39/40) and FMKD − OBC = +0.003 (23/40, p = 0.26): **the existing OBC
  closes v4's gap on held-out subjects too.** Final: v4's gap is dropped (rule 4, validation and test agree).
- What remains at 80 % for every existing method is ≈ 0.08 below the ceiling on test (OBC 0.083, FMKD 0.080), at
  accuracy within 1 pt.
- fmo90 fold 0 (fmkd from OBC uniform 90 %) 0.745: below lamp_recon_kd / calkd / park_kd on that fold (0.80–0.83).

## Rules for the 90 % cell (written 01:50, before any alloc90 / cap90 fold 1–4 / fmo90 fold 1–2 number is read)
1. **y90** = the highest 5-fold mean lci_indep at 90 % among existing methods (v3 / v4 / v5 rows: KD / fmkd / calfm /
   calkd / Park fine-tunes, one-shot OBC / allocations, and fmkd from the KL-chosen allocation) whose mean accuracy is
   within 1 pt of the original's; the best one-shot is reported separately (its accuracy loss stated).
2. KL-chosen allocation = per fold, the allocation with the lowest cal_kl (label-free, backprop-free); never chosen
   by LCI.
3. **x90** = capfit 90 % (capacity bound) on validation; on the held-out test it is a fair model (it only fitted
   inner-val windows), so x90_test − y90_test is the generalising part.
4. Gap kept if x90 − y90 ≥ 0.05 on validation with the fold-paired 95 % CI excluding 0 and ≥ 4/5 folds positive,
   occlusion in the same direction, x90's accuracy ≥ y90's − 1 pt; and confirmed on test (x90_test − y90_test > 0,
   one-sided Wilcoxon over the 40 subjects, alpha 0.05).
5. If 90 % fails, the report states that no between-method gap passes tonight, and gives the per-level ceiling gaps
   of the strongest existing combination and the capacity bounds.

## Decision point 8 (01:45): every route reaches the same 80 % plateau
- 80 % allocation, all 5 folds: KL-chosen per fold (a5, a5, a3, a5, a3; label-free, backprop-free) = **0.865**,
  accuracy 0.922 (−0.5 pt), = full-data fmkd 0.865. The LCI-best allocation per fold (an oracle choice) 0.875.
  capfit (λ 10, 20 ep) 5 folds 0.875. fmo 0.864. Every route lands at ≈ 0.865–0.875, 0.06 below the ceiling.
- Open question for "the highest LCI in theory": is 0.875 a capacity limit of 80 % TinyHAR or a weak oracle? capS =
  capfit with λ 100, 40 epochs, from the KL-chosen allocation's masks, on the two folds with the largest residual
  (1, 2).

| # | Time | Command | Config | Status | Result |
|---|---|---|---|---|---|
| K5 | 01:46 | — | stopped fmo90-a in fold 1 (fold 0 row kept: 0.745, below the existing 90 % fine-tunes) | done | — |
| V17 | 01:46 | `launch.ps1 capS --folds 1 2 --stages ref cap:alloc_a5-80pct-seed0:0.8:100:40 --device cpu --ft-device cuda --threads 2 --worker capS ...` | strong capacity oracle | running | `...-capS-*` |
| R3 | 02:00 | — | read: capfit 80 % 5 folds = 0.922 / 0.760 / 0.869 / 0.892 / 0.931 (mean 0.875); alloc90 folds 0 / 2 / 4: KL choice a3 on each, 0.779 / 0.777 / 0.835 (uniform 0.719 / 0.746 / 0.776; accuracy −2.6 / −1.5 / −0.7 pt) | — | — |
| T2 | chained (after capfit fold 4) | per fold: `python -m gap_study_v4.test_confirm --folds <k> --threads 2 --resume --out results/gap-study-v5/test --models CAP=<capfit 80 %> AKL=<KL-chosen allocation 80 % of fold k> FMO=<fmkd from OBC 80 %>` (folds 0 1 2 and 3 4 in two sequential chains) | held-out test of the oracle and of the strongest one-shot | running | `results/gap-study-v5/test/` |
| V18 | chained (after obcs) | `launch.ps1 fmoA90 --folds 0 2 4 --stages ref fmo:alloc_a3-90pct-seed0:0.9:20 --device cpu --ft-device cuda --threads 2 --worker fmoA90 ...`; folds 1 / 3 after alloc90 with their own KL choice | fmkd 20 ep (v4's 90 % budget) from the KL-chosen 90 % allocation: the strongest existing 90 % combination | queued | `...-fmoA90-*` |
| R4 | 02:40 | — | read: T2 (test, CAP / AKL / FMO, 5 folds) done; capfit 90 % folds 0 / 1 / 3 = 0.862 / 0.662 / 0.820; **fmoA90 fold 0 = 0.908** (accuracy 0.898 vs 0.891, F1 0.807 vs 0.818); v3's tuned 90 % baseline on fold 0 was 0.801 | — | — |
| T3 | 02:45 | per fold, sequential: `python -m gap_study_v4.test_confirm --folds <k> --threads 3 --resume --out results/gap-study-v5/test --models LAMPKD90=<v3>/fold{k}-Ansel-lamp_recon_kd-90pct-seed0-lr0.0003-e20.pt AKL90=<v5>/fold{k}-Ansel-<KL-chosen 90 % allocation>-90pct-seed0.pt` (FMOA90 added per fold when trained) | held-out test of v3's tuned 90 % baseline vs the strongest existing 90 % combination | running | `results/gap-study-v5/test/` |

## Decision point 9 (03:20): 90 % and the capacity probes
- fmoA90 (fmkd 20 ep from the KL-chosen 90 % allocation) folds 0 / 1 / 3 = **0.908 / 0.790 / 0.877** (accuracy
  0.898 / 0.902 / 0.931 vs original 0.891 / 0.911 / 0.935); v3's tuned 90 % baseline (lamp_recon_kd lr 3e-4) on the
  same folds 0.801 / 0.646 / 0.783. v3's second gap closes with existing pieces too (rule 4).
- capfit 90 % (λ 10, 20 ep), 5 folds: 0.862 / 0.662 / 0.791 / 0.820 / 0.856 = 0.798, at or below fmoA90 on every
  fold with both.
- capS (λ 100, 40 ep, from the 80 % KL allocation's masks), fold 1: **0.800** (fmo 0.782, capfit 0.760, AKL 0.737,
  ceiling 0.914). Even a hard fit to the evaluated windows stays at ≈ 0.80 on fold 1: its residual is a capacity /
  representability limit of the 80 % model, not a missing objective.
- Rules for the 90 % cell (written 01:50): y90 = the strongest existing combination; x90 = capfit 90 %;
  x90 − y90 < 0 on folds 0 / 1 / 3 → **the 90 % cell fails rule 4**.
- Per the 01:50 rule 5: no between-method gap passes tonight. The report gives the residual ceiling gaps of the
  strongest existing combination and the capacity bounds.
- T3 (test, LAMPKD90 / AKL90) running; T4 (FMOA90 test) chained to start when fmoA90 ends.
| T4 done | 03:35 | `analyze_test_v5 --models OBC OBC5X FMKD MIXCAL L1FR CAP AKL FMO LAMPKD90 AKL90 FMOA90 --pairs ...` | 90 % test: FMOA90 0.832 vs v3's LAMPKD90 0.767 (+0.065, 36/40 subjects, p < 1e-4, accuracy −0.3 pt); AKL90 0.777 (−2.7 pt) | done | `results/gap-study-v5/test/` |
| K6 | 03:39 | — | stopped capS in fold 2 (no row; fold 1's 0.800 is reported), keep-awake restarted | done | — |
| V19 | 03:40 | 5 workers `launch.ps1 allocs-f<k> --folds <k> --stages ref "allocs:0.8:1,2" --device cpu --threads 2 --worker allocs ...` | calibration seeds 1–2 of the 7 allocations at 80 % (KL choice per seed) | running | `...-allocs-*` |
| V20 | 03:40 | `launch.ps1 fmoA80-{a,b,c}` with `--stages ref fmo:<KL-chosen 80 % allocation>-80pct-seed0:0.8:20 --ft-device cuda --worker fmoA80` (a5: folds 0 1 3, a3: folds 2 4) | fmkd 20 ep from the KL-chosen 80 % allocation: does combining the existing pieces lift the best baseline further? | running | `...-fmoA80-*` |
| R5 | 04:10 | `python -m gap_study_v5.analyze_v5` | V19: alloc_kl over 3 calibration seeds **0.864** (seed SD 0.013), accuracy −0.5 pt; vs mixcal +0.091 [0.048, 0.134] 5/5; vs fmkd −0.001. V20: fmkd 20 ep from alloc_kl 80 % = 0.924 / 0.763 / 0.842 / 0.891 / 0.898 (0.864): no gain over the one-shot allocation | done | summary / paired-gaps CSVs |

## Final decision (04:15, by the rules written at decision point 4 and 01:50)
- **No between-method gap passes.**
  - 80 %: best existing = OBC + full refit with the KL-chosen layer budget, 0.864 (test 0.858), = fmkd 0.865;
    best oracle 0.875 (+0.011, CI [−0.011, 0.033]).
  - 90 %: best existing = fmkd from the KL-chosen allocation, 0.851 (test 0.832); capacity oracle 0.798.
  - v4's gap (0.773 vs 0.865) and v3's second gap (0.781 at 90 %) were weak-baseline artefacts: uniform / LAMP layer
    budgets and magnitude masks.
- What is left: the residual of the strongest existing compressors to the uncompressed ceiling, 0.068 / 0.081
  (validation), 0.059 / 0.085 (test). It concentrates on fold 1 (≈ 0.80 even for a direct fit), walking / standing,
  and the one-shot model's older-adult accuracy (−2.3 pt).
- Cleanup: `gap_study_v5/logs/`, `watch.sh`, `after.sh`, `keep_awake.ps1`, `launch.ps1`, `launch_mod.ps1` deleted
  (as in v4); the commands above and in `environment/requirements.md` are the run record.
