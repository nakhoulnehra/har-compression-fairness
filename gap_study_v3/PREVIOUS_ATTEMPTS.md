# Previous attempts (Phase 0 inventory, read-only)

**Cleanup (2026-11-26, after this study, at the user's request):** agent A's PAIR files and agent B's GMR method, docs and
results were deleted. Agent B's no-retraining baselines (`gap_study_v3/baselines.py`) and its baseline / test CSVs
(`results/gap-study-v3/agentB-*`) were kept because the confirmed gap uses them. The paths below are the original
locations.

2026-11-26. Everything below was found with `git status --untracked-files=all`, `git diff` (`--ignore-cr-at-eol`;
most of the raw diff is CRLF noise), `git diff --cached` (empty), `git stash list` (empty), `git branch -a`,
`git log --all --graph -n 50`, `git worktree list` (one worktree), and file modification times. Nothing of theirs was
changed, moved or deleted. This study lives in `gap_study_v3/` and `results/gap-study-v3/`, because `gap_study/` is
already agent B's folder.

Both agents ran on Ansel (this laptop, RTX 3050 Ti 4 GB). Their files are separated cleanly by time: agent A
2026-11-25 21:00 → 2026-11-26 01:14, agent B 2026-11-26 06:18 → 09:23. The only commit not on `main`
(`64720c8`, final test folds 2–4, branch `below-4bit-explainability`) is the team's own, not an agent's.

## Agent A: "PAIR" (calibration-only 2-bit QAT + an attribution-matching loss)

**Where:** all uncommitted.
- Code: `bit-reduction-test/bitred/{pair, run_pair, run_pair_scarce, run_pair_calcheck, run_pair_occlusion}.py`,
  `bit-reduction-test/tests/test_pair.py`.
- Analysis: `plan/gap-analysis-and-pair.md`.
- Results: `results/har-pair-calibration{,-smoke}/`, `results/har-pair-scarce{,-smoke}/`.
- Edits to tracked files: one allow-list line in `bit-reduction-test/tests/test_bitred.py` (lets `run_pair*` load
  test data), plus PAIR paragraphs added to `STATUS.md`, `environment/{requirements,time}.md` and `results/README.md`.

**What it tried.** It wrote a gap analysis, chose calibration-only compression (≈ 980 training windows, QAT + KD
at 2 bits per channel, 20.7 KB), and designed PAIR: KD plus a paired expected-gradient integrand matching term. It
pre-registered PAIR twice, at 978 windows and at ≈ 100 windows, ran both with a test phase, and added an occlusion
cross-check.

**What it found.**
- PAIR failed both pre-registered rules against the best baseline (KD at lr 1e-4):
  - 978 windows: +0.012 LCI, −2.2 pt F1.
  - ≈ 100 windows: −0.031 LCI, −4.4 pt F1. Test, 28 untouched subjects: −0.009 LCI (p = 0.92), −5.0 pt accuracy.
- Robust by-product (baselines only): KD at lr 1e-3 vs 1e-4 gives the same F1 but −0.11 LCI, lower in 5/5 folds,
  at both data sizes. This is a "silent explanation break".
- LCI on the calibration windows ranks models like held-out LCI (Spearman 0.84).

**What went wrong.**
1. **Method first, gap second.** The best baseline (KD lr 1e-4, LCI 0.808) was only 0.08 below the full-data
   reference (0.89): under a 0.10 bar, so there was little headroom to close.
2. **A weak comparator was mistaken for a gap.** The early "gap" partly came from a mis-set baseline lr (1e-3).
3. **Capacity trade-off.** At 2 bits, matching gradients costs accuracy, so PAIR moved along the trade-off
   instead of past it.
4. **Incomplete second setting.** The lamp_recon90 + 4-bit setting was stopped half way (folds 0–2, kd only).

**Results only in uncommitted files** (carried into `MASTER_TABLE.csv`): every `har-pair-*` validation and test
number above.

## Agent B: "gap study + GMR" (post-training pruning without retraining)

**Where:** all uncommitted.
- Code and docs: `gap_study/` (`AUDIT.md`, `DESIGN.md`, `REPORT.md`, `RUNS.md`, `baselines.py`, `gmr.py`,
  `run_phase2/3.py`, `run_test.py`, `diagnose_gradients.py`, `build_master_table.py`, `power_watch.ps1`,
  `jobs-*.json` with `.done` markers, `tests/`).
- Results: `results/gap-study/` (master table, phase2, phase3, test) and `results/gap-study-smoke/`.
- Checkpoints: `data/processed/har/checkpoints/gap-study/` (local).
- Edits to tracked files: 2 lines in `.gitignore`, plus gap-study paragraphs added to `STATUS.md`,
  `environment/time.md` and `results/README.md`.

**What it tried.**
- Phase 0–1: it built a 344-cell master table and ranked 5 candidates.
- Phase 2: it confirmed candidate 1 (unstructured pruning without retraining, 80 %) with 5 baselines × 5 folds
  × 3 calibration seeds: l1_recon, lamp_recon, l1_fullrecon (its own full ridge refit including LSTM / attention),
  wanda_fullrecon, sparsegpt. It also ran a dampening check for l1_fullrecon and an output-vs-gradient diagnostic.
- Phase 3: **it built a new method, GMR** (gradient-matched reconstruction), tuned its μ, extended the grid once
  (amendment 1), and ran a test phase.

**What it found.**
- Gap at 80 % without retraining: best baseline l1_fullrecon, LCI 0.787 vs ceiling 0.931 (gap 0.145, CI
  [0.04, 0.25], 4/5 folds). F1 0.791 vs 0.836. At 70 % there is no significant gap (0.044). At 90 % the gap is
  0.306, but F1 drops by 15 pt.
- GMR: +0.037 LCI on validation and +0.023 on test (p = 0.012). There was no gain on HAR70+, it did not replicate
  at 70 % on test, and about half of the gain is explained by an isotropic weight anchor.

**What went wrong (relative to this task's rules).**
1. **It built and tuned a new method** (GMR, Phase 3), which this task forbids. The μ grid was extended after the
   pre-registered grid produced no eligible value.
2. **SparseGPT was not fairly tuned.** It ran at 1 % dampening only and collapsed (F1 0.39 at 80 %). The report
   attributes this to model size without testing other dampening values. That gives the jury an easy "bad
   baseline" objection.
3. **No explanation-guided existing baseline** (attribution / Taylor-based selection) was run in the
   no-retraining cell.
4. **The mechanism evidence contradicts itself.** l1_fullrecon has the *worst* input-gradient error (1.66, cos 0.28)
   yet the *best* LCI, while lamp_recon has the best gradient cosine (0.50) but lower LCI. "Forward reconstruction
   does not preserve gradients" therefore does not explain the LCI ranking.
5. **Metric artefacts were not checked.**
   - The compressed models' LCI uses the **same SHAP seed** as the reference (common random numbers), while the
     "uncompressed reference" ceiling uses another seed. That is apples vs oranges: the 50 % row scoring above the
     ceiling (0.953 > 0.931) is the symptom.
   - No gradient-free explainer was used, the explainer's self-consistency on compressed models was not measured,
     and LCI was not restricted to windows where the predictions agree.
6. **A candidate was dismissed with a self-referential headroom rule.** The fine-tuned 90 % cell was rejected with
   "headroom 0, it is the full-FT cell". Yet 9A's l1_kd at 90 % keeps accuracy (inner val 0.911 vs 0.927; HARTH test
   0.917 vs 0.919) with LCI 0.744 vs 0.931. That is the "explanations fail while accuracy holds" case this task asks
   for.

**Results only in uncommitted files** (carried into `MASTER_TABLE.csv`): all of `results/gap-study/` (phase 2,
phase 3 / GMR, test).

## Earlier team work (committed; context for the master table)
- 9A grid: 5 folds, seed 0, with test.
- XGP rounds 1–4: folds 0–1, 3 seeds for XGP and 1 for baselines.
- Low-bit diagnostic, XGQ / bit-reduction tests, ultra-small and regime tests, XGP vs LAMP: 5 folds.
- Final test on HAR70+.

The lessons recorded in `plan/gap-analysis-and-pair.md` §2 still hold:
- With full data and KD there is no gap up to 70 % / 4 bits.
- The older-adult gap is an accuracy gap of the base model, not a compression gap.
- LCI ≈ 0.78 must be read against the SHAP ceiling (0.93) and the retrain-seed floor (0.66–0.69).
