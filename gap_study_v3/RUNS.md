> Cleanup 2026-11-27: the watcher scripts (`power_watch.ps1`, `keep_awake.ps1`, `chain_fold2.ps1`), the jobs files and the console logs
> were deleted. The commands below stay as the run record. Final decision: `GAP_SUMMARY.md`.

# Gap study v3: run log

**Setup**
- Machine: Ansel (RTX 3050 Ti Laptop 4 GB, Ryzen 7 5800U), torch 2.7.1+cu118, `.venv`, repo root,
  `PYTHONPATH=bit-reduction-test`.
- Real runs are launched by the power-gated watcher `gap_study_v3/power_watch.ps1`, copied unchanged from agent
  B's `gap_study/power_watch.ps1`. It:
  - waits for AC power;
  - stops the workers after 3 min on battery and resumes them with `--resume`;
  - keeps Windows awake;
  - logs AC detection and every job start to `results/gap-study-v3/phase2/logs/power-watch-*.log`.
- Launch:
  `Start-Process powershell -WindowStyle Hidden -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','gap_study_v3\power_watch.ps1','-Jobs','gap_study_v3\jobs-phase2.json'`

**Shared protocol**
- 50 explain windows per inner-val subject, GradientExplainer nsamples 200, background = 100 training windows.
- Calibration seed s = round 4's draw with rng 46 + s.
- Fine-tuning = the 9A recipe: 20 epochs, KD (T 3, α 0.7) + class-weighted CE, full HARTH training windows,
  batch-order seed s.

| # | Date / time | Command | Config | Seeds | Status | Result path |
|---|---|---|---|---|---|---|
| S | 2026-11-26 09:52 | `python -m gap_study_v3.confirm --folds 0 --stages ref rescore ftcore nr ftextra rescore9a --smoke --threads 4 --out <scratchpad>/smoke-v3` | smoke sizes (600 train windows, 3 explain windows / subject, nsamples 20, 1 FT epoch) | 0, 1 | pass, 57 s | scratchpad (not kept) |
| P2 | 2026-11-26 09:56 (AC detected at start) | watcher `-Jobs gap_study_v3/jobs-phase2.json`: 5 workers `python -u -m gap_study_v3.confirm --folds <k> --stages ref rescore ftcore nr ftextra rescore9a --device cpu --ft-device cuda --threads 3 --machine-name Ansel --worker all --resume` | **ref:** original SHAP seed 1 + occlusion. **rescore:** agent B's no-retrain checkpoints (80 %: l1_fullrecon / lamp_recon seeds 0–2, wanda / l1_recon seed 0; 70 / 90 %: l1_fullrecon, lamp_recon seed 0). **ftcore:** l1_kd and l1_fullrecon_kd 90 % seeds 0–2, lamp_recon_kd 90 % seed 0 (lr 1e-4). **nr:** sparsegpt percdamp 0.1 / 1.0 (80 % seeds 0–2; 70 / 90 % seed 0), taylor_fullrecon (80 % seeds 0–2; 70 / 90 % seed 0), l1_fullrecon_cal5x 80 % seed 0. **ftextra:** lr 3e-4 / 3e-5 for l1_kd and l1_fullrecon_kd 90 % seed 0; 80 % seed 0 for both. **rescore9a:** 9A l1_kd 70 / 90 %, park_kd / park 90 % (folds 0, 1, 3) | see config | done (all exit 0) | `results/gap-study-v3/phase2/gap-study-v3-phase2-tinyhar-fold<k>-Ansel-all-{models,per-subject-lci,config}.{csv,json}`; attributions in `data/processed/har/gap-study-v3/attributions/`; checkpoints in `data/processed/har/checkpoints/gap-study-v3/` |
| R2 | 2026-11-26 10:01 | watcher `-Jobs gap_study_v3/jobs-refs2.json`: `python -u -m gap_study_v3.confirm --folds 0 1 2 3 4 --stages refs2 --device cpu --threads 2 --machine-name Ansel --worker refs2 --resume` | reference points for the artefact metrics: agent B's l1_fullrecon 50 % seed 0 (near lossless) and the independently retrained TinyHAR seed 1 (folds 0, 1, 3) | 0 / 1 | see below | `...-fold<k>-Ansel-refs2-models.csv` |
| P2b | 2026-11-26 10:13 (stopped P2's watcher and workers at 10:12; finished models kept, `--resume` skips them; the fine-tunes in progress, ~1 min old, were redone) | watcher `-Jobs gap_study_v3/jobs-phase2b.json`: 5 workers `--folds <k> --stages ref rescore ftcore ftextra rescore9a --threads 2 --worker all`, plus 2 CPU workers `--folds 0 1 2` / `--folds 3 4` `--stages nr --threads 3 --worker nr` | same plan; `nr` moved to its own CPU workers so it runs while the GPU fine-tunes (the GPU was the bottleneck: ≈ 7 min per fine-tuned model per worker with 5 on the GPU) | as P2 | see below | as P2 (+ `...-fold<k>-Ansel-nr-*.csv`) |
| P2c | 2026-11-26 11:37–12:13 | watcher `-Jobs gap_study_v3/jobs-phase2c.json`: 5 workers `--folds <k> --stages ftseeds --ft-selected lamp_recon_kd:0.0003:0 lamp_recon_kd:0.0001:1,2 l1_fullrecon_kd:0.0003:1,2 --threads 2 --worker all` | fair tuning after ftcore/ftextra. lr rule (fixed 11:36, before these runs): per method, the lr with the highest mean LCI_val (seed 0, 5 folds) among lrs whose mean F1 is within 1 pt of that method's best-F1 lr; the selected lr gets seeds 1–2. l1_fullrecon_kd → 3e-4; lamp_recon_kd had only lr 1e-4, so it gets lr 3e-4 seed 0 plus seeds 1–2 at 1e-4. l1_kd → 3e-4, but its LCI is identical at 1e-4 and 3e-4 (0.750 / 0.750), so its 3e-4 seeds were skipped (stated) | 0–2 | done | as P2 |
| P2d | 2026-11-26 12:08 (queued after P2c's done marker) | watcher `-Jobs gap_study_v3/jobs-phase2d.json -WaitFor gap_study_v3/jobs-phase2c.json.done`: `--stages ftseeds --ft-selected lamp_recon_kd:0.0003:1,2` | the lr rule selected lr 3e-4 for lamp_recon_kd (F1 0.805 vs 0.791); seeds 1–2 for the best baseline of the cell | 1, 2 | done 12:29 (all exit 0) | as P2 |
| C | 2026-11-26 12:35 | cleanup at the user's request: deleted agent A's PAIR and agent B's GMR files; moved agent B's baselines and CSVs into gap_study_v3 / results/gap-study-v3 | — | — | done; analyze, diagnose, test_confirmation and build_master_table re-run, bit-reduction-test 38/38 pass | — |
| A | after P2 | `python -m gap_study_v3.analyze` | gate per cell (rules in the module docstring, fixed before any fine-tuned result was read) | — | see below | `results/gap-study-v3/phase2/gap-study-v3-phase2-tinyhar-{gate-per-cell,summary-every-method-level-mean-over-folds-seeds}.csv` |
| D | after P2 | `python -m gap_study_v3.diagnose` | Phase 3 diagnostics (channels, classes, shape, window level) | — | see below | `results/gap-study-v3/phase3/` |
| M | after P2 | `python -m gap_study_v3.build_master_table` | existing files + this study | — | see below | `gap_study_v3/MASTER_TABLE.csv` |
