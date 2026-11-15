# Time

Wall-clock time of each command, run from the repo root on the machine in [specs.md](specs.md).

This should only include the important time analysis that could be needed for paper submission.

machine model: Acer Swift X (SFX14-41G)
commands ran: python -m lci.run_lci_phishing --seeds 0 1 2 3 4 --shap_seed 0
time taken to finish: 817.0 s (13 min 37 s); per seed 162.3 ± 7.8 s (152.9, 170.0, 170.8, 157.9, 159.9)

machine model: Acer Swift X (SFX14-41G)
commands ran: python -m lci.time_lci_phishing_components --seed 0
time taken to finish: 141.6 s

Per-stage breakdown from that command (one seed, CPU only, SHAP cache bypassed):

| Stage | Time | Share |
|---|---|---|
| Data load + preprocess + split | 0.04 s | 0.0% |
| Teacher training (30 epochs, 7,738 rows) | 8.1 s | 5.7% |
| Compression, all 6 methods (KD training is 1.8 s; PTQ and pruning are under 0.02 s each) | 1.9 s | 1.3% |
| SHAP GradientExplainer, 7 models (300 samples × nsamples 200; 17.0–19.9 s each) | 131.4 s | 92.8% |
| LCI computation, 6 comparisons | 0.2 s | 0.1% |

The full run makes 8 SHAP calls per seed (7 here, plus the SHAP-noise call), so SHAP accounts for about 150 of the 162 s per seed.
