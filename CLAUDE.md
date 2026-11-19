# CLAUDE.md

## Results

- Every result goes in a single folder: `results/` at the repo root, grouped into one subfolder per set of related runs (e.g. `results/lci-phishing-reproduction/`).
- Give every file in `results/` a specific, descriptive name that says what it holds, e.g. `reproduce-lci-results-x-y.csv`. Don't use generic names like `output.csv` or `results.json`.
- Keep `results/README.md` up to date. It gives a brief, direct summary with numbers for every file in `results/`: one entry per file, listing its key numbers. Whenever a file is added or changed in `results/`, update its entry in the same step. keep it brief, direct, to the point, and has the main numbers findings; it should act as a summary that the research comparisons can rely on.
- keep `environment/requirements.md` up to date with all the project requirements and code that was ran.
- keep `environment/time.md` up to date with the important time analysis that could be needed for paper submission. 
## Working rules
- Never commit or push, and never add AI attribution anywhere. We push ourselves.
- Follow plan/HAR_IMPLEMENTATION.md. Read only the sections the current step needs, not the whole file.
- papers/ holds the source papers (git-ignored). Cite section/table for any architecture or method value.
- Long runs happen on GPU machines, run by us. Claude Code writes code, runs tests and --smoke runs only, then gives the exact commands.
- torch 2.7.1 on every machine (CPU build on laptops, cu118 on GPU machines).

## Current status
See STATUS.md.