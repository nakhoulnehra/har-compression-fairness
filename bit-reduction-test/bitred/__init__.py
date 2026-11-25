"""Bit-reduction test: explanation-guided mixed-precision weight quantization of the frozen TinyHAR.

Self-contained experiment (see bit-reduction-test/README.md). It reuses the repo's data, fold, model, SHAP
and LCI code (har/, lci/) read-only, and writes only inside bit-reduction-test/.
"""

from __future__ import annotations

import sys
from pathlib import Path

BITRED_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BITRED_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = BITRED_ROOT / "results"
RESULTS_SMOKE_DIR = BITRED_ROOT / "results-smoke"
