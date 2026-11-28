"""Gap study v5: per-class and LCI-part diagnostics from saved validation attributions (v4's diagnose_v4, unchanged,
pointed at the v5 attributions and output folder). No new model.

    python -m gap_study_v5.diagnose_v5 --folds 0 1 2 3 4 --models UNI=v5:alloc_a0-80pct-seed0 A3=v5:alloc_a3-80pct-seed0 ...
Writes results/gap-study-v5/validation/gap-study-v5-validation-tinyhar-diagnostics-{per-class,parts}.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gap_study_v4.diagnose_v4 as D  # noqa: E402

D.ATTR["v5"] = ROOT / "data" / "processed" / "har" / "gap-study-v5" / "attributions"
D.OUT = ROOT / "results" / "gap-study-v5" / "validation"
D.P = "gap-study-v5-validation-tinyhar-diagnostics"

if __name__ == "__main__":
    D.main()
