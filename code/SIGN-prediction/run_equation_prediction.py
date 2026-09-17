"""Generic joint-vector E2V3 prediction entry for SIGN.

Use ``run_fhn_prediction.py`` and ``run_sst_prediction.py`` for the two
dedicated prediction experiments.  This entry point is for one-dimensional
SIS/MM (and other canonical systems) used by the temporal prediction protocol.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from run_prediction import build_parser, main


if __name__ == "__main__":
    main(build_parser("equation").parse_args())
