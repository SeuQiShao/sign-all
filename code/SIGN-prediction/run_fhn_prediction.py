"""Standalone FHN prediction entry point using joint-vector E2V3 rollout."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from run_prediction import build_parser, main


if __name__ == "__main__":
    main(build_parser("fhn").parse_args())
