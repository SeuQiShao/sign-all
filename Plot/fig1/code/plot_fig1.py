"""Render the complete Fig. 1 from the bundled CSV plot data.

Run this file from any working directory.  It renders each panel separately,
then assembles the five final-size panel rasters on the 183-mm canvas.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = CODE_DIR.parent
PLOT_DATA_DIR = PACKAGE_DIR / "plot_data"
PANEL_DIR = CODE_DIR / "panel_outputs"
DEFAULT_OUTPUT_DIR = CODE_DIR / "output"


def _run(script: str, *arguments: str) -> None:
    command = [sys.executable, str(CODE_DIR / script), *arguments]
    subprocess.run(command, cwd=CODE_DIR, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--trace-node", type=int, default=121)
    parser.add_argument(
        "--context-nodes", type=int, nargs=4, default=(230, 112, 378, 934)
    )
    args = parser.parse_args()
    PANEL_DIR.mkdir(parents=True, exist_ok=True)

    _run(
        "panel_a_rebuild.py",
        "--plot-data", str(PLOT_DATA_DIR),
        "--trace-node", str(args.trace_node),
        "--output-dir", str(PANEL_DIR / "a"),
    )
    _run("panel_b_rebuild.py", "--plot-data", str(PLOT_DATA_DIR), "--output-dir", str(PANEL_DIR / "b"))
    _run("panel_c_rebuild.py", "--output-dir", str(PANEL_DIR / "c"))
    _run(
        "panel_d_rebuild.py",
        "--coefficients", str(PLOT_DATA_DIR / "coefficients.csv"),
        "--output-dir", str(PANEL_DIR / "d"),
    )
    _run(
        "panel_e_rebuild.py",
        "--plot-data", str(PLOT_DATA_DIR),
        "--trace-node", str(args.trace_node),
        "--context-nodes", *(str(node) for node in args.context_nodes),
        "--output-dir", str(PANEL_DIR / "e"),
    )
    _run(
        "assemble_fig1.py",
        "--panel-a", str(PANEL_DIR / "a" / "Fig1a_Rossler_rebuild.png"),
        "--panel-b", str(PANEL_DIR / "b" / "Fig1b_candidate_libraries_rebuild.png"),
        "--panel-c", str(PANEL_DIR / "c" / "Fig1c_support_identification_rebuild.png"),
        "--panel-d", str(PANEL_DIR / "d" / "Fig1d_shared_coefficients_rebuild.png"),
        "--panel-e", str(PANEL_DIR / "e" / "Fig1e_prediction_performance_rebuild.png"),
        "--output-dir", str(args.output_dir),
    )


if __name__ == "__main__":
    main()
