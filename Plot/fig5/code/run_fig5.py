"""Rebuild all panels and the final composite for the Fig. 5 submission package."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parent


def main() -> None:
    scripts = [
        "plot_panel_a.py",
        "plot_panel_b_v2.py",
        "plot_panel_c.py",
        "plot_panel_d.py",
        "plot_panel_e.py",
        "assemble_fig5.py",
    ]
    for name in scripts:
        path = CODE_ROOT / name
        print(f"Running {name} ...", flush=True)
        subprocess.run([sys.executable, str(path)], cwd=CODE_ROOT, check=True)
    print(f"Completed. Outputs are in {CODE_ROOT / 'output'}")


if __name__ == "__main__":
    main()

