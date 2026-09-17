"""Run the compact two-phase SIGN pipeline on synthetic Fig. 2 data."""

from __future__ import annotations

import argparse
from pathlib import Path

from trainer import run_case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="SIGN-data/synthetic")
    parser.add_argument("--output", default="SIGN-main/output/fig2")
    parser.add_argument("--epochs", type=int, default=300, help="Phase-II epochs")
    parser.add_argument("--phase2-min-epochs", type=int, default=30)
    parser.add_argument("--phase2-loss-patience", type=int, default=30)
    parser.add_argument("--phase2-support-patience", type=int, default=40)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    files = sorted(data_root.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No NPZ trajectories found under {data_root}")
    for source in files:
        run_case(
            source,
            output_root / source.stem,
            epochs=args.epochs,
            phase2_min_epochs=args.phase2_min_epochs,
            phase2_loss_patience=args.phase2_loss_patience,
            phase2_support_patience=args.phase2_support_patience,
            device=args.device,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
