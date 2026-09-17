"""Prepare the FHN prediction experiment data in SIGN-data/prediction."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from Data_generation.generate_dataset import generate
else:
    from .generate_dataset import generate


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="SIGN-data/prediction/fhn_2d.npz")
    p.add_argument("--num-nodes", type=int, default=1000)
    p.add_argument("--num-steps", type=int, default=200)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--edge-prob", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=20260913)
    p.add_argument("--snr-db", type=float, default=0.0)
    args = p.parse_args()
    job = argparse.Namespace(
        system="fhn", num_nodes=args.num_nodes, num_steps=args.num_steps, dt=args.dt,
        graph="erdos_renyi", edge_prob=args.edge_prob, edge_list=None, directed=True,
        seed=args.seed, init_scale=0.8, snr_db=args.snr_db, sample_every=1, out=args.out,
    )
    generate(job)


if __name__ == "__main__":
    main()
