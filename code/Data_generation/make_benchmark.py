"""Create the complete small smoke-data catalogue or a large benchmark set."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from Data_generation.generate_dataset import build_parser, generate
else:
    from .generate_dataset import generate


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="SIGN-data/synthetic")
    p.add_argument("--num-nodes", type=int, default=32)
    p.add_argument("--num-steps", type=int, default=128)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--edge-prob", type=float, default=0.12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--snr-db", type=float, default=0.0)
    args = p.parse_args()
    systems = ["kuramoto", "heat", "sis", "gene", "mutual", "fhn", "rossler", "hr", "chua"]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for offset, system in enumerate(systems):
        class Job:
            pass
        job = Job()
        job.system = system
        job.num_nodes = args.num_nodes
        job.num_steps = args.num_steps
        job.dt = args.dt
        job.graph = "erdos_renyi"
        job.edge_prob = args.edge_prob
        job.edge_list = None
        job.directed = True
        job.seed = args.seed + offset
        job.init_scale = 1.0
        job.snr_db = args.snr_db
        job.sample_every = 1
        job.out = str(out_dir / f"{system}_{'1d' if system in {'kuramoto','heat','sis','gene','mutual'} else '2d' if system == 'fhn' else '3d'}.npz")
        generate(job)


if __name__ == "__main__":
    main()

