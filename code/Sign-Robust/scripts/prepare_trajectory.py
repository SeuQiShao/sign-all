#!/usr/bin/env python3
"""Generate one clean trajectory and persist its true adjacency for robustness runs."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch


CANONICAL_ROOT = Path(__file__).resolve().parents[2] / "SIGN-phase"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--system", required=True, choices=["Kuramoto", "FHN", "Rossler"])
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--python-bin", default=sys.executable)
    p.add_argument("--node-count", type=int, default=1000)
    p.add_argument("--time-points", type=int, default=1000)
    p.add_argument("--time-interval", type=float, default=0.01)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    network = "power_law" if args.system == "Kuramoto" else "small_world"
    dims = {"Kuramoto": 1, "FHN": 2, "Rossler": 3}[args.system]
    init_scale = {"Kuramoto": 10.0, "FHN": 0.5, "Rossler": 1.0}[args.system]
    output_root = args.output_root.expanduser().resolve()
    dataset_root = output_root / f"{args.system}_{args.node_count}_{network}_{args.time_interval}_{args.time_points}"
    raw_file = dataset_root / "raw" / "data_1.pt"
    edge_file = dataset_root / "true_edge_index.pt"
    manifest_file = dataset_root / "trajectory_manifest.json"

    if not raw_file.exists():
        cmd = [
            args.python_bin, "data/generate_dataset.py", "--seed", str(args.seed),
            "--sample_interval", str(args.time_interval), "--num_points", str(args.time_points),
            "--init_dim", str(dims), "--sample_num", "1", "--save_path", str(output_root),
            "--method", "rk4", "--node_num", str(args.node_count), "--network", network,
            "--device", args.device, "--model_name", args.system, "--init_scale", str(init_scale),
            "--obnoise", "0",
        ]
        subprocess.run(cmd, cwd=CANONICAL_ROOT, check=True)

    try:
        data_list = torch.load(raw_file, weights_only=False)
    except TypeError:
        data_list = torch.load(raw_file)
    if not data_list:
        raise RuntimeError(f"No samples found in {raw_file}")
    edge_index = data_list[0].edge_index.detach().cpu().to(torch.long).contiguous()
    torch.save(edge_index, edge_file)
    manifest = {
        "system": args.system, "seed": args.seed, "node_count": args.node_count,
        "dims": dims, "network": network, "time_points": args.time_points,
        "time_interval": args.time_interval, "trajectory_source": "data/generate_dataset.py",
        "trajectory_is_clean": True, "true_edge_index": str(edge_file),
        "edge_count": int(edge_index.shape[1]),
    }
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"dataset_root": str(dataset_root), **manifest}, indent=2))


if __name__ == "__main__":
    main()
