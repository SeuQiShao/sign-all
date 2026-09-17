"""Generate one SIGN synthetic dataset.

Run from the NC directory, for example:

    python Data_generation/generate_dataset.py --system rossler \
        --num-nodes 1000 --num-steps 2000 --out SIGN-data/synthetic/rossler_3d.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from Data_generation.dynamics import make_dynamics
    from Data_generation.graph import make_graph, load_edge_list
    from Data_generation.integrate import integrate_rk4
else:
    from .dynamics import make_dynamics
    from .graph import make_graph, load_edge_list
    from .integrate import integrate_rk4


def _initial_state(system: str, num_nodes: int, dim: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    if system in {"sis", "gene", "mutual", "mutualistic"}:
        return np.clip(rng.uniform(0.05, 0.5, size=(num_nodes, dim)) * scale, 0.0, 2.0)
    return rng.normal(0.0, scale, size=(num_nodes, dim))


def _add_observation_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    if snr_db <= 0:
        return x.copy()
    signal_power = float(np.mean(x**2))
    noise_power = signal_power * 10.0 ** (-snr_db / 10.0)
    return x + rng.normal(size=x.shape) * np.sqrt(max(noise_power, 1e-12))


def generate(args: argparse.Namespace) -> Path:
    rng = np.random.default_rng(args.seed)
    spec = make_dynamics(args.system)
    if args.edge_list:
        graph = load_edge_list(args.edge_list, num_nodes=args.num_nodes, directed=args.directed)
        num_nodes = args.num_nodes or int(graph.edge_index.max()) + 1
    else:
        num_nodes = int(args.num_nodes)
        graph = make_graph(num_nodes, args.graph, args.edge_prob, args.seed + 17, directed=args.directed)
    times = np.arange(args.num_steps, dtype=np.float64) * float(args.dt)
    x0 = _initial_state(spec.name, num_nodes, spec.dimension, args.init_scale, rng)
    x, x_dot = integrate_rk4(spec.rhs, x0, times, graph.edge_index, graph.edge_weight)
    x_observed = _add_observation_noise(x, args.snr_db, rng)
    if args.sample_every > 1:
        indices = np.arange(0, len(times), args.sample_every, dtype=np.int64)
        times, x, x_dot, x_observed = times[indices], x[indices], x_dot[indices], x_observed[indices]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "system": spec.name,
        "dimension": spec.dimension,
        "num_nodes": num_nodes,
        "num_steps": int(len(times)),
        "dt": float(args.dt * args.sample_every),
        "seed": int(args.seed),
        "graph": args.graph if not args.edge_list else str(args.edge_list),
        "edge_prob": float(args.edge_prob),
        "directed": bool(args.directed),
        "init_scale": float(args.init_scale),
        "snr_db": float(args.snr_db),
        "sample_every": int(args.sample_every),
        "generator": "SIGN E2V3 delivery / Data_generation",
    }
    np.savez_compressed(
        out,
        x=x.astype(np.float32),
        x_observed=x_observed.astype(np.float32),
        x_dot=x_dot.astype(np.float32),
        t=times.astype(np.float64),
        edge_index=graph.edge_index.astype(np.int64),
        edge_weight=graph.edge_weight.astype(np.float32),
        system=np.asarray(spec.name),
        dimension=np.asarray(spec.dimension, dtype=np.int64),
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    out.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out} | system={spec.name} D={spec.dimension} shape={x.shape} edges={graph.edge_index.shape[1]}")
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--system", choices=["kuramoto", "heat", "sis", "gene", "michaelis_menten", "mutual", "fhn", "rossler", "hr", "chua"], required=True)
    p.add_argument("--num-nodes", type=int, default=64)
    p.add_argument("--num-steps", type=int, default=256)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--graph", choices=["erdos_renyi", "ring", "grid", "small_world"], default="erdos_renyi")
    p.add_argument("--edge-prob", type=float, default=0.08)
    p.add_argument("--edge-list", default=None)
    p.add_argument("--directed", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init-scale", type=float, default=1.0)
    p.add_argument("--snr-db", type=float, default=0.0, help="Observation SNR in dB; 0 disables noise")
    p.add_argument("--sample-every", type=int, default=1)
    p.add_argument("--out", required=True)
    return p


if __name__ == "__main__":
    generate(build_parser().parse_args())
