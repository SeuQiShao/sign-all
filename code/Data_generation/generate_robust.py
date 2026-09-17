"""Generate one reproducible SIGN robustness-case dataset.

This is a single-case entry point. Use ``NC/Sign-Robust/scripts/expand_matrix.py``
to expand the manuscript matrices, then schedule the resulting records through
the normal SIGN runner.
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
    from Data_generation.graph import make_graph
    from Data_generation.integrate import integrate_rk4
    from Data_generation.robustness import (
        add_gaussian_observation_noise,
        corrupt_adjacency_fn_fp,
        delete_edge_fraction,
        fhn_with_coupling,
        heterogeneous_kuramoto,
        make_structured_graph,
        remove_observed_nodes,
        temporal_sample_indices,
    )
else:
    from .dynamics import make_dynamics
    from .graph import make_graph
    from .integrate import integrate_rk4
    from .robustness import (
        add_gaussian_observation_noise,
        corrupt_adjacency_fn_fp,
        delete_edge_fraction,
        fhn_with_coupling,
        heterogeneous_kuramoto,
        make_structured_graph,
        remove_observed_nodes,
        temporal_sample_indices,
    )


def initial_state(system: str, n: int, dim: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    if system in {"sis", "gene", "michaelis_menten", "mutual", "mutualistic"}:
        return np.clip(rng.uniform(0.05, 0.5, size=(n, dim)) * scale, 0.0, 2.0)
    return rng.normal(0.0, scale, size=(n, dim))


def graph_for(args: argparse.Namespace):
    if args.network in {"dscm", "rgm", "rpg", "sbm"}:
        return make_structured_graph(
            args.network,
            args.num_nodes,
            seed=args.seed + 17,
            radius=args.rgm_radius,
            blocks=args.sbm_blocks,
            p_in=args.sbm_p_in,
            p_out=args.sbm_p_out,
        )
    return make_graph(args.num_nodes, args.network, args.edge_prob, args.seed + 17, directed=True)


def generate(args: argparse.Namespace) -> Path:
    rng = np.random.default_rng(args.seed)
    spec = make_dynamics(args.system)
    graph = graph_for(args)
    if args.heterogeneity_std > 0 and args.system == "kuramoto":
        rhs = heterogeneous_kuramoto(args.heterogeneity_std, seed=args.seed + 104)
    elif args.coupling_strength is not None and args.system == "fhn":
        rhs = fhn_with_coupling(args.coupling_strength)
    else:
        rhs = spec.rhs

    times = np.arange(args.num_steps, dtype=np.float64) * float(args.dt)
    x0 = initial_state(spec.name, args.num_nodes, spec.dimension, args.init_scale, rng)
    x_clean, x_dot = integrate_rk4(rhs, x0, times, graph.edge_index, graph.edge_weight)
    x_observed = add_gaussian_observation_noise(x_clean, args.snr_db, rng)
    supplied_edges = graph.edge_index.copy()
    true_edges_for_data = graph.edge_index.copy()
    true_edge_weight_for_data = graph.edge_weight.copy()
    metadata: dict[str, object] = {
        "generator": "NC/Data_generation/generate_robust.py",
        "system": spec.name,
        "dimension": spec.dimension,
        "num_nodes": int(args.num_nodes),
        "seed": int(args.seed),
        "network": args.network,
        "num_steps_before_sampling": int(args.num_steps),
        "dt_before_sampling": float(args.dt),
        "snr_db": None if args.snr_db is None else float(args.snr_db),
        "heterogeneity_std": float(args.heterogeneity_std),
        "coupling_strength": args.coupling_strength,
    }

    if args.sample_points is not None and args.sample_points < args.num_steps:
        indices = temporal_sample_indices(args.num_steps, args.sample_points)
        times, x_clean, x_dot, x_observed = (array[indices] for array in (times, x_clean, x_dot, x_observed))
        metadata["observed_points"] = int(args.sample_points)
    if args.sample_interval is not None:
        indices = np.arange(0, len(times), int(args.sample_interval), dtype=np.int64)
        if len(indices) >= 2:
            times, x_clean, x_dot, x_observed = (array[indices] for array in (times, x_clean, x_dot, x_observed))
            metadata["sample_every"] = int(args.sample_interval)

    if args.missing_edge_rate > 0:
        supplied_edges = delete_edge_fraction(supplied_edges, args.missing_edge_rate, args.seed + 101)
        metadata["missing_edge_rate"] = float(args.missing_edge_rate)
    if args.observed_node_fraction < 1:
        x_observed, supplied_edges, observed_nodes = remove_observed_nodes(
            x_observed, supplied_edges, args.observed_node_fraction, args.seed + 102
        )
        x_clean = x_clean[:, observed_nodes, :]
        x_dot = x_dot[:, observed_nodes, :]
        # The state has been compacted to the observed node subset.  Keep the
        # clean graph in the same compact index space; otherwise edge_index
        # would still refer to removed nodes and the canonical PyG loader
        # would receive an invalid graph.
        node_map = np.full(graph.edge_index.max() + 1, -1, dtype=np.int64)
        node_map[observed_nodes] = np.arange(len(observed_nodes), dtype=np.int64)
        true_keep = (node_map[graph.edge_index[0]] >= 0) & (node_map[graph.edge_index[1]] >= 0)
        true_edges_for_data = node_map[graph.edge_index[:, true_keep]]
        true_edge_weight_for_data = graph.edge_weight[true_keep]
        metadata["observed_node_fraction"] = float(args.observed_node_fraction)
        metadata["observed_node_count"] = int(len(observed_nodes))
    if args.fn_rate > 0 or args.fp_rate > 0:
        supplied_edges, corruption = corrupt_adjacency_fn_fp(
            supplied_edges, x_observed.shape[1], args.fn_rate, args.fp_rate, args.seed + 103
        )
        metadata["adjacency_corruption"] = corruption

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    metadata.update({"num_steps": int(len(times)), "supplied_edge_count": int(supplied_edges.shape[1])})
    np.savez_compressed(
        out,
        x=x_clean.astype(np.float32),
        x_observed=x_observed.astype(np.float32),
        x_dot=x_dot.astype(np.float32),
        t=times.astype(np.float64),
        edge_index=true_edges_for_data.astype(np.int64),
        supplied_edge_index=supplied_edges.astype(np.int64),
        supplied_edge_weight=np.ones(supplied_edges.shape[1], dtype=np.float32),
        edge_weight=true_edge_weight_for_data.astype(np.float32),
        system=np.asarray(spec.name),
        dimension=np.asarray(spec.dimension, dtype=np.int64),
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    out.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out} | system={spec.name} D={spec.dimension} shape={x_observed.shape} supplied_edges={supplied_edges.shape[1]}")
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=["kuramoto", "sis", "gene", "michaelis_menten", "mutual", "fhn", "rossler", "hr", "chua"], required=True)
    parser.add_argument("--network", choices=["erdos_renyi", "ring", "small_world", "dscm", "rgm", "rpg", "sbm"], default="small_world")
    parser.add_argument("--num-nodes", type=int, default=1000)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--edge-prob", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-scale", type=float, default=1.0)
    parser.add_argument("--snr-db", type=float, default=None)
    parser.add_argument("--sample-points", type=int, default=None)
    parser.add_argument("--sample-interval", type=int, default=None, help="Keep every k-th generated observation")
    parser.add_argument("--missing-edge-rate", type=float, default=0.0)
    parser.add_argument("--observed-node-fraction", type=float, default=1.0)
    parser.add_argument("--fn-rate", type=float, default=0.0)
    parser.add_argument("--fp-rate", type=float, default=0.0)
    parser.add_argument("--heterogeneity-std", type=float, default=0.0)
    parser.add_argument("--coupling-strength", type=float, default=None)
    parser.add_argument("--rgm-radius", type=float, default=0.05)
    parser.add_argument("--sbm-blocks", type=int, default=3)
    parser.add_argument("--sbm-p-in", type=float, default=0.05)
    parser.add_argument("--sbm-p-out", type=float, default=0.05)
    parser.add_argument("--out", required=True)
    return parser


if __name__ == "__main__":
    generate(build_parser().parse_args())
