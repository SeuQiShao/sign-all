"""Reusable perturbations for the SIGN robustness experiments.

The functions here operate on the delivery package's NumPy representation:
trajectories are ``[time, node, dimension]`` and graphs are sparse
``edge_index[2, edge]`` arrays with source -> target orientation.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .graph import GraphData


def add_gaussian_observation_noise(
    trajectory: np.ndarray,
    snr_db: float | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add zero-mean Gaussian noise using the SI SNR definition."""
    x = np.asarray(trajectory, dtype=np.float64)
    if snr_db is None or float(snr_db) <= 0:
        return x.copy()
    signal_power = float(np.mean(x**2))
    noise_power = signal_power * 10.0 ** (-float(snr_db) / 10.0)
    return x + rng.normal(size=x.shape) * np.sqrt(max(noise_power, 1e-12))


def temporal_sample_indices(num_steps: int, observed_points: int) -> np.ndarray:
    """Select exactly ``observed_points`` ordered samples from a trajectory."""
    n = int(num_steps)
    k = int(observed_points)
    if n < 2 or not 2 <= k <= n:
        raise ValueError("observed_points must satisfy 2 <= observed_points <= num_steps")
    return np.linspace(0, n - 1, k, dtype=np.int64)


def delete_edge_fraction(edge_index: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Randomly delete a fraction of supplied directed edges."""
    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    rate = float(np.clip(fraction, 0.0, 1.0))
    remove = int(np.floor(rate * edges.shape[1]))
    if remove == 0:
        return edges.copy()
    rng = np.random.default_rng(seed)
    keep = np.ones(edges.shape[1], dtype=bool)
    keep[rng.choice(edges.shape[1], size=remove, replace=False)] = False
    return edges[:, keep]


def remove_observed_nodes(
    trajectory: np.ndarray,
    edge_index: np.ndarray,
    observed_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep a random node subset and remap the retained edge list."""
    x = np.asarray(trajectory)
    edges = np.asarray(edge_index, dtype=np.int64)
    if x.ndim != 3 or edges.shape[0] != 2:
        raise ValueError("trajectory must be [T,N,D] and edge_index must be [2,E]")
    n = x.shape[1]
    keep_count = max(1, int(np.floor(n * float(np.clip(observed_fraction, 0.0, 1.0)))))
    rng = np.random.default_rng(seed)
    observed = np.sort(rng.choice(n, size=keep_count, replace=False))
    node_map = np.full(n, -1, dtype=np.int64)
    node_map[observed] = np.arange(keep_count, dtype=np.int64)
    mask = (node_map[edges[0]] >= 0) & (node_map[edges[1]] >= 0)
    remapped = node_map[edges[:, mask]]
    return x[:, observed, :], remapped, observed


def corrupt_adjacency_fn_fp(
    edge_index: np.ndarray,
    num_nodes: int,
    fn_rate: float,
    fp_rate: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Create an imperfect adjacency estimate using the SI FN/FP protocol.

    False negatives remove ``floor(fn_rate * |E|)`` true edges. False positives
    add ``floor(fp_rate * |E|)`` sampled non-edges, where ``|E|`` is the
    original true-edge count. Self-loops are excluded.
    """
    true_edges = np.asarray(edge_index, dtype=np.int64)
    if true_edges.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    n = int(num_nodes)
    if n < 2:
        raise ValueError("num_nodes must be at least 2")
    rng = np.random.default_rng(seed)
    original = {(int(src), int(dst)) for src, dst in true_edges.T if int(src) != int(dst)}
    original_count = len(original)
    fn_count = min(original_count, int(np.floor(np.clip(fn_rate, 0.0, 1.0) * original_count)))
    original_list = sorted(original)
    removed = (
        {original_list[int(index)] for index in rng.choice(original_count, size=fn_count, replace=False).tolist()}
        if fn_count
        else set()
    )
    retained = original - removed
    fp_count = int(np.floor(max(0.0, float(fp_rate)) * original_count))
    added: set[tuple[int, int]] = set()
    max_non_edges = n * (n - 1) - len(retained)
    target = min(fp_count, max_non_edges)
    while len(added) < target:
        batch = max(1024, min(target - len(added), 1_000_000))
        src = rng.integers(0, n, size=batch, dtype=np.int64)
        dst = rng.integers(0, n, size=batch, dtype=np.int64)
        for a, b in zip(src.tolist(), dst.tolist()):
            edge = (int(a), int(b))
            if a != b and edge not in retained and edge not in added:
                added.add(edge)
                if len(added) >= target:
                    break
    all_edges = np.asarray(sorted(retained | added), dtype=np.int64).T
    metadata = {
        "true_edge_count": original_count,
        "false_negative_count": len(removed),
        "false_positive_count": len(added),
        "fn_rate": float(fn_rate),
        "fp_rate": float(fp_rate),
    }
    return all_edges, metadata


def adjacency_f1(true_edge_index: np.ndarray, estimated_edge_index: np.ndarray) -> float:
    """Compute directed edge-set F1 for an adjacency estimate."""
    truth = {tuple(map(int, edge)) for edge in np.asarray(true_edge_index).T}
    estimate = {tuple(map(int, edge)) for edge in np.asarray(estimated_edge_index).T}
    tp = len(truth & estimate)
    precision = tp / len(estimate) if estimate else 0.0
    recall = tp / len(truth) if truth else 0.0
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def heterogeneous_kuramoto(
    std: float = 0.0,
    mean: float = 0.3,
    coupling: float = 0.15,
    seed: int = 0,
) -> Callable:
    """Return a Kuramoto RHS with node-specific ``omega_i`` values.

    The frequency vector is created lazily on the first RHS call, making it
    stable across all RK4 substeps for one generated trajectory.
    """
    frequencies: np.ndarray | None = None

    def rhs(t: float, x: np.ndarray, edge_index: np.ndarray, edge_weight: np.ndarray) -> np.ndarray:
        nonlocal frequencies
        if frequencies is None or frequencies.shape[0] != x.shape[0]:
            frequencies = np.random.default_rng(seed).normal(mean, std, size=x.shape[0])
        edges = np.asarray(edge_index, dtype=np.int64)
        weights = np.asarray(edge_weight, dtype=np.float64).reshape(-1)
        src, dst = edges[0], edges[1]
        out = np.zeros_like(x, dtype=np.float64)
        np.add.at(out[:, 0], dst, weights * coupling * np.sin(x[src, 0] - x[dst, 0]))
        out[:, 0] += frequencies
        return out

    return rhs


def fhn_with_coupling(coupling_strength: float = 0.1) -> Callable:
    """Return the E2V3 FHN RHS with configurable coupling strength."""

    def rhs(t: float, x: np.ndarray, edge_index: np.ndarray, edge_weight: np.ndarray) -> np.ndarray:
        edges = np.asarray(edge_index, dtype=np.int64)
        weights = np.asarray(edge_weight, dtype=np.float64).reshape(-1)
        src, dst = edges[0], edges[1]
        coupling = np.zeros(x.shape[0], dtype=np.float64)
        np.add.at(coupling, dst, coupling_strength * weights * (x[src, 0] - x[dst, 0]))
        out = np.empty_like(x, dtype=np.float64)
        v, r = x[:, 0], x[:, 1]
        out[:, 0] = v - v**3 - r + 1.0 + coupling
        out[:, 1] = 0.5 * (0.5 + v - 0.3 * r)
        return out

    return rhs


def make_structured_graph(kind: str, num_nodes: int, seed: int = 0, **kwargs) -> GraphData:
    """Build SI VI.D structured graph classes using the legacy definitions.

    NetworkX is imported only for these optional graph families. The standard
    E2V3 families remain available through :func:`Data_generation.graph.make_graph`.
    """
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("NetworkX is required for DSCM/RGM/RPG/SBM generation") from exc

    n = int(num_nodes)
    key = kind.lower()
    if key == "dscm":
        rng = np.random.default_rng(seed)
        indegree = rng.zipf(2, size=n)
        outdegree = rng.zipf(2, size=n)
        difference = int(indegree.sum() - outdegree.sum())
        if difference > 0:
            outdegree[int(rng.integers(n))] += difference
        elif difference < 0:
            indegree[int(rng.integers(n))] += abs(difference)
        graph = nx.directed_configuration_model(indegree.tolist(), outdegree.tolist(), create_using=nx.DiGraph, seed=seed)
    elif key == "rgm":
        graph = nx.random_geometric_graph(n, radius=float(kwargs.get("radius", 0.05)), seed=seed)
    elif key == "rpg":
        if n > 5000:
            raise ValueError("The legacy RPG construction is quadratic; use it only for the N=1000 structured-topology runs")
        rng = np.random.default_rng(seed)
        latent = rng.normal(size=(n, int(kwargs.get("latent_dim", 8))))
        distances = np.linalg.norm(latent[:, None] - latent[None, :], axis=2)
        probabilities = np.exp(-(distances**2)) * float(kwargs.get("sparsity", 0.5))
        mask = rng.random((n, n)) < probabilities
        np.fill_diagonal(mask, False)
        graph = nx.from_numpy_array(mask)
    elif key == "sbm":
        blocks = int(kwargs.get("blocks", 3))
        sizes = [n // blocks] * blocks
        sizes[-1] += n - sum(sizes)
        p_in = float(kwargs.get("p_in", 0.05))
        p_out = float(kwargs.get("p_out", 0.05))
        probs = np.full((blocks, blocks), p_out, dtype=float)
        np.fill_diagonal(probs, p_in)
        probs *= float(kwargs.get("sparsity_factor", 0.1))
        graph = nx.stochastic_block_model(sizes, probs, seed=seed)
    else:
        raise ValueError(f"unknown structured graph kind: {kind}")
    graph.remove_edges_from(nx.selfloop_edges(graph))
    if graph.number_of_nodes() < n:
        graph.add_nodes_from(range(n))
    edges = np.asarray(list(graph.edges()), dtype=np.int64).T
    if edges.size == 0:
        edges = np.asarray([[0], [1]], dtype=np.int64)
    return GraphData(edges, np.ones(edges.shape[1], dtype=np.float32))
