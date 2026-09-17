"""Graph generators used by the synthetic network-dynamics benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class GraphData:
    """Directed edge-list representation shared by the generator and SIGN."""

    edge_index: np.ndarray  # [2, E], source -> target
    edge_weight: np.ndarray  # [E]


def _ensure_connected_edges(num_nodes: int, src: list[int], dst: list[int], rng: np.random.Generator) -> None:
    """Add a ring backbone so every node participates in the dynamics."""
    if num_nodes < 2:
        return
    present = {(int(a), int(b)) for a, b in zip(src, dst)}
    for node in range(num_nodes):
        target = (node + 1) % num_nodes
        if (node, target) not in present:
            src.append(node)
            dst.append(target)
            present.add((node, target))


def make_graph(
    num_nodes: int,
    kind: str = "erdos_renyi",
    edge_prob: float = 0.08,
    seed: int = 0,
    weight_scale: float = 1.0,
    directed: bool = True,
) -> GraphData:
    """Create a graph without requiring torch-geometric.

    Edges use the same convention as E2V3: ``edge_index[0]`` is the source
    node and ``edge_index[1]`` is the receiving/target node.
    """
    n = int(num_nodes)
    if n < 2:
        raise ValueError("num_nodes must be at least 2")
    rng = np.random.default_rng(seed)
    kind = kind.lower().replace("-", "_")
    src: list[int] = []
    dst: list[int] = []

    if kind in {"ring", "cycle"}:
        for i in range(n):
            src.append(i)
            dst.append((i + 1) % n)
            if not directed:
                src.append((i + 1) % n)
                dst.append(i)
    elif kind in {"grid", "lattice"}:
        side = int(np.ceil(np.sqrt(n)))
        for node in range(n):
            row, col = divmod(node, side)
            for dr, dc in ((0, 1), (1, 0)):
                nr, nc = row + dr, col + dc
                other = nr * side + nc
                if nr < side and nc < side and other < n:
                    src.append(node)
                    dst.append(other)
                    if not directed:
                        src.append(other)
                        dst.append(node)
    elif kind in {"small_world", "watts_strogatz"}:
        # A light-weight Watts--Strogatz construction.
        k = max(2, min(n - 1, int(round(max(2.0, n * edge_prob)))))
        if k % 2:
            k += 1
        half = min(k // 2, (n - 1) // 2)
        for i in range(n):
            for step in range(1, half + 1):
                j = (i + step) % n
                target = int(rng.integers(n)) if rng.random() < 0.15 else j
                if target != i:
                    src.append(i)
                    dst.append(target)
    elif kind in {"erdos_renyi", "er", "random"}:
        p = float(np.clip(edge_prob, 0.0, 1.0))
        mask = rng.random((n, n)) < p
        np.fill_diagonal(mask, False)
        rows, cols = np.nonzero(mask)
        src.extend(rows.tolist())
        dst.extend(cols.tolist())
        if not directed:
            src.extend(cols.tolist())
            dst.extend(rows.tolist())
    else:
        raise ValueError(f"Unknown graph kind: {kind}")

    _ensure_connected_edges(n, src, dst, rng)
    edges = np.unique(np.asarray([src, dst], dtype=np.int64), axis=1)
    weights = rng.uniform(0.8, 1.2, size=edges.shape[1]).astype(np.float32) * float(weight_scale)
    return GraphData(edge_index=edges, edge_weight=weights)


def load_edge_list(path: str, num_nodes: Optional[int] = None, directed: bool = True) -> GraphData:
    """Load a two-column whitespace/CSV edge list."""
    raw = np.loadtxt(path, delimiter="," if str(path).lower().endswith(".csv") else None, ndmin=2)
    if raw.shape[1] < 2:
        raise ValueError("Edge list must contain at least two columns")
    edges = raw[:, :2].astype(np.int64).T
    if edges.min() < 0:
        raise ValueError("Node IDs must be non-negative")
    n = int(edges.max()) + 1 if num_nodes is None else int(num_nodes)
    keep = edges[0] != edges[1]
    edges = edges[:, keep]
    if not directed:
        edges = np.concatenate([edges, edges[::-1]], axis=1)
    if edges.size and int(edges.max()) >= n:
        raise ValueError("Edge list contains a node outside num_nodes")
    return GraphData(edges, np.ones(edges.shape[1], dtype=np.float32))

