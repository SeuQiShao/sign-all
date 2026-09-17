"""Network-coupled synthetic dynamics used by the SIGN delivery package.

Every right-hand side has the same signature::

    rhs(t, x, edge_index, edge_weight) -> dx_dt

The edge list is sparse and is aggregated at the target node, so the same
interface works for small examples and large networks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


RHS = Callable[[float, np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def _split_edges(edge_index: np.ndarray, edge_weight: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    return edges[0], edges[1], np.asarray(edge_weight, dtype=np.float64).reshape(-1)


def _aggregate_messages(num_nodes: int, target: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.zeros((num_nodes, values.shape[1]), dtype=np.float64)
    np.add.at(out, target, values)
    return out


def _scalar_message(x: np.ndarray, src: np.ndarray, dst: np.ndarray, weight: np.ndarray, fn) -> np.ndarray:
    return _aggregate_messages(x.shape[0], dst, weight[:, None] * fn(x[dst], x[src]))


def kuramoto(t, x, edge_index, edge_weight):
    src, dst, w = _split_edges(edge_index, edge_weight)
    out = np.full_like(x, 0.30, dtype=np.float64)
    msg = w * 0.15 * np.sin(x[src, 0] - x[dst, 0])
    out[:, 0] += _aggregate_messages(x.shape[0], dst, msg[:, None])[:, 0]
    return out


def heat_diffusion(t, x, edge_index, edge_weight):
    src, dst, w = _split_edges(edge_index, edge_weight)
    msg = -0.30 * w * (x[dst, 0] - x[src, 0])
    out = _aggregate_messages(x.shape[0], dst, msg[:, None])
    return out


def sis(t, x, edge_index, edge_weight):
    src, dst, w = _split_edges(edge_index, edge_weight)
    msg = w * (x[src, 0] - x[dst, 0] * x[src, 0])
    out = -x.copy()
    out[:, 0] += _aggregate_messages(x.shape[0], dst, msg[:, None])[:, 0]
    return out


def gene(t, x, edge_index, edge_weight):
    src, dst, w = _split_edges(edge_index, edge_weight)
    xj = np.maximum(x[src, 0], 0.0)
    msg = w * (xj / (1.0 + xj))
    out = -x.copy()
    out[:, 0] += _aggregate_messages(x.shape[0], dst, msg[:, None])[:, 0]
    return out


def mutualistic(t, x, edge_index, edge_weight):
    src, dst, w = _split_edges(edge_index, edge_weight)
    xi, xj = x[dst, 0], x[src, 0]
    msg = w * xi * xj / (5.0 + 0.9 * xi + 0.1 * xj)
    out = (x * (1.0 - x / 5.0) * (x / 1.0 - 1.0)).astype(np.float64)
    out[:, 0] += _aggregate_messages(x.shape[0], dst, msg[:, None])[:, 0]
    return out


def fitzhugh_nagumo(t, x, edge_index, edge_weight):
    src, dst, w = _split_edges(edge_index, edge_weight)
    coupling = _aggregate_messages(x.shape[0], dst, (0.1 * w[:, None] * (x[src, 0][:, None] - x[dst, 0][:, None])))
    out = np.empty_like(x, dtype=np.float64)
    v, r = x[:, 0], x[:, 1]
    out[:, 0] = v - v**3 - r + 1.0 + coupling[:, 0]
    out[:, 1] = 0.5 * (0.5 + v - 0.3 * r)
    return out


def rossler(t, x, edge_index, edge_weight):
    src, dst, w = _split_edges(edge_index, edge_weight)
    coupling = _aggregate_messages(x.shape[0], dst, 0.1 * w[:, None] * (x[src, 0][:, None] - x[dst, 0][:, None]))
    out = np.empty_like(x, dtype=np.float64)
    out[:, 0] = -x[:, 1] - x[:, 2] + coupling[:, 0]
    out[:, 1] = x[:, 0] + 0.2 * x[:, 1]
    out[:, 2] = 0.2 + x[:, 2] * (x[:, 0] - 6.0)
    return out


def hindmarsh_rose(t, x, edge_index, edge_weight):
    src, dst, w = _split_edges(edge_index, edge_weight)
    coupling = _aggregate_messages(x.shape[0], dst, 0.5 * w[:, None] * (x[src, 0][:, None] - x[dst, 0][:, None]))
    out = np.empty_like(x, dtype=np.float64)
    v, y, z = x[:, 0], x[:, 1], x[:, 2]
    out[:, 0] = y - v**3 + 3.0 * v**2 - z + 3.24 + coupling[:, 0]
    out[:, 1] = 1.0 - 5.0 * v**2 - y
    out[:, 2] = 0.01 * (4.0 * (v + 1.6) - z)
    return out


def chua(t, x, edge_index, edge_weight):
    src, dst, w = _split_edges(edge_index, edge_weight)
    coupling = _aggregate_messages(x.shape[0], dst, 0.1 * w[:, None] * (x[src, 0][:, None] - x[dst, 0][:, None]))
    v = x[:, 0]
    h = -0.714 * v + 0.5 * (-1.143 + 0.714) * (np.abs(v + 1.0) - np.abs(v - 1.0))
    out = np.empty_like(x, dtype=np.float64)
    out[:, 0] = 9.0 * (x[:, 1] - v - h) + coupling[:, 0]
    out[:, 1] = v - x[:, 1] + x[:, 2]
    out[:, 2] = -14.286 * x[:, 1]
    return out


@dataclass(frozen=True)
class DynamicsSpec:
    name: str
    dimension: int
    rhs: RHS
    default_scale: float


_SYSTEMS = {
    "kuramoto": DynamicsSpec("kuramoto", 1, kuramoto, 0.5),
    "heat": DynamicsSpec("heat", 1, heat_diffusion, 0.5),
    "sis": DynamicsSpec("sis", 1, sis, 0.5),
    "gene": DynamicsSpec("gene", 1, gene, 0.5),
    # The manuscript calls this benchmark Michaelis--Menten (MM); ``gene``
    # remains the historical CLI name used by the original delivery.
    "michaelis_menten": DynamicsSpec("michaelis_menten", 1, gene, 0.5),
    "mutual": DynamicsSpec("mutual", 1, mutualistic, 0.2),
    "mutualistic": DynamicsSpec("mutualistic", 1, mutualistic, 0.2),
    "fhn": DynamicsSpec("fhn", 2, fitzhugh_nagumo, 0.8),
    "rossler": DynamicsSpec("rossler", 3, rossler, 0.4),
    "hr": DynamicsSpec("hr", 3, hindmarsh_rose, 0.4),
    "chua": DynamicsSpec("chua", 3, chua, 0.3),
}


def available_systems() -> list[str]:
    return ["kuramoto", "heat", "sis", "gene", "michaelis_menten", "mutual", "fhn", "rossler", "hr", "chua"]


def system_dimension(name: str) -> int:
    key = name.lower()
    if key not in _SYSTEMS:
        raise ValueError(f"Unknown dynamics '{name}'. Available: {available_systems()}")
    return _SYSTEMS[key].dimension


def make_dynamics(name: str) -> DynamicsSpec:
    key = name.lower()
    if key not in _SYSTEMS:
        raise ValueError(f"Unknown dynamics '{name}'. Available: {available_systems()}")
    return _SYSTEMS[key]
