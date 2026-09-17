"""Prediction data adapters for the E2V3 PyTorch/NPZ delivery formats."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data


@dataclass
class Trajectory:
    """A public prediction trajectory with time-major state storage.

    ``x`` and ``x_observed`` are ``[time, node, dimension]``.  This is the
    delivery format; the E2V3 discovery code is given the transposed PyG view
    ``[node, time, dimension]`` by :meth:`to_pyg`.
    """

    x: torch.Tensor
    x_observed: torch.Tensor
    t: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    metadata: dict
    x_dot: torch.Tensor | None = None

    @property
    def num_steps(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[1])

    @property
    def dimension(self) -> int:
        return int(self.x.shape[2])

    def to_pyg(self, observed: bool = True) -> Data:
        """Return the shape and fields expected by E2V3 Phase-I code."""
        values = self.x_observed if observed else self.x
        return Data(
            x=values.permute(1, 0, 2).contiguous(),
            edge_index=self.edge_index,
            edge_attr=self.edge_weight.reshape(-1, 1),
            t=self.t,
            batch=torch.zeros(self.num_nodes, dtype=torch.long, device=values.device),
        )


def _read_metadata(raw, source: Path) -> dict:
    value = raw["metadata"] if "metadata" in raw else None
    if value is not None:
        try:
            item = value.item() if np.asarray(value).ndim == 0 else value
            parsed = json.loads(str(item))
            if isinstance(parsed, dict):
                return parsed
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    sidecar = source.with_suffix(".json")
    if sidecar.exists():
        return json.loads(sidecar.read_text(encoding="utf-8"))
    return {"source": str(source)}


def load_npz(path: str | Path, device: str | torch.device = "cpu") -> Trajectory:
    """Load an NPZ trajectory without changing the public time-major layout."""
    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=True) as raw:
        required = {"x", "t", "edge_index"}
        missing = sorted(required.difference(raw.files))
        if missing:
            raise ValueError(f"Dataset is missing required arrays: {missing}")
        x = np.asarray(raw["x"], dtype=np.float32)
        x_observed = np.asarray(raw["x_observed"] if "x_observed" in raw else x, dtype=np.float32)
        t = np.asarray(raw["t"], dtype=np.float32).reshape(-1)
        edge_index = np.asarray(raw["edge_index"], dtype=np.int64)
        edge_weight = np.asarray(
            raw["edge_weight"] if "edge_weight" in raw else np.ones(edge_index.shape[1]),
            dtype=np.float32,
        ).reshape(-1)
        x_dot = np.asarray(raw["x_dot"], dtype=np.float32) if "x_dot" in raw else None
        metadata = _read_metadata(raw, source)

    if x.ndim != 3:
        raise ValueError(f"x must have shape [time,node,dimension], got {x.shape}")
    if x_observed.shape != x.shape:
        raise ValueError("x_observed must have the same shape as x")
    if len(t) != x.shape[0] or len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing and align with x")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.shape[1] != len(edge_weight):
        raise ValueError("edge_index must be [2,E] and edge_weight must contain E values")
    if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= x.shape[1]):
        raise ValueError("edge_index contains a node id outside x")
    if x_dot is not None and x_dot.shape != x.shape:
        raise ValueError("x_dot must have the same shape as x when supplied")

    return Trajectory(
        x=torch.as_tensor(x, device=device),
        x_observed=torch.as_tensor(x_observed, device=device),
        t=torch.as_tensor(t, device=device),
        edge_index=torch.as_tensor(edge_index, dtype=torch.long, device=device),
        edge_weight=torch.as_tensor(edge_weight, device=device),
        metadata=metadata,
        x_dot=torch.as_tensor(x_dot, device=device) if x_dot is not None else None,
    )
