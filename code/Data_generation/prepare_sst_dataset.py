"""Convert an SST array or PyG trajectory to SIGN NPZ format.

Example:

    python Data_generation/prepare_sst_dataset.py \
        --source path/to/enso_sstg.npy \
        --out SIGN-data/prediction/sst_enso.npz

The large ENSO/SST delivery is a PyTorch-Geometric ``Data`` object saved in a
``.pt`` file.  Its source layout is ``x=[node,time,dimension]``; this script
normalizes it to SIGN's ``x=[time,node,dimension]`` convention and preserves
the source ``edge_index`` graph.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


def grid_edges(height: int, width: int, directed: bool = True) -> np.ndarray:
    """Build a sparse 4-neighbour grid graph in row-major node order."""
    src, dst = [], []
    for row in range(height):
        for col in range(width):
            node = row * width + col
            for dr, dc in ((0, 1), (1, 0)):
                nr, nc = row + dr, col + dc
                if nr < height and nc < width:
                    other = nr * width + nc
                    src.append(node); dst.append(other)
                    if not directed:
                        src.append(other); dst.append(node)
    return np.asarray([src, dst], dtype=np.int64)


def _numpy(value) -> np.ndarray:
    """Convert tensors and array-like values without retaining a computation graph."""
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def _load_sst_source(source_path: Path, lat_path: str | Path | None, lon_path: str | Path | None, directed: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load either a dense SST array or the delivered PyG graph trajectory."""
    if source_path.suffix.lower() == ".pt":
        loaded = torch.load(source_path, map_location="cpu", weights_only=False)
        if isinstance(loaded, (list, tuple)):
            if not loaded:
                raise ValueError("PyG SST source is an empty list")
            loaded = loaded[0]
        if not hasattr(loaded, "x") or not hasattr(loaded, "edge_index"):
            raise ValueError("PyG SST source must contain x and edge_index")

        raw_x = _numpy(loaded.x)
        if raw_x.ndim == 2:
            raw_x = raw_x[..., None]
        if raw_x.ndim != 3:
            raise ValueError(f"PyG x must have shape [N,T,D] or [T,N,D], got {raw_x.shape}")
        # The delivered 72k source is [N,T,D].  The public SIGN format is [T,N,D].
        x = np.transpose(raw_x, (1, 0, 2)) if raw_x.shape[0] > raw_x.shape[1] else raw_x
        edge_index = _numpy(loaded.edge_index).astype(np.int64, copy=False)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must have shape [2,E], got {edge_index.shape}")
        edge_attr = getattr(loaded, "edge_attr", None)
        weights = np.ones(edge_index.shape[1], dtype=np.float32) if edge_attr is None else _numpy(edge_attr).reshape(-1).astype(np.float32, copy=False)
        if len(weights) != edge_index.shape[1]:
            raise ValueError("edge_attr must contain one value per source edge")
        time_value = getattr(loaded, "t", None)
        t = np.arange(x.shape[0], dtype=np.float32) if time_value is None else _numpy(time_value).reshape(-1).astype(np.float32, copy=False)
        if len(t) != x.shape[0]:
            raise ValueError(f"source t has length {len(t)} but x has {x.shape[0]} time steps")

        extra: dict = {"source_type": "torch_geometric.data.Data", "graph": "source edge_index", "directed": bool(directed), "edge_index_source": "PyG Data.edge_index"}
        for field in ("xy", "lat", "lon"):
            value = getattr(loaded, field, None)
            if value is not None:
                extra[field] = _numpy(value)
        return x.astype(np.float32, copy=False), edge_index, weights, {"t": t, **extra}

    raw = np.load(source_path, mmap_mode="r")
    if raw.ndim == 3:
        time_steps, height, width = raw.shape
        x = np.asarray(raw, dtype=np.float32).reshape(time_steps, height * width, 1)
    elif raw.ndim == 2:
        time_steps, nodes = raw.shape
        height, width = 1, nodes
        x = np.asarray(raw, dtype=np.float32)[..., None]
    else:
        raise ValueError("SST source must have shape [T,H,W] or [T,N], or be a PyG .pt file")
    full_edges = grid_edges(height, width, directed=directed)
    lat = np.load(lat_path) if lat_path else np.arange(height, dtype=np.float32)
    lon = np.load(lon_path) if lon_path else np.arange(width, dtype=np.float32)
    return x, full_edges, np.ones(full_edges.shape[1], dtype=np.float32), {"t": np.arange(time_steps, dtype=np.float32), "source_type": "dense array", "height": height, "width": width, "lat": np.asarray(lat), "lon": np.asarray(lon), "graph": "4-neighbour grid", "directed": bool(directed)}


def prepare(source: str | Path, out: str | Path, lat_path: str | Path | None = None, lon_path: str | Path | None = None, directed: bool = True, invalid_threshold: float = 1e4) -> Path:
    source_path = Path(source)
    x, full_edges, weights, source_info = _load_sst_source(source_path, lat_path, lon_path, directed)
    time_steps, original_nodes, dimension = x.shape
    invalid = (~np.isfinite(x)).any(axis=(0, 2)) | (np.abs(x) > float(invalid_threshold)).any(axis=(0, 2))
    keep = ~invalid
    if not np.any(keep):
        raise ValueError("SST source has no valid nodes")
    if full_edges.size and (full_edges.min() < 0 or full_edges.max() >= original_nodes):
        raise ValueError("source edge_index contains a node id outside x")
    node_map = np.full(original_nodes, -1, dtype=np.int64)
    node_map[keep] = np.arange(int(keep.sum()), dtype=np.int64)
    edge_keep = keep[full_edges[0]] & keep[full_edges[1]]
    edges = node_map[full_edges[:, edge_keep]]
    x = x[:, keep]
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)[edge_keep]
    t = np.asarray(source_info.pop("t"), dtype=np.float32)
    if len(t) != time_steps or np.any(np.diff(t) <= 0):
        raise ValueError("source time coordinate must be strictly increasing and align with x")

    height = int(source_info.pop("height", 0))
    width = int(source_info.pop("width", 0))
    lat = source_info.pop("lat", np.asarray([], dtype=np.float32))
    lon = source_info.pop("lon", np.asarray([], dtype=np.float32))
    xy = source_info.pop("xy", np.asarray([], dtype=np.float32))
    node_lat = np.asarray([], dtype=np.float32)
    node_lon = np.asarray([], dtype=np.float32)
    if height and width:
        node_lat = np.repeat(np.asarray(lat).reshape(-1), width)[: height * width][keep]
        node_lon = np.tile(np.asarray(lon).reshape(-1), height)[: height * width][keep]
    elif xy.size and len(xy) == original_nodes:
        xy = np.asarray(xy, dtype=np.float32)[keep]
    else:
        xy = np.asarray([], dtype=np.float32)
    metadata = {
        "system": "sst",
        "source": "user-supplied torch_geometric.data.Data object",
        "source_file": source_path.name,
        "upstream_dataset": "SSTG (Cao et al., Earth System Science Data 13, 2111–2134, 2021)",
        "shape_source": [int(time_steps), int(original_nodes), int(dimension)],
        "dimension": int(dimension),
        "num_nodes": int(x.shape[1]),
        "original_num_nodes": int(original_nodes),
        "dropped_invalid_nodes": int(invalid.sum()),
        "num_steps": int(time_steps),
        "time_coordinate": "source Data.t when available; otherwise integer sample index",
        "graph": source_info.pop("graph"),
        "directed": bool(directed),
        "edge_convention": "edge_index[0] source -> edge_index[1] target",
        "generator": "Data_generation/prepare_sst_dataset.py",
    }
    metadata.update({key: value for key, value in source_info.items() if np.isscalar(value) or isinstance(value, str)})
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path, x=x, x_observed=x, t=t, edge_index=edges, edge_weight=weights,
        lat=np.asarray(lat), lon=np.asarray(lon), node_lat=node_lat, node_lon=node_lon, xy=xy, valid_node_mask=keep,
        system=np.asarray("sst"), dimension=np.asarray(dimension),
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    out_path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out_path} | shape={x.shape} edges={edges.shape[1]}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--out", default="SIGN-data/prediction/sst_enso.npz")
    p.add_argument("--lat", default=None)
    p.add_argument("--lon", default=None)
    p.add_argument("--directed", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--invalid-threshold", type=float, default=1e4, help="Drop nodes containing non-finite or absolute values above this sentinel threshold")
    args = p.parse_args()
    prepare(args.source, args.out, args.lat, args.lon, args.directed, args.invalid_threshold)


if __name__ == "__main__":
    main()
