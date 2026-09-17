"""Convert a delivery NPZ trajectory to the raw PyG layout expected by E2V3.

The converter keeps data generation in ``NC/Data_generation`` while allowing
the canonical ``NC/SIGN-phase/trainer.py`` to consume robustness cases.  State
arrays are transposed from ``[time,node,dimension]`` to ``[node,time,dimension]``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data


def convert(
    source: str | Path,
    out_root: str | Path,
    edge_key: str = "supplied_edge_index",
    observed: bool = True,
) -> Path:
    source_path = Path(source).expanduser()
    out = Path(out_root).expanduser()
    with np.load(source_path, allow_pickle=True) as raw:
        if "x" not in raw or "t" not in raw:
            raise ValueError("NPZ must contain x and t")
        state_source = "x_observed" if observed and "x_observed" in raw else "x"
        x = np.asarray(raw[state_source], dtype=np.float32)
        if edge_key not in raw:
            if edge_key == "supplied_edge_index" and "edge_index" in raw:
                edge_key = "edge_index"
            else:
                raise ValueError(f"NPZ does not contain {edge_key}")
        edge_index = np.asarray(raw[edge_key], dtype=np.int64)
        edge_weight_key = "supplied_edge_weight" if edge_key == "supplied_edge_index" else "edge_weight"
        edge_weight = np.asarray(
            raw[edge_weight_key] if edge_weight_key in raw else np.ones(edge_index.shape[1]),
            dtype=np.float32,
        ).reshape(-1)
        t = np.asarray(raw["t"], dtype=np.float32).reshape(-1)
        metadata_value = raw["metadata"].item() if "metadata" in raw else "{}"
        metadata = json.loads(str(metadata_value))

    if x.ndim != 3 or len(t) != x.shape[0]:
        raise ValueError("x must be [time,node,dimension] and t must align with x")
    if edge_index.shape != (2, len(edge_weight)):
        raise ValueError("edge_index and edge_weight are inconsistent")
    if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= x.shape[1]):
        raise ValueError("selected edge_index contains an invalid node id")
    if len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing")

    raw_dir = out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    data = Data(
        x=torch.from_numpy(np.transpose(x, (1, 0, 2)).copy()),
        edge_index=torch.from_numpy(edge_index.copy()),
        edge_attr=torch.from_numpy(edge_weight.reshape(-1, 1).copy()),
        t=torch.from_numpy(t.copy()),
        para=metadata,
    )
    torch.save([data], raw_dir / "data_1.pt")
    manifest = {
        "source_npz": str(source_path.resolve()),
        "state_source": state_source,
        "edge_source": edge_key,
        "shape_time_major": list(map(int, x.shape)),
        "shape_pyg": [int(x.shape[1]), int(x.shape[0]), int(x.shape[2])],
        "edge_count": int(edge_index.shape[1]),
        "canonical_e2v3_layout": True,
    }
    (out / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {raw_dir / 'data_1.pt'} | x={tuple(data.x.shape)} edges={edge_index.shape[1]}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--edge-key", choices=["edge_index", "supplied_edge_index"], default="supplied_edge_index")
    p.add_argument("--observed", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    convert(args.source, args.out_root, args.edge_key, args.observed)


if __name__ == "__main__":
    main()
