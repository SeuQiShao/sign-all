"""Convert an empirical edge-list file to a compact SIGN edge-only NPZ.

Empirical networks are downloaded by the user from the sources listed in the
manuscript's Data Availability statement. This utility is intentionally
opt-in: it normalizes node indexing and optionally expands an undirected
graph, but does not alter the downloaded source file.

Example:

    python Data_generation/prepare_empirical_networks.py \
        --source path/to/downloaded/catster.edges \
        --out Empirical_networks/processed/catster.npz \
        --one-indexed --undirected --expand-undirected
"""

from __future__ import annotations

import argparse
from array import array
import json
from pathlib import Path
import re

import numpy as np


def _numeric_row(line: str) -> list[str] | None:
    """Return tokens for a numeric edge row, accepting whitespace or CSV."""
    text = line.strip()
    if not text or text.startswith("#") or text.startswith("%"):
        return None
    tokens = [token.strip() for token in text.split(",")] if "," in text else text.split()
    if len(tokens) < 2:
        return None
    try:
        int(tokens[0]); int(tokens[1])
    except ValueError:
        return None
    return tokens


def load_edges(
    source: str | Path,
    *,
    num_nodes: int | None = None,
    one_indexed: bool | None = None,
    weighted: bool = False,
    undirected: bool = False,
    expand_undirected: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Parse an edge file and return normalized edges plus provenance metadata."""
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    src = array("q")
    dst = array("q")
    edge_weights = array("f")
    declared_nodes: int | None = None
    declared_edges: int | None = None
    minimum_id: int | None = None
    maximum_id: int | None = None
    saw_data = False

    with source_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.lower().startswith("%%matrixmarket"):
                continue
            if line.startswith("%") or line.startswith("#"):
                match = re.match(r"^\s*%\s*(\d+)\s+(\d+)\s+(\d+)\s*$", line)
                if match and declared_nodes is None:
                    declared_nodes = int(match.group(1))
                    declared_edges = int(match.group(3))
                continue
            tokens = _numeric_row(line)
            if tokens is None:
                continue
            values = [float(token) for token in tokens]
            # Standard MatrixMarket files put ``n n nnz`` on the first
            # non-comment line.  NetworkRepository exports also sometimes put
            # this declaration in a percent-comment, handled above.
            if not saw_data and len(values) == 3 and int(values[0]) == int(values[1]) and int(values[2]) >= 0 and declared_nodes is None:
                declared_nodes = int(values[0])
                declared_edges = int(values[2])
                continue
            saw_data = True
            left, right = int(values[0]), int(values[1])
            src.append(left)
            dst.append(right)
            if weighted and len(values) >= 3:
                edge_weights.append(float(values[2]))
            else:
                edge_weights.append(1.0)
            minimum_id = min(left, right) if minimum_id is None else min(minimum_id, left, right)
            maximum_id = max(left, right) if maximum_id is None else max(maximum_id, left, right)

    if not src:
        raise ValueError(f"no numeric edge rows found in {source_path}")
    if one_indexed is None:
        one_indexed = bool(minimum_id is not None and minimum_id >= 1)
    offset = 1 if one_indexed else 0
    edge_index = np.asarray([src, dst], dtype=np.int64) - offset
    edge_weight = np.asarray(edge_weights, dtype=np.float32)
    if edge_index.min() < 0:
        raise ValueError("one-index conversion produced a negative node id; pass --no-one-indexed if the source is zero-based")

    if expand_undirected:
        reverse_mask = edge_index[0] != edge_index[1]
        edge_index = np.concatenate((edge_index, edge_index[::-1, reverse_mask]), axis=1)
        edge_weight = np.concatenate((edge_weight, edge_weight[reverse_mask]))

    inferred_nodes = int(edge_index.max()) + 1
    total_nodes = int(num_nodes) if num_nodes is not None else (int(declared_nodes) if declared_nodes is not None else inferred_nodes)
    if total_nodes <= int(edge_index.max()):
        raise ValueError(f"num_nodes={total_nodes} is too small for maximum node id {int(edge_index.max())}")

    metadata = {
        "source": str(source_path),
        "source_format": source_path.suffix.lower().lstrip(".") or "text",
        "num_nodes": total_nodes,
        "num_edge_records": int(len(src)),
        "num_edges_written": int(edge_index.shape[1]),
        "declared_nodes": declared_nodes,
        "declared_edge_records": declared_edges,
        "source_one_indexed": bool(one_indexed),
        "weighted": bool(weighted),
        "undirected_semantics": bool(undirected),
        "expanded_undirected": bool(expand_undirected),
        "edge_convention": "edge_index[0] source -> edge_index[1] target",
        "generator": "SIGN E2V3 delivery / Data_generation/prepare_empirical_networks.py",
    }
    return edge_index, edge_weight, metadata


def prepare(
    source: str | Path,
    out: str | Path,
    *,
    num_nodes: int | None = None,
    one_indexed: bool | None = None,
    weighted: bool = False,
    undirected: bool = False,
    expand_undirected: bool = False,
) -> Path:
    """Parse an edge file and write ``edge_index``/``edge_weight`` NPZ data."""
    edge_index, edge_weight, metadata = load_edges(
        source, num_nodes=num_nodes, one_indexed=one_indexed, weighted=weighted,
        undirected=undirected, expand_undirected=expand_undirected,
    )
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, edge_index=edge_index, edge_weight=edge_weight, num_nodes=np.asarray(metadata["num_nodes"]), metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)))
    out_path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out_path} | nodes={metadata['num_nodes']} edges={edge_index.shape[1]} weighted={weighted} expanded_undirected={expand_undirected}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-nodes", type=int, default=None)
    parser.add_argument("--one-indexed", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--weighted", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--undirected", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--expand-undirected", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    prepare(args.source, args.out, num_nodes=args.num_nodes, one_indexed=args.one_indexed, weighted=args.weighted, undirected=args.undirected, expand_undirected=args.expand_undirected)


if __name__ == "__main__":
    main()
