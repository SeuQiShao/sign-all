"""Export the small, public plotting-data package used by Fig. 1.

The original Rössler NPZ files are large experiment artifacts.  This exporter
keeps the exact values needed by the figure: the five displayed
trajectory nodes, all node-level MSE values, the coefficient bars, and the
NetworkX/layout tables.  It is deterministic and is only needed when the
submission package is rebuilt from the original experiment outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


TRACE_NODE = 121
CONTEXT_NODES = (230, 112, 378, 934)
SEED_INDEX = 2


def _network_tables() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    edges = [
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
        (1, 6), (1, 7), (2, 8), (2, 9),
        (3, 9), (3, 10), (3, 11), (4, 11), (4, 12),
        (5, 7), (5, 13), (6, 7), (8, 9), (9, 10),
        (10, 11), (11, 12), (12, 13), (13, 7),
        (6, 14), (8, 15), (10, 16), (11, 17), (13, 18),
    ]
    pos = {
        0: (0.00, 0.00),
        1: (-0.98, 0.76), 2: (0.12, 1.24), 3: (1.12, 0.44),
        4: (0.67, -1.06), 5: (-0.78, -1.00),
        6: (-2.02, 1.26), 7: (-1.94, -0.18),
        8: (-0.82, 2.08), 9: (0.70, 2.08),
        10: (2.05, 1.18), 11: (2.08, -0.50),
        12: (0.92, -2.02), 13: (-1.10, -2.02),
        14: (-2.94, 1.92), 15: (-1.18, 3.02), 16: (2.82, 1.88),
        17: (3.05, -1.10), 18: (-1.88, -2.98),
    }
    neighbors = {1, 2, 3, 4, 5}
    continuation = {14, 15, 16, 17, 18}
    nodes = [
        {
            "node": node,
            "x": xy[0],
            "y": xy[1],
            "role": "target" if node == 0 else "neighbor" if node in neighbors else "other",
            "continuation": int(node in continuation),
        }
        for node, xy in pos.items()
    ]
    edge_rows = [
        {"source": source, "target": target, "continuation": int(source in continuation or target in continuation)}
        for source, target in edges
    ]
    return nodes, edge_rows


def _library_terms() -> list[dict[str, object]]:
    self_terms = [
        r"$1$", r"$xᵢ$", r"$yᵢ$", r"$zᵢ$", r"$xᵢ²$",
        r"$yᵢ²$", r"$zᵢ²$", r"$xᵢ yᵢ$", r"$xᵢ zᵢ$", r"$yᵢ zᵢ$",
        r"$xᵢ³$", r"$yᵢ³$", r"$zᵢ³$", r"$xᵢ² yᵢ$", r"$xᵢ yᵢ²$",
        r"$\sin(xᵢ)$", r"$\sin(yᵢ)$", r"$\cos(zᵢ)$", r"$e^{xᵢ}$", "…",
    ]
    coupling_terms = [
        r"$xᵢ\!-\!xⱼ$", r"$yᵢ\!-\!yⱼ$", r"$zᵢ\!-\!zⱼ$",
        r"$(xᵢ\!-\!xⱼ)²$", r"$(yᵢ\!-\!yⱼ)²$",
        r"$(zᵢ\!-\!zⱼ)²$", r"$\sin(xᵢ\!-\!xⱼ)$",
        r"$\sin(yᵢ\!-\!yⱼ)$", r"$\sin(zᵢ\!-\!zⱼ)$",
        r"$e^{xⱼ\!-\!xᵢ}$", r"$e^{yⱼ\!-\!yᵢ}$", r"$e^{zⱼ\!-\!zᵢ}$",
        r"$\cos(xᵢ\!-\!xⱼ)$", r"$\cos(yᵢ\!-\!yⱼ)$", r"$\cos(zᵢ\!-\!zⱼ)$",
        r"$e^{(xⱼ\!-\!xᵢ)²}$", r"$e^{(yⱼ\!-\!yᵢ)²}$",
        r"$e^{(zⱼ\!-\!zᵢ)²}$", r"$-\sin(xᵢ\!-\!xⱼ)$", "…",
    ]
    rows: list[dict[str, object]] = []
    for library, terms in (("self", self_terms), ("coupling", coupling_terms)):
        rows.extend({"library": library, "index": i, "expression": term} for i, term in enumerate(terms))
    return rows


def _support_tables() -> list[dict[str, object]]:
    self_base = [1, 0, 0, 1, 0, 1, 0, 0, 1, 0]
    coupling_base = [0, 1, 0, 0, 1, 0, 0, 1, 0, 1]
    changes = {
        0: ((8, 1), (9, 0)),
        1: ((2, 1), (7, 0)),
        2: ((8, 1), (2, 1)),
        3: ((6, 1), (0, 1)),
    }
    rows: list[dict[str, object]] = []
    for row in range(4):
        self_pattern = self_base.copy()
        coupling_pattern = coupling_base.copy()
        self_change, coupling_change = changes[row]
        self_pattern[self_change[0]] = self_change[1]
        coupling_pattern[coupling_change[0]] = coupling_change[1]
        for library, pattern in (("self", self_pattern), ("coupling", coupling_pattern)):
            for slot, active in enumerate(pattern):
                rows.append({"pattern": f"local_{row}", "library": library, "slot": slot, "active": active})
    core_rows = [
        (self_base, coupling_base),
        (self_base, [0, 1, 0, 0, 1, 0, 1, 1, 0, 1]),
        (self_base, coupling_base),
    ]
    for row, (self_pattern, coupling_pattern) in enumerate(core_rows):
        for library, pattern in (("self", self_pattern), ("coupling", coupling_pattern)):
            for slot, active in enumerate(pattern):
                rows.append({"pattern": f"core_{row}", "library": library, "slot": slot, "active": active})
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_plot_data(data_dir: Path, coefficient_json: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    observed_path = data_dir / "rossler_clean_dbscan_rollout_1_800.npz"
    forecast_path = data_dir / "rossler_clean_dbscan_rollout_800_1000.npz"
    with np.load(observed_path, allow_pickle=False) as observed:
        observed_true = np.asarray(observed["true_trajectory"][SEED_INDEX], dtype=np.float64)
    with np.load(forecast_path, allow_pickle=False) as forecast:
        forecast_true = np.asarray(forecast["true_trajectory"][SEED_INDEX], dtype=np.float64)
        forecast_pred = np.asarray(forecast["predicted_trajectory"][SEED_INDEX], dtype=np.float64)
        forecast_error = np.asarray(forecast["error"][SEED_INDEX], dtype=np.float64)

    node_ids = (TRACE_NODE, *CONTEXT_NODES)
    trajectory_rows: list[dict[str, object]] = []
    for node in node_ids:
        for time in range(1001):
            row: dict[str, object] = {"time": time, "node": node}
            for dim, name in enumerate(("x", "y", "z")):
                row[f"{name}_observed"] = observed_true[time, node, dim] if time <= 800 else ""
                row[f"{name}_forecast_true"] = forecast_true[time - 800, node, dim] if time >= 800 else ""
                row[f"{name}_forecast_pred"] = forecast_pred[time - 800, node, dim] if time >= 800 else ""
                row[f"{name}_forecast_error"] = forecast_error[time - 800, node, dim] if time >= 800 else ""
            trajectory_rows.append(row)
    _write_csv(
        output_dir / "trajectory.csv",
        trajectory_rows,
        ["time", "node", "x_observed", "y_observed", "z_observed", "x_forecast_true", "y_forecast_true", "z_forecast_true", "x_forecast_pred", "y_forecast_pred", "z_forecast_pred", "x_forecast_error", "y_forecast_error", "z_forecast_error"],
    )

    node_mse = np.mean(np.square(forecast_error), axis=0)
    _write_csv(
        output_dir / "node_mse.csv",
        [{"node": i, "x_mse": values[0], "y_mse": values[1], "z_mse": values[2]} for i, values in enumerate(node_mse)],
        ["node", "x_mse", "y_mse", "z_mse"],
    )

    payload = json.loads(coefficient_json.read_text(encoding="utf-8"))
    series_specs = [
        ("BA–1k", payload["rows"][0]),
        ("BA–100k", payload["rows"][1]),
        ("Human", payload["rows"][2]),
        ("Fly", payload["rows"][3]),
        ("Truth", payload["rows"][4]),
    ]
    terms = {
        "x": ["yᵢ", "zᵢ", "xⱼ−xᵢ"],
        "y": ["xᵢ", "yᵢ"],
        "z": ["1", "zᵢ", "xᵢzᵢ"],
    }
    rows: list[dict[str, object]] = []
    for series, source_row in series_specs:
        values = source_row[5:13]
        for dim, offset in (("x", 0), ("y", 3), ("z", 5)):
            for slot, term in enumerate(terms[dim]):
                rows.append({"series": series, "dimension": dim, "term": term, "slot": slot, "value": values[offset + slot]})
    _write_csv(output_dir / "coefficients.csv", rows, ["series", "dimension", "term", "slot", "value"])

    nodes, edges = _network_tables()
    _write_csv(output_dir / "network_nodes.csv", nodes, ["node", "x", "y", "role", "continuation"])
    _write_csv(output_dir / "network_edges.csv", edges, ["source", "target", "continuation"])
    _write_csv(output_dir / "library_terms.csv", _library_terms(), ["library", "index", "expression"])
    _write_csv(output_dir / "support_patterns.csv", _support_tables(), ["pattern", "library", "slot", "active"])

    (output_dir / "plot_data_manifest.json").write_text(
        json.dumps(
            {"seed_index": SEED_INDEX, "trace_node": TRACE_NODE, "context_nodes": list(CONTEXT_NODES), "trajectory_nodes": list(node_ids), "node_mse_nodes": 1000, "source": "Rössler DBSCAN rollout NPZ files supplied with the manuscript experiments"},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--coefficients", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "plot_data")
    args = parser.parse_args()
    export_plot_data(args.data_dir, args.coefficients, args.output_dir)
    print(f"Exported Fig. 1 plot data to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
