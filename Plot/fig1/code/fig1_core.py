"""Shared data model and CSV loader for the Fig. 1 submission package."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class RosslerData:
    """Exact plotting arrays retained in ``plot_data``.

    Trajectories are stored only for the representative node and the four
    context nodes used in panel e; the node-level MSE table retains all 1,000
    nodes.  ``node_ids`` maps the manuscript node labels to trajectory columns.
    """

    node_ids: tuple[int, ...]
    observed_true: np.ndarray
    forecast_true: np.ndarray
    forecast_pred: np.ndarray
    forecast_error: np.ndarray
    node_mse: np.ndarray
    learned_coefficients: dict[str, tuple[list[str], np.ndarray]]
    true_coefficients: dict[str, tuple[list[str], np.ndarray]]

    def node_index(self, node: int) -> int:
        try:
            return self.node_ids.index(int(node))
        except ValueError as exc:
            raise KeyError(
                f"Node {node} is not present in trajectory.csv; available nodes are {self.node_ids}"
            ) from exc

    def node_series(self, node: int, dim: int, *, predicted: bool = False, truth: bool = False) -> np.ndarray:
        index = self.node_index(node)
        if predicted:
            return np.concatenate((self.observed_true[:, index, dim], self.forecast_pred[1:, index, dim]))
        if truth:
            return np.concatenate((self.observed_true[:, index, dim], self.forecast_true[1:, index, dim]))
        return self.observed_true[:, index, dim]


def _float(value: str) -> float:
    if value == "":
        return float("nan")
    return float(value)


def load_plot_data(plot_data_dir: Path, seed_index: int = 2) -> RosslerData:
    """Load all arrays needed by panels a and e from public CSVs."""
    trajectory_path = plot_data_dir / "trajectory.csv"
    mse_path = plot_data_dir / "node_mse.csv"
    coefficient_path = plot_data_dir / "coefficients.csv"
    if not trajectory_path.exists() or not mse_path.exists() or not coefficient_path.exists():
        raise FileNotFoundError("plot_data must contain trajectory.csv, node_mse.csv, and coefficients.csv")

    rows = list(csv.DictReader(trajectory_path.open(encoding="utf-8", newline="")))
    node_ids = tuple(dict.fromkeys(int(row["node"]) for row in rows))
    if not node_ids:
        raise ValueError("trajectory.csv contains no nodes")
    node_index = {node: index for index, node in enumerate(node_ids)}
    observed_true = np.full((801, len(node_ids), 3), np.nan, dtype=np.float64)
    forecast_true = np.full((201, len(node_ids), 3), np.nan, dtype=np.float64)
    forecast_pred = np.full((201, len(node_ids), 3), np.nan, dtype=np.float64)
    forecast_error = np.full((201, len(node_ids), 3), np.nan, dtype=np.float64)
    dims = ("x", "y", "z")
    for row in rows:
        node = int(row["node"])
        time = int(row["time"])
        index = node_index[node]
        if time <= 800:
            for dim, name in enumerate(dims):
                observed_true[time, index, dim] = _float(row[f"{name}_observed"])
        if time >= 800:
            for dim, name in enumerate(dims):
                forecast_true[time - 800, index, dim] = _float(row[f"{name}_forecast_true"])
                forecast_pred[time - 800, index, dim] = _float(row[f"{name}_forecast_pred"])
                forecast_error[time - 800, index, dim] = _float(row[f"{name}_forecast_error"])
    if not np.all(np.isfinite(observed_true)) or not np.all(np.isfinite(forecast_true)):
        raise ValueError("trajectory.csv has missing or non-finite trajectory values")
    if not np.all(np.isfinite(forecast_pred)) or not np.all(np.isfinite(forecast_error)):
        raise ValueError("trajectory.csv has missing or non-finite forecast values")

    with mse_path.open(encoding="utf-8", newline="") as handle:
        mse_rows = list(csv.DictReader(handle))
    node_mse = np.asarray(
        [[float(row["x_mse"]), float(row["y_mse"]), float(row["z_mse"])] for row in mse_rows],
        dtype=np.float64,
    )
    if node_mse.shape != (1000, 3) or not np.all(np.isfinite(node_mse)) or np.any(node_mse <= 0):
        raise ValueError("node_mse.csv must contain 1,000 finite positive rows")

    terms = {"x": ["yᵢ", "zᵢ", "xⱼ−xᵢ"], "y": ["xᵢ", "yᵢ"], "z": ["1", "zᵢ", "xᵢzᵢ"]}
    values: dict[str, dict[str, list[float]]] = {}
    with coefficient_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            values.setdefault(row["series"], {}).setdefault(row["dimension"], []).append(float(row["value"]))
    ordered = ["BA–1k", "BA–100k", "Human", "Fly", "Truth"]
    learned = values[ordered[0]]
    truth = values[ordered[-1]]
    learned_coefficients = {dim: (terms[dim], np.asarray(learned[dim], dtype=np.float64)) for dim in terms}
    true_coefficients = {dim: (terms[dim], np.asarray(truth[dim], dtype=np.float64)) for dim in terms}
    return RosslerData(
        node_ids=node_ids,
        observed_true=observed_true,
        forecast_true=forecast_true,
        forecast_pred=forecast_pred,
        forecast_error=forecast_error,
        node_mse=node_mse,
        learned_coefficients=learned_coefficients,
        true_coefficients=true_coefficients,
    )


def combined_trajectory(data: RosslerData, node: int, dim: int, predicted: bool) -> np.ndarray:
    return data.node_series(node, dim, predicted=predicted, truth=not predicted)
