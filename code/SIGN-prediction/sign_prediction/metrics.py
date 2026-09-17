"""Prediction metrics and CSV helpers."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    truth = np.asarray(y, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    error = pred - truth
    return {
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
        "MAPE": float(np.mean(np.abs(error) / (np.abs(truth) + 1e-5)) * 100.0),
    }


def metric_rows(truth: np.ndarray, predictions: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for name, prediction in predictions.items():
        for horizon in range(len(truth)):
            row = {"model": name, "horizon": horizon + 1}
            row.update(metrics(truth[horizon], prediction[horizon]))
            rows.append(row)
    return rows


def write_csv(rows: list[dict], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["model", "horizon"]
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

