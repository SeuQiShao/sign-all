#!/usr/bin/env python3
"""Build raw/mean-std robustness tables and the four requested figures."""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


TRUTH = {
    "Kuramoto": [
        {"1": 0.3, "sin(x_j-x_i)": 0.15},
    ],
    "FHN": [
        {"1": 1.0, "x1^1": 1.0, "x2^1": -1.0, "x1^3": -1.0, "(x_j - x_i)^1": -0.1},
        {"1": 0.5, "x1^1": 1.0, "x2^1": -0.3},
    ],
    "Rossler": [
        {"x2^1": -1.0, "x3^1": -1.0, "(x_j - x_i)^1": 0.1},
        {"x1^1": 1.0, "x2^1": 0.2},
        {"1": 0.2, "x3^1": -6.0, "(x1x3)^1": 1.0},
    ],
}


def finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def coefficient_metrics(system, dim, phase2_root, metric):
    coef_path = phase2_root / f"dim{dim}" / "coefficients_pruned.npz"
    if not coef_path.exists():
        return np.nan, np.nan
    artifact = np.load(coef_path, allow_pickle=True)
    basis_artifact = np.load(coef_path.with_name("coefficients_raw.npz"), allow_pickle=True)
    truth = TRUTH[system][dim]
    predicted = {}
    for kind, key in (("f", "f_coef"), ("c", "c_coef")):
        basis_key = f"{kind}_basis"
        for name, value in zip(basis_artifact[basis_key].tolist(), artifact[key].reshape(-1)):
            predicted[str(name)] = float(value)
    names = set(truth) | set(predicted)
    smape = []
    for name in names:
        target = float(truth.get(name, 0.0))
        estimate = float(predicted.get(name, 0.0))
        smape.append(2.0 * abs(estimate - target) / (abs(estimate) + abs(target) + 1e-12))
    active_errors = [
        abs(float(predicted.get(name, 0.0)) - value) / (abs(value) + 1e-12)
        for name, value in truth.items()
    ]
    return float(np.mean(smape)) if smape else np.nan, float(np.mean(active_errors)) if active_errors else np.nan


def case_row(manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    system = manifest["system"]
    phase2_root = manifest_path.parent / "phase2" / manifest["phase2_metrics"][0]["run_id"]
    rows = []
    smapes, active_errors, mses, precisions, recalls, f1s, ks = [], [], [], [], [], [], []
    for dim, metric in enumerate(manifest.get("phase2_metrics", [])):
        if not metric:
            continue
        smape, active_error = coefficient_metrics(system, dim, phase2_root, metric)
        smapes.append(smape); active_errors.append(active_error)
        mses.append(float(metric.get("loss_mse", np.nan)))
        precisions.append(float(metric.get("support_precision", np.nan)))
        recalls.append(float(metric.get("support_recall", np.nan)))
        f1s.append(float(metric.get("support_f1", np.nan)))
        ks.append(float(metric.get("K_phase2", np.nan)))
    row = {
        "system": system, "seed": int(manifest["seed"]),
        "fn_rate": float(manifest["fn_rate"]), "fp_rate": float(manifest["fp_rate"]),
        "adjacency_F1": float(manifest["adjacency_F1"]),
        "support_precision": float(np.nanmean(precisions)), "support_recall": float(np.nanmean(recalls)),
        "support_F1": float(np.nanmean(f1s)), "coefficient_sMAPE": float(np.nanmean(smapes)),
        "active_coefficient_error": float(np.nanmean(active_errors)), "rollout_MSE": float(np.nanmean(mses)),
        "K_phase2": float(np.nanmean(ks)), "dimensions_completed": len(mses),
    }
    return row


def write_tsv(path, rows):
    fields = list(rows[0]) if rows else ["system"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader(); writer.writerows(rows)


def summary_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["system"], row["fn_rate"], row["fp_rate"])].append(row)
    numeric = ["adjacency_F1", "support_precision", "support_recall", "support_F1", "coefficient_sMAPE", "active_coefficient_error", "rollout_MSE", "K_phase2"]
    output = []
    for (system, fn, fp), group in sorted(grouped.items()):
        row = {"system": system, "fn_rate": fn, "fp_rate": fp, "seed_count": len(group)}
        for key in numeric:
            values = np.asarray([g[key] for g in group], dtype=float)
            row[f"{key}_mean"] = float(np.nanmean(values))
            row[f"{key}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
        output.append(row)
    return output


def make_figures(root, rows):
    import matplotlib.pyplot as plt

    systems = ["Kuramoto", "FHN", "Rossler"]
    all_values = [r["coefficient_sMAPE"] for r in rows if np.isfinite(r["coefficient_sMAPE"])]
    vmin, vmax = (min(all_values), max(all_values)) if all_values else (0.0, 1.0)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    image = None
    for ax, system in zip(axes, systems):
        grid = np.full((4, 4), np.nan)
        for row in rows:
            if row["system"] == system:
                yi = [0.01, 0.1, 0.15, 0.2].index(row["fn_rate"])
                xi = [0.01, 0.1, 0.15, 0.2].index(row["fp_rate"])
                grid[yi, xi] = row["coefficient_sMAPE"]
        image = ax.imshow(grid, origin="lower", vmin=vmin, vmax=vmax, aspect="equal", cmap="viridis")
        ax.set_title(system)
        ax.set_xticks(range(4), ["1", "10", "15", "20"])
        ax.set_yticks(range(4), ["1", "10", "15", "20"])
        ax.set_xlabel("Spurious edges (%)")
        ax.set_ylabel("Missing edges (%)")
    fig.colorbar(image, ax=axes, label="coefficient sMAPE")
    fig.savefig(root / "coefficient_smape_heatmaps.png", dpi=220)
    fig.savefig(root / "coefficient_smape_heatmaps.pdf")
    plt.close(fig)

    colors = {"Kuramoto": "#0072B2", "FHN": "#D55E00", "Rossler": "#009E73"}
    markers = {"Kuramoto": "o", "FHN": "s", "Rossler": "^"}
    fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    for system in systems:
        subset = [r for r in rows if r["system"] == system]
        ax.scatter([r["adjacency_F1"] for r in subset], [r["support_F1"] for r in subset],
                   label=system, color=colors[system], marker=markers[system], s=52, alpha=0.85)
    ax.set_xlabel("Adjacency F1"); ax.set_ylabel("Support F1"); ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.savefig(root / "adjacency_vs_support_summary_panel.png", dpi=220)
    fig.savefig(root / "adjacency_vs_support_summary_panel.pdf")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result-root", type=Path, required=True)
    args = p.parse_args()
    root = args.result_root.expanduser().resolve()
    manifests = sorted(root.glob("*/seed*/fn*_fp*/case_manifest.json"))
    rows = [case_row(path) for path in manifests]
    write_tsv(root / "robust_raw.tsv", rows)
    summaries = summary_rows(rows)
    write_tsv(root / "robust_mean_std.tsv", summaries)
    expected = 3 * 5 * 16
    audit = {"expected_cases": expected, "completed_cases": len(rows), "missing_cases": expected - len(rows),
             "systems": ["Kuramoto", "FHN", "Rossler"], "seeds": list(range(5)),
             "fn_fp_rates": [0.01, 0.1, 0.15, 0.2], "figures_use_all_completed_seeds_without_filtering": True}
    (root / "analysis_manifest.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    make_figures(root, rows)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
