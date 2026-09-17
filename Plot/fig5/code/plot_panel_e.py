from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


CODE_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = CODE_ROOT.parent
CSV_PATH = PACKAGE_ROOT / "plot_data" / "model_comparison.csv"
OUT = CODE_ROOT / "output" / "fig5e_model_comparison"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.sans-serif": ["Arial"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.65,
    "xtick.major.width": 0.55,
    "ytick.major.width": 0.55,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})

SNR30 = "#5C9BD1"
SNR50 = "#6E52B0"
NAVY = "#163B70"


def read_comparison() -> tuple[list[str], dict[tuple[str, str], np.ndarray]]:
    with CSV_PATH.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))

    models = rows[0][2:]
    if len(models) != 6:
        raise ValueError(f"Expected six models, found {len(models)}")

    values: dict[tuple[str, str], np.ndarray] = {}
    for row in rows[1:]:
        if len(row) < 2 or row[0] not in {"30", "50"}:
            continue
        metric = row[1].strip().upper()
        if metric not in {"RMSE", "MAPE"}:
            continue
        values[(row[0], metric)] = np.asarray([float(value) for value in row[2:]], dtype=float)

    for snr in ("30", "50"):
        for metric in ("RMSE", "MAPE"):
            if (snr, metric) not in values:
                raise ValueError(f"Missing {snr} dB {metric} row")
    return models, values


def plot_bars(ax, models: list[str], values: dict[tuple[str, str], np.ndarray], metric: str) -> None:
    x = np.arange(len(models), dtype=float)
    width = 0.34
    multiplier = 100.0 if metric == "MAPE" else 1.0
    values_30 = values[("30", metric)] * multiplier
    values_50 = values[("50", metric)] * multiplier

    ax.bar(x - width / 2, values_30, width=width, color=SNR30, edgecolor="none",
           label="SNR 30 dB", zorder=2)
    ax.bar(x + width / 2, values_50, width=width, color=SNR50, edgecolor="none",
           label="SNR 50 dB", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(length=2.0, width=0.5, labelsize=5.2, pad=1.0)
    ax.set_ylabel("RMSE" if metric == "RMSE" else "MAPE (%)", fontsize=6.2, labelpad=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.set_xlim(-0.65, len(models) - 0.35)
    if metric == "RMSE":
        ax.set_ylim(0, 0.31)
        ax.set_yticks(np.arange(0, 0.31, 0.05))
    else:
        ax.set_ylim(0, 120)
        ax.set_yticks(np.arange(0, 121, 20))


def main() -> None:
    models, values = read_comparison()
    fig = plt.figure(figsize=(3.50, 1.75), facecolor="white")
    fig.text(0.018, 0.965, "e", ha="left", va="top", color="#1C1C1C",
             fontsize=9.5, fontweight="bold", fontfamily="Arial")
    fig.text(0.500, 0.965, "Model comparison", ha="center", va="top",
             color="#1F5E9E", fontsize=8.6, fontweight="normal", fontfamily="Arial")

    grid = fig.add_gridspec(1, 2, left=0.105, right=0.985, bottom=0.31, top=0.84,
                            wspace=0.26)
    ax_rmse = fig.add_subplot(grid[0, 0])
    ax_mape = fig.add_subplot(grid[0, 1])
    plot_bars(ax_rmse, models, values, "RMSE")
    plot_bars(ax_mape, models, values, "MAPE")

    ax_mape.legend(loc="upper right", frameon=False, fontsize=5.2,
                   handlelength=1.0, borderaxespad=0.15,
                   handletextpad=0.35, labelspacing=0.25)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
