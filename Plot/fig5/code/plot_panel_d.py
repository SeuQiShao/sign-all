from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


CODE_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = CODE_ROOT.parent
METRICS = PACKAGE_ROOT / "plot_data"
OUT = CODE_ROOT / "output" / "fig5d_forecast_skill"

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

COLORS = {
    "SIGN": "#2D83B8",
    "Persistence": "#D95F02",
    "VAR(1)": "#2F7D50",
}


def read_horizon_error(snr: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    path = METRICS / f"horizon_error_snr{snr}.csv"
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        rows.extend(csv.DictReader(handle))

    expected_models = {"SIGN", "PERSISTENCE", "VAR"}
    horizons = sorted({int(row["baseline_horizon"]) for row in rows})
    if horizons != list(range(1, 99)):
        raise ValueError(f"{path.name}: expected baseline_horizon 1..98, got {horizons[:3]}..{horizons[-3:]}")
    if {row["model"] for row in rows} != expected_models:
        raise ValueError(f"{path.name}: unexpected model set")
    if len(rows) != 98 * 3:
        raise ValueError(f"{path.name}: expected 294 rows, got {len(rows)}")

    model_names = {"SIGN": "SIGN", "PERSISTENCE": "Persistence", "VAR": "VAR(1)"}
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for raw_name, display_name in model_names.items():
        selected = sorted(
            (row for row in rows if row["model"] == raw_name),
            key=lambda row: int(row["baseline_horizon"]),
        )
        result[display_name] = (
            np.asarray([int(row["baseline_horizon"]) for row in selected], dtype=float),
            np.asarray([float(row["mean_rmse"]) for row in selected], dtype=float),
        )
    return result


def plot_skill(ax, curves: dict[str, tuple[np.ndarray, np.ndarray]], snr: str) -> None:
    for model in ("SIGN", "Persistence", "VAR(1)"):
        horizon, rmse = curves[model]
        ax.plot(horizon, rmse, color=COLORS[model], lw=0.85,
                solid_capstyle="round", label=model, zorder=2)

    ax.set_title(f"{snr} dB", color="#163B70", fontsize=7.4,
                 fontweight="bold", pad=2.0)
    ax.set_xlabel("Baseline horizon", fontsize=6.2, labelpad=1.0)
    ax.set_ylabel("RMSE", fontsize=6.2, labelpad=1.0)
    ax.set_xlim(1, 100)
    ymax = max(rmse.max() for _, rmse in curves.values())
    ax.set_ylim(0, ymax * 1.08)
    ax.set_xticks([1, 20, 40, 60, 80, 100])
    ax.tick_params(length=2.0, width=0.5, labelsize=5.5, pad=1.0)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    fig = plt.figure(figsize=(3.50, 1.75), facecolor="white")
    fig.text(0.018, 0.965, "d", ha="left", va="top", color="#1C1C1C",
             fontsize=9.5, fontweight="bold", fontfamily="Arial")
    fig.text(0.500, 0.965, "Forecast skill", ha="center", va="top",
             color="#1F5E9E", fontsize=8.6, fontweight="normal", fontfamily="Arial")

    grid = fig.add_gridspec(1, 2, left=0.105, right=0.985, bottom=0.25, top=0.84,
                            wspace=0.26)
    ax_50 = fig.add_subplot(grid[0, 0])
    ax_30 = fig.add_subplot(grid[0, 1])

    curves_50 = read_horizon_error("50")
    curves_30 = read_horizon_error("30")
    plot_skill(ax_50, curves_50, "50")
    plot_skill(ax_30, curves_30, "30")

    handles = [
        Line2D([], [], color=COLORS["SIGN"], lw=0.85, label="SIGN"),
        Line2D([], [], color=COLORS["Persistence"], lw=0.85, label="Persistence"),
        Line2D([], [], color=COLORS["VAR(1)"], lw=0.85, label="VAR(1)"),
    ]
    ax_50.legend(handles=handles, loc="upper left", ncol=3, frameon=False,
                 fontsize=5.2, handlelength=1.2, columnspacing=0.75,
                 borderaxespad=0.15, handletextpad=0.35)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
