"""Reproduce the ENSO SST supplementary figure from shared compact CSV data.

The script reads the shared Fig. 6 plotting data from ../fig6/plot_data and
writes the figure files to ../plot.

The source trajectory contains 120 stored states by 71,987 nodes. Stored state
index 0 is the initial observed/true value and is not plotted; the compact
plotting data represent the remaining 119 monthly states. The plot data retain
every displayed node for spatial/error summaries and retain binned density
matrices and the four selected trajectories for the image panels.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyBboxPatch
import numpy as np


FIGURE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = FIGURE_DIR.parent / "fig6" / "plot_data"
PLOT_DIR = FIGURE_DIR / "plot"
FIGURE_KIND = FIGURE_DIR.name.lower()
# The source has 120 states; state 0 is the observed initialization value and
# the compact figure tables intentionally start at state 1.
N_SOURCE_STEPS = 120
INITIAL_STATE_INDEX = 0
N_PLOTTED_STEPS = N_SOURCE_STEPS - 1
N_TRAIN = 95
N_TEST = 24

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 6.5,
        "axes.titlesize": 7,
        "axes.labelsize": 6.5,
        "axes.linewidth": 0.55,
        "axes.spines.right": True,
        "axes.spines.top": True,
        "xtick.labelsize": 5.5,
        "ytick.labelsize": 5.5,
        "xtick.major.width": 0.45,
        "ytick.major.width": 0.45,
        "legend.frameon": False,
        "legend.fontsize": 5.5,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)

BLUE = "#2d59aa"
TRUE_COLOR = "#2f91c8"
PRED_COLOR = "#8150ba"
POINT_COLOR = "#75aac4"
MAGENTA = "#d96ba6"
CMAP = LinearSegmentedColormap.from_list("fig5_yellow_blue", ["#fffde4", "#005aa7"], N=256)


def add_panel_label(ax, label: str, x: float = -0.10) -> None:
    ax.text(x, 1.08, label, transform=ax.transAxes, fontsize=8,
            fontweight="bold", ha="left", va="bottom", clip_on=False, color="black")


def add_outer_border(fig) -> None:
    fig.patches.append(FancyBboxPatch(
        (0.008, 0.012), 0.984, 0.976, transform=fig.transFigure,
        boxstyle="round,pad=0.008,rounding_size=0.025", fill=False,
        linewidth=1.1, edgecolor="black", zorder=20,
    ))


def load_node_table() -> np.ndarray:
    table = np.genfromtxt(DATA_DIR / "spatial_node_metrics.csv", delimiter=",", names=True,
                          dtype=None, encoding="utf-8")
    if table.size != 71987:
        raise ValueError(f"expected 71987 node rows, got {table.size}")
    return table


def load_trajectory_table() -> np.ndarray:
    return np.genfromtxt(DATA_DIR / "trajectory_selected_nodes.csv", delimiter=",", names=True,
                         dtype=None, encoding="utf-8")


def load_forecast_table() -> np.ndarray:
    return np.genfromtxt(DATA_DIR / "forecast_error.csv", delimiter=",", names=True,
                         dtype=None, encoding="utf-8")


def load_density(split: str) -> tuple[np.ndarray, tuple[float, float, float, float, float]]:
    matrix = np.loadtxt(DATA_DIR / f"density_{split}.csv", delimiter=",")
    with (DATA_DIR / "density_ranges.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = next(item for item in rows if item["split"] == split)
    bounds = (float(row["true_min"]), float(row["true_max"]),
              float(row["pred_min"]), float(row["pred_max"]), float(row["r2"]))
    return matrix, bounds


def plot_map(ax, split: str, title: str, vmax: float, markers: np.ndarray | None = None) -> None:
    matrix = np.loadtxt(DATA_DIR / f"map_{split}.csv", delimiter=",")
    image = ax.imshow(matrix, extent=[120, 170, -5, 5], origin="lower", aspect="auto",
                      cmap=CMAP, vmin=0, vmax=vmax, interpolation="bilinear")
    ax.set_xlim(170, 120)
    ax.set_ylim(-5, 5)
    ax.set_xticks([170, 160, 150, 140, 130, 120])
    ax.set_xticklabels([f"{v}\N{DEGREE SIGN} W" for v in [170, 160, 150, 140, 130, 120]])
    ax.set_yticks([-5, 0, 5])
    ax.set_yticklabels([f"5\N{DEGREE SIGN} N", "0", f"5\N{DEGREE SIGN} S"])
    ax.set_xlabel("Longitude", labelpad=1)
    ax.set_ylabel("Latitude", labelpad=1)
    ax.set_title(title, color=BLUE, pad=3)
    ax.tick_params(direction="in", top=True, right=True, length=3)
    if markers is not None:
        ax.scatter(markers["longitude"], markers["latitude"], s=8, color="#8c5bbb", edgecolors="none", zorder=5)
        for row in markers:
            ax.text(float(row["longitude"]) + 1.2, float(row["latitude"]) + 0.15,
                    str(int(row["position"])), fontsize=5.5, zorder=6)
    cb = ax.figure.colorbar(image, ax=ax, fraction=0.035, pad=0.012)
    cb.set_label("MAPE", fontsize=6, labelpad=1)
    cb.ax.tick_params(labelsize=5, length=2)
    cb.set_ticks(np.linspace(0, vmax, 5))


def plot_trajectories(axes, trajectory: np.ndarray) -> None:
    positions = [1, 2, 3, 4]
    for ax, position in zip(axes, positions):
        rows = trajectory[trajectory["position"] == position]
        rows = np.sort(rows, order="month_index")
        time = rows["year"].astype(float)
        true = rows["true_sst_c"].astype(float)
        inferred = rows["inferred_sst_c"].astype(float)
        ax.plot(time[:N_TRAIN], true[:N_TRAIN], color=TRUE_COLOR, lw=0.65, label="True")
        ax.plot(time[:N_TRAIN], inferred[:N_TRAIN], color=PRED_COLOR, lw=0.65, label="Inferred")
        ax.plot(time[N_TRAIN - 1 :], true[N_TRAIN - 1 :], "--", color=TRUE_COLOR, lw=0.65)
        ax.plot(time[N_TRAIN - 1 :], inferred[N_TRAIN - 1 :], "--", color=PRED_COLOR, lw=0.65)
        lat, lon = float(rows["latitude"][0]), float(rows["longitude"][0])
        lat_dir = "N" if lat >= 0 else "S"
        lon_w = abs(lon)
        ax.set_title(f"Position {position} ({abs(lat):.2f}\N{DEGREE SIGN} {lat_dir}, {lon_w:.2f}\N{DEGREE SIGN} W) SST",
                     color=BLUE, pad=2)
        ax.set_ylabel("TEMP. (\N{DEGREE SIGN}C)", labelpad=1)
        ax.set_xlim(time[0], 2012.5)
        ax.tick_params(direction="in", top=True, right=True, length=2.5)
        if position < 4:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Year", labelpad=1)
            ax.set_xticks(np.arange(2003, 2013, 1))
        if position == 1:
            ax.legend(loc="upper right", bbox_to_anchor=(1.02, 1.02), handlelength=2.7, borderaxespad=0)


def plot_histogram(ax, nodes: np.ndarray, split: str, xlim: tuple[float, float], ylim: tuple[float, float]) -> dict[str, float]:
    values = nodes[f"mape_{split}_percent"].astype(float)
    if np.any(values <= 0):
        raise ValueError(f"non-positive MAPE values cannot be log-normal fitted for {split}")
    mean, median = float(values.mean()), float(np.median(values))
    log_values = np.log(values)
    log_mu, log_sigma = float(log_values.mean()), float(log_values.std())
    ax.hist(values, bins=20, density=True, color="#91b5c9", edgecolor="black", linewidth=0.35)
    xx = np.linspace(max(0.01, values.min()), values.max(), 300)
    fit = np.exp(-((np.log(xx) - log_mu) ** 2) / (2 * log_sigma ** 2)) / (xx * log_sigma * np.sqrt(2 * np.pi))
    ax.plot(xx, fit, color="#394fc1", lw=0.8)
    ax.axvline(mean, color=MAGENTA, ls="--", lw=0.65)
    ax.axvline(median, color="#8f75d2", ls="--", lw=0.65)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Node MAPE", labelpad=1)
    ax.set_ylabel("Node Count Density", labelpad=1)
    ax.set_title("Histogram of Node MAPE", color=BLUE, pad=3)
    ax.tick_params(direction="in", top=True, right=True, length=2.5)
    ax.text(0.56, 0.93, f"Mean = {mean:.2f}%\nMedian = {median:.2f}%\nLog-Normal Fit\nμ={log_mu:.2f}, σ={log_sigma:.2f}",
            transform=ax.transAxes, fontsize=5.1, va="top", color="#333333")
    return {"mean": mean, "median": median, "log_mu": log_mu, "log_sigma": log_sigma}


def plot_density(ax, split: str, xlim: tuple[float, float], ylim: tuple[float, float]) -> float:
    matrix, bounds = load_density(split)
    true_min, true_max, pred_min, pred_max, r2 = bounds
    image = ax.imshow(matrix, extent=[true_min, true_max, pred_min, pred_max], origin="lower",
                      aspect="auto", cmap=CMAP, interpolation="nearest")
    lo, hi = min(xlim[0], ylim[0]), max(xlim[1], ylim[1])
    ax.plot([lo, hi], [lo, hi], "--", color=MAGENTA, lw=0.65)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("True TEMP.(\N{DEGREE SIGN}C)", labelpad=1)
    ax.set_ylabel("Inferred TEMP.(\N{DEGREE SIGN}C)", labelpad=1)
    ax.set_title("Inferred Vs.True SST", color=BLUE, pad=3)
    # A common box aspect keeps all SI4 quantitative panels comparable in size.
    ax.set_aspect("auto")
    ax.tick_params(direction="in", top=True, right=True, length=2.5)
    # Place the colorbar in the inter-panel gutter so it does not shrink c/f.
    cax = ax.inset_axes([1.055, 0.0, 0.045, 1.0], transform=ax.transAxes)
    cb = ax.figure.colorbar(image, cax=cax)
    cb.set_label("Density", fontsize=5.5, labelpad=1)
    cb.ax.tick_params(labelsize=5, length=2)
    ax.text(0.04, 0.94, f"R² = {r2:.2f}", transform=ax.transAxes,
            fontsize=5.4, va="top", ha="left", color="#333333")
    return r2


def plot_variability(ax, nodes: np.ndarray, split: str, xlim: tuple[float, float], ylim: tuple[float, float]) -> dict[str, float]:
    x = nodes[f"sst_std_{split}_c"].astype(float)
    y = nodes[f"mape_{split}_percent"].astype(float)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    r2 = float(1.0 - np.sum((y - fitted) ** 2) / np.sum((y - y.mean()) ** 2))
    ax.scatter(x, y, s=4, color=POINT_COLOR, alpha=0.22, linewidths=0,
               rasterized=True, label="Error points")
    xx = np.linspace(x.min(), x.max(), 200)
    sign = "+" if intercept >= 0 else "−"
    fit_label = f"Linear fit: y = {slope:.2f}x {sign} {abs(intercept):.2f}"
    ax.plot(xx, slope * xx + intercept, color="#5136a3", lw=0.75, label=fit_label)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("SST Std Per Node", labelpad=1)
    ax.set_ylabel("MAPE", labelpad=1)
    ax.set_title("Node Error Vs. Variability", color=BLUE, pad=3)
    ax.tick_params(direction="in", top=True, right=True, length=2.5)
    ax.legend(loc="upper left", fontsize=5, handlelength=2.2, borderaxespad=0)
    return {"slope": float(slope), "intercept": float(intercept), "r2": r2}


def plot_time_error(ax, forecast: np.ndarray) -> None:
    months = forecast["month_index"].astype(float)
    values = forecast["mean_mape_percent"].astype(float)
    ax.plot(months - 1, values, "-o", color="#3989bb", lw=0.6, ms=1.7)
    ax.axvline(N_TRAIN, color=MAGENTA, ls="--", lw=0.6)
    ax.set_xlim(0, 120)
    ax.set_ylim(0.8, 7.0)
    ax.set_xlabel("Forecast Month", labelpad=1)
    ax.set_ylabel("Mean MAPE", labelpad=1)
    ax.set_title("Rollout Error Over Time", color=BLUE, pad=3)
    ax.tick_params(direction="in", top=True, right=True, length=2.5)


def save_figure(fig, name: str) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    stem = PLOT_DIR / name
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def build_fig6(nodes: np.ndarray, trajectories: np.ndarray, forecast: np.ndarray) -> None:
    fig = plt.figure(figsize=(8.76, 5.60), facecolor="white")
    gs = GridSpec(3, 12, figure=fig, height_ratios=[1.05, 1.05, 1.0], hspace=0.52, wspace=0.85)
    markers = trajectories[trajectories["month_index"] == 1]
    ax = fig.add_subplot(gs[0, :7])
    plot_map(ax, "train", "Inferred MAPE in Eastern Pacific SST on Training Data", 8, markers)
    add_panel_label(ax, "a")
    ax = fig.add_subplot(gs[1, :7])
    plot_map(ax, "test", "Inferred MAPE in Eastern Pacific SST on Test Data", 8, markers)
    add_panel_label(ax, "b")
    sub = GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[:2, 7:], hspace=0.32)
    trajectory_axes = [fig.add_subplot(sub[i, 0]) for i in range(4)]
    plot_trajectories(trajectory_axes, trajectories)
    add_panel_label(trajectory_axes[0], "c")
    ax = fig.add_subplot(gs[2, 0:3])
    plot_histogram(ax, nodes, "all", (0.9, 5.5), (0, 0.9))
    add_panel_label(ax, "d")
    ax = fig.add_subplot(gs[2, 3:6])
    plot_density(ax, "all", (24, 29), (24, 29))
    add_panel_label(ax, "e")
    ax = fig.add_subplot(gs[2, 6:9])
    plot_variability(ax, nodes, "all", (0.7, 1.5), (1.0, 5.5))
    add_panel_label(ax, "f")
    ax = fig.add_subplot(gs[2, 9:12])
    plot_time_error(ax, forecast)
    add_panel_label(ax, "g")
    add_outer_border(fig)
    fig.subplots_adjust(left=0.055, right=0.965, bottom=0.075, top=0.95)
    save_figure(fig, "fig6")
    plt.close(fig)


def build_si(nodes: np.ndarray) -> None:
    fig = plt.figure(figsize=(8.27, 6.70), facecolor="white")
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.12, 0.88, 0.88], hspace=0.40, wspace=0.43)
    ax = fig.add_subplot(gs[0, :])
    plot_map(ax, "all", "Inferred MAPE in Eastern Pacific SST on All Data", 5)
    # The wide map panel uses a closer relative offset so its label aligns with b/e.
    add_panel_label(ax, "a", x=-0.027)
    panel_specs = (("b", 1, 0), ("c", 1, 1), ("d", 1, 2),
                   ("e", 2, 0), ("f", 2, 1), ("g", 2, 2))
    panels = {label: fig.add_subplot(gs[row, col]) for label, row, col in panel_specs}
    for panel in panels.values():
        panel.set_box_aspect(0.72)

    # Keep the updated training-error bars and fitted curve fully visible.
    plot_histogram(panels["b"], nodes, "train", (0.8, 4.5), (0, 1.4))
    plot_density(panels["c"], "train", (24, 29), (24, 29))
    plot_variability(panels["d"], nodes, "train", (0.55, 1.4), (0.5, 4.5))
    # Keep the updated test-error bars and fitted curve fully visible.
    plot_histogram(panels["e"], nodes, "test", (1.0, 8.5), (0, 0.7))
    plot_density(panels["f"], "test", (22, 29), (25, 27.5))
    plot_variability(panels["g"], nodes, "test", (0.5, 2.0), (0.5, 9.0))
    for label, panel in panels.items():
        add_panel_label(panel, label)
    fig.subplots_adjust(left=0.065, right=0.965, bottom=0.075, top=0.94)
    # Shift the c/f column left by half of c's 1 °C x-axis interval (24–25 °C).
    c_position = panels["c"].get_position()
    half_tick_shift = c_position.width / 10
    for label in ("c", "f"):
        position = panels[label].get_position()
        panels[label].set_position([
            position.x0 - half_tick_shift,
            position.y0,
            position.width,
            position.height,
        ])
    save_figure(fig, "figSI4")
    plt.close(fig)


def main() -> None:
    nodes = load_node_table()
    if "figsi4" in FIGURE_KIND:
        build_si(nodes)
        print(f"generated SI figure in {PLOT_DIR}")
    else:
        build_fig6(nodes, load_trajectory_table(), load_forecast_table())
        print(f"generated main figure in {PLOT_DIR}")


if __name__ == "__main__":
    main()

