"""Reproduce the Fig. 6 working copy from compact CSV data.

The script is intentionally self-contained. It reads only ../plot_data and
writes the copied main-figure files to ../plot. The original Fig. 6 package is
left unchanged.

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
DATA_DIR = FIGURE_DIR / "plot_data"
PLOT_DIR = FIGURE_DIR / "plot"
FIGURE_KIND = FIGURE_DIR.name.lower()
FIGURE_WIDTH_MM = 180.0
FIGURE_HEIGHT_IN = 5.00
HORIZONTAL_BORDER_MARGIN = 2.0 / FIGURE_WIDTH_MM
FRAME_COLOR = "#2A2A2A"
FRAME_PAD = 0.006
FRAME_RADIUS = 0.012
FRAME_LINEWIDTH = 0.75
FRAME_LABEL_CLEARANCE_PT = 2.0
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
        "axes.titlesize": 8.6,
        "axes.titleweight": "normal",
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

BLUE = "#1F5E9E"
TRUE_COLOR = "#2f91c8"
PRED_COLOR = "#8150ba"
POINT_COLOR = "#75aac4"
MAGENTA = "#d96ba6"
CMAP = LinearSegmentedColormap.from_list("fig5_yellow_blue", ["#fffde4", "#005aa7"], N=256)


def add_panel_label(ax, label: str, x: float = -0.10):
    return ax.text(x, 1.08, label, transform=ax.transAxes, fontsize=9.5,
                   fontweight="bold", fontfamily="Arial", ha="left", va="bottom",
                   clip_on=False, color="#1C1C1C")


def align_panel_label_to_ylabel(ax, label) -> None:
    """Align a panel letter's left edge to the visible left edge of its ylabel."""
    fig = ax.figure
    fig.canvas.draw()
    ylabel_x = ax.yaxis.label.get_window_extent(fig.canvas.get_renderer()).x0
    label.set_x(ax.transAxes.inverted().transform((ylabel_x, 0))[0])


def keep_panel_labels_clear_of_frame(fig, labels) -> None:
    """Move only labels that would otherwise touch the left outer-frame stroke."""
    fig.canvas.draw()
    min_x = (fig.transFigure.transform((HORIZONTAL_BORDER_MARGIN, 0))[0]
             + fig.dpi * FRAME_LABEL_CLEARANCE_PT / 72.0)
    renderer = fig.canvas.get_renderer()
    for label in labels:
        if label.get_window_extent(renderer).x0 < min_x:
            label.set_x(label.axes.transAxes.inverted().transform((min_x, 0))[0])


def add_outer_border(fig) -> None:
    fig.patches.append(FancyBboxPatch(
        # Fig. 4's shared outer-frame style.  Its pad extends the path outward,
        # so the nominal x coordinates retain a true 2 mm blank edge margin.
        (HORIZONTAL_BORDER_MARGIN + FRAME_PAD, 0.014),
        1.0 - 2.0 * (HORIZONTAL_BORDER_MARGIN + FRAME_PAD), 0.972,
        transform=fig.transFigure,
        boxstyle=f"round,pad={FRAME_PAD},rounding_size={FRAME_RADIUS}", fill=False,
        linewidth=FRAME_LINEWIDTH, edgecolor=FRAME_COLOR, zorder=20,
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


def load_regression(panel: str, subset: str) -> tuple[float, float, float]:
    """Load a stored regression row for the requested panel and subset."""
    table = np.genfromtxt(DATA_DIR / "panel_f_regression.csv", delimiter=",", names=True,
                          dtype=None, encoding="utf-8")
    rows = np.atleast_1d(table)
    match = rows[(rows["panel"] == panel) & (rows["subset"] == subset)]
    if match.size != 1:
        raise ValueError(f"expected one paper regression row for {panel}/{subset}, got {match.size}")
    row = match[0]
    return float(row["slope"]), float(row["intercept"]), float(row["r2"])


def load_density(split: str) -> tuple[np.ndarray, tuple[float, float, float, float, float]]:
    matrix = np.loadtxt(DATA_DIR / f"density_{split}.csv", delimiter=",")
    with (DATA_DIR / "density_ranges.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = next(item for item in rows if item["split"] == split)
    bounds = (float(row["true_min"]), float(row["true_max"]),
              float(row["pred_min"]), float(row["pred_max"]), float(row["r2"]))
    return matrix, bounds


def plot_map(
    ax,
    split: str,
    title: str,
    vmax: float,
    markers: np.ndarray | None = None,
    metric_label: str = "sMAPE",
    percent_ticks: bool = False,
) -> object:
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
        ax.scatter(markers["longitude"], markers["latitude"], s=9, color="#8c5bbb", edgecolors="none", zorder=5)
        for row in markers:
            position = int(row["position"])
            longitude = float(row["longitude"])
            # The longitude axis is reversed (170° W on the left), so a
            # smaller longitude places the label visually to the right.
            label_longitude = longitude - 1.2 if position == 4 else longitude + 1.2
            ax.text(label_longitude, float(row["latitude"]) + 0.15,
                    str(position), fontsize=5.5, zorder=6)
    cb = ax.figure.colorbar(image, ax=ax, fraction=0.035, pad=0.012)
    # Place the metric name above the vertical color bar so it aligns with the
    # heatmap title baseline rather than running vertically beside the bar.
    cb.ax.set_title(metric_label, fontsize=6, pad=3)
    cb.ax.tick_params(labelsize=5, length=2)
    ticks = np.linspace(0, vmax, 5)
    cb.set_ticks(ticks)
    if percent_ticks:
        cb.set_ticklabels([f"{value:g}%" for value in ticks])
    return cb.ax


def plot_trajectories(axes, trajectory: np.ndarray):
    positions = [1, 2, 3, 4]
    legend = None
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
                     color=BLUE, pad=0)
        ax.set_ylabel("TEMP. (\N{DEGREE SIGN}C)", labelpad=1)
        ax.set_xlim(time[0], 2012.5)
        ax.tick_params(direction="in", top=True, right=True, length=2.5)
        if position < 4:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Year", labelpad=1)
            ax.set_xticks(np.arange(2003, 2013, 1))
        # The trajectory stack reads as a compact small-multiple strip; keep
        # only the left and bottom frame lines to reduce visual weight.
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(top=False, right=False)
        if position == 1:
            legend = ax.legend(loc="upper right", bbox_to_anchor=(1.02, 1.02),
                               handlelength=2.7, borderaxespad=0)
    return legend


def plot_histogram(
    ax,
    nodes: np.ndarray,
    split: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    metric_label: str = "sMAPE",
) -> dict[str, float]:
    values = nodes[f"mape_{split}_percent"].astype(float)
    if np.any(values <= 0):
        raise ValueError(f"non-positive {metric_label} values cannot be log-normal fitted for {split}")
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
    ax.set_xlabel(f"Node {metric_label}", labelpad=1)
    ax.set_ylabel("Node Count Density", labelpad=1)
    ax.set_title(f"Histogram of Node {metric_label}", color=BLUE, pad=3)
    ax.tick_params(direction="in", top=True, right=True, length=2.5)
    ax.text(0.56, 0.93, f"Mean = {mean:.2f}%\nMedian = {median:.2f}%\nLog-Normal Fit\nμ={log_mu:.2f}, σ={log_sigma:.2f}",
            transform=ax.transAxes, fontsize=5.1, va="top", color="#333333")
    return {"mean": mean, "median": median, "log_mu": log_mu, "log_sigma": log_sigma}


def plot_density(ax, split: str, xlim: tuple[float, float], ylim: tuple[float, float], equal_data: bool = True) -> float:
    matrix, bounds = load_density(split)
    true_min, true_max, pred_min, pred_max, r2 = bounds
    image = ax.imshow(matrix, extent=[true_min, true_max, pred_min, pred_max], origin="lower",
                      aspect="auto", cmap=CMAP, interpolation="nearest")
    lo, hi = min(xlim[0], ylim[0]), max(xlim[1], ylim[1])
    ax.plot([lo, hi], [lo, hi], "--", color=MAGENTA, lw=0.65)
    ax.text(0.18, 0.26, f"R² = {r2:.3f}", transform=ax.transAxes, fontsize=5.5)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("True TEMP.(\N{DEGREE SIGN}C)", labelpad=1)
    ax.set_ylabel("Inferred TEMP.(\N{DEGREE SIGN}C)", labelpad=1)
    ax.set_title("Inferred Vs.True SST", color=BLUE, pad=3)
    if equal_data:
        ax.set_aspect("equal", adjustable="box")
    else:
        ax.set_aspect("auto")
        ax.set_box_aspect(1)
    ax.tick_params(direction="in", top=True, right=True, length=2.5)
    cb = ax.figure.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
    # Place the density label above the vertical color bar and align its
    # baseline with the panel title (both use pad=3).
    cb.ax.set_title("Density", fontsize=5.5, pad=3)
    cb.ax.tick_params(labelsize=5, length=2)
    return r2


def plot_variability(
    ax,
    nodes: np.ndarray,
    split: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    regression: tuple[float, float, float] | None = None,
    metric_label: str = "sMAPE",
) -> dict[str, float]:
    x = nodes[f"sst_std_{split}_c"].astype(float)
    y = nodes[f"mape_{split}_percent"].astype(float)
    if regression is None:
        slope, intercept = np.polyfit(x, y, 1)
        fitted = slope * x + intercept
        r2 = float(1.0 - np.sum((y - fitted) ** 2) / np.sum((y - y.mean()) ** 2))
    else:
        slope, intercept, r2 = regression
    ax.scatter(x, y, s=4, color=POINT_COLOR, alpha=0.22, linewidths=0,
               rasterized=True, label="Error point")
    xx = np.linspace(x.min(), x.max(), 200)
    sign = "−" if intercept < 0 else "+"
    formula = f"y = {slope:.3f}x {sign} {abs(intercept):.3f}"
    ax.plot(xx, slope * xx + intercept, color="#5136a3", lw=0.75, label=formula)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("SST Std Per Node", labelpad=1)
    ax.set_ylabel(metric_label, labelpad=1)
    ax.set_title("Node Error Vs. Variability", color=BLUE, pad=3)
    ax.tick_params(direction="in", top=True, right=True, length=2.5)
    ax.text(0.68, 0.12, f"R² = {r2:.3f}", transform=ax.transAxes, fontsize=5.5)
    ax.legend(loc="upper left", fontsize=5, handlelength=2.2, borderaxespad=0)
    return {"slope": float(slope), "intercept": float(intercept), "r2": r2}


def plot_time_error(ax, forecast: np.ndarray, metric_label: str = "sMAPE") -> None:
    months = forecast["month_index"].astype(float)
    values = forecast["mean_mape_percent"].astype(float)
    ax.plot(months - 1, values, "-o", color="#3989bb", lw=0.6, ms=1.7)
    ax.axvline(N_TRAIN, color=MAGENTA, ls="--", lw=0.6)
    ax.set_xlim(0, 120)
    ax.set_ylim(0.8, 7.0)
    ax.set_xlabel("Forecast Month", labelpad=1)
    ax.set_ylabel(f"Mean {metric_label}", labelpad=1)
    ax.set_title("Forecast Error Over Time", color=BLUE, pad=3)
    ax.tick_params(direction="in", top=True, right=True, length=2.5)


def shift_axes_left(fig, axes, points: float) -> None:
    """Move an axis group left by a physical distance in points."""
    dx = (points / 72.0) / fig.get_figwidth()
    for axis in axes:
        pos = axis.get_position()
        axis.set_position([pos.x0 - dx, pos.y0, pos.width, pos.height])


def shift_axes_right(fig, axes, points: float) -> None:
    """Move an axis group right by a physical distance in points."""
    shift_axes_left(fig, axes, -points)


def extend_axis_right(fig, axis, points: float) -> None:
    """Increase one axis width to the right while keeping its left edge fixed."""
    dw = (points / 72.0) / fig.get_figwidth()
    pos = axis.get_position()
    axis.set_position([pos.x0, pos.y0, pos.width + dw, pos.height])


def shift_legend_left(fig, ax, legend, points: float) -> None:
    """Move an axes-anchored legend left by a physical distance in points."""
    if legend is None:
        return
    axes_width_points = ax.get_position().width * fig.get_figwidth() * 72.0
    dx_axes = points / axes_width_points
    legend.set_bbox_to_anchor((1.02 - dx_axes, 1.02), transform=ax.transAxes)


def shift_legend_right(fig, ax, legend, points: float) -> None:
    """Move an axes-anchored legend right by a physical distance in points."""
    shift_legend_left(fig, ax, legend, -points)


def shrink_axes_center(axes, factor: float) -> None:
    """Scale axes rectangles about their centers without changing text size."""
    for axis in axes:
        pos = axis.get_position()
        width, height = pos.width * factor, pos.height * factor
        center_x = pos.x0 + pos.width / 2.0
        center_y = pos.y0 + pos.height / 2.0
        axis.set_position([center_x - width / 2.0, center_y - height / 2.0,
                           width, height])


def save_figure(fig, name: str, *, tight: bool = True) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    stem = PLOT_DIR / name
    save_kwargs = {"bbox_inches": "tight"} if tight else {}
    fig.savefig(stem.with_suffix(".pdf"), **save_kwargs)
    fig.savefig(stem.with_suffix(".svg"), **save_kwargs)
    fig.savefig(stem.with_suffix(".png"), dpi=600, **save_kwargs)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, **save_kwargs)


def build_fig6(nodes: np.ndarray, trajectories: np.ndarray, forecast: np.ndarray) -> None:
    fig = plt.figure(figsize=(FIGURE_WIDTH_MM / 25.4, FIGURE_HEIGHT_IN), facecolor="white")
    # Equal top-row ratios make each map approximately twice the height of a
    # trajectory panel.  A larger bottom-row share and tighter inter-row gap
    # compress the c stack and pull the complete composite upward.
    gs = GridSpec(3, 12, figure=fig, height_ratios=[0.75, 0.75, 1.0], hspace=0.30, wspace=0.85)
    markers = trajectories[trajectories["month_index"] == 1]
    ax = fig.add_subplot(gs[0, :6])
    map_a_ax = ax
    map_a_cb = plot_map(ax, "train", "Inferred MAPE in Eastern Pacific SST on Training Data", 8, markers,
                        metric_label="MAPE", percent_ticks=True)
    a_label = add_panel_label(ax, "a")
    ax = fig.add_subplot(gs[1, :6])
    map_b_ax = ax
    map_b_cb = plot_map(ax, "test", "Inferred MAPE in Eastern Pacific SST on Test Data", 8, markers,
                        metric_label="MAPE", percent_ticks=True)
    b_label = add_panel_label(ax, "b")
    # Leave one narrow grid column as a breathing gap between the map
    # colorbars and the trajectory y-labels.
    sub = GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[:2, 7:], hspace=0.48)
    trajectory_axes = [fig.add_subplot(sub[i, 0]) for i in range(4)]
    trajectory_legend = plot_trajectories(trajectory_axes, trajectories)
    trajectory_axes[-1].set_yticks([25, 30])
    # The new horizontal MAPE title above the left color bar ends close to the
    # c-panel label; nudge c right to preserve a clear gap.
    c_label = add_panel_label(trajectory_axes[0], "c", x=-0.05)
    ax = fig.add_subplot(gs[2, 0:3])
    plot_histogram(ax, nodes, "all", (0.9, 5.5), (0, 0.9), metric_label="MAPE")
    ax.set_box_aspect(1)
    d_ax = ax
    # The d-panel tag is set slightly farther left so that, after the panel
    # itself is moved left, it shares the same physical column as the a/b tags.
    d_label = add_panel_label(d_ax, "d", x=-0.2269)
    bottom_scaled_axes = [ax]
    n_axes_before_e = len(fig.axes)
    e_ax = fig.add_subplot(gs[2, 3:6])
    plot_density(e_ax, "all", (24, 29), (24, 29))
    # The cumulative left shift places e closer to d; keep its panel tag at
    # the axes edge so it does not collide with the long d-panel title.
    e_label = add_panel_label(e_ax, "e", x=0.0)
    # plot_density adds one color-bar axis; move both axes together after the
    # final subplot adjustment so the cumulative physical 20 pt shift is
    # preserved.
    e_axes = [e_ax] + fig.axes[n_axes_before_e + 1 :]
    ax = fig.add_subplot(gs[2, 6:9])
    plot_variability(ax, nodes, "all", (0.7, 1.5), (1.0, 5.5), load_regression("f", "all"), metric_label="MAPE")
    ax.set_box_aspect(1)
    f_label = add_panel_label(ax, "f")
    f_ax = ax
    bottom_scaled_axes.append(ax)
    ax = fig.add_subplot(gs[2, 9:12])
    plot_time_error(ax, forecast, metric_label="MAPE")
    ax.set_box_aspect(1)
    g_label = add_panel_label(ax, "g")
    bottom_scaled_axes.append(ax)
    add_outer_border(fig)
    fig.subplots_adjust(left=0.055, right=0.965, bottom=0.04, top=0.95)
    # Add 8 pt to each map body while carrying its colorbar rightward by the
    # same amount, preserving the map-to-colorbar gap.
    extend_axis_right(fig, map_a_ax, 8)
    extend_axis_right(fig, map_b_ax, 8)
    shift_axes_right(fig, [map_a_cb, map_b_cb], 8)
    # Continue widening the a/b heatmap bodies by another 20 pt and move the
    # associated colorbars with them so the map-to-bar spacing is unchanged.
    extend_axis_right(fig, map_a_ax, 20)
    extend_axis_right(fig, map_b_ax, 20)
    shift_axes_right(fig, [map_a_cb, map_b_cb], 20)
    shift_axes_left(fig, e_axes, 20)
    # The prior revision moved the c legend left by 5 pt; move it back right
    # by 5 pt for the requested final position.
    shift_legend_left(fig, trajectory_axes[0], trajectory_legend, 5)
    shift_legend_right(fig, trajectory_axes[0], trajectory_legend, 5)
    # Apply the requested additional 5% reduction to the already 5%-reduced
    # d/f/g rectangles; labels and typography keep their publication sizes.
    shrink_axes_center(bottom_scaled_axes, 0.95)
    shrink_axes_center(bottom_scaled_axes, 0.95)
    # Match e's main plotting rectangle to the now-shrunken d/f/g rectangles.
    # The colorbar is carried along for alignment, but is excluded from the
    # size comparison itself.
    target_width = bottom_scaled_axes[0].get_position().width
    e_factor = target_width / e_ax.get_position().width
    shrink_axes_center(e_axes, e_factor)
    # Continue the previous e shift by another 8 pt, keeping clear gaps to d
    # and f after the equal-size correction.
    shift_axes_right(fig, e_axes, 5)
    shift_axes_right(fig, e_axes, 8)
    shift_axes_right(fig, [f_ax], 5)
    # Align the d ylabel with the map ylabels; the custom d tag above keeps
    # the panel letters in the same physical column as a/b after this shift.
    shift_axes_left(fig, [d_ax], 8.0955)
    # Anchor each tag to its actual rendered ylabel edge after every final
    # geometry adjustment; this also moves a/b/d safely inside the outer frame.
    labels = (a_label, b_label, c_label, d_label, e_label, f_label, g_label)
    for panel_ax, panel_label in ((map_a_ax, a_label), (map_b_ax, b_label),
                                  (trajectory_axes[0], c_label), (d_ax, d_label),
                                  (e_ax, e_label), (f_ax, f_label), (ax, g_label)):
        align_panel_label_to_ylabel(panel_ax, panel_label)
    keep_panel_labels_clear_of_frame(fig, labels)
    # Keep the exported canvas at the requested 180 mm width.  The SI figure
    # retains tight cropping, while this main composite uses its explicit
    # canvas so the journal dimension is exact in PDF/SVG metadata.
    save_figure(fig, "fig6", tight=False)
    plt.close(fig)


def build_si(nodes: np.ndarray) -> None:
    fig = plt.figure(figsize=(8.27, 6.70), facecolor="white")
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.2, 1, 1], hspace=0.70, wspace=0.65)
    ax = fig.add_subplot(gs[0, :])
    plot_map(ax, "all", "Inferred sMAPE in Eastern Pacific SST on All Data", 5)
    add_panel_label(ax, "a")
    ax = fig.add_subplot(gs[1, 0])
    plot_histogram(ax, nodes, "train", (0.8, 4.5), (0, 1.0))
    add_panel_label(ax, "b")
    ax = fig.add_subplot(gs[1, 1])
    plot_density(ax, "train", (24, 29), (24, 29))
    add_panel_label(ax, "c")
    ax = fig.add_subplot(gs[1, 2])
    plot_variability(ax, nodes, "train", (0.55, 1.4), (0.5, 4.5))
    add_panel_label(ax, "d")
    ax = fig.add_subplot(gs[2, 0])
    plot_histogram(ax, nodes, "test", (1.0, 8.5), (0, 0.4))
    add_panel_label(ax, "e")
    ax = fig.add_subplot(gs[2, 1])
    plot_density(ax, "test", (22, 29), (25, 27.5), equal_data=False)
    add_panel_label(ax, "f")
    ax = fig.add_subplot(gs[2, 2])
    plot_variability(ax, nodes, "test", (0.5, 2.0), (0.5, 9.0))
    add_panel_label(ax, "g")
    fig.subplots_adjust(left=0.065, right=0.965, bottom=0.075, top=0.94)
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
