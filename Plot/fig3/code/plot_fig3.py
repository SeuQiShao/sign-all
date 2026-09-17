"""Reproduce the Nature-style SIGN Fig. 3 (Phase-I/Phase-II) from CSV data.

This submission package is self-contained: the renderer reads only the
compact CSV files in ``../plot_data`` and writes publication exports to
``code/output``.  No private source checkout or raw experiment file is needed
to redraw the figure.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import ScalarFormatter
from matplotlib.transforms import ScaledTranslation
import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PLOT_DATA = PACKAGE_ROOT / "plot_data"
OUT = PACKAGE_ROOT / "code" / "output"
OUT.mkdir(parents=True, exist_ok=True)


# Shared publication figure contract: 183 mm wide, white background and
# editable text. Typography and the muted blue hierarchy match Figs. 2, 4
# and 5; all plotted data and panel content remain unchanged.
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "stix",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "font.size": 6.0,
        "axes.titlesize": 6.0,
        "axes.labelsize": 6.0,
        "xtick.labelsize": 6.0,
        "ytick.labelsize": 6.0,
        "legend.fontsize": 6.0,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)

BLACK = "#2A2A2A"
BLUE = "#2D7AB4"
ORANGE = "#E69F00"
VERMILION = "#D55E00"
PURPLE = "#8A6BB8"
MID_BLUE = "#79A9CB"
GRID = "#B7C5D0"
PALE_GRID = "#DCE6ED"
TITLE = "#1F5E9E"
FIG_HEIGHT_IN = 5.708661
FIG_WIDTH_IN = 7.204724
THREE_PT_FIG_Y = (3.0 / 72.0) / FIG_HEIGHT_IN
TWO_PT_FIG_X = (2.0 / 72.0) / FIG_WIDTH_IN
ONE_PT_FIG_X = (1.0 / 72.0) / FIG_WIDTH_IN
TWO_PT_FIG_Y = (2.0 / 72.0) / FIG_HEIGHT_IN
TEN_PT_FIG_X = (10.0 / 72.0) / FIG_WIDTH_IN
SEVEN_PT_FIG_X = (7.0 / 72.0) / FIG_WIDTH_IN


def cmap_from(stops, name):
    return LinearSegmentedColormap.from_list(name, stops, N=256)


# Restore the earlier blue scale for D/F/H and soften it one step further.
FREQ_CMAP = cmap_from(["#FFFFFF", "#D9E9F3", "#6FA9D0"], "freq_blue_light")
HEAT_CMAP = cmap_from(["#F7FBFD", "#9FC6DD", BLUE], "heat_blue")
DELTA_CMAP = cmap_from(["#6FA8CD", "#FFFFFF", "#E58A4A"], "delta")


def read_tsv(path):
    return pd.read_csv(path)


def style_axis(ax, *, grid=False):
    ax.tick_params(direction="out", pad=2)
    ax.spines["left"].set_color(BLACK)
    ax.spines["bottom"].set_color(BLACK)
    ax.spines["left"].set_linewidth(0.65)
    ax.spines["bottom"].set_linewidth(0.65)
    if grid:
        ax.grid(True, color=PALE_GRID, linewidth=0.40, alpha=0.8)
        ax.set_axisbelow(True)


def panel_title(fig, rect, letter, title, *, y_offset=0.0, letter_x=None,
                letter_y_offset=0.0):
    """Use the common individual-panel header treatment from Fig. 4."""
    x, y, w, h = rect
    fig.text(x + 0.014 if letter_x is None else letter_x,
             y + h - 0.018 + y_offset + letter_y_offset, letter, ha="left", va="top",
             fontsize=9.5, fontweight="bold", fontfamily="Arial", color="#1C1C1C")
    fig.text(x + w / 2, y + h - 0.015 + y_offset, title, ha="center", va="top",
             fontsize=8.6, fontweight="normal", fontfamily="Arial", color="#1F5E9E")


def draw_row_frame(fig, rect):
    """Retain the original three-row rounded enclosure structure."""
    x, y, w, h = rect
    fig.add_artist(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.010",
        transform=fig.transFigure,
        facecolor="none", edgecolor=BLACK, linewidth=0.70,
        clip_on=False, zorder=-1,
    ))


def load_library_summary():
    return read_tsv(PLOT_DATA / "panel_b.csv")


def panel_a_data():
    return read_tsv(PLOT_DATA / "panel_a.csv")


def draw_panel_a(fig, rect):
    d = panel_a_data()
    ax = fig.add_axes(rect)
    style_axis(ax)
    labels, nominal, phase1, phase1_sd, reduction = [], [], [], [], []
    for lib in ["L1", "L2", "L3"]:
        for system in ["Kuramoto", "FHN", "Rossler"]:
            r = d[(d.library == lib) & (d.system == system)].iloc[0]
            labels.append({"Kuramoto": "Kur", "FHN": "FHN", "Rossler": "Rös"}[system] + f"-{lib}")
            nominal.append(r.nominal)
            phase1.append(r.phase1_mean)
            phase1_sd.append(r.phase1_sd)
            reduction.append(r.reduction)
    x = np.arange(len(labels))
    width = 0.32
    bottom = 0.9
    ax.bar(x - width / 2, nominal, width, bottom=bottom, color=ORANGE,
           edgecolor=BLACK, linewidth=0.3, label="Original")
    ax.bar(x + width / 2, phase1, width, bottom=bottom, color=BLUE,
           edgecolor=BLACK, linewidth=0.3, label="Phase-I")
    ax.errorbar(x + width / 2, np.asarray(phase1) + bottom,
                yerr=np.asarray(phase1_sd), fmt="none", ecolor=BLACK,
                elinewidth=0.45, capsize=1.8, capthick=0.45, zorder=5)
    ax.set_yscale("log")
    ax.set_ylim(0.9, 170)
    ax.set_yticks([1, 10, 50, 100, 150])
    ax.set_yticklabels(["1", "10", "50", "100", "150"])
    ax.set_ylabel("Candidate terms", fontfamily="Arial", fontweight="normal")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Library", labelpad=4, fontfamily="Arial", fontweight="normal")
    ax.tick_params(axis="x", pad=3)
    for x_sep in [2.5, 5.5]:
        ax.axvline(x_sep, color=PALE_GRID, linewidth=0.5, zorder=0)

    ax2 = ax.twinx()
    ax2.plot(x, reduction, color=BLACK, marker="o", markersize=2.5,
             linewidth=0.85, label="Reduction")
    ax2.set_ylim(0.75, 1.0)
    ax2.set_yticks([0.75, 0.8, 0.9, 1.0])
    ax2.set_yticklabels(["75%", "80%", "90%", "100%"])
    ax2.set_ylabel("Reduction", labelpad=1, fontfamily="Arial", fontweight="normal")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(BLACK)
    ax2.tick_params(direction="out", pad=2)
    handles = [Rectangle((0, 0), 1, 1, facecolor=ORANGE, edgecolor=BLACK,
                         linewidth=0.3),
               Rectangle((0, 0), 1, 1, facecolor=BLUE, edgecolor=BLACK,
                         linewidth=0.3),
               Line2D([0], [0], color=BLACK, marker="o", markersize=2.5,
                      linewidth=0.85)]
    ax.legend(handles, ["Original", "Phase-I", "Reduction"],
              loc="upper left", bbox_to_anchor=(0.02, 1.015 + (TWO_PT_FIG_Y + THREE_PT_FIG_Y) / 0.150), ncol=3,
              frameon=False, handlelength=1.3, columnspacing=0.8,
              borderaxespad=0)
    return ax


def draw_panel_b(fig, rect):
    d = load_library_summary()
    # Three equal small multiples make metric-specific scales explicit.
    x0, y0, w, h = rect
    gap = 0.012
    aw = (w - 2 * gap) / 3
    axes = []
    for i, metric in enumerate(["precision", "recall", "F1"]):
        ax = fig.add_axes([x0 + i * (aw + gap), y0, aw, h])
        style_axis(ax)
        x = np.array([50, 100, 150])
        p1 = [d.loc[d.library == lib, f"P1_{metric}_mean"].iloc[0] for lib in ["L1", "L2", "L3"]]
        p2 = [d.loc[d.library == lib, f"P2_{metric}_mean"].iloc[0] for lib in ["L1", "L2", "L3"]]
        ax.plot(x, p1, "-o", color=BLUE, markerfacecolor=BLUE, markersize=3,
                linewidth=0.9, label="Phase-I")
        ax.plot(x, p2, "-s", color=ORANGE, markerfacecolor=ORANGE, markersize=3,
                linewidth=0.9, label="Phase-II")
        ax.set_xlim(42, 158)
        ax.set_ylim(0.7, 1.0)
        ax.set_yticks([0.7, 0.8, 0.9, 1.0])
        ax.set_xticks(x)
        ax.grid(axis="y", color=PALE_GRID, linewidth=0.35, alpha=0.75)
        ax.set_axisbelow(True)
        ax.set_xlabel("")
        ax.set_title(metric.capitalize(), fontsize=6.0, fontfamily="Arial",
                     fontweight="normal", color=BLACK, pad=3)
        if i == 0:
            ax.set_ylabel("")
            ax.legend(loc="lower left", frameon=False, handlelength=1.5,
                      borderaxespad=0.1)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
            ax.spines["left"].set_visible(True)
            ax.spines["left"].set_color(BLACK)
        axes.append(ax)
    fig.text(x0 + w / 2, y0 - 0.038, "Library size", ha="center",
             va="top", fontsize=6.0, fontfamily="Arial", fontweight="normal",
             color=BLACK)
    return axes


def heatmap(ax, values, *, vmin, vmax, cmap, labels=None, xlabels=None,
            ylabels=None, text_fmt=".2f", norm=None, grid_color=GRID,
            text_color=BLACK, cell_fontsize=6.0):
    values = np.asarray(values, dtype=float)
    im = ax.imshow(values, cmap=cmap, vmin=None if norm else vmin,
                   vmax=None if norm else vmax, norm=norm, aspect="auto",
                   interpolation="nearest")
    ax.set_xticks(np.arange(values.shape[1]))
    ax.set_yticks(np.arange(values.shape[0]))
    if xlabels is not None:
        ax.set_xticklabels(xlabels)
    if ylabels is not None:
        ax.set_yticklabels(ylabels)
    ax.tick_params(length=0, pad=2)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]) and labels is not None:
                ax.text(j, i, labels[i][j], ha="center", va="center",
                        color=text_color, fontsize=cell_fontsize)
    for i in range(values.shape[0] + 1):
        ax.axhline(i - 0.5, color=grid_color, linewidth=0.35, zorder=3)
    for j in range(values.shape[1] + 1):
        ax.axvline(j - 0.5, color=grid_color, linewidth=0.35, zorder=3)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(BLACK)
        spine.set_linewidth(0.55)
    return im


def style_fig2_colorbar(cb, title, *, title_pad=2.5):
    """Apply Fig. 2's compact colour-bar typography and line treatment."""
    cb.ax.set_title(title, fontsize=6.5, pad=title_pad, fontfamily="Times New Roman")
    cb.ax.tick_params(labelsize=6.5, width=0.42, length=1.7, pad=1.2)
    for label in cb.ax.get_yticklabels():
        label.set_fontfamily("Times New Roman")
        label.set_fontweight("normal")
    cb.outline.set_linewidth(0.42)
    cb.outline.set_edgecolor("#222222")


def x_tick_center_y(fig, ax):
    """Return the figure-coordinate centreline of an x-axis tick label."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = ax.get_xticklabels()[0].get_window_extent(renderer)
    return fig.transFigure.inverted().transform((0, (bbox.y0 + bbox.y1) / 2))[1]


def draw_panel_c(fig, rect):
    d = read_tsv(PLOT_DATA / "panel_c.csv")
    eps = sorted(d.eps_factor.unique())
    mins = sorted(d.min_samples_factor.unique())
    vals = np.zeros((len(mins), len(eps)))
    delta_vals = np.zeros_like(vals)
    for i, m in enumerate(mins):
        for j, e in enumerate(eps):
            vals[i, j] = d[(d.min_samples_factor == m) & (d.eps_factor == e)].f1_mean.iloc[0]
            delta_vals[i, j] = d[(d.min_samples_factor == m) &
                                 (d.eps_factor == e)].delta_f1_vs_nominal.iloc[0]
    x0, y0, w, h = rect
    ax = fig.add_axes([x0, y0, w * 0.78, h])
    labels = [[f"{v:.2f}" for v in row] for row in vals]
    norm = TwoSlopeNorm(vmin=-0.10, vcenter=0.0, vmax=0.025)
    im = heatmap(ax, delta_vals, vmin=-0.10, vmax=0.025, cmap=DELTA_CMAP,
                 norm=norm,
                 labels=labels, xlabels=[f"{v:g}" for v in eps],
                 ylabels=[f"{v:g}" for v in mins])
    ax.set_xlabel("")
    ax.set_ylabel("Minimum-sample", labelpad=3, fontfamily="Arial",
                  fontweight="normal")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.028,
                      ticks=[-0.10, -0.05, 0.0, 0.025])
    cb.ax.set_yticklabels(["−0.10", "−0.05", "0", "+0.025"])
    cb.ax.set_title("ΔF1", fontsize=6.0, pad=3)
    cb.ax.tick_params(labelsize=6.0, pad=1.5)
    cb.outline.set_linewidth(0.45)
    fig.text(x0 + 0.39 * w, y0 - 0.027 - THREE_PT_FIG_Y, "Neighborhood radius",
             ha="center", va="center", fontsize=6.0, fontfamily="Arial",
             fontweight="normal", color=BLACK)


def draw_panel_d(fig, rect):
    d = read_tsv(PLOT_DATA / "panel_d.csv")
    systems = ["Kuramoto", "FHN", "Rossler"]
    conds = ["clean", "snr50", "sparse200"]
    methods = ["DBSCAN", "All", "Voting", "HC"]
    vals = np.array([[d[(d.system == s) & (d.condition == c) & (d.method == m)].f1_mean.iloc[0]
                      for s in systems for c in conds] for m in methods])
    x0, y0, w, h = rect
    # Expand the heatmap body by 1 cm on the 183 mm-wide publication canvas.
    ax = fig.add_axes([x0 + 0.02 * w, y0, w * 0.97, h])
    labels = [[f"{v:.2f}" for v in row] for row in vals]
    im = heatmap(ax, vals, vmin=0.60, vmax=1.0, cmap=FREQ_CMAP,
                 labels=labels, xlabels=["C", "N", "S"] * 3,
                 ylabels=methods, cell_fontsize=6.0)
    for x in [2.5, 5.5]:
        ax.axvline(x, color=BLACK, linewidth=0.8, zorder=4)
    # Anchor each system heading to the exact centre of its middle heatmap
    # cell: FHN aligns with 0.95 and Rössler aligns with 0.87.
    for cell_x, name in zip([1, 4, 7], ["Kuramoto", "FHN", "Rössler"]):
        label_x = fig.transFigure.inverted().transform(
            ax.transData.transform((cell_x, 0))
        )[0]
        if name == "FHN":
            label_x -= TEN_PT_FIG_X - (5.0 / 72.0) / FIG_WIDTH_IN
        elif name == "Rössler":
            label_x -= TEN_PT_FIG_X - TWO_PT_FIG_X
        fig.text(label_x, y0 + h + 0.008, name,
                 ha="center", va="bottom", fontsize=6.0,
                 fontfamily="Arial", fontweight="normal", color=BLACK)
    ax.set_xlabel("")
    ax.set_ylabel("")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025,
                      ticks=[0.60, 0.80, 1.0])
    style_fig2_colorbar(cb, "")
    cbar_box = cb.ax.get_position()
    fig.text(cbar_box.x0 + cbar_box.width / 2, x_tick_center_y(fig, ax), "F1",
             ha="center", va="center", fontsize=6.5,
             fontfamily="Times New Roman")
    fig.text(x0 + 0.425 * w, y0 - 0.027 - THREE_PT_FIG_Y, "Condition", ha="center",
             va="center", fontsize=6.0, fontfamily="Arial", fontweight="normal",
             color=BLACK)


def draw_panel_e(fig, rect):
    d = read_tsv(PLOT_DATA / "panel_e.csv")
    horizons = [5, 10, 50, 100]
    methods = ["DBSCAN", "All", "HC", "Voting"]
    colors = [BLUE, "#737373", PURPLE, VERMILION]
    markers = ["o", "s", "D", "^"]
    x0, y0, w, h = rect
    ax = fig.add_axes([x0 + 0.14 * w, y0, 0.82 * w, h])
    style_axis(ax)
    ax.set_xlim(3, 103)
    ax.set_xticks(horizons)
    ax.set_xlabel("")
    ax.set_yscale("log")
    ax.set_ylim(2e-5, 6e-1)
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1])
    ax.set_yticklabels([
        r"$1\mathrm{e}^{-4}$", r"$1\mathrm{e}^{-3}$",
        r"$1\mathrm{e}^{-2}$", r"$1\mathrm{e}^{-1}$",
    ])
    ax.set_ylabel("Euler rollout error", labelpad=3, fontfamily="Arial",
                  fontsize=6.0, fontweight="normal")
    ax.grid(axis="y", which="major", color=PALE_GRID, linewidth=0.35)
    ax.set_axisbelow(True)
    handles = []
    for method, color, marker in zip(methods, colors, markers):
        sub = d[d.method == method].set_index("horizon").loc[horizons]
        y = sub.Euler_error_mean.to_numpy()
        e = sub.Euler_error_std.to_numpy()
        z = 6 if method == "DBSCAN" else 3
        ax.errorbar(horizons, y, yerr=e, color=color, marker=marker,
                    markerfacecolor=color, markersize=3.0, linewidth=0.85,
                    capsize=1.6, elinewidth=0.5, zorder=z)
        handles.append(Line2D([0], [0], color=color, marker=marker,
                              markerfacecolor=color, markersize=3.0,
                              linewidth=0.85, label=method))
    # Separate the two closely spaced early horizons without changing their
    # actual x positions or the linear horizon scale.
    tick_labels = ax.get_xticklabels()
    tick_labels[0].set_horizontalalignment("right")
    tick_labels[1].set_horizontalalignment("left")
    ax.tick_params(axis="x", labelsize=6.0)
    legend_order = [0, 2, 1, 3]
    ax.legend(handles=[handles[i] for i in legend_order], loc="center",
              bbox_to_anchor=(0.56, 0.39),
              frameon=False, ncol=2, handlelength=1.4, columnspacing=0.8,
              borderaxespad=0.15)
    fig.text(x0 + 0.55 * w, y0 - 0.027 - THREE_PT_FIG_Y, "Forecast horizon", ha="center",
             va="center", fontsize=6.0, fontfamily="Arial", fontweight="normal",
             color=BLACK)


def aggregate_l2_support():
    d = read_tsv(PLOT_DATA / "panel_f.csv")
    systems = ["Kuramoto", "FHN", "Rossler"]
    conditions = ["clean", "snr50", "sparse200"]
    z = np.zeros((3, 3))
    delta = np.zeros((3, 3))
    for i, system in enumerate(systems):
        for j, condition in enumerate(conditions):
            q = d[(d.system == system) & (d.condition == condition)]
            z[i, j] = q.p1_f1.iloc[0]
            delta[i, j] = q.delta_f1.iloc[0]
    return z, delta


def draw_panel_f(fig, rect):
    z, delta = aggregate_l2_support()
    x0, y0, w, h = rect
    ax = fig.add_axes([x0, y0, w * 0.80, h])
    labels = [[f"{z[i, j]:.2f}\n{delta[i, j]:+0.2f}" for j in range(3)] for i in range(3)]
    im = heatmap(ax, z, vmin=0.80, vmax=1.0, cmap=FREQ_CMAP, labels=labels,
                 xlabels=["Clean", "Noise", "Sparse"],
                 ylabels=["Kur", "FHN", "Rös"])
    ax.set_xlabel("")
    ax.set_ylabel("")
    cax = fig.add_axes([x0 + 0.835 * w, y0, 0.030 * w, h])
    cb = fig.colorbar(im, cax=cax, ticks=[0.80, 0.90, 1.0])
    cb.ax.yaxis.set_ticks_position("right")
    cb.ax.tick_params(labelleft=False, labelright=True)
    cb.ax.set_xlabel("")
    style_fig2_colorbar(cb, "")
    label_y = x_tick_center_y(fig, ax)
    fig.text(x0 + 0.850 * w, label_y, "F1",
             ha="center", va="center", fontsize=6.5,
             fontfamily="Times New Roman")
    return label_y


def paired_summary():
    d = read_tsv(PLOT_DATA / "panel_g.csv").set_index("metric")
    return d.loc[["precision", "recall", "F1", "K"]]


def draw_panel_g(fig, rect):
    d = paired_summary()
    x0, y0, w, h = rect
    ax = fig.add_axes(rect)
    style_axis(ax, grid=False)
    metric_lim = (0.88, 1.005)
    k_lim = (2.8, 3.8)
    map_k = lambda v: metric_lim[0] + (v - k_lim[0]) / (k_lim[1] - k_lim[0]) * (metric_lim[1] - metric_lim[0])
    rows = [("K", "K", 3, "D"), ("precision", "Precision", 2, "o"),
            ("recall", "Recall", 1, "s"), ("F1", "F1", 0, "^")]
    for metric, label, row, marker in rows:
        p1, p2 = float(d.loc[metric, "P1_mean"]), float(d.loc[metric, "P2_mean"])
        x1, x2 = (map_k(p1), map_k(p2)) if metric == "K" else (p1, p2)
        ax.plot([x1, x2], [row, row], color="#BFC2C4", linewidth=2.0, zorder=1)
        ax.plot(x1, row, marker=marker, color=BLUE, markerfacecolor=BLUE,
                markersize=4.0, zorder=3)
        ax.plot(x2, row, marker=marker, color=ORANGE, markerfacecolor=ORANGE,
                markersize=4.0, zorder=3)
        delta_color = VERMILION if (p2 - p1) < 0 else ORANGE
        ax.text((x1 + x2) / 2, row + 0.13, f"Δ{p2-p1:+.3f}",
                ha="center", va="bottom", color=delta_color, fontsize=6.0)
    ax.set_xlim(*metric_lim)
    ax.set_ylim(-0.65, 3.75)
    ax.set_xticks([0.90, 0.95, 1.00])
    ax.set_yticks([3, 2, 1, 0])
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.axhline(2.5, color="#C6CDD2", linewidth=0.55,
               linestyle=(0, (2.0, 2.0)), zorder=0)
    for row, label in zip([3, 2, 1, 0], ["K", "Precision", "Recall", "F1"]):
        ax.text(metric_lim[0] + 0.0025, row, label, ha="left", va="center",
                fontsize=6.0, color=BLACK, zorder=5, fontfamily="Arial",
                fontweight="normal")
    ax.set_xlabel("")
    ax.set_ylabel("")
    # Top manual K scale.
    top = ax.secondary_xaxis("top", functions=(
        lambda x: k_lim[0] + (x - metric_lim[0]) / (metric_lim[1] - metric_lim[0]) * (k_lim[1] - k_lim[0]),
        lambda k: metric_lim[0] + (k - k_lim[0]) / (k_lim[1] - k_lim[0]) * (metric_lim[1] - metric_lim[0]),
    ))
    top.set_xticks([2.8, 3.0, 3.2, 3.4, 3.6, 3.8])
    top.tick_params(axis="x", direction="in", pad=2)
    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(BLACK)
        ax.spines[side].set_linewidth(0.55)
    top.spines["top"].set_visible(True)
    top.spines["top"].set_color(BLACK)
    top.spines["top"].set_linewidth(0.55)
    # The tick values are sufficient to identify the K scale; omitting a
    # second title here prevents collision with the panel heading at final size.
    ax.legend([Line2D([0], [0], marker="o", color=BLUE, linestyle="none", markersize=3.5),
               Line2D([0], [0], marker="o", color=ORANGE, linestyle="none", markersize=3.5)],
              ["P1", "P2"], loc="lower left", bbox_to_anchor=(0.08, 0.0),
              frameon=False, ncol=2,
              handlelength=0.8, columnspacing=0.7, borderaxespad=0.1)
def format_term(name):
    # Keep the compact mathematical notation, but write the two exponential
    # terms with ``exp`` so nested mathtext scripts cannot drop below 5 pt in
    # the vector PDF.
    replacements = {
        "x1^1": r"$x_{1}$", "x2^1": r"$x_{2}$", "x3^1": r"$x_{3}$",
        "x1^2": r"$x_{1}^{2}$", "x3^2": r"$x_{3}^{2}$", "x1^3": r"$x_{1}^{3}$",
        "x3*exp(x3)": "",
        "(x_j - x_i)^1": r"$x_{j}-x_{i}$",
        "(x1x3)^1": r"$x_{1}x_{3}$", "(x1x3)^2": r"$(x_{1}x_{3})^{2}$",
        "(x1x3)^3": r"$(x_{1}x_{3})^{3}$", "x_j^-1": r"$x_{j}^{-1}$",
        "exp(x_j*x_i)": "", "1": r"$1$",
    }
    return replacements.get(name, name)


def support_heatmap_data():
    d = read_tsv(PLOT_DATA / "panel_h.csv")
    ordered = d.sort_values("column").drop_duplicates(["column"])
    cols = [(int(row.dim), row.term_index, str(row.term_name), str(row.role))
            for row in ordered.itertuples(index=False)]
    blocks = ["Clean-P1", "Clean-P2", "Noise-P1", "Noise-P2", "Sparse-P1", "Sparse-P2"]
    freq = np.full((len(blocks), len(cols)), np.nan)
    coeff = np.full((len(blocks), len(cols)), np.nan)
    for j, column in enumerate(ordered.column.to_numpy()):
        q = d[d.column == column].set_index("block")
        for i, block in enumerate(blocks):
            if block in q.index:
                freq[i, j] = float(q.loc[block, "frequency"])
                coeff[i, j] = float(q.loc[block, "mean_selected_coefficient"])
    return cols, freq, coeff


def coeff_text(v):
    if not np.isfinite(v):
        return ""
    if v == 0:
        return "0.00"
    if abs(v) < 0.01 or abs(v) >= 100:
        return f"{v:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    return f"{v:.2f}"


def draw_compact_exponential(ax, center, name, *, shift_pt=0.0):
    """Render the complete exponent through the common MathText engine."""
    formula = {
        "x3*exp(x3)": r"$x_3\,\mathrm{e}^{x_3}$",
        "exp(x_j*x_i)": r"$\mathrm{e}^{x_j x_i}$",
    }[name]
    ax.text(
        center,
        0.08,
        formula,
        transform=ax.transData + ScaledTranslation(
            shift_pt / 72.0, 0.0, ax.figure.dpi_scale_trans
        ),
        ha="center",
        va="baseline",
        fontsize=7.2,
        color=BLACK,
    )


def draw_panel_h(fig, rect, tick_row_y):
    cols, freq, coeff = support_heatmap_data()
    x0, y0, w, h = rect
    heat_rect = [x0 + 0.05 * w, y0, 0.89 * w, h]
    ax = fig.add_axes(heat_rect)
    # Color retains every selection frequency; text reports the mean active
    # coefficient for every cell where a coefficient is available.
    labels = [[coeff_text(coeff[i, j])
               for j in range(coeff.shape[1])] for i in range(coeff.shape[0])]
    im = heatmap(ax, freq, vmin=0, vmax=1, cmap=FREQ_CMAP, labels=labels,
                  xlabels=[""] * freq.shape[1],
                  ylabels=["C-P1", "C-P2", "N-P1", "N-P2", "S-P1", "S-P2"],
                  grid_color=GRID, text_color=BLACK,
                  cell_fontsize=6.0)
    ax.tick_params(axis="x", bottom=False, top=False)
    ax.set_yticklabels(["C-P1", "C-P2", "N-P1", "N-P2", "S-P1", "S-P2"])
    ax.set_ylabel("")
    # Term labels and dimension group brackets live in a separate top/bottom
    # band so they never collide with the cell values.
    centers = []
    start = 0
    for dim in [0, 1, 2]:
        idx = [i for i, c in enumerate(cols) if c[0] == dim]
        if not idx:
            continue
        end = idx[-1]
        centers.append((start, end, dim))
        start = end + 1
    ax_top = fig.add_axes([x0 + 0.05 * w, y0 + h + 0.002, 0.89 * w, 0.032],
                          frameon=False)
    ax_top.set_xlim(-0.5, freq.shape[1] - 0.5)
    ax_top.set_ylim(0, 1)
    ax_top.axis("off")
    formula_texts = {0: [], 1: [], 2: []}
    compact_exponentials = []
    for j, (dim, _, name, _) in enumerate(cols):
        # The compact exponential forms keep the dense H header inside its
        # dimension blocks.  Like the ordinary Fig. 2 term labels, they use
        # Matplotlib mathtext rather than the TeX-only fraction path.
        term_fontsize = 7.2
        term_font = None
        # Align every formula item to the same mathematical baseline as
        # x_j-x_i, regardless of its ascenders, descenders or scripts.
        txt = ax_top.text(j, 0.08, format_term(name), ha="center", va="baseline",
                          rotation=0, rotation_mode="anchor", fontsize=term_fontsize,
                          fontfamily=term_font)
        if name in {"x3*exp(x3)", "exp(x_j*x_i)"}:
            compact_exponentials.append((name, txt))
        formula_texts[dim].append((j, txt))
    # Resolve only actual glyph collisions.  Short terms retain their nominal
    # column centers; long terms borrow unused neighboring space while order
    # and dimension-block containment are preserved.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    data_per_px = freq.shape[1] / ax_top.get_window_extent(renderer).width
    for start, end, dim in centers:
        items = formula_texts[dim]
        half_widths = [0.5 * txt.get_window_extent(renderer).width * data_per_px
                       for _, txt in items]
        left_bound, right_bound = start - 0.42, end + 0.42
        available = right_bound - left_bound
        gap = 0.10
        if len(items) > 1 and sum(2 * x for x in half_widths) + gap * (len(items) - 1) > available:
            gap = (available - sum(2 * x for x in half_widths)) / (len(items) - 1)
        positions = [max(float(items[0][0]), left_bound + half_widths[0])]
        for i in range(1, len(items)):
            min_x = positions[-1] + half_widths[i - 1] + half_widths[i] + gap
            positions.append(max(float(items[i][0]), min_x))
        overflow = positions[-1] + half_widths[-1] - right_bound
        if overflow > 0:
            positions = [x - overflow for x in positions]
        underflow = left_bound - (positions[0] - half_widths[0])
        if underflow > 0:
            positions = [x + underflow for x in positions]
        for (_, txt), xpos in zip(items, positions):
            txt.set_x(xpos)
    # The final three terms form one long mathematical group. Shift the group
    # left together to give the colour-bar label a clear right-hand margin.
    for _, txt in formula_texts[2][-3:]:
        txt.set_x(txt.get_position()[0] - 0.18)
    for name, placeholder in compact_exponentials:
        center = placeholder.get_position()[0]
        if name == "x3*exp(x3)":
            center -= 0.08
        draw_compact_exponential(
            ax_top,
            center,
            name,
            shift_pt=7.0 if name == "exp(x_j*x_i)" else 0.0,
        )
    # A compact but fully readable lower band keeps the differential labels
    # optically aligned with F1/Freq. without leaving a loose footer.
    bottom_band_h = 0.035
    ax_bottom = fig.add_axes([x0 + 0.05 * w, y0 - bottom_band_h,
                              0.89 * w, bottom_band_h],
                             frameon=False)
    ax_bottom.set_xlim(-0.5, freq.shape[1] - 0.5)
    ax_bottom.set_ylim(0, 1)
    ax_bottom.axis("off")
    # Keep the grouped differential labels on the F1/Freq. optical baseline,
    # while retaining a deliberately short pair of boundary bars per group.
    label_y = np.clip((tick_row_y - (y0 - bottom_band_h)) / bottom_band_h,
                      0.25, 0.75)
    bar_half_height = 0.16
    boundaries = [centers[0][0] - 0.45]
    boundaries.extend(end + 0.5 for _, end, _ in centers[:-1])
    boundaries.append(centers[-1][1] + 0.45)
    for boundary in boundaries:
        ax_bottom.plot([boundary, boundary],
                       [label_y - bar_half_height, label_y + bar_half_height],
                       color="#555555",
                       linewidth=0.5, clip_on=False)
    for group_idx, (start, end, dim) in enumerate(centers):
        if dim < 2:
            ax.axvline(end + 0.5, color=BLACK, linewidth=0.8, zorder=4)
        center = (start + end) / 2
        left_edge, right_edge = boundaries[group_idx], boundaries[group_idx + 1]
        ax_bottom.annotate("", xy=(left_edge + 0.08, label_y),
                           xytext=(center - 0.58, label_y),
                           arrowprops=dict(arrowstyle="->", color="#555555",
                                           linewidth=0.5))
        ax_bottom.annotate("", xy=(right_edge - 0.08, label_y),
                           xytext=(center + 0.58, label_y),
                           arrowprops=dict(arrowstyle="->", color="#555555",
                                           linewidth=0.5))
        group_label = rf"$\mathrm{{d}}x_{{{dim + 1}}}/\mathrm{{d}}t$"
        ax_bottom.text(center, label_y, group_label,
                       ha="center", va="center", fontsize=7.5,
                       color="#444444")
    cbar_x = x0 + 0.965 * w - TWO_PT_FIG_X
    cbar_w = 0.014 * w
    cax = fig.add_axes([cbar_x, y0, cbar_w, h])
    cb = fig.colorbar(im, cax=cax, ticks=[0, 0.5, 1])
    style_fig2_colorbar(cb, "")
    fig.text(cbar_x + cbar_w / 2 + TWO_PT_FIG_X, tick_row_y, "Freq.",
             ha="center", va="center", fontsize=6.5,
             fontfamily="Arial", fontweight="normal", color=BLACK)


def main():
    # 183 mm x 145 mm: compact double-column layout with a strict 5 pt floor.
    fig = plt.figure(figsize=(7.204724, 5.708661), facecolor="white")
    # Keep the original three-row frame structure while applying the shared
    # typography, palette and title hierarchy.
    frames = {
        "a": (0.015, 0.652, 0.475, 0.270),
        "b": (0.505, 0.652, 0.480, 0.270),
        "c": (0.015, 0.375, 0.295, 0.270),
        "d": (0.325 - SEVEN_PT_FIG_X, 0.375, 0.375, 0.270),
        "e": (0.715, 0.375, 0.270, 0.270),
        "f": (0.015, 0.095, 0.240, 0.270),
        "g": (0.265, 0.095, 0.205, 0.270),
        "h": (0.480 - ONE_PT_FIG_X, 0.095, 0.505, 0.270),
    }
    for row_rect in [
        (0.009, 0.652, 0.988, 0.270),
        (0.009, 0.375, 0.988, 0.270),
        (0.009, 0.095, 0.988, 0.270),
    ]:
        draw_row_frame(fig, row_rect)
    # Row 1: A--B; Row 2: C--D--E; Row 3: F--G--H.
    # Keep the enclosing frame fixed while lifting the complete first-row
    # content by 3 pt, including titles, legends, axes and tick labels.
    panel_title(fig, frames["a"], "a", "Library pruning")
    # Match the compact row height below by shortening only the A/B plot
    # areas; their headers and all plotted content are retained.
    draw_panel_a(fig, [0.070, 0.700 + THREE_PT_FIG_Y, 0.405, 0.150])
    b_axes = draw_panel_b(fig, [0.560, 0.700 + THREE_PT_FIG_Y, 0.400, 0.150])
    fig.canvas.draw()
    b_tick_bbox = b_axes[0].get_yticklabels()[0].get_window_extent(fig.canvas.get_renderer())
    b_letter_x = fig.transFigure.inverted().transform((b_tick_bbox.x0, 0))[0]
    panel_title(fig, frames["b"], "b", "Width robustness", letter_x=b_letter_x)

    panel_title(fig, frames["c"], "c", "DBSCAN sensitivity")
    panel_title(fig, frames["d"], "d", "Consensus comparison")
    panel_title(fig, frames["e"], "e", "Rollout sensitivity")
    draw_panel_c(fig, [0.055, 0.425, 0.260, 0.155])
    draw_panel_d(fig, [0.360 - SEVEN_PT_FIG_X, 0.425, 0.340, 0.155])
    draw_panel_e(fig, [0.720, 0.425, 0.265, 0.155])

    panel_title(fig, frames["f"], "f", "F1 gain", y_offset=TWO_PT_FIG_Y,
                letter_y_offset=THREE_PT_FIG_Y)
    panel_title(fig, frames["g"], "g", "Support refinement", y_offset=TWO_PT_FIG_Y,
                letter_x=frames["g"][0] + 0.014 - (5.0 / 72.0) / FIG_WIDTH_IN,
                letter_y_offset=THREE_PT_FIG_Y)
    panel_title(fig, frames["h"], "h", "Rössler recovery", y_offset=TWO_PT_FIG_Y,
                letter_y_offset=THREE_PT_FIG_Y)
    # The third row uses the same compact frame height as the second row;
    # only the f/g/h plotting areas are compressed, not their title text.
    tick_row_y = draw_panel_f(fig, [0.055, 0.130, 0.205, 0.180])
    draw_panel_g(fig, [0.270, 0.130, 0.195, 0.180])
    draw_panel_h(fig, [0.485 - ONE_PT_FIG_X, 0.130, 0.505, 0.180], tick_row_y)

    # Preserve the declared 183 mm × 145 mm physical canvas exactly.  Tight
    # bounding boxes make font sizes and alignment drift between export types.
    fig.savefig(OUT / "Fig3_phase1_phase2.png", dpi=600)
    fig.savefig(OUT / "Fig3_phase1_phase2.tiff", dpi=600)
    fig.savefig(OUT / "Fig3_phase1_phase2.pdf")
    fig.savefig(OUT / "Fig3_phase1_phase2.svg")
    plt.close(fig)

    manifest = {
        "backend": "Python / matplotlib",
        "canvas_mm": [183, 145],
        "panels": "A-H",
        "plot_data_only": True,
        "sources": {
            "E1_E2": "plot_data/panel_a.csv, panel_b.csv, panel_f.csv, panel_g.csv",
            "C": "plot_data/panel_c.csv",
            "D": "plot_data/panel_d.csv",
            "E": "plot_data/panel_e.csv",
            "H": "plot_data/panel_h.csv",
        },
        "exports": ["Fig3_phase1_phase2.pdf", "Fig3_phase1_phase2.svg", "Fig3_phase1_phase2.tiff", "Fig3_phase1_phase2.png"],
        "notes": [
            "C uses the fixed 24-cell D1 snapshot stored in panel_c.csv.",
            "C cell labels show absolute F1 while the diverging fill shows delta F1 versus the nominal setting.",
            "E uses one logarithmic axis because Euler rollout errors span approximately four orders of magnitude.",
            "F uses the L2 system-condition F1 summary in panel_f.csv.",
            "Display terminology uses Clean, Noise and Sparse while archived condition keys and filenames remain unchanged.",
            "The visual system matches the other main figures: Times-style publication typography, muted blue headings, pastel blue heatmaps and centered headers, while retaining the original three-row rounded black frame structure. All plotted data, panel content, scales and labels are preserved. H formulas and differential labels use upright Times New Roman; equal-size Unicode script characters prevent export-time contraction. All heatmap annotations use black text, and redundant F Condition and G Recovery score footer titles are omitted.",
        ],
    }
    (OUT / "Fig3_phase1_phase2_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
