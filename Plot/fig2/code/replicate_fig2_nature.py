"""Recreate Fig. 2 from the portable submission data package.

The script is intentionally self-contained: it reads only CSV matrices from
the sibling ``plot_data`` directory and writes the publication exports next to
the three package directories.  All appearance controls and displayed
trajectory indices remain in the DISPLAY_SELECTIONS block below.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from tempfile import gettempdir

# Keep Matplotlib/TeX caches outside the submission tree so running the script
# never adds generated files to ``code/``.
os.environ.setdefault("MPLCONFIGDIR", str(Path(gettempdir()) / "fig2_mplconfig"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import ScaledTranslation


STYLE = {
    "font": "Times New Roman",
    "title_color": "#1F5E9E",
    # a-c: one highlighted True/Inferred pair plus four quiet background pairs.
    "primary_overlay_position": 1,
    "overlay_true_color": "#9ECAE1",
    "overlay_inferred_color": "#F2C14E",
    "overlay_true_lw": 1.18,
    "overlay_inferred_lw": 1.08,
    "overlay_inferred_linestyle": (0, (3.0, 4.8)),
    "background_true_color": "#9A9A9A",
    "background_inferred_color": "#C1C1C1",
    "background_lw": 0.62,
    "background_inferred_linestyle": (0, (2.2, 3.6)),
    # d-e: True/Inferred occupy separate axes, so color identifies the node.
    "phase_true_colors": ["#B7D7E8", "#8DBBD5", "#6BAED6", "#2F78A8", "#084A7C"],
    "phase_inferred_colors": ["#F8DB94", "#F4CC6A", "#F2C14E", "#E69F00", "#C97A00"],
    "trajectory_lw_scales": [1.55, 1.25, 1.00],
    # d-e remain solid because True/Inferred occupy separate labelled axes.
    "phase_lw_scales": [1.45, 1.32, 1.20, 1.10, 1.00],
    "phase_alphas": [0.34, 0.42, 0.50, 0.59, 0.68],
    "heat_grid_color": "#919B98",
    "panel_color": "#201B1B",
    "curve_lw": 1.12,          # a-c trajectory lines, pt
    "phase_lw": 1.10,          # d-e phase-space lines, pt
    "legend_lw": 0.80,         # a-c legend samples, pt
    "phase_legend_lw": 0.80,   # d-e legend samples; matched to a-c, pt
    "phase_frame_color": "#111111",
    "phase_frame_lw": 0.48,    # six outer 3-D frame edges, pt
    "phase_axis_lw": 0.58,     # x/y/z coordinate axes, pt
    "axis_lw": 0.42,
    "heat_grid_lw": 0.28,
    "panel_lw": 0.68,
    "arrow_lw": 0.48,
    "panel_radius_cm": 0.40,
    "tick_fs": 6.5,
    "term_fs": 7.2,
    "label_fs": 7.5,
    "legend_fs": 7.0,
    "panel_fs": 9.5,
}

# MATLAB-style 1-based indices; a-c show five trajectories with one highlighted pair.
IDX = {
    "kuramoto": [1, 4, 7, 2, 9],
    "sis": [1, 4, 7, 2, 9],
    "mm": [1, 4, 7, 2, 5],
    "fhn": [1, 5, 9, 8, 7],
    "hr": [1, 5, 9, 2, 7],
}

FIG_W_CM = 18.9                # 90% of A4 width
FIG_H_CM = 12.1
PT_TO_FIG_X = 2.54 / (72.0 * FIG_W_CM)
PT_TO_FIG_Y = 2.54 / (72.0 * FIG_H_CM)
E_AXIS_WIDTH_REDUCTION = 0.3 / FIG_W_CM
E_AXIS_CENTER_COMPENSATION = 0.15 / FIG_W_CM
OUTPUT_STEM = "Fig2_reproduced"
PNG_DPI = 600
TIFF_DPI = 600
# Fractional padding added to panel-e x/y/z limits so trajectories clear the frame.
E_RANGE_PADDING_FRACTION = (0.04, 0.04, 0.05)
FRAME_JOIN_EXTENSION_FRACTION = 0.006

# Normalized figure coordinates.  Change these to tune panel proportions.
PANEL_BOXES = {
    "a": (0.004, 0.690, 0.293, 0.292),
    "b": (0.305, 0.690, 0.350, 0.292),
    "c": (0.663, 0.690, 0.325, 0.292),
    "d": (0.006, 0.350, 0.982, 0.330),
    "e": (0.006, 0.020, 0.982, 0.320),
}

AXPOS = {
    "a_heat": (0.066, 0.720, 0.052, 0.175),
    "a_cbar": (0.122, 0.720, 0.007, 0.175),
    "a_traj": (0.170, 0.755, 0.120, 0.140),
    "b_heat": (0.365, 0.720, 0.075, 0.175),
    "b_cbar": (0.445, 0.720, 0.007, 0.175),
    "b_traj": (0.505, 0.755, 0.142, 0.140),
    "c_heat": (0.735, 0.720, 0.050, 0.175),
    "c_cbar": (0.790, 0.720, 0.007, 0.175),
    "c_traj": (0.850, 0.755, 0.130, 0.140),
    "d_heat": (0.084, 0.418, 0.394, 0.170),
    "d_cbar": (0.488, 0.418, 0.010, 0.170),
    "d_true": (0.565, 0.412, 0.175, 0.185),
    "d_inferred": (0.785, 0.412, 0.175, 0.185),
    "e_heat": (0.083, 0.092, 0.425, 0.174),
    "e_cbar": (0.517, 0.092, 0.010, 0.174),
    # Compress each e-panel 3-D axis by 0.3 cm while preserving its center.
    "e_true": (
        0.562 + 3 * PT_TO_FIG_X + E_AXIS_CENTER_COMPENSATION,
        0.075 + 3 * PT_TO_FIG_Y,
        0.165 - E_AXIS_WIDTH_REDUCTION,
        0.205,
    ),
    "e_inferred": (
        0.780 + 3 * PT_TO_FIG_X + E_AXIS_CENTER_COMPENSATION,
        0.075 + 3 * PT_TO_FIG_Y,
        0.165 - E_AXIS_WIDTH_REDUCTION,
        0.205,
    ),
}


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PLOT_DATA = PACKAGE_ROOT / "plot_data"
OUTDIR = PACKAGE_ROOT


def configure_matplotlib() -> None:
    # Literal assignments are retained for deterministic publication preflight.
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    mpl.rcParams["font.size"] = 7.5
    mpl.rcParams["text.usetex"] = False
    mpl.rcParams["text.latex.preamble"] = r"\usepackage{newtxtext,newtxmath}"
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [STYLE["font"], "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "mathtext.rm": STYLE["font"],
            "mathtext.it": f"{STYLE['font']}:italic",
            "mathtext.bf": f"{STYLE['font']}:bold",
            "font.size": STYLE["label_fs"],
            "axes.labelsize": STYLE["label_fs"],
            "axes.titlesize": STYLE["label_fs"],
            "xtick.labelsize": STYLE["tick_fs"],
            "ytick.labelsize": STYLE["tick_fs"],
            "axes.linewidth": STYLE["axis_lw"],
            "xtick.major.width": STYLE["axis_lw"],
            "ytick.major.width": STYLE["axis_lw"],
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def load_csv(path: Path) -> np.ndarray:
    values = np.genfromtxt(path, delimiter=",")
    if values.ndim != 2 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"Expected a non-empty numeric matrix in {path}")
    return values


def rounded_rect_path(box: tuple[float, float, float, float]) -> MplPath:
    x, y, w, h = box
    rx = min(STYLE["panel_radius_cm"] / FIG_W_CM, w / 2)
    ry = min(STYLE["panel_radius_cm"] / FIG_H_CM, h / 2)
    k = 0.5522847498307936
    vertices = [
        (x + rx, y), (x + w - rx, y),
        (x + w - rx + k * rx, y), (x + w, y + ry - k * ry), (x + w, y + ry),
        (x + w, y + h - ry),
        (x + w, y + h - ry + k * ry), (x + w - rx + k * rx, y + h), (x + w - rx, y + h),
        (x + rx, y + h),
        (x + rx - k * rx, y + h), (x, y + h - ry + k * ry), (x, y + h - ry),
        (x, y + ry),
        (x, y + ry - k * ry), (x + rx - k * rx, y), (x + rx, y),
        (x + rx, y),
    ]
    codes = [
        MplPath.MOVETO, MplPath.LINETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    return MplPath(vertices, codes)


def draw_panel_frames(fig: plt.Figure) -> None:
    # The three top-row subpanels form one logical comparison block.  Their
    # individual headers remain unchanged, while a single rounded frame makes
    # the shared a–c grouping explicit.
    top_a = PANEL_BOXES["a"]
    top_c = PANEL_BOXES["c"]
    frame_boxes = (
        (top_a[0], top_a[1], top_c[0] + top_c[2] - top_a[0], top_a[3]),
        PANEL_BOXES["d"],
        PANEL_BOXES["e"],
    )
    for box in frame_boxes:
        patch = PathPatch(
            rounded_rect_path(box), transform=fig.transFigure, facecolor="none",
            edgecolor=STYLE["panel_color"], linewidth=STYLE["panel_lw"],
            capstyle="round", joinstyle="round", clip_on=False, zorder=0,
        )
        fig.add_artist(patch)


def figure_text(fig: plt.Figure, x: float, y: float, text: str, **kwargs) -> None:
    defaults = dict(fontfamily=STYLE["font"], fontsize=STYLE["label_fs"], color="#1C1C1C")
    defaults.update(kwargs)
    fig.text(x, y, text, **defaults)


def add_panel_header(
    fig: plt.Figure, panel: str, title: str, title_y: float, title_fs: float
) -> None:
    panel_x, _, panel_w, _ = PANEL_BOXES[panel]
    # Panel b has the longest title in the compact top row; keep its tag at
    # the inner border so the Fig. 4-sized header remains collision-free.
    label_pos = (panel_x + (0.005 if panel == "b" else 0.020), title_y)
    title_x = panel_x + panel_w / 2
    figure_text(
        fig, *label_pos, panel, fontsize=STYLE["panel_fs"], fontweight="bold",
        color="#1C1C1C", fontfamily="Arial", ha="left", va="center",
    )
    figure_text(
        fig, title_x, title_y, title, fontsize=8.6, color=STYLE["title_color"],
        fontweight="normal", fontfamily="Arial", ha="center", va="center",
    )


def heat_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "fig2_smape",
        ["#F9F9E2", "#BED1D1", "#8CB0C2", "#5A8FB3", "#2D70A6", "#10426B"],
        N=256,
    )


def heatmap(
    fig: plt.Figure,
    pos: tuple[float, float, float, float],
    values: np.ndarray,
    vmax: float,
    row_labels: list[str],
    col_labels: list[str],
) -> tuple[plt.Axes, mpl.image.AxesImage]:
    ax = fig.add_axes(pos, zorder=2)
    image = ax.imshow(values, cmap=heat_cmap(), vmin=0, vmax=vmax, origin="upper", aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(values.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(values.shape[0]), labels=row_labels)
    ax.tick_params(
        axis="x", top=False, bottom=False, labeltop=True, labelbottom=False,
        length=0, pad=3.0, labelsize=STYLE["term_fs"],
    )
    ax.tick_params(axis="y", left=False, right=False, length=0, pad=4.0, labelsize=STYLE["label_fs"])
    ax.set_xticks(np.arange(-0.5, values.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, values.shape[0], 1), minor=True)
    ax.grid(which="minor", color=STYLE["heat_grid_color"], linewidth=STYLE["heat_grid_lw"])
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_linewidth(STYLE["axis_lw"])
        spine.set_color("#222222")
    return ax, image


def add_colorbar(
    fig: plt.Figure,
    image: mpl.image.AxesImage,
    pos: tuple[float, float, float, float],
    ticks: list[float],
) -> None:
    cax = fig.add_axes(pos, zorder=2)
    cb = fig.colorbar(image, cax=cax, ticks=ticks)
    cb.ax.set_yticklabels([f"{v:g}%" for v in ticks], fontweight="normal")
    cb.ax.tick_params(labelsize=STYLE["tick_fs"], width=STYLE["axis_lw"], length=1.7, pad=1.2)
    cb.outline.set_linewidth(STYLE["axis_lw"])
    figure_text(
        fig, pos[0] - 0.002, pos[1] + pos[3] + 0.012 + 2 * PT_TO_FIG_Y, "sMAPE",
        fontsize=STYLE["legend_fs"], ha="left", va="center",
    )


def term_label(fig: plt.Figure, x: float, y: float, text: str, ha: str = "center") -> None:
    figure_text(fig, x, y, text, fontsize=STYLE["term_fs"], ha=ha, va="center")


def trajectory_linewidth(base: float, position: int, scale_key: str = "trajectory_lw_scales") -> float:
    scales = STYLE[scale_key]
    return base * scales[position % len(scales)]


def style_xy_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(STYLE["axis_lw"])
    ax.spines["bottom"].set_linewidth(STYLE["axis_lw"])
    ax.tick_params(direction="out", width=STYLE["axis_lw"], length=2.0, pad=1.5, labelsize=STYLE["tick_fs"])
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("normal")


def set_time_axis(ax: plt.Axes, n_samples: int) -> None:
    labels = np.arange(2, 11, 2)
    positions = (labels / 10.0) * (n_samples - 1)
    ax.set_xticks(positions, [str(v) for v in labels])
    ax.set_xlim(0, n_samples - 1)
    ax.set_xlabel("Time", labelpad=1.5, fontsize=STYLE["label_fs"])
    ax.text(
        0.995, 0.045, "×10²", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=STYLE["tick_fs"], clip_on=False,
    )


def plot_pairs(
    ax: plt.Axes,
    true: np.ndarray,
    inferred: np.ndarray,
    indices_1based: list[int],
    transform=lambda x: x,
) -> None:
    indices = np.asarray(indices_1based, dtype=int) - 1
    if np.any(indices < 0) or np.any(indices >= true.shape[1]):
        raise IndexError(f"Trajectory indices {indices_1based} exceed matrix width {true.shape[1]}")
    primary = int(STYLE["primary_overlay_position"]) % len(indices)
    # Draw all non-primary node pairs first, using thin gray solid/dashed lines.
    for position, idx in enumerate(indices):
        if position == primary:
            continue
        ax.plot(
            transform(true[:, idx]), color=STYLE["background_true_color"],
            linewidth=STYLE["background_lw"], linestyle="-", alpha=0.78,
            solid_capstyle="round", dash_capstyle="round", zorder=1.0 + position * 0.05,
        )
        ax.plot(
            transform(inferred[:, idx]), color=STYLE["background_inferred_color"],
            linewidth=STYLE["background_lw"], linestyle=STYLE["background_inferred_linestyle"],
            alpha=0.82, solid_capstyle="round", dash_capstyle="round", zorder=1.1 + position * 0.05,
        )
    # The selected pair is the visual focal point.
    idx = indices[primary]
    ax.plot(
            transform(true[:, idx]), color=STYLE["overlay_true_color"],
            linewidth=STYLE["overlay_true_lw"],
            linestyle="-",
            solid_capstyle="round", dash_capstyle="round", zorder=3.0,
    )
    ax.plot(
            transform(inferred[:, idx]), color=STYLE["overlay_inferred_color"],
            linewidth=STYLE["overlay_inferred_lw"],
            linestyle=STYLE["overlay_inferred_linestyle"],
            solid_capstyle="round", dash_capstyle="round", zorder=4.0,
    )
    style_xy_axes(ax)


def top_pair_legend(fig: plt.Figure, ax: plt.Axes) -> None:
    x, y, w, h = ax.get_position().bounds
    baseline = y + h + 0.012
    line_y = baseline
    add_figure_line(
        fig, x + 0.005 * w, x + 0.145 * w, line_y,
        STYLE["overlay_true_color"], STYLE["legend_lw"], linestyle="-",
    )
    figure_text(fig, x + 0.175 * w, baseline, "True", fontsize=STYLE["legend_fs"], ha="left", va="center")
    add_figure_line(
        fig, x + 0.485 * w, x + 0.625 * w, line_y,
        STYLE["overlay_inferred_color"], STYLE["legend_lw"], linestyle=STYLE["overlay_inferred_linestyle"],
    )
    figure_text(fig, x + 0.655 * w, baseline, "Inferred", fontsize=STYLE["legend_fs"], ha="left", va="center")


def corner_ylabel(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.025, 0.965, label, transform=ax.transAxes, fontsize=STYLE["label_fs"],
        ha="left", va="top", bbox=dict(facecolor="white", edgecolor="none", pad=0.35),
        clip_on=True,
    )


def add_figure_line(
    fig: plt.Figure,
    x1: float,
    x2: float,
    y: float,
    color: str,
    linewidth: float,
    linestyle: object = "-",
) -> None:
    fig.add_artist(
        Line2D([x1, x2], [y, y], transform=fig.transFigure, color=color,
               linewidth=linewidth, linestyle=linestyle, solid_capstyle="butt",
               clip_on=False, zorder=10)
    )


def range_arrow(fig: plt.Figure, x1: float, x2: float, y: float, label: str, label_y: float) -> None:
    arrow = FancyArrowPatch(
        (x1 + 0.006, y), (x2 - 0.006, y), transform=fig.transFigure,
        arrowstyle="<->", mutation_scale=5.5, linewidth=STYLE["arrow_lw"],
        color="#262626", shrinkA=0, shrinkB=0, clip_on=False,
    )
    fig.add_artist(arrow)
    for x in (x1, x2):
        fig.add_artist(
            Line2D([x, x], [y - 0.010, y + 0.010], transform=fig.transFigure,
                   color="#262626", linewidth=STYLE["arrow_lw"], clip_on=False)
        )
    figure_text(fig, (x1 + x2) / 2, label_y, label, fontsize=STYLE["label_fs"], ha="center", va="center")


def plot_family_2d(
    ax: plt.Axes,
    xdata: np.ndarray,
    ydata: np.ndarray,
    indices_1based: list[int],
    role: str,
) -> None:
    indices = np.asarray(indices_1based, dtype=int) - 1
    phase_key = "phase_true_colors" if role == "true" else "phase_inferred_colors"
    colors = [STYLE[phase_key][position % len(STYLE[phase_key])] for position in range(len(indices))]
    for position, (idx, line_color) in enumerate(zip(indices, colors)):
        ax.plot(
            xdata[:, idx], ydata[:, idx], color=line_color,
            linewidth=trajectory_linewidth(STYLE["phase_lw"], position, "phase_lw_scales"),
            linestyle="-", alpha=STYLE["phase_alphas"][position % len(STYLE["phase_alphas"])],
            solid_capstyle="round", zorder=2 + position * 0.2,
        )
    style_xy_axes(ax)
    ax.set_xlabel(r"$x_{i1}$", fontsize=STYLE["label_fs"] + 0.5, labelpad=0.5)
    ax.set_ylabel(r"$x_{i2}$", fontsize=STYLE["label_fs"] + 0.5, labelpad=1.5)


def phase_legend(
    fig: plt.Figure,
    ax: plt.Axes,
    label: str,
    y: float,
    align: str = "right",
    x_offset: float = 0.0,
) -> None:
    x, _, w, _ = ax.get_position().bounds
    x_end = x + w + x_offset
    line_length = 0.020
    text_width = 0.027 if label == "True" else 0.050
    legend_width = line_length + 0.006 + text_width
    legend_left = x + (w - legend_width) / 2 if align == "center" else x_end - legend_width
    line_end = legend_left + line_length
    text_x = line_end + 0.006
    palette = STYLE["phase_true_colors"] if label == "True" else STYLE["phase_inferred_colors"]
    legend_color = palette[len(palette) // 2]
    add_figure_line(
        fig, line_end - line_length, line_end, y,
        legend_color, STYLE["phase_legend_lw"],
        linestyle="-",
    )
    figure_text(
        fig, text_x, y, label,
        fontsize=STYLE["legend_fs"], ha="left", va="center",
    )


def clean_number(value: float, _position: int | None = None) -> str:
    if abs(value - round(value)) < 1e-8:
        return str(int(round(value)))
    return f"{value:g}"


def style_phase3d(ax: plt.Axes) -> None:
    # Preserve explicit artist ordering so the selected nine frame edges stay
    # visible instead of being hidden by mplot3d's automatic depth sorting.
    ax.computed_zorder = False
    # Continue the selected clockwise rotation: -58 -> -43 -> -33 degrees.
    ax.view_init(elev=24, azim=-33)
    ax.set_box_aspect((1.45, 1.05, 0.82), zoom=1.20)
    ax.set_xlabel(r"$x_{i1}$", fontsize=STYLE["label_fs"] + 0.5, labelpad=-9.0)
    ax.set_ylabel(r"$x_{i2}$", fontsize=STYLE["label_fs"] + 0.5, labelpad=-9.0)
    ax.set_zlabel(r"$x_{i3}$", fontsize=STYLE["label_fs"], labelpad=-11.0)
    ax.tick_params(
        labelsize=STYLE["tick_fs"], width=STYLE["phase_axis_lw"],
        colors=STYLE["phase_frame_color"], pad=0.0,
    )
    ax.xaxis.set_tick_params(pad=-3.0)
    ax.yaxis.set_tick_params(pad=-3.0)
    ax.zaxis.set_tick_params(pad=-3.0)
    ax.xaxis.set_major_formatter(FuncFormatter(clean_number))
    ax.yaxis.set_major_formatter(FuncFormatter(clean_number))
    ax.zaxis.set_major_formatter(FuncFormatter(clean_number))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        # Pane rectangles stay transparent; a controlled nine-edge frame is
        # drawn after the final limits have been set.
        axis.pane.set_edgecolor((1, 1, 1, 0))
        axis.pane.set_linewidth(0.0)
        axis._axinfo["grid"].update({"color": "#888888", "linewidth": 0.32, "linestyle": ":"})
        axis._axinfo["axisline"].update(
            {"color": STYLE["phase_frame_color"], "linewidth": STYLE["phase_axis_lw"]}
        )
    ax.grid(True)


def draw_3d_frame9(ax: plt.Axes) -> None:
    """Draw nine black data-cuboid edges without crossing the trajectories."""
    x0, x1 = ax.get_xlim3d()
    y0, y1 = ax.get_ylim3d()
    z0, z1 = ax.get_zlim3d()
    axis_edges = [
        ((x0, y0, z0), (x1, y0, z0)),
        ((x0, y0, z0), (x0, y1, z0)),
        ((x0, y0, z0), (x0, y0, z1)),
    ]
    outer_edges = [
        ((x0, y1, z0), (x1, y1, z0)),
        ((x1, y0, z0), (x1, y1, z0)),
        ((x0, y1, z0), (x0, y1, z1)),
        ((x1, y1, z0), (x1, y1, z1)),
        # This left outer top edge stays clear of the displayed attractors.
        ((x0, y0, z1), (x0, y1, z1)),
    ]
    for edges, linewidth in (
        (outer_edges, STYLE["phase_frame_lw"]),
        (axis_edges, STYLE["phase_axis_lw"]),
    ):
        for p0, p1 in edges:
            ax.plot(
                [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                color=STYLE["phase_frame_color"], linewidth=linewidth,
                # At the selected azimuth, two valid cuboid endpoints project
                # just beyond the rectangular 2-D Axes patch.  Disable that
                # patch clipping so all nine requested 3-D edges remain whole.
                solid_capstyle="round", clip_on=False, zorder=4.5,
            )

    # Slightly extend the requested y=ymax, z=zmax edge along x so its
    # projected endpoints join the adjacent frame edges without a white seam.
    join = FRAME_JOIN_EXTENSION_FRACTION * (x1 - x0)
    ax.plot(
        [x0 - join, x1 + join], [y1, y1], [z1, z1],
        color=STYLE["phase_frame_color"], linewidth=STYLE["phase_frame_lw"],
        solid_capstyle="projecting", clip_on=False, zorder=4.6,
    )


def plot_family_3d(
    ax: plt.Axes,
    xdata: np.ndarray,
    ydata: np.ndarray,
    zdata: np.ndarray,
    indices_1based: list[int],
    role: str,
) -> None:
    indices = np.asarray(indices_1based, dtype=int) - 1
    phase_key = "phase_true_colors" if role == "true" else "phase_inferred_colors"
    colors = [STYLE[phase_key][position % len(STYLE[phase_key])] for position in range(len(indices))]
    for position, (idx, line_color) in enumerate(zip(indices, colors)):
        ax.plot(
            xdata[:, idx], ydata[:, idx], zdata[:, idx], color=line_color,
            linewidth=trajectory_linewidth(STYLE["phase_lw"], position, "phase_lw_scales"),
            linestyle="-", alpha=STYLE["phase_alphas"][position % len(STYLE["phase_alphas"])],
            solid_capstyle="round", zorder=2 + position * 0.2,
        )
    style_phase3d(ax)


def build_figure() -> tuple[plt.Figure, dict[str, object]]:
    configure_matplotlib()
    fig = plt.figure(figsize=(FIG_W_CM / 2.54, FIG_H_CM / 2.54), dpi=300)
    draw_panel_frames(fig)

    add_panel_header(fig, "a", "Kuramoto Dynamics", 0.958, 8.6)
    add_panel_header(fig, "b", "Susceptible-Infected-Susceptible Dynamics", 0.958, 8.6)
    add_panel_header(fig, "c", "Michaelis-Menten Dynamics", 0.958, 8.6)
    add_panel_header(fig, "d", "Brain: FitzHugh-Nagumo Neuronal Dynamics", 0.651, 8.6)
    add_panel_header(fig, "e", "Brain: Hindmarsh-Rose Neuronal Dynamics", 0.317, 8.6)

    # a: Kuramoto dynamics
    error_a = load_csv(PLOT_DATA / "error_kuramoto.csv")
    _, image_a = heatmap(fig, AXPOS["a_heat"], error_a, 10, ["BA-1k", "BA-100k", "Github", "Caster"], ["", ""])
    add_colorbar(fig, image_a, AXPOS["a_cbar"], [0, 2, 4, 6, 8, 10])
    term_label(fig, 0.093, 0.907, r"$\sin(x_j-x_i)$", ha="right")
    term_label(fig, 0.108, 0.907, r"$C$", ha="center")

    true_a = load_csv(PLOT_DATA / "true_Kuramoto_dim_0.csv")
    pred_a = load_csv(PLOT_DATA / "pred_Kuramoto_dim_0.csv")
    ax_a = fig.add_axes(AXPOS["a_traj"], zorder=2)
    plot_pairs(ax_a, true_a, pred_a, IDX["kuramoto"], transform=np.cos)
    ax_a.set_ylim(-1.05, 1.05)
    ax_a.set_yticks([-1, 0, 1])
    set_time_axis(ax_a, true_a.shape[0])
    corner_ylabel(ax_a, r"$\mathrm{cos}(x)$")
    top_pair_legend(fig, ax_a)

    # b: SIS dynamics
    error_b = load_csv(PLOT_DATA / "error_sis.csv")
    _, image_b = heatmap(
        fig, AXPOS["b_heat"], error_b, 5,
        ["SW-1k", "SW-100k", "Voles", "Caster"],
        [r"$x_i$", r"$x_j$", r"$x_ix_j$"],
    )
    add_colorbar(fig, image_b, AXPOS["b_cbar"], [0, 1, 2, 3, 4, 5])
    true_b = load_csv(PLOT_DATA / "true_SIS_dim_0.csv")
    pred_b = load_csv(PLOT_DATA / "pred_SIS_dim_0.csv")
    ax_b = fig.add_axes(AXPOS["b_traj"], zorder=2)
    plot_pairs(ax_b, true_b, pred_b, IDX["sis"])
    ax_b.set_ylim(0, 0.5)
    ax_b.set_yticks([0, 0.25, 0.5])
    ax_b.yaxis.set_major_formatter(FuncFormatter(clean_number))
    set_time_axis(ax_b, true_b.shape[0])
    corner_ylabel(ax_b, r"$x$")
    top_pair_legend(fig, ax_b)

    # c: Michaelis-Menten dynamics
    error_c = load_csv(PLOT_DATA / "error_gene.csv")
    ax_c_heat, image_c = heatmap(
        fig, AXPOS["c_heat"], error_c, 10,
        ["SW-1k", "SW-100k", "Bn-Mouse", "Bn-Human"],
        [r"$x_i$", r"$\frac{x_j}{x_j+1}$"],
    )
    # Compile the complete fraction with the installed TeX Live engine rather
    # than assembling numerator, rule, and denominator as separate artists.
    fraction_label = ax_c_heat.get_xticklabels()[1]
    fraction_label.set_fontsize(9.2)
    # Use the installed TeX Live engine when available; retain a portable
    # mathtext fallback so the submission package still runs on clean systems.
    fraction_label.set_usetex(shutil.which("latex") is not None)
    fraction_label.set_transform(
        fraction_label.get_transform()
        + ScaledTranslation(0, -1.8 / 72, fig.dpi_scale_trans)
    )
    add_colorbar(fig, image_c, AXPOS["c_cbar"], [0, 2, 4, 6, 8, 10])
    true_c = load_csv(PLOT_DATA / "true_Gene_dim_0.csv")
    pred_c = load_csv(PLOT_DATA / "pred_Gene_dim_0.csv")
    ax_c = fig.add_axes(AXPOS["c_traj"], zorder=2)
    plot_pairs(ax_c, true_c, pred_c, IDX["mm"])
    ax_c.set_ylim(0, 160)
    ax_c.set_yticks([0, 40, 80, 120, 160])
    set_time_axis(ax_c, true_c.shape[0])
    corner_ylabel(ax_c, r"$x$")
    top_pair_legend(fig, ax_c)

    # d: FitzHugh-Nagumo dynamics
    error_d = load_csv(PLOT_DATA / "error_fhn.csv")
    labels_d = [r"$C$", r"$x_{i1}$", r"$x_{i1}^3$", r"$x_{i2}$", r"$x_{j1}-x_{i1}$", r"$C$", r"$x_{i1}$", r"$x_{i2}$"]
    _, image_d = heatmap(fig, AXPOS["d_heat"], error_d, 6, ["SW-1k", "SW-100k", "Bn-Fly", "Bn-Human"], labels_d)
    add_colorbar(fig, image_d, AXPOS["d_cbar"], [0, 1, 2, 3, 4, 5, 6])
    range_arrow(fig, 0.084, 0.330, 0.397, r"$\mathrm{d}x_1/\mathrm{d}t$", 0.378)
    range_arrow(fig, 0.330, 0.478, 0.397, r"$\mathrm{d}x_2/\mathrm{d}t$", 0.378)

    true_d1 = load_csv(PLOT_DATA / "true_Fitz_dim_0.csv")
    true_d2 = load_csv(PLOT_DATA / "true_Fitz_dim_1.csv")
    pred_d1 = load_csv(PLOT_DATA / "pred_Fitz_dim_0.csv")
    pred_d2 = load_csv(PLOT_DATA / "pred_Fitz_dim_1.csv")
    ax_d_true = fig.add_axes(AXPOS["d_true"], zorder=2)
    ax_d_pred = fig.add_axes(AXPOS["d_inferred"], zorder=2)
    plot_family_2d(ax_d_true, true_d1, true_d2, IDX["fhn"], "true")
    plot_family_2d(ax_d_pred, pred_d1, pred_d2, IDX["fhn"], "inferred")
    d_idx = np.asarray(IDX["fhn"], dtype=int) - 1
    x_lo = min(true_d1[:, d_idx].min(), pred_d1[:, d_idx].min()) - 0.2
    x_hi = max(true_d1[:, d_idx].max(), pred_d1[:, d_idx].max()) + 0.2
    y_lo = min(true_d2[:, d_idx].min(), pred_d2[:, d_idx].min()) - 0.2
    y_hi = max(true_d2[:, d_idx].max(), pred_d2[:, d_idx].max()) + 0.2
    for ax in (ax_d_true, ax_d_pred):
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.xaxis.set_major_formatter(FuncFormatter(clean_number))
        ax.yaxis.set_major_formatter(FuncFormatter(clean_number))
    d_legend_y = AXPOS["d_cbar"][1] + AXPOS["d_cbar"][3] + 0.012
    phase_legend(fig, ax_d_true, "True", d_legend_y)
    phase_legend(fig, ax_d_pred, "Inferred", d_legend_y)

    # e: Hindmarsh-Rose dynamics
    error_e = load_csv(PLOT_DATA / "error_hr.csv")
    labels_e = [
        r"$C$", r"$x_{i2}$", r"$x_{i1}^2$", r"$x_{i3}$", r"$x_{i1}^3$", r"$x_{j1}-x_{i1}$",
        r"$C$", r"$x_{i2}$", r"$x_{i1}^2$", r"$C$", r"$x_{i1}$", r"$x_{i3}$",
    ]
    _, image_e = heatmap(fig, AXPOS["e_heat"], error_e, 15, ["SW-1k", "SW-100k", "Bn-Fly", "Bn-Human"], labels_e)
    add_colorbar(fig, image_e, AXPOS["e_cbar"], [0, 5, 10, 15])
    range_arrow(fig, 0.083, 0.296, 0.076, r"$\mathrm{d}x_1/\mathrm{d}t$", 0.052)
    range_arrow(fig, 0.296, 0.402, 0.076, r"$\mathrm{d}x_2/\mathrm{d}t$", 0.052)
    range_arrow(fig, 0.402, 0.508, 0.076, r"$\mathrm{d}x_3/\mathrm{d}t$", 0.052)

    true_e1 = load_csv(PLOT_DATA / "true_HR_dim_0.csv")
    true_e2 = load_csv(PLOT_DATA / "true_HR_dim_1.csv")
    true_e3 = load_csv(PLOT_DATA / "true_HR_dim_2.csv")
    pred_e1 = load_csv(PLOT_DATA / "pred_HR_dim_0.csv")
    pred_e2 = load_csv(PLOT_DATA / "pred_HR_dim_1.csv")
    pred_e3 = load_csv(PLOT_DATA / "pred_HR_dim_2.csv")
    # Explicit panel-e stack: heatmap (z=2) < True 3-D (z=3) < Inferred 3-D (z=4).
    ax_e_true = fig.add_axes(AXPOS["e_true"], projection="3d", zorder=3)
    ax_e_pred = fig.add_axes(AXPOS["e_inferred"], projection="3d", zorder=4)
    plot_family_3d(ax_e_true, true_e1, true_e2, true_e3, IDX["hr"], "true")
    plot_family_3d(ax_e_pred, pred_e1, pred_e2, pred_e3, IDX["hr"], "inferred")
    e_idx = np.asarray(IDX["hr"], dtype=int) - 1
    e_ranges = [
        (min(true_e1[:, e_idx].min(), pred_e1[:, e_idx].min()), max(true_e1[:, e_idx].max(), pred_e1[:, e_idx].max())),
        (min(true_e2[:, e_idx].min(), pred_e2[:, e_idx].min()), max(true_e2[:, e_idx].max(), pred_e2[:, e_idx].max())),
        (min(true_e3[:, e_idx].min(), pred_e3[:, e_idx].min()), max(true_e3[:, e_idx].max(), pred_e3[:, e_idx].max())),
    ]
    padded_e_ranges = [
        (lo - (hi - lo) * pad, hi + (hi - lo) * pad)
        for (lo, hi), pad in zip(e_ranges, E_RANGE_PADDING_FRACTION)
    ]
    for ax in (ax_e_true, ax_e_pred):
        ax.set_xlim(*padded_e_ranges[0])
        ax.set_ylim(*padded_e_ranges[1])
        ax.set_zlim(*padded_e_ranges[2])
        # Sparse ticks keep the compact 3-D panels readable at final size.
        x_grid = np.array([-1.0, 0.0, 1.0, 2.0])
        y_grid = np.array([-10.0, -5.0, 0.0])
        z_grid = np.array([0.3, 0.8, 1.3])
        ax.set_xticks(x_grid)
        ax.set_yticks(y_grid)
        ax.set_zticks(z_grid)
        ax.set_xticklabels([clean_number(v) for v in x_grid])
        ax.set_yticklabels([clean_number(v) for v in y_grid])
        ax.set_zticklabels([clean_number(v) for v in z_grid])
        draw_3d_frame9(ax)
    # Place status keys in the upper-right triangular whitespace beside the
    # projected top face: down/right from the title band, but
    # clear of the z=0.3 tick and lower frame.
    e_legend_y = AXPOS["e_true"][1] + 0.86 * AXPOS["e_true"][3] + 5 * PT_TO_FIG_Y
    # Axis movement already contributes +3 pt in x. Compensate the offsets so
    # the final legend movements relative to the preceding export are exactly
    # +1 pt for True and +10 pt for Inferred.
    phase_legend(
        fig, ax_e_true, "True", e_legend_y, align="right",
        x_offset=0.040 - 2 * PT_TO_FIG_X,
    )
    phase_legend(
        fig, ax_e_pred, "Inferred", e_legend_y, align="right",
        x_offset=0.035 + 7 * PT_TO_FIG_X,
    )

    data_manifest = {
        "figure_size_cm": [FIG_W_CM, FIG_H_CM],
        "data_policy": "No rows were filtered or smoothed; each displayed trajectory uses its complete time series.",
        "source_data_policy": "Empty by design: the figure is reproducible from the compact plot_data CSV package.",
        "plot_data_directory": "plot_data",
        "plot_data_files": {
            "heatmap_errors": [
                "error_kuramoto.csv", "error_sis.csv", "error_gene.csv",
                "error_fhn.csv", "error_hr.csv",
            ],
            "trajectories": [
                "true_Kuramoto_dim_0.csv", "pred_Kuramoto_dim_0.csv",
                "true_SIS_dim_0.csv", "pred_SIS_dim_0.csv",
                "true_Gene_dim_0.csv", "pred_Gene_dim_0.csv",
                "true_Fitz_dim_0.csv", "pred_Fitz_dim_0.csv",
                "true_Fitz_dim_1.csv", "pred_Fitz_dim_1.csv",
                "true_HR_dim_0.csv", "pred_HR_dim_0.csv",
                "true_HR_dim_1.csv", "pred_HR_dim_1.csv",
                "true_HR_dim_2.csv", "pred_HR_dim_2.csv",
            ],
        },
        "displayed_trajectory_indices_1based": IDX,
        "trajectory_visual_encoding": {
            "status_a_to_c": "five trajectories are shown; one True/Inferred pair is highlighted in light blue/yellow while four gray pairs form the bottom layer",
            "status_d_to_e": "both remain solid; True uses a blue family and Inferred uses a yellow family",
            "overlay_true_color": STYLE["overlay_true_color"],
            "overlay_inferred_color": STYLE["overlay_inferred_color"],
            "background_true_color": STYLE["background_true_color"],
            "background_inferred_color": STYLE["background_inferred_color"],
        },
        "source_matrix_shapes": {
            "kuramoto": list(true_a.shape), "sis": list(true_b.shape), "michaelis_menten": list(true_c.shape),
            "fhn_x1": list(true_d1.shape), "fhn_x2": list(true_d2.shape),
            "hr_x1": list(true_e1.shape), "hr_x2": list(true_e2.shape), "hr_x3": list(true_e3.shape),
        },
        "heatmap_shapes": {
            "a": list(error_a.shape), "b": list(error_b.shape), "c": list(error_c.shape),
            "d": list(error_d.shape), "e": list(error_e.shape),
        },
    }
    return fig, data_manifest


def save_figure(fig: plt.Figure, manifest: dict[str, object]) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    stem = OUTDIR / OUTPUT_STEM
    # Keep the requested physical page size: do not use bbox_inches='tight'.
    fig.savefig(stem.with_suffix(".svg"), format="svg")
    fig.savefig(stem.with_suffix(".pdf"), format="pdf")
    fig.savefig(stem.with_suffix(".png"), format="png", dpi=PNG_DPI)
    fig.savefig(stem.with_suffix(".tiff"), format="tiff", dpi=TIFF_DPI, pil_kwargs={"compression": "tiff_lzw"})
    (OUTDIR / f"{OUTPUT_STEM}_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    fig, manifest = build_figure()
    save_figure(fig, manifest)
    plt.close(fig)
    print(f"Saved publication figure files to: {OUTDIR}")


if __name__ == "__main__":
    main()
