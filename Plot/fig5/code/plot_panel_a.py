"""Draw panel a of the FHN prediction figure.

The panel is a schematic-led composite. Quantitative values are used only for
the two-node trajectory vignette; the brain-network marks are deliberately
schematic and are generated with NetworkX.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Ellipse
import networkx as nx
import numpy as np
from PIL import Image


CODE_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = CODE_ROOT.parent
DATA = PACKAGE_ROOT / "plot_data" / "trajectory_snr_30.csv"
BRAIN = CODE_ROOT / "assets" / "brain_outline_lateral.png"
OUT = CODE_ROOT / "output" / "fig5a_redraw"

# Brain outline source: Wikimedia Commons, File:Brain-outline-lateral.svg,
# https://commons.wikimedia.org/wiki/File:Brain-outline-lateral.svg
# CC0 1.0 Universal Public Domain Dedication.  The network marks below are
# schematic and are not a rendering of all 22,198 experimental nodes.


mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        # Kept only as an unused fallback for any explicitly sans-serif mark;
        # all figure text uses the serif Times New Roman family above.
        "font.sans-serif": ["Arial"],
        "mathtext.fontset": "stix",
        "font.size": 7.2,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.7,
    }
)


NAVY = "#163B70"
BLUE = "#2D83B8"
BLUE_DARK = "#1F5D91"
ORANGE = "#E9892E"
PURPLE = "#7051A5"
GREY = "#778393"
LIGHT_GREY = "#C9D2DC"
PALE_BLUE = "#EEF6FB"
PALE_ORANGE = "#FFF3E7"


def add_round_box(ax, xy, width, height, *, facecolor="white", edgecolor=LIGHT_GREY,
                  linewidth=0.9, radius=0.035, linestyle="-"):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        transform=ax.transAxes,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        clip_on=False,
    )
    ax.add_patch(patch)
    return patch


def draw_arrow(ax, start, end, *, color=GREY, lw=1.1, mutation_scale=11):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=lw,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def brain_rgba(path: Path) -> np.ndarray:
    """Convert the downloaded grayscale+alpha outline into a soft blue-gray RGBA image."""
    image = Image.open(path).convert("LA")
    # Converting the PIL image directly preserves the two-dimensional raster;

    pixels = np.asarray(image, dtype=float)
    alpha_array = pixels[..., 1] / 255.0
    rgba = np.empty((*alpha_array.shape, 4), dtype=float)
    rgba[..., 0] = 0.38
    rgba[..., 1] = 0.46
    rgba[..., 2] = 0.56
    rgba[..., 3] = alpha_array * 0.42
    return rgba


def draw_network(ax) -> None:
    """Show a selected local NetworkX subgraph over a faint brain outline."""
    # Stretch the lateral brain outline vertically so it reads as a brain
    # silhouette rather than a flattened horizontal strip.
    ax.imshow(brain_rgba(BRAIN), extent=(-0.02, 1.02, -0.16, 1.16), origin="upper", aspect="auto")

    # A sparse whole-brain context network.  NetworkX creates the topology,

    # across the full silhouette instead of collapsing into the selected area.
    whole_graph = nx.gnm_random_graph(42, 52, seed=4)
    brain_grid = np.array([
        [0.27, 0.22], [0.43, 0.22], [0.59, 0.22], [0.75, 0.22],
        [0.18, 0.31], [0.31, 0.31], [0.44, 0.31], [0.57, 0.31],
        [0.70, 0.31], [0.83, 0.31],
        [0.13, 0.40], [0.26, 0.40], [0.39, 0.40], [0.52, 0.40],
        [0.65, 0.40], [0.78, 0.40], [0.89, 0.40],
        [0.12, 0.49], [0.25, 0.49], [0.38, 0.49], [0.51, 0.49],
        [0.64, 0.49], [0.77, 0.49], [0.88, 0.49],
        [0.14, 0.58], [0.27, 0.58], [0.40, 0.58], [0.53, 0.58],
        [0.66, 0.58], [0.79, 0.58], [0.88, 0.58],
        [0.18, 0.67], [0.32, 0.67], [0.46, 0.67], [0.60, 0.67],
        [0.74, 0.67], [0.84, 0.67],
        [0.24, 0.76], [0.39, 0.76], [0.54, 0.76], [0.68, 0.76],
        [0.48, 0.84],
    ])
    jitter = np.random.default_rng(4).normal(0.0, 0.010, brain_grid.shape)
    brain_points = brain_grid + jitter
    # Rotate the complete network clockwise and lift it slightly so its
    # footprint follows the lateral brain silhouette more naturally.
    theta = np.deg2rad(-15.0)
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta),  np.cos(theta)]])
    rotation_center = np.array([0.50, 0.50])
    brain_points = (brain_points - rotation_center) @ rotation.T + rotation_center
    # Add a precise 5 pt vertical lift in the final rendered figure.
    lift_5pt = 5.0 / (72.0 * ax.figure.get_figheight() * ax.get_position().height)
    brain_points[:, 1] += 0.035 + lift_5pt
    brain_points = np.clip(brain_points, [0.10, 0.16], [0.92, 0.90])
    whole_pos = {node: tuple(point) for node, point in zip(whole_graph.nodes, brain_points)}
    nx.draw_networkx_edges(
        whole_graph, whole_pos, ax=ax, edge_color="#5C7188", width=0.65,
        alpha=0.78, arrows=False
    )

    # The dashed ellipse is visually circular in the final panel and marks
    # the local subnetwork used for the forecasting example.
    cx, cy = 0.47, 0.51 + lift_5pt
    rx, ry = 0.19, 0.24
    selected = [node for node, (x, y) in whole_pos.items()
                if ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0]
    selected_graph = whole_graph.subgraph(selected)
    nx.draw_networkx_nodes(
        whole_graph, whole_pos, ax=ax, node_color="#4E79A7", edgecolors="white",
        node_size=15, linewidths=0.45, alpha=0.96
    )
    nx.draw_networkx_nodes(
        selected_graph, whole_pos, ax=ax, node_color=ORANGE, edgecolors="white",
        node_size=20, linewidths=0.55, alpha=1.0
    )
    ax.add_patch(
        Ellipse((cx, cy), 2 * rx, 2 * ry, transform=ax.transAxes,
                fill=False, edgecolor="#111111", linewidth=0.9,
                linestyle=(0, (3, 2)), clip_on=False)
    )
    ax.text(0.06, -0.075, "N = 22,198 nodes  ·  2D FHN Dynamics",
            transform=ax.transAxes, color=NAVY, fontsize=6.8, ha="left",
            clip_on=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def draw_trajectory(ax, time, values, series_label, *, show_xlabel=False,
                    show_split_labels=False) -> None:
    boundary = time[0] + 0.8 * (time[-1] - time[0])

    ax.set_facecolor("#FFFFFF")
    ax.axvspan(time[0], boundary, color=PALE_BLUE, zorder=0)
    ax.axvspan(boundary, time[-1], color=PALE_ORANGE, zorder=0)
    ax.plot(time, values, color=ORANGE, lw=0.8, zorder=2)
    ax.axvline(boundary, color=PURPLE, lw=0.9, ls=(0, (3, 2)), zorder=3)
    if show_split_labels:
        ax.text(0.39, 0.975, "TRAIN 80%", transform=ax.transAxes, color=BLUE_DARK,
                fontsize=6.6, ha="center", va="top", fontweight="bold")
        ax.text(0.90, 0.975, "TEST 20%", transform=ax.transAxes, color=PURPLE,
                fontsize=6.6, ha="center", va="top", fontweight="bold")
    if show_xlabel:
        ax.set_xlabel("Time step", labelpad=0.5, fontsize=6.6)
        shift_down_2pt = 2.0 / (72.0 * ax.figure.get_figheight() * ax.get_position().height)
        ax.xaxis.set_label_coords(0.5, -0.14 - shift_down_2pt)
    else:
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax.set_ylabel(series_label, labelpad=0.5, fontsize=7.2)
    ax.tick_params(length=2.2, width=0.55, pad=1.2, labelsize=6.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E4EAF0", linewidth=0.45)


def draw_workflow(ax) -> None:
    ax.axis("off")
    title_lift = 5.0 / (72.0 * ax.figure.get_figheight() * ax.get_position().height)
    shift_down_3pt = 3.0 / (72.0 * ax.figure.get_figheight() * ax.get_position().height)
    label_inset_2pt = 2.0 / (72.0 * ax.figure.get_figheight() * ax.get_position().height)
    shift_left_10pt = 10.0 / (72.0 * ax.figure.get_figwidth() * ax.get_position().width)
    ax.text(0.50, 1.055 + title_lift, "Equation-based forecasting", transform=ax.transAxes,
            ha="center", va="top", color="#1F5E9E", fontsize=8.6,
            fontweight="normal", fontfamily="Arial")

    # 2 x 2 workflow: governing equations -> initial states -> Euler step -> rollout.
    box_w, box_h = 0.42, 0.35
    left_x, right_x = 0.02, 0.56
    top_y = 0.65 - shift_down_3pt
    bottom_y = 0.00
    left_center, right_center = left_x + box_w / 2, right_x + box_w / 2
    top_label_y = top_y + label_inset_2pt
    bottom_label_y = bottom_y + label_inset_2pt

    add_round_box(ax, (left_x, top_y), box_w, box_h, facecolor="#F7FAFD", edgecolor="#AFC1D5")
    equation_y = 0.84 - shift_down_3pt
    ax.text(0.13 - shift_left_10pt, equation_y, r"$\frac{\mathrm{d}x_i}{\mathrm{d}t}$",
            transform=ax.transAxes, color=BLUE_DARK, fontsize=10.5,
            ha="center", va="center")
    ax.text(0.31 - shift_left_10pt, equation_y, r"$=f(x_i)+\sum_j c(x_i,x_j)$",
            transform=ax.transAxes, color=BLUE_DARK, fontsize=7.4,
            ha="center", va="center")
    ax.text(left_center, top_label_y, "Governing equations", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=6.1, color=NAVY, fontweight="bold")

    # Initial states, top-right. The box is deliberately the same size as the
    # rollout box so both state representations have equal visual weight. Only
    # horizontal state markers are shown; no network is implied here.
    add_round_box(ax, (right_x, top_y), box_w, box_h, facecolor="#F4F7FA", edgecolor="#AFC1D5")
    initial_x = np.linspace(0.63, 0.91, 5)
    lift_5pt = 5.0 / (72.0 * ax.figure.get_figheight() * ax.get_position().height)
    ax.scatter(initial_x, np.full_like(initial_x, 0.86 + lift_5pt - shift_down_3pt), transform=ax.transAxes,
               s=62, color="#DCE2E8", edgecolors=BLUE_DARK,
               linewidths=0.8, zorder=2)
    ax.text(0.77, 0.76 + lift_5pt - shift_down_3pt,
            r"$y_i(0)\quad y_j(0)\quad \cdots\quad y_N(0)$",
            transform=ax.transAxes, color=NAVY, fontsize=7.4,
            ha="center", va="center", zorder=3)
    ax.text(right_center, top_label_y, "Initial states", transform=ax.transAxes,
            ha="center", va="bottom", color=NAVY, fontsize=6.1, fontweight="bold")

    # Rollout prediction vignette, bottom-left.
    add_round_box(ax, (left_x, bottom_y), box_w, box_h, facecolor="#FFF9F3", edgecolor="#E1B78C")
    rollout_x = np.linspace(0.07, 0.39, 90)
    rollout_y = 0.18 + 0.085 * np.sin(np.linspace(0, 3.7 * np.pi, 90)) * np.exp(-np.linspace(0, 1, 90) * 0.25)
    ax.add_patch(
        FancyArrowPatch((0.07, 0.08), (0.07, 0.30), transform=ax.transAxes,
                        arrowstyle="-|>", mutation_scale=7, linewidth=0.65,
                        color=GREY, shrinkA=0, shrinkB=0, zorder=1)
    )
    ax.add_patch(
        FancyArrowPatch((0.07, 0.08), (0.40, 0.08), transform=ax.transAxes,
                        arrowstyle="-|>", mutation_scale=7, linewidth=0.65,
                        color=GREY, shrinkA=0, shrinkB=0, zorder=1)
    )
    ax.plot(rollout_x, rollout_y, transform=ax.transAxes, color=ORANGE, lw=1.0)
    ax.scatter(rollout_x[::14], rollout_y[::14], transform=ax.transAxes,
               s=9, color=PURPLE, edgecolors="white", linewidths=0.35, zorder=3)
    ax.text(left_center, bottom_label_y, "Rollout", transform=ax.transAxes,
            ha="center", va="bottom", color=NAVY, fontsize=6.4, fontweight="bold")

    # Euler estimate, bottom-right. Mathtext is used so the update rule is
    # compiled as a proper equation rather than rendered as literal ASCII.
    add_round_box(ax, (right_x, bottom_y), box_w, box_h, facecolor="#F8F4FD", edgecolor="#B6A6D0")
    ax.text(0.77, 0.205, r"$\hat{y}_{t+1}=y_t+\Delta t\,f(y_t)$",
            transform=ax.transAxes, ha="center", va="center", color=PURPLE, fontsize=7.4)
    ax.text(right_center, bottom_label_y, "Euler step", transform=ax.transAxes,
            ha="center", va="bottom", color=NAVY, fontsize=6.4, fontweight="bold")

    top_arrow_y = 0.84 - shift_down_3pt
    draw_arrow(ax, (left_x + box_w + 0.01, top_arrow_y), (right_x - 0.01, top_arrow_y))
    draw_arrow(ax, (right_center, top_y - 0.02), (right_center, bottom_y + box_h + 0.02))
    draw_arrow(ax, (left_x + box_w + 0.01, bottom_y + box_h / 2),
               (right_x - 0.01, bottom_y + box_h / 2))


def main() -> None:
    if not DATA.exists():
        raise FileNotFoundError(DATA)
    if not BRAIN.exists():
        raise FileNotFoundError(BRAIN)

    fig = plt.figure(figsize=(7.2, 1.90), facecolor="white")
    # Keep the export footprint stable, but leave the visible row frame to the
    # final assembly so all three rows use one renderer and one style.
    fig.patches.append(
        FancyBboxPatch((0.0052, -0.036), 0.9838, 0.997,
                       transform=fig.transFigure,
                       boxstyle="round,pad=0.006,rounding_size=0.012",
                       facecolor="none", edgecolor="none", linewidth=0.0, zorder=-10)
    )
    grid = fig.add_gridspec(1, 3, left=0.025, right=0.985, bottom=0.10, top=0.84,
                            width_ratios=[1.2, 1.2, 1.6], wspace=0.18)
    ax_network = fig.add_subplot(grid[0, 0])
    traj_grid = grid[0, 1].subgridspec(2, 1, hspace=0.08)
    ax_traj_top = fig.add_subplot(traj_grid[0, 0])
    ax_traj_bottom = fig.add_subplot(traj_grid[1, 0], sharex=ax_traj_top)
    ax_workflow = fig.add_subplot(grid[0, 2])
    title_lift = 5.0 / (72.0 * fig.get_figheight() * ax_network.get_position().height)
    title_y = 1.055 + title_lift
    ax_network.text(-0.02, title_y, "a", transform=ax_network.transAxes,
                    fontsize=9.5, fontweight="bold", fontfamily="Arial",
                    color="#1C1C1C", va="top")
    ax_network.text(0.07, title_y, "Prediction Setting", transform=ax_network.transAxes,
                    fontsize=8.6, fontweight="normal", fontfamily="Arial",
                    color="#1F5E9E", va="top")
    # The trajectory column is split into two axes, so its local top is lower
    # than the full-height network/workflow axes. Place the title in figure
    # coordinates to align its top edge with the other two panel titles.
    title_global_y = fig.transFigure.inverted().transform(
        ax_network.transAxes.transform((0.0, title_y))
    )[1]
    traj_position = ax_traj_top.get_position()
    fig.text((traj_position.x0 + traj_position.x1) / 2.0, title_global_y,
             "Noisy observations", fontsize=8.6, fontweight="normal", fontfamily="Arial",
             color="#1F5E9E", ha="center", va="top")
    draw_network(ax_network)
    data = np.genfromtxt(DATA, delimiter=",", names=True)
    time = np.asarray(data["time_step"], dtype=float)
    draw_trajectory(ax_traj_top, time, np.asarray(data["true_dim1"], dtype=float),
                    r"$x_1$", show_split_labels=True)
    draw_trajectory(ax_traj_bottom, time, np.asarray(data["true_dim2"], dtype=float),
                    r"$x_2$", show_xlabel=True)
    draw_workflow(ax_workflow)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
