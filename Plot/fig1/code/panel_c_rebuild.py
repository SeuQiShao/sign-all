"""Render Fig. 1c as a standalone, final-size panel.

The NetworkX topology and node coordinates are imported from the canonical
panel-a master.  Panel c changes only the role encoding by highlighting the
sampled nodes used for Phase-I local sparse regressions.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
import networkx as nx
import numpy as np

from panel_a_rebuild import build_network_master


MM = 1.0 / 25.4
WIDTH_MM = 178.608
HEIGHT_MM = 39.600

BLUE = "#2369B3"
BLUE_DARK = "#124A86"
BLUE_LIGHT = "#EEF5FB"
PURPLE = "#7650A5"
PURPLE_DARK = "#56317F"
PURPLE_LIGHT = "#F5F0F9"
GREEN = "#2C9A51"
GREEN_DARK = "#19743A"
TITLE_BLUE = "#4F8FC3"
ORANGE = "#E97824"
SAMPLE = ORANGE
SAMPLE_DARK = ORANGE
GREY = "#93AEBB"
GREY_LIGHT = "#DCE8EC"
MID = "#8D9398"
EDGE = "#4C5156"
INK = "#151719"


mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif", "serif"],
    "font.sans-serif": ["Arial", "Helvetica", "sans-serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "mathtext.fallback": "stix",
    "font.size": 5.6,
    "axes.linewidth": 0.60,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("panel_c_output"),
    )
    return parser.parse_args()


def rounded_box(
    ax: plt.Axes,
    bounds: tuple[float, float, float, float],
    *,
    edge: str = "#666B70",
    face: str = "white",
    linewidth: float = 0.82,
    radius: float = 0.015,
) -> None:
    x, y, width, height = bounds
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            width,
            height,
            transform=ax.transAxes,
            boxstyle=f"round,pad=0.004,rounding_size={radius}",
            facecolor=face,
            edgecolor=edge,
            linewidth=linewidth,
            clip_on=False,
        )
    )


def flow_arrow(ax: plt.Axes, x0: float, x1: float, y: float = 0.500) -> None:
    ax.add_patch(
        patches.FancyArrowPatch(
            (x0, y),
            (x1, y),
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=8.5,
            linewidth=1.05,
            color=GREEN,
            shrinkA=0,
            shrinkB=0,
            zorder=10,
        )
    )


def draw_sample_network(ax: plt.Axes) -> None:
    graph, pos, _, _ = build_network_master()
    sampled = [1, 5, 7, 10, 12]
    continuation = list(range(14, 19))
    ordinary = [node for node in graph if node not in sampled + continuation]
    continuation_edges = [
        edge
        for edge in graph.edges
        if edge[0] in continuation or edge[1] in continuation
    ]
    solid_edges = [edge for edge in graph.edges if edge not in continuation_edges]

    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=solid_edges,
        ax=ax,
        edge_color="#AFC2CB",
        width=0.54,
    )
    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=continuation_edges,
        ax=ax,
        edge_color="#BDCBD1",
        width=0.50,
        style=(0, (2.0, 2.0)),
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=ordinary,
        ax=ax,
        node_color=GREY_LIGHT,
        node_size=17,
        edgecolors="#718B97",
        linewidths=0.40,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=sampled,
        ax=ax,
        node_color=SAMPLE,
        node_size=28,
        edgecolors=SAMPLE_DARK,
        linewidths=0.50,
    )
    continuation_artist = nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=continuation,
        ax=ax,
        node_color="white",
        node_size=12,
        edgecolors=MID,
        linewidths=0.42,
    )
    continuation_artist.set_linestyle((0, (2.0, 1.8)))

    label_offsets = {
        # Each label is placed in the open angular sector around its node.
        1: (-0.05, 0.43, "center", "bottom"),
        5: (-0.05, -0.44, "center", "top"),
        7: (-0.48, -0.02, "right", "center"),
        10: (0.49, 0.02, "left", "center"),
        12: (0.43, -0.31, "left", "top"),
    }
    labels = [r"$S_1$", r"$S_2$", r"$S_3$", r"$S_4$", r"$S_5$"]
    for node, label in zip(sampled, labels):
        x, y = pos[node]
        dx, dy, horizontal, vertical = label_offsets[node]
        ax.text(
            x + dx,
            y + dy,
            label,
            ha=horizontal,
            va=vertical,
            fontsize=7.2,
            color=SAMPLE_DARK,
        )
    ax.set_xlim(-3.22, 3.30)
    ax.set_ylim(-3.18, 3.22)
    ax.set_aspect("equal")
    ax.axis("off")


SELF_SUPPORT = [1, 0, 0, 1, 0, 1, 0, 0, 1, 0]
COUPLING_SUPPORT = [0, 1, 0, 0, 1, 0, 0, 1, 0, 1]


def _support_pattern(name: str) -> tuple[list[int], list[int]]:
    patterns = {"self": [0] * 10, "coupling": [0] * 10}
    path = Path(__file__).resolve().parents[1] / "plot_data" / "support_patterns.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["pattern"] == name:
                patterns[row["library"]][int(row["slot"])] = int(row["active"])
    return patterns["self"], patterns["coupling"]


def varied_support(row: int) -> tuple[list[int], list[int]]:
    return _support_pattern(f"local_{row}")


def draw_binary_strip(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    self_pattern: list[int],
    coupling_pattern: list[int],
    *,
    linewidth: float = 0.44,
) -> None:
    pattern = self_pattern + coupling_pattern
    n_cells = len(pattern)
    cell_width = width / n_cells
    for index, active in enumerate(pattern):
        if active:
            face = BLUE if index < len(self_pattern) else PURPLE
        else:
            face = "white"
        ax.add_patch(
            patches.Rectangle(
                (x + index * cell_width, y),
                cell_width,
                height,
                transform=ax.transAxes,
                facecolor=face,
                edgecolor=MID,
                linewidth=linewidth,
            )
        )


def draw_sparse_regressions(ax: plt.Axes) -> None:
    x = 0.246
    width = 0.218
    height = 0.058
    row_y = [0.585, 0.465, 0.345, 0.205]
    row_labels = [r"$ξ^{(S₁)}$", r"$ξ^{(S₂)}$", r"$ξ^{(S₃)}$", r"$ξ^{(Sₛ)}$"]

    ax.text(
        x + width * 0.25,
        0.690,
        "self coefficient",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=5.10,
        color=BLUE_DARK,
        fontweight="bold",
    )
    ax.text(
        x + width * 0.75,
        0.690,
        "coupling coefficient",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=5.10,
        color=PURPLE_DARK,
        fontweight="bold",
    )
    for row, (y, label) in enumerate(zip(row_y, row_labels)):
        if row == 3:
            ax.text(
                x - 0.021,
                0.290,
                ".\n.\n.",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=5.6,
                color=EDGE,
                linespacing=0.45,
            )
        ax.text(
            x - 0.003,
            y + height / 2,
            label,
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=7.2,
            color=INK,
        )
        draw_binary_strip(ax, x, y, width, height, *varied_support(row))


def draw_dbscan_consensus(parent: plt.Axes) -> None:
    ax = parent.inset_axes([0.507, 0.145, 0.225, 0.625])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    green_points = np.array([
        [0.23, 0.58], [0.27, 0.62], [0.30, 0.54], [0.32, 0.68],
        [0.35, 0.48], [0.38, 0.59], [0.40, 0.72], [0.43, 0.51],
        [0.46, 0.64], [0.49, 0.56], [0.52, 0.70], [0.55, 0.47],
        [0.57, 0.61], [0.60, 0.54], [0.63, 0.66], [0.36, 0.39],
        [0.48, 0.40], [0.58, 0.39],
    ])
    orange_points = np.array([
        [0.74, 0.30], [0.79, 0.37], [0.84, 0.28],
        [0.88, 0.39], [0.90, 0.23], [0.80, 0.18],
    ])
    grey_points = np.array([
        [0.17, 0.82], [0.27, 0.88], [0.43, 0.88], [0.63, 0.86],
        [0.80, 0.82], [0.93, 0.70], [0.17, 0.42], [0.19, 0.22],
        [0.30, 0.16], [0.49, 0.16], [0.68, 0.16], [0.96, 0.12],
    ])
    # Remove the grey illustrative node that overlaps the direct "core cluster"
    # label; retain every green node inside the DBSCAN core cluster.
    grey_points = grey_points[~np.all(grey_points == [0.49, 0.16], axis=1)]
    ax.scatter(grey_points[:, 0], grey_points[:, 1], s=8, color=GREY, zorder=2)
    ax.scatter(green_points[:, 0], green_points[:, 1], s=9, color=GREEN, zorder=4)
    ax.scatter(orange_points[:, 0], orange_points[:, 1], s=9, color=ORANGE, zorder=4)
    ax.add_patch(
        patches.Ellipse(
            (0.44, 0.56),
            0.52,
            0.53,
            angle=-12,
            transform=ax.transAxes,
            facecolor="none",
            edgecolor=GREEN,
            linewidth=0.85,
            linestyle=(0, (4, 3)),
        )
    )
    ax.add_patch(
        patches.Ellipse(
            (0.82, 0.29),
            0.26,
            0.34,
            angle=10,
            transform=ax.transAxes,
            facecolor="none",
            edgecolor=ORANGE,
            linewidth=0.82,
            linestyle=(0, (4, 3)),
        )
    )
    ax.text(
        0.46,
        0.18,
        "core cluster",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=5.35,
        fontweight="bold",
        color=GREEN_DARK,
    )
    origin_x, origin_y = 0.08, 0.06
    ax.plot([origin_x, 0.98], [origin_y, origin_y], transform=ax.transAxes,
            color=EDGE, lw=0.68, solid_capstyle="butt", clip_on=False)
    ax.plot([origin_x, origin_x], [origin_y, 0.96], transform=ax.transAxes,
            color=EDGE, lw=0.68, solid_capstyle="butt", clip_on=False)
    ax.scatter([0.98], [origin_y], transform=ax.transAxes, marker=">", s=9,
               facecolor=EDGE, edgecolor=EDGE, linewidth=0, clip_on=False, zorder=8)
    ax.scatter([origin_x], [0.96], transform=ax.transAxes, marker="^", s=9,
               facecolor=EDGE, edgecolor=EDGE, linewidth=0, clip_on=False, zorder=8)


def draw_core_to_mask(ax: plt.Axes) -> None:
    x = 0.787
    width = 0.178
    height = 0.045
    ax.text(
        x + width / 2,
        0.705,
        "core-node regressions",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.2,
        color=INK,
        fontfamily="Arial",
        fontweight="normal",
    )
    core_rows = [_support_pattern(f"core_{row}") for row in range(3)]
    for y, (self_pattern, coupling_pattern) in zip(
        [0.590, 0.490, 0.390], core_rows
    ):
        draw_binary_strip(
            ax, x, y, width, height, self_pattern, coupling_pattern, linewidth=0.42
        )
    ax.text(
        x - 0.012,
        0.505,
        ".\n.\n.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=5.5,
        color=EDGE,
        linespacing=0.45,
    )
    arrow_x = x + width / 2
    ax.plot(
        [arrow_x, arrow_x],
        [0.365, 0.300],
        transform=ax.transAxes,
        color=INK,
        lw=0.90,
        solid_capstyle="butt",
        zorder=8,
    )
    ax.scatter(
        [arrow_x],
        [0.290],
        transform=ax.transAxes,
        marker="v",
        s=14,
        facecolor=INK,
        edgecolor=INK,
        linewidth=0,
        zorder=9,
    )
    ax.text(
        x + width / 2,
        0.245,
        "support mask",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=6.2,
        color=INK,
        fontfamily="Arial",
        fontweight="normal",
    )
    ax.text(
        x + width / 2 + 0.005,
        0.245,
        r"$M=[M_{\mathrm{f}}\,|\,M_{\mathrm{c}}]$",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=7.2,
        color=INK,
    )
    draw_binary_strip(
        ax, x, 0.140, width, 0.052, core_rows[0][0], core_rows[0][1], linewidth=0.44
    )
    ax.text(x + width * 0.25, 0.111, r"$M_{\mathrm{f}}$", transform=ax.transAxes,
            ha="center", va="top", fontsize=7.2, color=BLUE_DARK)
    ax.text(x + width * 0.75, 0.111, r"$M_{\mathrm{c}}$", transform=ax.transAxes,
            ha="center", va="top", fontsize=7.2, color=PURPLE_DARK)


def make_panel() -> plt.Figure:
    fig = plt.figure(figsize=(WIDTH_MM * MM, HEIGHT_MM * MM), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.017, 0.927, "c", transform=ax.transAxes, ha="left", va="center",
            fontsize=9.5, fontweight="bold", fontfamily="Arial", color="#1C1C1C")
    ax.text(
        0.500,
        0.927,
        "Phase I: global support identification",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.6,
        fontweight="normal",
        fontfamily="Arial",
        color="#1F5E9E",
    )

    step_titles = [
        (0.050, "Sample nodes"),
        (0.230, "Local sparse coefficients"),
        (0.515, "DBSCAN consensus"),
        (0.765, "Mask from DBSCAN core nodes"),
    ]
    for x, title in step_titles:
        ax.text(x, 0.825, title, transform=ax.transAxes, ha="left", va="center",
                fontsize=6.2, fontfamily="Arial", fontweight="normal", color=INK)

    network_ax = ax.inset_axes([0.024, 0.185, 0.176, 0.585])
    draw_sample_network(network_ax)
    ax.text(
        0.112,
        0.105,
        r"S = {$S_1$, …, $S_s$},     $|S| \ll N$",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.2,
        color=INK,
    )

    draw_sparse_regressions(ax)
    draw_dbscan_consensus(ax)
    draw_core_to_mask(ax)

    flow_arrow(ax, 0.193, 0.216)
    flow_arrow(ax, 0.483, 0.506)
    flow_arrow(ax, 0.733, 0.756)
    return fig


def save_panel(fig: plt.Figure, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / "Fig1c_support_identification_rebuild"
    fig.savefig(prefix.with_suffix(".svg"), facecolor="white")
    fig.savefig(prefix.with_suffix(".pdf"), facecolor="white")
    fig.savefig(prefix.with_suffix(".png"), dpi=450, facecolor="white")
    fig.savefig(
        prefix.with_suffix(".tiff"),
        dpi=600,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    save_panel(make_panel(), args.output_dir)
    print(f"Saved rebuilt panel c to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
