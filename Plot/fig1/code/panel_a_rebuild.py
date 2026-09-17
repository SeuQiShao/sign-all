"""Render Fig. 1a as a standalone, final-size panel.

The fixed NetworkX topology and coordinates are the canonical network master
reused by panels c and d.
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

from fig1_core import RosslerData, load_plot_data


MM = 1.0 / 25.4
width_mm = 83.448
height_mm = 42.136

BLUE = "#2369B3"
BLUE_DARK = "#124A86"
TITLE_BLUE = "#4F8FC3"
PURPLE = "#7650A5"
PURPLE_DARK = "#56317F"
ORANGE = "#E97824"
ORANGE_DARK = "#B9540E"
GOLD = "#E6A000"
FORECAST_BG = "#FFF2E8"
EDGE = "#4C5156"
MID = "#8D9398"
LIGHT = "#D9DDE0"
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
    "axes.titlesize": 6.2,
    "axes.labelsize": 5.6,
    "xtick.labelsize": 5.2,
    "ytick.labelsize": 5.2,
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
        "--plot-data",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "plot_data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("panel_a_output"),
    )
    parser.add_argument("--seed-index", type=int, default=2)
    # Node 121 retains a visibly varying z trajectory after the transient while
    # keeping the rollout error representative and the curve clear of the axis.
    parser.add_argument("--trace-node", type=int, default=121)
    return parser.parse_args()


def rounded_box(
    ax: plt.Axes,
    bounds: tuple[float, float, float, float],
    *,
    edge: str = MID,
    face: str = "white",
    linewidth: float = 0.72,
    radius: float = 0.012,
) -> None:
    x, y, w, h = bounds
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y), w, h,
            transform=ax.transAxes,
            boxstyle=f"round,pad=0.004,rounding_size={radius}",
            facecolor=face,
            edgecolor=edge,
            linewidth=linewidth,
            clip_on=False,
        )
    )


def build_network_master() -> tuple[nx.Graph, dict[int, tuple[float, float]], int, list[int]]:
    """Load the shared NetworkX topology and coordinates from plot_data."""
    plot_data = Path(__file__).resolve().parents[1] / "plot_data"
    graph = nx.Graph()
    pos: dict[int, tuple[float, float]] = {}
    roles: dict[int, str] = {}
    continuation: set[int] = set()
    with (plot_data / "network_nodes.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            node = int(row["node"])
            graph.add_node(node)
            pos[node] = (float(row["x"]), float(row["y"]))
            roles[node] = row["role"]
            if int(row["continuation"]):
                continuation.add(node)
    with (plot_data / "network_edges.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            graph.add_edge(int(row["source"]), int(row["target"]))
    target = next(node for node, role in roles.items() if role == "target")
    neighbors = [node for node, role in roles.items() if role == "neighbor"]
    return graph, pos, target, neighbors


def draw_network_master(ax: plt.Axes) -> None:
    graph, pos, target, neighbors = build_network_master()
    continuation = list(range(14, 19))
    other = [
        node for node in graph
        if node not in neighbors and node != target and node not in continuation
    ]
    continuation_edges = [
        edge for edge in graph.edges if edge[0] in continuation or edge[1] in continuation
    ]
    solid_edges = [edge for edge in graph.edges if edge not in continuation_edges]

    nx.draw_networkx_edges(
        graph, pos, edgelist=solid_edges, ax=ax,
        edge_color="#B7BDC2", width=0.52,
    )
    nx.draw_networkx_edges(
        graph, pos, edgelist=continuation_edges, ax=ax,
        edge_color="#BFC4C8", width=0.50, style=(0, (2.0, 2.0)),
    )
    nx.draw_networkx_nodes(
        graph, pos, nodelist=other, ax=ax, node_color="#E1E4E6",
        node_size=15, edgecolors=EDGE, linewidths=0.42,
    )
    nx.draw_networkx_nodes(
        graph, pos, nodelist=neighbors, ax=ax, node_color=PURPLE,
        node_size=19, edgecolors=PURPLE_DARK, linewidths=0.46,
    )
    nx.draw_networkx_nodes(
        graph, pos, nodelist=[target], ax=ax, node_color=BLUE,
        node_size=34, edgecolors=BLUE_DARK, linewidths=0.56,
    )
    continuation_nodes = nx.draw_networkx_nodes(
        graph, pos, nodelist=continuation, ax=ax, node_color="white",
        node_size=11, edgecolors=MID, linewidths=0.44,
    )
    continuation_nodes.set_linestyle((0, (2.0, 1.8)))
    nx.draw_networkx_labels(
        graph, pos, labels={target: "i"}, ax=ax,
        font_size=5.0, font_color="white", font_weight="bold",
        font_family="Times New Roman",
    )
    ax.set_xlim(-3.22, 3.30)
    ax.set_ylim(-3.18, 3.22)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_role_key(ax: plt.Axes) -> None:
    items = [
        (0.735, BLUE, BLUE_DARK, "target i"),
        (0.620, PURPLE, PURPLE_DARK, "neighbor j"),
        (0.505, "#E1E4E6", EDGE, "other"),
    ]
    for y, face, edge, label in items:
        ax.scatter([0.278], [y], transform=ax.transAxes, s=16,
                   facecolor=face, edgecolor=edge, linewidth=0.60, zorder=5)
        ax.text(0.296, y, label, transform=ax.transAxes, ha="left", va="center",
                fontsize=6.2, color=INK, fontfamily="Arial", fontweight="normal")


def set_trace_limits(
    ax: plt.Axes,
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    lower_pad_fraction: float = 0.08,
) -> None:
    values = np.concatenate((observed, predicted))
    lo = float(values.min())
    hi = float(values.max())
    span = hi - lo
    lower_pad = max(lower_pad_fraction * span, 1e-6)
    upper_pad = max(0.08 * span, 1e-6)
    ax.set_ylim(lo - lower_pad, hi + upper_pad)


def draw_trajectories(ax: plt.Axes, data: RosslerData, trace_node: int) -> None:
    ax.axis("off")
    observed_step = np.arange(801)
    forecast_step = np.arange(800, 1001)
    labels = [r"$x_i(t)$", r"$y_i(t)$", r"$z_i(t)$"]

    ax.text(0.315, 1.035, "Observed trajectory", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=6.2, fontfamily="Arial",
            fontweight="normal", color=INK)
    ax.text(0.805, 1.035, "Predicted rollout", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=6.2, fontfamily="Arial",
            fontweight="normal", color=INK)

    for dim, label in enumerate(labels):
        sub = ax.inset_axes([0.105, 0.705 - 0.325 * dim, 0.885, 0.245])
        node_index = data.node_index(trace_node)
        observed = data.observed_true[:, node_index, dim]
        predicted = data.forecast_pred[:, node_index, dim]
        sub.axvspan(800, 1000, color=FORECAST_BG, zorder=0)
        sub.plot(observed_step, observed, color=BLUE, lw=1.08,
                 solid_capstyle="round", zorder=3)
        sub.plot(forecast_step, predicted, color=GOLD, lw=1.08,
                 ls=(0, (4.0, 2.5)), dash_capstyle="round", zorder=4)
        sub.axvline(800, color=EDGE, lw=0.70, ls=(0, (3.0, 2.2)), zorder=2)
        sub.set_xlim(0, 1000)
        set_trace_limits(
            sub,
            observed,
            predicted,
            lower_pad_fraction=0.14 if dim == 2 else 0.08,
        )
        sub.set_yticks([])
        sub.text(-0.035, 0.50, label, transform=sub.transAxes, ha="right", va="center",
                 fontsize=7.2, color=BLUE, clip_on=False)
        sub.spines["left"].set_visible(True)
        sub.spines["left"].set_color(EDGE)
        sub.spines["left"].set_linewidth(0.62)
        if dim < 2:
            sub.set_xticks([])
            sub.spines["bottom"].set_visible(False)
        else:
            sub.set_xticks([0, 800, 1000])
            sub.set_xticklabels(["0", "Tobs", "+H"])
            sub.get_xticklabels()[0].set_ha("left")
            sub.get_xticklabels()[-1].set_ha("right")
            sub.tick_params(axis="x", length=2.0, width=0.55, pad=1.0)
            sub.spines["bottom"].set_color(EDGE)
            sub.spines["bottom"].set_linewidth(0.62)


def draw_dynamics_boxes(ax: plt.Axes) -> None:
    left = (0.025, 0.028, 0.365, 0.297)
    right = (0.410, 0.028, 0.565, 0.297)
    rounded_box(ax, left, edge=MID, face="#FCFCFC", linewidth=0.75)
    rounded_box(ax, right, edge=MID, face="#FCFCFC", linewidth=0.75)

    ax.text(left[0] + left[2] / 2, left[1] + left[3] - 0.030,
            "Unknown true dynamics", transform=ax.transAxes,
            ha="center", va="center", fontsize=6.2, fontfamily="Arial",
            fontweight="normal", color=INK)
    equations = [
        r"$\dot{x}_i=-y_i-z_i+\mathregular{∑}_{j} A_{ij}(x_j-x_i)$",
        r"$\dot{y}_i=x_i+a y_i$",
        r"$\dot{z}_i=b+z_i(x_i-c)$",
    ]
    for y, equation in zip((0.210, 0.147, 0.083), equations):
        ax.text(left[0] + left[2] / 2, y, equation, transform=ax.transAxes,
                ha="center", va="center", fontsize=7.2, color=INK,
                fontfamily="Times New Roman")

    ax.text(right[0] + right[2] / 2, right[1] + right[3] - 0.030,
            "Shared-dynamics assumption", transform=ax.transAxes,
            ha="center", va="center", fontsize=6.2, fontfamily="Arial",
            fontweight="normal", color=INK)
    ax.text(right[0] + right[2] / 2, 0.186,
            r"$\dot{x}_i=F(x_i)+\mathregular{∑}_{j} A_{ij}C(x_i,x_j)$", transform=ax.transAxes,
            ha="center", va="center", fontsize=7.6, color=INK,
            fontfamily="Times New Roman")
    ax.plot([0.548, 0.640], [0.136, 0.136], transform=ax.transAxes,
            color=BLUE, lw=1.25, solid_capstyle="round")
    ax.plot([0.746, 0.868], [0.136, 0.136], transform=ax.transAxes,
            color=PURPLE, lw=1.25, solid_capstyle="round")
    ax.text(0.594, 0.090, r"self-dynamics  $F(\cdot)$", transform=ax.transAxes,
            ha="center", va="center", fontsize=5.0, color=BLUE_DARK)
    ax.text(0.807, 0.090, r"coupling dynamics  $C(\cdot,\cdot)$", transform=ax.transAxes,
            ha="center", va="center", fontsize=5.0, color=PURPLE_DARK)


def make_panel(data: RosslerData, trace_node: int) -> plt.Figure:
    fig = plt.figure(figsize=(width_mm * MM, height_mm * MM), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.017, 0.945, "a", transform=ax.transAxes, ha="left", va="center",
            fontsize=9.5, fontweight="bold", fontfamily="Arial", color="#1C1C1C")
    ax.text(0.500, 0.945, "Prediction of networked Rössler dynamics",
            transform=ax.transAxes, ha="center", va="center", fontsize=8.6,
            fontweight="normal", fontfamily="Arial", color="#1F5E9E")

    network_ax = ax.inset_axes([0.022, 0.390, 0.245, 0.445])
    draw_network_master(network_ax)
    draw_role_key(ax)

    trace_ax = ax.inset_axes([0.438, 0.390, 0.537, 0.435])
    draw_trajectories(trace_ax, data, trace_node)
    draw_dynamics_boxes(ax)
    return fig


def save_panel(fig: plt.Figure, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / "Fig1a_Rossler_rebuild"
    fig.savefig(prefix.with_suffix(".svg"), facecolor="white")
    fig.savefig(prefix.with_suffix(".pdf"), facecolor="white")
    fig.savefig(prefix.with_suffix(".png"), dpi=450, facecolor="white")
    fig.savefig(prefix.with_suffix(".tiff"), dpi=600, facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data = load_plot_data(args.plot_data, args.seed_index)
    data.node_index(args.trace_node)
    save_panel(make_panel(data, args.trace_node), args.output_dir)
    print(f"Saved rebuilt panel a to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
