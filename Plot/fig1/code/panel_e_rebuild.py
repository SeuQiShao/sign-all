"""Render Fig. 1e with equal-width error and rollout regions.

The left half summarizes node-level, per-dimension forecast MSE for all 1,000
nodes.  The right half shows the representative node together with
four context nodes, using every available time point in the supplied rollout.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

from fig1_core import RosslerData, combined_trajectory, load_plot_data


MM = 1.0 / 25.4
WIDTH_MM = 70.272
HEIGHT_MM = 43.783

BLUE = "#2369B3"
PURPLE = "#7650A5"
ORANGE = "#E97824"
GOLD = "#E6A000"
TITLE_BLUE = "#4F8FC3"
FORECAST_FILL = "#FFF1E7"
CONTEXT = "#B8BDC1"
MID = "#8D9398"
EDGE = "#4C5156"
INK = "#151719"
Z_MIN_BAR_PERCENT = 1.0


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
    "axes.labelsize": 5.4,
    "xtick.labelsize": 5.0,
    "ytick.labelsize": 5.0,
    "axes.linewidth": 0.55,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.50,
    "ytick.major.width": 0.50,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,
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
        default=Path(__file__).with_name("panel_e_output"),
    )
    parser.add_argument("--seed-index", type=int, default=2)
    parser.add_argument("--trace-node", type=int, default=121)
    parser.add_argument(
        "--context-nodes", type=int, nargs=4, default=(230, 112, 378, 934)
    )
    return parser.parse_args()


def rounded_frame(ax: plt.Axes) -> None:
    ax.add_patch(
        patches.FancyBboxPatch(
            (0.005, 0.010),
            0.990,
            0.980,
            boxstyle="round,pad=0.003,rounding_size=0.018",
            transform=ax.transAxes,
            facecolor="white",
            edgecolor=EDGE,
            linewidth=0.75,
            clip_on=False,
            zorder=0,
        )
    )


def style_axis(ax: plt.Axes) -> None:
    ax.spines["left"].set_color(EDGE)
    ax.spines["bottom"].set_color(EDGE)
    ax.tick_params(colors=INK, pad=1.2)
    ax.grid(False)


def draw_histograms(parent: plt.Axes, data: RosslerData) -> None:
    colors = (BLUE, PURPLE, ORANGE)
    names = ("x", "y", "z")
    row_bottoms = (0.555, 0.335, 0.115)
    if not np.all(np.isfinite(data.node_mse)) or np.any(data.node_mse <= 0):
        raise ValueError("Log-scaled node MSE values must be finite and positive")
    all_positive = data.node_mse[data.node_mse > 0]
    lower = 10.0 ** np.floor(np.log10(all_positive.min()))
    upper = 10.0 ** np.ceil(np.log10(all_positive.max()))
    bins = np.logspace(np.log10(lower), np.log10(upper), 41)

    for index, (name, color, bottom) in enumerate(zip(names, colors, row_bottoms)):
        ax = parent.inset_axes([0.070, bottom, 0.390, 0.155])
        values = data.node_mse[:, index]
        weights = np.full(values.shape, 100.0 / values.size)
        if index < 2:
            ax.hist(
                values,
                bins=bins,
                weights=weights,
                color=color,
                alpha=0.88,
                edgecolor="white",
                linewidth=0.22,
            )
            ax.set_xscale("log")
            ax.set_xlim(lower, upper)
        else:
            # The z errors occupy a very narrow range on the shared log axis.
            # Use separated local bars; bins representing <1% of nodes are
            # omitted from display only, while statistics use all 1,000 nodes.
            z_edges = np.linspace(float(values.min()), float(values.max()), 19)
            z_counts, _ = np.histogram(values, bins=z_edges)
            z_percent = 100.0 * z_counts / values.size
            z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
            z_widths = 0.72 * np.diff(z_edges)
            visible = z_percent >= Z_MIN_BAR_PERCENT
            ax.bar(
                z_centers[visible],
                z_percent[visible],
                width=z_widths[visible],
                color=color,
                alpha=0.88,
                edgecolor="white",
                linewidth=0.22,
                align="center",
            )
            ax.set_xlim(z_edges[0], z_edges[-1])
        median = float(np.median(values))
        ax.axvline(
            median,
            ymin=0.0,
            ymax=0.78,
            color=INK,
            linewidth=0.62,
            linestyle=(0, (3, 2)),
        )
        ax.set_ylim(bottom=0)
        ax.locator_params(axis="y", nbins=2)
        ax.annotate(
            f"{name}   med. {median:.1e}",
            xy=(0.03, 0.86),
            xycoords=ax.transAxes,
            xytext=(0, 5),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=5.0,
            fontweight="bold",
            color=color,
        )
        if index < 2:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xticks([4.5e-7, 5.0e-7, 5.5e-7])
            ax.set_xticklabels(["4.5", "5.0", "5.5"])
        style_axis(ax)

    parent.text(
        0.026,
        0.415,
        "Nodes (%)",
        transform=parent.transAxes,
        ha="center",
        va="center",
        rotation=90,
        rotation_mode="anchor",
        fontsize=5.2,
        fontfamily="Arial",
        fontweight="normal",
    )
    parent.text(
        0.265,
        0.040,
        "MSE (×10⁻⁷)",
        transform=parent.transAxes,
        ha="center",
        va="center",
        fontsize=5.2,
        fontfamily="DejaVu Sans",
        fontweight="normal",
    )


def draw_rollouts(
    parent: plt.Axes,
    data: RosslerData,
    trace_node: int,
    context_nodes: tuple[int, ...],
) -> None:
    names = ("x", "y", "z")
    row_bottoms = (0.555, 0.335, 0.115)
    time = np.arange(1001)

    # Compact encoding key within the right-hand 50% region.
    legend_y = 0.752
    key_specs = (
        (0.630, 0.670, BLUE, "-", "true", 0.677),
        (0.785, 0.825, GOLD, (0, (4, 2.4)), "pred.", 0.832),
    )
    for x0, x1, color, linestyle, label, text_x in key_specs:
        parent.plot(
            [x0, x1],
            [legend_y, legend_y],
            transform=parent.transAxes,
            color=color,
            linewidth=0.90,
            linestyle=linestyle,
            solid_capstyle="round",
            clip_on=False,
        )
        parent.text(
            text_x,
            legend_y,
            label,
            transform=parent.transAxes,
            ha="left",
            va="center",
            fontsize=6.2,
            color=EDGE,
            fontfamily="Arial",
            fontweight="normal",
        )

    for index, (name, bottom) in enumerate(zip(names, row_bottoms)):
        ax = parent.inset_axes([0.555, bottom, 0.410, 0.155])
        ax.axvspan(800, 1000, color=FORECAST_FILL, zorder=0)
        for node in context_nodes:
            truth = combined_trajectory(data, node, index, predicted=False)
            prediction = combined_trajectory(data, node, index, predicted=True)
            ax.plot(time, truth, color=CONTEXT, linewidth=0.44, alpha=0.78, zorder=1)
            ax.plot(
                time[800:],
                prediction[800:],
                color=CONTEXT,
                linewidth=0.48,
                alpha=0.82,
                linestyle=(0, (4, 2.4)),
                zorder=2,
            )

        truth = combined_trajectory(data, trace_node, index, predicted=False)
        prediction = combined_trajectory(data, trace_node, index, predicted=True)
        ax.plot(time, truth, color=BLUE, linewidth=0.80, zorder=4)
        ax.plot(
            time[800:],
            prediction[800:],
            color=GOLD,
            linewidth=0.92,
            linestyle=(0, (4, 2.4)),
            zorder=5,
        )
        selected_values = np.concatenate((truth, prediction))
        selected_lo = float(selected_values.min())
        selected_hi = float(selected_values.max())
        selected_span = selected_hi - selected_lo
        lower_pad = max((0.14 if index == 2 else 0.08) * selected_span, 1e-6)
        upper_pad = max(0.08 * selected_span, 1e-6)
        ax.set_ylim(selected_lo - lower_pad, selected_hi + upper_pad)
        ax.axvline(800, color=EDGE, linewidth=0.58, linestyle=(0, (3, 2)), zorder=3)
        ax.set_xlim(0, 1000)
        ax.set_yticks([])
        parent.text(
            0.548,
            bottom + 0.0775,
            rf"${name}_i\mathregular{{(t)}}$",
            transform=parent.transAxes,
            ha="right",
            va="center",
            fontsize=7.2,
            color=BLUE,
        )
        if index < 2:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xticks([0, 800, 1000])
            ax.set_xticklabels(["0", "Tobs", "+H"])
            ax.get_xticklabels()[0].set_ha("left")
            ax.get_xticklabels()[-1].set_ha("right")
        style_axis(ax)


def make_figure(
    data: RosslerData,
    trace_node: int,
    context_nodes: tuple[int, ...],
) -> plt.Figure:
    fig = plt.figure(figsize=(WIDTH_MM * MM, HEIGHT_MM * MM), facecolor="white")
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.set_xlim(0, 1)
    canvas.set_ylim(0, 1)
    canvas.axis("off")
    canvas.text(
        0.018, 0.968, "e", transform=canvas.transAxes, ha="left", va="top",
        fontsize=9.5, fontweight="bold", fontfamily="Arial", color="#1C1C1C",
    )
    canvas.text(
        0.500, 0.965, "Prediction performance", transform=canvas.transAxes,
        ha="center", va="top", fontsize=8.6, fontweight="normal",
        fontfamily="Arial", color="#1F5E9E",
    )
    canvas.text(
        0.035, 0.835, "Node MSE distributions", transform=canvas.transAxes,
        ha="left", va="top", fontsize=6.2, fontfamily="Arial",
        fontweight="normal", color=INK,
    )
    canvas.text(
        0.525, 0.835, "Representative rollouts", transform=canvas.transAxes,
        ha="left", va="top", fontsize=6.2, fontfamily="Arial",
        fontweight="normal", color=INK,
    )

    draw_histograms(canvas, data)
    draw_rollouts(canvas, data, trace_node, context_nodes)
    return fig


def save_figure(fig: plt.Figure, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / "Fig1e_prediction_performance_rebuild"
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
    data = load_plot_data(args.plot_data, args.seed_index)
    data.node_index(args.trace_node)
    for node in args.context_nodes:
        data.node_index(node)
    save_figure(
        make_figure(data, args.trace_node, tuple(args.context_nodes)),
        args.output_dir,
    )
    print(f"Saved rebuilt panel e to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
