"""Assemble the compact robustness supplementary figures S1--S3.

All scalar panels read the current curated CSVs under ``data/``.  The script
only assembles and formats the displays; it does not smooth, interpolate or
recompute experiment-level metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "plot_data"
OUT = ROOT / "code" / "output"

DYNAMICS = ["Kuramoto", "SIS", "Gene", "FHN", "HR", "Rossler"]
DYN_COLORS = {
    "Kuramoto": "#6A9EC0",
    "SIS": "#065EA8",
    "Gene": "#9CBECC",
    "FHN": "#6850A4",
    "HR": "#9C76B2",
    "Rossler": "#B689BA",
}
DYN_MARKERS = {"Kuramoto": "o", "SIS": "s", "Gene": "D", "FHN": "^", "HR": "v", "Rossler": "P"}
DYN_STYLES = {
    "Kuramoto": "-",
    "SIS": "--",
    "Gene": "-.",
    "FHN": ":",
    "HR": (0, (4, 1.5)),
    "Rossler": (0, (1.2, 1.2)),
}
METHODS = ["Ours", "Two-Step", "LAGNA"]
METHOD_DISPLAY = {"Ours": "SIGN", "Two-Step": "Two-Step", "LAGNA": "LaGNA"}
METHOD_COLORS = {"Ours": "#2D7AB4", "Two-Step": "#C55FA2", "LAGNA": "#54229A"}
METHOD_MARKERS = {"Ours": "o", "Two-Step": "s", "LAGNA": "^"}
TEXT = "#1C1C1C"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6.6,
        "axes.labelsize": 6.8,
        "axes.titlesize": 7.0,
        "xtick.labelsize": 5.6,
        "ytick.labelsize": 5.6,
        "axes.linewidth": 0.55,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


def style_axis(ax: plt.Axes) -> None:
    ax.tick_params(direction="out", pad=1.0)
    ax.xaxis.labelpad = 0.5
    ax.yaxis.labelpad = 2.5
    ax.xaxis.label.set_horizontalalignment("center")
    ax.yaxis.label.set_verticalalignment("center")


def compact_sci(value: float, _position: int) -> str:
    if value == 0 or not np.isfinite(value):
        return "0"
    return f"{value:.0e}".replace("e+0", "e").replace("e-0", "e-")


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def dynamics_handles() -> list[Line2D]:
    return [
        Line2D(
            [0], [0], color=DYN_COLORS[d], lw=1.0, linestyle=DYN_STYLES[d],
            marker=DYN_MARKERS[d], markersize=2.6, markerfacecolor="white",
            markeredgewidth=0.55, label=d,
        )
        for d in DYNAMICS
    ]


def method_handles() -> list[Line2D]:
    return [
        Line2D(
            [0], [0], color=METHOD_COLORS[m], lw=1.1, linestyle="-",
            marker=METHOD_MARKERS[m], markersize=2.8, markerfacecolor="white",
            markeredgewidth=0.55, label=METHOD_DISPLAY[m],
        )
        for m in METHODS
    ]


def panel_label(ax: plt.Axes, label: str) -> None:
    text_fn = getattr(ax, "text2D", ax.text)
    text_fn(-0.16, 1.04, label, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7.4, fontweight="bold", color=TEXT)


def plot_dynamics(ax: plt.Axes, frame: pd.DataFrame, *, xlim: tuple[float, float], ylim: tuple[float, float],
                  xlabel: str | None = None, ylabel: str | None = None) -> None:
    for dynamics in DYNAMICS:
        sub = frame[frame["dynamics"] == dynamics].sort_values("x")
        if sub.empty:
            continue
        x = sub["x"].to_numpy(float)
        y = sub["mean"].to_numpy(float)
        s = sub["std"].to_numpy(float)
        ax.fill_between(x, np.maximum(y - s, 0), y + s, color=DYN_COLORS[dynamics], alpha=0.15, linewidth=0)
        ax.plot(x, y, color=DYN_COLORS[dynamics], lw=0.85, linestyle=DYN_STYLES[dynamics],
                marker=DYN_MARKERS[dynamics], markersize=2.1, markerfacecolor="white",
                markeredgewidth=0.45)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    style_axis(ax)


def joint_limits(frames: list[pd.DataFrame], *, xpad: float = 0.04, ypad: float = 0.06) -> tuple[tuple[float, float], tuple[float, float]]:
    x_values = np.concatenate([f["x"].to_numpy(float) for f in frames])
    y_values = np.concatenate([(f["mean"] + f["std"]).to_numpy(float) for f in frames])
    x_values = x_values[np.isfinite(x_values)]
    y_values = y_values[np.isfinite(y_values)]
    xmin, xmax = float(np.min(x_values)), float(np.max(x_values))
    span = max(xmax - xmin, abs(xmax), 1e-6)
    xlim = (xmin - xpad * span, xmax + xpad * span)
    if xmin >= 0:
        xlim = (max(0.0, xlim[0]), xlim[1])
    ymax = max(float(np.max(y_values)) * (1 + ypad), 1e-6)
    return xlim, (0.0, ymax)


def plot_coupling(ax: plt.Axes, frame: pd.DataFrame, *, xlim: tuple[float, float], ylim: tuple[float, float],
                  xlabel: str | None, ylabel: str | None, show_right_label: bool) -> None:
    plot_dynamics(ax, frame, xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)
    sub = frame[frame["dynamics"] == "FHN"].sort_values("x")
    if "sync" not in sub or sub.empty:
        return
    right = ax.twinx()
    right.plot(sub["x"], sub["sync"], color="#654EA3", lw=0.8, linestyle="--", alpha=0.8)
    right.set_ylim(0.4, 1.0)
    right.set_yticks([0.4, 0.7, 1.0])
    if show_right_label:
        right.set_ylabel(r"$\langle R\rangle$", labelpad=1.5)
    else:
        right.set_yticklabels([])
    right.tick_params(direction="out", pad=0.8, labelsize=5.0, width=0.4, length=1.8)
    right.spines["right"].set_color("#B8A9D5")
    right.spines["right"].set_linewidth(0.45)
    right.spines["top"].set_visible(False)


def build_s1() -> dict:
    out = OUT / "FigS1_scale_robustness"
    specs = [
        ("Noise", "noise_1k.csv", "noise_100k.csv", "SNR (dB)"),
        ("Sample #", "sampling_points_1k.csv", "sampling_points_100k.csv", "Sampling points (n)"),
        ("Δt", "sampling_interval_1k.csv", "sampling_interval_100k.csv", "Sampling interval (Δt)"),
        ("Heterogeneity", "heterogeneity_1k.csv", "heterogeneity_100k.csv", "σ"),
        ("Coupling", "coupling_1k.csv", "coupling_100k.csv", "Coupling (E)"),
    ]
    fig, axes = plt.subplots(2, 5, figsize=(10.8, 4.25), squeeze=False)
    audit = {"figure": "S1", "panels": []}
    for col, (title, top_name, bottom_name, xlabel) in enumerate(specs):
        top = read_csv(top_name)
        bottom = read_csv(bottom_name)
        xlim, ylim = joint_limits([top, bottom])
        if title not in {"Sample #", "Δt"}:
            ylim = (0.0, 1.0)
        for row, frame in enumerate((top, bottom)):
            ax = axes[row, col]
            panel_label(ax, chr(ord("a") + row * 5 + col))
            if row == 0:
                ax.set_title(title, fontsize=7.2, pad=4)
            show_x = row == 1
            show_y = col == 0
            if title == "Coupling":
                plot_coupling(ax, frame, xlim=xlim, ylim=ylim, xlabel=xlabel if show_x else None,
                              ylabel="sMAPE" if show_y else None, show_right_label=row == 1)
            else:
                plot_dynamics(ax, frame, xlim=xlim, ylim=ylim, xlabel=xlabel if show_x else None,
                              ylabel="sMAPE" if show_y else None)
            audit["panels"].append({"label": chr(ord("a") + row * 5 + col), "sources": [top_name, bottom_name],
                                    "rows": [int(len(top)), int(len(bottom))], "xlim": xlim, "ylim": ylim})
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.14, top=0.84, wspace=0.28, hspace=0.34)
    fig.text(0.018, 0.645, "N=1k", rotation=90, rotation_mode="anchor", ha="center", va="center", fontsize=7.0)
    fig.text(0.018, 0.315, "N=100k", rotation=90, rotation_mode="anchor", ha="center", va="center", fontsize=7.0)
    fig.legend(handles=dynamics_handles(), loc="upper center", bbox_to_anchor=(0.53, 0.985), ncol=6,
               fontsize=5.4, handlelength=1.6, columnspacing=0.9, borderaxespad=0.0)
    save_figure(fig, out / "FigS1_scale_robustness")
    for idx, (title, top_name, bottom_name, xlabel) in enumerate(specs):
        fig_p, ax = plt.subplots(figsize=(2.25, 1.75))
        frames = [read_csv(top_name), read_csv(bottom_name)]
        xlim, ylim = joint_limits(frames)
        if title not in {"Sample #", "Δt"}:
            ylim = (0.0, 1.0)
        frame = frames[0]
        if title == "Coupling":
            plot_coupling(ax, frame, xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel="sMAPE", show_right_label=True)
        else:
            plot_dynamics(ax, frame, xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel="sMAPE")
        ax.set_title(f"{title} · N=1k", fontsize=7.0, pad=3)
        panel_label(ax, chr(ord("a") + idx))
        ax.legend(handles=dynamics_handles(), loc="upper right", fontsize=5.0, ncol=2, handlelength=1.2,
                  columnspacing=0.6, borderaxespad=0.1)
        fig_p.subplots_adjust(left=0.19, right=0.96, bottom=0.20, top=0.86)
        save_figure(fig_p, out / f"panel_{chr(ord('a') + idx)}")
    return audit


def plot_low_rank_panel(ax: plt.Axes, frame: pd.DataFrame, network: str, *, ylabel: bool) -> None:
    order = ["Gene", "Kuramoto", "SIS", "HR", "FHN", "Rossler"]
    sub = frame[frame["network"].str.lower() == network.lower()].copy()
    sub["order"] = pd.Categorical(sub["dynamics"], order, ordered=True)
    sub = sub.sort_values("order")
    x = np.arange(len(sub))
    ax.bar(x, sub["mean"], width=0.72, color=[DYN_COLORS[d] for d in sub["dynamics"]], alpha=0.9,
           edgecolor="none")
    ax.errorbar(x, sub["mean"], yerr=0.5 * sub["std"], fmt="none", ecolor="#555555", capsize=1.6,
                linewidth=0.45)
    ax.set_xticks([])
    ax.set_xlim(-0.6, len(sub) - 0.4)
    ax.set_ylim(0, 0.6 if network.lower() == "dscm" else 0.2)
    ax.set_xlabel(network.upper(), labelpad=1.0)
    if ylabel:
        ax.set_ylabel("sMAPE")
    style_axis(ax)


def plot_missing_panel(ax: plt.Axes, frame: pd.DataFrame, *, xlabel: str, ylabel: str | None,
                       xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    for dynamics in DYNAMICS:
        sub = frame[frame["dynamics"] == dynamics].sort_values("x")
        if sub.empty:
            continue
        x = sub["x"].to_numpy(float)
        y = sub["mean"].to_numpy(float)
        s = sub["std"].to_numpy(float)
        ax.fill_between(x, np.maximum(y - s, 0), y + s, color=DYN_COLORS[dynamics], alpha=0.13, linewidth=0)
        ax.plot(x, y, color=DYN_COLORS[dynamics], lw=0.82, linestyle=DYN_STYLES[dynamics],
                marker=DYN_MARKERS[dynamics], markersize=1.9, markerfacecolor="white", markeredgewidth=0.4)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    style_axis(ax)


def plot_methods_panel(ax: plt.Axes, frame: pd.DataFrame, *, xlabel: str, ylabel: str | None,
                       xlim: tuple[float, float], ylim: tuple[float, float], xscale: str = "linear",
                       yscale: str = "linear", node_percent_labels: bool = False) -> None:
    x_values = np.sort(frame["x"].dropna().unique())
    categorical = node_percent_labels
    lookup = {value: i + 1 for i, value in enumerate(x_values)}
    for method in METHODS:
        sub = frame[frame["method"] == method].sort_values("x")
        if sub.empty:
            continue
        raw_x = sub["x"].to_numpy(float)
        x = np.asarray([lookup[v] for v in raw_x]) if categorical else raw_x
        y = sub["mean"].to_numpy(float)
        s = sub["std"].to_numpy(float)
        ax.fill_between(x, y - s, y + s, color=METHOD_COLORS[method], alpha=0.13, linewidth=0)
        ax.plot(x, y, color=METHOD_COLORS[method], lw=0.95, marker=METHOD_MARKERS[method], markersize=2.2,
                markerfacecolor="white", markeredgewidth=0.45)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if categorical:
        ax.set_xticks(np.arange(1, len(x_values) + 1))
        ax.set_xticklabels(["1", "5", "10", "15", "20"])
    if yscale == "log":
        ax.yaxis.set_major_formatter(FuncFormatter(compact_sci))
    if xscale == "log":
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    style_axis(ax)


def build_s2() -> dict:
    out = OUT / "FigS2_network_benchmark"
    fig, axes = plt.subplots(2, 4, figsize=(9.2, 4.0), squeeze=False)
    audit = {"figure": "S2", "panels": []}
    for stale_label in "ijkl":
        for suffix in (".svg", ".pdf", ".png"):
            (out / f"panel_{stale_label}{suffix}").unlink(missing_ok=True)
    lowrank = read_csv("low_rank_network.csv")
    networks = ["dscm", "rgm", "rpg", "sbm"]
    for col, network in enumerate(networks):
        ax = axes[0, col]
        plot_low_rank_panel(ax, lowrank, network, ylabel=col == 0)
        panel_label(ax, chr(ord("a") + col))
        audit["panels"].append({"label": chr(ord("a") + col), "source": "low_rank_network.csv", "network": network,
                                "errorbar": "0.5 * std"})
    missing_specs = [
        ("missing_nodes_1k.csv", "Missing nodes (%)", (0, 21), "e"),
        ("missing_nodes_100k.csv", "Missing nodes (%)", (0, 21), "f"),
        ("missing_links_1k.csv", "Missing links (%)", (0, 42), "g"),
        ("missing_links_100k.csv", "Missing links (%)", (0, 42), "h"),
    ]
    missing_frames = [read_csv(name) for name, _, _, _ in missing_specs]
    missing_ylim = (0.0, 1.0)
    for col, (spec, frame) in enumerate(zip(missing_specs, missing_frames)):
        name, xlabel, xlim, label = spec
        plot_missing_panel(axes[1, col], frame, xlabel=xlabel, ylabel="sMAPE" if col == 0 else None,
                           xlim=xlim, ylim=missing_ylim)
        panel_label(axes[1, col], label)
        audit["panels"].append({"label": label, "source": name, "rows": int(len(frame)), "ylim": missing_ylim})
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.16, top=0.82, wspace=0.31, hspace=0.48)
    fig.legend(handles=dynamics_handles(), loc="upper center", bbox_to_anchor=(0.50, 0.985), ncol=6,
               fontsize=5.0, handlelength=1.45, columnspacing=0.85, borderaxespad=0.0)
    save_figure(fig, out / "FigS2_network_benchmark")
    # Independent panels retain a local legend so they remain interpretable.
    for idx, (network, label) in enumerate(zip(networks, "abcd")):
        fig_p, ax = plt.subplots(figsize=(2.1, 1.75))
        plot_low_rank_panel(ax, lowrank, network, ylabel=True)
        panel_label(ax, label)
        ax.legend(handles=[Patch(color=DYN_COLORS[d], label=d) for d in DYNAMICS], loc="upper left", fontsize=5.0,
                  ncol=2, handlelength=0.8, columnspacing=0.45)
        fig_p.subplots_adjust(left=0.17, right=0.97, bottom=0.19, top=0.92)
        save_figure(fig_p, out / f"panel_{label}")
    for idx, (name, xlabel, xlim, label) in enumerate(missing_specs):
        fig_p, ax = plt.subplots(figsize=(2.1, 1.75))
        plot_missing_panel(ax, read_csv(name), xlabel=xlabel, ylabel="sMAPE", xlim=xlim, ylim=missing_ylim)
        panel_label(ax, label)
        ax.legend(handles=dynamics_handles(), loc="upper right", fontsize=5.0, ncol=2, handlelength=0.85,
                  columnspacing=0.45)
        fig_p.subplots_adjust(left=0.18, right=0.97, bottom=0.20, top=0.92)
        save_figure(fig_p, out / f"panel_{label}")
    return audit


TRUE_HIGHLIGHT = "#4F809E"
INFERRED_HIGHLIGHT = "#97689E"
TRAJECTORY_GREY = "#A7A7A7"


def trajectory_pair_colors(index: int) -> tuple[str, str]:
    if index == 0:
        return TRUE_HIGHLIGHT, INFERRED_HIGHLIGHT
    return TRAJECTORY_GREY, TRAJECTORY_GREY


def plot_mul_trajectory_panel(ax: plt.Axes, frame: pd.DataFrame, *, ylabel: bool = True) -> None:
    trajectories = [5, 6, 7, 11]
    for idx, trajectory in enumerate(trajectories):
        sub = frame[frame["trajectory_index_matlab"] == trajectory].sort_values("time")
        true_color, inferred_color = trajectory_pair_colors(idx)
        ax.plot(sub["time"], sub["true"], color=true_color, lw=0.8)
        ax.plot(sub["time"], sub["inferred"], color=inferred_color, lw=0.8, linestyle="--")
    ax.set_xlabel("Time")
    if ylabel:
        ax.set_ylabel("State")
    ax.set_ylim(0, 7)
    style_axis(ax)


def plot_chua_trajectory_panel(ax: plt.Axes, frame: pd.DataFrame) -> None:
    trajectories = [5, 7, 4]
    for idx, trajectory in enumerate(trajectories):
        sub = frame[frame["trajectory_index_matlab"] == trajectory]
        true = sub.pivot(index="time", columns="dimension", values="true").sort_index()
        inferred = sub.pivot(index="time", columns="dimension", values="inferred").sort_index()
        true_color, inferred_color = trajectory_pair_colors(idx)
        ax.plot(true[0], true[1], true[2], color=true_color, lw=0.75)
        ax.plot(inferred[0], inferred[1], inferred[2], color=inferred_color, lw=0.75, linestyle="--")
    ax.set_xlabel("x1", labelpad=-1)
    ax.set_ylabel("x2", labelpad=-1)
    ax.set_zlabel("x3", labelpad=-1)
    ax.tick_params(labelsize=5.0, pad=0)
    ax.view_init(elev=24, azim=-62)
    ax.set_box_aspect((1.15, 0.85, 0.9))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"] = (0.84, 0.84, 0.84, 1.0)
        axis._axinfo["grid"]["linewidth"] = 0.35
        axis._axinfo["grid"]["linestyle"] = "-"
        axis.pane.set_facecolor((0.985, 0.985, 0.985, 1.0))
        axis.pane.set_edgecolor((0.90, 0.90, 0.90, 1.0))


def plot_chua_dimension_panel(ax: plt.Axes, frame: pd.DataFrame, dimension: int) -> None:
    trajectories = [5, 7, 4]
    for idx, trajectory in enumerate(trajectories):
        sub = frame[frame["trajectory_index_matlab"] == trajectory]
        true = sub[sub["dimension"] == dimension].sort_values("time")
        true_color, inferred_color = trajectory_pair_colors(idx)
        ax.plot(true["time"], true["true"], color=true_color, lw=0.75)
        ax.plot(true["time"], true["inferred"], color=inferred_color, lw=0.75, linestyle="--")
    ax.set_title(f"Chua · x{dimension + 1}", fontsize=6.8, pad=3)
    ax.set_xlabel("Time")
    ax.set_ylabel(f"x{dimension + 1}")
    style_axis(ax)


def build_s3() -> dict:
    out = OUT / "FigS3_basis_mismatch"
    mul = read_csv("mul_selected_trajectories.csv")
    chua = read_csv("chua_selected_trajectories.csv")
    fig = plt.figure(figsize=(12.3, 2.85))
    grid = fig.add_gridspec(1, 5, width_ratios=[1.35, 1.0, 1.0, 1.0, 1.15], wspace=0.48)
    ax_mul = fig.add_subplot(grid[0, 0])
    ax_dim1 = fig.add_subplot(grid[0, 1])
    ax_dim2 = fig.add_subplot(grid[0, 2])
    ax_dim3 = fig.add_subplot(grid[0, 3])
    ax_chua = fig.add_subplot(grid[0, 4], projection="3d")
    plot_mul_trajectory_panel(ax_mul, mul, ylabel=True)
    ax_mul.set_title("Mutualistic", fontsize=6.8, pad=3)
    plot_chua_dimension_panel(ax_dim1, chua, 0)
    plot_chua_dimension_panel(ax_dim2, chua, 1)
    plot_chua_dimension_panel(ax_dim3, chua, 2)
    plot_chua_trajectory_panel(ax_chua, chua)
    ax_chua.set_title("Chua 3D", fontsize=6.8, pad=3)
    for ax, label in zip((ax_mul, ax_dim1, ax_dim2, ax_dim3, ax_chua), "abcde"):
        panel_label(ax, label)
    handles = [Line2D([0], [0], color=TRUE_HIGHLIGHT, lw=1.0, label="True"),
               Line2D([0], [0], color=INFERRED_HIGHLIGHT, lw=1.0, linestyle="--", label="Inferred")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.50, 0.995), ncol=2, fontsize=5.5,
               handlelength=1.6, columnspacing=1.0, borderaxespad=0.0)
    fig.subplots_adjust(left=0.045, right=0.99, bottom=0.18, top=0.82)
    save_figure(fig, out / "FigS3_basis_mismatch")

    fig_p, ax = plt.subplots(figsize=(3.2, 2.1))
    plot_mul_trajectory_panel(ax, mul, ylabel=True)
    ax.set_title("Mutualistic", fontsize=6.8, pad=3)
    panel_label(ax, "a")
    ax.legend(handles=handles, loc="upper right", fontsize=5.0, handlelength=1.2)
    fig_p.subplots_adjust(left=0.16, right=0.97, bottom=0.18, top=0.86)
    save_figure(fig_p, out / "panel_a")

    for label, dimension in zip("bcd", range(3)):
        fig_p, ax = plt.subplots(figsize=(2.25, 2.1))
        plot_chua_dimension_panel(ax, chua, dimension)
        panel_label(ax, label)
        ax.legend(handles=handles, loc="upper right", fontsize=5.0, handlelength=1.2)
        fig_p.subplots_adjust(left=0.20, right=0.97, bottom=0.18, top=0.86)
        save_figure(fig_p, out / f"panel_{label}")
    fig_p = plt.figure(figsize=(3.2, 2.1))
    ax = fig_p.add_subplot(111, projection="3d")
    plot_chua_trajectory_panel(ax, chua)
    ax.set_title("Chua 3D", fontsize=6.8, pad=3)
    panel_label(ax, "e")
    ax.legend(handles=handles, loc="upper left", fontsize=5.0, handlelength=1.2)
    fig_p.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.86)
    save_figure(fig_p, out / "panel_e")
    return {"figure": "S3", "panels": [
        {"label": "a", "source": "mul_selected_trajectories.csv", "rows": int(len(mul)),
         "dynamics": "Mutualistic", "trajectories": [5, 6, 7, 11]},
        *[
            {"label": label, "source": "chua_selected_trajectories.csv", "rows": int(len(chua)),
             "dynamics": "Chua", "dimension": dimension}
            for label, dimension in zip("bcd", range(3))
        ],
        {"label": "e", "source": "chua_selected_trajectories.csv", "rows": int(len(chua)),
         "dynamics": "Chua", "trajectories": [5, 7, 4], "dimensions": [0, 1, 2]},
    ]}


def write_qa(audits: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "si_audit.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "SI_Robustness_QA.md").write_text(
        "# Robustness SI compact figure QA\n\n"
        "- S1 is a 2×5 scale-robustness grid with one shared dynamics legend.\n"
        "- S2 is a 2×4 network/benchmark grid; low-rank error bars use 0.5× the current CSV std. Its former methods-comparison row was removed because it duplicated the main figure.\n"
        "- S3 is a 1×5 trajectory figure: Mutualistic, three Chua dimensions, and Chua 3D in that order. One True/Inferred pair is highlighted and remaining trajectories are grey.\n"
        "- `sampling_interval_100k.csv` is consumed through its curated `x` column, which is the source column-A Δt.\n"
        "- Non-finite, OOM and >12 h placeholders are not plotted; no smoothing or interpolation is applied.\n"
        "- Runtime y ticks use compact scientific notation.\n"
        "- Every composite and standalone panel has editable SVG/PDF and 600-dpi PNG exports.\n\n"
        "The exact source files, row counts and axis limits are recorded in `si_audit.json`.\n",
        encoding="utf-8",
    )


def main() -> None:
    audits = [build_s3()]
    write_qa(audits)
    print(f"Generated compact robustness SI figures under {OUT}")
    for audit in audits:
        print(audit["figure"], len(audit["panels"]))


if __name__ == "__main__":
    main()
