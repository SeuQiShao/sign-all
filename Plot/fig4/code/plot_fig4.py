"""Assemble the six-panel robustness Fig. 3 main display.

The plotting inputs are the already curated CSVs under ``data/``.  This file
does not re-run the experiments or alter the source tables; it only assembles
the requested 100k-node main display and exports the six standalone panels.
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
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import Bbox, blended_transform_factory


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "plot_data"
OUT = ROOT / "code" / "output"

TITLE_BLUE = "#1F5E9E"
TEXT = "#1C1C1C"
GRID = "#D9DDE2"
DYNAMICS = ["Kuramoto", "SIS", "Gene", "FHN", "HR", "Rossler"]
DYN_COLORS = {
    "Kuramoto": "#6A9EC0",
    "SIS": "#065EA8",
    "Gene": "#9CBECC",
    "FHN": "#6850A4",
    "HR": "#9C76B2",
    "Rossler": "#B689BA",
}
DYN_MARKERS = {
    "Kuramoto": "o",
    "SIS": "s",
    "Gene": "D",
    "FHN": "^",
    "HR": "v",
    "Rossler": "P",
}
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

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.2,
        "axes.labelsize": 7.4,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


def add_row_frame(fig: plt.Figure, bbox: Bbox) -> None:
    """Add one compact frame around a complete row, not every subplot."""
    fig.patches.append(
        FancyBboxPatch(
            (bbox.x0 + 0.003, bbox.y0 + 0.004),
            bbox.width - 0.006,
            bbox.height - 0.008,
            transform=fig.transFigure,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            facecolor="white",
            edgecolor="#2A2A2A",
            linewidth=0.75,
            zorder=-10,
        )
    )


def add_panel_header(fig: plt.Figure, bbox: Bbox, label: str, title: str) -> None:
    fig.text(bbox.x0 + 0.014, bbox.y1 - 0.018, label, fontsize=9.5, fontweight="bold", color=TEXT, va="top")
    fig.text((bbox.x0 + bbox.x1) / 2, bbox.y1 - 0.015, title, fontsize=8.6, color=TITLE_BLUE, va="top", ha="center")


def add_frame(fig: plt.Figure, bbox: Bbox, label: str, title: str) -> None:
    """Standalone panel frame, where one frame remains meaningful."""
    add_row_frame(fig, bbox)
    add_panel_header(fig, bbox, label, title)


def style_axis(ax: plt.Axes) -> None:
    # Keep x-axis labels inside the enclosing row frame.  The row frame is
    # intentional; labels should belong to that frame rather than hang below
    # it into the inter-row gutter.
    ax.tick_params(direction="out", pad=1.2)
    ax.xaxis.labelpad = 0
    ax.yaxis.labelpad = 1.5
    # Keep every xlabel centered under its own axes and every ylabel centered
    # along its own y-axis; panel-specific outer labels override only x.
    ax.xaxis.label.set_horizontalalignment("center")
    ax.yaxis.label.set_verticalalignment("center")


def move_outer_ylabel(fig: plt.Figure, ax: plt.Axes) -> None:
    """Right-shift and align A/C/E outer axes and their y-labels."""
    is_main = fig.get_figwidth() > 5.0
    delta = 0.020 if is_main else 0.015
    pos = ax.get_position()
    ax.set_position([pos.x0 + delta, pos.y0, max(0.01, pos.width - delta), pos.height])
    # Use one absolute x-coordinate for alignment, while keeping each label's
    # y-coordinate relative to its own axes so A/C/E cannot stack vertically.
    x_label = 0.060 if is_main else 0.145
    label_transform = blended_transform_factory(fig.transFigure, ax.transAxes)
    ax.yaxis.set_label_coords(x_label, 0.5, transform=label_transform)
    ax.spines["left"].set_color("#222222")
    ax.spines["bottom"].set_color("#222222")
    ax.grid(False)


def align_inner_ylabel(ax: plt.Axes) -> None:
    """Align an inner-panel ylabel and keep it clear of y tick labels."""
    ax.yaxis.set_label_coords(-0.30, 0.5, transform=ax.transAxes)


def legend_dynamics(ax: plt.Axes, *, ncol: int = 1, loc: str = "upper right") -> None:
    handles = [
        Line2D(
            [0], [0], color=DYN_COLORS[d], lw=1.15, linestyle=DYN_STYLES[d],
            marker=DYN_MARKERS[d], markersize=2.7, markerfacecolor="white",
            markeredgewidth=0.75, label="MM" if d == "Gene" else d,
        )
        for d in DYNAMICS
    ]
    ax.legend(handles=handles, loc=loc, ncol=ncol, fontsize=5.5, handlelength=1.8,
              columnspacing=0.9, borderaxespad=0.15)


def legend_methods(ax: plt.Axes, *, loc: str = "upper right") -> None:
    handles = [
        Line2D([0], [0], color=METHOD_COLORS[m], lw=1.25, linestyle="-",
               marker=METHOD_MARKERS[m], markersize=3.0, markerfacecolor="white",
               markeredgewidth=0.75, label=METHOD_DISPLAY[m])
        for m in METHODS
    ]
    ax.legend(handles=handles, loc=loc, fontsize=5.7, handlelength=1.8, borderaxespad=0.15)


def plot_dynamics(ax: plt.Axes, frame: pd.DataFrame, xlabel: str, ylabel: str = "sMAPE",
                  *, xlim=None, ylim=None, legend=True) -> None:
    for dynamics in DYNAMICS:
        sub = frame[frame["dynamics"] == dynamics].sort_values("x")
        if sub.empty:
            continue
        x = sub["x"].to_numpy(float)
        y = sub["mean"].to_numpy(float)
        s = sub["std"].to_numpy(float)
        ax.fill_between(x, np.maximum(y - s, 0), y + s, color=DYN_COLORS[dynamics], alpha=0.15, linewidth=0)
        ax.plot(x, y, color=DYN_COLORS[dynamics], lw=1.0, linestyle=DYN_STYLES[dynamics],
                marker=DYN_MARKERS[dynamics], markersize=2.3, markerfacecolor="white",
                markeredgewidth=0.55, label=dynamics)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    style_axis(ax)
    if legend:
        legend_dynamics(ax)


def plot_methods(ax: plt.Axes, frame: pd.DataFrame, xlabel: str, *, xlim=None, ylim=None,
                 xscale: str = "linear", yscale: str = "linear", categorical_x: bool = False) -> None:
    x_values = np.sort(frame["x"].dropna().unique())
    x_lookup = {value: i + 1 for i, value in enumerate(x_values)}
    for method in METHODS:
        sub = frame[frame["method"] == method].sort_values("x")
        if sub.empty:
            continue
        raw_x = sub["x"].to_numpy(float)
        x = np.asarray([x_lookup[v] for v in raw_x], dtype=float) if categorical_x else raw_x
        y = sub["mean"].to_numpy(float)
        s = sub["std"].to_numpy(float)
        ax.fill_between(x, y - s, y + s, color=METHOD_COLORS[method], alpha=0.14, linewidth=0)
        ax.plot(x, y, color=METHOD_COLORS[method], lw=1.1, marker=METHOD_MARKERS[method],
                markersize=2.5, markerfacecolor="white", markeredgewidth=0.55, label=method)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Runtime (s)" if yscale == "log" else "sMAPE")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xscale("linear" if categorical_x else xscale)
    ax.set_yscale(yscale)
    if categorical_x:
        ax.set_xticks(np.arange(1, len(x_values) + 1))
        if "x_label" in frame.columns:
            labels = (
                frame[["x", "x_label"]]
                .dropna()
                .drop_duplicates("x")
                .sort_values("x")["x_label"]
                .to_numpy()
            )
        else:
            labels = x_values
        ax.set_xticklabels([f"{int(v):g}" if float(v).is_integer() else f"{v:g}" for v in labels])
    # Matplotlib's default log formatter uses mathtext exponents, which can
    # render below the 5 pt floor in compact production PDFs. Plain numeric
    # labels preserve the log-axis meaning and remain fully editable.
    if xscale == "log":
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    if yscale == "log":
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    style_axis(ax)


def make_axes(fig: plt.Figure, subplotspec, ncols: int = 1, *, gap_fraction: float = 0.08) -> list[plt.Axes]:
    # Reserve the upper strip for the panel title and the lower strip for the
    # shared footnote/frame breathing room.  Without this inset, the compact
    # panel titles collide with legends at final figure size.
    parent = subplotspec.get_position(fig)
    x0 = parent.x0 + 0.085 * parent.width
    x1 = parent.x1 - 0.025 * parent.width
    y0 = parent.y0 + 0.115 * parent.height
    y1 = parent.y0 + 0.825 * parent.height
    gap = gap_fraction * parent.width if ncols == 2 else 0.0
    width = (x1 - x0 - gap * (ncols - 1)) / ncols
    return [fig.add_axes([x0 + i * (width + gap), y0, width, y1 - y0]) for i in range(ncols)]


def panel_a(fig: plt.Figure, ss) -> tuple[list[plt.Axes], dict]:
    ax = make_axes(fig, ss, 1)[0]
    frame = read_csv("noise_100k.csv")
    plot_dynamics(ax, frame, "SNR (dB)", xlim=(28, 72), ylim=(0, 1.05))
    move_outer_ylabel(fig, ax)
    return [ax], {"source": "noise_100k.csv", "rows": len(frame), "note": "SNR is plotted directly; high SNR denotes lower observation noise."}


def panel_b(fig: plt.Figure, ss) -> tuple[list[plt.Axes], dict]:
    # A wider inter-panel gap keeps both y-labels clear of the neighboring
    # plotting area at the compact main-figure size.
    axes = make_axes(fig, ss, 2, gap_fraction=0.16)
    points = read_csv("sampling_points_100k.csv")
    interval = read_csv("sampling_interval_100k.csv")
    plot_dynamics(axes[0], points, "Sampling points (n)", xlim=(150, 1050), ylim=(0, 0.0185), legend=False)
    plot_dynamics(axes[1], interval, "Sampling interval (Δt)", xlim=(0.005, 0.055), ylim=(0, 0.0365), legend=False)
    for ax in axes:
        align_inner_ylabel(ax)
    return axes, {"source": ["sampling_points_100k.csv", "sampling_interval_100k.csv"], "rows": [len(points), len(interval)],
                  "note": "Interval x is the original recorded column-A Delta t; mean and SD, and the resulting displayed bands, are original recorded values rather than estimates."}


def panel_c(fig: plt.Figure, ss) -> tuple[list[plt.Axes], dict]:
    axes = make_axes(fig, ss, 2)
    mul = read_csv("mul_errors_all_trajectories.csv")
    chua = read_csv("chua_errors_all_trajectories.csv")
    # Both distributions are plotted on their normal value scale. Chua is the
    # raw sqrt(sum of the three per-dimension MSEs) used by HR_c.m.
    mul_values = pd.to_numeric(mul["mse"], errors="coerce").to_numpy(float)
    mul_values = mul_values[np.isfinite(mul_values)]
    chua_values = pd.to_numeric(chua["matlab_error"], errors="coerce").to_numpy(float)
    chua_values = chua_values[np.isfinite(chua_values)]

    def plot_error_hist(ax: plt.Axes, values: np.ndarray, system: str, color: str, xlabel: str) -> dict:
        if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(f"{system} plotted error values must be finite")
        mean = float(np.mean(values))
        median = float(np.quantile(values, 0.50))
        bins = np.histogram_bin_edges(values, bins="auto")
        ax.hist(values, bins=bins, color=color, edgecolor="#333333", linewidth=0.30, alpha=0.7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")
        log_values = np.log(values)
        log_mu = float(np.mean(log_values))
        log_sigma = float(np.std(log_values, ddof=1))
        xi = np.linspace(float(np.min(values)), float(np.max(values)), 400)
        pdf = np.exp(-0.5 * ((np.log(xi) - log_mu) / log_sigma) ** 2) / (xi * log_sigma * np.sqrt(2 * np.pi))
        fit_color = "#1F5E9E" if system == "Mul" else "#2E8B57"
        ax.plot(xi, pdf * len(values) * float(np.mean(np.diff(bins))), color=fit_color, linewidth=1.25)
        ax.legend(
            handles=[
                Patch(facecolor=color, edgecolor="#333333", linewidth=0.30, alpha=0.7),
                Line2D([0], [0], color=fit_color, linewidth=1.25),
            ],
            labels=[f"{system}\nμ = {mean:.2e}", "Lognormal fit"],
            loc="upper right",
            fontsize=5.2,
            handlelength=1.3,
            borderpad=0.35,
            labelspacing=0.25,
        )
        style_axis(ax)
        return {"n": int(values.size), "mean": mean, "median": median}

    mul_stats = plot_error_hist(axes[0], mul_values, "Mul", "#4682B4", "MSE")
    chua_stats = plot_error_hist(axes[1], chua_values, "Chua", "#6A9EC0", "MSE")
    move_outer_ylabel(fig, axes[0])
    return axes, {"source": ["mul_errors_all_trajectories.csv", "chua_errors_all_trajectories.csv"], "rows": [len(mul), len(chua)],
                  "statistics": {"Mul": mul_stats, "Chua": chua_stats},
                  "note": "Mul and Chua use normal-scale values and lognormal fit curves; Chua uses raw sqrt(MSE_dim0+MSE_dim1+MSE_dim2)."}


def panel_d(fig: plt.Figure, ss) -> tuple[list[plt.Axes], dict]:
    axes = make_axes(fig, ss, 2, gap_fraction=0.16)
    nodes = read_csv("missing_nodes_100k.csv")
    links = read_csv("missing_links_100k.csv")
    plot_dynamics(axes[0], nodes, "Missing nodes (%)", xlim=(0, 21), ylim=(0, 0.70), legend=False)
    plot_dynamics(axes[1], links, "Missing links (%)", xlim=(0, 42), ylim=(0, 0.70), legend=False)
    for ax in axes:
        align_inner_ylabel(ax)
    return axes, {"source": ["missing_nodes_100k.csv", "missing_links_100k.csv"], "rows": [len(nodes), len(links)],
                  "note": "Missing-node percentage is derived from (100000 - observed nodes)/1000; missing-link percentage is source x*100."}


def panel_e(fig: plt.Figure, ss) -> tuple[list[plt.Axes], dict]:
    axes = make_axes(fig, ss, 2, gap_fraction=0.16)
    nodes = read_csv("comparison_missing_nodes.csv")
    links = read_csv("comparison_missing_links.csv")
    plot_methods(axes[0], nodes, "Missing nodes (%)", xlim=(0.5, 5.5), ylim=(-0.01, 0.4), categorical_x=True)
    axes[0].set_xticks(np.arange(1, 6))
    axes[0].set_xticklabels(["1", "5", "10", "15", "20"])
    plot_methods(axes[1], links, "Missing links (%)", xlim=(0, 42), ylim=(0, 0.41))
    move_outer_ylabel(fig, axes[0])
    align_inner_ylabel(axes[1])
    legend_methods(axes[1], loc="upper left")
    return axes, {"source": ["comparison_missing_nodes.csv", "comparison_missing_links.csv"], "rows": [len(nodes), len(links)],
                  "note": "J1 uses the exact MATLAB fig4k.m arrays; E1 displays missing-node percentages 1, 5, 10, 15 and 20; comparison bands are MATLAB mean ± std."}


def panel_f(fig: plt.Figure, ss) -> tuple[list[plt.Axes], dict]:
    axes = make_axes(fig, ss, 2, gap_fraction=0.16)
    smape = read_csv("comparison_smape_nodes.csv")
    runtime = read_csv("comparison_runtime.csv")
    plot_methods(axes[0], smape, "Number of nodes", xlim=(8, 130000), ylim=(0, 0.23), xscale="log")
    plot_methods(axes[1], runtime, "Number of nodes", xlim=(8, 130000), ylim=(30, 70000), xscale="log", yscale="log")
    for ax in axes:
        align_inner_ylabel(ax)
    # Compact scientific notation keeps the F2 runtime ticks legible at the
    # final figure size (e.g. 1e2, 1e3, 1e4).
    axes[1].yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:.0e}".replace("e+0", "e").replace("e-0", "e-"))
    )
    legend_methods(axes[1], loc="upper left")
    return axes, {"source": ["comparison_smape_nodes.csv", "comparison_runtime.csv"], "rows": [len(smape), len(runtime)],
                  "note": "Both axes use log-scaled node count; runtime y-axis is log-scaled and retains MATLAB-recorded SD."}


PANEL_BUILDERS = [panel_a, panel_b, panel_c, panel_d, panel_e, panel_f]
PANEL_TITLES = [
    "Observation noise",
    "Limited sampling",
    "Basis mismatch",
    "Missing network",
    "Baseline comparison",
    "Scalability",
]


def save_figure(fig: plt.Figure, path: Path, *, dpi: int = 600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight", facecolor="white")


def build_main() -> list[dict]:
    OUT.mkdir(parents=True, exist_ok=True)
    # Approx. 183 mm wide (double-column) while retaining the requested 1.4:1
    # aspect ratio.
    fig = plt.figure(figsize=(7.2, 4.95), facecolor="white")
    gs = fig.add_gridspec(3, 2, left=0.045, right=0.990, bottom=0.080, top=0.955,
                          wspace=0.06, hspace=0.12)
    audits: list[dict] = []
    for idx, (builder, title) in enumerate(zip(PANEL_BUILDERS, PANEL_TITLES)):
        row, col = divmod(idx, 2)
        ss = gs[row, col]
        if col == 0:
            left_bbox = gs[row, 0].get_position(fig)
            right_bbox = gs[row, 1].get_position(fig)
            # Extend the row frame slightly below the parent GridSpec cell so
            # the x-labels are enclosed with breathing room rather than
            # touching or crossing the border.
            row_bbox = Bbox.from_extents(left_bbox.x0, left_bbox.y0 - 0.015, right_bbox.x1, left_bbox.y1)
            add_row_frame(fig, row_bbox)
        add_panel_header(fig, ss.get_position(fig), chr(ord("a") + idx), title)
        axes, audit = builder(fig, ss)
        audit.update({"panel": chr(ord("a") + idx), "title": title})
        audits.append(audit)
    save_figure(fig, OUT / "Fig4_main")
    plt.close(fig)

    # Save each panel independently with the same visual language.
    for idx, (builder, title) in enumerate(zip(PANEL_BUILDERS, PANEL_TITLES)):
        fig_p = plt.figure(figsize=(3.55, 2.25), facecolor="white")
        ss_p = fig_p.add_gridspec(1, 1, left=0.12, right=0.96, bottom=0.16, top=0.88)[0, 0]
        bbox_p = ss_p.get_position(fig_p)
        add_frame(fig_p, bbox_p, chr(ord("a") + idx), title)
        builder(fig_p, ss_p)
        save_figure(fig_p, OUT / f"panel_{chr(ord('a') + idx)}")
        plt.close(fig_p)

    with (OUT / "numeric_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audits, handle, ensure_ascii=False, indent=2)
    pd.DataFrame(audits).to_csv(OUT / "numeric_audit.csv", index=False, encoding="utf-8-sig")
    if len(audits) >= 3 and "statistics" in audits[2]:
        statistic_rows = []
        for system, values in audits[2]["statistics"].items():
            statistic_rows.append({"system": system, **values})
        pd.DataFrame(statistic_rows).to_csv(OUT / "c_mse_statistics.csv", index=False, encoding="utf-8-sig")

    return audits


if __name__ == "__main__":
    result = build_main()
    print(f"Generated Fig. 4 main and six standalone panels under {OUT}")
    for item in result:
        print(item["panel"], item["title"], item["rows"])

