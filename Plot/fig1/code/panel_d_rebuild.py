"""Render Fig. 1d as a standalone, final-size panel.

The network topology and coordinates are imported from the canonical panel-a
master.  The Phase-II coefficient bars are read directly from
``fig1f/rossler_coef_bar.xlsx`` using only Python's standard library.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import zipfile
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.colors import to_rgb
import networkx as nx
import numpy as np

from panel_a_rebuild import build_network_master


MM = 1.0 / 25.4
WIDTH_MM = 106.872
HEIGHT_MM = 43.783

BLUE = "#2369B3"
BLUE_DARK = "#124A86"
BLUE_FILL = "#D3E2F1"
PURPLE = "#7650A5"
PURPLE_DARK = "#56317F"
PURPLE_FILL = "#E3D9EC"
ORANGE = "#E97824"
ORANGE_DARK = "#B9540E"
TITLE_BLUE = "#4F8FC3"
GREEN = "#2C9A51"
GREEN_DARK = "#19743A"
GREY = "#B9BEC2"
GREY_LIGHT = "#E1E4E6"
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
    "mathtext.cal": "Times New Roman:italic",
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
    "hatch.linewidth": 0.35,
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coefficients",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "plot_data" / "coefficients.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("panel_d_output"),
    )
    return parser.parse_args()


def _column_index(cell_reference: str) -> int:
    letters = re.match(r"[A-Z]+", cell_reference)
    if letters is None:
        raise ValueError(f"Invalid Excel cell reference: {cell_reference}")
    value = 0
    for letter in letters.group(0):
        value = value * 26 + ord(letter) - ord("A") + 1
    return value - 1


def read_xlsx_rows(path: Path) -> list[list[object]]:
    """Read the first XLSX sheet without adding an Excel-engine dependency."""
    if not path.exists():
        raise FileNotFoundError(path)
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as workbook:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            for item in root.findall("x:si", namespace):
                shared_strings.append(
                    "".join(text.text or "" for text in item.findall(".//x:t", namespace))
                )

        sheet = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
        rows: list[list[object]] = []
        for row_node in sheet.findall(".//x:sheetData/x:row", namespace):
            cells: dict[int, object] = {}
            for cell in row_node.findall("x:c", namespace):
                ref = cell.attrib["r"]
                value_node = cell.find("x:v", namespace)
                if value_node is None:
                    value: object = None
                elif cell.attrib.get("t") == "s":
                    value = shared_strings[int(value_node.text or "0")]
                elif cell.attrib.get("t") == "b":
                    value = (value_node.text == "1")
                else:
                    raw = value_node.text or ""
                    value = float(raw) if any(char in raw for char in ".eE") else int(raw)
                cells[_column_index(ref)] = value
            width = max(cells, default=-1) + 1
            rows.append([cells.get(index) for index in range(width)])
    return rows


def load_coefficients(
    path: Path,
) -> list[tuple[str, dict[str, tuple[list[str], np.ndarray]]]]:
    if path.suffix.lower() == ".csv":
        order = ["BA–1k", "BA–100k", "Human", "Fly", "Truth"]
        terms = {
            "x": ["yᵢ", "zᵢ", "xⱼ−xᵢ"],
            "y": ["xᵢ", "yᵢ"],
            "z": ["1", "zᵢ", "xᵢzᵢ"],
        }
        values: dict[str, dict[str, list[tuple[int, float]]]] = {}
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                values.setdefault(row["series"], {}).setdefault(row["dimension"], []).append(
                    (int(row["slot"]), float(row["value"]))
                )
        return [
            (
                series,
                {
                    dimension: (
                        terms[dimension],
                        np.asarray([value for _, value in sorted(values[series][dimension])], dtype=float),
                    )
                    for dimension in terms
                },
            )
            for series in order
        ]
    rows = read_xlsx_rows(path)
    if not rows or rows[0][:5] != ["Dynamics", "Network", "nodes", "inter", "samples"]:
        raise ValueError("Unexpected Rössler coefficient workbook structure")
    row_specs = [
        ("BA–1k", lambda row: row[1] == "BA" and row[2] == "1k"),
        ("BA–100k", lambda row: row[1] == "BA" and row[2] == "100k"),
        ("Human", lambda row: row[1] == "bn-hunman" and row[2] == "17.8k"),
        ("Fly", lambda row: row[1] == "bn-fly" and row[2] == "2k"),
        ("Truth", lambda row: row[1] is True),
    ]

    groups = {
        "x": (["yᵢ", "zᵢ", "xⱼ−xᵢ"], slice(0, 3)),
        "y": (["xᵢ", "yᵢ"], slice(3, 5)),
        "z": (["1", "zᵢ", "xᵢzᵢ"], slice(5, 8)),
    }
    series: list[tuple[str, dict[str, tuple[list[str], np.ndarray]]]] = []
    for series_name, selector in row_specs:
        row = next(
            row for row in rows[1:]
            if row[0] == "Rossler" and selector(row)
        )
        values = np.asarray(row[5:13], dtype=float)
        if values.shape != (8,) or not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid active Rössler coefficients for {series_name}")
        grouped = {
            dimension: (labels, values[index])
            for dimension, (labels, index) in groups.items()
        }
        series.append((series_name, grouped))
    return series


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


def draw_message_network(parent: plt.Axes) -> None:
    ax = parent.inset_axes([0.004, 0.095, 0.272, 0.655])
    graph, _, target, neighbors = build_network_master()
    # Preserve the canonical topology but open a larger message-passing cavity
    # around i and spread the outer context accordingly.
    pos = {
        0: (0.00, 0.00),
        1: (-1.35, 0.95), 2: (0.00, 1.65), 3: (1.55, 0.65),
        4: (0.95, -1.45), 5: (-1.15, -1.35),
        6: (-2.40, 1.65), 7: (-2.35, -0.10),
        8: (-1.00, 2.65), 9: (0.75, 2.70),
        10: (2.50, 1.65), 11: (2.65, -0.35),
        12: (1.20, -2.55), 13: (-1.35, -2.60),
        14: (-3.30, 2.40), 15: (-1.40, 3.55), 16: (3.35, 2.25),
        17: (3.50, -1.10), 18: (-2.05, -3.50),
    }
    continuation = list(range(14, 19))
    ordinary = [
        node for node in graph
        if node not in neighbors and node != target and node not in continuation
    ]
    continuation_edges = [
        edge for edge in graph.edges
        if edge[0] in continuation or edge[1] in continuation
    ]
    solid_edges = [edge for edge in graph.edges if edge not in continuation_edges]

    ax.add_patch(
        patches.Circle(
            (0.0, 0.0), 2.08, fill=False,
            edgecolor="#7C6F69", linewidth=0.55,
            linestyle=(0, (4.0, 3.0)), zorder=0,
        )
    )

    nx.draw_networkx_edges(
        graph, pos, edgelist=solid_edges, ax=ax,
        edge_color="#B9CAD7", width=0.48,
    )
    nx.draw_networkx_edges(
        graph, pos, edgelist=continuation_edges, ax=ax,
        edge_color="#BFC4C8", width=0.48, style=(0, (2.0, 2.0)),
    )
    nx.draw_networkx_nodes(
        graph, pos, nodelist=ordinary, ax=ax, node_color="#D9E7E8",
        node_size=17, edgecolors="#83979E", linewidths=0.40,
    )
    nx.draw_networkx_nodes(
        graph, pos, nodelist=neighbors, ax=ax, node_color=PURPLE,
        node_size=27, edgecolors=PURPLE_DARK, linewidths=0.48,
    )
    nx.draw_networkx_nodes(
        graph, pos, nodelist=[target], ax=ax, node_color=BLUE,
        node_size=43, edgecolors=BLUE_DARK, linewidths=0.60,
    )
    continuation_artist = nx.draw_networkx_nodes(
        graph, pos, nodelist=continuation, ax=ax, node_color="white",
        node_size=10, edgecolors=MID, linewidths=0.42,
    )
    continuation_artist.set_linestyle((0, (2.0, 1.8)))
    nx.draw_networkx_labels(
        graph, pos, labels={target: "i", 4: "j"}, ax=ax,
        font_size=5.0, font_color="white", font_weight="bold",
        font_family="Times New Roman",
    )

    # Structural edges stay gray; message flow is overlaid as curved arrows.
    # Bend the upper-left coupling path away from the external self-loop.
    curvatures = [0.38, 0.24, -0.26, 0.31, -0.34]
    for node, curvature in zip(neighbors, curvatures):
        ax.annotate(
            "",
            xy=pos[target],
            xytext=pos[node],
            arrowprops={
                "arrowstyle": "-|>",
                "color": PURPLE_DARK,
                "lw": 0.70 if node == 4 else 0.54,
                "mutation_scale": 5.6,
                "shrinkA": 4.8,
                "shrinkB": 6.0,
                "connectionstyle": f"arc3,rad={curvature}",
            },
            zorder=6,
        )

    # The self-message is represented by an external loop so it cannot be
    # mistaken for a loop drawn directly around the target node.
    ax.add_patch(
        patches.Arc(
            (0.00, -0.68), 0.72, 0.58, theta1=125, theta2=415,
            linewidth=0.68, color=BLUE, zorder=8,
        )
    )
    ax.add_patch(
        patches.FancyArrowPatch(
            (0.35, -0.52), (0.21, -0.44),
            arrowstyle="-|>", mutation_scale=6.4,
            linewidth=0.68, color=BLUE, zorder=9,
        )
    )
    ax.text(
        0.00, -1.37, "MSGᵢᵢ", ha="center", va="center",
        fontsize=5.0, color=BLUE, zorder=11,
    )
    ax.text(
        1.42, -0.25, "MSGᵢⱼ", ha="center", va="center",
        fontsize=5.0, fontweight="bold", color=PURPLE_DARK, zorder=11,
    )

    ax.set_xlim(-3.62, 3.72)
    ax.set_ylim(-3.68, 3.72)
    ax.set_aspect("equal")
    ax.axis("off")

    parent.text(
        0.145, 0.075, r"$j_1,\ldots,j_k\in\mathcal{N}_i$",
        transform=parent.transAxes, ha="center", va="center",
        fontsize=7.6, color=INK,
    )


def blend_with_white(color: str, strength: float) -> tuple[float, float, float]:
    rgb = np.asarray(to_rgb(color))
    strength = float(np.clip(strength, 0.0, 1.0))
    return tuple(1.0 - strength * (1.0 - rgb))


def draw_cell_strip(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    color: str,
    values: np.ndarray | None = None,
    active: np.ndarray | None = None,
    cell_labels: list[str] | None = None,
    solid_face: str | None = None,
    edge_color: str | None = None,
) -> None:
    count = 6
    if values is not None and values.shape != (count,):
        raise ValueError("Coefficient strips require six cells")
    if active is not None and active.shape != (count,):
        raise ValueError("Mask strips require six cells")
    cell_width = width / count
    for index in range(count):
        if solid_face is not None:
            face = solid_face
        elif active is not None:
            face = color if active[index] else "white"
        elif values is not None:
            normalized = abs(float(values[index])) / max(float(np.max(np.abs(values))), 1e-12)
            face = blend_with_white(color, 0.25 + 0.72 * normalized)
        else:
            face = "white"
        ax.add_patch(
            patches.Rectangle(
                (x + index * cell_width, y - height / 2),
                cell_width,
                height,
                transform=ax.transAxes,
                facecolor=face,
                edgecolor=edge_color if edge_color is not None else MID,
                linewidth=0.44,
                zorder=2,
            )
        )
    if cell_labels is not None:
        if len(cell_labels) != count:
            raise ValueError("Library strips require six labels")
        for index, cell_label in enumerate(cell_labels):
            ax.text(
                x + (index + 0.5) * cell_width, y, cell_label,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=5.0, color=color, zorder=4,
            )


def draw_message_row(
    ax: plt.Axes,
    *,
    y: float,
    row_label: str,
    color: str,
    library_color: str,
    library_fill: str,
    mask: np.ndarray,
    coefficients: np.ndarray,
    row_label_size: float = 5.2,
    row_label_style: str = "italic",
) -> None:
    ax.text(0.366, y, row_label, transform=ax.transAxes,
            ha="right", va="center", fontsize=row_label_size, color=color,
            fontfamily="Times New Roman", fontstyle=row_label_style)
    draw_cell_strip(ax, x=0.374, y=y, width=0.086, height=0.046,
                    color=color, active=mask)
    ax.text(0.469, y, r"$\odot$", transform=ax.transAxes,
            ha="center", va="center", fontsize=6.5, color=INK,
            fontfamily="Times New Roman")
    draw_cell_strip(ax, x=0.478, y=y, width=0.086, height=0.046,
                    color=color, values=coefficients)
    ax.text(0.573, y, r"$\odot$", transform=ax.transAxes,
            ha="center", va="center", fontsize=6.5, color=INK,
            fontfamily="Times New Roman")
    draw_cell_strip(ax, x=0.582, y=y, width=0.086, height=0.046,
                    color=library_color, solid_face=library_fill,
                    edge_color=library_color)


def draw_masked_messages(ax: plt.Axes) -> None:
    ax.text(0.417, 0.720, "Mask", transform=ax.transAxes,
            ha="center", va="center", fontsize=6.2, fontweight="normal", color=INK,
            fontfamily="Arial")
    ax.text(0.521, 0.720, "Coefficient", transform=ax.transAxes,
            ha="center", va="center", fontsize=6.2, fontweight="normal", color=INK,
            fontfamily="Arial")
    ax.text(0.625, 0.720, "Library", transform=ax.transAxes,
            ha="center", va="center", fontsize=6.2, fontweight="normal", color=INK,
            fontfamily="Arial")

    self_mask = np.asarray([1, 1, 0, 0, 1, 0], dtype=bool)
    coupling_mask = np.asarray([1, 0, 1, 0, 0, 1], dtype=bool)
    self_coefficients = np.asarray([0.92, 0.78, 0.18, 0.10, 0.55, 0.20])
    coupling_coefficients = np.asarray([0.82, 0.15, 0.63, 0.09, 0.22, 0.74])
    draw_message_row(
        ax, y=0.625, row_label="MSGᵢᵢ", color=BLUE,
        library_color=BLUE, library_fill=BLUE_FILL,
        mask=self_mask, coefficients=self_coefficients,
        row_label_size=5.5, row_label_style="normal",
    )

    ax.text(0.314, 0.305, "MSGᵢⱼ", transform=ax.transAxes,
            ha="center", va="center", fontsize=5.2, rotation=90,
            rotation_mode="anchor",
            color=PURPLE_DARK, fontweight="bold",
            fontfamily="Times New Roman", fontstyle="normal")
    ax.plot([0.345, 0.345], [0.122, 0.488], transform=ax.transAxes,
            color=PURPLE_DARK, lw=0.55)
    ax.plot([0.345, 0.353], [0.488, 0.488], transform=ax.transAxes,
            color=PURPLE_DARK, lw=0.55)
    ax.plot([0.345, 0.353], [0.122, 0.122], transform=ax.transAxes,
            color=PURPLE_DARK, lw=0.55)
    rows = [
        (0.465, "j₁"),
        (0.305, "j₂"),
        (0.145, "jₖ"),
    ]
    for y, row_label in rows:
        draw_message_row(
            ax, y=y, row_label=row_label, color=PURPLE,
            library_color=PURPLE, library_fill=PURPLE_FILL,
            mask=coupling_mask, coefficients=coupling_coefficients,
        )
    ax.text(0.366, 0.225, r"$\vdots$", transform=ax.transAxes,
            ha="right", va="center", fontsize=7.4, color=PURPLE_DARK,
            fontfamily="Times New Roman")


def coefficient_bars(
    parent: plt.Axes,
    bounds: tuple[float, float, float, float],
    series: list[tuple[str, dict[str, tuple[list[str], np.ndarray]]]],
    color: str,
    dimension: str,
    *,
    show_legend: bool = False,
) -> None:
    if len(series) != 5:
        raise ValueError("Each function term requires four network bars and one truth bar")
    labels = series[0][1][dimension][0]
    value_rows = []
    for _, grouped in series:
        row_labels, row_values = grouped[dimension]
        if row_labels != labels:
            raise ValueError("All coefficient series must use identical function terms")
        value_rows.append(row_values)
    values = np.vstack(value_rows)
    ax = parent.inset_axes(bounds)
    x = np.arange(len(labels), dtype=float)
    ax.axhline(0.0, color=MID, lw=0.55, zorder=1)
    bar_width = 0.135
    offsets = (np.arange(5) - 2.0) * (bar_width + 0.012)
    series_colors = ["#2F6FA6", "#3F8C89", "#6B72B5", "#89AFC2", "#B98573"]
    series_edges = ["#194C76", "#286B68", "#4D548D", "#5E8799", "#7C5648"]
    for index in range(5):
        ax.bar(
            x + offsets[index], values[index], width=bar_width,
            color=series_colors[index], edgecolor=series_edges[index],
            linewidth=0.38 if index < 4 else 0.58, zorder=2,
        )
    combined = np.concatenate([values.ravel(), np.asarray([0.0])])
    low = float(combined.min())
    high = float(combined.max())
    span = max(high - low, 0.20)
    ax.set_ylim(low - 0.14 * span, high + 0.15 * span)
    # All dimensions share three function slots and identical bar geometry.
    ax.set_xlim(-0.58, 2.65)
    ax.set_xticks(x)
    ax.set_xticklabels(
        labels, fontsize=5.4, fontfamily="Times New Roman",
        fontstyle="italic",
    )
    ax.tick_params(axis="x", length=0, pad=0.8)
    ax.tick_params(axis="y", length=1.8, width=0.50, pad=1.2, labelsize=5.1)
    for tick_label in ax.get_yticklabels():
        tick_label.set_fontfamily("Times New Roman")
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=2))
    ax.spines["left"].set_color(EDGE)
    ax.spines["bottom"].set_color(EDGE)
    ax.spines["left"].set_linewidth(0.55)
    ax.spines["bottom"].set_linewidth(0.55)
    # Extend the vertical axis above the dimension letter for a clear axis cap.
    ax.plot(
        [0.0, 0.0], [1.0, 1.20], transform=ax.transAxes,
        color=EDGE, linewidth=0.55, clip_on=False, zorder=5,
    )
    dimension_y = 1.020 if dimension in {"x", "z"} else 1.000
    dimension_va = "bottom" if dimension in {"x", "z"} else "top"
    ax.text(
        0.015, dimension_y, dimension, transform=ax.transAxes,
        ha="left", va=dimension_va, fontsize=5.7,
        fontweight="bold", color=INK,
        fontfamily="Times New Roman", fontstyle="italic",
        clip_on=False,
    )
    if show_legend:
        legend_labels = ["BA-1K", "BA-100K", "Human", "Fly", "Truth"]
        legend_columns = [
            ((0.395, 0.435), [(0, 0.78), (1, 0.51)]),
            ((0.770, 0.810), [(2, 0.78), (3, 0.55), (4, 0.32)]),
        ]
        for (marker_x, text_x), items in legend_columns:
            for index, legend_y in items:
                ax.scatter(
                    [marker_x], [legend_y], transform=ax.transAxes,
                    marker="s", s=7.8,
                    facecolor=series_colors[index],
                    edgecolor=series_edges[index],
                    linewidth=0.42, zorder=8,
                )
                ax.text(
                    text_x, legend_y, legend_labels[index],
                    transform=ax.transAxes, ha="left", va="center",
                    fontsize=5.0, color=INK, zorder=8,
                    fontfamily="Arial", fontweight="normal",
                )


def make_panel(
    series: list[tuple[str, dict[str, tuple[list[str], np.ndarray]]]],
) -> plt.Figure:
    fig = plt.figure(figsize=(WIDTH_MM * MM, HEIGHT_MM * MM), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.020, 0.955, "d", transform=ax.transAxes,
            ha="left", va="top", fontsize=9.5, fontweight="bold",
            fontfamily="Arial", color="#1C1C1C")
    ax.text(
        0.500, 0.952, "Phase II: shared coefficient estimation",
        transform=ax.transAxes, ha="center", va="top", fontsize=8.6,
        fontweight="normal", fontfamily="Arial", color="#1F5E9E",
    )

    ax.text(0.028, 0.840, "Full-graph message passing",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=6.2, fontfamily="Arial", fontweight="normal", color=INK)
    ax.text(0.495, 0.840, "Masked message construction",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=6.2, fontfamily="Arial", fontweight="normal", color=INK)
    ax.text(0.720, 0.840, "Shared coefficients",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=6.2, fontfamily="Arial", fontweight="normal", color=INK)

    draw_message_network(ax)
    draw_masked_messages(ax)

    coefficient_bars(ax, (0.735, 0.575, 0.250, 0.150),
                     series, BLUE, "x")
    coefficient_bars(ax, (0.735, 0.085, 0.250, 0.150),
                     series, ORANGE, "z")
    coefficient_bars(ax, (0.735, 0.335, 0.250, 0.190),
                     series, PURPLE, "y", show_legend=True)
    return fig


def save_outputs(fig: plt.Figure, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / "Fig1d_shared_coefficients_rebuild"
    fig.savefig(prefix.with_suffix(".svg"), bbox_inches=None, pad_inches=0)
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches=None, pad_inches=0)
    fig.savefig(prefix.with_suffix(".png"), dpi=450, bbox_inches=None, pad_inches=0)
    fig.savefig(prefix.with_suffix(".tiff"), dpi=600, bbox_inches=None, pad_inches=0)


def main() -> None:
    args = parse_args()
    series = load_coefficients(args.coefficients)
    fig = make_panel(series)
    save_outputs(fig, args.output_dir)
    plt.close(fig)


if __name__ == "__main__":
    main()
