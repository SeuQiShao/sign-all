"""Render Fig. 1b as a standalone, final-size panel."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches


MM = 1.0 / 25.4
width_mm = 93.696
height_mm = 42.136

BLUE = "#2369B3"
BLUE_DARK = "#124A86"
BLUE_FILL = "#F5F9FD"
PURPLE = "#7650A5"
PURPLE_DARK = "#56317F"
PURPLE_FILL = "#FAF7FC"
TITLE_BLUE = "#4F8FC3"
EDGE = "#666B70"
MID = "#8D9398"
INK = "#151719"

mpl.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif", "serif"],
        "font.sans-serif": ["Arial", "Helvetica", "sans-serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "mathtext.fallback": "stix",
        "font.size": 5.6,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


SELF_TERMS = [
    r"$1$", r"$xᵢ$", r"$yᵢ$", r"$zᵢ$", r"$xᵢ²$",
    r"$yᵢ²$", r"$zᵢ²$", r"$xᵢ yᵢ$", r"$xᵢ zᵢ$", r"$yᵢ zᵢ$",
    r"$xᵢ³$", r"$yᵢ³$", r"$zᵢ³$", r"$xᵢ² yᵢ$", r"$xᵢ yᵢ²$",
    r"$\sin(xᵢ)$", r"$\sin(yᵢ)$", r"$\cos(zᵢ)$", r"$e^{xᵢ}$", "…",
]

COUPLING_TERMS = [
    r"$xᵢ\!-\!xⱼ$", r"$yᵢ\!-\!yⱼ$", r"$zᵢ\!-\!zⱼ$",
    r"$(xᵢ\!-\!xⱼ)²$", r"$(yᵢ\!-\!yⱼ)²$",
    r"$(zᵢ\!-\!zⱼ)²$", r"$\sin(xᵢ\!-\!xⱼ)$",
    r"$\sin(yᵢ\!-\!yⱼ)$", r"$\sin(zᵢ\!-\!zⱼ)$",
    r"$e^{xⱼ\!-\!xᵢ}$",
    r"$e^{yⱼ\!-\!yᵢ}$", r"$e^{zⱼ\!-\!zᵢ}$",
    r"$\cos(xᵢ\!-\!xⱼ)$", r"$\cos(yᵢ\!-\!yⱼ)$",
    r"$\cos(zᵢ\!-\!zⱼ)$",
    r"$e^{(xⱼ\!-\!xᵢ)²}$", r"$e^{(yⱼ\!-\!yᵢ)²}$",
    r"$e^{(zⱼ\!-\!zᵢ)²}$", r"$-\sin(xᵢ\!-\!xⱼ)$", "…",
]


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
        default=Path(__file__).with_name("panel_b_output"),
    )
    return parser.parse_args()


def load_library_terms(plot_data_dir: Path) -> tuple[list[str], list[str]]:
    """Read the two candidate-library grids from a public CSV table."""
    terms: dict[str, list[str]] = {"self": [], "coupling": []}
    with (plot_data_dir / "library_terms.csv").open(encoding="utf-8", newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: (row["library"], int(row["index"])))
        for row in rows:
            terms[row["library"]].append(row["expression"])
    if len(terms["self"]) != 20 or len(terms["coupling"]) != 20:
        raise ValueError("library_terms.csv must contain 20 self and 20 coupling terms")
    return terms["self"], terms["coupling"]


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
            (x, y),
            w,
            h,
            transform=ax.transAxes,
            boxstyle=f"round,pad=0.004,rounding_size={radius}",
            facecolor=face,
            edgecolor=edge,
            linewidth=linewidth,
            clip_on=False,
        )
    )


def draw_formula_grid(
    ax: plt.Axes,
    bounds: tuple[float, float, float, float],
    terms: list[str],
    *,
    color: str,
    fill: str,
    fontsize: float,
) -> None:
    if len(terms) != 20:
        raise ValueError("Each candidate library must contain exactly 20 cells")
    x, y, w, h = bounds
    rows, cols = 4, 5
    cell_w = w / cols
    cell_h = h / rows
    for row in range(rows):
        for col in range(cols):
            index = row * cols + col
            xx = x + col * cell_w
            yy = y + (rows - 1 - row) * cell_h
            face = fill if row % 2 == 0 else "white"
            ax.add_patch(
                patches.Rectangle(
                    (xx, yy),
                    cell_w,
                    cell_h,
                    transform=ax.transAxes,
                    facecolor=face,
                    edgecolor=color,
                    linewidth=0.58,
                )
            )
            ax.text(
                xx + cell_w / 2,
                yy + cell_h / 2,
                terms[index],
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=7.2 if terms[index].startswith("$e^") else fontsize,
                color=INK,
                fontfamily="Times New Roman",
                linespacing=0.90,
            )


def make_panel(terms: tuple[list[str], list[str]] | None = None) -> plt.Figure:
    if terms is None:
        terms = (SELF_TERMS, COUPLING_TERMS)
    fig = plt.figure(figsize=(width_mm * MM, height_mm * MM), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.017, 0.945, "b", transform=ax.transAxes, ha="left", va="center",
            fontsize=9.5, fontweight="bold", fontfamily="Arial", color="#1C1C1C")
    ax.text(0.500, 0.945, "Broad candidate function libraries",
            transform=ax.transAxes, ha="center", va="center", fontsize=8.6,
            fontweight="normal", fontfamily="Arial", color="#1F5E9E")

    ax.text(0.215, 0.810, "Self-dynamics library",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=6.15, fontweight="bold", color=BLUE_DARK)
    ax.text(0.370, 0.810, r"$θ_{\mathrm{f}}(·)$",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=7.2, fontweight="bold", color=BLUE_DARK)
    ax.text(0.705, 0.810, "Coupling library",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=6.15, fontweight="bold", color=PURPLE_DARK)
    ax.text(0.844, 0.810, r"$θ_{\mathrm{c}}(·,·)$",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=7.2, fontweight="bold", color=PURPLE_DARK)

    draw_formula_grid(
        ax, (0.025, 0.110, 0.455, 0.610), terms[0],
        color=BLUE, fill=BLUE_FILL, fontsize=5.35,
    )
    draw_formula_grid(
        ax, (0.520, 0.110, 0.455, 0.610), terms[1],
        color=PURPLE, fill=PURPLE_FILL, fontsize=5.05,
    )
    return fig


def save_panel(fig: plt.Figure, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / "Fig1b_candidate_libraries_rebuild"
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
    save_panel(make_panel(load_library_terms(args.plot_data)), args.output_dir)
    print(f"Saved rebuilt panel b to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
