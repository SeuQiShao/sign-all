"""Assemble the Fig. 1 panels at the final 183-mm canvas size."""

from __future__ import annotations

import argparse
import copy
import io
import re
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:  # Vector-only assembly can run with the lightweight runtime.
    class _MatplotlibPlaceholder:
        rcParams = {}
    mpl = _MatplotlibPlaceholder()
    plt = None
try:
    from pypdf import PageObject, PdfReader, PdfWriter, Transformation
except ImportError:  # Keep the renderer usable before optional vector tools are installed.
    PageObject = PdfReader = PdfWriter = Transformation = None
try:
    from reportlab.pdfgen import canvas as reportlab_canvas
except ImportError:
    reportlab_canvas = None


MM = 1.0 / 25.4
WIDTH_MM = 183.0
HEIGHT_MM = 133.2045
_MARGIN_Y_MM = 1.647
_ROW_GAP_MM = 2.196
_TOP_HEIGHT_MM = 42.13575
_C_HEIGHT_MM = 39.600
_BOTTOM_HEIGHT_MM = 43.78275
_BOTTOM_Y = _MARGIN_Y_MM / HEIGHT_MM
_C_Y = (_MARGIN_Y_MM + _BOTTOM_HEIGHT_MM + _ROW_GAP_MM) / HEIGHT_MM
_TOP_Y = (_MARGIN_Y_MM + _BOTTOM_HEIGHT_MM + _ROW_GAP_MM + _C_HEIGHT_MM + _ROW_GAP_MM) / HEIGHT_MM
PANEL_SPECS = {
    "a": [0.012, _TOP_Y, 0.456, _TOP_HEIGHT_MM / HEIGHT_MM],
    "b": [0.476, _TOP_Y, 0.512, _TOP_HEIGHT_MM / HEIGHT_MM],
    "c": [0.012, _C_Y, 0.976, _C_HEIGHT_MM / HEIGHT_MM],
    "d": [0.012, _BOTTOM_Y, 0.584, _BOTTOM_HEIGHT_MM / HEIGHT_MM],
    "e": [0.604, _BOTTOM_Y, 0.384, _BOTTOM_HEIGHT_MM / HEIGHT_MM],
}
FRAME_SPECS = (
    (0.012, _TOP_Y, 0.976, _TOP_HEIGHT_MM / HEIGHT_MM),
    (0.012, _C_Y, 0.976, _C_HEIGHT_MM / HEIGHT_MM),
    (0.012, _BOTTOM_Y, 0.976, _BOTTOM_HEIGHT_MM / HEIGHT_MM),
)
FRAME_INSET = 0.003
FRAME_RADIUS = 0.012
FRAME_COLOR = "#2A2A2A"
FRAME_LINEWIDTH = 0.75

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        "font.size": 6.0,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    for letter, filename in (
        ("a", "Fig1a_Rossler_rebuild.png"),
        ("b", "Fig1b_candidate_libraries_rebuild.png"),
        ("c", "Fig1c_support_identification_rebuild.png"),
        ("d", "Fig1d_shared_coefficients_rebuild.png"),
        ("e", "Fig1e_prediction_performance_rebuild.png"),
    ):
        parser.add_argument(
            f"--panel-{letter}",
            type=Path,
            default=root / "code" / "panel_outputs" / letter / filename,
        )
    parser.add_argument("--output-dir", type=Path, default=root / "output")
    return parser.parse_args()


def _load(path: Path, letter: str):
    if not path.exists():
        raise FileNotFoundError(path)
    image = plt.imread(path)
    slot = PANEL_SPECS[letter]
    image_ratio = image.shape[1] / image.shape[0]
    slot_ratio = (WIDTH_MM * slot[2]) / (HEIGHT_MM * slot[3])
    if abs(image_ratio / slot_ratio - 1.0) > 0.005:
        raise ValueError(f"Panel {letter} aspect ratio does not match its final slot")
    return image


def make_figure(panel_paths: dict[str, Path]) -> plt.Figure:
    fig = plt.figure(figsize=(WIDTH_MM * MM, HEIGHT_MM * MM), facecolor="white")
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.set_xlim(0, 1)
    canvas.set_ylim(0, 1)
    canvas.axis("off")
    for letter, bounds in PANEL_SPECS.items():
        panel_ax = canvas.inset_axes(bounds)
        panel_ax.set_xlim(0, 1)
        panel_ax.set_ylim(0, 1)
        panel_ax.axis("off")
        panel_ax.imshow(
            _load(panel_paths[letter], letter),
            extent=(0, 1, 0, 1),
            origin="upper",
            interpolation="antialiased",
            aspect="auto",
        )
    for x, y, width, height in FRAME_SPECS:
        canvas.add_patch(
            mpl.patches.FancyBboxPatch(
                (x + FRAME_INSET, y + FRAME_INSET),
                width - 2 * FRAME_INSET,
                height - 2 * FRAME_INSET,
                transform=canvas.transAxes,
                boxstyle=f"round,pad=0.006,rounding_size={FRAME_RADIUS}",
                facecolor="none",
                edgecolor=FRAME_COLOR,
                linewidth=FRAME_LINEWIDTH,
                zorder=20,
                clip_on=False,
            )
        )
    return fig


def _frame_overlay_page(width_pt: float, height_pt: float):
    if reportlab_canvas is None:
        raise RuntimeError("reportlab is required for vector frame composition")
    buffer = io.BytesIO()
    overlay = reportlab_canvas.Canvas(buffer, pagesize=(width_pt, height_pt))
    overlay.setStrokeColorRGB(42 / 255, 42 / 255, 42 / 255)
    overlay.setLineWidth(FRAME_LINEWIDTH)
    for x, y, width, height in FRAME_SPECS:
        overlay.roundRect(
            width_pt * (x + FRAME_INSET),
            height_pt * (y + FRAME_INSET),
            width_pt * (width - 2 * FRAME_INSET),
            height_pt * (height - 2 * FRAME_INSET),
            min(width_pt, height_pt) * FRAME_RADIUS,
            stroke=1,
            fill=0,
        )
    overlay.save()
    buffer.seek(0)
    return PdfReader(buffer).pages[0]


def _source_viewbox(path: Path) -> tuple[float, float]:
    root = ET.parse(path).getroot()
    values = [float(value) for value in root.attrib["viewBox"].replace(",", " ").split()]
    return values[2], values[3]


def _prefixed_svg(root: ET.Element, prefix: str) -> list[ET.Element]:
    cloned = copy.deepcopy(root)
    ids = {
        element.attrib["id"]: f"{prefix}{element.attrib['id']}"
        for element in cloned.iter()
        if "id" in element.attrib
    }
    for element in cloned.iter():
        for key, value in list(element.attrib.items()):
            for old, new in ids.items():
                value = value.replace(f"url(#{old})", f"url(#{new})")
                value = value.replace(f"#{old}", f"#{new}")
            element.attrib[key] = value
        if element.text and element.tag.endswith("style"):
            for old, new in ids.items():
                element.text = element.text.replace(f"#{old}", f"#{new}")
    return list(cloned)


def compose_vector_pdf(panel_paths: dict[str, Path], output_path: Path) -> None:
    if PageObject is None:
        raise RuntimeError("pypdf is required for vector PDF composition")
    width_pt = WIDTH_MM * MM * 72.0
    height_pt = HEIGHT_MM * MM * 72.0
    page = PageObject.create_blank_page(width=width_pt, height=height_pt)
    for letter, bounds in PANEL_SPECS.items():
        source_path = panel_paths[letter].with_suffix(".pdf")
        source_page = PdfReader(str(source_path)).pages[0]
        source_width = float(source_page.mediabox.width)
        source_height = float(source_page.mediabox.height)
        target_width = width_pt * bounds[2]
        target_height = height_pt * bounds[3]
        transform = (
            Transformation()
            .scale(target_width / source_width, target_height / source_height)
            .translate(width_pt * bounds[0], height_pt * bounds[1])
        )
        page.merge_transformed_page(source_page, transform)
    page.merge_page(_frame_overlay_page(width_pt, height_pt))
    writer = PdfWriter()
    writer.add_page(page)
    with output_path.open("wb") as handle:
        writer.write(handle)


def compose_vector_svg(panel_paths: dict[str, Path], output_path: Path) -> None:
    width_pt = WIDTH_MM * MM * 72.0
    height_pt = HEIGHT_MM * MM * 72.0
    svg_ns = "http://www.w3.org/2000/svg"
    ET.register_namespace("", svg_ns)
    root = ET.Element(
        f"{{{svg_ns}}}svg",
        {
            "width": f"{WIDTH_MM}mm",
            "height": f"{HEIGHT_MM}mm",
            "viewBox": f"0 0 {width_pt:.6f} {height_pt:.6f}",
        },
    )
    for index, (letter, bounds) in enumerate(PANEL_SPECS.items()):
        source = ET.parse(panel_paths[letter].with_suffix(".svg")).getroot()
        source_width, source_height = _source_viewbox(panel_paths[letter].with_suffix(".svg"))
        target_width = width_pt * bounds[2]
        target_height = height_pt * bounds[3]
        group = ET.SubElement(
            root,
            f"{{{svg_ns}}}g",
            {
                "transform": (
                    f"translate({width_pt * bounds[0]:.6f},{height_pt * (1 - bounds[1] - bounds[3]):.6f}) "
                    f"scale({target_width / source_width:.9f},{target_height / source_height:.9f})"
                )
            },
        )
        for child in _prefixed_svg(source, f"panel{index}_"):
            if child.tag.endswith("svg"):
                group.extend(list(child))
            else:
                group.append(child)
    for x, y, width, height in FRAME_SPECS:
        ET.SubElement(
            root,
            f"{{{svg_ns}}}rect",
            {
                "x": f"{width_pt * (x + FRAME_INSET):.6f}",
                "y": f"{height_pt * (1 - y - height + FRAME_INSET):.6f}",
                "width": f"{width_pt * (width - 2 * FRAME_INSET):.6f}",
                "height": f"{height_pt * (height - 2 * FRAME_INSET):.6f}",
                "rx": f"{min(width_pt, height_pt) * FRAME_RADIUS:.6f}",
                "ry": f"{min(width_pt, height_pt) * FRAME_RADIUS:.6f}",
                "fill": "none",
                "stroke": FRAME_COLOR,
                "stroke-width": str(FRAME_LINEWIDTH),
            },
        )
    ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True)


def save_figure(fig: plt.Figure, panel_paths: dict[str, Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / "Fig1_SIGN"
    fig.savefig(prefix.with_suffix(".png"), dpi=450, facecolor="white")
    fig.savefig(
        prefix.with_suffix(".tiff"),
        dpi=600,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    if PageObject is None:
        # Do not downgrade an existing vector release merely because a local
        # plotting environment lacks the optional pypdf dependency.
        if not prefix.with_suffix(".pdf").exists():
            fig.savefig(prefix.with_suffix(".pdf"), facecolor="white")
        if not prefix.with_suffix(".svg").exists():
            fig.savefig(prefix.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    if PageObject is not None:
        compose_vector_pdf(panel_paths, prefix.with_suffix(".pdf"))
        compose_vector_svg(panel_paths, prefix.with_suffix(".svg"))


def main() -> None:
    args = parse_args()
    panel_paths = {letter: getattr(args, f"panel_{letter}") for letter in PANEL_SPECS}
    if plt is None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        compose_vector_pdf(panel_paths, args.output_dir / "Fig1_SIGN.pdf")
        compose_vector_svg(panel_paths, args.output_dir / "Fig1_SIGN.svg")
        print(f"Saved vector Fig. 1 to {args.output_dir.resolve()}")
        return
    save_figure(make_figure(panel_paths), panel_paths, args.output_dir)
    print(f"Saved Fig. 1 to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
