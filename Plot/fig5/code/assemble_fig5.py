from __future__ import annotations

import copy
import io
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:  # PIL plus the vector libraries are enough for assembly.
    class _MatplotlibPlaceholder:
        rcParams = {}
    mpl = _MatplotlibPlaceholder()
    plt = None
import numpy as np
from PIL import Image, ImageDraw
try:
    from pypdf import PageObject, PdfReader, PdfWriter, Transformation
    from reportlab.pdfgen import canvas as reportlab_canvas
except ImportError:  # Optional; the raster outputs remain runnable without it.
    PageObject = PdfReader = PdfWriter = Transformation = reportlab_canvas = None


CODE_ROOT = Path(__file__).resolve().parent
OUT_DIR = CODE_ROOT / "output"
OUT = OUT_DIR / "fig5_full_composite"
DPI = 600
CANVAS_WIDTH = 4320  # 7.2 in at 600 dpi; matches the full-width panel a.
COL_GAP = 54
ROW_GAP = 10
FRAME_LEFT = 31
FRAME_RIGHT = 56
FRAME_Y_INSET = 20

# 0.012-figure-width corner.  Pixels are used here because this script also
# produces the raster composite at 600 dpi.
FRAME_WIDTH = 6.25
FRAME_RADIUS = 52
A_SHIFT = 33  # about 4 pt at 600 dpi; shift panel a down as a complete unit.
LOWER_SHIFT = -3  # keep the two lower rows aligned after moving panel a down.
TOP_FRAME_INSET = 56

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.sans-serif": ["Arial"],
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


def load(name: str) -> Image.Image:
    path = OUT_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def resize_to(image: Image.Image, width: int, height: int) -> Image.Image:
    return image.resize((width, height), Image.Resampling.LANCZOS)


def build_composite() -> Image.Image:
    panel_a = load("fig5a_redraw.png")
    panel_b = load("fig5b_phase_portrait_v2.png")
    panel_c = load("fig5c_error_distribution.png")
    panel_d = load("fig5d_forecast_skill.png")
    panel_e = load("fig5e_model_comparison.png")

    row_a_height = round(CANVAS_WIDTH * panel_a.height / panel_a.width)
    pair_width = (CANVAS_WIDTH - COL_GAP) // 2
    pair_height = max(panel_b.height, panel_c.height, panel_d.height, panel_e.height)
    canvas_height = row_a_height + 2 * ROW_GAP + 2 * pair_height
    canvas = Image.new("RGB", (CANVAS_WIDTH, canvas_height), "white")

    y = 0
    # Stretch only the horizontal canvas of panel a by 3 pt and shift it left,
    # keeping its right border aligned with the two lower frames.
    left_extension = FRAME_RIGHT - FRAME_LEFT
    canvas.paste(
        resize_to(panel_a, CANVAS_WIDTH + left_extension, row_a_height),
        (-left_extension, y + A_SHIFT),
    )
    y += row_a_height + ROW_GAP + LOWER_SHIFT
    canvas.paste(resize_to(panel_b, pair_width, pair_height), (0, y))
    canvas.paste(resize_to(panel_c, pair_width, pair_height), (pair_width + COL_GAP, y))
    y += pair_height + ROW_GAP
    canvas.paste(resize_to(panel_d, pair_width, pair_height), (0, y))
    canvas.paste(resize_to(panel_e, pair_width, pair_height), (pair_width + COL_GAP, y))

    # Draw every visible outer frame in one pass, using the same Fig. 4 style.
    draw = ImageDraw.Draw(canvas)
    frame_bounds = [
        (FRAME_LEFT, A_SHIFT + TOP_FRAME_INSET,
         CANVAS_WIDTH - FRAME_RIGHT, A_SHIFT + row_a_height - TOP_FRAME_INSET),
    ]
    for row_top in (row_a_height + ROW_GAP + LOWER_SHIFT,
                    row_a_height + 2 * ROW_GAP + pair_height + LOWER_SHIFT):
        frame_bounds.append(
            (FRAME_LEFT, row_top + FRAME_Y_INSET,
             CANVAS_WIDTH - FRAME_RIGHT, row_top + pair_height - FRAME_Y_INSET)
        )
    for left, top, right, bottom in frame_bounds:
        draw.rounded_rectangle(
            [left, top, right, bottom],
            radius=FRAME_RADIUS, outline="#2A2A2A", width=round(FRAME_WIDTH),
        )
    return canvas


def _panel_names() -> list[str]:
    return [
        "fig5a_redraw",
        "fig5b_phase_portrait_v2",
        "fig5c_error_distribution",
        "fig5d_forecast_skill",
        "fig5e_model_comparison",
    ]


def _panel_sizes() -> list[tuple[float, float]]:
    sizes = []
    for name in _panel_names():
        if PdfReader is not None:
            page = PdfReader(str(OUT_DIR / f"{name}.pdf")).pages[0]
            sizes.append((float(page.mediabox.width), float(page.mediabox.height)))
        else:
            image = Image.open(OUT_DIR / f"{name}.png")
            sizes.append((float(image.width), float(image.height)))
    return sizes


def _placements() -> tuple[int, list[tuple[str, int, int, int, int]]]:
    sizes = _panel_sizes()
    row_a_height = round(CANVAS_WIDTH * sizes[0][1] / sizes[0][0])
    pair_width = (CANVAS_WIDTH - COL_GAP) // 2
    pair_height = max(round(pair_width * height / width) for width, height in sizes[1:])
    canvas_height = row_a_height + 2 * ROW_GAP + 2 * pair_height
    left_extension = FRAME_RIGHT - FRAME_LEFT
    placements = [
        (_panel_names()[0], -left_extension, A_SHIFT, CANVAS_WIDTH + left_extension, row_a_height),
    ]
    y = row_a_height + ROW_GAP + LOWER_SHIFT
    placements.extend([
        (_panel_names()[1], 0, y, pair_width, pair_height),
        (_panel_names()[2], pair_width + COL_GAP, y, pair_width, pair_height),
    ])
    y += pair_height + ROW_GAP
    placements.extend([
        (_panel_names()[3], 0, y, pair_width, pair_height),
        (_panel_names()[4], pair_width + COL_GAP, y, pair_width, pair_height),
    ])
    return canvas_height, placements


def _svg_viewbox(path: Path) -> tuple[float, float]:
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
                value = value.replace(f"url(#{old})", f"url(#{new})").replace(f"#{old}", f"#{new}")
            element.attrib[key] = value
        if element.text and element.tag.endswith("style"):
            for old, new in ids.items():
                element.text = element.text.replace(f"#{old}", f"#{new}")
    return list(cloned)


def _frame_specs(placements: list[tuple[str, int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    pair_height = placements[1][4]
    row_tops = (placements[1][2], placements[3][2])
    specs = [
        (FRAME_LEFT, A_SHIFT + TOP_FRAME_INSET,
         CANVAS_WIDTH - FRAME_RIGHT - FRAME_LEFT,
         placements[0][4] - 2 * TOP_FRAME_INSET),
    ]
    specs.extend(
        (FRAME_LEFT, top + FRAME_Y_INSET,
         CANVAS_WIDTH - FRAME_RIGHT - FRAME_LEFT, pair_height - 2 * FRAME_Y_INSET)
        for top in row_tops
    )
    return specs


def _write_vector_pdf(canvas_height: int) -> None:
    if PageObject is None or reportlab_canvas is None:
        raise RuntimeError("pypdf and reportlab are required for vector PDF composition")
    page_width = CANVAS_WIDTH / DPI * 72.0
    page_height = canvas_height / DPI * 72.0
    page = PageObject.create_blank_page(width=page_width, height=page_height)
    for name, x, top, width, height in _placements()[1]:
        source = PdfReader(str(OUT_DIR / f"{name}.pdf")).pages[0]
        source_width = float(source.mediabox.width)
        source_height = float(source.mediabox.height)
        transform = (
            Transformation()
            .scale((width / DPI * 72.0) / source_width, (height / DPI * 72.0) / source_height)
            .translate(x / DPI * 72.0, (canvas_height - top - height) / DPI * 72.0)
        )
        page.merge_transformed_page(source, transform)

    overlay_bytes = io.BytesIO()
    overlay = reportlab_canvas.Canvas(overlay_bytes, pagesize=(page_width, page_height))
    overlay.setStrokeColorRGB(0.165, 0.165, 0.165)
    overlay.setLineWidth(FRAME_WIDTH / DPI * 72.0)
    placements = _placements()[1]
    for x, top, width, height in _frame_specs(placements):
        overlay.roundRect(x / DPI * 72.0, (canvas_height - top - height) / DPI * 72.0,
                          width / DPI * 72.0, height / DPI * 72.0,
                          FRAME_RADIUS / DPI * 72.0, stroke=1, fill=0)
    overlay.save()
    overlay_page = PdfReader(io.BytesIO(overlay_bytes.getvalue())).pages[0]
    page.merge_page(overlay_page)

    writer = PdfWriter()
    writer.add_page(page)
    with OUT.with_suffix(".pdf").open("wb") as handle:
        writer.write(handle)


def _write_vector_svg(canvas_height: int) -> None:
    svg_ns = "http://www.w3.org/2000/svg"
    ET.register_namespace("", svg_ns)
    width_pt = CANVAS_WIDTH / DPI * 72.0
    height_pt = canvas_height / DPI * 72.0
    root = ET.Element(
        f"{{{svg_ns}}}svg",
        {"width": f"{width_pt / 72.0:.6f}in", "height": f"{height_pt / 72.0:.6f}in",
         "viewBox": f"0 0 {width_pt:.6f} {height_pt:.6f}"},
    )
    placements = _placements()[1]
    for index, (name, x, top, width, height) in enumerate(placements):
        source_path = OUT_DIR / f"{name}.svg"
        source = ET.parse(source_path).getroot()
        source_width, source_height = _svg_viewbox(source_path)
        group = ET.SubElement(
            root, f"{{{svg_ns}}}g",
            {"transform": (
                f"translate({x / DPI * 72.0:.6f},{(canvas_height - top - height) / DPI * 72.0:.6f}) "
                f"scale({(width / DPI * 72.0) / source_width:.9f},{(height / DPI * 72.0) / source_height:.9f})")},
        )
        for child in _prefixed_svg(source, f"panel{index}_"):
            group.append(child)
    for x, top, width, height in _frame_specs(placements):
        ET.SubElement(root, f"{{{svg_ns}}}rect", {
            "x": f"{x / DPI * 72.0:.6f}",
            "y": f"{(canvas_height - top - height) / DPI * 72.0:.6f}",
            "width": f"{width / DPI * 72.0:.6f}",
            "height": f"{height / DPI * 72.0:.6f}",
            "rx": f"{FRAME_RADIUS / DPI * 72.0:.6f}",
            "fill": "none", "stroke": "#2A2A2A",
            "stroke-width": f"{FRAME_WIDTH / DPI * 72.0:.6f}",
        })
    ET.ElementTree(root).write(OUT.with_suffix(".svg"), encoding="utf-8", xml_declaration=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    canvas = build_composite()
    canvas.save(OUT.with_suffix(".png"), dpi=(DPI, DPI))
    canvas.save(OUT.with_suffix(".tiff"), dpi=(DPI, DPI), compression="tiff_lzw")
    canvas_height, _ = _placements()
    if PageObject is not None and reportlab_canvas is not None:
        _write_vector_pdf(canvas_height)
    else:
        # Keep the renderer usable in a minimal plotting environment.  The
        # release environment should install requirements.txt to get vector PDF.
        fig = plt.figure(figsize=(CANVAS_WIDTH / DPI, canvas_height / DPI), dpi=DPI, facecolor="white")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(np.asarray(canvas), interpolation="none")
        ax.set_axis_off()
        if not OUT.with_suffix(".pdf").exists():
            fig.savefig(OUT.with_suffix(".pdf"), dpi=DPI, facecolor="white")
        plt.close(fig)
    _write_vector_svg(canvas_height)
    print(f"Saved {OUT}.* at {canvas.width} x {canvas.height} px ({DPI} dpi)")


if __name__ == "__main__":
    main()
