# FigSI2 network benchmark

This folder contains the Python redraw package for the 2 × 4 network-benchmark
SI figure. Run from this directory with:

```powershell
python code\plot_si2.py
```

The script reads only `plot_data` and writes the complete figure plus panels
a–h to `code/output` as editable SVG/PDF and 600-dpi PNG/TIFF files.

Statistical display convention: panels a–d (the four low-rank network bar
panels) draw error bars as `mean ± 0.5 × std`. Panels e–h draw their shaded
uncertainty bands as `mean ± 1 × std`. These are display conventions applied
to the unmodified `std` values in `plot_data`; the underlying data are not
rescaled.
