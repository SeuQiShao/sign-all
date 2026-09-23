# Fig4 robustness main figure

This is the Python submission package for the robustness figure. The package
identifier is `fig4`; manuscript-facing figure labels should be defined by the
release manifest and manuscript rather than historical source directory names.

Run from this directory with:

```powershell
python code\plot_fig4.py
```

The script reads only the CSV files in `plot_data` and writes the `Fig4_main`
submission composite plus six standalone panels as editable SVG/PDF and
600-dpi PNG/TIFF exports to `code/output`. Numeric audit files are written
alongside the plots.

`source_data` is intentionally empty of raw experiment files. The compact CSVs
in `plot_data` are the direct plotting inputs and are sufficient to reproduce
the figure.



