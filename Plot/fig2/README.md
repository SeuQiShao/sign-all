# Fig. 2 Python reproduction package

This directory is a self-contained, data-driven reproduction of Fig. 2. The
figure is an asymmetric five-panel composite: coefficient-error heatmaps,
trajectory comparisons, 2-D phase portraits, and 3-D phase trajectories.

## Directory contract

- `code/` — the reproducible Python plotting script and its instructions.
- `source_data/` — intentionally empty of raw observations for this figure;
  see its README.
- `plot_data/` — compact CSV matrices used directly by the plotting script.

The generated publication files are written at this directory level so the
three data/code directories remain easy to archive or upload:

- `Fig2_reproduced.pdf` — vector PDF for submission.
- `Fig2_reproduced.svg` — editable vector companion.
- `Fig2_reproduced.png` — 600-dpi preview.
- `Fig2_reproduced.tiff` — 600-dpi TIFF export.
- `Fig2_reproduced_manifest.json` — dimensions, node selections, and data map.

## Reproduce

From this directory, run:

```powershell
python code\replicate_fig2_nature.py
```

The script uses `numpy` and `matplotlib`; no external XLSX reader is needed
because the heatmap error matrices have been materialized as CSV files. If a
TeX Live `latex` executable is available, the c-panel fraction uses it; the
script automatically falls back to Matplotlib's LaTeX-compatible mathtext
otherwise.
The current layout is 18.9 x 12.1 cm, and all trajectories use their complete
time series without smoothing or row filtering.

## Data provenance

The trajectory CSVs are the supplied True/Inferred matrices. The five
`error_*.csv` files are the deterministic symmetric-percent-error matrices
used by the heatmaps, calculated from the supplied coefficient tables as
`abs(pred-true)/(abs(pred)+abs(true))` before packaging. No scientific
values are invented in the submission package.
