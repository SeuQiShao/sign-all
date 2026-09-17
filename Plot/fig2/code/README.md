# Plotting code

`replicate_fig2_nature.py` is the only executable needed to regenerate Fig. 2.
It resolves paths relative to its parent package, so it can be copied or
uploaded without depending on the original workspace layout.

The user-tunable block at the top controls fonts, colors, line widths, panel
positions, and the displayed MATLAB-style 1-based node indices. The script
reads only `../plot_data/*.csv`, builds panels a-e, and writes all exports to
the package root.

The c-panel fraction uses TeX Live when `latex` is on `PATH`; otherwise the
same LaTeX expression is rendered by Matplotlib's built-in mathtext fallback.

For a new data release, replace the CSV matrices while preserving their shape
and file names. Heatmap errors are already plot-ready; do not recompute them
inside the plotting script.
