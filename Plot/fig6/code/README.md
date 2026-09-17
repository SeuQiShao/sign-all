# ENSO/SST Fig. 6 code

Run from any directory with Python, NumPy and Matplotlib:

```powershell
python code/reproduce_figure.py
```

The script reads only `../plot_data` and writes `fig6.pdf`, `fig6.svg`,
`fig6.png`, and `fig6.tiff` to `../plot`.

The source SST arrays were inverse-normalized with `x * 1.3180 + 26.9876`.
The source trajectory contains 120 stored states and 71,987 nodes. Stored
index 0 is the initial observed/true state and is not plotted; the package
plots indices 1–119 as 119 monthly states. The first 95 plotted months are
the training period and the last 24 plotted months are the test period.

`plot_data` is a compact public plotting package: every node is retained for
spatial/error summaries, while large point clouds are represented by complete
180 × 180 density matrices. No observations were sampled or dropped.
Panel f reads the ordinary least-squares regression over all 71,987 nodes from
`../plot_data/panel_f_regression.csv`: `y = 2.311112248631588x -
0.188568155717975`, `R² = 0.249703763834046`.

The SST source is identified in the manuscript as the SSTG dataset [45] (Cao
et al., *Earth System Science Data* 13, 2111–2134, 2021; SI [S25]).

The main Fig. 6 export is 180 mm wide on a 5.00-inch-high canvas. Its d/e/f/g
panels use equal square plotting areas, and all visible error labels in those
panels use MAPE.
