# Supplementary ENSO/SST figure submission code

Run from the `Plot/figSI4` directory with Python, NumPy and Matplotlib:

```powershell
python code/reproduce_figure.py
```

The script reads the shared Fig. 6 plotting data from `../../fig6/plot_data` and
writes `figSI4.pdf`, `figSI4.svg`, `figSI4.png`, and `figSI4.tiff` to `../plot`.
SI4 intentionally does not duplicate the 71,987-node CSV package; Fig. 6 is
the single source of truth for these derived SST plotting inputs.

The source SST arrays were inverse-normalized with `x * 1.3180 + 26.9876`.
The source trajectory contains 120 stored states and 71,987 nodes. Stored
index 0 is the initial observed/true state and is not plotted; the package
plots indices 1–119 as 119 monthly states. The first 95 plotted months are
the training period and the last 24 plotted months are the test period.

`plot_data` is a compact public plotting package: every node is retained for
spatial/error summaries, while large point clouds are represented by complete
180 × 180 density matrices. No observations were sampled or dropped.

The SST source is identified in the manuscript as the SSTG dataset [45] (Cao
et al., *Earth System Science Data* 13, 2111–2134, 2021; SI [S25]).


