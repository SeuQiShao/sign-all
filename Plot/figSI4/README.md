# Supplementary Fig. SI4 SST submission package

The package identifier is `figSI4`; manuscript-facing labels should be
defined by the release manifest and manuscript rather than historical working
directory names.

- `code/`: executable Python plotting code and run instructions.
- `plot_data/`: compact CSV inputs used directly by the plotting code.
- `source_data/`: intentionally empty because the compact plot data are
  sufficient to regenerate every panel without the large raw trajectories.
- `plot/`: generated PDF/SVG submission files and PNG/TIFF previews.

Run `python code/reproduce_figure.py` to regenerate the supplementary figure.

The supplied SST trajectory contains 120 stored states. Stored index 0 is the
initial observed/true state used to initialize the rollout and is not plotted;
the figure displays the remaining 119 monthly states (indices 1–119). Of
those displayed states, the first 95 months are training and the last 24 are
held-out test months.

The SST data are aggregated from the manuscript's SSTG dataset [45] (Cao et
al., *Earth System Science Data* 13, 2111–2134, 2021; SI [S25]).
