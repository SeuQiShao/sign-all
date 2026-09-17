# Fig. 1 reproduction package

This package reproduces the five-panel SIGN Fig. 1 using the Python
matplotlib/NetworkX backend.  The figure is a schematic-led composite: panels
a–b introduce the networked dynamics and candidate libraries, panel c shows
global support identification, panel d shows shared coefficient estimation,
and panel e validates node-level prediction performance.

## One-command reproduction

From the `fig1/` directory, run:

```text
cd Plot/fig1
python code/plot_fig1.py
```

The final files are written to `code/output/` as `Fig1_SIGN.png`,
`Fig1_SIGN.tiff`, `Fig1_SIGN.pdf`, and `Fig1_SIGN.svg`.  The PDF and SVG
composites preserve the standalone panel vector content.  The default representative node is 121
and the context nodes are 230, 112, 378, and 934, matching the packaged figure.

## Package layout

- `code/`: panel renderers, CSV loader, exporter, and the top-level runner.
- `plot_data/`: compact CSV tables used directly by the renderers.
- `source_data/`: intentionally contains no large raw experiment files; see its
  README.  `code/export_plot_data.py` can regenerate the compact tables when
  the original Rössler NPZ files and coefficient JSON are available.
- `code/output/`: generated figure files (created or refreshed by `plot_fig1.py`).

## Plot-data contract

`trajectory.csv` contains the observed and forecast trajectories for the
representative node and four context nodes. `node_mse.csv` contains all 1,000
node-level MSE values used for panel e. `coefficients.csv` contains the five
coefficient series used in panel d. `network_nodes.csv` and
`network_edges.csv` define the shared sparse NetworkX topology used in panels
a, c, and d. `library_terms.csv` and `support_patterns.csv` contain the
candidate-library and support-mask tables used in panels b and c.

The CSVs are derived from the supplied Rössler DBSCAN rollout data without
downsampling, except that trajectory rows are limited to the five nodes
actually displayed in panels a and e. The MSE table retains all 1,000 nodes.
