# Figure reproduction packages

This directory contains the self-contained Python packages used to redraw the
main and supplementary figures for the SIGN manuscript. Each package keeps its
renderer, compact plotting inputs, and generated exports together, so a figure
can be reproduced without rerunning the underlying discovery or prediction
experiments.

## Quick start

Use Python 3 and install the requirements in the figure package you want to
render. Run the entry point from that package directory. For example:

```powershell
cd Plot\fig3
python -m pip install -r code\requirements.txt
python code\plot_fig3.py
```

The renderers read only their packaged `plot_data/` files. They do not modify
the experiment code or the source data under `../code/`.

## Package index

| Package | Manuscript content | Entry point | Generated exports |
|---|---|---|---|
| `fig1` | SIGN workflow and discovery overview | `python code\plot_fig1.py` | `code\output\` |
| `fig2` | Equation-discovery benchmark | `python code\replicate_fig2_nature.py` | package root |
| `fig3` | Phase-I/Phase-II and consensus analysis | `python code\plot_fig3.py` | `code\output\` |
| `fig4` | Robustness analyses | `python code\plot_fig4.py` | `code\output\` |
| `fig5` | FHN network prediction | `python code\run_fig5.py` | `code\output\` |
| `fig6` | ENSO/SST prediction | `python code\reproduce_figure.py` | `plot\` |
| `figSI1` | Scale robustness | `python code\plot_si1.py` | `code\output\` |
| `figSI2` | Network benchmark | `python code\plot_si2.py` | `code\output\` |
| `figSI3` | Basis-mismatch trajectories | `python code\plot_si3.py` | `code\output\` |
| `figSI4` | Supplementary ENSO/SST analysis | `python code\reproduce_figure.py` | `plot\` |

`main-figures/` contains the final rendered exports for Figs. 1–6 in PDF,
SVG, and PNG form. It is an archive of deliverables, not a plotting package.

## Data layout and provenance

Most packages use this common layout:

```text
fig*/
├── code/          renderer, package-specific requirements, and output folder
├── plot_data/     compact CSV inputs read directly by the renderer
└── source_data/   source-data notes; large raw experiment files are omitted
```

The compact CSVs preserve every value needed by the published visualizations.
Where original experiment matrices are too large to redistribute, the
corresponding `source_data/README.md` documents that choice. Individual
package READMEs record panel-to-data mappings, transformations, output names,
and any figure-specific conventions.

## Reproducibility notes

- Outputs are normally written as editable SVG/PDF and 600-dpi PNG/TIFF.
- Install dependencies per package rather than assuming a single plotting
  environment; several packages have a minimal, separate `requirements.txt`.
- The figure packages are presentation-layer artifacts. To regenerate data or
  rerun the SIGN experiments, use [`../code/README.md`](../code/README.md).
- The packaged plot data and renderers are deterministic; local font and PDF
  backend differences can cause minor non-scientific rendering variation.
