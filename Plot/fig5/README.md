# FHN prediction figure package

This is the self-contained Python plotting package for the final FHN figure. It contains exactly the three requested data/code directories:

- `code/`: panel scripts, composite assembly, the brain-outline asset, and the runnable entry point.
- `plot_data/`: compact CSV inputs used directly by the plotting scripts.
- `source_data/`: intentionally contains no raw matrices; see its README.

## Reproduce the figure

From this directory, run:

```powershell
python code/run_fig5.py
```

The script renders panels a–e and then assembles the final composite. Outputs
are written to `code/output/` as 600-dpi PNG/TIFF files plus PDF/SVG files.
The composite PDF/SVG is assembled from the standalone panel exports and
preserves their vector content when `pypdf` and `reportlab` from
`code/requirements.txt` are installed.  Without those optional assembly
dependencies, the runner keeps a raster PDF fallback for local inspection.

## Data-to-panel map

| Panel | Direct input | Contents |
|---|---|---|
| a | `trajectory_snr_30.csv` | Two-node 30-dB trajectory vignette used in the prediction-setting schematic |
| b | `trajectory_snr_30.csv`, `trajectory_snr_50.csv` | 20 independent trajectory segments; observed values are plotted as points and predictions as train/test solid lines |
| c | `node_mse_snr_30.csv`, `node_mse_snr_50.csv` | Per-node MSE values and their base-10 log values; histograms are fitted in log space with a normal distribution |
| d | `horizon_error_snr30.csv`, `horizon_error_snr50.csv` | Horizon-dependent RMSE for SIGN, Persistence, and VAR(1), all horizons 1–98 |
| e | `model_comparison.csv` | Six-model RMSE and MAPE comparison at 30 and 50 dB |

## Provenance and scope

The CSVs are compact plot-ready derivatives of the project data, retained so the figure can be regenerated without shipping the very large node-by-time raw matrices. Column definitions, row counts, and transformations are documented in `plot_data/README.md`. The raw-data decision is documented in `source_data/README.md`.

Panel e contains the manuscript's external traffic-baseline comparison. The
baseline implementations/results were sourced from the unified GitHub
benchmark repository STG4Traffic
(https://github.com/trainingl/STG4Traffic), not reimplemented in this plotting
package. The individual baseline projects retain their own licenses and
citations.

The main rendered output is `code/output/fig5_full_composite.png`. Individual
panels and the composite are also exported as PDF, SVG, and TIFF.
