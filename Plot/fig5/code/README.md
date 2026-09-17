# Plotting code

Run `run_fig5.py` from the package root or from any working directory. The scripts resolve all inputs relative to this package, so no project-level paths or external data directories are needed.

The plotting environment requires Python 3.10+ and the packages listed in `requirements.txt` (`matplotlib`, `numpy`, `scipy`, `Pillow`, and `networkx`).

- `plot_panel_a.py`: prediction-setting schematic with the brain-outline asset, NetworkX schematic nodes, a 30-dB trajectory vignette, and the training/test split.
- `plot_panel_b_v2.py`: V2 phase portraits for 30 and 50 dB.
- `plot_panel_c.py`: log10 node-error histograms with normal fits and fitted mean/SD annotations.
- `plot_panel_d.py`: forecast skill across all 98 horizons.
- `plot_panel_e.py`: six-model RMSE/MAPE comparison.
- `assemble_fig5.py`: fixed-layout three-row composite assembly.

Panel e reads the external comparison values from `../plot_data/model_comparison.csv`.
The non-SIGN traffic baselines originate from the unified GitHub benchmark
repository STG4Traffic (https://github.com/trainingl/STG4Traffic); this package
does not reimplement or rerun those baselines.

Each panel exports PNG, PDF, SVG, and TIFF. The composite is written in the same four formats under `code/output/`.

The brain outline is a decorative schematic asset, not experimental data. Its source and public-domain status are recorded in the comment in `plot_panel_a.py`.
