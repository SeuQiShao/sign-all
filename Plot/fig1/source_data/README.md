# Source data

No large raw source files are copied into this submission folder. The packaged
figure is generated directly from the compact, public-facing CSV tables in
`../plot_data/`. This keeps the package small while preserving every value used
by the displayed panels.

To rebuild those CSVs from the original Rössler experiment outputs, run
`python ../code/export_plot_data.py --data-dir <rollout-directory>
--coefficients <coefficient-json> --output-dir ../plot_data`.
