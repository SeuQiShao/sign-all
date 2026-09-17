# Plot-ready data

All files in this directory are CSV inputs consumed directly by the Fig. 5 Python scripts.

## Trajectories (`trajectory_snr_30.csv`, `trajectory_snr_50.csv`)

Columns: `time_step`, `true_dim1`, `pred_dim1`, `true_dim2`, `pred_dim2`.

Each file contains 1,980 rows: 20 independent segments concatenated in segment order, with 99 time steps per segment. These compact traces support panels a and b.

## Node errors (`node_mse_snr_30.csv`, `node_mse_snr_50.csv`)

Columns: `dim`, `node_index`, `mse_all`, `log_mse`.

Each file contains 44,396 node/component values. `log_mse` is the base-10 logarithm of the positive `mse_all` values. Panel c plots the log-transformed values and fits a normal distribution in that same log10 space; it does not treat the displayed fit as an RMSE distribution.

## Forecast skill (`horizon_error_snr30.csv`, `horizon_error_snr50.csv`)

Each file contains 294 rows: 98 horizons × 3 models. `mean_rmse` is the plotted quantity. The files also contain MSE-space uncertainty columns; because these are not RMSE-space confidence intervals, panel d does not draw them as RMSE bands.

## Model comparison (`model_comparison.csv`)

The first row gives the six model names. Subsequent rows give RMSE or MAPE for clean, 30-dB, and 50-dB conditions. Panel e uses the 30- and 50-dB rows; MAPE is converted from fraction to percent for display.

The non-SIGN traffic baselines in this table are external comparison results
from the unified GitHub benchmark repository STG4Traffic
(https://github.com/trainingl/STG4Traffic). They are not implementations or
new reruns contained in this release.
