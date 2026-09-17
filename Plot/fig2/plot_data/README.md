# Plot-ready CSV data

All files are numeric comma-separated matrices with no header row. Rows retain
the supplied time/sample order; columns retain the supplied node order.

| Files | Shape | Used for |
|---|---:|---|
| `error_kuramoto.csv` | 4 x 2 | panel a coefficient-error heatmap |
| `error_sis.csv` | 4 x 3 | panel b coefficient-error heatmap |
| `error_gene.csv` | 4 x 2 | panel c coefficient-error heatmap |
| `error_fhn.csv` | 4 x 8 | panel d coefficient-error heatmap |
| `error_hr.csv` | 4 x 12 | panel e coefficient-error heatmap |
| `true_Kuramoto_dim_0.csv`, `pred_Kuramoto_dim_0.csv` | 399 x 10 | panel a trajectories |
| `true_SIS_dim_0.csv`, `pred_SIS_dim_0.csv` | 999 x 10 | panel b trajectories |
| `true_Gene_dim_0.csv`, `pred_Gene_dim_0.csv` | 999 x 10 | panel c trajectories |
| `true_Fitz_dim_0/1.csv`, `pred_Fitz_dim_0/1.csv` | 399 x 10 | panel d phase portraits |
| `true_HR_dim_0/1/2.csv`, `pred_HR_dim_0/1/2.csv` | 1199 x 10 | panel e 3-D trajectories |

The error matrices are deterministic plot inputs, not re-estimated statistics.
The plotting script does not filter, smooth, normalize, or downsample the
trajectory matrices.
