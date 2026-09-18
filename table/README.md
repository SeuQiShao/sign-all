# Supplementary table source data

Each `SNN_*` directory corresponds to Supplementary Table SNN. CSV/TSV files
are machine-readable source tables; units and aggregation definitions are kept
in the column names or the table-specific notes below.

| Directory | Content | Statistical unit |
|---|---|---|
| `S01_Libraries` | Candidate basis-library definitions | Basis term |
| `S02_Network_properties` | Network size and topology descriptors | Network |
| `S03_Inference_error` | Coefficient sMAPE by network category | Recorded experiment summary |
| `S04_Support_recovery` | Phase-I/Phase-II support recovery | Dimension-level record over five seeds |
| `S05_Consensus_comparison` | Consensus-strategy comparison | Dimension-level record over five seeds |
| `S06_Phase2_refinement` | Paired Phase-I/Phase-II refinement | Paired record |
| `S07_Rollout_ablation` | Raw and summarized Euler-rollout error | Five random seeds; sample SD |
| `S08_Error_reduction` | Euler-rollout method comparison plus Phase-I-to-Phase-II relative error reduction | Method/horizon summaries and Rössler 50-dB paired comparison; n = 5 |
| `S09_Adjacency_robustness_summary` | Adjacency-error robustness summary | Recorded experiment summary |
| `S10_Adjacency_robustness_grid` | False-positive/false-negative grid | Grid condition |
| `S11_FHN_forecast_protocol` | FHN forecast protocol | Protocol record |
| `S12_FHN_baselines` | FHN horizon and final-horizon errors | Method and horizon |
| `S13_Graph_model_comparison` | Graph-model RMSE/MAPE comparison | Method, SNR condition and metric |
| `S14_Node_statistics_70dB` | Node-level MSE and summary | Node–dimension record |
| `S15_Segment_statistics_70dB` | Segment-level MSE and summary | Segment–dimension record |
| `S16_SIS_MM_rollout` | SIS/MM horizon errors | Method, seed and horizon |
| `S17_DeltaX_MSE` | State-increment MSE | System, method and horizon |
| `S18_SST_fourier` | Retained SST Fourier components and fitted coefficients | Fourier component and coefficient |
| `S19_SST_baselines` | SST prediction matrices and baseline summary | See the directory README |

