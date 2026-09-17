# Manuscript/SI crosswalk

The registry is based on the manuscript and Supplementary Information. No
machine-specific path is required to use this project. This repository keeps
the SIGN-side implementation and records external comparison protocols only.
Traffic baselines such as STGNN/MTGNN/ASTGCN/MSTGCN/STSGCN/STGCN were sourced
from the unified GitHub benchmark repository STG4Traffic
(https://github.com/trainingl/STG4Traffic); their implementations are not
redistributed here.

| Configuration | Manuscript/SI location | Covered protocol |
|---|---|---|
| `support_library_recovery` | Main 2.3; SI V.A | L1/L2/L3, clean/noise-50/sparse-200 support recovery |
| `consensus_sensitivity` | Main Fig. 3c; SI V.B | DBSCAN radius and minimum-sample sensitivity |
| `consensus_strategies` | Main Fig. 3d-e; SI V.B | DBSCAN, All, Voting, and Ward-HC aggregation |
| `phase2_refinement` | Main Fig. 3f-g; SI V.C-D | Fixed-support refinement, pruning, and Euler rollout |
| `observation_noise_sampling` | Main 2.4; SI VI.A | SNR 30–70 dB, 1000→200 points, dt 0.01→0.05 |
| `heterogeneity_coupling` | SI VI.B | Kuramoto frequency spread and FHN coupling strength |
| `basis_mismatch` | Main Fig. 4c; SI VI.C | Mutualistic fractional and Chua piecewise nonlinearities |
| `structure_incompleteness` | Main Fig. 4d-f; SI VI.D | DSCM/RGM/RPG/SBM and missing nodes/edges |
| `imperfect_adjacency_fn_fp` | SI VI.E; Tables S9-S10 | 3 systems × 5 seeds × 16 FN/FP cells |
| `baseline_scalability` | Main Fig. 4f; SI VI.F | SIGN scalability protocol; comparison methods external |
| `fhn_prediction_robustness` | Main 2.5; SI VII.A-B | 22,198-node bn-human partition, 2,000 steps, 30/50/70 dB |
| `prediction_temporal_baselines` | SI VII.B.4 | SIGN recursive SIS/MM prediction; comparison methods external |

The SST empirical SIGN prediction entry point is implemented by
`SIGN_sst_pred/trainer.py`. It uses a 96/24 split and training-only Fourier
period selection. SST comparison baselines are not included in this
repository.

## Parameter decisions

- The SI range SNR 30–70 dB is used when the main-text Fig. 4 caption and body
  differ.
- DBSCAN `min_samples=2` is the default because it is the explicit E2V3
  implementation setting.
- SI gives endpoint ranges for some heterogeneity parameters rather than a
  complete discrete grid; those endpoints are recorded and any intermediate
  grid must be supplied explicitly at run time.
- External comparison entries preserve the paper protocol but do not
  redistribute baseline implementations or dependencies.
