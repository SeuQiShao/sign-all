# Experiment coverage

This matrix maps the SIGN-side experiments described in the manuscript and
Supplementary Information to the public code. The repository does not contain
third-party baseline implementations.

`SIGN-phase` is the primary algorithm entry point. `SIGN-benchmarks` is a
compact Fig. 2 reproduction runner built on that implementation; the FHN and
SST prediction directories are independent, experiment-specific projects with
their own data interfaces.

| Paper location | Experiment | Public implementation | Status |
|---|---|---|---|
| Fig. 1; Methods | SIGN pipeline and two-phase training | `SIGN-phase/trainer.py`, `model/`, `utils_file/` | Runnable |
| Fig. 2; SI IV | Compact benchmark equation discovery | `SIGN-benchmarks/run_fig2.py`, `SIGN-benchmarks/trainer.py`, `SIGN-data/synthetic/` | Runnable on smoke data; invokes the canonical `SIGN-phase` Phase-I/Phase-II path |
| Fig. 2; SI IV | Kuramoto, SIS, MM, FHN, HR, Rössler benchmark generation | `Data_generation/dynamics.py`, `generate_dataset.py` | Runnable; large outputs on demand |
| Fig. 2; SI III | Synthetic graph construction | `Data_generation/graph.py` | Runnable |
| Fig. 2; SI III | Empirical graph preparation | `Data_generation/prepare_empirical_networks.py` | Runnable after external download |
| Fig. 3a-b; SI V.A | L1/L2/L3 support-library recovery | `Sign-Robust/configs/support_library_recovery.json` + `SIGN-phase` | Runnable |
| Fig. 3c; SI V.B | DBSCAN sensitivity | `configs/consensus_sensitivity.json`, `expand_matrix.py` | Matrix runnable |
| Fig. 3d-e; SI V.B | DBSCAN, All, Voting, HC consensus | `configs/consensus_strategies.json` | Protocol recorded; external aggregation code may be required |
| Fig. 3f-g; SI V.C-D | Phase-II fixed-support refinement and rollout | `configs/phase2_refinement.json`, `SIGN-phase` | Runnable |
| Fig. 4a-b; SI VI.A | Noise, fewer observations, coarse dt | `Data_generation/robustness.py`, `configs/observation_noise_sampling.json` | Matrix runnable |
| SI VI.B | Parameter heterogeneity and strong coupling | `configs/heterogeneity_coupling.json` | Matrix runnable |
| Fig. 4c; SI VI.C | Mutualistic/Chua basis mismatch | `configs/basis_mismatch.json`, `generate_robust.py` | Matrix runnable |
| Fig. 4d-e; SI VI.D | Structured topology, missing nodes/edges | `configs/structure_incompleteness.json` | Matrix runnable |
| SI VI.E; Tables S9-S10 | FN/FP adjacency estimates | `run_robust_case.py`, `configs/imperfect_adjacency_fn_fp.json` | Matrix runnable |
| Fig. 4f; SI VI.F | SIGN scalability protocol | `configs/baseline_scalability.json` | SIGN path runnable; comparison methods external |
| Fig. 5; SI VII.A-B | FHN 1,000-node / partitioned-network prediction | `SIGN_fhn_pred/trainer.py` | Runnable after staging the required PyG input; public NPZ record is archival |
| Fig. 5; SI VII.A-B | FHN 22,198-node bn-human prediction | `configs/fhn_prediction_robustness.json`, `prepare_fhn_prediction.py` | Requires external partition input |
| Fig. 5; SI VII.A-B | MTGNN/ASTGCN/MSTGCN/STSGCN/STGCN comparison | Protocol metadata in `fhn_prediction_robustness.json` | External baselines sourced from the unified GitHub STG4Traffic repository; implementations omitted |
| SI VII.B.4; Tables S16-S17 | SIS/MM recursive SIGN prediction | `SIGN-phase/trainer.py`, `Sign-Robust/configs/prediction_temporal_baselines.json` | Protocol recorded; the retired dedicated equation-prediction runner is not distributed |
| Fig. 6; SI VII.C; Table S18 | SST 96/24 SIGN prediction and Fourier terms | `SIGN_sst_pred/trainer.py`, `SIGN_sst_pred/SIGN_data/enso_71987/raw/data1.pt` | Runnable after configuring its local PyG data root; comparison methods omitted |

## Data policy

`SIGN-data/synthetic` and the 1,000-node FHN file are smoke-scale examples.
Large synthetic trajectories are generated on demand. `Empirical_networks`
contains download instructions only; source networks and the 22,198-node FHN
partition input are not redistributed.

## Validation scope

The distributed package provides import, basis, catalog, configuration, and
small-case execution checks.
