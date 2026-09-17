# SIGN robustness project

`Sign-Robust` is the configuration-driven project for the SIGN robustness
experiments in the manuscript and Supplementary Information. It calls the
canonical E2V3 trainer in `../SIGN-phase` and the shared generators in
`../Data_generation`.

This repository contains SIGN implementations and experiment protocols only.
Third-party baseline implementations are not redistributed. External traffic
baselines such as STGNN/MTGNN/ASTGCN/MSTGCN/STSGCN/STGCN were sourced from the
unified GitHub benchmark repository STG4Traffic
(https://github.com/trainingl/STG4Traffic), with the original projects'
licenses and citations retained outside this repository.

## Experiment families

The registry covers:

1. nested L1/L2/L3 support-library recovery;
2. DBSCAN sensitivity and consensus strategies;
3. Phase-II fixed-support refinement and Euler rollout;
4. observational noise, limited observations, and coarse sampling intervals;
5. heterogeneous Kuramoto frequencies and strong FHN coupling;
6. Mutualistic and Chua basis mismatch;
7. structured topologies, missing nodes, and missing edges;
8. false-negative/false-positive adjacency estimates;
9. SIGN scalability protocols;
10. 22,198-node FHN prediction robustness; and
11. SIS/MM recursive SIGN temporal prediction.

Comparison-method names remain in configuration metadata where required by the
paper protocol, but their implementations and dependencies are external.

## Use

```powershell
python Sign-Robust/scripts/validate_configs.py
python Sign-Robust/scripts/expand_matrix.py `
  --experiment observation_noise_sampling `
  --out Sign-Robust/output/observation_noise_sampling.jsonl

python Data_generation/generate_robust.py `
  --system fhn --network small_world --num-nodes 32 `
  --num-steps 64 --dt 0.01 --snr-db 50 `
  --out Sign-Robust/output/fhn_smoke.npz

python Sign-Robust/scripts/run_canonical_npz_case.py `
  --source Sign-Robust/output/fhn_smoke.npz `
  --data-root Sign-Robust/output/canonical_data `
  --system FHN --phase e1 --device cpu
```

The matrix expander writes deterministic JSONL records and does not start a
training job. `run_canonical_npz_case.py` converts a generated NPZ to the
canonical PyG format and launches unchanged E2V3. FN/FP cases apply adjacency
corruption only to the graph provided to inference while retaining the true
graph for trajectory generation.

## Output contract

Discovery outputs record the system, condition, graph, size, dimension,
library, phase, support precision/recall/F1, support size, coefficient error,
and rollout error. Prediction outputs record the SIGN forecast split,
horizon, RMSE, MAPE, and MSE.

`configs/*.json` is the authoritative experiment specification and
`manifest.json` is the registry. `references/SCOPE.md` gives the
manuscript/SI crosswalk without machine-specific paths.
