# SIGN-benchmarks: compact Fig. 2 runner

`SIGN-benchmarks` is the compact reproduction runner for the Fig. 2 benchmark.
It is not the primary algorithm entry point: `../SIGN-phase/` contains the
canonical SIGN implementation. This package fixes the Fig. 2 experiment
surface while delegating the complete E2V3 core algorithm to `SIGN-phase`:
Phase-I discovers a coefficient-valued sparse support mask, and Phase-II
trains coefficients on that fixed support. Wide-library scans, ablation
matrices, and auxiliary experiment controls are omitted here.

The compact runner fixes the paper's clean Fig. 2 setting to the `L1` library
with the `trig_exp_v2` basis and delegates the canonical Phase-I/Phase-II
modules to `SIGN-phase`. This keeps both public entry points algorithmically
consistent while avoiding a second copy of the model implementation.

The implementation keeps the SIGN conventions used by the paper:

- trajectories are stored as `x=[time,node,state_dimension]`;
- graph edges use `edge_index[0] -> edge_index[1]`;
- self and graph-coupling terms are evaluated with dimension-aware libraries;
- multidimensional discovery equations are fitted one coordinate at a time;
- derivatives are estimated with the same five-point finite-difference stencil
  used by the canonical phase implementation;
- Phase-II cannot add terms that were absent from the Phase-I support mask.

## Quick start

Run from the `code` directory after installing `requirements.txt` (or this
package's smaller `requirements.txt`):

```powershell
python SIGN-benchmarks/run_fig2.py `
  --data-root SIGN-data/synthetic `
  --output SIGN-benchmarks/output/fig2
```

To fit one trajectory with the same two phases:

```powershell
python SIGN-benchmarks/trainer.py `
  --data SIGN-data/synthetic/fhn_2d.npz `
  --output SIGN-benchmarks/output/fhn_compact `
  --device cpu
```

Each output directory contains converted canonical data, Phase-I and Phase-II
outputs, and a `manifest.json` recording the fixed library and settings. Use
`SIGN-phase` directly for the primary algorithm workflow, and use
`Sign-Robust` for robustness matrices and wide-library or ablation studies.
