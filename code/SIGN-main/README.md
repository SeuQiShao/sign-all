# SIGN main: compact two-phase SIGN for Fig. 2

`SIGN-main` is the compact public entry point for the Fig. 2 benchmark. It
keeps the complete E2V3 core algorithm: Phase-I discovers a coefficient-valued
sparse support mask, and Phase-II trains coefficients on that fixed support.
The reduction is in the trainer and experiment surface, not in the core
algorithm: wide-library scans, ablation matrices, and auxiliary trainer
branches are omitted.

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

Run from the `NC` directory after installing `SIGN-main/requirements.txt`:

```powershell
python SIGN-main/run_fig2.py `
  --data-root SIGN-data/synthetic `
  --output SIGN-main/output/fig2
```

To fit one trajectory with the same two phases:

```powershell
python SIGN-main/trainer.py `
  --data SIGN-data/synthetic/fhn_2d.npz `
  --output SIGN-main/output/fhn_compact `
  --device cpu
```

Each output directory contains converted canonical data, Phase-I and Phase-II
outputs, and a `manifest.json` recording the fixed library and settings. Use
`SIGN-phase` for the full E2V3 trainer, robustness matrices, and the remaining
wide-library or ablation study.
