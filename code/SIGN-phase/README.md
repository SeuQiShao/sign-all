# SIGN phase: canonical E2V3

This folder contains the reference implementation of SIGN E2V3. It preserves
the original PyTorch-Geometric data interface and module layout:

- `trainer.py`: Phase-I support discovery and Phase-II fixed-support training;
- `model/`: encoder, decoder, graph message passing, and dimension-aware basis libraries;
- `data/`: compatibility data generator used by the canonical trainer;
- `utils_file/`: argument parsing, loading, derivative masks, logging, and evaluation;
- `scripts/verify_package.py`: public import, basis, data, and configuration smoke checks.

## Algorithm

Phase-I estimates a coefficient-valued support mask with the five-point
derivative and sparse regression/consensus procedure. Phase-II starts from
that mask and refines coefficients with the E2V3 training objective. Phase-II
can remove weak terms but cannot add terms that were absent from the Phase-I
mask. The default final basis is `trig_exp_v2`.

For a multidimensional system, discovery is equation-wise: set `args.k` to
the target coordinate and run the same library/model for each coordinate. The
library is generated from the requested state dimension, so the 1D, 2D, and
3D self/coupling bases share one implementation.

## Run

From the `NC` directory:

```powershell
python SIGN-phase/trainer.py --help
python SIGN-phase/scripts/verify_package.py
```

The trainer consumes a canonical dataset directory containing the PyG
`data_1.pt` file expected by `utils_file/data_loader.py`. NPZ trajectories from
`Data_generation` can be converted with
`Sign-Robust/scripts/run_canonical_npz_case.py`, which then launches this
trainer without changing its algorithm.

The canonical model code does not contain ablation-result files or baseline
implementations. Use `SIGN-main/` for the compact
complete Phase-I/Phase-II Fig. 2 implementation,
`Sign-Robust/` for robustness matrices,
[`../SIGN_fhn_pred/`](../SIGN_fhn_pred/README.md) for the FHN prediction
experiment, and [`../SIGN_sst_pred/`](../SIGN_sst_pred/README.md) for the
ENSO/SST experiment.
