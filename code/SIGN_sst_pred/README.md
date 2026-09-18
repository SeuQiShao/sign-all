# SIGN SST prediction

`SIGN_sst_pred` is the experiment-specific implementation for the ENSO/SST prediction study (Fig. 6, SI VII.C, and Table S18). It fits SIGN on the first 96 stored monthly states, augments the state library with Fourier terms, and evaluates the remaining 24-state forecast interval. This package replaces the retired `SIGN-prediction` SST runner; it uses its own PyTorch Geometric data layout rather than the retired NPZ command-line interface.

## Position in the repository

This is a standalone prediction project, not the primary SIGN algorithm
entry point. Its layout intentionally follows the SST data record, Fourier
time basis, and 96/24 prediction protocol. For canonical Phase-I / Phase-II
discovery workflows, use [`../SIGN-phase/`](../SIGN-phase/README.md).

## Included data and layout

The supplied source record is `SIGN_data/enso_71987/raw/data1.pt`. It is a PyTorch file containing the PyG data list consumed by `utils_file/data_loader.py`. On the first run, PyTorch Geometric creates `SIGN_data/enso_71987/processed/geometric_data_processed.pt`. Do not commit that generated cache.

The loaded record supplies `x`, `edge_index`, `t`, and the spatial coordinate field `xy`. The loader standardizes `x` using the first 96 states only. The stored trajectory has 120 states over 71,987 nodes; state 0 initializes the sequence, while the figure packages display states 1--119. The separately catalogued NPZ copy and upstream attribution are documented in [`../SIGN-data/prediction/SST_PROVENANCE.md`](../SIGN-data/prediction/SST_PROVENANCE.md).

The original code contains a machine-specific `args.root` in [`utils_file/arg_parser.py`](utils_file/arg_parser.py). Before a local run, change it to the absolute path of this package's `SIGN_data/enso_71987` directory (or make an equivalent local path mapping). The data root must have the `raw/` directory described above.

## Run

Install the shared environment from `code/`, then start the trainer from this directory:

```powershell
cd code
python -m pip install -r requirements.txt
cd SIGN_sst_pred
python trainer.py --ode_model enso --num-atoms 71987 --dims 1 --time-stamp 120
```

The defaults use 500 epochs, 40 Fourier basis terms, and the available CUDA device when present. Adjust `--epochs`, `--k-num`, `--GPU_to_use`, or the other options in `utils_file/arg_parser.py` as required by the local runtime. The trainer writes timestamped checkpoints, loss records, and plots under `logs/`. With saving enabled, it also writes `true_enso_dim_0.csv`, `pred_enso_dim_0.csv`, `enso_xy.csv`, `fftx_enso.csv`, and `result.log` in the current working directory.

## Method and scope

`utils_file/primary_mask.py` constructs the sparse support masks; `model/` contains the basis classifier and GSI decoder. Dominant seasonal components are selected from the 96-state training window and incorporated into the model's time basis. SST comparison baselines are not redistributed. For the source-data licence and citation requirements, see [`../DATA_LICENSES.md`](../DATA_LICENSES.md) and the provenance record above.
