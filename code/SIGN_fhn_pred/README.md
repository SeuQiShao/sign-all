# SIGN FHN prediction

`SIGN_fhn_pred` is the experiment-specific implementation used for the FitzHugh--Nagumo (FHN) network-prediction study (Fig. 5 and SI VII.A--B). It learns a sparse governing-equation model on each network partition and evaluates autonomous prediction on held-out partitions. The project replaces the retired `SIGN-prediction` entry point; it is not a drop-in runner for the public NPZ files.

## Position in the repository

This is a standalone prediction project, not the primary SIGN algorithm
entry point. Its structure intentionally follows the FHN partitioned-network
protocol and native PyTorch-Geometric input layout. For canonical Phase-I /
Phase-II discovery workflows, use [`../SIGN-phase/`](../SIGN-phase/README.md).

## What this package expects

The trainer uses PyTorch Geometric's `InMemoryDataset` layout. Its data root must contain a raw PyTorch file, for example:

```text
<data-root>/
├── raw/
│   └── data1.pt                     # a saved list of PyG Data objects
└── processed/
    └── geometric_data_processed.pt  # created automatically on first run
```

Each `Data` object must provide `x`, `edge_index`, and `t`; the FHN study also expects the state dimension and partitioning to agree with the command-line configuration. The 22,198-node `bn-human` partition used for the paper is an external input. See [`../Empirical_networks/README.md`](../Empirical_networks/README.md) for its acquisition scope. `../SIGN-data/prediction/fhn_2d.npz` is retained as a time-major public data record for the retired runner and figure/audit workflows; convert or stage it into the PyG layout above before using this trainer.

The root is currently assembled in [`utils_file/arg_parser.py`](utils_file/arg_parser.py) from the FHN run parameters. Before running locally, set `args.root` there to the directory that contains your `raw/` data (or adapt that path construction to your data layout). This explicit setup is necessary because the original experiment's partition data are not redistributed.

## Run

Install the shared environment from `code/`, then run from this directory so that the local `model` and `utils_file` imports resolve:

```powershell
cd code
python -m pip install -r requirements.txt
cd SIGN_fhn_pred
python trainer.py --ode_model FHN --network from_file --dims 2
```

Set `--num-atoms`, `--time-stamp`, `--time-interval`, `--epochs`, and `--GPU_to_use` to match the staged data. The trainer divides the node axis into 20 partitions, fits the first 16, and evaluates the remaining four. It writes timestamped logs and model checkpoints beneath `logs/`; when saving is enabled it also writes sampled true/predicted trajectories and `result.log` in the current working directory.

## Method and scope

The model comprises the basis classifier, encoder, and GSI decoder in `model/`; support masks are constructed by `utils_file/primary_mask.py`. Prediction evaluates each held-out partition without using its future target values. Third-party graph-forecasting baselines are not included. The paper protocol and reported comparison settings are recorded in [`../Sign-Robust/configs/fhn_prediction_robustness.json`](../Sign-Robust/configs/fhn_prediction_robustness.json).
