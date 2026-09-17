# SIGN source code and experiment protocols

This directory contains the public SIGN implementation used for network
dynamical-system discovery and temporal prediction, together with data
preparation utilities and the configuration-driven robustness protocols used
in the manuscript. Third-party baseline implementations are not redistributed.
The standalone figure renderers are in [`../Plot/`](../Plot/README.md).

## Repository layout

```text
code/
├── SIGN-main/           Compact, complete two-phase SIGN benchmark path (Fig. 2)
├── SIGN-phase/          Canonical E2V3 Phase-I/Phase-II implementation
├── SIGN-prediction/     Joint-vector prediction runners for FHN, SST, and equations
├── Sign-Robust/         Configuration registry and robustness-matrix runners
├── Data_generation/     Dynamics, graphs, simulation, and data-conversion tools
├── SIGN-data/           Bundled synthetic smoke data and FHN/SST examples
├── Empirical_networks/  Instructions for externally sourced networks
├── docs/                Manuscript-to-code coverage map
├── requirements.txt     Full shared Python environment
├── DATA_LICENSES.md     Data and external-method licensing notes
└── CITATION.cff         Citation metadata
```

## Installation

Run the following from this `code/` directory. The full environment supports
all local components; individual packages also carry smaller requirements
files when only one workflow is needed.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The project uses PyTorch and PyTorch Geometric. If your operating system or
CUDA version requires a specific PyTorch build, install that compatible build
first, then install the remaining dependencies.

## Main entry points

### Verify the canonical implementation

```powershell
python SIGN-phase\scripts\verify_package.py
```

This performs public import, basis-library, data, and configuration smoke
checks without starting a full experiment.

### Run the compact Fig. 2 discovery benchmark

```powershell
python SIGN-main\run_fig2.py `
  --data-root SIGN-data\synthetic `
  --output SIGN-main\output\fig2
```

`SIGN-main` retains the complete two-phase algorithm in the manuscript's
clean L1 / `trig_exp_v2` setting. Use it as the shortest runnable discovery
example.

### Run canonical Phase-I / Phase-II discovery

```powershell
python SIGN-phase\trainer.py --help
```

The canonical trainer consumes the PyTorch-Geometric dataset format expected
by `SIGN-phase/utils_file/data_loader.py`. `Data_generation/` creates
synthetic trajectories, and `Sign-Robust/scripts/run_canonical_npz_case.py`
converts a generated NPZ case to this canonical input format.

### Run prediction examples

```powershell
python SIGN-prediction\run_fhn_prediction.py `
  --data SIGN-data\prediction\fhn_2d.npz `
  --output SIGN-prediction\output\fhn

python SIGN-prediction\run_sst_prediction.py `
  --data SIGN-data\prediction\sst_enso_71987.npz `
  --output SIGN-prediction\output\sst
```

Discovery fits multidimensional systems one coordinate at a time. Prediction
instead advances the complete state vector jointly after the forecast
boundary, without accessing future components.

### Inspect or expand robustness protocols

```powershell
python Sign-Robust\scripts\validate_configs.py
python Sign-Robust\scripts\expand_matrix.py `
  --experiment observation_noise_sampling `
  --out Sign-Robust\output\observation_noise_sampling.jsonl
```

`Sign-Robust/configs/` is the authoritative experiment registry. Matrix
expansion writes deterministic JSONL task records; it does not start training.

## Data policy and scope

`SIGN-data/synthetic` and the 1,000-node FHN data are smoke-scale examples.
Large synthetic trajectories are generated on demand. Raw empirical networks
and the 22,198-node FHN partition input are not included; see
`Empirical_networks/README.md` for acquisition instructions. The bundled SST
example contains the time-major 71,987-node trajectory used by the SST runner;
its provenance is documented in `SIGN-data/prediction/SST_PROVENANCE.md`.

The codebase distributes SIGN methods only. Baselines including MTGNN,
ASTGCN, MSTGCN, STSGCN, STGCN, TP-SINDy, LaGNA, and related implementations
remain external and retain their own licenses and citations. The manuscript
crosswalk, including what is directly runnable versus protocol-only, is in
[`docs/EXPERIMENT_COVERAGE.md`](docs/EXPERIMENT_COVERAGE.md).

## Outputs and citation

Each runnable component writes results beneath its local `output/` directory;
these directories retain only their README files in a clean package. Outputs
typically record run configuration, metrics, phase provenance, and model
artifacts. Cite the work using [`CITATION.cff`](CITATION.cff), and consult
[`DATA_LICENSES.md`](DATA_LICENSES.md) before redistributing data or comparing
against external methods.
