# SIGN source code and experiment protocols

This directory contains the public SIGN implementation used for network
dynamical-system discovery and temporal prediction, together with data
preparation utilities and the configuration-driven robustness protocols used
in the manuscript. Third-party baseline implementations are not redistributed.
The standalone figure renderers are in [`../Plot/`](../Plot/README.md).

## Repository layout

```text
code/
├── SIGN-phase/          Primary SIGN algorithm entry point and canonical E2V3 implementation
├── SIGN-benchmarks/     Compact Fig. 2 benchmark runner built on SIGN-phase
├── SIGN_fhn_pred/       Standalone FHN partitioned-network prediction project
├── SIGN_sst_pred/       Standalone ENSO/SST prediction project with Fourier time terms
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

## Choose an entry point

### Verify the canonical implementation

```powershell
python SIGN-phase\scripts\verify_package.py
```

This performs public import, basis-library, data, and configuration smoke
checks without starting a full experiment.

### Run the primary SIGN algorithm

```powershell
python SIGN-phase\trainer.py --help
```

`SIGN-phase` is the canonical implementation of Phase-I support discovery and
Phase-II fixed-support coefficient refinement. It is the repository's primary
algorithm entry point. The trainer consumes the PyTorch-Geometric dataset
format expected by `SIGN-phase/utils_file/data_loader.py`.
`Data_generation/` creates synthetic trajectories, and
`Sign-Robust/scripts/run_canonical_npz_case.py` converts a generated NPZ case
to this canonical input format.

### Reproduce the compact Fig. 2 benchmark

```powershell
python SIGN-benchmarks\run_fig2.py `
  --data-root SIGN-data\synthetic `
  --output SIGN-benchmarks\output\fig2
```

`SIGN-benchmarks` is a paper-specific convenience runner, not a second
algorithm entry point. It delegates Phase-I and Phase-II execution to
`SIGN-phase` using the clean L1 / `trig_exp_v2` Fig. 2 setting.

### Run prediction examples

```powershell
cd SIGN_fhn_pred
python trainer.py --ode_model FHN --network from_file --dims 2

cd ..\SIGN_sst_pred
python trainer.py --ode_model enso --num-atoms 71987 --dims 1 --time-stamp 120
```

These are standalone, experiment-specific projects rather than alternative
`SIGN-phase` entry points. Their layouts intentionally follow their native
PyTorch-Geometric data and prediction workflows. `SIGN_sst_pred` includes its
raw PyG input under `SIGN_data/enso_71987`; `SIGN_fhn_pred` requires a staged
PyG partition input. Read their package README files before running:
[`SIGN_fhn_pred/README.md`](SIGN_fhn_pred/README.md) and
[`SIGN_sst_pred/README.md`](SIGN_sst_pred/README.md).

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

`SIGN-data/synthetic` and the NPZ FHN record are smoke-scale/audit examples.
Large synthetic trajectories are generated on demand. Raw empirical networks
and the 22,198-node FHN partition input are not included; see
`Empirical_networks/README.md` for acquisition instructions. The bundled SST
NPZ record is a time-major 71,987-node archive; the native SST experiment input
is the PyG record under `SIGN_sst_pred/SIGN_data/`. Its provenance is documented
in `SIGN-data/prediction/SST_PROVENANCE.md`.

The codebase distributes SIGN methods only. Baselines including MTGNN,
ASTGCN, MSTGCN, STSGCN, STGCN, TP-SINDy, LaGNA, and related implementations
remain external and retain their own licenses and citations. The manuscript
crosswalk, including what is directly runnable versus protocol-only, is in
[`docs/EXPERIMENT_COVERAGE.md`](docs/EXPERIMENT_COVERAGE.md).

## Outputs and citation

Most runnable components write results beneath a local `output/` directory;
these directories retain only their README files in a clean package.
`SIGN_fhn_pred` and `SIGN_sst_pred` write timestamped runs beneath `logs/`
and selected CSV outputs in their working directories. Outputs typically
record run configuration, metrics, phase provenance, and model artifacts.
Cite the work using [`CITATION.cff`](CITATION.cff), and consult
[`DATA_LICENSES.md`](DATA_LICENSES.md) before redistributing data or comparing
against external methods.
