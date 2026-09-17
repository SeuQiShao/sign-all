# Predicting Dynamics of Ultra-Large Complex Systems by Inferring Governing Equations

This repository accompanies the manuscript *Predicting Dynamics of Ultra-Large Complex Systems by Inferring Governing Equations*. It provides the public implementation of **SIGN**, a two-phase approach for discovering governing equations in networked dynamical systems and using them for prediction. The release also contains reproducible figure-rendering packages and machine-readable source data for the supplementary tables.

## Contents

| Path | Purpose |
|---|---|
| [`code/`](code/README.md) | SIGN implementations, data-generation tools, prediction runners, and robustness protocols |
| [`Plot/`](Plot/README.md) | Self-contained Python packages for all main and supplementary figures |


## Quick start

The shortest end-to-end check is the canonical-package verification:

```powershell
git clone <repository-url>
cd <repository-directory>\code

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python SIGN-phase\scripts\verify_package.py
```

The project relies on PyTorch and PyTorch Geometric. For GPU use, install the PyTorch build compatible with your CUDA version before installing the remaining requirements.

### Run representative workflows

From `code/`, the compact Fig. 2 discovery benchmark can be run with:

```powershell
python SIGN-main\run_fig2.py `
  --data-root SIGN-data\synthetic `
  --output SIGN-main\output\fig2
```

The supplied prediction examples are:

```powershell
python SIGN-prediction\run_fhn_prediction.py `
  --data SIGN-data\prediction\fhn_2d.npz `
  --output SIGN-prediction\output\fhn

python SIGN-prediction\run_sst_prediction.py `
  --data SIGN-data\prediction\sst_enso_71987.npz `
  --output SIGN-prediction\output\sst
```

For the full experiment map—including what is directly runnable, what needs external inputs, and what is protocol-only—see [`code/docs/EXPERIMENT_COVERAGE.md`](code/docs/EXPERIMENT_COVERAGE.md).

## Reproduce figures

Every figure package can be rendered independently from its own directory, using its `code/` renderer and compact `plot_data/` inputs. For example:

```powershell
cd ..\Plot\fig3
python -m pip install -r code\requirements.txt
python code\plot_fig3.py
```

`Plot/` covers main Figs. 1–6 and Supplementary Figs. SI1–SI4. The packages write editable SVG/PDF outputs and high-resolution PNG/TIFF previews. Their data are presentation-ready inputs: rendering a figure does not rerun a model or modify experiment data. See [`Plot/README.md`](Plot/README.md) for the full package index and output locations.

## Code structure

The main implementation components are:

- `SIGN-main/`: compact complete Phase-I/Phase-II path for the Fig. 2 benchmark.
- `SIGN-phase/`: canonical E2V3 implementation for support discovery and fixed-support coefficient refinement.
- `SIGN-prediction/`: joint-vector FHN, SST, and generic equation-prediction runners.
- `Sign-Robust/`: declarative robustness configurations, validators, and experiment-matrix utilities.
- `Data_generation/`: simulation, graph, robustness, and data-preparation tools.
- `SIGN-data/`: bundled smoke-scale synthetic data plus FHN and SST examples.

SIGN discovers equations for multidimensional systems one coordinate at a time. During prediction, it advances the complete state vector jointly after the forecast boundary and does not read future values from another state component.

## Data, external methods, and outputs

The distributed synthetic and 1,000-node FHN data are smoke-scale examples. Large synthetic trajectories can be generated on demand. Raw empirical networks and the 22,198-node FHN partition input are intentionally omitted; follow [`code/Empirical_networks/README.md`](code/Empirical_networks/README.md) to obtain and prepare them. The included SST trajectory and its provenance are documented in [`code/SIGN-data/prediction/SST_PROVENANCE.md`](code/SIGN-data/prediction/SST_PROVENANCE.md).

This release contains SIGN code and experiment protocols only. It does not redistribute third-party baseline implementations (including MTGNN, ASTGCN, MSTGCN, STSGCN, STGCN, TP-SINDy, and LaGNA). Details of data rights, external baselines, and required attribution are in [`code/DATA_LICENSES.md`](code/DATA_LICENSES.md).

Generated outputs are written beneath the relevant component's `output/` directory. The clean repository retains output-directory README files but not large generated artifacts.

## Citation and license

Please cite the associated manuscript when using SIGN. Citation metadata and the complete author list are provided in [`code/CITATION.cff`](code/CITATION.cff).

SIGN-authored source code is released under the MIT License; see [`code/LICENSE`](code/LICENSE). The license does not extend to third-party data, external baseline implementations, or dependencies.
