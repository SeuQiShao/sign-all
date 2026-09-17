# SIGN prediction

This folder contains separate FHN and SST prediction entry points plus a
generic equation runner for SIGN temporal prediction. This repository
contains SIGN implementations only; persistence, AR, climatology, and
third-party graph-forecasting baselines are not included.

## Information policy

Discovery is equation-wise, but prediction is joint-vector. At the forecast
boundary the model receives one complete observed state; after that boundary
it advances all dimensions together and never reads future values from any
other dimension. This is essential for the 2D FHN experiment.

## Run

From the `NC` directory:

```powershell
python SIGN-prediction/run_fhn_prediction.py `
  --data SIGN-data/prediction/fhn_2d.npz `
  --output SIGN-prediction/output/fhn

python SIGN-prediction/run_sst_prediction.py `
  --data SIGN-data/prediction/sst_enso_71987.npz `
  --output SIGN-prediction/output/sst

python SIGN-prediction/run_equation_prediction.py `
  --system SIS --data path/to/sis.npz `
  --output SIGN-prediction/output/sis
```

`--train-steps` and `--horizon` override the default split. Defaults are an
80/20 split for synthetic systems and 96 observed months plus 24 forecast
months for SST. Normalization, when enabled, is fitted on the observed
training window only.

SST additionally selects dominant Fourier periods from that same training
window and fits fixed Fourier coefficients jointly with the E2V3 state and
graph terms. Fourier phase is evaluated on the absolute source time axis;
future SST values are never used to select periods or coefficients.

Outputs include the SIGN prediction, `metrics.csv`, `config.json`,
`training_history.json`, `joint_e2v3_model.pt`, and Phase-I provenance under
`phase1/`.

## Data

The runnable examples are in `../SIGN-data/prediction`. The 1,000-node FHN
file is a smoke example. The large FHN protocol uses 22,198 nodes from the
external `bn-human` partition workflow and can be supplied in the same NPZ
format. The supplied SST file has 71,987 nodes, 120 stored time steps, and the
source graph already converted to the public time-major layout. The first
stored state (index 0) is the observed/true initialization state; figure
packages exclude it from the plotted sequence and display the remaining 119
months. The manuscript identifies the source as the SSTG dataset [45] (Cao
et al., *Earth System Science Data* 13, 2111–2134, 2021; SI [S25]).
