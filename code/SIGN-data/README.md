# SIGN data

This is the single public data directory for the SIGN source repository.

## Contents

| Directory | Contents |
|---|---|
| `synthetic/` | Smoke-scale trajectories for compact SIGN and format checks |
| `prediction/` | 1,000-node FHN smoke data and the supplied 71,987-node ENSO/SST data |

All arrays use `x=[time,node,dimension]`, `t=[time]`, and
`edge_index=[2,edge]`; edges follow `source -> target`. The machine-readable
catalog is `catalog.json`.

The synthetic files are small examples, not the large trajectories used for
paper-scale runs. Generate those trajectories with
`Data_generation/generate_dataset.py` using the parameters in the manuscript
or `Sign-Robust/configs`.

The FHN prediction smoke file has shape `[200,1000,2]`. The large FHN result
uses an externally prepared 22,198-node partition of `bn-human`; the source
network and partition output are not redistributed. See
`Empirical_networks/README.md`.

The SST/ENSO file has shape `[120,71987,1]` and preserves the supplied graph
and source time coordinate. Stored index 0 is the initial observed/true state
used to initialize the rollout; it is not included in the 119-month plotted
sequence (indices 1–119). The manuscript identifies the source as the SSTG
dataset [45], with the full citation given as Cao et al., *Earth System
Science Data* 13, 2111–2134 (2021) [SI S25]. The supplied file is a publicly
released cleaned derivative; cite both the upstream SSTG dataset and this
repository's provenance record when reusing it.
