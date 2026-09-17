# ENSO/SST provenance

`sst_enso_71987.npz` is a publicly released cleaned derivative of the public
SSTG dataset. It was converted to the repository's NPZ layout by
`Data_generation/prepare_sst_dataset.py`; conversion preserves the supplied
trajectory and graph and does not reconstruct the upstream dataset.

The upstream source is SSTG [45]: Cao, M. et al., “A new global gridded sea
surface temperature data product based on multisource data,” *Earth System
Science Data* 13, 2111–2134 (2021) [manuscript 45; SI S25]. SSTG Version 1.0
is available at `https://doi.org/10.5281/zenodo.4419804` under CC BY 4.0.
The upstream attribution and license remain applicable to this derivative;
they are not replaced by this repository's MIT license.

The cleaned derivative used here represents the equatorial East
Pacific/Niño3.4 experiment described in the manuscript, with spatial
aggregation. It contains 120 monthly states at 71,987 retained spatial nodes.
Figure CSVs inverse-normalize the data with `x * 1.3180 + 26.9876`.
`enso_xy_71987.csv` rows align with SST node columns; the figure workflow
uses `latitude = -xy[:, 0]` and western longitude `= 180 - xy[:, 1]`.

Stored index 0 is the observed/true state used to initialize rollout. The
figure packages display indices 1–119: 95 training months and 24 held-out
test months.
