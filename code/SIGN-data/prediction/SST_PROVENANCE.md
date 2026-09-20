# ENSO/SST Data Provenance

`data1.pt` contains the cleaned SST data used for the ENSO/Niño3.4 prediction experiment in this study.

The file is publicly available in this repository at:

https://github.com/SeuQiShao/sign-all/releases/tag/SSTdata

The dataset is derived from the publicly available SSTG dataset and was prepared for the experimental workflow using `Data_generation/prepare_sst_dataset.py`. The preprocessing preserves the supplied SST trajectory and graph structure and does not reconstruct the upstream dataset.

The upstream data source is SSTG [45]:

Cao, M. et al., “A new global gridded sea surface temperature data product based on multisource data,” *Earth System Science Data* **13**, 2111–2134 (2021) [Ref. 45 in the manuscript; Supplementary Information, Section S25].

SSTG Version 1.0 is publicly available at:

https://doi.org/10.5281/zenodo.4419804

and is distributed under the CC BY 4.0 license. The original SSTG attribution and license remain applicable to this derived dataset and are not replaced by the MIT license of this repository.

## Dataset used in this study

The processed dataset corresponds to the equatorial eastern Pacific/Niño3.4 experiment described in the manuscript, with spatial aggregation applied during preprocessing. It contains 120 monthly states over 71,987 retained spatial nodes.

For visualization, the SST values are inverse-normalized as:

`x * 1.3180 + 26.9876`

The spatial coordinates in `enso_xy_71987.csv` are aligned row-by-row with the SST node dimension. The figure-generation workflow uses:

- `latitude = -xy[:, 0]`
- `western longitude = 180 - xy[:, 1]`

The state at index 0 is the observed ground-truth state used to initialize the rollout. The figure-generation workflow displays states at indices 1–119, corresponding to 95 training months followed by 24 held-out test months.
