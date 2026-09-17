# Empirical networks

This directory is intentionally a placeholder. Raw empirical networks are
not redistributed with the SIGN source repository. The manuscript's set
includes the voles social network, human and mouse gene-regulatory networks,
human and fly brain networks, the GitHub collaboration network, and Catster.
The manuscript's Data Availability statement directs readers to the Network
Repository and the Stanford Large Network Dataset Collection (SNAP); the
manuscript references these sources as [33], [41]–[44] and the SI gives the
repository citations as [S10] and [S13]. Download each dataset from its
original repository, verify its individual terms and citation, and pass the
local file to `Data_generation/prepare_empirical_networks.py`.

The preparation utility accepts whitespace- or comma-separated edge lists and
can normalize indexing, directedness, weights, and self-loops. For example:

```powershell
python Data_generation/prepare_empirical_networks.py `
  --source path/to/downloaded/network.edges `
  --out Empirical_networks/processed/network.npz `
  --one-indexed --undirected --expand-undirected
```

The large FHN experiment uses the `bn-human` network and the existing
partition workflow to select 22,198 nodes. That source data and its partition
output must be obtained externally. ENSO/SST data are supplied separately in
`SIGN-data/prediction`.

Do not assume that the source repositories grant redistribution rights; cite
each original dataset as required by its provider and by the manuscript.
