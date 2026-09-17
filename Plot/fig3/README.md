# Fig. 3 — Phase-I / Phase-II

This folder is a self-contained Python redraw package for the Nature-style Fig. 3 composite (panels a–h). The renderer uses the compact CSV files in `plot_data`; no raw experimental files are required for the final redraw.

## Run

From this folder, run:

```powershell
python code\plot_fig3.py
```

The publication exports are written to `code\output`:

- `Fig3_phase1_phase2.pdf` — vector PDF for submission;
- `Fig3_phase1_phase2.svg` — editable vector SVG;
- `Fig3_phase1_phase2.png` — 600 dpi preview;
- `Fig3_phase1_phase2.tiff` — 600 dpi TIFF export;
- `Fig3_phase1_phase2_manifest.json` — export manifest.

## Data map

| Panel | CSV | Content |
|---|---|---|
| a | `panel_a.csv` | Original library size, Phase-I retained support and reduction by system/library |
| b | `panel_b.csv` | Precision, recall, F1 and K summaries for L1/L2/L3 |
| c | `panel_c.csv` | 24-cell DBSCAN sensitivity snapshot; cell text is absolute F1 and fill is ΔF1 |
| d | `panel_d.csv` | DBSCAN, All, Voting and HC consensus comparison |
| e | `panel_e.csv` | Euler rollout error at horizons 5, 10, 50 and 100 |
| f | `panel_f.csv` | L2 Phase-I/Phase-II F1 by system and condition |
| g | `panel_g.csv` | Global paired Phase-I → Phase-II refinement summary |
| h | `panel_h.csv` | Rössler/L2 term-wise selection frequency and active-coefficient means |

`export_plot_data.py` is an optional archival helper for regenerating these compact CSVs from the experiment archive. The final plotting script itself reads only `plot_data`.

## Scope and conventions

- Library widths are L1/L2/L3 = 50/100/150 candidate terms.
- Conditions are Clean, Noise (SNR 50) and Sparse (200 observations).
- Phase II refines the fixed Phase-I mask and therefore cannot add terms excluded in Phase I.
- The nominal DBSCAN parameters are `eps = 0.15 × sqrt(number of features)` and `min_samples = 2`; panel c reports parameter factors relative to these values.
- Rollout error is the cumulative normalized squared error over each forecast horizon, summarized over five seeds.
- Panel h selection frequencies are over five seeds; coefficient labels are means conditional on the term being selected.

The `source_data` directory intentionally contains no raw observations. This
keeps the submission package small while preserving exact redraw capability
through the compact plot data.
