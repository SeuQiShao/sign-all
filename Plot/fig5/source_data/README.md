# Source data

This directory is intentionally empty of raw numeric files for the submission plotting package.

The final figure can be regenerated entirely from the compact CSV files in `../plot_data/`. The original server-side FHN files are large node-by-time matrices (true/predicted trajectories for each condition and segment) and are not required by the plotting code. Keeping them out of this package avoids duplicating bulky experimental data while leaving the submitted plot data directly auditable and usable.

If a journal or repository later requires the full raw matrices, they should be deposited separately with their original acquisition metadata; they are not silently reconstructed from the compact CSVs here.

