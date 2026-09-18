# Supplementary Table S8 source data

`relative_error_reduction.tsv` contains the Phase-I-to-Phase-II Euler-rollout
error reduction reported for Rössler dynamics with the L2 library under 50 dB
observational noise. Means and sample standard deviations were recomputed from
`../S07_Rollout_ablation/euler_rollout_raw.tsv` after filtering
`system = Rossler`, `condition = snr50`, and methods `Phase-I` and
`Phase-II fixed` (five random seeds per horizon).

The reported percentage is
`(phase1_mean - phase2_mean) / phase1_mean × 100`.

`Panel_e_rollout_summary.tsv` is retained as a separate consensus-method
comparison table and is not the input to the relative-reduction calculation.

