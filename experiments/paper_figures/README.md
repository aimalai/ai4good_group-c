# Paper figures

These figures are generated directly from the final model runs referenced in
the TESSERA and Dual L-TAE RESULTS.md files.

## Run provenance

- Baseline: `runs/full_baseline_10ep_20261021-165735`
- TESSERA S2: `runs/full_tessera_s2_10ep_20261020-213828`
- TESSERA S1+S2: `runs/full_tessera_s1s2_10ep_20261021-051601`
- Dual L-TAE S1+S2: `runs/full_dual_ltae_s1s2_10ep_20261021-212830`

## Figures

1. `01_training_convergence_macro60`
   - Validation operational Macro F1 @ 60d over all 10 epochs.
   - Four models, fold 0.

2. `02_operational_performance`
   - Final-epoch operational Binary and Macro F1.
   - Horizons: 14d, 30d, 60d, 365d.
   - Values correspond to the four-way RESULTS.md comparison.

3. `03_dual_ltae_sample_timeline_zoom.png`
   - Real validation sample from Dual L-TAE fold-0 epoch 009 predictions.
   - Epoch 009 is both the final and best Macro60 epoch for this run.
   - Selected sample: 28
   - Predicted class: Wildfire
   - True class: Wildfire
   - Alert lag: 0 days
   - Triangle marker(s) in the probability panel are operational CUSUM alerts.
   - The matching shown here is evaluated at the 60-day horizon.

4. `04_multifold_peak_macro60`
   - Peak validation Macro F1 @ 60d for Baseline vs Dual L-TAE.
   - Folds 0, 1 and 2.

Important: these are validation-fold results, not a separate held-out test set.
