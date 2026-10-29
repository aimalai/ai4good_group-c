# Fold-1 Replication

Matched 10-epoch replication of the original StatisticalMLP baseline and Dual L-TAE S1+S2 on fold 1.

- Baseline: `runs/fold1_baseline_10ep_20261028-124147`
- Dual L-TAE: `runs/fold1_dual_ltae_s1s2_10ep_20261028-155121`

## Final epoch

| Metric | Baseline | Dual L-TAE | Change |
|---|---:|---:|---:|
| Binary 14d | 0.167 | 0.215 | +0.048 |
| Binary 30d | 0.275 | 0.297 | +0.022 |
| Binary 60d | 0.440 | 0.453 | +0.013 |
| Binary 365d | 0.645 | 0.703 | +0.058 |
| Macro 14d | 0.158 | 0.167 | +0.009 |
| Macro 30d | 0.270 | 0.250 | -0.019 |
| Macro 60d | 0.401 | 0.371 | -0.030 |
| Macro 365d | 0.576 | 0.589 | +0.013 |

## Best Macro F1 @ 60 days

- Baseline: 0.401 (epoch 9)
- Dual L-TAE: 0.420 (epoch 3)
- Improvement: +0.019
