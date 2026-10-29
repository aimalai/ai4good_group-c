# Fold-2 Replication and Checkpoint Selection

Matched 10-epoch fold-2 replication of StatisticalMLP and Dual L-TAE S1+S2.

Every Dual L-TAE epoch was checkpointed. The validation checkpoint with the highest Macro F1 @ 60 days is preserved as `best_macro60.ckpt`.

- Baseline: `runs/fold2_baseline_10ep_20261029-123554`
- Dual L-TAE: `runs/fold2_dual_ltae_s1s2_10ep_20261029-154044`

## Final epoch

| Metric | Baseline | Dual L-TAE | Change |
|---|---:|---:|---:|
| Binary 14d | 0.126 | 0.194 | +0.068 |
| Binary 30d | 0.186 | 0.269 | +0.084 |
| Binary 60d | 0.303 | 0.403 | +0.100 |
| Binary 365d | 0.533 | 0.644 | +0.111 |
| Macro 14d | 0.127 | 0.147 | +0.020 |
| Macro 30d | 0.159 | 0.193 | +0.034 |
| Macro 60d | 0.232 | 0.303 | +0.071 |
| Macro 365d | 0.384 | 0.474 | +0.090 |

## Validation-selected Macro F1 @ 60 days

- Baseline: 0.264 (epoch 5)
- Dual L-TAE: 0.305 (epoch 6)
- Difference: +0.041

Checkpoint selection is performed on validation performance; these numbers should therefore be described as validation results, not held-out test results.
