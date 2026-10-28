# NASA FIRMS External-Data Ablation

Dual L-TAE S1+S2 plus eight strictly past-only NASA FIRMS VIIRS active-fire features.

- Control: `runs/full_dual_ltae_s1s2_10ep_20261021-212830`
- +FIRMS: `runs/full_dual_ltae_s1s2_firms_10ep_20261022-211959`

No FIRMS detection from the target date or future is used.

## Final epoch

| Metric | Dual L-TAE | +FIRMS | Change |
|---|---:|---:|---:|
| Binary 14d | 0.223 | 0.225 | +0.002 |
| Binary 30d | 0.330 | 0.325 | -0.005 |
| Binary 60d | 0.500 | 0.492 | -0.008 |
| Binary 365d | 0.680 | 0.683 | +0.003 |
| Macro 14d | 0.192 | 0.200 | +0.008 |
| Macro 30d | 0.309 | 0.305 | -0.004 |
| Macro 60d | 0.473 | 0.459 | -0.013 |
| Macro 365d | 0.616 | 0.605 | -0.011 |

## Best Macro F1 @ 60d

- Dual L-TAE: 0.473 (epoch 9)
- +FIRMS: 0.466 (epoch 6)
