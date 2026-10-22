# Dual L-TAE S1+S2 — matched fold-0 result

The new model is fully trainable from scratch and does not use TESSERA or another foundation model.

## Final epoch

| Metric | Baseline | TESSERA S2 | TESSERA S1+S2 | Dual L-TAE S1+S2 |
|---|---:|---:|---:|---:|
| Binary 14d | 0.117 | 0.170 | 0.207 | **0.223** |
| Binary 30d | 0.178 | 0.239 | 0.307 | **0.330** |
| Binary 60d | 0.352 | 0.371 | 0.458 | **0.500** |
| Binary 365d | 0.627 | 0.557 | 0.679 | **0.680** |
| Macro 14d | 0.125 | 0.173 | 0.222 | **0.192** |
| Macro 30d | 0.173 | 0.231 | 0.315 | **0.309** |
| Macro 60d | 0.314 | 0.386 | 0.425 | **0.473** |
| Macro 365d | 0.504 | 0.540 | 0.579 | **0.616** |

## Best epoch by Macro F1 @ 60 days

- **baseline**: epoch 4, Macro60=0.319, Binary60=0.362
- **tessera_s2**: epoch 6, Macro60=0.442, Binary60=0.426
- **tessera_s1s2**: epoch 4, Macro60=0.431, Binary60=0.440
- **dual_ltae_s1s2**: epoch 9, Macro60=0.473, Binary60=0.500

## Run folders

- **baseline**: `runs/full_baseline_10ep_20261021-165735`
- **tessera_s2**: `runs/full_tessera_s2_10ep_20261020-213828`
- **tessera_s1s2**: `runs/full_tessera_s1s2_10ep_20261021-051601`
- **dual_ltae_s1s2**: `runs/full_dual_ltae_s1s2_10ep_20261021-212830`
