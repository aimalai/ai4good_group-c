# 10-Epoch Fold-0 Comparison

All three models use the same fold and operational evaluation.

- Baseline: `runs/full_baseline_10ep_20261021-165735`
- TESSERA S2: `runs/full_tessera_s2_10ep_20261020-213828`
- TESSERA S1+S2: `runs/full_tessera_s1s2_10ep_20261021-051601`

## Final epoch

| Metric | Baseline | TESSERA S2 | TESSERA S1+S2 | Fusion vs baseline |
|---|---:|---:|---:|---:|
| Binary 14d | 0.117 | 0.170 | 0.207 | +0.090 |
| Binary 30d | 0.178 | 0.239 | 0.307 | +0.129 |
| Binary 60d | 0.352 | 0.371 | 0.458 | +0.106 |
| Binary 365d | 0.627 | 0.557 | 0.679 | +0.052 |
| Macro 14d | 0.125 | 0.173 | 0.222 | +0.097 |
| Macro 30d | 0.173 | 0.231 | 0.315 | +0.142 |
| Macro 60d | 0.314 | 0.386 | 0.425 | +0.111 |
| Macro 365d | 0.504 | 0.540 | 0.579 | +0.076 |

## Best epoch by Macro F1 @ 60 days

- **baseline**: epoch 4, Macro60=0.319, Binary60=0.362, Macro365=0.477, Binary365=0.615
- **tessera_s2**: epoch 6, Macro60=0.442, Binary60=0.426, Macro365=0.602, Binary365=0.659
- **tessera_s1s2**: epoch 4, Macro60=0.431, Binary60=0.440, Macro365=0.569, Binary365=0.669
