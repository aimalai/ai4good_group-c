# Gated Dual L-TAE S1+S2

Matched fold-0 ablation of learned multimodal gating versus the original Dual L-TAE concatenation fusion.

- Same S1/S2 temporal encoders
- Same fold 0
- Same 10 epochs
- Same batch size and sampling
- Same operational evaluation
- No foundation model

- Control: `runs/full_dual_ltae_s1s2_10ep_20261021-212830`
- Gated: `runs/full_gated_dual_ltae_s1s2_10ep_20261027-142119`

## Final epoch

| Metric | Dual L-TAE | Gated | Change |
|---|---:|---:|---:|
| Binary 14d | 0.223 | 0.242 | +0.019 |
| Binary 30d | 0.330 | 0.318 | -0.012 |
| Binary 60d | 0.500 | 0.472 | -0.028 |
| Binary 365d | 0.680 | 0.676 | -0.004 |
| Macro 14d | 0.192 | 0.194 | +0.002 |
| Macro 30d | 0.309 | 0.285 | -0.024 |
| Macro 60d | 0.473 | 0.425 | -0.047 |
| Macro 365d | 0.616 | 0.580 | -0.036 |

## Best Macro F1 @ 60d

- Original Dual L-TAE: 0.473 (epoch 9)
- Gated Dual L-TAE: 0.427 (epoch 4)
- Change: -0.046
