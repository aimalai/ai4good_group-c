# Sentinel-1 + Sentinel-2 Fusion Experiment

## Goal

Test whether adding Sentinel-1 radar information improves forest-disturbance
monitoring compared with:

1. the original StatisticalMLP baseline;
2. a Sentinel-2-only TESSERA temporal foundation-model representation.

## Experimental setup

All results below use:

- fold 0
- 10 full training epochs
- batch size 16
- sampling alpha 0.5
- Sentinel-2 target dates
- the same operational event evaluation
- F1 evaluated at 14, 30, 60 and 365 day horizons

### Models

**Baseline**
Original StatisticalMLP using Sentinel-2 centre-pixel observations.

**TESSERA S2**
Frozen pretrained TESSERA-v2 Sentinel-2 temporal encoder with a trainable
classification head.

**TESSERA S1+S2**
The same frozen TESSERA Sentinel-2 encoder combined with a trainable GRU
encoding the Sentinel-1 VV/VH temporal sequence. Recent and previous-year
representations are fused before classification.

## Final-epoch results

| Metric | Baseline | TESSERA S2 | TESSERA S1+S2 |
|---|---:|---:|---:|
| Binary F1 14d | 0.117 | 0.170 | **0.207** |
| Binary F1 30d | 0.178 | 0.239 | **0.307** |
| Binary F1 60d | 0.352 | 0.371 | **0.458** |
| Binary F1 365d | 0.627 | 0.557 | **0.679** |
| Macro F1 14d | 0.125 | 0.173 | **0.222** |
| Macro F1 30d | 0.173 | 0.231 | **0.315** |
| Macro F1 60d | 0.314 | 0.386 | **0.425** |
| Macro F1 365d | 0.504 | 0.540 | **0.579** |

The S1+S2 model has the strongest final-epoch score on all eight reported
operational metrics.

## Important qualification

Performance varies across epochs. In particular, the highest single
Macro-F1@60d checkpoint was obtained by TESSERA S2:

- Baseline: 0.319 (epoch 4)
- TESSERA S2: 0.442 (epoch 6)
- TESSERA S1+S2: 0.431 (epoch 4)

Therefore the current result should be described as a strong fold-0 result,
not evidence that fusion is universally superior. Additional folds are needed.

## Run locations

- Baseline:
  `runs/full_baseline_10ep_20261021-165735`
- TESSERA S2:
  `runs/full_tessera_s2_10ep_20261020-213828`
- TESSERA S1+S2:
  `runs/full_tessera_s1s2_10ep_20261021-051601`

Machine-readable comparison results are stored in:

`experiments/tristan_baseline_10ep/comparison_3way.csv`

and

`experiments/tristan_baseline_10ep/comparison_3way.json`
