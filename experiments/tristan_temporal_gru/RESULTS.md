# TemporalGRU experiment results

Keep this file updated after every meaningful run. Do not overwrite old rows; add a new row.

## Reference baseline from the course repository (fold 0)

| Model | 14d binary F1 | 14d macro F1 | 30d binary F1 | 30d macro F1 | 60d binary F1 | 60d macro F1 | 365d binary F1 | 365d macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| StatisticalMLP reference | 0.12 | 0.10 | 0.19 | 0.15 | 0.34 | 0.28 | 0.60 | 0.48 |

These are the published starter-code reference values, not results from our machine. We should first
reproduce the baseline on `sushko` before claiming any improvement.

## Our runs

| Date | Run name | Git branch/commit | Epochs / train batches | 14d bin | 14d macro | 30d bin | 30d macro | 60d bin | 60d macro | 365d bin | 365d macro | Notes |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| YYYY-MM-DD | example | branch / commit | 1 / 200 |  |  |  |  |  |  |  |  | smoke run |

## Experiment decisions

Record decisions here so that later results remain interpretable.

- 2026-10-08: Experiment 01 chosen: replace baseline mean/std temporal aggregation with a small GRU.
- 2026-10-08: Keep S2-only, centre-pixel input and all loss/evaluation settings unchanged for the
  first comparison. This isolates temporal modelling as much as practical.
- 2026-10-08: No course baseline files are modified; experiment code is isolated in its own folder.

## Cluster validation — 2026-10-09

Environment:
- Node: sushko
- GPU: NVIDIA L40S
- Python: 3.12.12
- PyTorch: 2.11.0+cu128
- CUDA available: True

TemporalGRU smoke test:
- PASS
- Trainable parameters: 72,583
- Logits shape: (4, 7)
- Synthetic losses: 1.8415 -> 1.4683 -> 1.2259

Official repository test suite:
- 11 passed
- 0 failed
- 0 skipped
- Runtime: 116.81 s

Dataset:
- DISFOR_DATA_ROOT=/zfs/ai4good/datasets/forest_disturbance

Conclusion:
The original repository, real dataset, CUDA environment, and isolated TemporalGRU experiment all pass validation on sushko.

## Cluster validation — 2026-10-09

Environment:
- Node: sushko
- GPU: NVIDIA L40S
- Python: 3.12.12
- PyTorch: 2.11.0+cu128
- CUDA available: True

TemporalGRU smoke test:
- PASS
- Trainable parameters: 72,583
- Logits shape: (4, 7)
- Synthetic losses: 1.8415 -> 1.4683 -> 1.2259

Official repository test suite:
- 11 passed
- 0 failed
- 0 skipped
- Runtime: 116.81 s

Dataset:
- DISFOR_DATA_ROOT=/zfs/ai4good/datasets/forest_disturbance

Conclusion:
The original repository, real dataset, CUDA environment, and isolated TemporalGRU experiment all pass validation on sushko.
