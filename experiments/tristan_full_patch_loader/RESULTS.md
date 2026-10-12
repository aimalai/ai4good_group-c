# Full-Patch Loader Results

Status: implementation prepared; real-data benchmark pending.

Hardware: sushko, NVIDIA L40S.

Dataset: /zfs/ai4good/datasets/forest_disturbance

## Experiments

Compare 0, 2, 4 and 8 workers, plus GPU transfer and
CUDA prefetching.

Benchmark output is saved to results/*.csv and results/*.json.

## Important distinction

The baseline uses a precomputed center-pixel cache.
This experiment reads image patches from Zarr.

A speedup is only meaningful when the compared runs
read the same patch data with the same workload.
