# Tristan — Full-Patch Data Loading Experiment

Standalone benchmark for real satellite image patches.

The professor's baseline code is not modified.

## Features

- Lazy Zarr reads: no full dataset copy
- Reads only the requested spatial crop
- Multiworker PyTorch DataLoader
- Persistent workers and prefetching
- Per-worker Zarr handle cache
- Pinned memory for CUDA transfers
- Optional CUDA stream prefetching
- GPU convolution workload for realistic overlap testing
- Synthetic correctness tests
- CPU/GPU throughput sweep
- JSON and CSV results

## Scope

This measures image patch I/O, not the full labelled DISFOR
training pipeline. Each sample is one image frame, not a
complete irregular temporal sequence.

The selected Zarr array and layout must be verified against
the real dataset before interpreting results.

## Run

From the repository root on sushko:

    bash experiments/tristan_full_patch_loader/run_all.sh

Or inspect only:

    uv run --frozen python experiments/tristan_full_patch_loader/benchmark_loader.py --mode inspect

Results appear under:

    experiments/tristan_full_patch_loader/results/

## Interpretation

Compare equivalent configurations only.

CPU-only runs measure patch decoding and DataLoader overhead.
GPU runs additionally measure host-to-device transfer.
Convolution runs add GPU computation.

Logical GiB/s is tensor payload throughput, not disk bandwidth.

The fastest worker count is machine- and workload-dependent.
More workers are not automatically better.
