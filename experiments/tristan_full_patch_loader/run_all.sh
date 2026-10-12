#!/usr/bin/env bash
set -euo pipefail

cd /zfs/ai4good/student/thein/ai4good_group-c
export PATH="$HOME/.local/bin:$PATH"
export DISFOR_DATA_ROOT=/zfs/ai4good/datasets/forest_disturbance

EXP=experiments/tristan_full_patch_loader
mkdir -p "$EXP/results"

echo "=== Environment ==="
uv run --frozen python -c \
  'import torch,zarr; print("Torch:",torch.__version__); print("Zarr:",zarr.__version__); print("CUDA:",torch.cuda.is_available()); print("GPU:",torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'

echo
echo "=== Synthetic correctness tests ==="
uv run --frozen python "$EXP/self_test.py" \
  2>&1 | tee "$EXP/results/self_test.log"

echo
echo "=== Real dataset inspection ==="
uv run --frozen python "$EXP/benchmark_loader.py" \
  --mode inspect \
  --root "$DISFOR_DATA_ROOT/patches" \
  --max-stores 3 \
  2>&1 | tee "$EXP/results/inspection.log"

echo
echo "=== CPU/GPU benchmark sweep ==="
uv run --frozen python "$EXP/benchmark_loader.py" \
  --mode sweep \
  --root "$DISFOR_DATA_ROOT/patches" \
  --max-stores 12 \
  --crop 64 \
  --batch-size 8 \
  --frames-per-store 64 \
  --warmup 3 \
  --steps 15 \
  2>&1 | tee "$EXP/results/sweep.log"

echo
echo "=== GRU real-data GPU smoke ==="
uv run --frozen python experiments/tristan_temporal_gru/train.py \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=200 \
  logger.kind=csv \
  run_name=tristan_gru_amp_fixed_200 \
  2>&1 | tee "$EXP/results/gru_smoke.log"

echo
echo "=== ALL STEPS FINISHED ==="
echo "Loader results: $EXP/results/"
echo "GRU run: check runs/tristan_gru_amp_fixed_200_*"
