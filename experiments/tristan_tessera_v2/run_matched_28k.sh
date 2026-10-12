#!/usr/bin/env bash
set -euo pipefail

cd /zfs/ai4good/student/thein/ai4good_group-c

export DISFOR_DATA_ROOT=/zfs/ai4good/datasets/forest_disturbance
export HF_HOME=/zfs/ai4good/student/thein/hf_cache
export UV_CACHE_DIR=/zfs/ai4good/student/thein/uv_cache

echo "=================================================="
echo " TESSERA v2 FOUNDATION MODEL"
echo " 28,000 matched training batches"
echo " S2 only / same fold / same loss / SAME METRICS"
echo "=================================================="

uv run \
  --with "huggingface_hub>=0.26" \
  python experiments/tristan_tessera_v2/train.py \
  data.reader=pixel_cache \
  data.image_size=1 \
  data.sensors='[s2]' \
  data.target_sensors='[s2]' \
  data.normalization=null \
  data.batch_size=16 \
  data.num_workers=8 \
  data.sampling_alpha=0.5 \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=28000 \
  logger.kind=csv \
  run_name=matched_tessera_v2_28k

echo
echo "=================================================="
echo " THREE-WAY MATCHED COMPARISON"
echo "=================================================="

python3 - <<'PY'
import csv
import glob


def latest(prefix):
    files = sorted(
        glob.glob(
            f"runs/{prefix}_*/metrics.csv"
        )
    )

    if not files:
        return None

    return files[-1]


def metric(path, key):
    if path is None:
        return float("nan")

    rows = list(
        csv.DictReader(
            open(path)
        )
    )

    for row in reversed(rows):
        if row.get(key):
            return float(row[key])

    return float("nan")


runs = [
    (
        "Baseline",
        latest("matched_baseline_28k"),
    ),
    (
        "GRU",
        latest("matched_gru_28k"),
    ),
    (
        "TESSERA",
        latest("matched_tessera_v2_28k"),
    ),
]

metrics = [
    (
        "Binary 14d",
        "operational/binary_f1_14d",
    ),
    (
        "Binary 30d",
        "operational/binary_f1_30d",
    ),
    (
        "Binary 60d",
        "operational/binary_f1_60d",
    ),
    (
        "Binary 365d",
        "operational/binary_f1_365d",
    ),
    (
        "Macro 14d",
        "operational/macro_f1_14d",
    ),
    (
        "Macro 30d",
        "operational/macro_f1_30d",
    ),
    (
        "Macro 60d",
        "operational/macro_f1_60d",
    ),
    (
        "Macro 365d",
        "operational/macro_f1_365d",
    ),
]

print()

print(
    f"{'Metric':<15}"
    f"{'Baseline':>12}"
    f"{'GRU':>12}"
    f"{'TESSERA':>12}"
)

print("-" * 51)

for label, key in metrics:
    vals = [
        metric(path, key)
        for _, path in runs
    ]

    print(
        f"{label:<15}"
        + "".join(
            f"{x:>12.3f}"
            for x in vals
        )
    )

print()
print("Validation loss:")

for name, path in runs:
    print(
        f"{name:<10}: "
        f"{metric(path, 'val/loss'):.4f}"
    )

print()
print("Files:")

for name, path in runs:
    print(
        f"{name:<10}: {path}"
    )
PY

echo
echo "=================================================="
echo " TESSERA EXPERIMENT COMPLETE"
echo "=================================================="
