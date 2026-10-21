#!/usr/bin/env bash
set -euo pipefail

cd /zfs/ai4good/student/thein/ai4good_group-c

export DISFOR_DATA_ROOT=/zfs/ai4good/datasets/forest_disturbance
export HF_HOME=/zfs/ai4good/student/thein/hf_cache
export UV_CACHE_DIR=/zfs/ai4good/student/thein/uv_cache

EPOCHS=10
START_ALL=$SECONDS

echo "============================================================"
echo " SENTINEL-1 + SENTINEL-2 FUSION EXPERIMENT"
echo
echo " CONTROL:"
echo "   TESSERA v2 / Sentinel-2 only"
echo
echo " EXPERIMENT:"
echo "   frozen TESSERA S2 + trainable S1 GRU late fusion"
echo
echo " MATCHED:"
echo "   fold 0"
echo "   S2 target dates only"
echo "   sampling alpha 0.5"
echo "   full training sampler"
echo "   10 epochs"
echo "   same loss"
echo "   same operational evaluation"
echo "============================================================"

echo
echo "============================================================"
echo " 1 / 2 -- TESSERA S2-ONLY CONTROL"
echo "============================================================"

START=$SECONDS

uv run \
  --with "huggingface_hub>=0.26" \
  python experiments/tristan_tessera_v2/train.py \
  data.reader=pixel_cache \
  data.image_size=1 \
  data.sensors='[s2]' \
  data.target_sensors='[s2]' \
  data.normalization=null \
  data.fold=0 \
  data.batch_size=16 \
  data.num_workers=8 \
  data.sampling_alpha=0.5 \
  trainer.max_epochs=$EPOCHS \
  trainer.limit_train_batches=1.0 \
  logger.kind=csv \
  run_name=full_tessera_s2_10ep

echo
echo "S2-only finished in $(( (SECONDS-START)/60 )) minutes"

echo
echo "============================================================"
echo " 2 / 2 -- TESSERA S2 + SENTINEL-1 FUSION"
echo "============================================================"

START=$SECONDS

uv run \
  --with "huggingface_hub>=0.26" \
  python experiments/tristan_s1s2_fusion/train.py \
  data.reader=pixel_cache \
  data.image_size=1 \
  data.sensors='[s1,s2]' \
  data.target_sensors='[s2]' \
  data.normalization=null \
  data.fold=0 \
  data.batch_size=16 \
  data.num_workers=8 \
  data.sampling_alpha=0.5 \
  trainer.max_epochs=$EPOCHS \
  trainer.limit_train_batches=1.0 \
  logger.kind=csv \
  run_name=full_tessera_s1s2_10ep

echo
echo "S1+S2 fusion finished in $(( (SECONDS-START)/60 )) minutes"

echo
echo "============================================================"
echo " FINAL SENSOR-FUSION COMPARISON"
echo "============================================================"

python3 - <<'PY'
import glob
import json
from pathlib import Path


def latest(prefix):
    runs = sorted(
        glob.glob(
            f"runs/{prefix}_*"
        )
    )

    if not runs:
        raise RuntimeError(
            f"No run found for {prefix}"
        )

    return Path(runs[-1])


def epochs(run):
    files = sorted(
        (
            run
            / "predictions"
        ).glob(
            "epoch_*_metrics.json"
        )
    )

    return [
        (
            int(
                f.stem.split("_")[1]
            ),
            json.loads(
                f.read_text()
            ),
        )
        for f in files
    ]


def metric(m, key):
    v = m.get(key)

    if v is None:
        return float("nan")

    return float(v)


s2_run = latest(
    "full_tessera_s2_10ep"
)

fusion_run = latest(
    "full_tessera_s1s2_10ep"
)

s2 = epochs(s2_run)
fusion = epochs(fusion_run)

print()
print("RUNS")
print("S2 only :", s2_run)
print("S1 + S2 :", fusion_run)
print()

keys = [
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

print(
    f"{'Epoch':<7}"
    f"{'S2 Bin60':>11}"
    f"{'Fusion Bin60':>14}"
    f"{'S2 Mac60':>11}"
    f"{'Fusion Mac60':>14}"
    f"{'S2 Bin365':>12}"
    f"{'Fusion Bin365':>15}"
)

print("-" * 84)

for (
    (epoch_s2, ms2),
    (epoch_f, mf),
) in zip(
    s2,
    fusion,
    strict=True,
):
    assert epoch_s2 == epoch_f

    print(
        f"{epoch_s2:<7}"
        f"{metric(ms2, 'operational/binary_f1_60d'):>11.3f}"
        f"{metric(mf, 'operational/binary_f1_60d'):>14.3f}"
        f"{metric(ms2, 'operational/macro_f1_60d'):>11.3f}"
        f"{metric(mf, 'operational/macro_f1_60d'):>14.3f}"
        f"{metric(ms2, 'operational/binary_f1_365d'):>12.3f}"
        f"{metric(mf, 'operational/binary_f1_365d'):>15.3f}"
    )

print()
print("============================================================")
print(" FINAL EPOCH")
print("============================================================")
print()

ms2 = s2[-1][1]
mf = fusion[-1][1]

print(
    f"{'Metric':<15}"
    f"{'S2 only':>12}"
    f"{'S1 + S2':>12}"
    f"{'Difference':>14}"
)

print("-" * 53)

for label, key in keys:
    a = metric(
        ms2,
        key,
    )

    b = metric(
        mf,
        key,
    )

    print(
        f"{label:<15}"
        f"{a:>12.3f}"
        f"{b:>12.3f}"
        f"{b-a:>+14.3f}"
    )

print()
print("============================================================")
print(" BEST EPOCH BY MACRO F1 @ 60 DAYS")
print("============================================================")
print()

best_s2 = max(
    s2,
    key=lambda x: metric(
        x[1],
        "operational/macro_f1_60d",
    ),
)

best_fusion = max(
    fusion,
    key=lambda x: metric(
        x[1],
        "operational/macro_f1_60d",
    ),
)

print(
    "S2-only:"
    f" epoch {best_s2[0]}"
    f" macro60={metric(best_s2[1], 'operational/macro_f1_60d'):.3f}"
    f" binary60={metric(best_s2[1], 'operational/binary_f1_60d'):.3f}"
)

print(
    "Fusion :"
    f" epoch {best_fusion[0]}"
    f" macro60={metric(best_fusion[1], 'operational/macro_f1_60d'):.3f}"
    f" binary60={metric(best_fusion[1], 'operational/binary_f1_60d'):.3f}"
)

print()
print("============================================================")
PY

echo
echo "TOTAL WALL TIME: $(( (SECONDS-START_ALL)/60 )) minutes"
echo "SENSOR-FUSION EXPERIMENT COMPLETE"
