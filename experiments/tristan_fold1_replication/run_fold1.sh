#!/usr/bin/env bash
set -euo pipefail

cd /zfs/ai4good/student/thein/ai4good_group-c

export DISFOR_DATA_ROOT=/zfs/ai4good/datasets/forest_disturbance
export UV_CACHE_DIR=/zfs/ai4good/student/thein/uv_cache

EXP=experiments/tristan_fold1_replication
mkdir -p "$EXP"

START=$SECONDS

echo
echo "============================================================"
echo " FOLD-1 REPLICATION"
echo
echo " 1) Original StatisticalMLP baseline"
echo " 2) Dual L-TAE S1+S2"
echo
echo " Same 10 epochs"
echo " Same operational evaluation"
echo "============================================================"
echo


echo
echo "============================================================"
echo " STEP 1/2 — FOLD-1 BASELINE"
echo "============================================================"
echo

uv run python scripts/train.py \
  --config configs/statistical_mlp.yaml \
  data.fold=1 \
  data.sensors='[s2]' \
  data.target_sensors='[s2]' \
  data.reader=pixel_cache \
  data.image_size=1 \
  data.batch_size=16 \
  data.num_workers=8 \
  data.sampling_alpha=0.5 \
  trainer.max_epochs=10 \
  trainer.limit_train_batches=1.0 \
  logger.kind=csv \
  run_name=fold1_baseline_10ep


echo
echo "============================================================"
echo " STEP 2/2 — FOLD-1 DUAL L-TAE S1+S2"
echo "============================================================"
echo

uv run python \
  experiments/tristan_ltae_late_fusion/train.py \
  data.reader=pixel_cache \
  data.image_size=1 \
  data.sensors='[s1,s2]' \
  data.target_sensors='[s2]' \
  data.normalization=null \
  data.fold=1 \
  data.batch_size=16 \
  data.num_workers=8 \
  data.sampling_alpha=0.5 \
  optimizer.lr=0.0003 \
  optimizer.weight_decay=0.01 \
  trainer.max_epochs=10 \
  trainer.limit_train_batches=1.0 \
  trainer.precision=bf16-mixed \
  logger.kind=csv \
  run_name=fold1_dual_ltae_s1s2_10ep


echo
echo "============================================================"
echo " BUILDING FOLD-1 COMPARISON"
echo "============================================================"

python3 - <<'PY'
import csv
import glob
import json
from pathlib import Path


OUT = Path(
    "experiments/tristan_fold1_replication"
)


def latest(prefix):
    paths = sorted(
        glob.glob(
            f"runs/{prefix}_*"
        )
    )

    if not paths:
        raise RuntimeError(
            f"No run found: {prefix}"
        )

    return Path(paths[-1])


def load_epochs(run):
    files = sorted(
        (
            run
            / "predictions"
        ).glob(
            "epoch_*_metrics.json"
        )
    )

    if not files:
        raise RuntimeError(
            f"No metrics in {run}"
        )

    rows = []

    for path in files:
        epoch = int(
            path.stem.split("_")[1]
        )

        rows.append(
            (
                epoch,
                json.loads(
                    path.read_text()
                ),
            )
        )

    return rows


def v(m, key):
    value = m.get(key)

    return (
        float("nan")
        if value is None
        else float(value)
    )


METRICS = [
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


baseline_run = latest(
    "fold1_baseline_10ep"
)

ltae_run = latest(
    "fold1_dual_ltae_s1s2_10ep"
)


baseline = load_epochs(
    baseline_run
)

ltae = load_epochs(
    ltae_run
)


baseline_final = (
    baseline[-1][1]
)

ltae_final = (
    ltae[-1][1]
)


baseline_best_epoch, baseline_best = max(
    baseline,
    key=lambda row: v(
        row[1],
        "operational/macro_f1_60d",
    ),
)


ltae_best_epoch, ltae_best = max(
    ltae,
    key=lambda row: v(
        row[1],
        "operational/macro_f1_60d",
    ),
)


csv_path = (
    OUT
    / "comparison_fold1.csv"
)


with csv_path.open(
    "w",
    newline="",
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "metric",
            "baseline_fold1",
            "dual_ltae_fold1",
            "ltae_minus_baseline",
        ]
    )

    for label, key in METRICS:

        a = v(
            baseline_final,
            key,
        )

        b = v(
            ltae_final,
            key,
        )

        writer.writerow(
            [
                label,
                a,
                b,
                b - a,
            ]
        )


print()
print(
    "============================================================"
)

print(
    " FOLD-1 FINAL-EPOCH COMPARISON"
)

print(
    "============================================================"
)

print(
    f"{'Metric':<15}"
    f"{'Baseline':>12}"
    f"{'Dual LTAE':>12}"
    f"{'Change':>11}"
)

print("-" * 50)


for label, key in METRICS:

    a = v(
        baseline_final,
        key,
    )

    b = v(
        ltae_final,
        key,
    )

    print(
        f"{label:<15}"
        f"{a:>12.3f}"
        f"{b:>12.3f}"
        f"{b-a:>+11.3f}"
    )


baseline_macro60 = v(
    baseline_best,
    "operational/macro_f1_60d",
)

ltae_macro60 = v(
    ltae_best,
    "operational/macro_f1_60d",
)


print()
print(
    "Best Macro60 checkpoints:"
)

print(
    f"  baseline: "
    f"epoch {baseline_best_epoch} "
    f"= {baseline_macro60:.3f}"
)

print(
    f"  Dual L-TAE: "
    f"epoch {ltae_best_epoch} "
    f"= {ltae_macro60:.3f}"
)

print(
    f"  improvement: "
    f"{ltae_macro60-baseline_macro60:+.3f}"
)


lines = [
    "# Fold-1 Replication",
    "",
    "Matched 10-epoch replication of the original StatisticalMLP baseline and Dual L-TAE S1+S2 on fold 1.",
    "",
    f"- Baseline: `{baseline_run}`",
    f"- Dual L-TAE: `{ltae_run}`",
    "",
    "## Final epoch",
    "",
    "| Metric | Baseline | Dual L-TAE | Change |",
    "|---|---:|---:|---:|",
]


for label, key in METRICS:

    a = v(
        baseline_final,
        key,
    )

    b = v(
        ltae_final,
        key,
    )

    lines.append(
        f"| {label} | "
        f"{a:.3f} | "
        f"{b:.3f} | "
        f"{b-a:+.3f} |"
    )


lines += [
    "",
    "## Best Macro F1 @ 60 days",
    "",
    (
        f"- Baseline: "
        f"{baseline_macro60:.3f} "
        f"(epoch {baseline_best_epoch})"
    ),
    (
        f"- Dual L-TAE: "
        f"{ltae_macro60:.3f} "
        f"(epoch {ltae_best_epoch})"
    ),
    (
        f"- Improvement: "
        f"{ltae_macro60-baseline_macro60:+.3f}"
    ),
]


(
    OUT
    / "RESULTS.md"
).write_text(
    "\n".join(lines)
    + "\n"
)


print()
print(
    "Saved:",
    csv_path,
)

print(
    "Saved:",
    OUT / "RESULTS.md",
)
PY


echo
echo "============================================================"
echo " FOLD-1 REPLICATION COMPLETE"
echo " TOTAL WALL TIME: $(( (SECONDS-START)/60 )) minutes"
echo "============================================================"
