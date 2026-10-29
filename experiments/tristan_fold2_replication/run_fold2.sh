#!/usr/bin/env bash
set -euo pipefail

cd /zfs/ai4good/student/thein/ai4good_group-c

export DISFOR_DATA_ROOT=/zfs/ai4good/datasets/forest_disturbance
export UV_CACHE_DIR=/zfs/ai4good/student/thein/uv_cache

EXP=experiments/tristan_fold2_replication
START=$SECONDS

echo
echo "============================================================"
echo " FINAL ROBUSTNESS CHECK — FOLD 2"
echo
echo " 1) StatisticalMLP baseline — 10 epochs"
echo " 2) Dual L-TAE S1+S2 — 10 epochs"
echo " 3) save every L-TAE epoch"
echo " 4) select best validation Macro60 checkpoint"
echo "============================================================"
echo


echo
echo "============================================================"
echo " STEP 1/2 — FOLD-2 BASELINE"
echo "============================================================"
echo

uv run python scripts/train.py \
  --config configs/statistical_mlp.yaml \
  data.fold=2 \
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
  run_name=fold2_baseline_10ep


echo
echo "============================================================"
echo " STEP 2/2 — FOLD-2 DUAL L-TAE"
echo "============================================================"
echo

uv run python \
  "$EXP/train_ltae_save_all.py" \
  data.reader=pixel_cache \
  data.image_size=1 \
  data.sensors='[s1,s2]' \
  data.target_sensors='[s2]' \
  data.normalization=null \
  data.fold=2 \
  data.batch_size=16 \
  data.num_workers=8 \
  data.sampling_alpha=0.5 \
  optimizer.lr=0.0003 \
  optimizer.weight_decay=0.01 \
  trainer.max_epochs=10 \
  trainer.limit_train_batches=1.0 \
  trainer.precision=bf16-mixed \
  logger.kind=csv \
  run_name=fold2_dual_ltae_s1s2_10ep


python3 - <<'PY'
import csv
import glob
import json
import shutil
from pathlib import Path


OUT = Path(
    "experiments/tristan_fold2_replication"
)


def latest(prefix):
    paths = sorted(
        glob.glob(f"runs/{prefix}_*")
    )

    if not paths:
        raise RuntimeError(
            f"No run found: {prefix}"
        )

    return Path(paths[-1])


def epochs(run):
    files = sorted(
        (run / "predictions").glob(
            "epoch_*_metrics.json"
        )
    )

    if not files:
        raise RuntimeError(
            f"No metrics found in {run}"
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


def v(metrics, key):
    x = metrics.get(key)

    return (
        float("nan")
        if x is None
        else float(x)
    )


METRICS = [
    ("Binary 14d", "operational/binary_f1_14d"),
    ("Binary 30d", "operational/binary_f1_30d"),
    ("Binary 60d", "operational/binary_f1_60d"),
    ("Binary 365d", "operational/binary_f1_365d"),
    ("Macro 14d", "operational/macro_f1_14d"),
    ("Macro 30d", "operational/macro_f1_30d"),
    ("Macro 60d", "operational/macro_f1_60d"),
    ("Macro 365d", "operational/macro_f1_365d"),
]


baseline_run = latest(
    "fold2_baseline_10ep"
)

ltae_run = latest(
    "fold2_dual_ltae_s1s2_10ep"
)


baseline = epochs(
    baseline_run
)

ltae = epochs(
    ltae_run
)


baseline_final = baseline[-1][1]
ltae_final = ltae[-1][1]


baseline_best_epoch, baseline_best = max(
    baseline,
    key=lambda x: v(
        x[1],
        "operational/macro_f1_60d",
    ),
)


ltae_best_epoch, ltae_best = max(
    ltae,
    key=lambda x: v(
        x[1],
        "operational/macro_f1_60d",
    ),
)


# Preserve actual model weights for the
# validation-selected best epoch.
source_ckpt = (
    ltae_run
    / "checkpoints"
    / f"epoch-{ltae_best_epoch:02d}.ckpt"
)

best_ckpt = (
    ltae_run
    / "checkpoints"
    / "best_macro60.ckpt"
)


if source_ckpt.exists():
    shutil.copy2(
        source_ckpt,
        best_ckpt,
    )

    print(
        "Best checkpoint copied:",
        source_ckpt,
        "->",
        best_ckpt,
    )
else:
    print(
        "WARNING: expected checkpoint missing:",
        source_ckpt,
    )


csv_path = (
    OUT
    / "comparison_fold2.csv"
)


with csv_path.open(
    "w",
    newline="",
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "metric",
            "baseline_final",
            "ltae_final",
            "final_change",
            "baseline_at_best_macro60_checkpoint",
            "ltae_at_best_macro60_checkpoint",
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
                v(
                    baseline_best,
                    key,
                ),
                v(
                    ltae_best,
                    key,
                ),
            ]
        )


print()
print(
    "============================================================"
)

print(
    " FOLD-2 FINAL-EPOCH COMPARISON"
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


base60 = v(
    baseline_best,
    "operational/macro_f1_60d",
)

ltae60 = v(
    ltae_best,
    "operational/macro_f1_60d",
)


print()
print(
    "VALIDATION-SELECTED BEST Macro60:"
)

print(
    f"  baseline: "
    f"epoch {baseline_best_epoch} "
    f"= {base60:.3f}"
)

print(
    f"  Dual L-TAE: "
    f"epoch {ltae_best_epoch} "
    f"= {ltae60:.3f}"
)

print(
    f"  improvement: "
    f"{ltae60-base60:+.3f}"
)


lines = [
    "# Fold-2 Replication and Checkpoint Selection",
    "",
    "Matched 10-epoch fold-2 replication of StatisticalMLP and Dual L-TAE S1+S2.",
    "",
    "Every Dual L-TAE epoch was checkpointed. The validation checkpoint with the highest Macro F1 @ 60 days is preserved as `best_macro60.ckpt`.",
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
    "## Validation-selected Macro F1 @ 60 days",
    "",
    (
        f"- Baseline: {base60:.3f} "
        f"(epoch {baseline_best_epoch})"
    ),
    (
        f"- Dual L-TAE: {ltae60:.3f} "
        f"(epoch {ltae_best_epoch})"
    ),
    (
        f"- Difference: {ltae60-base60:+.3f}"
    ),
    "",
    "Checkpoint selection is performed on validation performance; these numbers should therefore be described as validation results, not held-out test results.",
]


(
    OUT
    / "RESULTS.md"
).write_text(
    "\n".join(lines) + "\n"
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

print(
    "Dual L-TAE run:",
    ltae_run,
)
PY


echo
echo "============================================================"
echo " FOLD-2 ROBUSTNESS CHECK COMPLETE"
echo " TOTAL WALL TIME: $(( (SECONDS-START)/60 )) minutes"
echo "============================================================"
