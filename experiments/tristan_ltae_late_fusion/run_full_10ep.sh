#!/usr/bin/env bash
set -euo pipefail

cd /zfs/ai4good/student/thein/ai4good_group-c

export DISFOR_DATA_ROOT=/zfs/ai4good/datasets/forest_disturbance
export UV_CACHE_DIR=/zfs/ai4good/student/thein/uv_cache

EXP_DIR=experiments/tristan_ltae_late_fusion
mkdir -p "$EXP_DIR"

START_ALL=$SECONDS

echo
echo "============================================================"
echo " DUAL L-TAE S1+S2 -- FULL 10-EPOCH MATCHED RUN"
echo " fully trainable from scratch"
echo " NO TESSERA / NO FOUNDATION MODEL"
echo
echo " fold 0"
echo " batch 16"
echo " sampling alpha 0.5"
echo " S2 target dates only"
echo " same operational evaluator"
echo "============================================================"
echo

uv run python \
  "$EXP_DIR/train.py" \
  data.reader=pixel_cache \
  data.image_size=1 \
  data.sensors='[s1,s2]' \
  data.target_sensors='[s2]' \
  data.normalization=null \
  data.fold=0 \
  data.batch_size=16 \
  data.num_workers=8 \
  data.sampling_alpha=0.5 \
  optimizer.lr=0.0003 \
  optimizer.weight_decay=0.01 \
  trainer.max_epochs=10 \
  trainer.limit_train_batches=1.0 \
  trainer.precision=bf16-mixed \
  logger.kind=csv \
  run_name=full_dual_ltae_s1s2_10ep

python3 - <<'PY'
import csv
import glob
import json
from pathlib import Path


EXP = Path(
    "experiments/"
    "tristan_ltae_late_fusion"
)


def latest(prefix):

    paths = sorted(
        glob.glob(
            f"runs/{prefix}_*"
        )
    )

    if not paths:
        raise RuntimeError(
            f"No run found "
            f"for {prefix}"
        )

    return Path(
        paths[-1]
    )


def epoch_metrics(run):

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
            f"No metrics found "
            f"in {run}"
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


def val(m, key):

    x = m.get(key)

    return (
        float("nan")
        if x is None
        else float(x)
    )


runs = {
    "baseline":
        latest(
            "full_baseline_10ep"
        ),

    "tessera_s2":
        latest(
            "full_tessera_s2_10ep"
        ),

    "tessera_s1s2":
        latest(
            "full_tessera_s1s2_10ep"
        ),

    "dual_ltae_s1s2":
        latest(
            "full_dual_ltae_s1s2_10ep"
        ),
}


series = {
    name:
        epoch_metrics(path)

    for name, path
    in runs.items()
}


metrics = [
    (
        "Binary 14d",
        "operational/"
        "binary_f1_14d",
    ),

    (
        "Binary 30d",
        "operational/"
        "binary_f1_30d",
    ),

    (
        "Binary 60d",
        "operational/"
        "binary_f1_60d",
    ),

    (
        "Binary 365d",
        "operational/"
        "binary_f1_365d",
    ),

    (
        "Macro 14d",
        "operational/"
        "macro_f1_14d",
    ),

    (
        "Macro 30d",
        "operational/"
        "macro_f1_30d",
    ),

    (
        "Macro 60d",
        "operational/"
        "macro_f1_60d",
    ),

    (
        "Macro 365d",
        "operational/"
        "macro_f1_365d",
    ),
]


final = {
    name: rows[-1][1]

    for name, rows
    in series.items()
}


csv_path = (
    EXP
    / "comparison_4way.csv"
)


with csv_path.open(
    "w",
    newline="",
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "metric",
            *runs.keys(),

            "ltae_minus_baseline",

            "ltae_minus_"
            "tessera_fusion",
        ]
    )

    for label, key in metrics:

        values = [
            val(
                final[name],
                key,
            )
            for name in runs
        ]

        writer.writerow(
            [
                label,
                *values,

                values[3]
                - values[0],

                values[3]
                - values[2],
            ]
        )


best = {
    name:
        max(
            rows,

            key=lambda row:
                val(
                    row[1],
                    "operational/"
                    "macro_f1_60d",
                ),
        )

    for name, rows
    in series.items()
}


lines = [
    "# Dual L-TAE S1+S2 "
    "— matched fold-0 result",

    "",

    "The new model is fully "
    "trainable from scratch and "
    "does not use TESSERA or "
    "another foundation model.",

    "",

    "## Final epoch",

    "",

    "| Metric | Baseline | "
    "TESSERA S2 | "
    "TESSERA S1+S2 | "
    "Dual L-TAE S1+S2 |",

    "|---|---:|---:|---:|---:|",
]


for label, key in metrics:

    lines.append(
        f"| {label} | "
        f"{val(final['baseline'], key):.3f} | "
        f"{val(final['tessera_s2'], key):.3f} | "
        f"{val(final['tessera_s1s2'], key):.3f} | "
        f"**{val(final['dual_ltae_s1s2'], key):.3f}** |"
    )


lines += [
    "",
    "## Best epoch by "
    "Macro F1 @ 60 days",
    "",
]


for name, (
    epoch,
    m,
) in best.items():

    lines.append(
        f"- **{name}**: "
        f"epoch {epoch}, "
        f"Macro60="
        f"{val(m, 'operational/macro_f1_60d'):.3f}, "
        f"Binary60="
        f"{val(m, 'operational/binary_f1_60d'):.3f}"
    )


lines += [
    "",
    "## Run folders",
    "",
]


for name, path in runs.items():

    lines.append(
        f"- **{name}**: "
        f"`{path}`"
    )


(
    EXP
    / "RESULTS.md"
).write_text(
    "\n".join(lines)
    + "\n"
)


print()
print(
    "============================================================"
)
print(
    " FINAL 4-WAY COMPARISON"
)
print(
    "============================================================"
)

print(
    f"{'Metric':<15}"
    f"{'Baseline':>10}"
    f"{'TESS S2':>10}"
    f"{'TESS Fuse':>11}"
    f"{'Dual LTAE':>11}"
    f"{'vs best':>10}"
)

print("-" * 67)


for label, key in metrics:

    b = val(
        final["baseline"],
        key,
    )

    s2 = val(
        final["tessera_s2"],
        key,
    )

    tf = val(
        final["tessera_s1s2"],
        key,
    )

    lt = val(
        final[
            "dual_ltae_s1s2"
        ],
        key,
    )

    print(
        f"{label:<15}"
        f"{b:>10.3f}"
        f"{s2:>10.3f}"
        f"{tf:>11.3f}"
        f"{lt:>11.3f}"
        f"{lt-tf:>+10.3f}"
    )


print()
print(
    "Best Macro60 checkpoints:"
)


for name, (
    epoch,
    m,
) in best.items():

    print(
        f"  {name:<18} "
        f"epoch {epoch}: "
        f"{val(m, 'operational/macro_f1_60d'):.3f}"
    )


print()
print(
    "Saved:",
    csv_path,
)

print(
    "Saved:",
    EXP / "RESULTS.md",
)
PY

echo
echo "TOTAL WALL TIME: $(( (SECONDS-START_ALL)/60 )) minutes"
echo "DUAL L-TAE EXPERIMENT COMPLETE"
