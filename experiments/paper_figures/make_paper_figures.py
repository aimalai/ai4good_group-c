from __future__ import annotations

import csv
import json
import os
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from forest_disturbance.build import build_mapping
from forest_disturbance.config import load_config
from forest_disturbance.data.labels import load_labels
from forest_disturbance.metrics.operational import evaluate_operational
from forest_disturbance.viz import plot_sample, style


ROOT = Path(".")
OUT = ROOT / "experiments" / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# EXACT RUNS USED IN THE CORRESPONDING RESULTS.md FILES
# ----------------------------------------------------------------------

RUNS = {
    "Baseline": ROOT / "runs/full_baseline_10ep_20261021-165735",
    "TESSERA S2": ROOT / "runs/full_tessera_s2_10ep_20261020-213828",
    "TESSERA S1+S2": ROOT / "runs/full_tessera_s1s2_10ep_20261021-051601",
    "Dual L-TAE S1+S2": ROOT / "runs/full_dual_ltae_s1s2_10ep_20261021-212830",
}

FOLD_RUNS = {
    0: {
        "Baseline": RUNS["Baseline"],
        "Dual L-TAE S1+S2": RUNS["Dual L-TAE S1+S2"],
    },
    1: {
        "Baseline": ROOT / "runs/fold1_baseline_10ep_20261028-124147",
        "Dual L-TAE S1+S2": ROOT / "runs/fold1_dual_ltae_s1s2_10ep_20261028-155121",
    },
    2: {
        "Baseline": ROOT / "runs/fold2_baseline_10ep_20261029-123554",
        "Dual L-TAE S1+S2": ROOT / "runs/fold2_dual_ltae_s1s2_10ep_20261029-154044",
    },
}

for name, run in RUNS.items():
    if not run.exists():
        raise FileNotFoundError(f"Missing {name} run: {run}")

for fold, models in FOLD_RUNS.items():
    for name, run in models.items():
        if not run.exists():
            raise FileNotFoundError(
                f"Missing fold {fold} {name} run: {run}"
            )


def load_epochs(run: Path):
    files = sorted(
        (run / "predictions").glob("epoch_*_metrics.json")
    )

    if len(files) != 10:
        raise RuntimeError(
            f"Expected 10 epoch metric files in {run}, found {len(files)}"
        )

    result = []

    for path in files:
        epoch = int(path.stem.split("_")[1])
        result.append((epoch, json.loads(path.read_text())))

    return result


def metric(metrics: dict, key: str) -> float:
    return float(metrics[key])


series = {
    name: load_epochs(run)
    for name, run in RUNS.items()
}


# ======================================================================
# FIGURE 1
# Training Convergence Curve: Macro F1 @ 60d across all 10 epochs
# ======================================================================

macro60_key = "operational/macro_f1_60d"

curve_rows = []

fig, ax = plt.subplots(figsize=(8.2, 5.0))

for name in RUNS:
    epochs = [epoch for epoch, _ in series[name]]
    values = [
        metric(metrics, macro60_key)
        for _, metrics in series[name]
    ]

    ax.plot(
        epochs,
        values,
        marker="o",
        linewidth=2,
        markersize=5,
        label=name,
    )

    for epoch, value in zip(epochs, values):
        curve_rows.append(
            {
                "epoch": epoch,
                "model": name,
                "macro_f1_60d": value,
            }
        )

ax.set_xlabel("Training epoch (0-indexed, matching run logs)")
ax.set_ylabel("Validation Macro F1 @ 60 days")
ax.set_title("Training Convergence: Macro F1 @ 60 Days")
ax.set_xticks(range(10))
ax.set_ylim(bottom=0)
ax.grid(axis="y", alpha=0.25)
ax.legend(frameon=False)
fig.tight_layout()

fig.savefig(
    OUT / "01_training_convergence_macro60.png",
    dpi=300,
    bbox_inches="tight",
)
fig.savefig(
    OUT / "01_training_convergence_macro60.pdf",
    bbox_inches="tight",
)
plt.close(fig)

with (OUT / "01_training_convergence_macro60.csv").open(
    "w", newline=""
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["epoch", "model", "macro_f1_60d"],
    )
    writer.writeheader()
    writer.writerows(curve_rows)


# ======================================================================
# FIGURE 2
# Multi-horizon operational performance
# Uses FINAL EPOCH values, exactly as comparison_4way.csv / RESULTS.md
# ======================================================================

horizons = [14, 30, 60, 365]

binary_keys = {
    h: f"operational/binary_f1_{h}d"
    for h in horizons
}

macro_keys = {
    h: f"operational/macro_f1_{h}d"
    for h in horizons
}

final_metrics = {
    name: rows[-1][1]
    for name, rows in series.items()
}

performance_rows = []

for name in RUNS:
    for horizon in horizons:
        performance_rows.append(
            {
                "model": name,
                "horizon_days": horizon,
                "metric": "Binary F1",
                "f1": metric(
                    final_metrics[name],
                    binary_keys[horizon],
                ),
            }
        )

        performance_rows.append(
            {
                "model": name,
                "horizon_days": horizon,
                "metric": "Macro F1",
                "f1": metric(
                    final_metrics[name],
                    macro_keys[horizon],
                ),
            }
        )

with (OUT / "02_operational_performance.csv").open(
    "w", newline=""
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "model",
            "horizon_days",
            "metric",
            "f1",
        ],
    )
    writer.writeheader()
    writer.writerows(performance_rows)


fig, axes = plt.subplots(
    1,
    2,
    figsize=(13.0, 5.0),
    sharey=True,
)

x = np.arange(len(horizons))
width = 0.19

for panel_ax, metric_name, keys in [
    (axes[0], "Binary F1 — Detection", binary_keys),
    (axes[1], "Macro F1 — Attribution", macro_keys),
]:
    for i, name in enumerate(RUNS):
        values = [
            metric(final_metrics[name], keys[h])
            for h in horizons
        ]

        panel_ax.bar(
            x + (i - 1.5) * width,
            values,
            width=width,
            label=name,
        )

    panel_ax.set_title(metric_name)
    panel_ax.set_xlabel("Evaluation horizon")
    panel_ax.set_xticks(x)
    panel_ax.set_xticklabels(
        [f"{h}d" for h in horizons]
    )
    panel_ax.set_ylim(0, 0.75)
    panel_ax.grid(axis="y", alpha=0.20)

axes[0].set_ylabel("Operational F1")
axes[1].legend(
    frameon=False,
    fontsize=8,
    loc="upper left",
)

fig.suptitle(
    "Multi-Horizon Operational Performance — Final Epoch"
)
fig.tight_layout()

fig.savefig(
    OUT / "02_operational_performance.png",
    dpi=300,
    bbox_inches="tight",
)
fig.savefig(
    OUT / "02_operational_performance.pdf",
    bbox_inches="tight",
)
plt.close(fig)


# ======================================================================
# FIGURE 4
# Multi-fold replication comparison:
# PEAK VALIDATION Macro F1 @60d for Baseline vs Dual L-TAE
# ======================================================================

fold_rows = []

for fold, models in FOLD_RUNS.items():
    for name, run in models.items():
        epochs = load_epochs(run)

        best_epoch, best_metrics = max(
            epochs,
            key=lambda row: metric(
                row[1],
                macro60_key,
            ),
        )

        fold_rows.append(
            {
                "fold": fold,
                "model": name,
                "best_epoch": best_epoch,
                "peak_macro_f1_60d": metric(
                    best_metrics,
                    macro60_key,
                ),
            }
        )

with (OUT / "04_multifold_peak_macro60.csv").open(
    "w", newline=""
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "fold",
            "model",
            "best_epoch",
            "peak_macro_f1_60d",
        ],
    )
    writer.writeheader()
    writer.writerows(fold_rows)


fig, ax = plt.subplots(figsize=(7.8, 5.0))

folds = [0, 1, 2]
x = np.arange(len(folds))
width = 0.34

for i, name in enumerate(
    ["Baseline", "Dual L-TAE S1+S2"]
):
    values = []

    for fold in folds:
        row = next(
            r
            for r in fold_rows
            if r["fold"] == fold
            and r["model"] == name
        )
        values.append(row["peak_macro_f1_60d"])

    bars = ax.bar(
        x + (i - 0.5) * width,
        values,
        width,
        label=name,
    )

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.008,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

ax.set_xticks(x)
ax.set_xticklabels(
    [f"Fold {fold}" for fold in folds]
)
ax.set_ylabel("Peak validation Macro F1 @ 60 days")
ax.set_title(
    "Multi-Fold Replication: Baseline vs Dual L-TAE"
)
ax.set_ylim(0, 0.55)
ax.grid(axis="y", alpha=0.20)
ax.legend(frameon=False)
fig.tight_layout()

fig.savefig(
    OUT / "04_multifold_peak_macro60.png",
    dpi=300,
    bbox_inches="tight",
)
fig.savefig(
    OUT / "04_multifold_peak_macro60.pdf",
    bbox_inches="tight",
)
plt.close(fig)


# ======================================================================
# FIGURE 3
# Real sample timeline from final Dual L-TAE fold-0 run.
#
# epoch 009 is BOTH:
# - final epoch
# - best Macro60 epoch of this run
#
# We automatically choose a real validation event with a correctly matched
# operational class alert within 60 days.
# ======================================================================

data_root = os.environ.get(
    "DISFOR_DATA_ROOT",
    "/zfs/ai4good/datasets/forest_disturbance",
)

data_root = Path(data_root)

prediction_path = (
    RUNS["Dual L-TAE S1+S2"]
    / "predictions"
    / "epoch_009.parquet"
)

if not prediction_path.exists():
    raise FileNotFoundError(prediction_path)

predictions = pl.read_parquet(prediction_path)

config = load_config(
    "configs/statistical_mlp.yaml",
    [f"data.root={data_root}"],
)

mapping = build_mapping(config)
labels = load_labels(data_root / "labels.parquet")
frames = pl.read_parquet(
    data_root / "zarr_frames.parquet"
)

op = dict(
    config["evaluation"]["operational"]
)

result60 = evaluate_operational(
    predictions,
    labels=labels,
    timeline=frames,
    mapping=mapping,
    t_after_days=60,
    **op,
)

matched = [
    alert
    for alert in result60["alerts"]
    if alert["matched_event_id"] is not None
    and alert["lag_days"] is not None
]

# Prefer an intuitive alert after the annotated start and within 60 days.
positive = [
    a
    for a in matched
    if 0 <= int(a["lag_days"]) <= 60
]

candidates = positive if positive else matched

if not candidates:
    raise RuntimeError(
        "Could not find a correctly matched Dual L-TAE alert."
    )

# Pick the fastest correct alert for a clean qualitative example.
selected = sorted(
    candidates,
    key=lambda a: (
        abs(int(a["lag_days"])),
        int(a["sample_id"]),
    ),
)[0]

sample_id = int(selected["sample_id"])

with (
    OUT / "03_selected_sample_alert.csv"
).open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(selected.keys()),
    )
    writer.writeheader()
    writer.writerow(selected)

(OUT / "03_selected_sample.txt").write_text(
    f"{sample_id}\n"
)

# Get the matching annotation so we can also make a useful zoomed panel.
event_id = selected["matched_event_id"]
period_idx = int(str(event_id).split(":")[-1])

event_row = (
    labels
    .filter(
        (pl.col("sample_id") == sample_id)
        & (pl.col("period_idx") == period_idx)
    )
    .row(0, named=True)
)

zoom_start = event_row["start"] - timedelta(days=365)
zoom_end = event_row["end_evidence"] + timedelta(days=365)

style.use()

common_kwargs = dict(
    root=data_root,
    patches=("s2", "s1"),
    n_patches=5,
    crop=96,
    series=(
        ("B04", "B08", "B11"),
        ("VV", "VH"),
    ),
    predictions=predictions,
    alerts=result60["alerts"],
    threshold=0.5,
)

figure = plot_sample(
    sample_id,
    **common_kwargs,
)

style.save(
    figure,
    OUT / "03_dual_ltae_sample_timeline_full.png",
    dpi=220,
)
plt.close(figure)

figure = plot_sample(
    sample_id,
    span=(zoom_start, zoom_end),
    **common_kwargs,
)

style.save(
    figure,
    OUT / "03_dual_ltae_sample_timeline_zoom.png",
    dpi=220,
)
plt.close(figure)


# ======================================================================
# README / PROVENANCE
# ======================================================================

readme = f"""# Paper figures

These figures are generated directly from the final model runs referenced in
the TESSERA and Dual L-TAE RESULTS.md files.

## Run provenance

- Baseline: `{RUNS["Baseline"]}`
- TESSERA S2: `{RUNS["TESSERA S2"]}`
- TESSERA S1+S2: `{RUNS["TESSERA S1+S2"]}`
- Dual L-TAE S1+S2: `{RUNS["Dual L-TAE S1+S2"]}`

## Figures

1. `01_training_convergence_macro60`
   - Validation operational Macro F1 @ 60d over all 10 epochs.
   - Four models, fold 0.

2. `02_operational_performance`
   - Final-epoch operational Binary and Macro F1.
   - Horizons: 14d, 30d, 60d, 365d.
   - Values correspond to the four-way RESULTS.md comparison.

3. `03_dual_ltae_sample_timeline_zoom.png`
   - Real validation sample from Dual L-TAE fold-0 epoch 009 predictions.
   - Epoch 009 is both the final and best Macro60 epoch for this run.
   - Selected sample: {sample_id}
   - Predicted class: {selected["predicted_class"]}
   - True class: {selected["true_class"]}
   - Alert lag: {selected["lag_days"]} days
   - Triangle marker(s) in the probability panel are operational CUSUM alerts.
   - The matching shown here is evaluated at the 60-day horizon.

4. `04_multifold_peak_macro60`
   - Peak validation Macro F1 @ 60d for Baseline vs Dual L-TAE.
   - Folds 0, 1 and 2.

Important: these are validation-fold results, not a separate held-out test set.
"""

(OUT / "README.md").write_text(readme)

print()
print("=" * 70)
print("PAPER FIGURES COMPLETE")
print("=" * 70)
print()
print("Selected qualitative sample:")
print(" sample_id       =", sample_id)
print(" true class      =", selected["true_class"])
print(" predicted class =", selected["predicted_class"])
print(" alert lag       =", selected["lag_days"], "days")
print()
print("Files:")
for path in sorted(OUT.iterdir()):
    if path.is_file():
        print(" ", path)
