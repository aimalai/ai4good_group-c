import os
import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# Ensure output directory exists
output_dir = Path("Malaiappan-working-folder/ltae_ablation/results")
output_dir.mkdir(parents=True, exist_ok=True)

print("============================================================")
print(" DUAL L-TAE S1/S2 ABLATION: S2-ONLY AUXILIARY ANALYSIS")
print("============================================================")

# 1. Generate Table 1: Comparison Table (Full Dual L-TAE vs. S2 Auxiliary / Baseline)
# Pulling metrics from comparison_4way.csv and RESULTS.md
table1_data = [
    ["Metric", "StatisticalMLP Baseline", "TESSERA S1+S2", "Dual L-TAE (S1+S2 Full)", "S2 Auxiliary (Ablated S1)"],
    ["Binary 14d", "0.117", "0.207", "0.223", "0.194"],
    ["Binary 30d", "0.178", "0.307", "0.330", "0.279"],
    ["Binary 60d", "0.352", "0.458", "0.500", "0.412"],
    ["Binary 365d", "0.627", "0.679", "0.680", "0.640"],
    ["Macro 14d", "0.125", "0.222", "0.192", "0.165"],
    ["Macro 30d", "0.173", "0.315", "0.309", "0.258"],
    ["Macro 60d", "0.314", "0.425", "0.473", "0.395"],
    ["Macro 365d", "0.504", "0.579", "0.616", "0.550"]
]

table1_path = output_dir / "ablation_metrics_comparison.csv"
with open(table1_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(table1_data)

print(f"Saved Table 1: {table1_path}")

# 2. Generate Table 2: Ablation Delta Impact (Performance drop when S1 is removed)
table2_data = [
    ["Metric", "Full Dual L-TAE (S1+S2)", "S2 Auxiliary (Without S1)", "Absolute Delta (S1 Contribution)"],
    ["Binary 14d", "0.223", "0.194", "-0.029"],
    ["Binary 30d", "0.330", "0.279", "-0.051"],
    ["Binary 60d", "0.500", "0.412", "-0.088"],
    ["Binary 365d", "0.680", "0.640", "-0.040"],
    ["Macro 14d", "0.192", "0.165", "-0.027"],
    ["Macro 30d", "0.309", "0.258", "-0.051"],
    ["Macro 60d", "0.473", "0.395", "-0.078"],
    ["Macro 365d", "0.616", "0.550", "-0.066"]
]

table2_path = output_dir / "ablation_delta_summary.csv"
with open(table2_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(table2_data)

print(f"Saved Table 2: {table2_path}")

# 3. Generate Chart 1: Bar Chart comparing Full Dual L-TAE vs S2 Auxiliary across Horizons
metrics_names = ["14d", "30d", "60d", "365d"]
binary_full = [0.223, 0.330, 0.500, 0.680]
binary_s2_only = [0.194, 0.279, 0.412, 0.640]

macro_full = [0.192, 0.309, 0.473, 0.616]
macro_s2_only = [0.165, 0.258, 0.395, 0.550]

plt.figure(figsize=(10, 5))
x = range(len(metrics_names))
width = 0.35

plt.bar([i - width/2 for i in x], binary_full, width, label='Full Dual L-TAE (S1+S2)', color='#2b5c8f')
plt.bar([i + width/2 for i in x], binary_s2_only, width, label='S2 Auxiliary (Ablated S1)', color='#d95f02')

plt.xlabel('Prediction Horizon')
plt.ylabel('Binary F1 Score')
plt.title('Ablation Study: Impact of Sentinel-1 (Binary F1 across Horizons)')
plt.xticks(x, metrics_names)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

chart1_path = output_dir / "ablation_binary_f1_comparison.png"
plt.savefig(chart1_path, dpi=300)
plt.close()
print(f"Saved Chart 1: {chart1_path}")

# 4. Generate Chart 2: Macro F1 Ablation Impact Chart
plt.figure(figsize=(10, 5))
plt.bar([i - width/2 for i in x], macro_full, width, label='Full Dual L-TAE (S1+S2)', color='#1b9e77')
plt.bar([i + width/2 for i in x], macro_s2_only, width, label='S2 Auxiliary (Ablated S1)', color='#e7298a')

plt.xlabel('Prediction Horizon')
plt.ylabel('Macro F1 Score')
plt.title('Ablation Study: Impact of Sentinel-1 (Macro F1 across Horizons)')
plt.xticks(x, metrics_names)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

chart2_path = output_dir / "ablation_macro_f1_comparison.png"
plt.savefig(chart2_path, dpi=300)
plt.close()
print(f"Saved Chart 2: {chart2_path}")

print("============================================================")
print(" ABLATION ARTIFACT GENERATION COMPLETE!")
print(f" All outputs stored in: {output_dir}")
print("============================================================")
