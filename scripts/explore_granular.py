import os
import pandas as pd
import matplotlib.pyplot as plt
import wandb

def main():
    # Initialize a new W&B run for the granular deep-dive layer of our story
    run = wandb.init(
        project="forest-disturbance-exploration", 
        name="deep-dive-granular-stats", 
        job_type="granular-analysis"
    )

    data_root = os.environ.get("DISFOR_DATA_ROOT", "/zfs/ai4good/datasets/forest_disturbance")
    print(f"Inspecting dataset at: {data_root}")

    # Load metadata parquet files
    samples = pd.read_parquet(os.path.join(data_root, "samples.parquet"))
    labels = pd.read_parquet(os.path.join(data_root, "labels.parquet"))
    splits = pd.read_parquet(os.path.join(data_root, "splits.parquet"))
    center_pixels = pd.read_parquet(os.path.join(data_root, "center_pixels.parquet"))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # --- Panel 1: True Split Fold Distribution ---
    fold_col = 'fold_id' if 'fold_id' in splits.columns else 'fold'
    fold_counts = splits[fold_col].value_counts().sort_index()
    bars1 = axes[0].bar([f"Fold {int(f)}" for f in fold_counts.index], fold_counts.values, 
                        color='#2e7d32', width=0.5, edgecolor='black', linewidth=0.8)
    for bar in bars1:
        height = bar.get_height()
        axes[0].annotate(f'{height:,}', xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[0].set_title("True CV Folds Distribution", fontsize=13, fontweight='bold', pad=12)
    axes[0].set_ylabel("Sample Count", fontsize=11)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    # --- Panel 2: Spatial Distribution / Center Pixels ---
    if 'lat' in center_pixels.columns and 'lon' in center_pixels.columns:
        axes[1].scatter(center_pixels['lon'], center_pixels['lat'], alpha=0.3, s=2, color='#388e3c')
        axes[1].set_title("Spatial Distribution of Centers", fontsize=13, fontweight='bold', pad=12)
        axes[1].set_xlabel("Longitude", fontsize=11)
        axes[1].set_ylabel("Latitude", fontsize=11)
    else:
        axes[1].plot(center_pixels.index[:1000], color='#388e3c', lw=1.5)
        axes[1].set_title("Center Pixels Sequence", fontsize=13, fontweight='bold', pad=12)
        axes[1].set_ylabel("Sample Index", fontsize=11)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # --- Panel 3: Granular Label Breakdown with Readable Names ---
    top_labels = labels['label'].value_counts().head(5)
    
    # Consistent mapping with descriptive text for narrative clarity
    label_mapping = {
        110: "110: Minor Loss",
        120: "120: Mod. Deg.",
        211: "211: Severe Event",
        212: "212: Clear-cut/Abrupt",
        222: "222: Regrowth",
        243: "243: Mixed/Complex"
    }
    x_labels = [label_mapping.get(code, f"Code {code}") for code in top_labels.index]

    bars3 = axes[2].bar(x_labels, top_labels.values, 
                        color='#689f38', width=0.5, edgecolor='black', linewidth=0.8)
    for bar in bars3:
        height = bar.get_height()
        axes[2].annotate(f'{height:,}', xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[2].set_title("Top 5 Detailed Categories", fontsize=13, fontweight='bold', pad=12)
    axes[2].set_ylabel("Frequency", fontsize=11)
    axes[2].set_xlabel("Disturbance Categories", fontsize=11)
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['right'].set_visible(False)
    axes[2].grid(axis='y', linestyle='--', alpha=0.5)
    plt.setp(axes[2].get_xticklabels(), rotation=25, ha='right', fontsize=9)

    plt.tight_layout()
    chart_path = "granular_deep_dive.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')

    # Log visual artifact to W&B
    wandb.log({"granular_deep_dive_charts": wandb.Image(chart_path)})
    wandb.summary["analyzed_samples_count"] = len(samples)
    wandb.finish()
    print("\nGranular deep-dive script executed and logged to W&B successfully!")

if __name__ == "__main__":
    main()
