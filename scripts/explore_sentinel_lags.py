import os
import pandas as pd
import matplotlib.pyplot as plt
import wandb

def main():
    # Initialize W&B for the 3rd chapter in our exploratory narrative
    run = wandb.init(
        project="forest-disturbance-exploration", 
        name="sentinel-lags-and-spatial-analysis", 
        job_type="exploratory-analysis"
    )

    data_root = '/zfs/ai4good/datasets/forest_disturbance'
    print(f"Loading dataset for Sentinel-2 lag & spatial analysis from: {data_root}")

    samples = pd.read_parquet(os.path.join(data_root, "samples.parquet"))
    labels = pd.read_parquet(os.path.join(data_root, "labels.parquet"))

    # Calculate temporal gap / evidence lag where applicable
    if 'start' in labels.columns and 'end_evidence' in labels.columns:
        labels['start_dt'] = pd.to_datetime(labels['start'])
        labels['evidence_dt'] = pd.to_datetime(labels['end_evidence'])
        labels['evidence_lag_days'] = (labels['evidence_dt'] - labels['start_dt']).dt.days
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # --- Panel 1: Evidence Lag Distribution (Highlighting the 3-4 day lag concept) ---
    if 'evidence_lag_days' in labels.columns:
        lags = labels['evidence_lag_days'].clip(lower=0, upper=60) # clip extreme outliers for clean viz
        axes[1].hist(lags, bins=30, color='#388e3c', edgecolor='black', linewidth=0.6)
        axes[1].set_title("Evidence Window Lag Distribution", fontsize=13, fontweight='bold', pad=12)
        axes[1].set_xlabel("Lag / Window Duration (Days)", fontsize=11)
        axes[1].set_ylabel("Frequency", fontsize=11)
    else:
        axes[1].text(0.5, 0.5, "Lag data unavailable", ha='center', va='center')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    # --- Panel 2: MGRS Tile Distribution (Sentinel-2 Grid Spread) ---
    if 'mgrs_tile' in samples.columns:
        tile_counts = samples['mgrs_tile'].value_counts().head(6)
        bars2 = axes[0].bar(tile_counts.index.astype(str), tile_counts.values, 
                            color='#1b5e20', width=0.5, edgecolor='black', linewidth=0.8)
        for bar in bars2:
            height = bar.get_height()
            axes[0].annotate(f'{height:,}', xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')
        axes[0].set_title("Top MGRS Tiles (Sentinel-2)", fontsize=13, fontweight='bold', pad=12)
        axes[0].set_ylabel("Sample Count", fontsize=11)
        axes[0].set_xlabel("MGRS Tile ID", fontsize=11)
        plt.setp(axes[0].get_xticklabels(), rotation=20, ha='right', fontsize=9)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    # --- Panel 3: Interpreter Confidence Levels ---
    if 'confidence' in samples.columns:
        conf_counts = samples['confidence'].value_counts()
        bars3 = axes[2].bar(conf_counts.index.astype(str), conf_counts.values, 
                            color='#7cb342', width=0.4, edgecolor='black', linewidth=0.8)
        for bar in bars3:
            height = bar.get_height()
            axes[2].annotate(f'{height:,}', xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')
        axes[2].set_title("Annotation Confidence Levels", fontsize=13, fontweight='bold', pad=12)
        axes[2].set_ylabel("Sample Count", fontsize=11)
        axes[2].set_xlabel("Confidence Category", fontsize=11)
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['right'].set_visible(False)
    axes[2].grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    chart_path = "sentinel_lags_and_spatial.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')

    # Log artifact to W&B
    wandb.log({"sentinel_lags_spatial_chart": wandb.Image(chart_path)})
    wandb.summary["total_mgrs_tiles"] = samples['mgrs_tile'].nunique() if 'mgrs_tile' in samples.columns else 0
    
    wandb.finish()
    print("\nSentinel-2 spatial and lag analysis successfully logged to Weights & Biases!")

if __name__ == "__main__":
    main()
