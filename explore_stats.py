import os
import pandas as pd
import matplotlib.pyplot as plt
import wandb

def main():
    run = wandb.init(
        project="forest-disturbance-exploration", 
        name="dataset-overview-macro-complete", 
        job_type="exploratory-analysis"
    )

    data_root = os.environ.get("DISFOR_DATA_ROOT", "/zfs/ai4good/datasets/forest_disturbance")
    print(f"Loading dataset from: {data_root}")

    samples = pd.read_parquet(os.path.join(data_root, "samples.parquet"))
    labels = pd.read_parquet(os.path.join(data_root, "labels.parquet"))
    splits = pd.read_parquet(os.path.join(data_root, "splits.parquet"))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # --- Panel 1: Dataset Volume & Scale (Theme: Deep Forest Green) ---
    categories = ['Sample Locations', 'Annotated Periods']
    values = [len(samples), len(labels)]
    bars1 = axes[0].bar(categories, values, color=['#1b5e20', '#2e7d32'], width=0.4, edgecolor='black', linewidth=0.8)
    for bar in bars1:
        height = bar.get_height()
        axes[0].annotate(f'{height:,}', xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=11, fontweight='bold')
    axes[0].set_title("Dataset Core Scale", fontsize=13, fontweight='bold', pad=12)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    # --- Panel 2: True Cross-Validation Folds Distribution (Theme: Slate Teal - Matches Structural Rigor) ---
    fold_col = 'fold_id' if 'fold_id' in splits.columns else 'fold'
    fold_counts = splits[fold_col].value_counts().sort_index()
    bars2 = axes[1].bar([f"Fold {int(f)}" for f in fold_counts.index], fold_counts.values, 
                        color='#00838f', width=0.5, edgecolor='black', linewidth=0.8)
    for bar in bars2:
        height = bar.get_height()
        axes[1].annotate(f'{height:,}', xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[1].set_title("CV Folds Distribution", fontsize=13, fontweight='bold', pad=12)
    axes[1].set_ylabel("Samples per Fold", fontsize=11)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    # --- Panel 3: ALL Disturbance Classes with Comprehensive Mapping ---
    all_labels = labels['label'].value_counts().sort_index()
    
    # Comprehensive label dictionary mapping raw codes to readable descriptions
    label_mapping = {
        110: "110: Minor Loss",
        120: "120: Mod. Deg.",
        121: "121: Tree Cover (Post-Cut)",
        122: "122: Degraded State",
        123: "123: Transitional",
        211: "211: Severe Event",
        212: "212: Clear-cut/Abrupt",
        222: "222: Regrowth",
        243: "243: Mixed/Complex"
    }
    x_labels = [label_mapping.get(code, f"Code {code}") for code in all_labels.index]

    # Generate a smooth graded color scale matching the length of classes
    bar_colors = plt.cm.Greens(pd.Series(range(len(all_labels))) / max(1, len(all_labels) - 1) * 0.6 + 0.3)

    bars3 = axes[2].bar(x_labels, all_labels.values, 
                        color=bar_colors, width=0.5, edgecolor='black', linewidth=0.8)
    for bar in bars3:
        height = bar.get_height()
        axes[2].annotate(f'{height:,}', xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
    axes[2].set_title("Complete Disturbance Class Distribution", fontsize=13, fontweight='bold', pad=12)
    axes[2].set_ylabel("Frequency", fontsize=11)
    axes[2].set_xlabel("Disturbance Categories", fontsize=11)
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['right'].set_visible(False)
    axes[2].grid(axis='y', linestyle='--', alpha=0.5)
    
    # Rotate x-axis text sharply so all labels fit cleanly without crowding
    plt.setp(axes[2].get_xticklabels(), rotation=40, ha='right', fontsize=8)

    plt.tight_layout()
    chart_path = "dataset_overview_macro_complete.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')

    wandb.log({"macro_overview_complete": wandb.Image(chart_path)})
    wandb.summary["total_samples"] = len(samples)
    wandb.summary["total_labels"] = len(labels)
    wandb.summary["total_classes"] = len(all_labels)
    
    wandb.finish()
    print(f"\nComplete macro stats with all {len(all_labels)} classes logged to Weights & Biases!")

if __name__ == "__main__":
    main()
