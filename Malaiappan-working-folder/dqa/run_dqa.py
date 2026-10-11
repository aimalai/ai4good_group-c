"""Data Quality Analysis (DQA) script for Forest Disturbance dataset."""

import os
import pandas as pd
import wandb
from pathlib import Path


def run_data_quality_analysis():
    data_root = Path(os.environ.get("DISFOR_DATA_ROOT", "/zfs/ai4good/datasets/forest_disturbance"))
    
    # Initialize wandb tracking for data quality audit
    wandb.init(
        project="forest-disturbance-exploration",
        name="data_quality_analysis",
        job_type="dqa"
    )
    
    print(f"Loading verified dataset metadata from: {data_root}")
    
    # Load primary parquet artifacts using exact schema columns
    samples_df = pd.read_parquet(data_root / "samples.parquet")
    labels_df = pd.read_parquet(data_root / "labels.parquet")
    splits_df = pd.read_parquet(data_root / "splits.parquet")
    zarr_frames_df = pd.read_parquet(data_root / "zarr_frames.parquet")
    
    # 1. Source / Dataset Composition Audit
    if "source" in samples_df.columns:
        source_counts = samples_df["source"].value_counts().reset_index()
        source_counts.columns = ["source", "sample_count"]
        wandb.log({"dqa/source_distribution": wandb.Table(dataframe=source_counts)})
    
    # 2. Label Disturbance Class Skew Audit
    if "label" in labels_df.columns:
        label_counts = labels_df["label"].value_counts().reset_index()
        label_counts.columns = ["label_code", "frequency"]
        wandb.log({"dqa/label_distribution": wandb.Table(dataframe=label_counts)})
        
    # 3. Event vs. Period Distribution Audit (`is_event`)
    if "is_event" in labels_df.columns:
        event_counts = labels_df["is_event"].value_counts().reset_index()
        event_counts.columns = ["is_event", "count"]
        wandb.log({"dqa/is_event_distribution": wandb.Table(dataframe=event_counts)})
    
    # 4. Fold Stratification Balance Audit
    if "fold_id" in splits_df.columns and "split" in splits_df.columns:
        fold_distribution = splits_df.groupby(["fold_id", "split"]).size().reset_index(name="count")
        wandb.log({"dqa/fold_splits": wandb.Table(dataframe=fold_distribution)})
    
    # 5. Sensor Availability / Frame Density Audit
    if "sensor" in zarr_frames_df.columns:
        sensor_counts = zarr_frames_df["sensor"].value_counts().reset_index()
        sensor_counts.columns = ["sensor", "frame_count"]
        wandb.log({"dqa/sensor_frame_counts": wandb.Table(dataframe=sensor_counts)})
    
    print("DQA audit complete. Metrics and tables successfully synced to Weights & Biases.")
    wandb.finish()


if __name__ == "__main__":
    run_data_quality_analysis()
