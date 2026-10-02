import os
import pandas as pd
import wandb
from forest_disturbance.viz import plot_sample, style

def main():
    # Initialize a new W&B run for the sample time-series chapter
    run = wandb.init(
        project="forest-disturbance-exploration", 
        name="sample-timeseries-deep-dive", 
        job_type="exploratory-analysis"
    )

    style.use()
    data_root = os.environ.get("DISFOR_DATA_ROOT", "/zfs/ai4good/datasets/forest_disturbance")
    print(f"Loading samples from: {data_root}")

    samples = pd.read_parquet(os.path.join(data_root, "samples.parquet"))
    
    # Pick 3 representative sample IDs
    sample_ids = samples.index[:3].tolist()
    print(f"Selected sample IDs for W&B logging: {sample_ids}")

    logged_images = {}
    for sample_id in sample_ids:
        try:
            print(f"Generating time series plot for sample {sample_id}...")
            fig = plot_sample(
                sample_id, 
                data_root, 
                patches=("s2", "s1"), 
                series=("NDVI", ("VV", "VH"))
            )
            output_path = f"sample_{sample_id}_timeseries.png"
            style.save(fig, output_path)
            
            # Log each sample image to W&B
            logged_images[f"sample_{sample_id}_timeseries"] = wandb.Image(output_path)
            print(f"Successfully generated and logged: {output_path}")
        except Exception as e:
            print(f"Skipping sample {sample_id} due to error: {e}")

    # Log all artifacts together to W&B dashboard
    if logged_images:
        wandb.log(logged_images)
        
    wandb.summary["analyzed_sample_timeseries_count"] = len(sample_ids)
    wandb.finish()
    print("\nSample time series analysis successfully logged to Weights & Biases!")

if __name__ == "__main__":
    main()
