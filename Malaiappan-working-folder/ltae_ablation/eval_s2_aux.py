import torch
from pathlib import Path
from experiments.tristan_ltae_late_fusion.model import DualLTAEFusion
from forest_disturbance.build import build_datamodule, build_mapping
from forest_disturbance.config import load_config

def evaluate_s2_auxiliary(ckpt_path, config_path="configs/statistical_mlp.yaml"):
    print(f"Loading config from {config_path}")
    config = load_config(config_path, [])
    
    mapping = build_mapping(config)
    datamodule = build_datamodule(config, mapping)
    datamodule.setup("validate")
    val_loader = datamodule.val_dataloader()

    print(f"Loading model checkpoint from {ckpt_path}")
    model = DualLTAEFusion(num_classes=mapping.num_classes)
    
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            cleaned_state_dict[k[6:]] = v
        else:
            cleaned_state_dict[k] = v

    model.load_state_dict(cleaned_state_dict, strict=False)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print("Running evaluation on validation set using S2 Auxiliary Head...")
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            target_date = batch["target_date"].to(device)
            s2_inputs = {
                "recent": {
                    "values": batch["inputs"]["s2"]["recent"]["values"].to(device),
                    "dates": batch["inputs"]["s2"]["recent"]["dates"].to(device),
                    "mask": batch["inputs"]["s2"]["recent"]["mask"].to(device),
                },
                "yearly": [
                    {
                        "values": y["values"].to(device),
                        "dates": y["dates"].to(device),
                        "mask": y["mask"].to(device),
                    }
                    for y in batch["inputs"]["s2"].get("yearly", [])
                ]
            }
            
            s2_features = model._modality_features(
                model.s2_branch,
                s2_inputs,
                target_date,
                model.s2_dim,
            )
            
            logits = model.s2_aux_head(s2_features)
            preds = logits.argmax(dim=-1)
            
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(batch["target"].cpu().tolist())
            
    print(f"Evaluated {len(all_targets)} validation samples successfully.")

if __name__ == "__main__":
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/full_dual_ltae_s1s2_10ep/checkpoints/last.ckpt"
    evaluate_s2_auxiliary(ckpt)
