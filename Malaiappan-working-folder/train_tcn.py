"""Custom training script that registers TemporalTCNModel locally without touching core files."""

import sys
import importlib.util
from pathlib import Path

# Ensure root repository is in path
repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from forest_disturbance.build import MODELS

# Dynamically load temporal_tcn.py by path
model_path = repo_root / "Malaiappan-working-folder" / "models" / "temporal_tcn.py"
spec = importlib.util.spec_from_file_location("temporal_tcn", model_path)
seq_module = importlib.util.module_from_spec(spec)
sys.modules["temporal_tcn"] = seq_module
spec.loader.exec_module(seq_module)

# Register the model class in-memory
MODELS["temporal_tcn"] = seq_module.TemporalTCNModel

from scripts.train import main as baseline_train_main

if __name__ == "__main__":
    print("Successfully registered custom temporal_tcn model in-memory via dynamic loader.")
    
    # Point to our working config file
    config_file = str(repo_root / "Malaiappan-working-folder" / "configs" / "temporal_tcn.yaml")
    
    # Build sys.argv mimicking: python scripts/train.py --config <path> [overrides...]
    new_argv = [sys.argv[0], "--config", config_file]
    
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg in ("--config-dir", "--config-name"):
            skip_next = True
            continue
        if not arg.startswith("--config"):
            new_argv.append(arg)
            
    sys.argv = new_argv
    print(f"Launching training with args: {sys.argv}")
    
    baseline_train_main()
