"""Custom training script that registers SequentialFusionModel locally without touching core files."""

import sys
import importlib.util
from pathlib import Path

# Ensure root repository is in path
repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from forest_disturbance.build import MODELS

# Dynamically load sequential_fusion.py by path to avoid hyphen syntax errors
model_path = repo_root / "Malaiappan-working-folder" / "models" / "sequential_fusion.py"
spec = importlib.util.spec_from_file_location("sequential_fusion", model_path)
seq_module = importlib.util.module_from_spec(spec)
sys.modules["sequential_fusion"] = seq_module
spec.loader.exec_module(seq_module)

# Register the model class in-memory
MODELS["sequential_fusion"] = seq_module.SequentialFusionModel

from scripts.train import main as baseline_train_main

if __name__ == "__main__":
    print("Successfully registered custom sequential_fusion model in-memory via dynamic loader.")
    
    # Point to our working config file using the --config argument format scripts/train.py expects
    config_file = str(repo_root / "Malaiappan-working-folder" / "configs" / "sequential_fusion.yaml")
    
    # Build sys.argv mimicking: python scripts/train.py --config <path> [overrides...]
    new_argv = [sys.argv[0], "--config", config_file]
    
    # Forward any command-line overrides, filtering out any old hydra flags if passed
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
