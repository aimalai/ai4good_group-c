"""Custom training script for Temporal Attention Network."""

import sys
import importlib.util
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from forest_disturbance.build import MODELS

model_path = repo_root / "Malaiappan-working-folder" / "models" / "temporal_attention.py"
spec = importlib.util.spec_from_file_location("temporal_attention", model_path)
seq_module = importlib.util.module_from_spec(spec)
sys.modules["temporal_attention"] = seq_module
spec.loader.exec_module(seq_module)

MODELS["temporal_attention"] = seq_module.TemporalAttentionModel

from scripts.train import main as baseline_train_main

if __name__ == "__main__":
    config_file = str(repo_root / "Malaiappan-working-folder" / "configs" / "temporal_attention.yaml")
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
    baseline_train_main()
