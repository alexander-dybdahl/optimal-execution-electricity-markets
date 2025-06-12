import json
import torch

def load_model_config(path):
    with open(path, "r") as f:
        cfg = json.load(f)
    cfg["dt"] = cfg["T"] / cfg["N"]
    return cfg

def load_run_config(path):
    with open(path, "r") as f:
        cfg = json.load(f)
    return cfg

def load_nn_config(path):
    """Load architecture-specific neural network configuration."""
    with open(path, "r") as f:
        cfg = json.load(f)
    return cfg

def load_combined_config(path):
    """Load run config and merge with architecture-specific nn config."""
    run_cfg = load_run_config(path)
    
    # Load NN config if specified
    if "nn_config_path" in run_cfg:
        nn_cfg = load_nn_config(run_cfg["nn_config_path"])
        # Merge nn config into run config
        run_cfg.update(nn_cfg)
    
    return run_cfg