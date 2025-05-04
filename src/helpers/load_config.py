import json
import torch

def load_config(path="config.json"):
    with open(path, "r") as f:
        cfg = json.load(f)

    # Derived values
    cfg["dt"] = cfg["T"] / cfg["N"]
    cfg["device"] = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    cfg["y0"] = torch.tensor([cfg["initial_state"]], device=cfg["device"])
    cfg["c_prod"] = lambda xi: cfg["c_prod_base"] + 0.0 * xi  # keep flexible

    return cfg
