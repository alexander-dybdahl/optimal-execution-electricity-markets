import json
import torch

def load_model_config(path):
    with open(path, "r") as f:
        cfg = json.load(f)
    cfg["dt"] = cfg["T"] / cfg["N"]
    return cfg

def load_run_config(path="config/run_config.json"):
    with open(path, "r") as f:
        cfg = json.load(f)
    cfg["architecture"] = cfg["architecture"].lower()
    return cfg