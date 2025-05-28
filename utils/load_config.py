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

    if torch.cuda.is_available() and cfg["device"] != "cuda":
        print("Warning: CUDA is available but the config file does not set device to cuda.") 

    return cfg