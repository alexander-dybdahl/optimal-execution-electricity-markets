import json

def load_config(path):
    with open(path, "r") as f:
        cfg = json.load(f)
    return cfg

def load_dynamics_config(path):
    cfg = load_config(path)
    cfg["dt"] = cfg["T"] / cfg["N"]
    return cfg

def load_train_config(path):
    cfg = load_config(path)
    cfg["architecture"] = cfg["architecture"].lower()
    return cfg