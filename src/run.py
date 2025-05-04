import argparse
import numpy as np
import torch
from model.hjb_bsde import HJMDeepBSDE
from simulation.simulator import simulate_all
from helpers.plots import plot_all_diagnostics
from helpers.load_config import load_config

cfg = load_config()

T = cfg["T"]
N = cfg["N"]
dt = cfg["dt"]
gamma = cfg["gamma"]
device = cfg["device"]
y0 = cfg["y0"]
dim = cfg["dim"]
dim_w = cfg["dim_w"]
rho = cfg["rho"]
xi = cfg["xi"]
mu_P = cfg["mu_P"]
eta = cfg["eta"]
batch_size = cfg["batch_size"]

def main():
    parser = argparse.ArgumentParser(description="Train and simulate Deep BSDE model.")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save_path", type=str, default="model.pth", help="Path to save model")
    parser.add_argument("--n_paths", type=int, default=5, help="Number of simulation paths")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for simulation")
    parser.add_argument("--train", action="store_true", default=True, help="Train the model")
    parser.add_argument("--load_if_exists", action="store_true", default=True, help="Load existing model if it exists")
    parser.add_argument("--verbose", action="store_true", help="Print training progress")

    args = parser.parse_args()

    model = HJMDeepBSDE(y0, xi, batch_size)
    if args.train:
        if args.load_if_exists:
            try:
                model.load_state_dict(torch.load(args.save_path))
                print("Model loaded successfully.")
            except FileNotFoundError:
                print("No model found, starting training from scratch.")
        else:
            print("Starting training from scratch.")
        model.train_model(epochs=args.epochs, lr=args.lr, save_path=args.save_path)

    timesteps = np.linspace(0, T, N)
    results = simulate_all(model, n_paths=args.n_paths, batch_size=args.batch_size, y0=y0)
    plot_all_diagnostics(results, timesteps)

if __name__ == "__main__":
    main()
