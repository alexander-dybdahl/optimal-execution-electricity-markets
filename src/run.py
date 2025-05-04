import argparse
import numpy as np
from model.hjb_bsde import HJMDeepBSDE
from simulation.simulator import simulate_all
from helpers.plots import plot_all_diagnostics
from config import y0, xi, batch_size, T, N

def main():
    parser = argparse.ArgumentParser(description="Train and simulate Deep BSDE model.")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save_path", type=str, default="model.pth", help="Path to save model")
    parser.add_argument("--n_paths", type=int, default=5, help="Number of simulation paths")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for simulation")

    args = parser.parse_args()

    model = HJMDeepBSDE(y0, xi, batch_size)
    model.train_model(epochs=args.epochs, lr=args.lr, save_path=args.save_path)

    timesteps = np.linspace(0, T, N)
    results = simulate_all(model, n_paths=args.n_paths, batch_size=args.batch_size, y0=y0)
    plot_all_diagnostics(results, timesteps)

if __name__ == "__main__":
    main()
