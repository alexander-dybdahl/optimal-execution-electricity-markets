from core.hjb_bsde import HJMDeepBSDE
from simulation.simulator import simulate_all
from helpers.plots import plot_all_diagnostics
from config import y0, xi, batch_size, T, N
import numpy as np

def main():
    model = HJMDeepBSDE(y0, xi, batch_size)
    model.train_model(epochs=1000, lr=1e-3, save_path="model.pth")
    timesteps = np.linspace(0, T, N)
    results = simulate_all(model, n_paths=5, batch_size=5, y0=y0)
    plot_all_diagnostics(results, timesteps)

if __name__ == "__main__":
    main()
