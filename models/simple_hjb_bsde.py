import torch
import numpy as np
from core.base_bsde import BaseDeepBSDE
import matplotlib.pyplot as plt

class SimpleHJB(BaseDeepBSDE):
    def __init__(self, args, model_cfg):
        super().__init__(args, model_cfg)
        self.sigma_x = model_cfg["sigma"]
        self.G = model_cfg["G"]

    def generator(self, y, q):
        x = y[:, 0:1]
        return q**2 + x**2

    def terminal_cost(self, y):
        x_T = y[:, 0]
        return self.G * x_T**2

    def mu(self, t, y, q):
        return q  # dx = q dt

    def sigma(self, t, y):
        batch_size = y.shape[0]
        σ = torch.zeros(batch_size, 1, 1, device=self.device)
        σ[:, 0, 0] = self.sigma_x
        return σ

    def simulate_paths(self, n_paths=1000, batch_size=256, seed=42, y0_single=None):
        torch.manual_seed(seed)
        self.eval()

        all_q, all_Y, all_y = [], [], []

        for _ in range(n_paths // batch_size):
            y = y0_single.repeat(batch_size, 1) if y0_single is not None else self.y0.repeat(batch_size, 1)
            t = torch.zeros(batch_size, 1, device=self.device)
            Y = self.Y0.repeat(batch_size, 1)

            q_traj, Y_traj, y_traj = [], [], []

            for i in range(self.N):
                t_input = t.clone()  # (batch, 1)
                q = self.q_net(t_input, y)  # (batch, 1)
                z = self.z_net(t_input, y)  # (batch, dim_W)
                f = self.generator(y, q)

                dW = torch.randn(batch_size, self.dim_W, device=self.device) * self.dt**0.5
                y = self.forward_dynamics(y, q, dW, t, self.dt)
                Y = Y - f * self.dt + (z * dW).sum(dim=1, keepdim=True)
                t += self.dt

                q_traj.append(q.detach().cpu().numpy())
                Y_traj.append(Y.detach().cpu().numpy())
                y_traj.append(y.detach().cpu().numpy())

            all_q.append(np.stack(q_traj))     # shape: (N, batch)
            all_Y.append(np.stack(Y_traj))
            all_y.append(np.stack(y_traj))

        timesteps = np.linspace(0, self.T, self.N)

        return timesteps, {
            "q": np.concatenate(all_q, axis=1),         # (N, n_paths)
            "Y": np.concatenate(all_Y, axis=1),
            "final_y": np.concatenate(all_y, axis=1)    # (N, n_paths, dim)
        }

    def K_analytic(self, t):
        """Analytical solution to Riccati equation"""
        G = self.G
        numerator = 1 + G * np.tanh(self.T - t)
        denominator = G + np.tanh(self.T - t)
        return numerator / denominator  # scalar

    def optimal_control_analytic(self, t, x):
        """Optimal q(t, x) = -K(t) * x"""
        t_np = t.detach().cpu().numpy()
        K_t = self.K_analytic(t_np)
        K_tensor = torch.tensor(K_t, dtype=torch.float32, device=self.device).unsqueeze(1)
        return -K_tensor * x

    def plot_approx_vs_analytic(self, results, timesteps):
        approx_q = results["q"]              # shape: (T, N_paths)
        x_vals = results["final_y"][:, :, 0] # shape: (T, N_paths)
        Y_vals = results["Y"]                # shape: (T, N_paths, 1)

        t_tensor = torch.tensor(timesteps, dtype=torch.float32).unsqueeze(1).to(self.device)  # shape: (T, 1)
        x_tensor = torch.tensor(x_vals, dtype=torch.float32).to(self.device)                  # shape: (T, N)

        with torch.no_grad():
            true_q = self.optimal_control_analytic(t_tensor, x_tensor).cpu().numpy()          # shape: (T, N)

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        # --- Subplot 1: Learned q(t) paths ---
        for i in range(approx_q.shape[1]):
            axs[0, 0].plot(timesteps, approx_q[:, i], alpha=0.4)
        axs[0, 0].set_title("Learned Control $q(t)$")
        axs[0, 0].set_xlabel("Time $t$")
        axs[0, 0].set_ylabel("q(t)")
        axs[0, 0].grid(True)

        # --- Subplot 2: Analytical q*(t) paths ---
        for i in range(true_q.shape[1]):
            axs[0, 1].plot(timesteps, true_q[:, i], linestyle="--", alpha=0.4)
        axs[0, 1].set_title("Analytical Optimal Control $q^*(t)$")
        axs[0, 1].set_xlabel("Time $t$")
        axs[0, 1].set_ylabel("q*(t)")
        axs[0, 1].grid(True)

        # --- Subplot 3: x(t) paths ---
        for i in range(x_vals.shape[1]):
            axs[1, 0].plot(timesteps, x_vals[:, i], alpha=0.4)
        axs[1, 0].set_title("State $x(t)$")
        axs[1, 0].set_xlabel("Time $t$")
        axs[1, 0].set_ylabel("x(t)")
        axs[1, 0].grid(True)

        # --- Subplot 4: Y(t) paths ---
        for i in range(Y_vals.shape[1]):
            axs[1, 1].plot(timesteps, Y_vals[:, i, 0], alpha=0.3)
        axs[1, 1].set_title("Cost-to-Go $Y(t)$")
        axs[1, 1].set_xlabel("Time $t$")
        axs[1, 1].set_ylabel("Y(t)")
        axs[1, 1].grid(True)

        plt.tight_layout()
        plt.show()
