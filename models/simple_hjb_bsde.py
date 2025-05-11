import torch
import numpy as np
from core.base_bsde import BaseDeepBSDE
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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
        return q

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
        A = ((G + 1) / (G - 1)) * np.exp(2 * (self.T - t))
        return (A + 1) / (A - 1)

    def optimal_control_analytic(self, t, x):
        """Optimal q(t, x) = -K(t) * x"""
        K_t = self.K_analytic(t).reshape(-1, 1)           # shape: (T, 1)
        return -K_t * x  # shape: (T, N_paths)

    def phi_analytic(self, t):
        """Analytical phi(t) from backward integral of K"""
        sigma = self.sigma_x
        G = self.G
        def A(s): return (G + 1) / (G - 1) * np.exp(2 * (self.T - s))
        A_t = A(t)
        A_T = A(self.T)
        log_expr = (A_t / A_T) * ((A_T - 1)**2 / (A_t - 1)**2)
        phi = -0.5 * sigma**2 * np.log(log_expr)
        return phi

    def optimal_cost_analytic(self, t, x):
        """Analytical cost-to-go Y(t) = phi(t) + K(t) * x^2"""
        K_t = self.K_analytic(t).reshape(-1, 1)  # (T, 1)
        phi_t = self.phi_analytic(t).reshape(-1, 1)  # (T, 1)
        return phi_t + K_t * x**2  # shape (T, N)

    def plot_approx_vs_analytic(self, results, timesteps):
        approx_q = results["q"]              # shape: (T, N_paths)
        x_vals = results["final_y"][:, :, 0] # shape: (T, N_paths)
        Y_vals = results["Y"]                # shape: (T, N_paths, 1)

        with torch.no_grad():
            true_q = self.optimal_control_analytic(timesteps, x_vals)          # shape: (T, N)
            true_Y = self.optimal_cost_analytic(timesteps, x_vals)             # shape (T, N)

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        # --- Subplot 1: Learned q(t) paths ---
        colors = cm.get_cmap("tab10", approx_q.shape[1])  # Get a colormap with enough distinct colors

        for i in range(approx_q.shape[1]):
            axs[0, 0].plot(timesteps, approx_q[:, i], color=colors(i), alpha=0.6, label=f"Learned {i+1}")
            axs[0, 0].plot(timesteps, true_q[:, i], linestyle="--", color=colors(i), label=f"Analytical {i+1}")

        axs[0, 0].set_title("Control $q(t)$: Learned vs Analytical")
        axs[0, 0].set_xlabel("Time $t$")
        axs[0, 0].set_ylabel("$q(t)$")
        axs[0, 0].grid(True)
        axs[0, 0].legend(ncol=2, fontsize=8)

        # --- Subplot 2: Absolute Difference ---
        diff = (approx_q.squeeze(-1) - true_q)  # (T, N_paths)
        for i in range(diff.shape[1]):
            axs[0, 1].plot(timesteps, diff[:, i], label=f"Diff Path {i+1}")
        axs[0, 1].set_title("Difference: Learned $-$ Analytical")
        axs[0, 1].set_xlabel("Time $t$")
        axs[0, 1].set_ylabel("$q(t) - q^*(t)$")
        axs[0, 1].grid(True)

        # --- Subplot 3: Y(t) paths ---
        for i in range(Y_vals.shape[1]):
            axs[1, 0].plot(timesteps, Y_vals[:, i, 0], color=colors(i), alpha=0.6, label=f"Learned {i+1}")
            axs[1, 0].plot(timesteps, true_Y[:, i], linestyle="--", color=colors(i), label=f"Analytical {i+1}")
        axs[1, 0].set_title("Cost-to-Go $Y(t)$")
        axs[1, 0].set_xlabel("Time $t$")
        axs[1, 0].set_ylabel("Y(t)")
        axs[1, 0].grid(True)
        axs[1, 0].legend(ncol=2, fontsize=8)

        # --- Subplot 4: x(t) paths ---
        for i in range(x_vals.shape[1]):
            axs[1, 1].plot(timesteps, x_vals[:, i])
        axs[1, 1].set_title("State $x(t)$")
        axs[1, 1].set_xlabel("Time $t$")
        axs[1, 1].set_ylabel("x(t)")
        axs[1, 1].grid(True)


        plt.tight_layout()
        plt.show()
