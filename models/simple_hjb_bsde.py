import torch
import numpy as np
from core.fbsnn import FBSNN
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class SimpleHJB(FBSNN):
    def __init__(self, args, model_cfg):
        super().__init__(args, model_cfg)
        self.sigma_y = model_cfg["sigma"]
        self.G = model_cfg["G"]

    def generator(self, y, q):
        return q**2 + y**2

    def terminal_cost(self, y):
        return self.G * y**2

    def mu(self, t, y, q):
        return q

    def sigma(self, t, y):
        batch_size = y.shape[0]
        σ = torch.zeros(batch_size, 1, 1, device=self.device)
        σ[:, 0, 0] = self.sigma_y
        return σ
    
    def trading_rate(self, t, y, Y, create_graph=False):
        dV = torch.autograd.grad(
            outputs=Y,
            inputs=y,
            grad_outputs=torch.ones_like(Y),
            create_graph=create_graph,
            retain_graph=True,
        )[0]
        q = - 0.5 * dV
        return q

    def K_analytic(self, t):
        """Analytical solution to Riccati equation"""
        G = self.G
        T = torch.tensor(self.T, device=self.device)
        A = (G + 1) * torch.exp(2 * (T - t))
        return (A + (G - 1)) / (A - (G - 1))

    def optimal_control_analytic(self, t, y):
        """Optimal q(t, y) = -K(t) * y"""
        K_t = self.K_analytic(t)
        return -K_t * y

    def phi_analytic(self, t):
        """Analytical phi(t) from backward integral of K"""
        sigma = self.sigma_y
        G = self.G
        T = torch.tensor(self.T, device=self.device)
        def A(s): return (G + 1) * torch.exp(2 * (T - s))
        A_t = A(t)
        A_T = A(self.T)
        log_expr = (A_T / A_t) * ((A_t - (G - 1))**2 / (A_T - (G - 1))**2)
        phi = 0.5 * sigma**2 * torch.log(log_expr)
        return phi

    def value_function_analytic(self, t, y):
        """Analytical cost-to-go Y(t) = phi(t) + K(t) * y^2"""
        K_t = self.K_analytic(t)
        phi_t = self.phi_analytic(t)
        return phi_t + K_t * y**2

    def plot_approx_vs_analytic(self, results, timesteps, plot=True, save_dir=None):
        approx_q = results["q"]              # shape: (T + 1, N_paths)
        y_vals = results["y"][:, :, 0]       # shape: (T + 1, N_paths)
        Y_vals = results["Y"]                # shape: (T + 1, N_paths, 1)

        with torch.no_grad():
            t_tensor = torch.tensor(timesteps, dtype=torch.float32, device=self.device).unsqueeze(1).repeat(1, y_vals.shape[1])
            y_tensor = torch.tensor(y_vals, dtype=torch.float32, device=self.device)

            true_q = self.optimal_control_analytic(t_tensor, y_tensor).detach().cpu().numpy()
            true_Y = self.value_function_analytic(t_tensor, y_tensor).detach().cpu().numpy()

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        colors = cm.get_cmap("tab10", approx_q.shape[1])  # Get a colormap with enough distinct colors

        # --- Subplot 1: Learned q(t) paths ---
        for i in range(approx_q.shape[1]):
            axs[0, 0].plot(timesteps, approx_q[:, i], color=colors(i), alpha=0.6, label=f"Learned {i+1}")
            axs[0, 0].plot(timesteps, true_q[:, i], linestyle="--", color=colors(i), alpha=0.4, label=f"Analytical {i+1}")
        axs[0, 0].set_title("Control $q(t)$: Learned vs Analytical")
        axs[0, 0].set_xlabel("Time $t$")
        axs[0, 0].set_ylabel("$q(t)$")
        axs[0, 0].grid(True)

        # --- Subplot 2: Absolute Difference ---
        diff = (approx_q.squeeze(-1) - true_q)  # (T, N_paths)
        for i in range(diff.shape[1]):
            axs[0, 1].plot(timesteps, diff[:, i], color=colors(i), alpha=0.6, label=f"Diff Path {i+1}")
        axs[0, 1].axhline(0, color='red', linestyle='--', linewidth=0.8)
        axs[0, 1].set_title("Difference: Learned $-$ Analytical")
        axs[0, 1].set_xlabel("Time $t$")
        axs[0, 1].set_ylabel("$q(t) - q^*(t)$")
        axs[0, 1].grid(True)

        # --- Subplot 3: Y(t) paths ---
        for i in range(Y_vals.shape[1]):
            axs[1, 0].plot(timesteps, Y_vals[:, i, 0], color=colors(i), alpha=0.6, label=f"Learned {i+1}")
            axs[1, 0].plot(timesteps, true_Y[:, i], linestyle="--", color=colors(i), alpha=0.4, label=f"Analytical {i+1}")
        axs[1, 0].set_title("Cost-to-Go $Y(t)$")
        axs[1, 0].set_xlabel("Time $t$")
        axs[1, 0].set_ylabel("Y(t)")
        axs[1, 0].grid(True)

        # --- Subplot 4: y(t) paths ---
        for i in range(y_vals.shape[1]):
            axs[1, 1].plot(timesteps, y_vals[:, i], color=colors(i), alpha=0.6, label=f"$x_{i}(t)$" if i == 0 else None)
        axs[1, 1].set_title("State $x(t)$")
        axs[1, 1].set_xlabel("Time $t$")
        axs[1, 1].set_ylabel("x(t)")
        axs[1, 1].grid(True)

        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/approx_vs_analytic.png", dpi=300, bbox_inches='tight')
        if plot:
            plt.show()

    def plot_approx_vs_analytic_expectation(self, results, timesteps, plot=True, save_dir=None):
        approx_q = results["q"]              # shape: (T + 1, N_paths)
        y_vals = results["y"][:, :, 0]       # shape: (T + 1, N_paths)
        Y_vals = results["Y"]                # shape: (T + 1, N_paths, 1)

        with torch.no_grad():
            t_tensor = torch.tensor(timesteps, dtype=torch.float32, device=self.device).unsqueeze(1).repeat(1, y_vals.shape[1])
            y_tensor = torch.tensor(y_vals, dtype=torch.float32, device=self.device)

            true_q = self.optimal_control_analytic(t_tensor, y_tensor).squeeze().detach().cpu().numpy()
            true_Y = self.value_function_analytic(t_tensor, y_tensor).squeeze().detach().cpu().numpy()

        mean_Y = Y_vals[:, :, 0].mean(axis=1).squeeze()
        std_Y = Y_vals[:, :, 0].std(axis=1).squeeze()
        mean_q = approx_q.mean(axis=1).squeeze()
        std_q = approx_q.std(axis=1).squeeze()

        mean_true_Y = true_Y.mean(axis=1)
        std_true_Y = true_Y.std(axis=1)
        mean_q_true = true_q.mean(axis=1)
        std_q_true = true_q.std(axis=1)

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        colors = cm.get_cmap("tab10", approx_q.shape[1])  # Get a colormap with enough distinct colors

        # --- Subplot 1: Learned q(t) paths ---
        axs[0, 0].plot(timesteps, mean_q, label='Learned Mean', color='blue')
        axs[0, 0].fill_between(timesteps, mean_q - std_q, mean_q + std_q, color='blue', alpha=0.3, label='Learned ±1 Std')

        axs[0, 0].plot(timesteps, mean_q_true, label='Analytical Mean', color='black', linestyle='--')
        axs[0, 0].fill_between(timesteps, mean_q_true - std_q_true, mean_q_true + std_q_true, color='black', alpha=0.2, label='Analytical ±1 Std')

        axs[0, 0].set_title("Control $q(t)$: Learned vs Analytical")
        axs[0, 0].set_xlabel("Time $t$")
        axs[0, 0].set_ylabel("$q(t)$")
        axs[0, 0].grid(True)
        axs[0, 0].legend()

        # --- Subplot 2: Absolute Difference ---
        diff = (approx_q.squeeze(-1) - true_q)  # (T, N_paths)
        mean_diff = np.mean(diff, axis=1)
        std_diff = np.std(diff, axis=1)
        axs[0, 1].fill_between(timesteps, mean_diff - std_diff, mean_diff + std_diff, color='red', alpha=0.4, label='±1 Std Dev')
        axs[0, 1].plot(timesteps, mean_diff, color='red', label='Mean Difference')
        axs[0, 1].set_title("Difference: Learned $-$ Analytical")
        axs[0, 1].set_xlabel("Time $t$")
        axs[0, 1].set_ylabel("$q(t) - q^*(t)$")
        axs[0, 1].grid(True)
        axs[0, 1].legend()

        # --- Subplot 3: Y(t) paths ---
        axs[1, 0].plot(timesteps, mean_Y, color='blue', label='Learned Mean')
        axs[1, 0].fill_between(timesteps, mean_Y - std_Y, mean_Y + std_Y, color='blue', alpha=0.3, label='Learned ±1 Std')

        axs[1, 0].plot(timesteps, mean_true_Y, color='black', linestyle='--', label='Analytical Mean')
        axs[1, 0].fill_between(timesteps, mean_true_Y - std_true_Y, mean_true_Y + std_true_Y, color='black', alpha=0.2, label='Analytical ±1 Std')

        axs[1, 0].set_title("Cost-to-Go $Y(t)$")
        axs[1, 0].set_xlabel("Time $t$")
        axs[1, 0].set_ylabel("Y(t)")
        axs[1, 0].grid(True)
        axs[1, 0].legend()

        # --- Subplot 4: Difference ---
        diff_Y = (Y_vals[:, :, 0] - true_Y)  # (T, N_paths)
        mean_diff_Y = np.mean(diff_Y, axis=1)
        std_diff_Y = np.std(diff_Y, axis=1)
        axs[1, 1].fill_between(timesteps, mean_diff_Y - std_diff_Y, mean_diff_Y + std_diff_Y, color='red', alpha=0.4, label='±1 Std Dev')
        axs[1, 1].plot(timesteps, mean_diff_Y, color='red', label='Mean Difference')
        axs[1, 1].set_title("Difference: Learned $Y(t) - Y^*(t)$")
        axs[1, 1].set_xlabel("Time $t$")
        axs[1, 1].set_ylabel("$Y(t) - Y^*(t)$")
        axs[1, 1].grid(True)
        axs[1, 1].legend()

        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/approx_vs_analytic_expectation.png", dpi=300, bbox_inches='tight')
        if plot:
            plt.show()

    def plot_terminal_histogram(self, results, plot=True, save_dir=None):
        y_vals = results["y"][:, :, 0]       # shape: (T + 1, N_paths)
        Y_vals = results["Y"]                # shape: (T + 1, N_paths, 1)

        Y_T_approx = Y_vals[-1, :, 0]  # shape: (N_paths,)
        y_T = y_vals[-1, :]            # shape: (N_paths,)
        Y_T_true = self.terminal_cost(torch.tensor(y_T, device=self.device).unsqueeze(1)).cpu().numpy()

        plt.figure(figsize=(8, 6))
        bins = 30

        plt.hist(Y_T_approx, bins=bins, alpha=0.6, label="Approx. $Y_T$", color="blue", density=True)
        plt.hist(Y_T_true, bins=bins, alpha=0.6, label="Analytical $g(y_T)$", color="green", density=True)

        mean_approx = np.mean(Y_T_approx)
        mean_true = np.mean(Y_T_true)

        plt.axvline(mean_approx, color='blue', linestyle='--', label=f"Mean approx: {mean_approx:.3f}")
        plt.axvline(mean_true, color='green', linestyle='--', label=f"Mean true: {mean_true:.3f}")

        plt.title("Distribution of Terminal Values")
        plt.xlabel("$Y(T)$ / $g(y_T)$")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/terminal_histogram.png", dpi=300, bbox_inches='tight')
        if plot:
            plt.show()