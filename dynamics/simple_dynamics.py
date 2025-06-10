import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

from dynamics.dynamics import Dynamics


class SimpleDynamics(Dynamics):
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
        Sigma = torch.zeros(batch_size, 1, 1, device=self.device)
        Sigma[:, 0, 0] = self.sigma_y
        return Sigma
    
    def optimal_control(self, t, y, Y, create_graph=False):
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


    # TODO: Implement these functions to be more flexible depending if analytical is known
    def plot_approx_vs_analytic(self, results, timesteps, plot=True, save_dir=None, num=None):

        approx_q = results["q_learned"]
        y_vals = results["y_learned"]
        Y_vals = results["Y_learned"]
        true_q = results["q_analytical"]
        true_y = results["y_analytical"]
        true_Y = results["Y_analytical"]

        fig, axs = plt.subplots(3, 2, figsize=(14, 10))
        colors = cm.get_cmap("tab10", approx_q.shape[1])

        for i in range(approx_q.shape[1]):
            axs[0, 0].plot(timesteps, approx_q[:, i], color=colors(i), alpha=0.6, label=f"Learned $q_{i}(t)$" if i == 0 else None)
            axs[0, 0].plot(timesteps, true_q[:, i], linestyle="--", color=colors(i), alpha=0.4, label=f"Analytical $q^*_{i}(t)$" if i == 0 else None)
        axs[0, 0].set_title("Control $q(t)$: Learned vs Analytical")
        axs[0, 0].set_xlabel("Time $t$")
        axs[0, 0].set_ylabel("$q(t)$")
        axs[0, 0].grid(True)
        axs[0, 0].legend(loc='upper left')

        for i in range(approx_q.shape[1]):
            diff = (approx_q[:, i] - true_q[:, i]) ** 2
            axs[0, 1].plot(timesteps, diff, color=colors(i), alpha=0.6, label=f"$q_{i}(t) - q^*_{i}(t)$" if i == 0 else None)
        axs[0, 1].axhline(0, color='red', linestyle='--', linewidth=0.8)
        axs[0, 1].set_title("Error in Control $q(t)$")
        axs[0, 1].set_xlabel("Time $t$")
        axs[0, 1].set_ylabel("$|q(t) - q^*(t)|^2$")
        axs[0, 1].grid(True)
        axs[0, 1].legend(loc='upper left')

        for i in range(Y_vals.shape[1]):
            axs[1, 0].plot(timesteps, Y_vals[:, i, 0], color=colors(i), alpha=0.6, label=f"Learned $Y_{i}(t)$" if i == 0 else None)
            axs[1, 0].plot(timesteps, true_Y[:, i, 0], linestyle="--", color=colors(i), alpha=0.4, label=f"Analytical $Y^*_{i}(t)$" if i == 0 else None)
        axs[1, 0].set_title("Cost-to-Go $Y(t)$")
        axs[1, 0].set_xlabel("Time $t$")
        axs[1, 0].set_ylabel("Y(t)")
        axs[1, 0].grid(True)
        axs[1, 0].legend(loc='upper left')

        for i in range(Y_vals.shape[1]):
            diff_Y = (Y_vals[:, i, 0] - true_Y[:, i, 0]) ** 2
            axs[1, 1].plot(timesteps, diff_Y, color=colors(i), alpha=0.6, label=f"$Y_{i}(t) - Y^*_{i}(t)$" if i == 0 else None)
        axs[1, 1].axhline(0, color='red', linestyle='--', linewidth=0.8)
        axs[1, 1].set_title("Error in Cost-to-Go $Y(t)$")
        axs[1, 1].set_xlabel("Time $t$")
        axs[1, 1].set_ylabel("$|Y(t) - Y^*(t)|^2$")
        axs[1, 1].grid(True)
        axs[1, 1].legend(loc='upper left')

        for i in range(y_vals.shape[1]):
            axs[2, 0].plot(timesteps, y_vals[:, i, 0], color=colors(i), alpha=0.6, label=f"$x_{i}(t)$" if i == 0 else None)
            axs[2, 0].plot(timesteps, true_y[:, i, 0], linestyle="--", color=colors(i), alpha=0.4, label=f"$x^*_{i}(t)$" if i == 0 else None)
        axs[2, 0].set_title("States: $x(t)$ and $d(t)$")
        axs[2, 0].set_xlabel("Time $t$")
        axs[2, 0].set_ylabel("x(t), d(t)")
        axs[2, 0].grid(True)
        axs[2, 0].legend(loc='upper left')

        plt.tight_layout()
        if save_dir:
            if num:
                plt.savefig(f"{save_dir}/approx_vs_analytic_{num}.png", dpi=300, bbox_inches='tight')
            else:
                plt.savefig(f"{save_dir}/approx_vs_analytic.png", dpi=300, bbox_inches='tight')
            plt.close()
        if plot:
            plt.show()

    def plot_approx_vs_analytic_expectation(self, results, timesteps, plot=True, save_dir=None, num=None):
        approx_q = results["q_learned"]
        Y_vals = results["Y_learned"]
        true_q = results["q_analytical"]
        true_Y = results["Y_analytical"]

        # Learned results
        mean_q = approx_q.mean(axis=1).squeeze()
        std_q = approx_q.std(axis=1).squeeze()
        mean_Y = Y_vals[:, :, 0].mean(axis=1).squeeze()
        std_Y = Y_vals[:, :, 0].std(axis=1).squeeze()

        # Analytic results
        mean_q_analytical = true_q.mean(axis=1).squeeze()
        std_q_analytical = true_q.std(axis=1).squeeze()
        mean_true_Y = true_Y.mean(axis=1).squeeze()
        std_true_Y = true_Y.std(axis=1).squeeze()

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        axs[0, 0].plot(timesteps, mean_q, label='Learned Mean', color='blue')
        axs[0, 0].fill_between(timesteps, mean_q - std_q, mean_q + std_q, color='blue', alpha=0.3, label='Learned ±1 Std')
        axs[0, 0].plot(timesteps, mean_q_analytical, label='Analytical Mean', color='black', linestyle='--')
        axs[0, 0].fill_between(timesteps, mean_q_analytical - std_q_analytical, mean_q_analytical + std_q_analytical, color='black', alpha=0.2, label='Analytical ±1 Std')
        axs[0, 0].set_title("Control $q(t)$: Learned vs Analytical")
        axs[0, 0].set_xlabel("Time $t$")
        axs[0, 0].set_ylabel("$q(t)$")
        axs[0, 0].grid(True)
        axs[0, 0].legend(loc='upper left')

        diff = (approx_q - true_q) ** 2
        mean_diff = np.mean(diff, axis=1).squeeze()
        std_diff = np.std(diff, axis=1).squeeze()
        axs[0, 1].fill_between(timesteps, mean_diff - std_diff, mean_diff + std_diff, color='red', alpha=0.4, label='±1 Std Dev')
        axs[0, 1].plot(timesteps, mean_diff, color='red', label='Mean Difference')
        axs[0, 1].set_title("Error in Control $q(t)$")
        axs[0, 1].set_xlabel("Time $t$")
        axs[0, 1].set_ylabel("$|q(t) - q^*(t)|^2$")
        axs[0, 1].grid(True)
        axs[0, 1].legend(loc='upper left')

        axs[1, 0].plot(timesteps, mean_Y, color='blue', label='Learned Mean')
        axs[1, 0].fill_between(timesteps, mean_Y - std_Y, mean_Y + std_Y, color='blue', alpha=0.3, label='Learned ±1 Std')
        axs[1, 0].plot(timesteps, mean_true_Y, color='black', linestyle='--', label='Analytical Mean')
        axs[1, 0].fill_between(timesteps, mean_true_Y - std_true_Y, mean_true_Y + std_true_Y, color='black', alpha=0.2, label='Analytical ±1 Std')
        axs[1, 0].set_title("Cost-to-Go $Y(t)$")
        axs[1, 0].set_xlabel("Time $t$")
        axs[1, 0].set_ylabel("Y(t)")
        axs[1, 0].grid(True)
        axs[1, 0].legend(loc='upper left')

        diff_Y = (Y_vals - true_Y) ** 2
        mean_diff_Y = np.mean(diff_Y, axis=1).squeeze()
        std_diff_Y = np.std(diff_Y, axis=1).squeeze()
        axs[1, 1].fill_between(timesteps, mean_diff_Y - std_diff_Y, mean_diff_Y + std_diff_Y, color='red', alpha=0.4, label='±1 Std Dev')
        axs[1, 1].plot(timesteps, mean_diff_Y, color='red', label='Mean Difference')
        axs[1, 1].set_title("Error in Cost-to-Go $Y(t)$")
        axs[1, 1].set_xlabel("Time $t$")
        axs[1, 1].set_ylabel("$|Y(t) - Y^*(t)|^2$")
        axs[1, 1].grid(True)
        axs[1, 1].legend(loc='upper left')

        plt.tight_layout()
        if save_dir:
            if num:
                plt.savefig(f"{save_dir}/approx_vs_analytic_expectation_{num}.png", dpi=300, bbox_inches='tight')
            else:
                plt.savefig(f"{save_dir}/approx_vs_analytic_expectation.png", dpi=300, bbox_inches='tight')
            plt.close()
        if plot:
            plt.show()
        
    def plot_terminal_histogram(self, results, plot=True, save_dir=None, num=None):
        y_vals = results["y_learned"]  # shape: (T+1, N_paths, dim)
        q_vals = results["q_learned"]
        Y_vals = results["Y_learned"]  # shape: (T+1, N_paths, 1)

        y_T = y_vals[-1, :, :]
        Y_T_approx = Y_vals[-1, :, 0]
        q_T_approx = q_vals[-1, :, 0]
        y_T_tensor = torch.tensor(y_T, dtype=torch.float32, device=self.device)
        Y_T_true = self.terminal_cost(y_T_tensor).detach().cpu().numpy().squeeze()
        q_T_true = self.optimal_control_analytic(self.T, y_T_tensor).detach().cpu().numpy().squeeze()

        # Filter out NaN or Inf
        mask = np.isfinite(Y_T_approx) & np.isfinite(Y_T_true)
        Y_T_approx = Y_T_approx[mask]
        Y_T_true = Y_T_true[mask]

        mask_q = np.isfinite(q_T_approx) & np.isfinite(q_T_true)
        q_T_approx = q_T_approx[mask_q]
        q_T_true = q_T_true[mask_q]

        if len(Y_T_approx) == 0 or len(Y_T_true) == 0:
            print("Warning: No valid terminal values to plot.")
            return

        range_approx = np.ptp(Y_T_approx)  # Peak-to-peak (max - min)
        range_true = np.ptp(Y_T_true)
        range_combined = max(range_approx, range_true)

        if range_combined == 0:
            print("Warning: No variation in terminal values. Skipping histogram.")
            return

        # Choose bins depending on data spread
        bins = min(30, max(1, int(range_combined / 1e-2)))
        
        plt.figure(figsize=(14, 10))

        plt.subplot(2, 1, 1)
        plt.hist(Y_T_approx, bins=bins, alpha=0.6, label="Approx. $Y_T$", color="blue", density=True)
        plt.hist(Y_T_true, bins=bins, alpha=0.6, label="Analytical $g(y_T)$", color="green", density=True)
        plt.axvline(np.mean(Y_T_approx), color='blue', linestyle='--', label=f"Mean approx: {np.mean(Y_T_approx):.3f}")
        plt.axvline(np.mean(Y_T_true), color='green', linestyle='--', label=f"Mean true: {np.mean(Y_T_true):.3f}")
        plt.title("Distribution of Terminal Values")
        plt.xlabel("$Y(T)$ / $g(y_T)$")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.subplot(2, 1, 2)
        plt.hist(q_T_approx, bins=bins, alpha=0.6, label="Approx. $q_T$", color="blue", density=True)
        plt.hist(q_T_true, bins=bins, alpha=0.6, label="Analytical $q^*(y_T)$", color="green", density=True)
        plt.axvline(np.mean(q_T_approx), color='blue', linestyle='--', label=f"Mean approx: {np.mean(q_T_approx):.3f}")
        plt.axvline(np.mean(q_T_true), color='green', linestyle='--', label=f"Mean true: {np.mean(q_T_true):.3f}")
        plt.title("Distribution of Terminal Controls")
        plt.xlabel("$q(T)$ / $q^*(y_T)$")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if save_dir:
            if num:
                plt.savefig(f"{save_dir}/terminal_histogram_{num}.png", dpi=300, bbox_inches='tight')
            else:
                plt.savefig(f"{save_dir}/terminal_histogram.png", dpi=300, bbox_inches='tight')
            plt.close()
        if plot:
            plt.show()