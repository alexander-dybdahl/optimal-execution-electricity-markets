import torch
import numpy as np
from core.fbsnn import FBSNN
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class SimpleHJB(FBSNN):
    def __init__(self, args, model_cfg):
        super().__init__(args, model_cfg)
        self.sigma_x = model_cfg["sigma"]
        self.G = model_cfg["G"]

    def generator(self, y, q):
        x = y[:, 0:1]
        return q**2 + x**2

    def terminal_cost(self, y):
        x_T = y
        return self.G * x_T**2

    def mu(self, t, y, q):
        return q

    def sigma(self, t, y):
        batch_size = y.shape[0]
        σ = torch.zeros(batch_size, 1, 1, device=self.device)
        σ[:, 0, 0] = self.sigma_x
        return σ

    def simulate_paths(self, n_sim=5, seed=42, y0_single=None):
        torch.manual_seed(seed)
        self.eval()

        all_q, all_Y, all_y = [], [], []

        y = y0_single.repeat(n_sim, 1) if y0_single is not None else self.y0.repeat(n_sim, 1)
        t = torch.zeros(n_sim, 1, device=self.device)
        
        q_traj, Y_traj, y_traj = [], [], []

        # Initial value function and its gradient
        Y = self.Y_net(t, y)
        dY = torch.autograd.grad(
            outputs=Y,
            inputs=y,
            grad_outputs=torch.ones_like(Y),
            create_graph=False,
            retain_graph=False
        )[0]

        q = -0.5 * dY

        q_traj.append(q.detach().cpu().numpy())
        Y_traj.append(Y.detach().cpu().numpy())
        y_traj.append(y.detach().cpu().numpy())

        for i in range(self.N):
            dW = torch.randn(n_sim, self.dim_W, device=self.device) * self.dt**0.5
            y = self.forward_dynamics(y, q, dW, t, self.dt)
            t += self.dt
            
            Y = self.Y_net(t, y)
            dY = torch.autograd.grad(
                outputs=Y,
                inputs=y,
                grad_outputs=torch.ones_like(Y),
                create_graph=False,
                retain_graph=False
            )[0]

            q = -0.5 * dY

            q_traj.append(q.detach().cpu().numpy())
            Y_traj.append(Y.detach().cpu().numpy())
            y_traj.append(y.detach().cpu().numpy())

        all_q.append(np.stack(q_traj))     # shape: (N, batch)
        all_Y.append(np.stack(Y_traj))
        all_y.append(np.stack(y_traj))

        timesteps = np.linspace(0, self.T, self.N + 1)

        return timesteps, {
            "q": np.concatenate(all_q, axis=1),         # shape (N, n_paths)
            "Y": np.concatenate(all_Y, axis=1),
            "y": np.concatenate(all_y, axis=1)          # shape (N, n_paths, dim)
        }

    def K_analytic(self, t):
        """Analytical solution to Riccati equation"""
        G = self.G
        A = (G + 1) * np.exp(2 * (self.T - t))
        return (A + (G - 1)) / (A - (G - 1))

    def optimal_control_analytic(self, t, x):
        """Optimal q(t, x) = -K(t) * x"""
        K_t = self.K_analytic(t).reshape(-1, 1)           # shape: (T, 1)
        return -K_t * x                                   # shape: (T, N_paths)

    def phi_analytic(self, t):
        """Analytical phi(t) from backward integral of K"""
        sigma = self.sigma_x
        G = self.G
        def A(s): return (G + 1) * np.exp(2 * (self.T - s))
        A_t = A(t)
        A_T = A(self.T)
        log_expr = (A_T / A_t) * ((A_t - (G - 1))**2 / (A_T - (G - 1))**2)
        phi = 0.5 * sigma**2 * np.log(log_expr)
        return phi

    def optimal_cost_analytic(self, t, x):
        """Analytical cost-to-go Y(t) = phi(t) + K(t) * x^2"""
        K_t = self.K_analytic(t).reshape(-1, 1)      # shape (T, 1)
        phi_t = self.phi_analytic(t).reshape(-1, 1)  # shape (T, 1)
        return phi_t + K_t * x**2                    # shape (T, N)

    def plot_approx_vs_analytic(self, results, timesteps):
        approx_q = results["q"]              # shape: (T + 1, N_paths)
        x_vals = results["y"][:, :, 0]       # shape: (T + 1, N_paths)
        Y_vals = results["Y"]                # shape: (T + 1, N_paths, 1)

        with torch.no_grad():
            true_q = self.optimal_control_analytic(timesteps, x_vals)          # shape: (T + 1, N)
            true_Y = self.optimal_cost_analytic(timesteps, x_vals)             # shape (T + 1, N)

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

        # --- Subplot 4: x(t) paths ---
        for i in range(x_vals.shape[1]):
            axs[1, 1].plot(timesteps, x_vals[:, i], color=colors(i), alpha=0.6, label=f"Path {i+1}")
        axs[1, 1].set_title("State $x(t)$")
        axs[1, 1].set_xlabel("Time $t$")
        axs[1, 1].set_ylabel("x(t)")
        axs[1, 1].grid(True)

        plt.tight_layout()
        plt.show()

    def plot_approx_vs_analytic_expectation(self, results, timesteps):
        approx_q = results["q"]              # shape: (T + 1, N_paths)
        x_vals = results["y"][:, :, 0]       # shape: (T + 1, N_paths)
        Y_vals = results["Y"]                # shape: (T + 1, N_paths, 1)

        with torch.no_grad():
            true_q = self.optimal_control_analytic(timesteps, x_vals)          # shape: (T + 1, N)
            true_Y = self.optimal_cost_analytic(timesteps, x_vals)             # shape (T + 1, N)

        mean_Y = Y_vals[:, :, 0].mean(axis=1).squeeze()
        std_Y = Y_vals[:, :, 0].std(axis=1).squeeze()

        mean_true_Y = true_Y.mean(axis=1).squeeze()
        std_true_Y = true_Y.std(axis=1).squeeze()

        mean_q = approx_q.mean(axis=1).squeeze()
        std_q = approx_q.std(axis=1).squeeze()

        mean_q_true = true_q.mean(axis=1).squeeze()
        std_q_true = true_q.std(axis=1).squeeze()

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

        # --- Subplot 2: Absolute Difference ---
        diff = (approx_q.squeeze(-1) - true_q)  # (T, N_paths)
        mean_diff = np.mean(diff, axis=1)
        std_diff = np.std(diff, axis=1)
        axs[0, 1].fill_between(timesteps, mean_diff - std_diff, mean_diff + std_diff, color='red', alpha=0.4, label='±1 Std Dev')

        axs[0, 1].set_title("Difference: Learned $-$ Analytical")
        axs[0, 1].set_xlabel("Time $t$")
        axs[0, 1].set_ylabel("$q(t) - q^*(t)$")
        axs[0, 1].grid(True)

        # --- Subplot 3: Y(t) paths ---
        axs[1, 0].plot(timesteps, mean_Y, color='blue', label='Learned Mean')
        axs[1, 0].fill_between(timesteps, mean_Y - std_Y, mean_Y + std_Y, color='blue', alpha=0.3, label='Learned ±1 Std')

        axs[1, 0].plot(timesteps, mean_true_Y, color='black', linestyle='--', label='Analytical Mean')
        axs[1, 0].fill_between(timesteps, mean_true_Y - std_true_Y, mean_true_Y + std_true_Y, color='black', alpha=0.2, label='Analytical ±1 Std')

        axs[1, 0].set_title("Cost-to-Go $Y(t)$")
        axs[1, 0].set_xlabel("Time $t$")
        axs[1, 0].set_ylabel("Y(t)")
        axs[1, 0].grid(True)

        # --- Subplot 4: x(t) paths ---
        for i in range(x_vals.shape[1]):
            axs[1, 1].plot(timesteps, x_vals[:, i], color=colors(i), alpha=0.6, label=f"Path {i+1}")
        axs[1, 1].set_title("State $x(t)$")
        axs[1, 1].set_xlabel("Time $t$")
        axs[1, 1].set_ylabel("x(t)")
        axs[1, 1].grid(True)

        plt.tight_layout()
        plt.show()

    def plot_terminal_histogram(self, results):
        x_vals = results["y"][:, :, 0]       # shape: (T + 1, N_paths)
        Y_vals = results["Y"]                # shape: (T + 1, N_paths, 1)

        Y_T_approx = Y_vals[-1, :, 0]  # shape: (N_paths,)
        x_T = x_vals[-1, :]            # shape: (N_paths,)
        Y_T_true = self.terminal_cost(torch.tensor(x_T, device=self.device).unsqueeze(1)).cpu().numpy()

        plt.figure(figsize=(8, 6))
        bins = 30

        plt.hist(Y_T_approx, bins=bins, alpha=0.6, label="Approx. $Y_T$", color="blue", density=True)
        plt.hist(Y_T_true, bins=bins, alpha=0.6, label="Analytical $g(x_T)$", color="green", density=True)

        mean_approx = np.mean(Y_T_approx)
        mean_true = np.mean(Y_T_true)

        plt.axvline(mean_approx, color='blue', linestyle='--', label=f"Mean approx: {mean_approx:.3f}")
        plt.axvline(mean_true, color='green', linestyle='--', label=f"Mean true: {mean_true:.3f}")

        plt.title("Distribution of Terminal Values")
        plt.xlabel("$Y(T)$ / $g(x_T)$")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def forward_supervised(self, t_paths, W_paths):
        batch_size = self.batch_size

        t = torch.rand(batch_size, 1, device=self.device) * self.T
        x = torch.randn(batch_size, 1, device=self.device, requires_grad=True)

        # -- Supervised loss
        with torch.no_grad():
            K = torch.tensor(self.K_analytic(t.cpu().numpy()), device=self.device).float()
            phi = torch.tensor(self.phi_analytic(t.cpu().numpy()), device=self.device).float()
            V_target = phi + K * x**2
            dV_target = 2 * K * x

        V_pred = self.Y_net(t, x)
        supervised_loss = torch.mean((V_pred - V_target)**2)

        dV_pred = torch.autograd.grad(
            outputs=V_pred,
            inputs=x,
            grad_outputs=torch.ones_like(V_pred),
            create_graph=True
        )[0]

        gradient_loss = torch.mean((dV_pred - dV_target)**2)

        Y_loss = supervised_loss + gradient_loss

        # Terminal losses
        t_terminal = torch.full((batch_size, 1), self.T, device=self.device)
        YT = self.Y_net(t_terminal, x)
        terminal = self.terminal_cost(x)
        terminal_loss = torch.mean(torch.pow(YT - terminal, 2))

        # Terminal gradient loss
        dYT = torch.autograd.grad(
            outputs=YT,
            inputs=x,
            grad_outputs=torch.ones_like(YT),
            create_graph=True,
            retain_graph=True
        )[0]
        terminal_gradient = self.terminal_cost_grad(x)
        terminal_gradient_loss = torch.mean(torch.pow(dYT - terminal_gradient, 2))

        self.λ_T, self.λ_TG = 0, 0
        self.total_Y_loss = self.λ_Y * Y_loss.detach().item()
        self.terminal_loss = self.λ_T * terminal_loss.detach().item()
        self.terminal_gradient_loss = self.λ_TG * terminal_gradient_loss.detach().item()
        self.terminal_hessian_loss = 0.0
        self.pinn_loss = 0.0

        return self.λ_Y * Y_loss + self.λ_T * terminal_loss + self.λ_TG * terminal_gradient_loss