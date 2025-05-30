import torch
import numpy as np
from core.fbsnn import FBSNN
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class AidIntradayLQ(FBSNN):
    def __init__(self, args, model_cfg):
        super().__init__(args, model_cfg)
        self.sigma_P = model_cfg["sigma_P"]
        self.sigma_D = model_cfg["sigma_D"]
        self.rho = model_cfg["rho"]
        self.gamma = model_cfg["gamma"]  # temp impact
        self.nu = model_cfg["nu"]        # perm impact
        self.eta = model_cfg["eta"]        # terminal penalty
        self.mu_D = model_cfg["mu_D"]     # drift for D
        self.mu_P = model_cfg["mu_P"]     # drift for D

    def generator(self, y, q):
        P = y[:, 1:2]
        temporary_impact = self.gamma * q
        execution_price = P + temporary_impact
        return q * execution_price

    def terminal_cost(self, y):
        D = y[:, 2:3]
        X = y[:, 0:1]
        return 0.5 * self.eta * (D - X)**2

    def mu(self, t, y, q):
        X = y[:, 0:1]
        P = y[:, 1:2]
        D = y[:, 2:3]

        dX = q
        dP = self.mu_P + self.nu * q # drift + permanent impact
        dD = self.mu_D * torch.ones_like(D)

        return torch.cat([dX, dP, dD], dim=1)

    def sigma(self, t, y):
        batch = y.shape[0]

        Sigma = torch.zeros(batch, 3, 2, device=self.device)
        Sigma[:, 1, 0] = self.sigma_P                          # dP = ... dW1
        Sigma[:, 2, 0] = self.rho * self.sigma_D               # dD = ... dW1
        Sigma[:, 2, 1] = (1 - self.rho**2)**0.5 * self.sigma_D # dD = ... dW2
        return Sigma

    def trading_rate(self, t, y, Y, create_graph=True):
        dV = torch.autograd.grad(
            outputs=Y,
            inputs=y,
            grad_outputs=torch.ones_like(Y),
            create_graph=create_graph
        )[0]

        dV_X = dV[:, 0:1]
        dV_P = dV[:, 1:2]
        P = y[:, 1:2]

        q = -0.5 / self.gamma * (P + dV_X + self.nu * dV_P)
        return q

    def optimal_control_analytic(self, t, y):
        X = y[:, 0:1]
        P = y[:, 1:2]
        D = y[:, 2:3]
        tau = self.T - t
        return (self.eta * (self.mu_D * tau + D - X) - P) / ((self.eta + self.nu) * tau + 2 * self.gamma)

    def value_function_analytic(self, t, y):
        X = y[:, 0:1]
        P = y[:, 1:2]
        D = y[:, 2:3]

        T = self.T
        eta = self.eta
        nu = self.nu
        gamma = self.gamma
        mu = self.mu_D
        sigma_0 = self.sigma_P
        sigma_d = self.sigma_D
        rho = self.rho

        tau = T - t
        denom = (eta + nu) * tau + 2 * gamma

        # Coefficient functions
        A = eta * (0.5 * nu * tau + gamma) / denom
        B = -0.5 * tau / denom
        F = eta * tau / denom
        G = 2 * mu * tau * A
        H = -2 * eta * mu * tau * B

        # Constant K(t)
        log_term = gamma * (sigma_0**2 + sigma_d**2 * eta**2 - 2 * rho * sigma_0 * sigma_d * eta) / (eta + nu)**2
        log_expr = 1 + ((eta + nu) * tau) / (2 * gamma)
        K1 = log_term * torch.log(log_expr)

        K2 = (sigma_d**2 * eta * nu + 2 * rho * sigma_0 * sigma_d * eta - sigma_0**2) / (2 * (eta + nu)) * tau
        K3 = eta * mu**2 * tau**2 * (0.5 * nu * tau + gamma) / denom
        K = K1 + K2 + K3

        # Value function assembly
        diff = D - X
        V = A * diff**2 + B * P**2 + F * diff * P + G * diff + H * P + K
        return V

    def pinn_loss(self, t, y, Y):
        return super().pinn_loss(t, y, Y)

    def simulate_paths(self, n_sim=5, seed=42, y0_single=None):
        torch.manual_seed(seed)
        self.eval()

        y = y0_single.repeat(n_sim, 1) if y0_single is not None else self.y0.repeat(n_sim, 1)
        t = torch.zeros(n_sim, 1, device=self.device)
  
        q_traj, Y_traj, y_traj = [], [], []

        Y = self.Y_net(t, y)
        q = self.trading_rate(t, y, Y, create_graph=False)

        q_traj.append(q.detach().cpu().numpy())
        Y_traj.append(Y.detach().cpu().numpy())
        y_traj.append(y.detach().cpu().numpy())

        for _ in range(self.N):
            dW = torch.randn(n_sim, self.dim_W, device=self.device) * self.dt**0.5
            y = self.forward_dynamics(y, q, dW, t, self.dt)
            t += self.dt

            Y = self.Y_net(t, y)
            q = self.trading_rate(t, y, Y, create_graph=False)

            q_traj.append(q.detach().cpu().numpy())
            Y_traj.append(Y.detach().cpu().numpy())
            y_traj.append(y.detach().cpu().numpy())

        return torch.linspace(0, self.T, self.N + 1).cpu().numpy(), {
            "q": np.stack(q_traj),
            "Y": np.stack(Y_traj),
            "y": np.stack(y_traj)
        }

    # def forward_supervised(self, t_paths, W_paths):
    #     """
    #     Two-phase supervised training:
    #     Phase 1: simulate forward paths using q = trading_rate
    #     Phase 2: compute analytic and predicted value + gradient loss on the whole trajectory
    #     """
    #     batch_size = self.batch_size
    #     dt = self.T / self.N
    #     device = self.device

    #     # Initial state
    #     t0 = t_paths[:, 0, :]
    #     W0 = W_paths[:, 0, :]
    #     y0 = self.y0.repeat(batch_size, 1).to(device)

    #     # Collect full trajectories
    #     y_path = [y0]
    #     t_path = [t0]

    #     for n in range(self.N):
    #         t1 = t_paths[:, n + 1, :]
    #         W1 = W_paths[:, n + 1, :]

    #         Y0 = self.Y_net(t0, y0)
    #         q = self.trading_rate(t0, y0, Y0)
    #         y1 = self.forward_dynamics(y0, q, W1 - W0, t0, t1 - t0)

    #         y_path.append(y1)
    #         t_path.append(t1)

    #         t0, W0, y0 = t1, W1, y1

    #     # Phase 2: compare with analytic solution
    #     Y_loss = 0.0
    #     dY_loss = 0.0
    #     for i, (t_n, y_n_orig) in enumerate(zip(t_path, y_path)):
    #         # Detach and re-enable autograd cleanly
    #         y_n = y_n_orig.detach()
    #         y_n.requires_grad_(True)

    #         t_n = t_n.detach()  # just to be safe
    #         t_n = t_n.requires_grad_(False)

    #         # Compute V_pred and V_target
    #         V_target = self.value_function_analytic(t_n, y_n).detach()  # no need to keep graph
    #         V_pred = self.Y_net(t_n, y_n)

    #         # Compute gradients
    #         y1_ = y1.clone().detach().requires_grad_(True)
    #         V_temp = self.value_function_analytic(t1, y1_)
    #         dV_target = torch.autograd.grad(
    #             outputs=V_temp,
    #             inputs=y1_,
    #             grad_outputs=torch.ones_like(V_temp),
    #             create_graph=False
    #         )[0]
            
    #         dV_pred = torch.autograd.grad(
    #             outputs=V_pred,
    #             inputs=y_n,
    #             grad_outputs=torch.ones_like(V_pred),
    #             create_graph=True,
    #             retain_graph=True
    #         )[0]

    #         loss_V = torch.mean((V_pred - V_target) ** 2)
    #         loss_dV = torch.mean((dV_pred - dV_target) ** 2)
    #         Y_loss += loss_V
    #         dY_loss += loss_dV

    #     self.total_Y_loss = self.λ_Y * Y_loss.detach().item()
    #     self.terminal_loss = self.λ_dY * dY_loss.detach().item()
    #     self.terminal_gradient_loss = 0.0

    #     return self.λ_Y * Y_loss + self.λ_dY * dY_loss

    # def forward_supervised(self, t_paths, W_paths):
    #     batch_size = self.batch_size
    #     t0 = t_paths[:, 0, :]
    #     W0 = W_paths[:, 0, :]
    #     y0 = self.y0.repeat(batch_size, 1).to(self.device)
    #     Y0 = self.Y_net(t0, y0)

    #     Y_loss = 0.0

    #     for n in range(self.N):
    #         t1 = t_paths[:, n + 1, :]
    #         W1 = W_paths[:, n + 1, :]

    #         q = self.trading_rate(t0, y0, Y0)
    #         y1 = self.forward_dynamics(y0, q, W1 - W0, t0, t1 - t0)

    #         # --- Supervised loss with analytic V and dV at t1, y1 ---
    #         V_target = self.value_function_analytic(t1, y1.detach())

    #         y1_ = y1.clone().detach().requires_grad_(True)
    #         V_temp = self.value_function_analytic(t1, y1_)
    #         dV_target = torch.autograd.grad(
    #             outputs=V_temp,
    #             inputs=y1_,
    #             grad_outputs=torch.ones_like(V_temp),
    #             create_graph=False
    #         )[0]

    #         V_pred = self.Y_net(t1, y1)
    #         supervised_loss = torch.mean(torch.pow(V_pred - V_target, 2))

    #         dV_pred = torch.autograd.grad(
    #             outputs=V_pred,
    #             inputs=y1,
    #             grad_outputs=torch.ones_like(V_pred),
    #             create_graph=True
    #         )[0]
    #         gradient_loss = torch.mean(torch.pow(dV_pred - dV_target, 2))

    #         Y_loss += supervised_loss + gradient_loss

    #         # Advance
    #         t0, W0, y0, Y0 = t1, W1, y1, V_pred

    #     t_terminal = torch.full((batch_size, 1), self.T, device=self.device)
    #     YT = self.Y_net(t_terminal, y1)
    #     terminal = self.terminal_cost(y1)
    #     terminal_loss = torch.mean(torch.pow(YT - terminal, 2))

    #     dYT = torch.autograd.grad(
    #         outputs=YT,
    #         inputs=y1,
    #         grad_outputs=torch.ones_like(YT),
    #         create_graph=True,
    #         retain_graph=True
    #     )[0]
    #     terminal_gradient = self.terminal_cost_grad(y1)
    #     terminal_gradient_loss = torch.mean(torch.pow(dYT - terminal_gradient, 2))

    #     # Log and return
    #     self.total_Y_loss = self.λ_Y * Y_loss.detach().item()
    #     self.terminal_loss = 0 # self.λ_T * terminal_loss.detach().item()
    #     self.terminal_gradient_loss = 0 # self.λ_TG * terminal_gradient_loss.detach().item()

    #     return self.λ_Y * Y_loss + self.λ_T * terminal_loss + self.λ_TG * terminal_gradient_loss

    def forward_supervised(self, t_paths, W_paths):
        batch_size = self.batch_size
        dim = self.dim

        t = torch.rand(batch_size, 1, device=self.device) * self.T
        y = torch.randn(batch_size, dim, device=self.device, requires_grad=True)

        # -- Get V_target analytically (no need for autograd)
        V_target = self.value_function_analytic(t, y.detach())

        # -- Compute dV_target using autograd
        y_ = y.clone().detach().requires_grad_(True)
        V_temp = self.value_function_analytic(t, y_)
        dV_target = torch.autograd.grad(
            outputs=V_temp,
            inputs=y_,
            grad_outputs=torch.ones_like(V_temp),
            create_graph=False
        )[0]

        # -- Predict and compute losses
        V_pred = self.Y_net(t, y)
        supervised_loss = torch.mean((V_pred - V_target)**2)

        dV_pred = torch.autograd.grad(
            outputs=V_pred,
            inputs=y,
            grad_outputs=torch.ones_like(V_pred),
            create_graph=True
        )[0]

        gradient_loss = torch.mean((dV_pred - dV_target)**2)
        Y_loss = supervised_loss + gradient_loss

        # Terminal condition
        t_terminal = torch.full((batch_size, 1), self.T, device=self.device)
        YT = self.Y_net(t_terminal, y)
        terminal = self.terminal_cost(y)
        terminal_loss = torch.mean(torch.pow(YT - terminal, 2))

        dYT = torch.autograd.grad(
            outputs=YT,
            inputs=y,
            grad_outputs=torch.ones_like(YT),
            create_graph=True,
            retain_graph=True
        )[0]
        terminal_gradient = self.terminal_cost_grad(y)
        terminal_gradient_loss = torch.mean(torch.pow(dYT - terminal_gradient, 2))

        self.λ_T, self.λ_TG = 0, 0
        self.total_Y_loss = self.λ_Y * Y_loss.detach().item()
        self.terminal_loss = self.λ_T * terminal_loss.detach().item()
        self.terminal_gradient_loss = self.λ_TG * terminal_gradient_loss.detach().item()

        return self.λ_Y * Y_loss + self.λ_T * terminal_loss + self.λ_TG * terminal_gradient_loss

    def plot_approx_vs_analytic(self, results, timesteps, plot=True, save_dir=None):
        approx_q = results["q"]
        y_vals = results["y"]
        Y_vals = results["Y"]

        T, N_paths = y_vals.shape[:2]

        with torch.no_grad():
            t_grid = torch.linspace(0, self.T, self.N + 1, device=self.device).view(self.N + 1, 1).expand(self.N + 1, N_paths)  # shape: (N + 1, N_paths)
            y_tensor = torch.tensor(results["y"], dtype=torch.float32, device=self.device)          # shape: (N + 1, N_paths, dim)
            flat_y = y_tensor.reshape(-1, self.dim)                   # (N + 1) * n_sim, dim
            flat_t = t_grid.reshape(-1, 1).expand_as(flat_y[:, :1])   # match shape: (N + 1) * n_sim, 1
            true_q = self.optimal_control_analytic(flat_t, flat_y).view(self.N + 1, N_paths)
            true_Y = self.value_function_analytic(flat_t, flat_y).view(self.N + 1, N_paths)

        true_q = true_q.cpu().numpy()
        true_Y = true_Y.cpu().numpy()

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
            diff = approx_q[:, i].squeeze() - true_q[:, i].squeeze()
            axs[0, 1].plot(timesteps, diff, color=colors(i), alpha=0.6, label=f"$q_{i}(t) - q^*_{i}(t)$" if i == 0 else None)
        axs[0, 1].axhline(0, color='red', linestyle='--', linewidth=0.8)
        axs[0, 1].set_title("Difference: Learned $-$ Analytical")
        axs[0, 1].set_xlabel("Time $t$")
        axs[0, 1].set_ylabel("$q(t) - q^*(t)$")
        axs[0, 1].grid(True)
        axs[0, 1].legend(loc='upper left')

        for i in range(Y_vals.shape[1]):
            axs[1, 0].plot(timesteps, Y_vals[:, i, 0], color=colors(i), alpha=0.6, label=f"Learned $Y_{i}(t)$" if i == 0 else None)
            axs[1, 0].plot(timesteps, true_Y[:, i], linestyle="--", color=colors(i), alpha=0.4, label=f"Analytical $Y^*_{i}(t)$" if i == 0 else None)
        axs[1, 0].set_title("Cost-to-Go $Y(t)$")
        axs[1, 0].set_xlabel("Time $t$")
        axs[1, 0].set_ylabel("Y(t)")
        axs[1, 0].grid(True)
        axs[1, 0].legend(loc='upper left')

        for i in range(Y_vals.shape[1]):
            diff_Y = Y_vals[:, i, 0] - true_Y[:, i]
            axs[1, 1].plot(timesteps, diff_Y, color=colors(i), alpha=0.6, label=f"$Y_{i}(t) - Y^*_{i}(t)$" if i == 0 else None)
        axs[1, 1].axhline(0, color='red', linestyle='--', linewidth=0.8)
        axs[1, 1].set_title("Difference: Learned $Y(t) - Y^*(t)$")
        axs[1, 1].set_xlabel("Time $t$")
        axs[1, 1].set_ylabel("$Y(t) - Y^*(t)$")
        axs[1, 1].grid(True)
        axs[1, 1].legend(loc='upper left')

        x_star = np.zeros_like(true_q)
        x_star[0] = y_vals[0, :, 0]
        for n in range(1, self.N + 1):
            x_star[n] = x_star[n - 1] + true_q[n - 1] * self.dt

        for i in range(y_vals.shape[1]):
            axs[2, 0].plot(timesteps, y_vals[:, i, 0], color=colors(i), alpha=0.6, label=f"$x_{i}(t)$" if i == 0 else None)
            axs[2, 0].plot(timesteps, x_star[:, i], linestyle="--", color=colors(i), alpha=0.4, label=f"$x^*_{i}(t)$" if i == 0 else None)
            axs[2, 0].plot(timesteps, y_vals[:, i, 2], linestyle="-.", color=colors(i), alpha=0.6, label=f"$d_{i}(t)$" if i == 0 else None)
        axs[2, 0].set_title("States")
        axs[2, 0].set_xlabel("Time $t$")
        axs[2, 0].set_ylabel("x(t)/p(t)/d(t)")
        axs[2, 0].grid(True)
        axs[2, 0].legend(loc='upper left')

        for i in range(y_vals.shape[1]):
            axs[2, 1].plot(timesteps, y_vals[:, i, 1], color=colors(i), alpha=0.6, label=f"$p_{i}(t)$" if i == 0 else None)
        axs[2, 1].set_title("States")
        axs[2, 1].set_xlabel("Time $t$")
        axs[2, 1].set_ylabel("x(t)/p(t)/d(t)")
        axs[2, 1].grid(True)
        axs[2, 1].legend(loc='upper left')

        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/approx_vs_analytic.png", dpi=300, bbox_inches='tight')
        if plot:
            plt.show()

    def plot_approx_vs_analytic_expectation(self, results, timesteps, plot=True, save_dir=None):
        approx_q = results["q"]
        y_vals = results["y"]
        Y_vals = results["Y"]

        T, N_paths = y_vals.shape[:2]

        with torch.no_grad():
            t_grid = torch.linspace(0, self.T, self.N + 1, device=self.device).view(self.N + 1, 1).expand(self.N + 1, N_paths)  # shape: (N + 1, N_paths)
            y_tensor = torch.tensor(results["y"], dtype=torch.float32, device=self.device)          # shape: (N + 1, N_paths, dim)
            flat_y = y_tensor.reshape(-1, self.dim)                   # (N + 1) * n_sim, dim
            flat_t = t_grid.reshape(-1, 1).expand_as(flat_y[:, :1])   # match shape: (N + 1) * n_sim, 1
            true_q = self.optimal_control_analytic(flat_t, flat_y).view(self.N + 1, N_paths)
            true_Y = self.value_function_analytic(flat_t, flat_y).view(self.N + 1, N_paths)

        true_q = true_q.cpu().numpy()
        true_Y = true_Y.cpu().numpy()

        # Learned results
        mean_q = approx_q.mean(axis=1).squeeze()
        std_q = approx_q.std(axis=1).squeeze()
        mean_Y = Y_vals[:, :, 0].mean(axis=1).squeeze()
        std_Y = Y_vals[:, :, 0].std(axis=1).squeeze()

        # Analytic results
        mean_q_true = true_q.mean(axis=1).squeeze()
        std_q_true = true_q.std(axis=1).squeeze()
        mean_true_Y = true_Y.mean(axis=1).squeeze()
        std_true_Y = true_Y.std(axis=1).squeeze()

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        axs[0, 0].plot(timesteps, mean_q, label='Learned Mean', color='blue')
        axs[0, 0].fill_between(timesteps, mean_q - std_q, mean_q + std_q, color='blue', alpha=0.3, label='Learned ±1 Std')
        axs[0, 0].plot(timesteps, mean_q_true, label='Analytical Mean', color='black', linestyle='--')
        axs[0, 0].fill_between(timesteps, mean_q_true - std_q_true, mean_q_true + std_q_true, color='black', alpha=0.2, label='Analytical ±1 Std')
        axs[0, 0].set_title("Control $q(t)$: Learned vs Analytical")
        axs[0, 0].set_xlabel("Time $t$")
        axs[0, 0].set_ylabel("$q(t)$")
        axs[0, 0].grid(True)
        axs[0, 0].legend(loc='upper left')

        diff = (approx_q.squeeze(-1) - true_q)
        mean_diff = np.mean(diff, axis=1)
        std_diff = np.std(diff, axis=1)
        axs[0, 1].fill_between(timesteps, mean_diff - std_diff, mean_diff + std_diff, color='red', alpha=0.4, label='±1 Std Dev')
        axs[0, 1].plot(timesteps, mean_diff, color='red', label='Mean Difference')
        axs[0, 1].set_title("Difference: Learned $-$ Analytical")
        axs[0, 1].set_xlabel("Time $t$")
        axs[0, 1].set_ylabel("$q(t) - q^*(t)$")
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

        diff_Y = (Y_vals[:, :, 0] - true_Y)
        mean_diff_Y = np.mean(diff_Y, axis=1)
        std_diff_Y = np.std(diff_Y, axis=1)
        axs[1, 1].fill_between(timesteps, mean_diff_Y - std_diff_Y, mean_diff_Y + std_diff_Y, color='red', alpha=0.4, label='±1 Std Dev')
        axs[1, 1].plot(timesteps, mean_diff_Y, color='red', label='Mean Difference')
        axs[1, 1].set_title("Difference: Learned $Y(t) - Y^*(t)$")
        axs[1, 1].set_xlabel("Time $t$")
        axs[1, 1].set_ylabel("$Y(t) - Y^*(t)$")
        axs[1, 1].grid(True)
        axs[1, 1].legend(loc='upper left')

        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/approx_vs_analytic_expectation.png", dpi=300, bbox_inches='tight')
        if plot:
            plt.show()
        
    def plot_terminal_histogram(self, results, plot=True, save_dir=None):
        y_vals = results["y"]  # shape: (T+1, N_paths, dim)
        Y_vals = results["Y"]  # shape: (T+1, N_paths, 1)

        Y_T_approx = Y_vals[-1, :, 0]
        y_T = y_vals[-1, :, :]  # full final states
        y_T_tensor = torch.tensor(y_T, dtype=torch.float32, device=self.device)
        Y_T_true = self.terminal_cost(y_T_tensor).detach().cpu().numpy().squeeze()

        # Filter out NaN or Inf
        mask = np.isfinite(Y_T_approx) & np.isfinite(Y_T_true)
        Y_T_approx = Y_T_approx[mask]
        Y_T_true = Y_T_true[mask]

        if len(Y_T_approx) == 0 or len(Y_T_true) == 0:
            print("Warning: No valid terminal values to plot.")
            return

        plt.figure(figsize=(8, 6))
        bins = 30
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
        if save_dir:
            plt.savefig(f"{save_dir}/terminal_histogram.png", dpi=300, bbox_inches='tight')
        if plot:
            plt.show()