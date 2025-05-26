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
        self.nu = model_cfg["nu"]        # perm impact and terminal penalty
        self.mu_D = model_cfg["mu_D"]     # drift for D

    def generator(self, y, q):
        P = y[:, 1:2]
        return q * P + self.gamma * q**2

    def terminal_cost(self, y):
        D = y[:, 2:3]
        X = y[:, 0:1]
        return 0.5 * self.nu * (D - X)**2

    def mu(self, t, y, q):
        X = y[:, 0:1]
        P = y[:, 1:2]
        D = y[:, 2:3]

        dX = q
        dP = self.nu * q
        dD = self.mu_D * torch.ones_like(D)

        return torch.cat([dX, dP, dD], dim=1)

    def sigma(self, t, y):
        batch = y.shape[0]

        Sigma = torch.zeros(batch, 3, 2, device=self.device)
        Sigma[:, 1, 0] = self.sigma_P                          # dP = ... dW1
        Sigma[:, 2, 0] = self.rho * self.sigma_D               # dD = ... dW1
        Sigma[:, 2, 1] = (1 - self.rho**2)**0.5 * self.sigma_D # dD = ... dW2
        return Sigma

    def trading_rate(self, t, y, Y):

        dV = torch.autograd.grad(
            outputs=Y,
            inputs=y,
            grad_outputs=torch.ones_like(Y),
            create_graph=True
        )[0]  # shape: [batch, dim] = [batch, 3]

        dV_X = dV[:, 0:1]
        dV_P = dV[:, 1:2]
        P = y[:, 1:2]

        q = -0.5 / self.gamma * (P + dV_X + self.nu * dV_P)  # shape: [batch, 1]

        return q

    def optimal_control_analytic(self, t, y):
        D = y[:, 2:3]
        P = y[:, 1:2]
        num = self.eta * (self.mu_D * t + D) - P
        denom = (self.eta + self.nu) * t + 2 * self.gamma
        return num / denom

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
        eps = 1e-8
        denom = (eta + nu) * tau + 2 * gamma

        term1 = (eta * (0.5 * nu * tau + gamma) / denom) * ((D - X)**2 + 2 * mu * tau * (D - X))
        term2 = (tau / denom) * (-0.5 * P**2 + eta * mu * tau * P)
        term3 = (eta * tau / denom) * (D - X) * P

        log_term = gamma * (sigma_0**2 + sigma_d**2 * eta**2 - 2 * rho * sigma_0 * sigma_d * eta) / (eta + nu)**2
        log_expr = 1 + ((eta + nu) * tau) / (2 * gamma)
        term4 = log_term * torch.log(torch.tensor(log_expr, device=self.device))

        term5 = (sigma_d**2 * eta * nu + 2 * rho * sigma_0 * sigma_d * eta - sigma_0**2) / (2 * (eta + nu)) * tau

        term6 = (eta * mu**2 * tau**2 * (0.5 * nu * tau + gamma)) / denom

        return term1 + term2 + term3 + term4 + term5 + term6

    def simulate_paths(self, n_sim=5, seed=42, y0_single=None):
        torch.manual_seed(seed)
        self.eval()

        y = y0_single.repeat(n_sim, 1) if y0_single is not None else self.y0.repeat(n_sim, 1)
        t = torch.zeros(n_sim, 1, device=self.device)

        q_traj, Y_traj, y_traj = [], [], []

        Y = self.Y_net(t, y)
        dY = torch.autograd.grad(Y, y, torch.ones_like(Y), create_graph=False)[0]
        q = -0.5 * dY[:, 0:1]  # control acts on X

        q_traj.append(q.detach().cpu().numpy())
        Y_traj.append(Y.detach().cpu().numpy())
        y_traj.append(y.detach().cpu().numpy())

        for _ in range(self.N):
            dW = torch.randn(n_sim, self.dim_W, device=self.device) * self.dt**0.5
            y = self.forward_dynamics(y, q, dW, t, self.dt)
            t += self.dt

            Y = self.Y_net(t, y)
            dY = torch.autograd.grad(Y, y, torch.ones_like(Y), create_graph=False)[0]
            q = -0.5 * dY[:, 0:1]

            q_traj.append(q.detach().cpu().numpy())
            Y_traj.append(Y.detach().cpu().numpy())
            y_traj.append(y.detach().cpu().numpy())

        return torch.linspace(0, self.T, self.N + 1).cpu().numpy(), {
            "q": np.stack(q_traj),
            "Y": np.stack(Y_traj),
            "y": np.stack(y_traj)
        }

    def plot_approx_vs_analytic(self, results, timesteps):
        approx_q = results["q"]
        y_vals = results["y"]
        Y_vals = results["Y"]

        with torch.no_grad():
            t_grid = torch.tensor(timesteps, dtype=torch.float32, device=self.device).unsqueeze(1).repeat(1, y_vals.shape[1])
            y_tensor = torch.tensor(y_vals, dtype=torch.float32, device=self.device)
            true_q = self.optimal_control_analytic(t_grid.flatten(), y_tensor.reshape(-1, y_tensor.shape[-1])).view(*t_grid.shape)
            true_Y = self.value_function_analytic(t_grid.flatten(), y_tensor.reshape(-1, y_tensor.shape[-1])).view(*t_grid.shape)

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        colors = cm.get_cmap("tab10", approx_q.shape[1])

        for i in range(approx_q.shape[1]):
            axs[0, 0].plot(timesteps, approx_q[:, i], color=colors(i), alpha=0.6)
            axs[0, 0].plot(timesteps, true_q[:, i].cpu(), linestyle="--", color=colors(i), alpha=0.4)
        axs[0, 0].set_title("Control $q(t)$: Learned vs Analytical")
        axs[0, 0].set_xlabel("Time $t$")
        axs[0, 0].set_ylabel("$q(t)$")
        axs[0, 0].grid(True)

        for i in range(approx_q.shape[1]):
            diff = approx_q[:, i] - true_q[:, i].cpu()
            axs[0, 1].plot(timesteps, diff, color=colors(i), alpha=0.6)
        axs[0, 1].axhline(0, color='red', linestyle='--', linewidth=0.8)
        axs[0, 1].set_title("Difference: Learned $-$ Analytical")
        axs[0, 1].set_xlabel("Time $t")
        axs[0, 1].set_ylabel("$q(t) - q^*(t)$")
        axs[0, 1].grid(True)

        for i in range(Y_vals.shape[1]):
            axs[1, 0].plot(timesteps, Y_vals[:, i, 0], color=colors(i), alpha=0.6)
            axs[1, 0].plot(timesteps, true_Y[:, i].cpu(), linestyle="--", color=colors(i), alpha=0.4)
        axs[1, 0].set_title("Cost-to-Go $Y(t)$")
        axs[1, 0].set_xlabel("Time $t")
        axs[1, 0].set_ylabel("Y(t)")
        axs[1, 0].grid(True)

        for i in range(y_vals.shape[1]):
            axs[1, 1].plot(timesteps, y_vals[:, i, 0], color=colors(i), alpha=0.6)
        axs[1, 1].set_title("State $x(t)$")
        axs[1, 1].set_xlabel("Time $t")
        axs[1, 1].set_ylabel("x(t)")
        axs[1, 1].grid(True)

        plt.tight_layout()
        plt.show()

    def plot_approx_vs_analytic_expectation(self, results, timesteps):
        approx_q = results["q"]
        y_vals = results["y"]
        Y_vals = results["Y"]

        with torch.no_grad():
            t_grid = torch.tensor(timesteps, dtype=torch.float32, device=self.device).unsqueeze(1).repeat(1, y_vals.shape[1])
            y_tensor = torch.tensor(y_vals, dtype=torch.float32, device=self.device)
            true_q = self.optimal_control_analytic(t_grid.flatten(), y_tensor.reshape(-1, y_tensor.shape[-1])).view(*t_grid.shape)
            true_Y = self.value_function_analytic(t_grid.flatten(), y_tensor.reshape(-1, y_tensor.shape[-1])).view(*t_grid.shape)

        mean_Y = Y_vals[:, :, 0].mean(axis=1).squeeze()
        std_Y = Y_vals[:, :, 0].std(axis=1).squeeze()
        mean_true_Y = true_Y.mean(dim=1).cpu().numpy().squeeze()
        std_true_Y = true_Y.std(dim=1).cpu().numpy().squeeze()
        mean_q = approx_q.mean(axis=1).squeeze()
        std_q = approx_q.std(axis=1).squeeze()
        mean_q_true = true_q.mean(dim=1).cpu().numpy().squeeze()
        std_q_true = true_q.std(dim=1).cpu().numpy().squeeze()

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        axs[0, 0].plot(timesteps, mean_q, label='Learned Mean', color='blue')
        axs[0, 0].fill_between(timesteps, mean_q - std_q, mean_q + std_q, color='blue', alpha=0.3, label='Learned ±1 Std')
        axs[0, 0].plot(timesteps, mean_q_true, label='Analytical Mean', color='black', linestyle='--')
        axs[0, 0].fill_between(timesteps, mean_q_true - std_q_true, mean_q_true + std_q_true, color='black', alpha=0.2, label='Analytical ±1 Std')
        axs[0, 0].set_title("Control $q(t)$: Learned vs Analytical")
        axs[0, 0].set_xlabel("Time $t")
        axs[0, 0].set_ylabel("$q(t)$")
        axs[0, 0].grid(True)
        axs[0, 0].legend()

        diff = (approx_q.squeeze(-1) - true_q.cpu().numpy())
        mean_diff = np.mean(diff, axis=1)
        std_diff = np.std(diff, axis=1)
        axs[0, 1].fill_between(timesteps, mean_diff - std_diff, mean_diff + std_diff, color='red', alpha=0.4, label='±1 Std Dev')
        axs[0, 1].plot(timesteps, mean_diff, color='red', label='Mean Difference')
        axs[0, 1].set_title("Difference: Learned $-$ Analytical")
        axs[0, 1].set_xlabel("Time $t")
        axs[0, 1].set_ylabel("$q(t) - q^*(t)$")
        axs[0, 1].grid(True)
        axs[0, 1].legend()

        axs[1, 0].plot(timesteps, mean_Y, color='blue', label='Learned Mean')
        axs[1, 0].fill_between(timesteps, mean_Y - std_Y, mean_Y + std_Y, color='blue', alpha=0.3, label='Learned ±1 Std')
        axs[1, 0].plot(timesteps, mean_true_Y, color='black', linestyle='--', label='Analytical Mean')
        axs[1, 0].fill_between(timesteps, mean_true_Y - std_true_Y, mean_true_Y + std_true_Y, color='black', alpha=0.2, label='Analytical ±1 Std')
        axs[1, 0].set_title("Cost-to-Go $Y(t)$")
        axs[1, 0].set_xlabel("Time $t")
        axs[1, 0].set_ylabel("Y(t)")
        axs[1, 0].grid(True)
        axs[1, 0].legend()

        diff_Y = (Y_vals[:, :, 0] - true_Y.cpu().numpy())
        mean_diff_Y = np.mean(diff_Y, axis=1)
        std_diff_Y = np.std(diff_Y, axis=1)
        axs[1, 1].fill_between(timesteps, mean_diff_Y - std_diff_Y, mean_diff_Y + std_diff_Y, color='red', alpha=0.4, label='±1 Std Dev')
        axs[1, 1].plot(timesteps, mean_diff_Y, color='red', label='Mean Difference')
        axs[1, 1].set_title("Difference: Learned $Y(t) - Y^*(t)$")
        axs[1, 1].set_xlabel("Time $t")
        axs[1, 1].set_ylabel("$Y(t) - Y^*(t)$")
        axs[1, 1].grid(True)
        axs[1, 1].legend()

        plt.tight_layout()
        plt.show()
        
    def plot_terminal_histogram(self, results):
        y_vals = results["y"][:, :, 0]  # X_T
        Y_vals = results["Y"]  # shape: (T+1, N_paths, 1)

        Y_T_approx = Y_vals[-1, :, 0]
        y_T = y_vals[-1, :]
        Y_T_true = self.terminal_cost(torch.tensor(y_T, device=self.device).unsqueeze(1)).cpu().numpy()

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
        plt.show()

    def forward_supervised(self, t_paths, W_paths):
        batch_size = self.batch_size
        dim = self.dim

        t = torch.rand(batch_size, 1, device=self.device) * self.T
        y = torch.randn(batch_size, dim, device=self.device, requires_grad=True)

        with torch.no_grad():
            V_target = self.value_function_analytic(t, y)
            dV_target = torch.autograd.grad(
                outputs=V_target,
                inputs=y,
                grad_outputs=torch.ones_like(V_target),
                create_graph=False
            )[0]

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

        # Terminal losses
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
