import torch
import numpy as np
from core.fbsnn import FBSNN
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class HJB(FBSNN):
    def __init__(self, args, model_cfg):
        super().__init__(args, model_cfg)
        self.xi = torch.tensor(model_cfg["xi"], device=self.device)
        self.gamma = model_cfg["gamma"]
        self.eta = model_cfg["eta"]
        self.mu_P = model_cfg["mu_P"]
        self.rho = model_cfg["rho"]
        self.vol_P = model_cfg["vol_P"]
        self.vol_D = model_cfg["vol_D"]
        self.vol_B = model_cfg["vol_B"]
        self.dim_W = model_cfg["dim_W"]

    def psi(self, q):
        return self.gamma * q

    def phi(self, q):
        return self.eta * q

    def sigma_P(self, t):
        return self.vol_P * torch.ones_like(t, device=self.device)

    def sigma_D(self, t):
        return self.vol_D * torch.ones_like(t, device=self.device) # (self.T - t) / self.T

    def sigma_B(self, t):
        return self.vol_B * torch.ones_like(t, device=self.device) # (self.T - t) / self.T

    def generator(self, y, q):
        P = y[:, 1:2]
        sign_q = torch.sign(q)
        P_exec = P + sign_q * self.psi(q) + self.phi(q)
        return -q * P_exec

    def terminal_cost(self, y):
        X = y[:, 0]
        D = y[:, 2]
        B = y[:, 3]
        I = X - D + self.xi
        I_plus = torch.clamp(I, min=0.0)
        I_minus = torch.clamp(-I, min=0.0)
        return I_plus * B - I_minus * B

    def mu(self, t, y, q):
        dX = q
        dP = self.mu_P + self.gamma * q
        dD = torch.zeros_like(dP)
        dB = torch.zeros_like(dP)
        return torch.cat([dX, dP, dD, dB], dim=1)

    def sigma(self, t, y):
        batch_size = t.shape[0]
        Sigma = torch.zeros(batch_size, 4, 3, device=self.device)
        Sigma[:, 1, 0] = self.sigma_P(t).squeeze()
        Sigma[:, 2, 0] = self.rho * self.sigma_D(t).squeeze()
        Sigma[:, 2, 1] = (1 - self.rho**2) ** 0.5 * self.sigma_D(t).squeeze()
        Sigma[:, 3, 2] = self.sigma_B(t).squeeze()
        return Sigma
    
    def simulate_paths(self, n_sim=1024, seed=42, y0_single=None):
        torch.manual_seed(seed)
        self.eval()

        terminal_stats = {"X": [], "D": [], "B": [], "I": []}

        y = y0_single.repeat(n_sim, 1) if y0_single is not None else self.y0.repeat(n_sim, 1)
        t = torch.zeros(n_sim, 1, device=self.device)
        Y = self.Y0.repeat(n_sim, 1)

        q_traj, Y_traj, y_traj = [], [], []

        for _ in range(self.N):
            t_input = t.clone()
            q = self.q_net(t_input, y).squeeze(-1)
            z = self.z_net(t_input, y)
            f = self.generator(y, q.unsqueeze(-1))
            dW = torch.randn(n_sim, self.dim_W, device=self.device) * self.dt**0.5

            y = self.forward_dynamics(y, q.unsqueeze(-1), dW, t, self.dt)
            Y = Y - f * self.dt + (z * dW).sum(dim=1, keepdim=True)
            t += self.dt

            q_traj.append(q.detach().cpu().numpy())
            Y_traj.append(Y.detach().cpu().numpy())
            y_traj.append(y.detach().cpu().numpy())

        X, D, B = y[:, 0], y[:, 2], y[:, 3]
        I = X - D + self.xi

        terminal_stats["X"].append(X.detach().cpu())
        terminal_stats["D"].append(D.detach().cpu())
        terminal_stats["B"].append(B.detach().cpu())
        terminal_stats["I"].append(I.detach().cpu())

        timesteps = np.linspace(0, self.T, self.N)

        return timesteps, {
            "q": np.concatenate(q_traj, axis=1),
            "Y": np.concatenate(Y_traj, axis=1),
            "final_y": np.concatenate(y_traj, axis=1),
            "X_T": torch.cat(terminal_stats["X"]).squeeze().numpy(),
            "D_T": torch.cat(terminal_stats["D"]).squeeze().numpy(),
            "B_T": torch.cat(terminal_stats["B"]).squeeze().numpy(),
            "I_T": torch.cat(terminal_stats["I"]).squeeze().numpy()
        }

    def plot_all_diagnostics(self, results, timesteps):

        q_vals = results["q"]  # shape: (T, N)
        Y_vals = results["Y"]
        X_T, D_T, B_T, I_T = results["X_T"], results["D_T"], results["B_T"], results["I_T"]

        # Retrieve the state trajectories (assumed shape: (N*T, 4))
        y_all = results["final_y"]  # shape: (T * n_paths, 4)
        n_paths = q_vals.shape[1]
        T = q_vals.shape[0]
        state_trajectories = y_all.reshape(T, -1, 4)

        mean_states = state_trajectories.mean(axis=1)
        std_states = state_trajectories.std(axis=1)
        ci_states = 1.96 * std_states / np.sqrt(n_paths)

        fig, axs = plt.subplots(3, 2, figsize=(16, 14))

        # Subplot 1: Control over time
        mean_q = q_vals.mean(axis=1)
        std_q = q_vals.std(axis=1)
        ci_q = 1.96 * std_q / np.sqrt(n_paths)
        axs[0, 0].plot(timesteps, mean_q, label="Mean $q(t)$")
        axs[0, 0].fill_between(timesteps, mean_q - ci_q, mean_q + ci_q, alpha=0.3, label="95% CI")
        axs[0, 0].set_title("Optimal Trading Rate $q(t)$", pad=10)
        axs[0, 0].set_xlabel("Time $t$")
        axs[0, 0].set_ylabel("$q(t)$")
        axs[0, 0].grid(True)
        axs[0, 0].legend()

        # Subplot 2: Value function over time
        for i in range(Y_vals.shape[1]):  # Iterate over all trajectories
            axs[0, 1].plot(timesteps, Y_vals[:, i, 0], alpha=0.1, color="blue")
        axs[0, 1].set_title("Cost-to-Go $Y(t)$ (Value Function)", pad=10)
        axs[0, 1].set_xlabel("Time $t$")
        axs[0, 1].set_ylabel("$Y(t)$")
        axs[0, 1].grid(True)
        # mean_Y = Y_vals.mean(axis=1).squeeze()
        # std_Y = Y_vals.std(axis=1).squeeze()
        # ci_Y = 1.96 * std_Y / np.sqrt(Y_vals.shape[1])
        # axs[0, 1].plot(timesteps, mean_Y, label="Mean $Y(t)$")
        # axs[0, 1].fill_between(timesteps, mean_Y - ci_Y, mean_Y + ci_Y, alpha=0.3, label="95% CI")
        # axs[0, 1].set_title("Cost-to-Go $Y(t)$ (Value Function)")
        # axs[0, 1].set_xlabel("Time $t$")
        # axs[0, 1].set_ylabel("$Y(t)$")
        # axs[0, 1].grid(True)
        # axs[0, 1].legend()

        # Subplot 3: Scatter of imbalance vs B(T)
        axs[1, 0].scatter(B_T, I_T, alpha=0.3, s=10)
        axs[1, 0].set_xlabel("Terminal Imbalance Price $B(T)$")
        axs[1, 0].set_ylabel("Imbalance $I(T)$")
        axs[1, 0].set_title("Imbalance $I(T)$ vs. Imbalance Price $B(T)$", pad=10)
        axs[1, 0].grid(True)

        # Subplot 4: X(T) vs D(T)
        axs[1, 1].scatter(D_T, X_T, alpha=0.3, s=10)
        corr = np.corrcoef(X_T, D_T)[0, 1]
        axs[1, 1].set_title(f"$X(T)$ vs. $D(T)$ (Corr = {corr:.3f})", pad=10)
        axs[1, 1].set_xlabel("$D(T)$ (Residual Demand)")
        axs[1, 1].set_ylabel("$X(T)$ (Cumulative Position)")
        axs[1, 1].grid(True)

        # Subplot 5: All states with confidence bands
        labels = ["$X(t)$ (Cumulative)", "$P(t)$ (Mid Price)", "$D(t)$ (Demand)", "$B(t)$ (Imbalance Price)"]
        colors = ["blue", "orange", "green", "red"]
        for i in range(4):
            alpha = 0.5 if n_paths < 100 else 0.1
            for j in range(n_paths):
                if j == 0:
                    axs[2, i % 2].plot(timesteps, state_trajectories[:, j, i], label=labels[i], alpha=alpha, color=colors[i])
                else:
                    axs[2, i % 2].plot(timesteps, state_trajectories[:, j, i], alpha=alpha, color=colors[i])

            # axs[2, i % 2].plot(timesteps, mean_states[:, i], label=f"Mean {labels[i]}")
            # axs[2, i % 2].fill_between(timesteps,
            #                            mean_states[:, i] - ci_states[:, i],
            #                            mean_states[:, i] + ci_states[:, i],
            #                            alpha=0.3, label="95% CI")
            axs[2, i % 2].set_title(labels[i], pad=10)
            axs[2, i % 2].set_xlabel("Time $t$")
            axs[2, i % 2].set_ylabel(labels[i])
            axs[2, i % 2].grid(True)
            axs[2, i % 2].legend()

        # plt.tight_layout()
        plt.show()