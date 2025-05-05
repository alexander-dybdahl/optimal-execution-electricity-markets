import torch
import numpy as np
from core.base_bsde import BaseDeepBSDE

class HJB(BaseDeepBSDE):
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
        return self.eta * q ** 2

    def sigma_P(self, t):
        return self.vol_P * t / self.T

    def sigma_D(self, t):
        return self.vol_D * (self.T - t) / self.T

    def sigma_B(self, t):
        return self.vol_B * (self.T - t) / self.T

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
        sigma_tensor = torch.zeros(batch_size, 4, 3, device=self.device)
        sigma_tensor[:, 1, 0] = self.sigma_P(t).squeeze()
        sigma_tensor[:, 2, 0] = self.rho * self.sigma_D(t).squeeze()
        sigma_tensor[:, 2, 1] = (1 - self.rho**2) ** 0.5 * self.sigma_D(t).squeeze()
        sigma_tensor[:, 3, 2] = self.sigma_B(t).squeeze()
        return sigma_tensor
    
    def simulate_paths(self, n_paths=1000, batch_size=256, seed=42, y0_single=None):
        torch.manual_seed(seed)
        self.eval()

        all_q, all_Y, all_y = [], [], []
        terminal_stats = {"X": [], "D": [], "B": [], "I": []}

        for _ in range(n_paths // batch_size):
            y = y0_single.repeat(batch_size, 1) if y0_single is not None else self.y0.repeat(batch_size, 1)
            t = torch.zeros(batch_size, 1, device=self.device)
            Y = self.Y0.repeat(batch_size, 1)

            q_traj, Y_traj, y_traj = [], [], []

            for _ in range(self.N):
                t_input = t.clone()
                q = self.q_net(t_input, y).squeeze(-1)
                z = self.z_net(t_input, y)
                f = self.generator(y, q.unsqueeze(-1))
                dW = torch.randn(batch_size, self.dim_W, device=self.device) * self.dt**0.5

                y = self.forward_dynamics(y, q.unsqueeze(-1), dW, t, self.dt)
                Y = Y - f * self.dt + (z * dW).sum(dim=1, keepdim=True)
                t += self.dt

                q_traj.append(q.detach().cpu().numpy())
                Y_traj.append(Y.detach().cpu().numpy())
                y_traj.append(y.detach().cpu().numpy())

            all_q.append(np.stack(q_traj))
            all_Y.append(np.stack(Y_traj))
            all_y.append(np.stack(y_traj))

            X, D, B = y[:, 0], y[:, 2], y[:, 3]
            I = X - D + self.xi

            terminal_stats["X"].append(X.detach().cpu())
            terminal_stats["D"].append(D.detach().cpu())
            terminal_stats["B"].append(B.detach().cpu())
            terminal_stats["I"].append(I.detach().cpu())

        timesteps = np.linspace(0, self.T, self.N)

        return timesteps, {
            "q": np.concatenate(all_q, axis=1),
            "Y": np.concatenate(all_Y, axis=1),
            "final_y": np.concatenate(all_y, axis=1),
            "X_T": torch.cat(terminal_stats["X"]).squeeze().numpy(),
            "D_T": torch.cat(terminal_stats["D"]).squeeze().numpy(),
            "B_T": torch.cat(terminal_stats["B"]).squeeze().numpy(),
            "I_T": torch.cat(terminal_stats["I"]).squeeze().numpy()
        }
