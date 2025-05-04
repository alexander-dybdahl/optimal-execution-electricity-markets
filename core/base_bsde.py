import torch
import torch.nn as nn
import numpy as np
import time
from abc import ABC, abstractmethod

class ZNetwork(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 64),
            nn.ReLU(), nn.Linear(64, 64),
            nn.ReLU(), nn.Linear(64, 3)
        )

    def forward(self, t, y):
        return self.net(torch.cat([t, y], dim=1))

class QNetwork(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 64),
            nn.ReLU(), nn.Linear(64, 64),
            nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, t, y):
        return self.net(torch.cat([t, y], dim=1))


class BaseDeepBSDE(nn.Module, ABC):
    def __init__(self, args, model_cfg):
        super().__init__()
        self.device = args.device
        self.batch_size = args.batch_size
        self.y0 = torch.tensor([model_cfg["y0"]], device=self.device)
        self.dim = model_cfg["dim"]
        self.dim_W = model_cfg["dim_W"]
        self.T = model_cfg["T"]
        self.N = model_cfg["N"]
        self.dt = model_cfg["dt"]
        self.Y0 = nn.Parameter(torch.tensor([[0.0]], device=self.device))
        self.z_net = ZNetwork(self.dim).to(self.device)
        self.q_net = QNetwork(self.dim).to(self.device)
        self.lowest_loss = float("inf")

    @abstractmethod
    def generator(self, y, q): pass

    @abstractmethod
    def terminal_cost(self, y): pass

    @abstractmethod
    def mu(self, t, y, q): pass  # shape: (batch, dim)

    @abstractmethod
    def sigma(self, t, y): pass  # shape: (batch, dim, dW_dim)

    def forward_dynamics(self, y, q, dW, t, dt):
        μ = self.mu(t, y, q)                  # shape: (batch, dim)
        σ = self.sigma(t, y)                  # shape: (batch, dim, dW_dim)
        diffusion = torch.bmm(σ, dW.unsqueeze(-1)).squeeze(-1)  # shape: (batch, dim)
        return y + μ * dt + diffusion         # shape: (batch, dim)

    def forward(self):
        batch_size = self.batch_size
        y = self.y0.repeat(batch_size, 1).to(self.device)
        t = torch.zeros(batch_size, 1, device=self.device)
        Y = self.Y0.repeat(batch_size, 1)
        total_residual_loss = 0.0

        for _ in range(self.N):
            z = self.z_net(t, y)
            q = self.q_net(t, y)
            f = self.generator(y, q)
            dW = torch.randn(batch_size, self.dim_W, device=self.device) * self.dt**0.5
            y = self.forward_dynamics(y, q, dW, t, self.dt)
            Y_next = Y - f * self.dt + (z * dW).sum(dim=1, keepdim=True)
            with torch.no_grad():
                residual = Y_next.detach() - (Y - f * self.dt + (z * dW).sum(dim=1, keepdim=True))
            total_residual_loss += torch.mean(residual**2)
            Y = Y_next
            t += self.dt

        terminal = self.terminal_cost(y)
        terminal_loss = torch.mean((Y - terminal)**2)
        return terminal_loss + total_residual_loss

    def train_model(self, epochs=1000, lr=1e-3, save_path="model.pth", verbose=True):
        self.device = next(self.parameters()).device
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        losses = []

        self.train()

        header_printed = False
    
        start_time = time.time()
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = self()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if (epoch % 50 == 0 or epoch == epochs - 1) and verbose:
                elapsed = time.time() - start_time
                if not header_printed:
                    print(f"{'Epoch':>8} | {'Loss':>12} | {'Memory [MB]':>12} | {'Time [s]':>10} | {'Status'}")
                    print("-" * 70)
                    header_printed = True

                mem_mb = torch.cuda.memory_allocated() / 1e6
                status = ""
                if loss.item() < self.lowest_loss:
                    self.lowest_loss = loss.item()
                    torch.save(self.state_dict(), save_path)
                    status = "Model saved ↓"

                print(f"{epoch:8} | {loss.item():12.6f} | {mem_mb:12.2f} | {elapsed:10.2f} | {status}")
                start_time = time.time()  # Reset timer for next block

        return losses

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
