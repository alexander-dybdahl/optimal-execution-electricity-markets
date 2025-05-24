import torch
import torch.nn as nn
import matplotlib.pyplot as plt
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
            dZ = torch.randn(batch_size, self.dim_W, device=self.device) * self.dt**0.5
            y = self.forward_dynamics(y, q, dW, t, self.dt)
            Y_next = Y - f * self.dt + (z * dW).sum(dim=1, keepdim=True)
            residual = Y_next - (Y - f * self.dt + (z * dZ).sum(dim=1, keepdim=True))
            total_residual_loss += torch.mean(residual**2)
            Y = Y_next
            t += self.dt

        terminal = self.terminal_cost(y)
        terminal_loss = torch.mean((Y - terminal)**2)
        return terminal_loss + total_residual_loss

    def train_model(self, epochs=1000, lr=1e-3, save_path="models/saved/model.pth", verbose=True, plot=True):
        self.device = next(self.parameters()).device
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        losses = []

        self.train()

        header_printed = False
    
        init_time = time.time()
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

        if verbose:
            print("-" * 70)
            print(f"Training completed. Lowest loss: {self.lowest_loss:.6f}. Total time: {time.time() - init_time:.2f} seconds")
            print(f"Model saved to {save_path}")

        if plot:
            plt.plot(losses, label="Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training Loss")
            plt.legend()
            plt.grid()
            plt.show()

        return losses
