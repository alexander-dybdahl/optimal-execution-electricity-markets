import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from abc import ABC, abstractmethod
from core.nnets import Sine, FCnet, Resnet


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

class FBSNN(nn.Module, ABC):
    def __init__(self, args, model_cfg):
        super().__init__()
        self.device = args.device
        self.batch_size = args.batch_size
        self.t0 = 0.0
        self.y0 = torch.tensor([model_cfg["y0"]], device=self.device, requires_grad=True)
        self.dim = model_cfg["dim"]
        self.dim_W = model_cfg["dim_W"]
        self.T = model_cfg["T"]
        self.N = model_cfg["N"]
        self.dt = model_cfg["dt"]

        if args.activation == "Sine":
            self.activation = Sine()
        elif args.activation == "ReLU":
            self.activation = nn.ReLU()

        if args.architecture == "Default":
            self.Y_net = FCnet(layers=[2]+[64, 64, 64, 64, 1], activation=self.activation).to(self.device)
            self.q_net = FCnet(layers=[self.dim+1]+[64, 64, 64, 64, 1], activation=self.activation).to(self.device)
        elif args.architecture == "FC":
            self.Y_net = FCnet(layers=[2]+model_cfg["Y_layers"], activation=self.activation).to(self.device)
            self.q_net = FCnet(layers=[self.dim+1]+model_cfg["q_layers"], activation=self.activation).to(self.device)
        elif args.architecture == "NAISnet":
            self.Y_net = Resnet(layers=[2]+model_cfg["Y_layers"], activation=self.activation, stable=True).to(self.device)
            self.q_net = Resnet(layers=[self.dim+1]+model_cfg["q_layers"], activation=self.activation, stable=True).to(self.device)
        elif args.architecture == "Resnet":
            self.Y_net = Resnet(layers=[2]+model_cfg["Y_layers"], activation=self.activation, stable=False).to(self.device)
            self.q_net = Resnet(layers=[self.dim+1]+model_cfg["q_layers"], activation=self.activation, stable=False).to(self.device)
        else:
            raise ValueError(f"Unknown architecture: {args.architecture}")

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
        t = torch.zeros(batch_size, 1, device=self.device)
        y0 = self.y0.repeat(batch_size, 1).to(self.device)
        Y0 = self.Y_net(t, y0)

        dY0 = torch.autograd.grad(
            outputs=Y0,
            inputs=y0,
            grad_outputs=torch.ones_like(Y0),
            # allow_unused=True,
            create_graph=True,
            retain_graph=True
        )[0]

        total_residual_loss = 0.0

        for _ in range(self.N):
            
            q0 = self.q_net(t, y0)
            dW = torch.randn(batch_size, self.dim_W, device=self.device) * self.dt**0.5
            y1 = self.forward_dynamics(y0, q0, dW, t, self.dt)
            
            σ0 = self.sigma(t, y0)
            z0 = torch.bmm(σ0, dY0.unsqueeze(-1)).squeeze(-1)

            t = t + self.dt

            Y1 = self.Y_net(t, y1)
            dY1 = torch.autograd.grad(
                outputs=Y1,
                inputs=y1,
                grad_outputs=torch.ones_like(Y0),
                # allow_unused=True,
                create_graph=True,
                retain_graph=True
            )[0]
            
            f = self.generator(y0, q0)
            Y1_tilde = Y0 - f * self.dt + (z0 * dW).sum(dim=1, keepdim=True)

            residual = Y1 - Y1_tilde
            total_residual_loss += torch.mean(residual**2)
            
            y0 = y1
            Y0 = Y1
            dY0 = dY1

        terminal = self.terminal_cost(y1)
        terminal_loss = torch.mean((Y1 - terminal)**2)
        return terminal_loss + total_residual_loss

    def train_model(self, epochs=1000, lr=1e-3, save_path=None, verbose=True, plot=True):
        """Train the model using Adam optimizer"""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=0.5,
            patience=50,
            min_lr=1e-6
        )
        
        self.train()
        best_loss = float('inf')
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = self()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step(loss)

            current_loss = loss.item()
            
            # Print status every 100 epochs
            if verbose and epoch % 100 == 0:
                status = f"Epoch {epoch}: Loss = {current_loss:.6f}"
                # Check if this is a new best loss
                if current_loss < best_loss:
                    if save_path is not None:
                        torch.save(self.state_dict(), save_path)
                    status += " (Best model saved ↓)"
                    best_loss = current_loss
                print(status)
            
        return best_loss
