import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from abc import ABC, abstractmethod
from core.nnets import Sine, FCnet, Resnet


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
            create_graph=True,
            retain_graph=True
        )[0]

        total_residual_loss = 0.0

        for _ in range(self.N):
            
            q0 = self.q_net(t, y0)
            dW = torch.randn(batch_size, self.dim_W, device=self.device) * self.dt**0.5
            y1 = self.forward_dynamics(y0, q0, dW, t, self.dt)

            σ0 = self.sigma(t, y0)
            Z0 = torch.bmm(σ0, dY0.unsqueeze(-1)).squeeze(-1)

            t = t + self.dt

            Y1 = self.Y_net(t, y1)
            dY1 = torch.autograd.grad(
                outputs=Y1,
                inputs=y1,
                grad_outputs=torch.ones_like(Y1),
                create_graph=True,
                retain_graph=True
            )[0]
            
            f = self.generator(y0, q0)

            Y1_tilde = Y0 - f * self.dt + (Z0 * dW)

            residual = Y1 - Y1_tilde
            total_residual_loss += torch.mean(torch.pow(residual, 2))
            
            y0 = y1
            Y0 = Y1
            dY0 = dY1

        terminal = self.terminal_cost(y1)
        terminal_loss = torch.mean(torch.pow(Y1 - terminal, 2))

        terminal_gradient = torch.autograd.grad(
            outputs=terminal.unsqueeze(-1),
            inputs=y1,
            grad_outputs=torch.ones_like(terminal.unsqueeze(-1)),
            create_graph=True,
            retain_graph=True
        )[0]
        terminal_gradient_loss = torch.mean(torch.pow(dY1 - terminal_gradient, 2))

        return total_residual_loss + terminal_loss + terminal_gradient_loss

    def train_model(self, epochs=1000, lr=1e-3, save_path="models/saved/model.pth", verbose=True, plot=True, save_best_only=True):
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
                if save_best_only:
                    if loss.item() < self.lowest_loss:
                        self.lowest_loss = loss.item()
                        torch.save(self.state_dict(), save_path)
                        status = "Model saved ↓"
                else:
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
