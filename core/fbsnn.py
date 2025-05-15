import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from abc import ABC, abstractmethod
from core.nnets import Sine, FCnet, Resnet, YLSTM


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
        self.save = args.save
        self.save_n = args.save_n
        self.total_Y_loss = None
        self.terminal_loss = None
        self.terminal_gradient_loss = None

        if args.activation == "Sine":
            self.activation = Sine()
        elif args.activation == "ReLU":
            self.activation = nn.ReLU()

        if args.architecture == "Default":
            self.Y_net = FCnet(layers=[2]+[64, 64, 64, 64, 1], activation=self.activation).to(self.device)
        elif args.architecture == "FC":
            self.Y_net = FCnet(layers=[2]+model_cfg["Y_layers"], activation=self.activation).to(self.device)
        elif args.architecture == "NAISnet":
            self.Y_net = Resnet(layers=[2]+model_cfg["Y_layers"], activation=self.activation, stable=True).to(self.device)
        elif args.architecture == "Resnet":
            self.Y_net = Resnet(layers=[2]+model_cfg["Y_layers"], activation=self.activation, stable=False).to(self.device)
        elif args.architecture == "LSTM":
            self.Y_net = YLSTM(input_dim=2, hidden_dim=64, output_dim=1).to(self.device)
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

        total_Y_loss = 0.0

        for _ in range(self.N):
        
            # Compute sigma and Z = sigma * dY
            σ0 = self.sigma(t, y0)                      # shape: (batch, 1, 1)
            Z0 = torch.bmm(σ0, dY0.unsqueeze(-1)).squeeze(-1)  # shape: (batch, 1)

            # Compute q analytically: q = -0.5 * Z / sigma_x^2
            q0 = -0.5 * Z0 / (self.sigma_x ** 2)

            # Simulate forward
            dW = torch.randn(batch_size, self.dim_W, device=self.device) * self.dt**0.5
            y1 = self.forward_dynamics(y0, q0, dW, t, self.dt)
            t = t + self.dt

            # Propagate value network and compute gradients
            Y1 = self.Y_net(t, y1)
            dY1 = torch.autograd.grad(
                outputs=Y1,
                inputs=y1,
                grad_outputs=torch.ones_like(Y1),
                create_graph=True,
                retain_graph=True
            )[0]
            
            f = self.generator(y0, q0)
            
            Y1_tilde = Y0 - f * self.dt + (Z0 * dW).sum(dim=1, keepdim=True)
            total_Y_loss += torch.mean(torch.pow(Y1 - Y1_tilde, 2))

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

        self.total_Y_loss = total_Y_loss.detach().item()
        self.terminal_loss = terminal_loss.detach().item()
        self.terminal_gradient_loss = terminal_gradient_loss.detach().item()

        return total_Y_loss + terminal_loss + terminal_gradient_loss

    def train_model(self, epochs=1000, lr=1e-3, save_path="models/saved/model", verbose=True, plot=True):
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
                    print(f"{'Epoch':>8} | {'Total loss':>12} | {'Y loss':>12} | {'T. loss':>12} | {'T.G. loss':>12} | {'Memory [MB]':>12} | {'Time [s]':>10} | {'Status'}")
                    print("-" * 120)
                    header_printed = True

                mem_mb = torch.cuda.memory_allocated() / 1e6
                
                status = ""

                if "every" in self.save and epoch % self.save_n == 0:
                    torch.save(self.state_dict(), save_path + ".pth")
                    status = f"Model saved ↓"

                if "best" in self.save and loss.item() < self.lowest_loss:
                    self.lowest_loss = loss.item()
                    torch.save(self.state_dict(), save_path + "_best.pth")
                    status = "Model saved ↓ (best)"

                print(f"{epoch:8} | {loss.item():12.6f} | {self.total_Y_loss:12.6f} | {self.terminal_loss:12.6f} | {self.terminal_gradient_loss:12.6f} | {mem_mb:12.2f} | {elapsed:10.2f} | {status}")
                start_time = time.time()  # Reset timer for next block

        if "last" in self.save:
            torch.save(self.state_dict(), save_path + ".pth")
            status = f"Model saved ↓"
            print(f"{epoch:8} | {loss.item():12.6f} | {self.total_Y_loss:12.6f} | {self.terminal_loss:12.6f} | {self.terminal_gradient_loss:12.6f} | {mem_mb:12.2f} | {elapsed:10.2f} | {status}")

        if verbose:
            print("-" * 70)
            print(f"Training completed. Lowest loss: {self.lowest_loss:.6f}. Total time: {time.time() - init_time:.2f} seconds")
            print(f"Model saved to {save_path}.pth")

        if plot:
            plt.plot(losses, label="Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training Loss")
            plt.legend()
            plt.grid()
            plt.show()

        return losses
