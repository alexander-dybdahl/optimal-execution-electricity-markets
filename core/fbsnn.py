import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import numpy as np
import os
import logging
from abc import ABC, abstractmethod
from core.nnets import Sine, FCnet, Resnet, LSTMNet


class FBSNN(nn.Module, ABC):
    def __init__(self, args, model_cfg):
        super().__init__()
        self.device = args.device
        self.architecture = args.architecture
        self.supervised = args.supervised
        self.Y_layers = args.Y_layers
        self.n_paths = args.n_paths
        self.batch_size = args.batch_size
        self.λ_Y = args.lambda_Y
        self.λ_T = args.lambda_T
        self.λ_TG = args.lambda_TG
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
        elif args.activation == "Tanh":
            self.activation = nn.Tanh()
        elif args.activation == "LeakyReLU":
            self.activation = nn.LeakyReLU()
        elif args.activation == "ELU":
            self.activation = nn.ELU()
        elif args.activation == "Softplus":
            self.activation = nn.Softplus()
        elif args.activation == "Softsign":
            self.activation = nn.Softsign()
        elif args.activation == "GELU":
            self.activation = nn.GELU()
        elif args.activation == "Sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation function: {args.activation}")

        if args.architecture == "Default":
            self.Y_net = FCnet(layers=[self.dim + 1] + [64, 64, 64, 64, 1], activation=self.activation).to(self.device)
        elif args.architecture == "FC":
            self.Y_net = FCnet(layers=[self.dim + 1] + args.Y_layers, activation=self.activation).to(self.device)
        elif args.architecture == "NAISnet":
            self.Y_net = Resnet(layers=[self.dim + 1] + args.Y_layers, activation=self.activation, stable=True).to(self.device)
        elif args.architecture == "Resnet":
            self.Y_net = Resnet(layers=[self.dim + 1] + args.Y_layers, activation=self.activation, stable=False).to(self.device)
        elif args.architecture == "LSTM" or args.architecture == "ResLSTM" or args.architecture == "NaisLSTM":
            self.Y_net = LSTMNet(layers=[self.dim + 1] + args.Y_layers, activation=self.activation, type=args.architecture).to(self.device)
        else:
            raise ValueError(f"Unknown architecture: {args.architecture}")

        self.lowest_loss = float("inf")

    @abstractmethod
    def generator(self, y, q): pass

    @abstractmethod
    def terminal_cost(self, y): pass

    def terminal_cost_grad(self, y):
        return torch.autograd.grad(
            outputs=self.terminal_cost(y),
            inputs=y,
            grad_outputs=torch.ones_like(self.terminal_cost(y)),
            create_graph=True,
            retain_graph=True
        )[0]

    @abstractmethod
    def mu(self, t, y, q): pass  # shape: (batch, dim)

    @abstractmethod
    def sigma(self, t, y): pass  # shape: (batch, dim, dW_dim)

    @abstractmethod
    def sigma(self, t, y): pass  # shape: (batch, dim, dW_dim)

    @abstractmethod
    def trading_rate(self, t, y, Y): pass 

    @abstractmethod
    def pinn_loss(self, t, y, Y): pass

    @abstractmethod
    def forward_supervised(self, t_paths, W_paths): pass

    def forward_dynamics(self, y, q, dW, t, dt):
        μ = self.mu(t, y, q)                  # shape: (batch, dim)
        σ = self.sigma(t, y)                  # shape: (batch, dim, dW_dim)
        diffusion = torch.bmm(σ, dW.unsqueeze(-1)).squeeze(-1)  # shape: (batch, dim)
        return y + μ * dt + diffusion         # shape: (batch, dim)

    def normalize_inputs(self, t, y):
        return (t - self.t_mean) / self.t_std, (y - self.y_mean) / self.y_std
    
    def fetch_minibatch(self):
        batch_size = self.batch_size
        dim_W = self.dim_W
        T = self.T
        N = self.N
        dt = T / N
        dW = torch.randn(batch_size, N, dim_W, device=self.device) * np.sqrt(dt)
        W = torch.cat([
            torch.zeros(batch_size, 1, dim_W, device=self.device),
            torch.cumsum(dW, dim=1)
        ], dim=1)  # Shape: [batch_size, N+1, dim_W]
        
        t = torch.linspace(0, T, N + 1, device=self.device).view(1, -1, 1).repeat(batch_size, 1, 1)  # [M, N+1, 1]
        return t, W

    def forward(self, t_paths, W_paths):
        if self.supervised:
            return self.forward_supervised(t_paths, W_paths)
        else:
            return self.forward_fc(t_paths, W_paths)

    def forward_fc(self, t_paths, W_paths):

        batch_size = self.batch_size
        t0 = t_paths[:, 0, :]
        W0 = W_paths[:, 0, :]
        y0 = self.y0.repeat(batch_size, 1).to(self.device)
        Y0 = self.Y_net(t0, y0)

        dY0 = torch.autograd.grad(
            outputs=Y0,
            inputs=y0,
            grad_outputs=torch.ones_like(Y0),
            create_graph=True,
            retain_graph=True
        )[0]

        Y_loss = 0.0

        for n in range(self.N):
            t1 = t_paths[:, n + 1, :]
            W1 = W_paths[:, n + 1, :]
            σ0 = self.sigma(t0, y0)
            Z0 = torch.bmm(σ0.transpose(1, 2), dY0.unsqueeze(-1)).squeeze(-1)
            q = self.trading_rate(t0, y0, Y0)
            y1 = self.forward_dynamics(y0, q, W1 - W0, t0, t1 - t0)

            Y1 = self.Y_net(t1, y1)
            dY1 = torch.autograd.grad(
                outputs=Y1,
                inputs=y1,
                grad_outputs=torch.ones_like(Y1),
                create_graph=True,
                retain_graph=True
            )[0]

            f = self.generator(y0, q)
            Y1_tilde = Y0 - f * (t1 - t0) + (Z0 * (W1 - W0)).sum(dim=1, keepdim=True)
            Y_loss += torch.sum(torch.pow(Y1 - Y1_tilde, 2))

            t0, W0, y0, Y0, dY0 = t1, W1, y1, Y1, dY1

        terminal_loss = torch.sum(torch.pow(Y1 - self.terminal_cost(y1), 2))
        terminal_gradient_loss = torch.sum(torch.pow(dY1 - self.terminal_cost_grad(y1), 2))

        self.total_Y_loss = self.λ_Y * Y_loss.detach().item()
        self.terminal_loss = self.λ_T * terminal_loss.detach().item()
        self.terminal_gradient_loss = self.λ_TG * terminal_gradient_loss.detach().item()

        return self.λ_Y * Y_loss + self.λ_T * terminal_loss + self.λ_TG * terminal_gradient_loss

    def train_model(self, epochs=1000, K=50, lr=1e-3, verbose=True, plot=True, adaptive=True, save_dir=None):
        save_path = os.path.join(save_dir, "model")
        log_file = os.path.join(save_dir, "training.log")
        logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO, format='%(message)s')
        logger = logging.getLogger()

        def log(msg):
            if verbose:
                print(msg)
            logger.info(msg)

        self.device = next(self.parameters()).device
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.80, patience=200)
            if adaptive else
            torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
        )

        losses, losses_Y, losses_terminal, losses_terminal_gradient = [], [], [], []
        self.train()

        log(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        log(f"Logging training to {log_file}")
        log("\n+---------------------------+---------------------------+")
        log("| Training Configuration    |                           |")
        log("+---------------------------+---------------------------+")
        log(f"| Epochs                    | {epochs:<25} |")
        log(f"| Learning Rate             | {lr:<25} |")
        log(f"| Adaptive LR               | {'True' if adaptive else 'False':<25} |")
        log(f"| lambda_Y (Y loss)         | {self.λ_Y:<25} |")
        log(f"| lambda_T (Terminal loss)  | {self.λ_T:<25} |")
        log(f"| lambda_TG (Gradient loss) | {self.λ_TG:<25} |")
        log(f"| Number of Paths           | {self.n_paths:<25} |")
        log(f"| Batch Size                | {self.batch_size:<25} |")
        log(f"| Architecture              | {self.architecture:<25} |")
        log(f"| Depth                     | {len(self.Y_layers):<25} |")
        log(f"| Width                     | {self.Y_layers[0]:<25} |")
        log(f"| Activation                | {self.activation.__class__.__name__:<25} |")
        log(f"| T                         | {self.T:<25} |")
        log(f"| N                         | {self.N:<25} |")
        log(f"| Supervised                | {'True' if self.supervised else 'False':<25} |")
        log("+---------------------------+---------------------------+\n")

        header_printed = False
        init_time = time.time()
        start_time = time.time()
        
        max_widths = {
            "epoch": 8,
            "loss": 12,
            "lr": 10,
            "mem": 12,
            "time": 8,
            "eta": 8,
            "status": 20
        }
        width = (max_widths['epoch'] + max_widths['loss'] * 4 + max_widths['lr'] + max_widths['mem'] + max_widths['time'] + max_widths['eta'] + max_widths['status'] + 8 * 3 + 1)

        for epoch in range(epochs):
            optimizer.zero_grad()
            t_paths, W_paths = self.fetch_minibatch()
            loss = self(t_paths, W_paths)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            losses_Y.append(self.total_Y_loss)
            losses_terminal.append(self.terminal_loss)
            losses_terminal_gradient.append(self.terminal_gradient_loss)

            scheduler.step(loss.item() if adaptive else epoch)

            if (epoch % K == 0 or epoch == epochs - 1):
                
                avg_time_per_K = (time.time() - init_time) / (epoch + 1e-8)  # avoid div-by-zero
                eta_seconds = avg_time_per_K * (epochs - epoch) if epoch > 0 else 0
                eta_minutes_part = eta_seconds // 60
                eta_seconds_part = eta_seconds % 60
                if epoch == 0:
                    eta_str = "N/A"
                else:
                    eta_seconds = int(avg_time_per_K * (epochs - epoch))
                    eta_minutes_part = eta_seconds // 60
                    eta_seconds_part = eta_seconds % 60
                    eta_str = f"{eta_minutes_part}m {eta_seconds_part:02d}s" if eta_seconds >= 60 else f"{eta_seconds}s"

                elapsed = time.time() - start_time
                if not header_printed:
                    log(f"{'Epoch':>{max_widths['epoch']}} | "
                        f"{'Total loss':>{max_widths['loss']}} | "
                        f"{'Y loss':>{max_widths['loss']}} | "
                        f"{'T. loss':>{max_widths['loss']}} | "
                        f"{'T.G. loss':>{max_widths['loss']}} | "
                        f"{'LR':>{max_widths['lr']}} | "
                        f"{'Memory [MB]':>{max_widths['mem']}} | "
                        f"{'Time [s]':>{max_widths['time']}} | "
                        f"{'ETA':>{max_widths['eta']}} | "
                        f"{'Status':<{max_widths['status']}}")
                    log("-" * width)
                    header_printed = True

                mem_mb = torch.cuda.memory_allocated() / 1e6
                current_lr = optimizer.param_groups[0]['lr']
                status = ""

                if "every" in self.save and (epoch % self.save_n == 0 or epoch == epochs - 1):
                    try:
                        torch.save(self.state_dict(), save_path + ".pth")
                    except Exception as e:
                        log(f"Error saving model: {e}")
                    status = "Model saved"

                if "best" in self.save and np.mean(losses[-K:]) < self.lowest_loss:
                    self.lowest_loss = np.mean(losses[-K:])
                    try:
                        torch.save(self.state_dict(), save_path + "_best.pth")
                    except Exception as e:
                        log(f"Error saving best model: {e}")
                    status = "Model saved (best)"

                log(f"{epoch:>{max_widths['epoch']}} | "
                    f"{np.mean(losses[-K:]):>{max_widths['loss']}.2e} | "
                    f"{np.mean(losses_Y[-K:]):>{max_widths['loss']}.2e} | "
                    f"{np.mean(losses_terminal[-K:]):>{max_widths['loss']}.2e} | "
                    f"{np.mean(losses_terminal_gradient[-K:]):>{max_widths['loss']}.2e} | "
                    f"{current_lr:>{max_widths['lr']}.2e} | "
                    f"{mem_mb:>{max_widths['mem']}.2f} | "
                    f"{elapsed:>{max_widths['time']}.2f} | "
                    f"{eta_str:>{max_widths['eta']}} | "
                    f"{status:<{max_widths['status']}}")
                start_time = time.time()

        if "last" in self.save:
            try:
                torch.save(self.state_dict(), save_path + ".pth")
            except Exception as e:
                log(f"Error saving last model: {e}")
            status = "Model saved"
            log(f"{epoch:>{max_widths['epoch']}} | "
                f"{np.mean(losses[-K:]):>{max_widths['loss']}.2e} | "
                f"{np.mean(losses_Y[-K:]):>{max_widths['loss']}.2e} | "
                f"{np.mean(losses_terminal[-K:]):>{max_widths['loss']}.2e} | "
                f"{np.mean(losses_terminal_gradient[-K:]):>{max_widths['loss']}.2e} | "
                f"{current_lr:>{max_widths['lr']}.2e} | "
                f"{mem_mb:>{max_widths['mem']}.2f} | "
                f"{elapsed:>{max_widths['time']}.2f} | "
                f"{eta_str:>{max_widths['eta']}} | "
                f"{status:<{max_widths['status']}}")

        log("-" * width)
        log(f"Training completed. Lowest loss: {self.lowest_loss:.6f}. Total time: {time.time() - init_time:.2f} seconds")
        log(f"Model saved to {save_path}.pth")

        plt.figure(figsize=(10, 6))
        plt.plot(losses, label="Total Loss")
        plt.plot(losses_Y, label="Y Loss")
        plt.plot(losses_terminal, label="Terminal Loss")
        plt.plot(losses_terminal_gradient, label="Terminal Gradient Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.yscale("log")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/loss.png", dpi=300, bbox_inches='tight')
        if plot:
            plt.show()

        return losses