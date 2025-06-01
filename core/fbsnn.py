import os
import time
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from core.nnets import FCnet, LSTMNet, Resnet, Sine
from utils.logger import Logger


class FBSNN(nn.Module, ABC):
    def __init__(self, args, model_cfg):
        super().__init__()
        # System & Execution Settings
        self.is_distributed = dist.is_initialized()
        self.device = torch.device(f"cuda:{dist.get_rank()}" if args.device == "cuda" else "cpu") if self.is_distributed else torch.device(args.device)
        self.world_size = dist.get_world_size() if self.is_distributed else 1
        self.is_main = not self.is_distributed or dist.get_rank() == 0

        # Data & Batch Settings
        self.batch_size = args.batch_size_per_rank
        self.supervised = args.supervised
        self.simulate_true = args.simulate_true

        # Network Architecture
        self.architecture = args.architecture
        self.Y_layers = args.Y_layers      # e.g., [64, 64, 64]
        self.adaptive_factor = args.adaptive_factor

        # Loss Weights
        self.lambda_Y = args.lambda_Y      # value function loss
        self.lambda_dY = args.lambda_dY    # spatial gradient loss
        self.lambda_dYt = args.lambda_dYt  # temporal gradient loss
        self.lambda_T = args.lambda_T      # terminal value loss
        self.lambda_TG = args.lambda_TG    # terminal gradient loss
        self.lambda_pinn = args.lambda_pinn  # physics residual loss

        # Loss Tracking
        self.Y_loss = torch.tensor(0.0, device=self.device)
        self.dY_loss = torch.tensor(0.0, device=self.device)
        self.dYt_loss = torch.tensor(0.0, device=self.device)
        self.terminal_loss = torch.tensor(0.0, device=self.device)
        self.terminal_gradient_loss = torch.tensor(0.0, device=self.device)
        self.pinn_loss = torch.tensor(0.0, device=self.device)

        # Time Discretization
        self.t0 = 0.0
        self.T = model_cfg["T"]
        self.N = model_cfg["N"]
        self.dt = model_cfg["dt"]

        # Problem Setup
        self.dim = model_cfg["dim"]        # state space dimension
        self.dim_W = model_cfg["dim_W"]    # Brownian motion dimension
        self.y0 = torch.tensor([model_cfg["y0"]], device=self.device, requires_grad=True)

        # Saving & Checkpointing
        self.save = args.save              # e.g., "best", "every", "last"
        self.save_n = args.save_n          # save every n epochs if "every"

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
            self.Y_net = FCnet(
                layers=[self.dim + 1] + [64, 64, 64, 64, 1], activation=self.activation
            ).to(self.device)
        elif args.architecture == "FC":
            self.Y_net = FCnet(
                layers=[self.dim + 1] + args.Y_layers, activation=self.activation
            ).to(self.device)
        elif args.architecture == "NAISnet":
            self.Y_net = Resnet(
                layers=[self.dim + 1] + args.Y_layers,
                activation=self.activation,
                stable=True,
            ).to(self.device)
        elif args.architecture == "Resnet":
            self.Y_net = Resnet(
                layers=[self.dim + 1] + args.Y_layers,
                activation=self.activation,
                stable=False,
            ).to(self.device)
        elif (
            args.architecture == "LSTM"
            or args.architecture == "ResLSTM"
            or args.architecture == "NaisLSTM"
        ):
            self.Y_net = LSTMNet(
                layers=[self.dim + 1] + args.Y_layers,
                activation=self.activation,
                type=args.architecture,
            ).to(self.device)
        else:
            raise ValueError(f"Unknown architecture: {args.architecture}")

        self.lowest_loss = float("inf")

    @abstractmethod
    def generator(self, y, q):
        pass

    @abstractmethod
    def terminal_cost(self, y):
        pass

    def terminal_cost_grad(self, y):
        return torch.autograd.grad(
            outputs=self.terminal_cost(y),
            inputs=y,
            grad_outputs=torch.ones_like(self.terminal_cost(y)),
            create_graph=True,
            retain_graph=True,
        )[0]

    @abstractmethod
    def mu(self, t, y, q):
        pass                                # shape: (batch, dim)

    @abstractmethod
    def sigma(self, t, y):
        pass                                # shape: (batch, dim, dW_dim)

    @abstractmethod
    def optimal_control(self, t, y, Y):
        pass

    @abstractmethod
    def value_function_analytic(self, t, y):
        pass

    def forward_dynamics(self, y, q, dW, t, dt):
        mu = self.mu(t, y, q)               # shape: (batch, dim)
        Sigma = self.sigma(t, y)            # shape: (batch, dim, dW_dim)
        diffusion = torch.bmm(Sigma, dW.unsqueeze(-1)).squeeze(
            -1
    )                                       # shape: (batch, dim)
        return y + mu * dt + diffusion      # shape: (batch, dim)

    def physics_loss(self, t, y, Y):
        dim = y.shape[1]
        t.requires_grad_(True)
        y.requires_grad_(True)

        dV = torch.autograd.grad(
            outputs=Y, inputs=y, grad_outputs=torch.ones_like(Y), create_graph=True
        )[0]

        dV_t = torch.autograd.grad(
            outputs=Y, inputs=t, grad_outputs=torch.ones_like(Y), create_graph=True
        )[0]

        # Compute Hessian H_ij = ∂²V / ∂y_i ∂y_j
        grads = [
            torch.autograd.grad(
                outputs=dV[:, i],
                inputs=y,
                grad_outputs=torch.ones_like(dV[:, i]),
                create_graph=True,
            )[0]
            for i in range(dim)
        ]
        H = torch.stack(grads, dim=1)       # shape: (batch, dim, dim)

        q = self.optimal_control(t, y, Y)
        mu = self.mu(t, y, q)
        sigma = self.sigma(t, y)
        f = self.generator(y, q)

        # Trace(σ σᵀ H)
        sigma_T = sigma.transpose(1, 2)     # shape: (batch, dW_dim, dim)
        sigma_sigma_T = torch.bmm(sigma, sigma_T)  # shape: (batch, dim, dim)
        diffusion_term = 0.5 * torch.einsum("bij,bij->b", sigma_sigma_T, H).unsqueeze(
            -1
        )                                   # shape: (batch, 1)

        residual = dV_t + torch.sum(mu * dV, dim=1, keepdim=True) + diffusion_term - f
        return torch.mean(residual**2)

    def fetch_minibatch(self, batch_size):
        dim_W = self.dim_W
        T = self.T
        N = self.N
        dt = T / N
        dW = torch.randn(batch_size, N, dim_W, device=self.device) * np.sqrt(dt)
        W = torch.cat(
            [
                torch.zeros(batch_size, 1, dim_W, device=self.device),
                torch.cumsum(dW, dim=1),
            ],
            dim=1,
        )                                   # shape: (batch_size, N+1, dim_W)

        t = (
            torch.linspace(0, T, N + 1, device=self.device)
            .view(1, -1, 1)
            .repeat(batch_size, 1, 1)
        )                                   # shape: (batch_size, N+1, 1)
        return t, W

    def forward(self, t_paths, W_paths):
        if self.supervised:
            return self.forward_supervised(t_paths, W_paths)
        else:
            return self.forward_fc(t_paths, W_paths)

    def forward_fc(self, t_paths, W_paths):
        self.Y_net.eval()
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
            retain_graph=True,
        )[0]

        Y_loss = 0.0

        for n in range(self.N):
            t1 = t_paths[:, n + 1, :]
            W1 = W_paths[:, n + 1, :]
            Sigma0 = self.sigma(t0, y0)
            Z0 = torch.bmm(Sigma0.transpose(1, 2), dY0.unsqueeze(-1)).squeeze(-1)
            q = self.optimal_control(t0, y0, Y0)
            y1 = self.forward_dynamics(y0, q, W1 - W0, t0, t1 - t0)

            Y1 = self.Y_net(t1, y1)
            dY1 = torch.autograd.grad(
                outputs=Y1,
                inputs=y1,
                grad_outputs=torch.ones_like(Y1),
                create_graph=True,
                retain_graph=True,
            )[0]

            f = self.generator(y0, q)
            Y1_tilde = Y0 - f * (t1 - t0) + (Z0 * (W1 - W0)).sum(dim=1, keepdim=True)
            Y_loss += torch.sum(torch.pow(Y1 - Y1_tilde, 2))

            t0, W0, y0, Y0, dY0 = t1, W1, y1, Y1, dY1

        terminal_loss = torch.sum(torch.pow(Y1 - self.terminal_cost(y1), 2))
        terminal_gradient_loss = torch.sum(torch.pow(dY1 - self.terminal_cost_grad(y1), 2))

        # Record losses while maintaining location on device
        self.Y_loss = self.lambda_Y * Y_loss.detach()
        self.terminal_loss = self.lambda_T * terminal_loss.detach()
        self.terminal_gradient_loss = self.lambda_TG * terminal_gradient_loss.detach()

        return (
            self.lambda_Y * Y_loss
            + self.lambda_T * terminal_loss
            + self.lambda_TG * terminal_gradient_loss
        )

    def forward_supervised(self, t_paths, W_paths):
        batch_size = self.batch_size
        device = self.device
        y0 = self.y0.repeat(batch_size, 1).to(device)

        # === Precompute optimal trajectory ===
        y_traj, t_traj = [y0], []
        t0 = t_paths[:, 0, :]
        W0 = W_paths[:, 0, :]

        for n in range(self.N):
            t1 = t_paths[:, n + 1, :]
            W1 = W_paths[:, n + 1, :]
            dW = W1 - W0

            V = self.value_function_analytic(t0, y0)
            q = self.optimal_control(t0, y0, V, create_graph=False)
            y1 = self.forward_dynamics(y0, q, dW, t0, t1 - t0)

            t_traj.append(t1)
            y_traj.append(y1)

            t0, W0, y0 = t1, W1, y1

        t_traj = torch.cat(t_traj, dim=0).requires_grad_(
            True
        )                                   # shape: (N * batch_size, 1)
        y_traj = torch.cat(y_traj[1:], dim=0).requires_grad_(
            True
        )                                   # shape: (N * batch_size, dim)

        # === Compute target value and gradients ===
        V_target = self.value_function_analytic(t_traj, y_traj)
        V_pred = self.Y_net(t_traj, y_traj)

        Y_loss = torch.sum(torch.pow(V_pred - V_target, 2))
        if self.lambda_Y > 0:
            self.Y_loss = self.lambda_Y * Y_loss.detach()

        dY_loss, dYt_loss = 0.0, 0.0
        if self.lambda_dY > 0:
            dV_target = torch.autograd.grad(
                V_target,
                y_traj,
                grad_outputs=torch.ones_like(V_target),
                create_graph=False,
                retain_graph=True,
            )[0]
            dV_pred = torch.autograd.grad(
                V_pred, y_traj, grad_outputs=torch.ones_like(V_pred), create_graph=True
            )[0]
            dY_loss = torch.sum(torch.pow(dV_pred - dV_target, 2))
            self.dY_loss = self.lambda_dY * dY_loss.detach()

        if self.lambda_dYt > 0:
            dV_target_t = torch.autograd.grad(
                V_target,
                t_traj,
                grad_outputs=torch.ones_like(V_target),
                create_graph=False,
                retain_graph=True,
            )[0]
            dV_pred_t = torch.autograd.grad(
                V_pred, t_traj, grad_outputs=torch.ones_like(V_pred), create_graph=True
            )[0]
            dYt_loss = torch.sum(torch.pow(dV_pred_t - dV_target_t, 2))
            self.dYt_loss = self.lambda_dYt * dYt_loss.detach()

        # === Terminal supervision ===
        terminal_loss, terminal_gradient_loss = 0.0, 0.0
        if self.lambda_T > 0:
            YT = self.Y_net(t1, y1)
            terminal_loss = torch.mean(torch.pow(YT - self.terminal_cost(y1), 2))
            self.terminal_loss = self.lambda_T * terminal_loss.detach()
        
        if self.lambda_TG > 0:
            dYT = torch.autograd.grad(
                YT, y1, grad_outputs=torch.ones_like(YT), create_graph=True
            )[0]
            terminal_gradient_loss = torch.mean(torch.pow(dYT - self.terminal_cost_grad(y1), 2))
            self.terminal_gradient_loss = self.lambda_TG * terminal_gradient_loss.detach()

        # === Optional physics-based loss ===
        pinn_loss = 0.0
        if self.lambda_pinn > 0:
            with torch.enable_grad():
                for i in range(self.N):
                    idx = slice(i * batch_size, (i + 1) * batch_size)
                    t_i, y_i, V_i = t_traj[idx], y_traj[idx], V_pred[idx]
                    pinn_loss += self.physics_loss(t_i, y_i, V_i)
                pinn_loss /= self.N
                self.pinn_loss = self.lambda_pinn * pinn_loss.detach()

        return (
            self.lambda_Y * Y_loss
            + self.lambda_dY * dY_loss
            + self.lambda_dYt * dYt_loss
            + self.lambda_T * terminal_loss
            + self.lambda_TG * terminal_gradient_loss
            + self.lambda_pinn * pinn_loss
        )

    def simulate_paths(self, n_sim=5, seed=42, y0_single=None):
        torch.manual_seed(seed)

        y0 = (
            y0_single.repeat(n_sim, 1)
            if y0_single is not None
            else self.y0.repeat(n_sim, 1)
        )
        t_scalar = 0.0

        # Initialize both
        y_learned = y0.clone()
        y_analytic = y0.clone()

        y_learned_traj = []
        q_learned_traj = []
        Y_learned_traj = []
        if self.simulate_true:
            y_true_traj = []
            q_true_traj = []
            Y_true_traj = []

        for step in range(self.N + 1):
            t_tensor = torch.full((n_sim, 1), t_scalar, device=self.device)

            # Predict Y and compute control
            Y_learned = self.Y_net(t_tensor, y_learned)
            q_learned = self.optimal_control(
                t_tensor, y_learned, Y_learned, create_graph=False
            )
            if self.analytical_known:
                Y_true = self.value_function_analytic(t_tensor, y_analytic)
                q_true = self.optimal_control(
                    t_tensor, y_analytic, Y_true, create_graph=False
                )

            # Save states and controls
            q_learned_traj.append(q_learned.detach().cpu().numpy())
            y_learned_traj.append(y_learned.detach().cpu().numpy())
            Y_learned_traj.append(Y_learned.detach().cpu().numpy())

            if self.simulate_true:
                q_true_traj.append(q_true.detach().cpu().numpy())
                y_true_traj.append(y_analytic.detach().cpu().numpy())
                Y_true_traj.append(Y_true.detach().cpu().numpy())

            if step < self.N:
                dW = torch.randn(n_sim, self.dim_W, device=self.device) * self.dt**0.5
                y_learned = self.forward_dynamics(
                    y_learned, q_learned, dW, t_tensor, self.dt
                )
                if self.supervised:
                    y_analytic = self.forward_dynamics(
                        y_analytic, q_true, dW, t_tensor, self.dt
                    )
                t_scalar += self.dt

        return torch.linspace(0, self.T, self.N + 1).cpu().numpy(), {
            "y_learned": np.stack(y_learned_traj),
            "q_learned": np.stack(q_learned_traj),
            "Y_learned": np.stack(Y_learned_traj),
            "y_true": np.stack(y_true_traj) if self.analytical_known else None,
            "q_true": np.stack(q_true_traj) if self.analytical_known else None,
            "Y_true": np.stack(Y_true_traj) if self.analytical_known else None
        }
    
    def _save_model(self, save_path):
        state_dict = self.state_dict()

        try:
            torch.save(state_dict, save_path + ".pth")
        except Exception as e:
            return f"Error saving model: {e}"
        
        return "Model saved"
        
    def train_model(
        self,
        epochs=1000,
        K=50,
        lr=1e-3,
        verbose=True,
        plot=True,
        adaptive=True,
        save_dir=None,
        logger=None,
    ):

        # Prepare save directory and logging
        save_path = os.path.join(save_dir, "model")
        logger = logger if logger is not None else Logger(save_dir, is_main=self.is_main, verbose=verbose, filename="training.log")
        
        # Dynamic width calculation based on loss components
        max_widths = {
            "epoch": 8,
            "loss": 8,
            "lr": 10,
            "mem": 12,
            "time": 8,
            "eta": 8,
            "status": 20,
        }

        (
            losses,
            losses_Y,
            losses_dY,
            losses_dYt,
            losses_terminal,
            losses_terminal_gradient,
            losses_pinn,
        ) = [], [], [], [], [], [], []

        # Define active loss components
        loss_components = [
            ("Y loss", self.lambda_Y, lambda: np.mean(losses_Y[-K:])),
            ("dY loss", self.lambda_dY, lambda: np.mean(losses_dY[-K:])),
            ("dYt loss", self.lambda_dYt, lambda: np.mean(losses_dYt[-K:])),
            ("T. loss", self.lambda_T, lambda: np.mean(losses_terminal[-K:])),
            ("T.G. loss", self.lambda_TG, lambda: np.mean(losses_terminal_gradient[-K:])),
            ("pinn loss", self.lambda_pinn, lambda: np.mean(losses_pinn[-K:])),
        ]
        active_losses = [
            (name, fn) for name, lambda_, fn in loss_components if lambda_ > 0
        ]

        # Calculate max widths
        for name, _ in active_losses:
            max_widths[name] = max(10, len(name) + 2)
        width = (
            max_widths["epoch"]
            + sum(max_widths[name] for name, _ in active_losses)
            + max_widths["lr"]
            + max_widths["mem"]
            + max_widths["time"]
            + max_widths["eta"]
            + max_widths["status"]
            + 3 * (7 + len(active_losses))
        )

        if self.is_main:
            ### Log training configuration
            logger.log(
                f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
            )
            logger.log(f"Logging training to {logger.log_path}")
            logger.log("\n+---------------------------+---------------------------+")
            logger.log("| Training Configuration    |                           |")
            logger.log("+---------------------------+---------------------------+")
            logger.log(f"| Epochs                    | {epochs:<25} |")
            logger.log(f"| Learning Rate             | {lr:<25} |")
            logger.log(f"| Adaptive LR               | {'True' if adaptive else 'False':<25} |")
            logger.log(f"| Adaptive Factor           | {self.adaptive_factor:<25} |")
            logger.log(f"| lambda_Y (Y loss)         | {self.lambda_Y:<25} |")
            logger.log(f"| lambda_T (Terminal loss)  | {self.lambda_T:<25} |")
            logger.log(f"| lambda_TG (Gradient loss) | {self.lambda_TG:<25} |")
            logger.log(f"| Batch Size per Epoch      | {self.batch_size * self.world_size:<25} |")
            logger.log(f"| Batch Size per Rank       | {self.batch_size:<25} |")
            logger.log(f"| Architecture              | {self.architecture:<25} |")
            logger.log(f"| Depth                     | {len(self.Y_layers):<25} |")
            logger.log(f"| Width                     | {self.Y_layers[0]:<25} |")
            logger.log(f"| Activation                | {self.activation.__class__.__name__:<25} |")
            logger.log(f"| T                         | {self.T:<25} |")
            logger.log(f"| N                         | {self.N:<25} |")
            logger.log(f"| Supervised                | {'True' if self.supervised else 'False':<25} |")
            logger.log("+---------------------------+---------------------------+\n")

            init_time = time.time()
            start_time = time.time()

            # Print header
            header_parts = [
                f"{'Epoch':>{max_widths['epoch']}}",
                f"{'Total loss':>10}",
            ]
            for name, _ in active_losses:
                header_parts.append(f"{name:>{max_widths[name]}}")
            header_parts += [
                f"{'LR':>{max_widths['lr']}}",
                f"{'Memory [MB]':>{max_widths['mem']}}",
                f"{'Time [s]':>{max_widths['time']}}",
                f"{'ETA':>{max_widths['eta']}}",
                f"{'Status':<{max_widths['status']}}",
            ]
            logger.log(" | ".join(header_parts))
            logger.log("-" * width)

        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=self.adaptive_factor, patience=200
            )
            if adaptive
            else torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
        )

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            t_paths, W_paths = self.fetch_minibatch(self.batch_size)
            loss = self(t_paths, W_paths)
            loss.backward()
            optimizer.step()
            scheduler.step(loss.item() if adaptive else epoch)

            # Reduce losses across all processes if distributed
            Y_loss = self.Y_loss
            dY_loss = self.dY_loss
            dYt_loss = self.dYt_loss
            terminal_loss = self.terminal_loss
            terminal_gradient_loss = self.terminal_gradient_loss
            pinn_loss = self.pinn_loss

            if self.is_distributed:
                dist.reduce(loss, op=dist.ReduceOp.SUM, dst=0)
                dist.reduce(Y_loss, op=dist.ReduceOp.SUM, dst=0)
                dist.reduce(dY_loss, op=dist.ReduceOp.SUM, dst=0)
                dist.reduce(dYt_loss, op=dist.ReduceOp.SUM, dst=0)
                dist.reduce(terminal_loss, op=dist.ReduceOp.SUM, dst=0)
                dist.reduce(terminal_gradient_loss, op=dist.ReduceOp.SUM, dst=0)
                dist.reduce(pinn_loss, op=dist.ReduceOp.SUM, dst=0)

                loss /= self.world_size
                Y_loss /= self.world_size
                dY_loss /= self.world_size
                dYt_loss /= self.world_size
                terminal_loss /= self.world_size
                terminal_gradient_loss /= self.world_size
                pinn_loss /= self.world_size

            if self.is_main:
                losses.append(loss.item())
                losses_Y.append(Y_loss.item())
                losses_dY.append(dY_loss.item())
                losses_dYt.append(dYt_loss.item())
                losses_terminal.append(terminal_loss.item())
                losses_terminal_gradient.append(terminal_gradient_loss.item())
                losses_pinn.append(pinn_loss.item())

                if epoch % K == 0 or epoch == 1:
                    status = ""

                    # Save model if conditions are met
                    if "every" in self.save and (
                        epoch % self.save_n == 0 or epoch == epochs - 1
                    ):
                        status = self._save_model(save_path)

                    if "best" in self.save and np.mean(losses[-K:]) < self.lowest_loss:
                        self.lowest_loss = np.mean(losses[-K:])
                        status = self._save_model(save_path + "_best") + " (best)"

                    # Calculate average time per K epochs and ETA
                    avg_time_per_K = (time.time() - init_time) / (epoch + 1e-8)  # avoid div-by-zero
                    
                    if epoch == 1:
                        eta_str = "N/A"
                    else:
                        eta_seconds = int(avg_time_per_K * (epochs - epoch))
                        eta_minutes_part = eta_seconds // 60
                        eta_seconds_part = eta_seconds % 60
                        eta_str = (
                            f"{eta_minutes_part}m {eta_seconds_part:02d}s"
                            if eta_seconds >= 60
                            else f"{eta_seconds}s"
                        )

                    # Log the current epoch's results
                    elapsed = time.time() - start_time
                    mem_mb = torch.cuda.memory_allocated() / 1e6
                    current_lr = optimizer.param_groups[0]["lr"]

                    row_parts = [
                        f"{epoch:>{max_widths['epoch']}}",
                        f"{np.mean(losses[-K:]):>10.2e}",
                    ]
                    for name, fn in active_losses:
                        row_parts.append(f"{fn():>{max_widths[name]}.2e}")
                    row_parts += [
                        f"{current_lr:>{max_widths['lr']}.2e}",
                        f"{mem_mb:>{max_widths['mem']}.2f}",
                        f"{elapsed:>{max_widths['time']}.2f}",
                        f"{eta_str:>{max_widths['eta']}}",
                        f"{status:<{max_widths['status']}}",
                    ]
                    logger.log(" | ".join(row_parts))
                    start_time = time.time()

        if self.is_main:
            if self.is_distributed:
                dist.barrier()

            logger.log(f"Training completed. Lowest loss: {self.lowest_loss:.6f}. Total time: {time.time() - init_time:.2f} seconds")
            logger.log(f"Model saved to {save_path}.pth")

            # Plotting losses
            plt.figure(figsize=(10, 6))
            plt.plot(losses, label="Total Loss")
            plt.plot(losses_Y, label="Y Loss")
            plt.plot(losses_dY, label="dY Loss")
            plt.plot(losses_dYt, label="dYt Loss")
            plt.plot(losses_terminal, label="Terminal Loss")
            plt.plot(losses_terminal_gradient, label="Terminal Gradient Loss")
            plt.plot(losses_pinn, label="PINN Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training Loss")
            plt.yscale("log")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            if save_dir:
                plt.savefig(f"{save_dir}/loss.png", dpi=300, bbox_inches="tight")
            if plot:
                plt.show()

        return losses

    # def train_model(self, epochs=1000, K=50, lr=1e-3, save_path="saved/", verbose=True, plot=True, adaptive=True):
    #     self.device = next(self.parameters()).device
    #     optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    #     if adaptive:
    #         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #             optimizer, mode='min', factor=0.80, patience=200
    #         )
    #     else:
    #         scheduler = torch.optim.lr_scheduler.StepLR(
    #             optimizer, step_size=5000, gamma=0.5
    #         )

    #     losses = []
    #     losses_Y = []
    #     losses_terminal = []
    #     losses_terminal_gradient = []

    #     self.train()

    #     header_printed = False

    #     init_time = time.time()
    #     start_time = time.time()

    #     if verbose:
    #         print("\n+---------------------------+---------------------------+")
    #         print("| Training Configuration    |                           |")
    #         print("+---------------------------+---------------------------+")
    #         print(f"| Epochs                    | {epochs:<25} |")
    #         print(f"| Learning Rate             | {lr:<25} |")
    #         print(f"| Adaptive LR               | {'True' if adaptive else 'False':<25} |")
    #         print(f"| lambda_Y (Y loss)              | {self.lambda_Y:<25} |")
    #         print(f"| lambda_T (Terminal loss)       | {self.lambda_T:<25} |")
    #         print(f"| lambda_TG (Gradient loss)      | {self.lambda_TG:<25} |")
    #         print(f"| Number of Paths           | {self.n_paths:<25} |")
    #         print(f"| Batch Size                | {self.batch_size:<25} |")
    #         print(f"| Architecture              | {self.architecture:<25} |")
    #         print(f"| Depth                     | {len(self.Y_layers):<25} |")
    #         print(f"| Width                     | {self.Y_layers[0]:<25} |")
    #         print(f"| Activation                | {self.activation.__class__.__name__:<25} |")
    #         print(f"| T                         | {self.T:<25} |")
    #         print(f"| N                         | {self.N:<25} |")
    #         print(f"| Supervised                | {'True' if self.supervised else 'False':<25} |")
    #         print("+---------------------------+---------------------------+\n")

    #     for epoch in range(epochs):
    #         optimizer.zero_grad()
    #         t_paths, W_paths = self.fetch_minibatch()
    #         loss = self(t_paths, W_paths)
    #         loss.backward()
    #         optimizer.step()

    #         losses.append(loss.item())
    #         losses_Y.append(self.total_Y_loss)
    #         losses_terminal.append(self.terminal_loss)
    #         losses_terminal_gradient.append(self.terminal_gradient_loss)

    #         if adaptive:
    #             scheduler.step(loss.item())
    #         else:
    #             scheduler.step()

    #         if (epoch % K == 0 or epoch == epochs - 1) and verbose and epoch > 0:
    #             elapsed = time.time() - start_time
    #             if not header_printed:
    #                 print(f"{'Epoch':>8} | {'Total loss':>12} | {'Y loss':>12} | {'T. loss':>12} | {'T.G. loss':>12} | {'LR':>10} | {'Memory [MB]':>12} | {'Time [s]':>10} | {'Status'}")
    #                 print("-" * 120)
    #                 header_printed = True

    #             mem_mb = torch.cuda.memory_allocated() / 1e6
    #             current_lr = optimizer.param_groups[0]['lr']
    #             status = ""

    #             if "every" in self.save and (epoch % self.save_n == 0 or epoch == epochs - 1):
    #                 torch.save(self.state_dict(), save_path + ".pth")
    #                 status = f"Model saved ↓"

    #             if "best" in self.save and np.mean(losses[-K:]) < self.lowest_loss:
    #                 self.lowest_loss = np.mean(losses[-K:])
    #                 torch.save(self.state_dict(), save_path + "_best.pth")
    #                 status = "Model saved ↓ (best)"

    #             print(f"{epoch:8} | {np.mean(losses[-K:]):12.6f} | {np.mean(losses_Y[-K:]):12.6f} | {np.mean(losses_terminal[-K:]):12.6f} | {np.mean(losses_terminal_gradient[-K:]):12.6f} | {current_lr:10.2e} | {mem_mb:12.2f} | {elapsed:10.2f} | {status}")
    #             start_time = time.time()

    #     if "last" in self.save:
    #         torch.save(self.state_dict(), save_path + ".pth")
    #         status = f"Model saved ↓"
    #         print(f"{epoch:8} | {loss.item():12.6f} | {self.total_Y_loss:12.6f} | {self.terminal_loss:12.6f} | {self.terminal_gradient_loss:12.6f} | {current_lr:10.2e} | {mem_mb:12.2f} | {elapsed:10.2f} | {status}")

    #         t0, W0, y0 = t1, W1, y1

    #     Y_loss = 0.0
    #     for t_n, y_n_orig in zip(t_path, y_path):
    #         y_n = y_n_orig.detach()
    #         y_n.requires_grad_(True)

    #         yn_detached = y1.clone().detach().requires_grad_(True)
    #         V_target = self.value_function_analytic(t_n, yn_detached)
    #         dV_target = torch.autograd.grad(
    #             outputs=V_target,
    #             inputs=yn_detached,
    #             grad_outputs=torch.ones_like(V_target),
    #             create_graph=False,
    #             retain_graph=True
    #         )[0]

    #         V_pred = self.Y_net(t_n, y_n)
    #         dV_pred = torch.autograd.grad(
    #             outputs=V_pred,
    #             inputs=y_n,
    #             grad_outputs=torch.ones_like(V_pred),
    #             create_graph=True
    #         )[0]

    #         loss_Y = torch.mean(torch.pow(V_pred - V_target, 2))
    #         loss_dY = torch.mean(torch.pow(dV_pred - dV_target, 2))

    #         Y_loss += loss_Y + loss_dY

    #     YT = self.Y_net(t_n, y_n)
    #     terminal_loss = torch.mean(torch.pow(YT - self.terminal_cost(y_n), 2))

    #     dYT = torch.autograd.grad(
    #         outputs=YT,
    #         inputs=y_n,
    #         grad_outputs=torch.ones_like(YT),
    #         create_graph=True,
    #         retain_graph=True
    #     )[0]
    #     terminal_gradient_loss = torch.mean(torch.pow(dYT - self.terminal_cost_grad(y_n), 2))

    #     self.Y_loss = self.lambda_Y * Y_loss.detach().item()
    #     self.terminal_loss = self.lambda_T * terminal_loss.detach().item()
    #     self.terminal_gradient_loss = self.lambda_TG * terminal_gradient_loss.detach().item()

    #     return self.lambda_Y * Y_loss + self.lambda_T * terminal_loss + self.lambda_TG * terminal_gradient_loss
