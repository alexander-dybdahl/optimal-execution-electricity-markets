import os
import time

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.quasirandom import SobolEngine

from core.nnets import FCnet, LSTMNet, Resnet, Sine
from utils.logger import Logger


class FBSNN(nn.Module):
    def __init__(self, dynamics, args):
        super().__init__()
        # System & Execution Settings
        self.is_distributed = dist.is_initialized()
        self.device = args.device_set
        self.world_size = dist.get_world_size() if self.is_distributed else 1
        self.is_main = not self.is_distributed or dist.get_rank() == 0

        # Underlying dynamics
        self.dynamics = dynamics

        # Data & Batch Settings
        self.batch_size = args.batch_size_per_rank
        self.supervised = args.supervised
        if self.supervised and not self.dynamics.analytical_known:
            raise ValueError("Cannot proceed with supervised training when analytical value function is not know")

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

        # Saving & Checkpointing
        self.save = args.save              # e.g., "best", "every", "last"
        self.save_n = args.save_n          # save every n epochs if "every"
        self.plot_n = args.plot_n          # save every n epochs if "every"
        self.n_simulations = args.n_simulations

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
                layers=[self.dynamics.dim + 1] + [64, 64, 64, 64, 1], activation=self.activation
            ).to(self.device)
        elif args.architecture == "FC":
            self.Y_net = FCnet(
                layers=[self.dynamics.dim + 1] + args.Y_layers, activation=self.activation
            ).to(self.device)
        elif args.architecture == "NAISnet":
            self.Y_net = Resnet(
                layers=[self.dynamics.dim + 1] + args.Y_layers,
                activation=self.activation,
                stable=True,
            ).to(self.device)
        elif args.architecture == "Resnet":
            self.Y_net = Resnet(
                layers=[self.dynamics.dim + 1] + args.Y_layers,
                activation=self.activation,
                stable=False,
            ).to(self.device)
        elif (
            args.architecture == "LSTM"
            or args.architecture == "ResLSTM"
            or args.architecture == "NaisLSTM"
        ):
            self.Y_net = LSTMNet(
                layers=[self.dynamics.dim + 1] + args.Y_layers,
                activation=self.activation,
                type=args.architecture,
            ).to(self.device)
        else:
            raise ValueError(f"Unknown architecture: {args.architecture}")

        self.lowest_loss = float("inf")

    def sample_latin_hypercube(self, device, N, t_range, y_range, dim_y):
        # Sobol sampling gives better stratification than pure random
        sobol = SobolEngine(dimension=dim_y + 1, scramble=True)
        samples = sobol.draw(N).to(device)

        # Scale t and y
        t_samples = t_range[0] + (t_range[1] - t_range[0]) * samples[:, [0]]
        y_samples = y_range[0] + (y_range[1] - y_range[0]) * samples[:, 1:]
        return t_samples, y_samples
    
    def physics_loss(self, t, y, V):
        dim = y.shape[1]

        # Ensure gradients
        t = t if t.requires_grad else t.requires_grad_(True)
        y = y if y.requires_grad else y.requires_grad_(True)

        # First derivatives
        dV_dy, dV_dt = torch.autograd.grad(
            outputs=V,
            inputs=(y, t),
            grad_outputs=torch.ones_like(V),
            create_graph=True,
            retain_graph=True
        )

        # Compute full Hessian: ∇²_y V (batch, dim, dim)
        hessian_rows = [
            torch.autograd.grad(
                outputs=dV_dy[:, i],
                inputs=y,
                grad_outputs=torch.ones_like(dV_dy[:, i]),
                create_graph=True,
                retain_graph=True
            )[0] for i in range(dim)
        ]
        H = torch.stack(hessian_rows, dim=1)

        # Dynamics quantities
        q = self.dynamics.optimal_control(t, y, V)
        mu = self.dynamics.mu(t, y, q)
        sigma = self.dynamics.sigma(t, y)
        f = self.dynamics.generator(y, q)

        # Diffusion term: ½ Tr[σσᵀ H]
        sigma_sigma_T = torch.bmm(sigma, sigma.transpose(1, 2))  # (batch, dim, dim)
        trace_term = torch.einsum("bij,bij->b", sigma_sigma_T, H).unsqueeze(-1)

        # Residual from HJB
        residual = dV_dt + (mu * dV_dy).sum(dim=1, keepdim=True) + 0.5 * trace_term + f
        return (residual**2).sum()

    def forward(self, t_paths, W_paths, supervised=False):
        if supervised:
            return self.forward_supervised(t_paths, W_paths)
        else:
            return self.forward_fc(t_paths, W_paths)

    def forward_fc(self, t_paths, W_paths):
        y0 = self.dynamics.y0.repeat(self.batch_size, 1).to(self.device)
        y_traj, t_traj = [y0], []
        t0 = t_paths[:, 0, :]
        W0 = W_paths[:, 0, :]
        Y0 = self.Y_net(t0, y0)

        dY0 = torch.autograd.grad(
            outputs=Y0,
            inputs=y0,
            grad_outputs=torch.ones_like(Y0),
            create_graph=True,
            retain_graph=True,
        )[0]

        # === Compute fbsnn loss ===
        Y_loss = 0.0
        for n in range(self.dynamics.N):
            t1 = t_paths[:, n + 1, :]
            W1 = W_paths[:, n + 1, :]
            Sigma0 = self.dynamics.sigma(t0, y0)
            Z0 = torch.bmm(Sigma0.transpose(1, 2), dY0.unsqueeze(-1)).squeeze(-1)
            q = self.dynamics.optimal_control(t0, y0, Y0)
            y1 = self.dynamics.forward_dynamics(y0, q, W1 - W0, t0, t1 - t0)

            Y1 = self.Y_net(t1, y1)
            dY1 = torch.autograd.grad(
                outputs=Y1,
                inputs=y1,
                grad_outputs=torch.ones_like(Y1),
                create_graph=True,
                retain_graph=True,
            )[0]

            f = self.dynamics.generator(y0, q)
            Y1_tilde = Y0 - f * (t1 - t0) + (Z0 * (W1 - W0)).sum(dim=1, keepdim=True)
            Y_loss += torch.sum(torch.pow(Y1 - Y1_tilde, 2))

            # print out Y and Y_tilde for debugging
            print(f"Y1: {Y1.mean().item()}, Y1_tilde: {Y1_tilde.mean().item()}")

            t_traj.append(t1)
            y_traj.append(y1)

            t0, W0, y0, Y0, dY0 = t1, W1, y1, Y1, dY1

        self.Y_loss = self.lambda_Y * Y_loss.detach()

        # === Terminal supervision ===
        terminal_loss, terminal_gradient_loss = 0.0, 0.0
        if self.lambda_T > 0:
            YT = self.Y_net(t1, y1)
            terminal_loss = torch.sum(torch.pow(YT - self.dynamics.terminal_cost(y1), 2))
            self.terminal_loss = self.lambda_T * terminal_loss.detach()
        if self.lambda_TG > 0:
            dYT = torch.autograd.grad(
                YT, 
                y1, 
                grad_outputs=torch.ones_like(YT), 
                create_graph=True
            )[0]
            terminal_gradient_loss = torch.sum(torch.pow(dYT - self.dynamics.terminal_cost_grad(y1), 2))
            self.terminal_gradient_loss = self.lambda_TG * terminal_gradient_loss.detach()

        # === Physics-based loss ===
        t_traj = torch.cat(t_traj, dim=0).requires_grad_(True)            # shape: [N * batch_size, 1]
        y_traj = torch.cat(y_traj[1:], dim=0).requires_grad_(True)        # shape: [N * batch_size, state_dim]

        pinn_loss = 0.0
        if self.lambda_pinn > 0:
            N_pinn = self.batch_size * self.dynamics.N
            t_pinn, y_pinn = self.sample_latin_hypercube(
                device=self.device,
                N=N_pinn,
                t_range=(0.0, self.dynamics.T),
                y_range=(-10.0, 10.0),  # adjust to your problem's domain
                dim_y=self.dynamics.dim,
            )
            t_pinn.requires_grad_(True)
            y_pinn.requires_grad_(True)

            with torch.enable_grad():
                V = self.Y_net(t_pinn, y_pinn)
                pinn_loss = self.physics_loss(t_pinn, y_pinn, V)

            self.pinn_loss = self.lambda_pinn * pinn_loss.detach()

        return (
            self.lambda_Y * Y_loss
            + self.lambda_T * terminal_loss
            + self.lambda_TG * terminal_gradient_loss
            + self.lambda_pinn * pinn_loss
        )

    def forward_supervised(self, t_paths, W_paths):
        y0 = self.dynamics.y0.repeat(self.batch_size, 1).to(self.device)

        # === Precompute optimal trajectory ===
        y_traj, t_traj = [y0], []
        t0 = t_paths[:, 0, :]
        W0 = W_paths[:, 0, :]

        for n in range(self.dynamics.N):
            t1 = t_paths[:, n + 1, :]
            W1 = W_paths[:, n + 1, :]
            dW = W1 - W0

            V = self.dynamics.value_function_analytic(t0, y0)
            q = self.dynamics.optimal_control(t0, y0, V, create_graph=False)
            y1 = self.dynamics.forward_dynamics(y0, q, dW, t0, t1 - t0)

            t_traj.append(t1)
            y_traj.append(y1)

            t0, W0, y0 = t1, W1, y1

        t_traj = torch.cat(t_traj, dim=0).requires_grad_(
            True
        )                                   # shape: (N * batch_size, 1)
        y_traj = torch.cat(y_traj[1:], dim=0).requires_grad_(
            True
        )                                   # shape: (N * batch_size, dim)

        # === Y loss ===
        V_target = self.dynamics.value_function_analytic(t_traj, y_traj)
        V_pred = self.Y_net(t_traj, y_traj)

        Y_loss = torch.sum(torch.pow(V_pred - V_target, 2))
        if self.lambda_Y > 0:
            self.Y_loss = self.lambda_Y * Y_loss.detach()

        # === dY loss ===
        dY_loss = 0.0
        if self.lambda_dY > 0:
            dV_target = torch.autograd.grad(
                V_target,
                y_traj,
                grad_outputs=torch.ones_like(V_target),
                create_graph=False,
                retain_graph=True,
            )[0]
            dV_pred = torch.autograd.grad(
                V_pred,
                y_traj,
                grad_outputs=torch.ones_like(V_pred),
                create_graph=True
            )[0]
            dY_loss = torch.sum(torch.pow(dV_pred - dV_target, 2))
            self.dY_loss = self.lambda_dY * dY_loss.detach()

        # === dYt loss ===
        dYt_loss = 0.0
        if self.lambda_dYt > 0:
            dV_target_t = torch.autograd.grad(
                V_target, 
                t_traj, 
                grad_outputs=torch.ones_like(V_target),
                create_graph=False, 
                retain_graph=True
            )[0]
            dV_pred_t = torch.autograd.grad(
                V_pred, 
                t_traj, 
                grad_outputs=torch.ones_like(V_pred),
                create_graph=True
            )[0]
            dYt_loss = torch.sum(torch.pow(dV_pred_t - dV_target_t, 2))
            self.dYt_loss = self.lambda_dYt * dYt_loss.detach()

        # === Terminal loss ===
        terminal_loss = 0.0
        if self.lambda_T > 0:
            YT = self.Y_net(t1, y1)
            terminal_loss = torch.sum(torch.pow(YT - self.dynamics.terminal_cost(y1), 2))
            self.terminal_loss = self.lambda_T * terminal_loss.detach()

        # === Terminal gradient loss ===
        terminal_gradient_loss = 0.0
        if self.lambda_TG > 0:
            dYT = torch.autograd.grad(
                YT, 
                y1, 
                grad_outputs=torch.ones_like(YT), 
                create_graph=True
            )[0]
            terminal_gradient_loss = torch.sum(torch.pow(dYT - self.dynamics.terminal_cost_grad(y1), 2))
            self.terminal_gradient_loss = self.lambda_TG * terminal_gradient_loss.detach()

        # === Physics-informed loss ===
        pinn_loss = 0.0
        if self.lambda_pinn > 0:
            with torch.enable_grad():
                for i in range(self.dynamics.N):
                    idx = slice(i * self.batch_size, (i + 1) * self.batch_size)
                    t_i = t_traj[idx].detach().clone().requires_grad_(True)
                    y_i = y_traj[idx].detach().clone().requires_grad_(True)
                    V_i = self.Y_net(t_i, y_i)
                    pinn_loss += self.physics_loss(t_i, y_i, V_i)
                pinn_loss /= self.dynamics.N
                self.pinn_loss = self.lambda_pinn * pinn_loss.detach()

        return (
            self.lambda_Y * Y_loss
            + self.lambda_dY * dY_loss
            + self.lambda_dYt * dYt_loss
            + self.lambda_T * terminal_loss
            + self.lambda_TG * terminal_gradient_loss
            + self.lambda_pinn * pinn_loss
        )
    
    def predict(self, t_tensor, y_tensor):
        return self.Y_net(t_tensor, y_tensor)

    def fetch_minibatch(self, batch_size):
        dim_W = self.dynamics.dim_W
        T = self.dynamics.T
        N = self.dynamics.N
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

    def save_model(self, save_path):
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
            logger.log(f"| lambda_dY (Spatial loss)  | {self.lambda_dY:<25} |")
            logger.log(f"| lambda_dYt (Temporal loss)| {self.lambda_dYt:<25} |")
            logger.log(f"| lambda_T (Terminal loss)  | {self.lambda_T:<25} |")
            logger.log(f"| lambda_TG (Gradient loss) | {self.lambda_TG:<25} |")
            logger.log(f"| lambda_pinn (PINN loss)   | {self.lambda_pinn:<25} |")
            logger.log(f"| Batch Size per Epoch      | {self.batch_size * self.world_size:<25} |")
            logger.log(f"| Batch Size per Rank       | {self.batch_size:<25} |")
            logger.log(f"| Architecture              | {self.architecture:<25} |")
            logger.log(f"| Depth                     | {len(self.Y_layers):<25} |")
            logger.log(f"| Width                     | {self.Y_layers[0]:<25} |")
            logger.log(f"| Activation                | {self.activation.__class__.__name__:<25} |")
            logger.log(f"| T                         | {self.dynamics.T:<25} |")
            logger.log(f"| N                         | {self.dynamics.N:<25} |")
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
            loss = self(t_paths=t_paths, W_paths=W_paths, supervised=self.supervised)
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
                        status = self.save_model(save_path)

                    if "best" in self.save and np.mean(losses[-K:]) < self.lowest_loss:
                        self.lowest_loss = np.mean(losses[-K:])
                        status = self.save_model(save_path + "_best") + " (best)"

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

                if self.plot_n is not None and epoch % self.plot_n == 0:
                    timesteps, results = self.dynamics.simulate_paths(agent=self, n_sim=self.n_simulations, seed=42)
                    self.dynamics.plot_approx_vs_analytic(results, timesteps, plot=False, save_dir=save_dir, num=epoch)
                    timesteps, results = self.dynamics.simulate_paths(agent=self, n_sim=1000, seed=42)
                    self.dynamics.plot_approx_vs_analytic_expectation(results, timesteps, plot=False, save_dir=save_dir, num=epoch)
                    self.dynamics.plot_terminal_histogram(results, plot=False, save_dir=save_dir, num=epoch)

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
                plt.close()
            if plot:
                plt.show()

        return losses