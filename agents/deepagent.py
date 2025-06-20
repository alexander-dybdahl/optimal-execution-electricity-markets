import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.quasirandom import SobolEngine

from utils.logger import Logger
from utils.plots import plot_approx_vs_analytic, plot_approx_vs_analytic_expectation, plot_terminal_histogram
from utils.simulator import simulate_paths
from utils.nnets import (
    FCnet,
    FCnet_init,
    LSTMNet,
    LSTMWithSubnets,
    Resnet,
    SeparateSubnets,
    Sine,
    UncertaintyWeightedLoss,
)


class DeepAgent(nn.Module):
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
        self.sobol_points = args.sobol_points

        # Network Architecture
        self.architecture = args.architecture
        self.Y_layers = args.Y_layers      # e.g., [64, 64, 64]
        self.adaptive_factor = args.adaptive_factor
        self.strong_grad_output = args.strong_grad_output  # whether to use strong output gradient
        self.scale_output = args.scale_output  # how much to scale the output of the network

        # Loss Weights
        self.adaptive_loss = args.adaptive_loss
        self.loss_weights = {
            "lambda_Y0": args.lambda_Y0,
            "lambda_Y": args.lambda_Y,
            "lambda_dY": args.lambda_dY,
            "lambda_dYt": args.lambda_dYt,
            "lambda_T": args.lambda_T,
            "lambda_TG": args.lambda_TG,
            "lambda_pinn": args.lambda_pinn,
            "lambda_reg": args.lambda_reg,
            "lambda_cost": args.lambda_cost,
        }
        
        # Loss threshold for linear approximation
        self.loss_threshold = args.loss_threshold
        self.use_linear_approx = args.use_linear_approx
        self.second_order_taylor = args.second_order_taylor

        # Validation Losses
        self.val_q_loss = torch.tensor(0.0, device=self.device)
        self.val_Y_loss = torch.tensor(0.0, device=self.device)
        self.validation = {
            "q_loss": [],
            "Y_loss": []
        }

        # Loss Tracking
        self.Y0_loss = torch.tensor(0.0, device=self.device)
        self.Y_loss = torch.tensor(0.0, device=self.device)
        self.dY_loss = torch.tensor(0.0, device=self.device)
        self.dYt_loss = torch.tensor(0.0, device=self.device)
        self.terminal_loss = torch.tensor(0.0, device=self.device)
        self.terminal_gradient_loss = torch.tensor(0.0, device=self.device)
        self.pinn_loss = torch.tensor(0.0, device=self.device)
        self.reg_loss = torch.tensor(0.0, device=self.device)
        self.cost_loss = torch.tensor(0.0, device=self.device)

        # Saving & Checkpointing
        self.epoch = 0
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

        self.Y_init_net = FCnet_init(
            layers=[self.dynamics.dim] + args.Y0_layers + [1],
            activation=self.activation,
            y0=self.dynamics.y0,
            rescale_y0=args.rescale_y0,
            strong_grad_output=args.strong_grad_output,
            scale_output=args.scale_output,
        )

        if args.architecture == "default":
            self.dY_net = FCnet(
                layers=[self.dynamics.dim + 1] + [64, 64, 64, 64] + [self.dynamics.dim + 1],
                activation=self.activation,
                T=self.dynamics.T,
                input_bn=args.input_bn,
                affine=args.affine,
            )
        elif args.architecture == "fc":
            self.dY_net = FCnet(
                layers=[self.dynamics.dim + 1] + args.Y_layers + [self.dynamics.dim + 1],
                activation=self.activation,
                T=self.dynamics.T,
                input_bn=args.input_bn,
                affine=args.affine,
            )
        elif args.architecture == "naisnet":
            self.dY_net = Resnet(
                layers=[self.dynamics.dim + 1] + args.Y_layers + [self.dynamics.dim + 1],
                activation=self.activation,
                stable=True,
                T=self.dynamics.T,
                input_bn=args.input_bn,
                affine=args.affine,
            )
        elif args.architecture == "resnet":
            self.dY_net = Resnet(
                layers=[self.dynamics.dim + 1] + args.Y_layers + [self.dynamics.dim + 1],
                activation=self.activation,
                stable=False,
                T=self.dynamics.T,
                input_bn=args.input_bn,
                affine=args.affine,
            )
        elif (
            args.architecture == "lstm"
            or args.architecture == "reslstm"
            or args.architecture == "naislstm"
        ):
            self.dY_net = LSTMNet(
                layers=[self.dynamics.dim + 1] + args.Y_layers + [self.dynamics.dim + 1],
                activation=self.activation,
                type=args.architecture,
                T=self.dynamics.T,
                input_bn=args.input_bn,
                affine=args.affine,
            )
        elif args.architecture == "separatesubnets":
            # Get subnet configuration from args
            subnet_type = args.subnet_type
            self.dY_net = SeparateSubnets(
                layers=[self.dynamics.dim + 1] + args.Y_layers + [self.dynamics.dim + 1],
                activation=self.activation,
                num_time_steps=self.dynamics.N,
                T=self.dynamics.T,
                subnet_type=subnet_type,
                input_bn=args.input_bn,
                affine=args.affine,
            )
        elif args.architecture == "lstmwithsubnets":
            # Get LSTM and subnet configuration from args
            lstm_layers = args.lstm_layers
            lstm_type = args.lstm_type
            subnet_type = args.subnet_type
            self.dY_net = LSTMWithSubnets(
                layers=[self.dynamics.dim + 1] + args.Y_layers + [self.dynamics.dim + 1],
                activation=self.activation,
                num_time_steps=self.dynamics.N,
                T=self.dynamics.T,
                lstm_layers=lstm_layers,
                subnet_type=subnet_type,
                lstm_type=lstm_type,
                input_bn=args.input_bn,
                affine=args.affine,
            )
        else:
            raise ValueError(f"Unknown architecture: {args.architecture}")

        self.lowest_loss = float("inf")

    def hjb_residual(self, t, y):
        dY_outputs = self.dY_net(t, y)
        dY_dt = dY_outputs[:, 0:1]  # temporal derivative
        dY_dy = dY_outputs[:, 1:]   # spatial derivatives
        
        # Compute Hessian using autograd only on dY_dy (single autograd call)
        dim = y.shape[1]
        hessian_rows = [
            torch.autograd.grad(
                outputs=dY_dy[:, i],
                inputs=y,
                grad_outputs=torch.ones_like(dY_dy[:, i]),
                create_graph=True,
                retain_graph=True
            )[0] for i in range(dim)
        ]
        H = torch.stack(hessian_rows, dim=1)

        # Dynamics quantities
        q = self.dynamics.optimal_control(t, y, dY_outputs)
        mu = self.dynamics.mu(t, y, q)
        sigma = self.dynamics.sigma(t, y)
        f = self.dynamics.generator(y, q)

        # Diffusion term: ½ Tr[σσᵀ H]
        sigma_sigma_T = torch.bmm(sigma, sigma.transpose(1, 2))  # (batch, dim, dim)
        trace_term = torch.einsum("bij,bij->b", sigma_sigma_T, H).unsqueeze(-1)

        # Residual from HJB
        residual = dY_dt + (mu * dY_dy).sum(dim=1, keepdim=True) + 0.5 * trace_term + f
        return self.apply_thresholded_loss(residual).mean()

    def apply_thresholded_loss(self, diff):

        if not self.use_linear_approx:
            return diff.pow(2)
        
        large_diff_mask = (diff.abs() > self.loss_threshold).float()
        squared_loss = diff.pow(2)
        linear_loss = 2 * self.loss_threshold * diff.abs() - self.loss_threshold**2
        
        return large_diff_mask * linear_loss + (1 - large_diff_mask) * squared_loss
    
    def forward(self, t, dW):
        t0 = t[:, 0, :]
        y0 = self.dynamics.y0.repeat(self.batch_size, 1).to(self.device)

        if self.second_order_taylor:
            t0 = t0.requires_grad_(True)
            y0 = y0.requires_grad_(True)

        Y0 = self.Y_init_net(y0)
        Y0_init = Y0

        t_traj, y_traj, q_traj, Y_traj = [t0], [y0], [], [Y0]
        
        losses_dict = {}

        # === Compute FBSNN loss ===
        Y_loss = 0.0
        for n in range(self.dynamics.N):
            t1 = t[:, n + 1, :]
            dt = t1 - t0

            if self.second_order_taylor:
                t0 = t0.requires_grad_(True)
                y0 = y0.requires_grad_(True)
            
            dY_outputs = self.dY_net(t0, y0)
            dY_dt = dY_outputs[:, 0:1]
            dY_dy = dY_outputs[:, 1:]
            
            q = self.dynamics.optimal_control(t0, y0, dY_dy)
            y1 = self.dynamics.forward_dynamics(y0, q, dW[:, n, :], t0, dt)
            dy = y1 - y0
            Y1 = Y0 + dY_dt * dt + (dY_dy * dy).sum(dim=1, keepdim=True)

            Z0 = torch.bmm(self.dynamics.sigma(t0, y0).transpose(1, 2), dY_dy.unsqueeze(-1)).squeeze(-1)
            Y1_tilde = Y0 - self.dynamics.generator(y0, q) * dt + (Z0 * (dW[:, n, :])).sum(dim=1, keepdim=True)

            if self.second_order_taylor:
                # Second-order Taylor approximation
                ddY_ddt = torch.autograd.grad(
                    outputs=dY_dt,
                    inputs=t0,
                    grad_outputs=torch.ones_like(dY_dt),
                    create_graph=False,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                hvp = torch.autograd.grad(
                    outputs=(dY_dy * dy).sum(),
                    inputs=y0,
                    create_graph=False,
                    retain_graph=True
                )[0]

                # Handle cases where gradient with respect to t might be None due to separate subnet per time
                if ddY_ddt is None:
                    ddY_ddt = 0

                quadratic_term = (hvp * dy).sum(dim=1, keepdim=True)  # (delta_y)^T @ H @ delta_y
                Y1 += 0.5 * (ddY_ddt * dt ** 2 + quadratic_term)

            Y_loss += self.apply_thresholded_loss(Y1 - Y1_tilde).mean()

            t_traj.append(t1)
            y_traj.append(y1)
            q_traj.append(q)
            Y_traj.append(Y1)

            t0, y0, Y0 = t1, y1, Y1

        if self.loss_weights["lambda_Y"] > 0:
            losses_dict["lambda_Y"] = self.Y_loss
            self.Y_loss = Y_loss.detach()

        t_all = torch.cat(t_traj, dim=0).detach()
        y_all = torch.cat(y_traj, dim=0).detach()
        q_all = torch.cat(q_traj, dim=0).detach()
        Y_all = torch.cat(Y_traj, dim=0).detach()

        # === Terminal loss ===
        terminal_loss = 0.0
        if self.loss_weights["lambda_T"] > 0:
            terminal_loss = self.apply_thresholded_loss(Y0 - self.dynamics.terminal_cost(y0)).mean()
            losses_dict["lambda_T"] = terminal_loss
            self.terminal_loss = terminal_loss.detach()

        # === Terminal gradient loss ===
        terminal_gradient_loss = 0.0
        if self.loss_weights["lambda_TG"] > 0:
            terminal_gradient_loss = self.apply_thresholded_loss(dY_dy - self.dynamics.terminal_cost_grad(y0)).mean()
            losses_dict["lambda_TG"] = terminal_gradient_loss
            self.terminal_gradient_loss = terminal_gradient_loss.detach()

        # === Physics-based loss ===
        pinn_loss = 0.0
        if self.loss_weights["lambda_pinn"] > 0:
            t_pinn = torch.cat(t_traj, dim=0).requires_grad_(True)
            y_pinn = torch.cat(y_traj, dim=0).requires_grad_(True)

            # Sobol points for additional PINN loss
            if self.sobol_points:
                sobol = SobolEngine(dimension=self.dynamics.dim, scramble=True)
                sobol_points = sobol.draw(self.batch_size * self.dynamics.N).to(self.device)
                y0 = self.dynamics.y0.detach().flatten().cpu()  # shape: (dim,)
                T = self.dynamics.T
                std_mult = 3.0
                abs_padding = 10.0  # fallback for dimensions with no dynamics-based std
                y_min_list, y_max_list = [], []
                for i in range(self.dynamics.dim):
                    base = y0[i].item()
                    try:
                        # Construct dummy inputs for sigma (batch=1)
                        dummy_t = torch.zeros(1, 1, device=self.device)
                        dummy_y = self.dynamics.y0.detach()
                        sigma = self.dynamics.sigma(dummy_t, dummy_y)[0]  # shape: (dim, dW)
                        std_estimate = torch.norm(sigma[i]).item() * T**0.5
                        delta = std_mult * std_estimate
                    except Exception:
                        delta = abs_padding  # fallback for unknown std
                    y_min_list.append(base - delta)
                    y_max_list.append(base + delta)
                y_min = torch.tensor(y_min_list, device=self.device)
                y_max = torch.tensor(y_max_list, device=self.device)
                sobol_points = y_min + (y_max - y_min) * sobol_points
                t_sobol = torch.rand(self.batch_size * self.dynamics.N, 1, device=self.device) * T
                t_pinn = torch.cat([t_pinn, t_sobol], dim=0).requires_grad_(True)
                y_pinn = torch.cat([y_pinn, sobol_points], dim=0).requires_grad_(True)

            pinn_loss = self.hjb_residual(t_pinn, y_pinn)
            losses_dict["lambda_pinn"] = pinn_loss
            self.pinn_loss = pinn_loss.detach()

        # === q regularization loss ===
        reg_loss = 0.0
        if self.loss_weights["lambda_reg"] > 0:
            q_diffs = [q_traj[i+1] - q_traj[i] for i in range(len(q_traj) - 1)]
            dq_all = torch.cat(q_diffs, dim=0)
            reg_loss = self.apply_thresholded_loss(q_all).mean() + self.apply_thresholded_loss(dq_all).mean()
            losses_dict["lambda_reg"] = reg_loss
            self.reg_loss = reg_loss.detach()

        # === Cost objective ===
        cost_objective = 0.0
        if self.loss_weights["lambda_cost"] > 0:
            running_cost = 0.0
            dt = self.dynamics.dt
            for n in range(self.dynamics.N):
                start = n * self.batch_size
                end = (n + 1) * self.batch_size
                y_n = y_all[start:end]
                q_n = q_all[start:end]
                f_n = self.dynamics.generator(y_n, q_n)
                running_cost += f_n * dt

            terminal_cost = self.dynamics.terminal_cost(y_all[-self.batch_size:])  # final time step
            cost_objective = (running_cost + terminal_cost).mean()
            losses_dict["lambda_cost"] = cost_objective
            self.cost_loss = cost_objective.detach()

        # === Y0 fbsnn loss ===
        Y0_loss = 0.0
        if self.loss_weights["lambda_Y0"] > 0:
            Y0_loss = self.apply_thresholded_loss(Y0_init - self.cost_loss).mean()
            losses_dict["lambda_Y0"] = Y0_loss
            self.Y0_loss = Y0_loss.detach()

        # === Y and q validation loss ===
        if self.dynamics.analytical_known:
            Y_true = self.dynamics.value_function_analytic(t_all, y_all)
            Y_val_loss = self.apply_thresholded_loss(Y_all - Y_true).mean()
            self.val_Y_loss = Y_val_loss.detach()

            q_true = self.dynamics.optimal_control_analytic(t_all[:-self.batch_size], y_all[:-self.batch_size])
            q_val_loss = self.apply_thresholded_loss(q_all - q_true).mean()
            self.val_q_loss = q_val_loss.detach()

        # === Total loss computation ===
        if self.adaptive_loss:
            loss_fn = UncertaintyWeightedLoss(self.loss_weights)
            total_loss = loss_fn(losses_dict)
        else:
            total_loss = sum(
                self.loss_weights[key] * value
                for key, value in losses_dict.items()
            ) 

        # === Supervised loss ===
        if self.dynamics.analytical_known and self.supervised:
            total_loss += Y_val_loss
            total_loss += q_val_loss

        return total_loss

    def predict_Y_initial(self, y0):
        self.Y_init_net.eval()
        with torch.no_grad():
            Y0 = self.Y_init_net(y0)
        self.Y_init_net.train()
        return Y0
    
    def predict_Y_next(self, t0, y0, dt, dy, Y0):
        self.dY_net.eval()
        if self.second_order_taylor:
            t0 = t0.requires_grad_(True)
            y0 = y0.requires_grad_(True)
            dY_outputs = self.dY_net(t0, y0)
        else:
            with torch.no_grad():
                dY_outputs = self.dY_net(t0, y0)
        self.dY_net.train()

        dY_dt = dY_outputs[:, 0:1]
        dY_dy = dY_outputs[:, 1:]
        
        Y1 = Y0 + dY_dt * dt + (dY_dy * dy).sum(dim=1, keepdim=True)

        if self.second_order_taylor:
            # Second-order Taylor approximation
            ddY_ddt_tilde = torch.autograd.grad(
                outputs=dY_dt,
                inputs=t0,
                grad_outputs=torch.ones_like(dY_dt),
                create_graph=False,
                retain_graph=True,
                allow_unused=True,
            )[0]
            ddY_ddy_tilde = torch.autograd.grad(
                outputs=dY_dy,
                inputs=y0,
                grad_outputs=torch.ones_like(dY_dy),
                create_graph=False,
                retain_graph=True,
            )[0]

            # Handle cases where gradient with respect to t might be None due to separate subnet per time
            if ddY_ddt_tilde is None:
                ddY_ddt_tilde = 0

            Y1 += 0.5 * (ddY_ddt_tilde * dt ** 2 + (ddY_ddy_tilde * dy ** 2).sum(dim=1, keepdim=True))

        return Y1

    def predict(self, t, y):
        self.dY_net.eval()
        with torch.no_grad():
            dY_outputs = self.dY_net(t, y)
        self.dY_net.train()
        
        dY_dy = dY_outputs[:, 1:]
        return self.dynamics.optimal_control(t, y, dY_dy)

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
            "val loss": 10 if self.dynamics.analytical_known else 0,
            "loss": 8,
            "lr": 10,
            "mem": 12,
            "time": 8,
            "eta": 10,
            "status": 20,
        }

        (
            losses,
            losses_Y0,
            losses_Y,
            losses_dY,
            losses_dYt,
            losses_terminal,
            losses_terminal_gradient,
            losses_pinn,
            losses_reg,
            losses_cost,
        ) = [], [], [], [], [], [], [], [], [], []

        # Define active loss components
        loss_components = [
            ("Y0 loss", self.loss_weights["lambda_Y0"], lambda: np.mean(losses_Y0[-K:])),
            ("Y loss", self.loss_weights["lambda_Y"], lambda: np.mean(losses_Y[-K:])),
            ("dY loss", self.loss_weights["lambda_dY"], lambda: np.mean(losses_dY[-K:])),
            ("dYt loss", self.loss_weights["lambda_dYt"], lambda: np.mean(losses_dYt[-K:])),
            ("T. loss", self.loss_weights["lambda_T"], lambda: np.mean(losses_terminal[-K:])),
            ("T.G. loss", self.loss_weights["lambda_TG"], lambda: np.mean(losses_terminal_gradient[-K:])),
            ("pinn loss", self.loss_weights["lambda_pinn"], lambda: np.mean(losses_pinn[-K:])),
            ("reg loss", self.loss_weights["lambda_reg"], lambda: np.mean(losses_reg[-K:])),
            ("cost loss", self.loss_weights["lambda_cost"], lambda: np.mean(losses_cost[-K:])),
        ]
        active_losses = [
            (name, fn) for name, lambda_, fn in loss_components if lambda_ > 0
        ]

        # Calculate max widths
        for name, _ in active_losses:
            max_widths[name] = max(10, len(name) + 2)
        width = (
            max_widths["epoch"]
            + 2 * max_widths["val loss"]
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
            logger.log(f"| lambda_Y0 (Y0 loss)       | {self.loss_weights['lambda_Y0']:<25} |")
            logger.log(f"| lambda_Y (Y loss)         | {self.loss_weights['lambda_Y']:<25} |")
            logger.log(f"| lambda_dY (Spatial loss)  | {self.loss_weights['lambda_dY']:<25} |")
            logger.log(f"| lambda_dYt (Temporal loss)| {self.loss_weights['lambda_dYt']:<25} |")
            logger.log(f"| lambda_T (Terminal loss)  | {self.loss_weights['lambda_T']:<25} |")
            logger.log(f"| lambda_TG (Gradient loss) | {self.loss_weights['lambda_TG']:<25} |")
            logger.log(f"| lambda_pinn (PINN loss)   | {self.loss_weights['lambda_pinn']:<25} |")
            logger.log(f"| lambda_reg (Reg loss)     | {self.loss_weights['lambda_reg']:<25} |")
            logger.log(f"| lambda_cost (Cost loss)   | {self.loss_weights['lambda_cost']:<25} |")
            logger.log(f"| Batch Size per Epoch      | {self.batch_size * self.world_size:<25} |")
            logger.log(f"| Batch Size per Rank       | {self.batch_size:<25} |")
            logger.log(f"| Architecture              | {self.architecture.upper():<25} |")
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
                f"{'Y val':>{max_widths['val loss']}}" if self.dynamics.analytical_known else "",
                f"{'q val':>{max_widths['val loss']}}" if self.dynamics.analytical_known else "",
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
        current_lr = optimizer.param_groups[0]["lr"]
        lr_decay_epochs = []

        for epoch in range(1, epochs + 1):
            self.epoch = epoch
            optimizer.zero_grad()
            t, dW, _ = self.dynamics.generate_paths(self.batch_size)
            loss = self(t=t, dW=dW)
            loss.backward()
            optimizer.step()
            scheduler.step(loss.item() if adaptive else epoch)

            # Check if the LR was reduced
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr < current_lr:
                lr_decay_epochs.append(epoch)
                current_lr = new_lr

            # Reduce losses across all processes if distributed
            val_Y_loss = self.val_Y_loss
            val_q_loss = self.val_q_loss
            Y0_loss = self.Y0_loss
            Y_loss = self.Y_loss
            dY_loss = self.dY_loss
            dYt_loss = self.dYt_loss
            terminal_loss = self.terminal_loss
            terminal_gradient_loss = self.terminal_gradient_loss
            pinn_loss = self.pinn_loss
            reg_loss = self.reg_loss
            cost_loss = self.cost_loss

            if self.is_distributed:
                dist.reduce(val_Y_loss, op=dist.ReduceOp.SUM, dst=0)
                dist.reduce(val_q_loss, op=dist.ReduceOp.SUM, dst=0)
                dist.reduce(loss, op=dist.ReduceOp.SUM, dst=0)
                dist.reduce(Y0_loss, op=dist.ReduceOp.SUM, dst=0)
                dist.reduce(Y_loss, op=dist.ReduceOp.SUM, dst=0)
                dist.reduce(dY_loss, op=dist.ReduceOp.SUM, dst=0)
                dist.reduce(dYt_loss, op=dist.ReduceOp.SUM, dst=0)
                dist.reduce(terminal_loss, op=dist.ReduceOp.SUM, dst=0)
                dist.reduce(terminal_gradient_loss, op=dist.ReduceOp.SUM, dst=0)
                dist.reduce(pinn_loss, op=dist.ReduceOp.SUM, dst=0)
                dist.reduce(reg_loss, op=dist.ReduceOp.SUM, dst=0)
                dist.reduce(cost_loss, op=dist.ReduceOp.SUM, dst=0)

                val_Y_loss /= self.world_size
                val_q_loss /= self.world_size
                loss /= self.world_size
                Y0_loss /= self.world_size
                Y_loss /= self.world_size
                dY_loss /= self.world_size
                dYt_loss /= self.world_size
                terminal_loss /= self.world_size
                terminal_gradient_loss /= self.world_size
                pinn_loss /= self.world_size
                reg_loss /= self.world_size
                cost_loss /= self.world_size

            if self.is_main:
                self.validation["Y_loss"].append(val_Y_loss.item())
                self.validation["q_loss"].append(val_q_loss.item())
                losses.append(loss.item())
                losses_Y0.append(Y0_loss.item())
                losses_Y.append(Y_loss.item())
                losses_dY.append(dY_loss.item())
                losses_dYt.append(dYt_loss.item())
                losses_terminal.append(terminal_loss.item())
                losses_terminal_gradient.append(terminal_gradient_loss.item())
                losses_pinn.append(pinn_loss.item())
                losses_reg.append(reg_loss.item())
                losses_cost.append(cost_loss.item())

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
                        f"{np.mean(self.validation['Y_loss'][-K:]):>{max_widths['val loss']}.2e}" if self.dynamics.analytical_known else "",
                        f"{np.mean(self.validation['q_loss'][-K:]):>{max_widths['val loss']}.2e}" if self.dynamics.analytical_known else "",
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

                if epoch % self.plot_n == 0:
                    timesteps, results = simulate_paths(dynamics=self.dynamics, agent=self, n_sim=self.n_simulations, seed=42)
                    
                    plot_approx_vs_analytic(
                        results = results,
                        timesteps = timesteps,
                        validation = self.validation,
                        plot=False,
                        save_dir=save_dir,
                        num=epoch
                    )
                    timesteps, results = simulate_paths(dynamics=self.dynamics, agent=self, n_sim=1000, seed=42)
                    # plot_approx_vs_analytic_expectation(results, timesteps, plot=False, save_dir=save_dir, num=epoch)
                    plot_terminal_histogram(results=results, dynamics=self.dynamics, plot=False, save_dir=save_dir, num=epoch)

                if epoch % 1000 == 0 and epoch > 0:
                    # === Plot training losses ===
                    plt.figure(figsize=(12, 8))
                    plt.plot(losses, label="Total Loss")
                    if self.loss_weights["lambda_Y0"] > 0: plt.plot(losses_Y0, label="Y0 Loss")
                    if self.loss_weights["lambda_Y"] > 0: plt.plot(losses_Y, label="Y Loss")
                    if self.loss_weights["lambda_dY"] > 0: plt.plot(losses_dY, label="dY Loss")
                    if self.loss_weights["lambda_dYt"] > 0: plt.plot(losses_dYt, label="dYt Loss")
                    if self.loss_weights["lambda_T"] > 0: plt.plot(losses_terminal, label="Terminal Loss")
                    if self.loss_weights["lambda_TG"] > 0: plt.plot(losses_terminal_gradient, label="Terminal Gradient Loss")
                    if self.loss_weights["lambda_pinn"] > 0: plt.plot(losses_pinn, label="PINN Loss")
                    if self.loss_weights["lambda_reg"] > 0: plt.plot(losses_reg, label="Regularization Loss")
                    if self.loss_weights["lambda_cost"] > 0: plt.plot(losses_cost, label="Cost Loss")

                    for decay_epoch in lr_decay_epochs:
                        plt.axvline(x=decay_epoch, color='red', linestyle='--', alpha=0.6,
                                    label='LR Drop' if decay_epoch == lr_decay_epochs[0] else "")

                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.title("Training Loss")
                    plt.yscale("symlog", linthresh=1e-1)  # Use symlog scale with linear region near 0
                    plt.axhline(0, color='black', linestyle='--', linewidth=1)
                    plt.legend()
                    plt.grid(True, which='both', linestyle='--', alpha=0.4)
                    plt.tight_layout()
                    if save_dir:
                        plt.savefig(f"{save_dir}/imgs/loss_{epoch}.png", dpi=300, bbox_inches="tight")
                        plt.close()
                    if plot:
                        plt.show()

                    # === Plot validation losses ===
                    if self.dynamics.analytical_known:
                        plt.figure(figsize=(12, 8))
                        plt.plot(self.validation["q_loss"], label="Validation q Loss")
                        plt.plot(self.validation["Y_loss"], label="Validation Y Loss")
                        plt.plot(losses, label="Total Loss")
                        plt.xlabel("Epoch")
                        plt.ylabel("Loss")
                        plt.title("Validation Losses")
                        plt.yscale("symlog", linthresh=1e-1)  # same symlog scale
                        plt.axhline(0, color='black', linestyle='--', linewidth=1)
                        plt.legend()
                        plt.grid(True, which='both', linestyle='--', alpha=0.4)
                        plt.tight_layout()
                        if save_dir:
                            plt.savefig(f"{save_dir}/imgs/val_loss_{epoch}.png", dpi=300, bbox_inches="tight")
                            plt.close()
                        if plot:
                            plt.show()

        if self.is_main:
            if self.is_distributed:
                dist.barrier()

            logger.log(f"Training completed. Lowest loss: {self.lowest_loss:.6f}. Total time: {time.time() - init_time:.2f} seconds")
            logger.log(f"Model saved to {save_path}.pth")

        return losses