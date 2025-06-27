import os
import time

import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.quasirandom import SobolEngine

from utils.logger import Logger
from utils.plots import plot_approx_vs_analytic, plot_approx_vs_analytic_expectation, plot_terminal_histogram
from utils.simulator import simulate_paths, compute_cost_objective
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
    def __init__(self, dynamics, model_cfg, device):
        super().__init__()
        # System & Execution Settings
        self.is_distributed = dist.is_initialized()
        self.device = torch.device(device)
        self.world_size = dist.get_world_size() if self.is_distributed else 1
        self.is_main = not self.is_distributed or dist.get_rank() == 0

        # Underlying Dynamics
        self.dynamics = dynamics

        # Data & Batch Settings
        self.batch_size = model_cfg["batch_size_per_rank"]
        self.supervised = model_cfg["supervised"]
        if self.supervised and not self.dynamics.analytical_known:
            raise ValueError("Cannot proceed with supervised training when analytical value function is not know")
        self.sobol_points = model_cfg["sobol_points"]

        # Network Architecture
        self.network_type = model_cfg["network_type"]
        self.architecture = model_cfg["architecture"]
        self.Y_layers = model_cfg["Y_layers"]      # e.g., [64, 64, 64]
        self.adaptive_factor = model_cfg["adaptive_factor"]
        self.strong_grad_output = model_cfg["strong_grad_output"]  # whether to use strong output gradient
        self.scale_output = model_cfg["scale_output"]  # how much to scale the output of the network
        self.careful_init = model_cfg["careful_init"]  # whether to use careful initialization
        self.detach_control = model_cfg["detach_control"]  # whether to detach the control from the network output

        # Loss and Loss Weights
        self.lowest_loss = float("inf")
        self.annealing = model_cfg["annealing"]
        self.adaptive_loss = model_cfg["adaptive_loss"]
        self.loss_weights = {
            "lambda_Y0": model_cfg["lambda_Y0"],
            "lambda_Y": model_cfg["lambda_Y"],
            "lambda_dY": model_cfg["lambda_dY"],
            "lambda_dYt": model_cfg["lambda_dYt"],
            "lambda_T": model_cfg["lambda_T"],
            "lambda_TG": model_cfg["lambda_TG"],
            "lambda_pinn": model_cfg["lambda_pinn"],
            "lambda_reg": model_cfg["lambda_reg"],
            "lambda_cost": model_cfg["lambda_cost"],
        }
        
        # Loss threshold for linear approximation
        self.loss_threshold = model_cfg["loss_threshold"]
        self.use_linear_approx = model_cfg["use_linear_approx"]
        self.second_order_taylor = model_cfg["second_order_taylor"]

        # Validation Losses
        self.val_q_loss = torch.tensor(0.0, device=self.device)
        self.val_Y_loss = torch.tensor(0.0, device=self.device)
        self.validation = {
            "q_loss": [],
            "Y_loss": []
        } if self.dynamics.analytical_known else None

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
        self.reset_lr = model_cfg["reset_lr"]       # whether to reset the learning rate
        self.reset_best = model_cfg["reset_best"]  # whether to reset the best loss
        self.save = model_cfg["save"]               # e.g., "best", "every", "last"
        self.save_n = model_cfg["save_n"]           # save every n epochs if "every"
        self.plot_n = model_cfg["plot_n"]           # save every n epochs if "every"
        self.n_simulations = model_cfg["n_simulations"]

        if model_cfg["activation"] == "Sine":
            self.activation = Sine()
        elif model_cfg["activation"] == "ReLU":
            self.activation = nn.ReLU()
        elif model_cfg["activation"] == "Tanh":
            self.activation = nn.Tanh()
        elif model_cfg["activation"] == "LeakyReLU":
            self.activation = nn.LeakyReLU()
        elif model_cfg["activation"] == "ELU":
            self.activation = nn.ELU()
        elif model_cfg["activation"] == "Softplus":
            self.activation = nn.Softplus()
        elif model_cfg["activation"] == "Softsign":
            self.activation = nn.Softsign()
        elif model_cfg["activation"] == "GELU":
            self.activation = nn.GELU()
        elif model_cfg["activation"] == "Sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation function: {model_cfg['activation']}]")

        if model_cfg["architecture"] == "default":
            internal_net = FCnet(
                layers=[self.dynamics.dim + 1] + [64, 64, 64, 64] + [self.dynamics.dim + 1],
                activation=self.activation,
                T=self.dynamics.T,
                input_bn=model_cfg["input_bn"],
                affine=model_cfg["affine"],
            )
        elif model_cfg["architecture"] == "fc":
            internal_net = FCnet(
                layers=[self.dynamics.dim + 1] + model_cfg["Y_layers"] + [self.dynamics.dim + 1],
                activation=self.activation,
                T=self.dynamics.T,
                input_bn=model_cfg["input_bn"],
                affine=model_cfg["affine"],
            )
        elif model_cfg["architecture"] == "naisnet":
            internal_net = Resnet(
                layers=[self.dynamics.dim + 1] + model_cfg["Y_layers"] + [self.dynamics.dim + 1],
                activation=self.activation,
                stable=True,
                T=self.dynamics.T,
                input_bn=model_cfg["input_bn"],
                affine=model_cfg["affine"],
            )
        elif model_cfg["architecture"] == "resnet":
            internal_net = Resnet(
                layers=[self.dynamics.dim + 1] + model_cfg["Y_layers"] + [self.dynamics.dim + 1],
                activation=self.activation,
                stable=False,
                T=self.dynamics.T,
                input_bn=model_cfg["input_bn"],
                affine=model_cfg["affine"],
            )
        elif (
            model_cfg["architecture"] == "lstm"
            or model_cfg["architecture"] == "reslstm"
            or model_cfg["architecture"] == "naislstm"
        ):
            internal_net = LSTMNet(
                layers=[self.dynamics.dim + 1] + model_cfg["Y_layers"] + [self.dynamics.dim + 1],
                activation=self.activation,
                type=model_cfg["architecture"],
                T=self.dynamics.T,
                input_bn=model_cfg["input_bn"],
                affine=model_cfg["affine"],
            )
        elif model_cfg["architecture"] == "separatesubnets":
            # Get subnet configuration from args
            subnet_type = model_cfg["subnet_type"]
            internal_net = SeparateSubnets(
                layers=[self.dynamics.dim + 1] + model_cfg["Y_layers"] + [self.dynamics.dim + 1],
                activation=self.activation,
                num_time_steps=self.dynamics.N,
                T=self.dynamics.T,
                subnet_type=subnet_type,
                input_bn=model_cfg["input_bn"],
                affine=model_cfg["affine"],
            )
        elif model_cfg["architecture"] == "lstmwithsubnets":
            # Get LSTM and subnet configuration from args
            lstm_layers = model_cfg["lstm_layers"]
            lstm_type = model_cfg["lstm_type"]
            subnet_type = model_cfg["subnet_type"]
            internal_net = LSTMWithSubnets(
                layers=[self.dynamics.dim + 1] + model_cfg["Y_layers"] + [self.dynamics.dim + 1],
                activation=self.activation,
                num_time_steps=self.dynamics.N,
                T=self.dynamics.T,
                lstm_layers=lstm_layers,
                subnet_type=subnet_type,
                lstm_type=lstm_type,
                input_bn=model_cfg["input_bn"],
                affine=model_cfg["affine"],
            )
        else:
            raise ValueError(f"Unknown architecture: {model_cfg['architecture']}")

        # Initialize networks based on the type
        if self.network_type == "dY":
            self.dY_net = internal_net
            self.Y_init_net = FCnet_init(
                layers=[self.dynamics.dim] + model_cfg["Y0_layers"] + [1],
                activation=self.activation,
                y0=self.dynamics.y0,
                rescale_y0=model_cfg["rescale_y0"],
                strong_grad_output=model_cfg["strong_grad_output"],
                scale_output=model_cfg["scale_output"],
            )
        elif self.network_type == "Y":
            self.Y_net = internal_net

        self.to(torch.device(device))
        if self.careful_init:
            self.initialize_weights(self.activation)

    def initialize_weights(self, activation):
        """Applies careful initialization to all nn.Linear layers in the model."""

        if isinstance(activation, nn.Module):
            act_name = activation.__class__.__name__
        else:
            act_name = str(activation)

        if act_name in ['ReLU', 'LeakyReLU', 'ELU', 'GELU', 'SELU', 'SiLU', 'Swish', 'Hardswish']:
            init_type = "kaiming"
        elif act_name in ['Sigmoid', 'Tanh', 'Hardsigmoid', 'Hardtanh', 'Softmax', 'Softsign']:
            init_type = "xavier"
        elif act_name == 'Sine':
            init_type = "sine"
        else:
            raise ValueError(f"Unsupported activation function for initialization: {act_name}")

        def init_fn(m):
            if isinstance(m, nn.Linear):
                if init_type == "kaiming":
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5), nonlinearity='relu')
                elif init_type == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif init_type == "sine":
                    nn.init.uniform_(m.weight, -1 / m.in_features, 1 / m.in_features)

                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)

        self.apply(init_fn)


    @classmethod
    def load_from_checkpoint(cls, dynamics, model_cfg, device, model_dir, best=True):
        """
        Class method to create a DeepAgent instance and load checkpoint weights.
        
        Args:
            dynamics: Dynamics object for the model
            model_cfg: Configuration dictionary for the model
            device: Device to load the model on
            model_dir: Directory containing the model checkpoint
            best: Whether to load the best model or latest model
            
        Returns:
            model: A loaded DeepAgent instance
        """
        # Create a new instance
        model = cls(dynamics, model_cfg, device)
        
        # Load the checkpoint
        model.load_checkpoint(model_dir, best)
        
        return model
        
    def load_checkpoint(self, model_dir, best=True):
        """
        Load a checkpoint into the current model instance.
        
        Args:
            model_dir: Directory containing the model checkpoint
            best: Whether to load the best model or latest model
            
        Returns:
            optimizer_state: Optimizer state dict if available, else None
            scheduler_state: Scheduler state dict if available, else None
            start_epoch: The epoch to resume from if available, else 0
        """
        model_path = os.path.join(model_dir, "model")
        load_path = model_path + "_best.pth" if best else model_path + ".pth"
        
        # Load checkpoint
        checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)
        
        # Load model state
        self.load_state_dict(checkpoint['model_state_dict'])
        
        # Get training states
        optimizer_state = checkpoint.get('optimizer_state_dict')
        scheduler_state = checkpoint.get('scheduler_state_dict')
        start_epoch = checkpoint['epoch'] + 1  # Start from next epoch
        
        # Restore lowest loss
        if not self.reset_best:
            self.lowest_loss = checkpoint['lowest_loss']

        # Restore validation history if applicable
        if 'validation' in checkpoint and hasattr(self, 'validation'):
            self.validation = checkpoint['validation']
        
        # Store training history as temporary attribute
        if 'history' in checkpoint:
            self._training_history = checkpoint['history']
        
        return optimizer_state, scheduler_state, start_epoch

    def save_model(self, save_path, optimizer=None, scheduler=None, epoch=None, history=None):
        """
        Save model state, and optionally optimizer and scheduler state for training resumption.
        
        Args:
            save_path: Path to save the model
            optimizer: Optional optimizer to save state
            scheduler: Optional scheduler to save state
            epoch: Current epoch number
            history: Optional dictionary containing training history/losses
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'epoch': epoch if epoch is not None else self.epoch,
            'lowest_loss': self.lowest_loss
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
        if hasattr(self, 'validation') and self.validation is not None:
            checkpoint['validation'] = self.validation
            
        # Save history if provided
        if history is not None:
            checkpoint['history'] = history

        try:
            torch.save(checkpoint, save_path + ".pth")
        except Exception as e:
            return f"Error saving model: {e}"

        return "Model saved"

    def hjb_residual(self, t, y):

        if self.network_type == "dY":
            dY = self.dY_net(t, y)
            dY_dt = dY[:, 0:1]  # temporal derivative
            dY_dy = dY[:, 1:]   # spatial derivatives
        elif self.network_type == "Y":
            Y = self.Y_net(t, y)
            dY_dt, dY_dy = torch.autograd.grad(
                outputs=Y,
                inputs=(t, y),
                grad_outputs=torch.ones_like(Y),
                create_graph=True,
                retain_graph=True
            )
        
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
        q = self.dynamics.optimal_control(t, y, dY_dy)
        mu = self.dynamics.mu(t, y, q)
        sigma = self.dynamics.sigma(t, y)
        f = self.dynamics.generator(y, q)

        # Diffusion term: ½ Tr[σσᵀ H]
        sigma_sigma_T = torch.bmm(sigma, sigma.transpose(1, 2))  # (batch, dim, dim)
        trace_term = torch.einsum("bij,bij->b", sigma_sigma_T, H).unsqueeze(-1)

        # Residual from HJB
        residual = dY_dt + (mu * dY_dy).sum(dim=1, keepdim=True) + 0.5 * trace_term + f
        
        # Validate intermediate values
        self.validate_loss(dY_dt, "hjb_dY_dt")
        self.validate_loss((mu * dY_dy).sum(dim=1, keepdim=True), "hjb_drift_term")
        self.validate_loss(trace_term, "hjb_trace_term")
        self.validate_loss(f, "hjb_generator_term")
        self.validate_loss(residual, "hjb_residual")
        
        return self.apply_thresholded_loss(residual).mean()

    def sample_random_points(self, n_samples, point_type="initial"):
        """
        Sample random points for training.
        
        Args:
            n_samples: Number of random samples to generate
            point_type: "initial" for t=0 samples, "terminal" for t=T samples, "random" for random t
            
        Returns:
            t_samples: Time points
            y_samples: State samples
        """
        if point_type == "initial":
            # Sample around initial state y0
            t_samples = torch.zeros(n_samples, 1, device=self.device)
            y0_base = self.dynamics.y0.detach().flatten().cpu()
            
            # Create random perturbations around y0
            std_mult = 2.0  # Standard deviation multiplier
            perturbations = []
            for i in range(self.dynamics.dim):
                base = y0_base[i].item()
                try:
                    # Estimate std from dynamics
                    dummy_t = torch.zeros(1, 1, device=self.device)
                    dummy_y = self.dynamics.y0.detach()
                    sigma = self.dynamics.sigma(dummy_t, dummy_y)[0]
                    std_estimate = torch.norm(sigma[i]).item() * self.dynamics.T**0.5
                    std = std_mult * std_estimate
                except Exception:
                    std = abs(base) * 0.1 + 0.1  # Fallback: 10% of value + small constant
                
                perturbation = torch.normal(0, std, (n_samples,))
                perturbations.append(perturbation)
            
            perturbation_tensor = torch.stack(perturbations, dim=1).to(self.device)
            y_samples = self.dynamics.y0.repeat(n_samples, 1) + perturbation_tensor
            
        elif point_type == "terminal":
            # Sample around terminal time T
            t_samples = torch.full((n_samples, 1), self.dynamics.T, device=self.device)
            
            # Generate random terminal states by forward simulation
            _, _, y_paths = self.dynamics.generate_paths(n_samples)
            y_samples = y_paths[-1]  # Take terminal states
            
        elif point_type == "random":
            # Sample random time and state points
            t_samples = torch.rand(n_samples, 1, device=self.device) * self.dynamics.T
            
            # Sample states in a reasonable range
            y0_base = self.dynamics.y0.detach().flatten().cpu()
            std_mult = 3.0
            y_min_list, y_max_list = [], []
            
            for i in range(self.dynamics.dim):
                base = y0_base[i].item()
                try:
                    dummy_t = torch.zeros(1, 1, device=self.device)
                    dummy_y = self.dynamics.y0.detach()
                    sigma = self.dynamics.sigma(dummy_t, dummy_y)[0]
                    std_estimate = torch.norm(sigma[i]).item() * self.dynamics.T**0.5
                    delta = std_mult * std_estimate
                except Exception:
                    delta = abs(base) * 0.5 + 1.0  # Fallback
                
                y_min_list.append(base - delta)
                y_max_list.append(base + delta)
            
            y_min = torch.tensor(y_min_list, device=self.device)
            y_max = torch.tensor(y_max_list, device=self.device)
            y_samples = y_min + (y_max - y_min) * torch.rand(n_samples, self.dynamics.dim, device=self.device)
        
        return t_samples, y_samples

    def validate_loss(self, loss_value, loss_name, epoch=None):
        """
        Validate that a loss value is finite (not NaN or inf).
        
        Args:
            loss_value: The loss tensor to validate
            loss_name: Name of the loss for error reporting
            epoch: Current epoch number (optional)
        """
        if torch.isnan(loss_value).any():
            error_msg = f"NaN detected in {loss_name}"
            if epoch is not None:
                error_msg += f" at epoch {epoch}"
            print(f"ERROR: {error_msg}")
            print(f"Loss value: {loss_value}")
            raise ValueError(error_msg)
        
        if torch.isinf(loss_value).any():
            error_msg = f"Inf detected in {loss_name}"
            if epoch is not None:
                error_msg += f" at epoch {epoch}"
            print(f"ERROR: {error_msg}")
            print(f"Loss value: {loss_value}")
            raise ValueError(error_msg)
        
        return True

    def apply_thresholded_loss(self, diff):

        if not self.use_linear_approx:
            return diff.pow(2)
        
        large_diff_mask = (diff.abs() > self.loss_threshold).float()
        squared_loss = diff.pow(2)
        linear_loss = 2 * self.loss_threshold * diff.abs() - self.loss_threshold**2
        
        return large_diff_mask * linear_loss + (1 - large_diff_mask) * squared_loss
    
    def forward(self, t, dW, epoch=None):
        t0 = t[0]
        y0 = self.dynamics.y0.repeat(self.batch_size, 1).to(self.device)

        if self.second_order_taylor:
            t0 = t0.requires_grad_(True)
            y0 = y0.requires_grad_(True)

        if self.network_type == "dY":
            Y0 = self.Y_init_net(y0)
        elif self.network_type == "Y":
            Y0 = self.Y_net(t0, y0)
        Y0_init = Y0

        t_traj, y_traj, q_traj, Y_traj = [t0], [y0], [], [Y0]
        
        losses_dict = {}

        # === Compute FBSNN loss ===
        Y_loss = 0.0
        for n in range(self.dynamics.N):
            t1 = t[n + 1]
            dt = t1 - t0

            if self.second_order_taylor:
                t1 = t1.requires_grad_(True)
                y0 = y0.requires_grad_(True)

            # === Compute dY using the network ===
            if self.network_type == "dY":
                dY0 = self.dY_net(t0, y0)
                dY0_dt = dY0[:, 0:1]
                dY0_dy = dY0[:, 1:]
            elif self.network_type == "Y":
                dY0_dy = torch.autograd.grad(
                    outputs=Y0,
                    inputs=y0,
                    grad_outputs=torch.ones_like(Y0),
                    create_graph=True,
                    retain_graph=True
                )[0]
            
            q = self.dynamics.optimal_control(t0, y0, dY0_dy)
            if self.detach_control:
                q = q.detach()
            y1 = self.dynamics.forward_dynamics(y0, q, dW[n, :, :], t0, dt)
            dy = y1 - y0

            # === Compute Y1 using the network ===
            if self.network_type == "dY":
                # Predict next state using the dY network
                Y1_tilde = Y0 + dY0_dt * dt + (dY0_dy * dy).sum(dim=1, keepdim=True)

                if self.second_order_taylor:
                    # Second-order Taylor approximation
                    ddY_ddt = torch.autograd.grad(
                        outputs=dY0_dt,
                        inputs=t0,
                        grad_outputs=torch.ones_like(dY0_dt),
                        create_graph=False,
                        retain_graph=True,
                    )[0]
                    cross_term = torch.autograd.grad(
                        outputs=dY0_dt,
                        inputs=y0,
                        grad_outputs=torch.ones_like(dY0_dt),
                        create_graph=False,
                        retain_graph=True,
                    )[0]
                    hvp = torch.autograd.grad(
                        outputs=(dY0_dy * dy).sum(),
                        inputs=y0,
                        create_graph=False,
                        retain_graph=True
                    )[0]

                    # Handle cases where gradient with respect to t might be None due to separate subnet per time
                    if ddY_ddt is None:
                        ddY_ddt = 0

                    Y1_tilde += 0.5 * (ddY_ddt * dt ** 2 
                                 + 2 * (cross_term * dy).sum(dim=1, keepdim=True) * dt 
                                 + (hvp * dy).sum(dim=1, keepdim=True)
                                 )
            elif self.network_type == "Y":
                # Predict next state using the Y network
                Y1_tilde = self.Y_net(t1, y1)

            Z0 = torch.bmm(self.dynamics.sigma(t0, y0).transpose(1, 2), dY0_dy.unsqueeze(-1)).squeeze(-1)
            Y1 = Y0 - self.dynamics.generator(y0, q) * dt + (Z0 * (dW[n, :, :])).sum(dim=1, keepdim=True)

            step_loss = self.apply_thresholded_loss(Y1 - Y1_tilde).mean()
            self.validate_loss(step_loss, f"Y_step_loss_n{n}", epoch)
            Y_loss += step_loss

            t_traj.append(t1)
            y_traj.append(y1)
            q_traj.append(q)
            Y_traj.append(Y1)

            t0, y0, Y0 = t1, y1, Y1

        if self.loss_weights["lambda_Y"] > 0:
            self.validate_loss(Y_loss, "Y_loss (FBSNN)", epoch)
            losses_dict["lambda_Y"] = Y_loss
            self.Y_loss = Y_loss.detach()

        t_all = torch.cat(t_traj, dim=0)
        y_all = torch.cat(y_traj, dim=0)
        q_all = torch.cat(q_traj, dim=0)
        Y_all = torch.cat(Y_traj, dim=0)

        # === Terminal loss ===
        terminal_loss = 0.0
        if self.loss_weights["lambda_T"] > 0:
            terminal_loss = self.apply_thresholded_loss(Y0 - self.dynamics.terminal_cost(y0)).mean()
            self.validate_loss(terminal_loss, "terminal_loss", epoch)
            losses_dict["lambda_T"] = terminal_loss
            self.terminal_loss = terminal_loss.detach()

        # === Terminal gradient loss ===
        terminal_gradient_loss = 0.0
        if self.loss_weights["lambda_TG"] > 0:
            terminal_gradient_loss = self.apply_thresholded_loss(dY0_dy - self.dynamics.terminal_cost_grad(y0)).mean()
            self.validate_loss(terminal_gradient_loss, "terminal_gradient_loss", epoch)
            losses_dict["lambda_TG"] = terminal_gradient_loss
            self.terminal_gradient_loss = terminal_gradient_loss.detach()

        # === Physics-based loss ===
        pinn_loss = 0.0
        if self.loss_weights["lambda_pinn"] > 0:
            t_pinn = torch.cat(t_traj, dim=0)
            y_pinn = torch.cat(y_traj, dim=0)

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
            self.validate_loss(pinn_loss, "pinn_loss", epoch)
            losses_dict["lambda_pinn"] = pinn_loss
            self.pinn_loss = pinn_loss.detach()

        # === q regularization loss ===
        reg_loss = 0.0
        if self.loss_weights["lambda_reg"] > 0:
            q_diffs = [q_traj[i+1] - q_traj[i] for i in range(len(q_traj) - 1)]
            dq_all = torch.cat(q_diffs, dim=0)
            reg_loss = self.apply_thresholded_loss(q_all).mean() #+ self.apply_thresholded_loss(dq_all).mean()
            self.validate_loss(reg_loss, "reg_loss", epoch)
            losses_dict["lambda_reg"] = reg_loss
            self.reg_loss = reg_loss.detach()

        # === Cost objective ===
        cost_objective = 0.0
        if self.loss_weights["lambda_cost"] > 0:
            cost_objective = compute_cost_objective(
                dynamics=self.dynamics,
                q_traj=torch.stack(q_traj, dim=0),
                y_traj=torch.stack(y_traj, dim=0),
                terminal_cost=True
            ).mean()
            self.validate_loss(cost_objective, "cost_objective", epoch)
            losses_dict["lambda_cost"] = cost_objective
            self.cost_loss = cost_objective.detach()

        # === Y0 fbsnn loss ===
        Y0_loss = 0.0
        if self.loss_weights["lambda_Y0"] > 0:
            Y0_loss = self.apply_thresholded_loss(Y0_init - self.cost_loss).mean()
            self.validate_loss(Y0_loss, "Y0_loss", epoch)
            losses_dict["lambda_Y0"] = Y0_loss
            self.Y0_loss = Y0_loss.detach()

        # === Y and q validation loss ===
        if self.dynamics.analytical_known:
            Y_true = self.dynamics.value_function_analytic(t_all, y_all)
            Y_val_loss = self.apply_thresholded_loss(Y_all - Y_true).mean()
            self.validate_loss(Y_val_loss, "Y_val_loss", epoch)
            self.val_Y_loss = Y_val_loss.detach()

            q_true = self.dynamics.optimal_control_analytic(t_all[:-self.batch_size], y_all[:-self.batch_size])
            q_val_loss = self.apply_thresholded_loss(q_all - q_true).mean()
            self.validate_loss(q_val_loss, "q_val_loss", epoch)
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

        self.validate_loss(total_loss, "total_loss", epoch)

        # === Supervised loss ===
        if self.dynamics.analytical_known and self.supervised:
            total_loss += Y_val_loss
            total_loss += q_val_loss
            self.validate_loss(total_loss, "total_loss_with_supervised", epoch)

        return total_loss

    def predict_Y_initial(self, y0):
        if self.network_type == "dY":
            # Predict initial state using the Y_init network
            self.Y_init_net.eval()
            with torch.no_grad():
                Y0 = self.Y_init_net(y0)
            self.Y_init_net.train()
        elif self.network_type == "Y":
            # Predict initial state using the Y network
            self.Y_net.eval()
            batch_size = y0.shape[0]
            t0 = torch.zeros(batch_size, 1, device=self.device)
            Y0 = self.Y_net(t0, y0)
            self.Y_net.train()
        return Y0
    
    def predict_Y_next(self, t0, y0, dt, dy, Y0):
        if self.network_type == "dY":
            # Predict next state using the dY network
            self.dY_net.eval()
            if self.second_order_taylor:
                t0 = t0.requires_grad_(True)
                y0 = y0.requires_grad_(True)
                dY = self.dY_net(t0, y0)
            else:
                with torch.no_grad():
                    dY = self.dY_net(t0, y0)
            self.dY_net.train()

            dY_dt = dY[:, 0:1]
            dY_dy = dY[:, 1:]
            
            Y1 = Y0 + dY_dt * dt + (dY_dy * dy).sum(dim=1, keepdim=True)

            if self.second_order_taylor:
                # Second-order Taylor approximation
                ddY_ddt = torch.autograd.grad(
                    outputs=dY_dt,
                    inputs=t0,
                    grad_outputs=torch.ones_like(dY_dt),
                    create_graph=False,
                    retain_graph=True,
                )[0]
                cross_term = torch.autograd.grad(
                    outputs=dY_dt,
                    inputs=y0,
                    grad_outputs=torch.ones_like(dY_dt),
                    create_graph=False,
                    retain_graph=True,
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

                Y1 += 0.5 * (ddY_ddt * dt ** 2
                             + 2 * (cross_term * dy).sum(dim=1, keepdim=True) * dt
                             + (hvp * dy).sum(dim=1, keepdim=True)
                             )

        elif self.network_type == "Y":
            # Predict next state using the Y network
            self.Y_net.eval()
            Y1 = self.Y_net(t0 + dt, y0 + dy)
            self.Y_net.train()

        return Y1

    def predict(self, t, y):
        if self.network_type == "dY":
            # Predict optimal control using the dY network
            self.dY_net.eval()
            with torch.no_grad():
                dY_outputs = self.dY_net(t, y)
            self.dY_net.train()
            
            dY_dy = dY_outputs[:, 1:]
        elif self.network_type == "Y":
            # Predict optimal control using the Y network
            self.Y_net.eval()
            t = t.requires_grad_(True)
            y = y.requires_grad_(True)
            Y = self.Y_net(t, y)
            self.Y_net.train()
            
            dY_dy = torch.autograd.grad(
                outputs=Y,
                inputs=y,
                grad_outputs=torch.ones_like(Y),
                create_graph=True,
                retain_graph=True
            )[0]
        return self.dynamics.optimal_control(t, y, dY_dy)

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
        start_epoch=1,
        optimizer_state=None,
        scheduler_state=None,
    ):
        """
        Train the model, with option to resume from a checkpoint.
        
        Args:
            epochs: Total number of epochs to train
            K: Print interval
            lr: Learning rate (used only if optimizer_state is None)
            verbose: Whether to print progress
            plot: Whether to show plots
            adaptive: Whether to use adaptive learning rate
            save_dir: Directory to save models and plots
            logger: Logger object
            start_epoch: Epoch to start from (for resuming training)
            optimizer_state: Optimizer state dict to resume from
            scheduler_state: Scheduler state dict to resume from
        """
        # Prepare save directory and logging
        save_path = os.path.join(save_dir, "model")
        logger = logger if logger is not None else Logger(save_dir, is_main=self.is_main, verbose=verbose, filename="training.log")

        # === Initialize training history arrays - will be restored from checkpoint if provided ===
        if hasattr(self, '_training_history') and self._training_history is not None:
            # Restore from previously loaded history
            losses = self._training_history.get('losses', [])
            losses_Y0 = self._training_history.get('losses_Y0', [])
            losses_Y = self._training_history.get('losses_Y', [])
            losses_dY = self._training_history.get('losses_dY', [])
            losses_dYt = self._training_history.get('losses_dYt', [])
            losses_terminal = self._training_history.get('losses_terminal', [])
            losses_terminal_gradient = self._training_history.get('losses_terminal_gradient', [])
            losses_pinn = self._training_history.get('losses_pinn', [])
            losses_reg = self._training_history.get('losses_reg', [])
            losses_cost = self._training_history.get('losses_cost', [])
            lr_decay_epochs = self._training_history.get('lr_decay_epochs', [])
            
            # Clear the reference to avoid keeping it twice in memory
            del self._training_history
        else:
            # Start with empty lists
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
            lr_decay_epochs = []

        # === Define active loss components ===
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

        # === Width of the training log ===
        num_cols = 1  # for epoch
        if self.dynamics.analytical_known:
            num_cols += 2  # Y val, q val
        num_cols += 1  # total loss
        num_cols += len(active_losses)
        num_cols += 5  # lr, mem, time, eta, status
        
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
        for name, _ in active_losses:
            max_widths[name] = max(10, len(name) + 2)
        width = (
            max_widths["epoch"]
            + (2 * max_widths["val loss"] if self.dynamics.analytical_known else 0)
            + max_widths["loss"]
            + sum(max_widths[name] for name, _ in active_losses)
            + max_widths["lr"]
            + max_widths["mem"]
            + max_widths["time"]
            + max_widths["eta"]
            + max_widths["status"]
            + 3 * (num_cols - 1)
        )

        if self.is_main:
            # === Log training configuration ===
            logger.log(
                f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
            )
            logger.log(f"Logging training to {logger.log_path}")
            init_time = time.time()
            start_time = time.time()

            # === Print header for training progress ===
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
        
        # === Initialize optimizer ===
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        # === Load optimizer state if provided ===
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            # Make sure we're using the correct device for optimizer state
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
        
        # === Initialize scheduler ===
        scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=self.adaptive_factor, patience=200
            )
            if adaptive
            else torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
        )
        
        # === Load scheduler state if provided ===
        if scheduler_state is not None and not self.reset_lr:
            scheduler.load_state_dict(scheduler_state)
        elif self.reset_lr:
            # Reset learning rate to initial value when resetting scheduler
            logger.log(f"Resetting learning rate scheduler and optimizer LR to {lr:.2e}")
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        current_lr = optimizer.param_groups[0]["lr"]
        lr_decay_epochs = []
        
        # === Log if resuming training ===
        if start_epoch > 1 and self.is_main:
            logger.log(f"\nResuming training from epoch {start_epoch}/{epochs}")
            logger.log(f"Current learning rate: {current_lr:.6e}")
            logger.log(f"Lowest loss so far: {self.lowest_loss:.6e}")
            logger.log("-" * width)

        # === Training loop ===
        for epoch in range(start_epoch, epochs + 1):
            self.epoch = epoch
            if self.annealing:
                self.dynamics.anneal(epoch, epochs)
            optimizer.zero_grad()
            t, dW, _ = self.dynamics.generate_paths(self.batch_size)
            loss = self(t=t, dW=dW, epoch=epoch)
            loss.backward()

            # === Optimizer and scheduler step ===
            optimizer.step()
            scheduler.step(loss.item() if adaptive else epoch)

            # === Check if the LR was reduced ===
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr < current_lr:
                lr_decay_epochs.append(epoch)
                current_lr = new_lr

            # === Reduce losses across all processes if distributed ===
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

            # === Log losses and print plots ===
            if self.is_main:
                if self.dynamics.analytical_known:
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

                    # === Save model if conditions are met ===
                    if "every" in self.save and (
                        epoch % self.save_n == 0 or epoch == epochs - 1
                    ):
                        # Create history dictionary with all losses
                        history = {
                            'losses': losses,
                            'losses_Y0': losses_Y0,
                            'losses_Y': losses_Y,
                            'losses_dY': losses_dY,
                            'losses_dYt': losses_dYt,
                            'losses_terminal': losses_terminal,
                            'losses_terminal_gradient': losses_terminal_gradient,
                            'losses_pinn': losses_pinn,
                            'losses_reg': losses_reg,
                            'losses_cost': losses_cost,
                            'lr_decay_epochs': lr_decay_epochs
                        }
                        status = self.save_model(save_path, optimizer, scheduler, epoch, history)

                    # === Save best model if conditions are met ===
                    if "best" in self.save and np.mean(losses_cost[-K:]) < self.lowest_loss:
                        self.lowest_loss = np.mean(losses_cost[-K:])
                        # Create history dictionary with all losses
                        history = {
                            'losses': losses,
                            'losses_Y0': losses_Y0,
                            'losses_Y': losses_Y,
                            'losses_dY': losses_dY,
                            'losses_dYt': losses_dYt,
                            'losses_terminal': losses_terminal,
                            'losses_terminal_gradient': losses_terminal_gradient,
                            'losses_pinn': losses_pinn,
                            'losses_reg': losses_reg,
                            'losses_cost': losses_cost,
                            'lr_decay_epochs': lr_decay_epochs
                        }
                        status = self.save_model(save_path + "_best", optimizer, scheduler, epoch, history) + " (best)"

                    # === Calculate average time per K epochs and ETA ===
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

                    # === Log training progress ===
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

                # === Plot approximation vs analytical ===
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

                # === Plot training losses ===
                if epoch % 1000 == 0 and epoch > 0:
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
                    if plot:
                        plt.show()
                    else:
                        plt.close()

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
                        if plot:
                            plt.show()
                        else:
                            plt.close()

        if self.is_main:
            if self.is_distributed:
                dist.barrier()

            logger.log(f"Training completed. Lowest loss: {self.lowest_loss:.6f}. Total time: {time.time() - init_time:.2f} seconds")
            logger.log(f"Model saved to {save_path}.pth")

        return losses