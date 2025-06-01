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
        self.analytical_known = args.analytical_known
        self.Y_layers = args.Y_layers
        self.n_paths = args.n_paths
        self.batch_size = args.batch_size
        self.λ_Y = args.lambda_Y
        self.λ_dY = args.lambda_dY
        self.λ_dYt = args.lambda_dYt
        self.λ_T = args.lambda_T
        self.λ_TG = args.lambda_TG
        self.λ_pinn = args.lambda_pinn
        self.t0 = 0.0
        self.y0 = torch.tensor([model_cfg["y0"]], device=self.device, requires_grad=True)
        self.dim = model_cfg["dim"]
        self.dim_W = model_cfg["dim_W"]
        self.T = model_cfg["T"]
        self.N = model_cfg["N"]
        self.dt = model_cfg["dt"]
        self.save = args.save
        self.save_n = args.save_n
        self.Y_loss = 0
        self.dY_loss = 0
        self.dYt_loss = 0
        self.terminal_loss = 0
        self.terminal_gradient_loss = 0
        self.pinn_loss = 0

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
    def trading_rate(self, t, y, Y): pass 

    @abstractmethod
    def forward_supervised(self, t_paths, W_paths): pass

    @abstractmethod
    def value_function_analytic(self, t, y): pass

    def forward_dynamics(self, y, q, dW, t, dt):
        μ = self.mu(t, y, q)                  # shape: (batch, dim)
        σ = self.sigma(t, y)                  # shape: (batch, dim, dW_dim)
        diffusion = torch.bmm(σ, dW.unsqueeze(-1)).squeeze(-1)  # shape: (batch, dim)
        return y + μ * dt + diffusion         # shape: (batch, dim)

    def physics_loss(self, t, y, Y):
        dim = y.shape[1]

        # Ensure t and y have gradients enabled
        if not t.requires_grad:
            t = t.detach().requires_grad_(True)
        if not y.requires_grad:
            y = y.detach().requires_grad_(True)

        dV = torch.autograd.grad(
            outputs=Y, 
            inputs=y, 
            grad_outputs=torch.ones_like(Y),
            create_graph=True,
            retain_graph=True
        )[0]

        dV_t = torch.autograd.grad(
            outputs=Y, 
            inputs=t, 
            grad_outputs=torch.ones_like(Y),
            create_graph=True,
            retain_graph=True
        )[0]

        # Compute Hessian
        H = []
        for i in range(dim):
            d2V_dyi = torch.autograd.grad(
                outputs=dV[:, i], 
                inputs=y, 
                grad_outputs=torch.ones_like(dV[:, i]),
                create_graph=True,
                retain_graph=True
            )[0]
            H.append(d2V_dyi)
        H = torch.stack(H, dim=1)  # (batch, dim, dim)

        # Dynamics
        q = self.trading_rate(t, y, Y)
        mu = self.mu(t, y, q)
        sigma = self.sigma(t, y)
        f = self.generator(y, q)

        # Trace term: Tr[σ σᵀ H]
        sigma_T = sigma.transpose(1, 2)
        sigma_sigma_T = torch.bmm(sigma, sigma_T)
        trace_term = 0.5 * torch.einsum("bij,bij->b", sigma_sigma_T, H).unsqueeze(-1)

        # HJB residual
        residual = dV_t + (mu * dV).sum(dim=1, keepdim=True) + trace_term - f
        return torch.mean(torch.pow(residual, 2))

    def fetch_minibatch(self, batch_size):
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
        y0 = self.y0.repeat(batch_size, 1).to(self.device)
        y_traj, t_traj = [y0], []
        t0 = t_paths[:, 0, :]
        W0 = W_paths[:, 0, :]
        Y0 = self.Y_net(t0, y0)

        dY0 = torch.autograd.grad(
            outputs=Y0,
            inputs=y0,
            grad_outputs=torch.ones_like(Y0),
            create_graph=True,
            retain_graph=True
        )[0]

        Y_loss = 0.0

        # === 1. Compute fbsnn loss ===
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

            t_traj.append(t1)
            y_traj.append(y1)

            t0, W0, y0, Y0, dY0 = t1, W1, y1, Y1, dY1

        # === 2. Terminal supervision ===
        terminal_loss, terminal_gradient_loss = 0.0, 0.0
        if self.λ_T > 0:
            YT = self.Y_net(t1, y1)
            terminal_loss = torch.sum(torch.pow(YT - self.terminal_cost(y1), 2))
            self.terminal_loss = self.λ_T * terminal_loss.detach().item()
        if self.λ_TG > 0:
            dYT = torch.autograd.grad(
                YT, 
                y1, 
                grad_outputs=torch.ones_like(YT), 
                create_graph=True
            )[0]
            terminal_gradient_loss = torch.sum(torch.pow(dYT - self.terminal_cost_grad(y1), 2))
            self.terminal_gradient_loss = self.λ_TG * terminal_gradient_loss.detach().item()

        # === 3. Optional physics-based loss ===
        t_traj = torch.cat(t_traj, dim=0).requires_grad_(True)            # shape: [N * batch_size, 1]
        y_traj = torch.cat(y_traj[1:], dim=0).requires_grad_(True)        # shape: [N * batch_size, state_dim]

        pinn_loss = 0.0
        if self.λ_pinn > 0:
            with torch.enable_grad():
                for i in range(self.N):
                    idx = slice(i * batch_size, (i + 1) * batch_size)
                    t_i = t_traj[idx].detach().clone().requires_grad_(True)
                    y_i = y_traj[idx].detach().clone().requires_grad_(True)
                    V_i = self.Y_net(t_i, y_i)
                    pinn_loss += self.physics_loss(t_i, y_i, V_i)
                pinn_loss /= self.N
                self.pinn_loss = self.λ_pinn * pinn_loss.detach().item()

        self.Y_loss = self.λ_Y * Y_loss.detach().item()
        self.terminal_loss = self.λ_T * terminal_loss.detach().item()
        self.terminal_gradient_loss = self.λ_TG * terminal_gradient_loss.detach().item()

        return self.λ_Y * Y_loss + self.λ_T * terminal_loss + self.λ_TG * terminal_gradient_loss

    def forward_supervised(self, t_paths, W_paths):
        batch_size = self.batch_size
        device = self.device
        y0 = self.y0.repeat(batch_size, 1).to(device)

        # === 1. Precompute optimal trajectory ===
        y_traj, t_traj = [y0], []
        t0 = t_paths[:, 0, :]
        W0 = W_paths[:, 0, :]

        for n in range(self.N):
            t1 = t_paths[:, n + 1, :]
            W1 = W_paths[:, n + 1, :]
            dW = W1 - W0

            V = self.value_function_analytic(t0, y0)
            q = self.trading_rate(t0, y0, V, create_graph=False)
            y1 = self.forward_dynamics(y0, q, dW, t0, t1 - t0)

            t_traj.append(t1)
            y_traj.append(y1)

            t0, W0, y0 = t1, W1, y1

        t_traj = torch.cat(t_traj, dim=0).requires_grad_(True)            # shape: [N * batch_size, 1]
        y_traj = torch.cat(y_traj[1:], dim=0).requires_grad_(True)        # shape: [N * batch_size, state_dim]

        # === 2. Compute target value + gradients ===
        V_target = self.value_function_analytic(t_traj, y_traj)
        V_pred = self.Y_net(t_traj, y_traj)

        Y_loss = torch.mean(torch.pow(V_pred - V_target, 2))

        dY_loss, dYt_loss = 0.0, 0.0

        if self.λ_dY > 0:
            dV_target = torch.autograd.grad(
                V_target, 
                y_traj, 
                grad_outputs=torch.ones_like(V_target),
                create_graph=False, 
                retain_graph=True
            )[0]
            dV_pred = torch.autograd.grad(
                V_pred, 
                y_traj, 
                grad_outputs=torch.ones_like(V_pred),
                create_graph=True
            )[0]
            dY_loss = torch.mean(torch.pow(dV_pred - dV_target, 2))

        if self.λ_dYt > 0:
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
            dYt_loss = torch.mean(torch.pow(dV_pred_t - dV_target_t, 2))

        # === 3. Terminal supervision ===
        terminal_loss, terminal_gradient_loss = 0.0, 0.0
        if self.λ_T > 0:
            YT = self.Y_net(t1, y1)
            terminal_loss = torch.mean(torch.pow(YT - self.terminal_cost(y1), 2))
            self.terminal_loss = self.λ_T * terminal_loss.detach().item()
        if self.λ_TG > 0:
            dYT = torch.autograd.grad(
                YT, 
                y1, 
                grad_outputs=torch.ones_like(YT), 
                create_graph=True
            )[0]
            terminal_gradient_loss = torch.mean(torch.pow(dYT - self.terminal_cost_grad(y1), 2))
            self.terminal_gradient_loss = self.λ_TG * terminal_gradient_loss.detach().item()

        # === 4. Optional physics-based loss ===
        pinn_loss = 0.0
        if self.λ_pinn > 0:
            with torch.enable_grad():
                for i in range(self.N):
                    idx = slice(i * batch_size, (i + 1) * batch_size)
                    t_i = t_traj[idx].detach().clone().requires_grad_(True)
                    y_i = y_traj[idx].detach().clone().requires_grad_(True)
                    V_i = self.Y_net(t_i, y_i)
                    pinn_loss += self.physics_loss(t_i, y_i, V_i)
                pinn_loss /= self.N
                self.pinn_loss = self.λ_pinn * pinn_loss.detach().item()

        # === 5. Log losses ===
        if self.λ_Y > 0: self.Y_loss = self.λ_Y * Y_loss.detach().item()
        if self.λ_dY > 0: self.dY_loss = self.λ_dY * dY_loss.detach().item()
        if self.λ_dYt > 0: self.dYt_loss = self.λ_dYt * dYt_loss.detach().item()

        return (
            self.λ_Y * Y_loss +
            self.λ_dY * dY_loss +
            self.λ_dYt * dYt_loss +
            self.λ_T * terminal_loss +
            self.λ_TG * terminal_gradient_loss +
            self.λ_pinn * pinn_loss
        )

    def simulate_paths(self, n_sim=5, seed=42, y0_single=None):
        torch.manual_seed(seed)
        self.eval()

        y0 = y0_single.repeat(n_sim, 1) if y0_single is not None else self.y0.repeat(n_sim, 1)
        t_scalar = 0.0

        # Initialize both
        y_learned = y0.clone()
        y_analytic = y0.clone()

        y_learned_traj = []
        q_learned_traj = []
        Y_learned_traj = []
        if self.analytical_known:
            y_true_traj = []
            q_true_traj = []
            Y_true_traj = []

        for step in range(self.N + 1):
            t_tensor = torch.full((n_sim, 1), t_scalar, device=self.device)

            # Predict Y and compute control
            Y_learned = self.Y_net(t_tensor, y_learned)
            q_learned = self.trading_rate(t_tensor, y_learned, Y_learned, create_graph=False)
            if self.analytical_known:
                Y_true = self.value_function_analytic(t_tensor, y_analytic)
                q_true = self.trading_rate(t_tensor, y_analytic, Y_true, create_graph=False)

            # Save states and controls
            q_learned_traj.append(q_learned.detach().cpu().numpy())
            y_learned_traj.append(y_learned.detach().cpu().numpy())
            Y_learned_traj.append(Y_learned.detach().cpu().numpy())

            if self.analytical_known:
                q_true_traj.append(q_true.detach().cpu().numpy())
                y_true_traj.append(y_analytic.detach().cpu().numpy())
                Y_true_traj.append(Y_true.detach().cpu().numpy())

            if step < self.N:
                dW = torch.randn(n_sim, self.dim_W, device=self.device) * self.dt**0.5
                y_learned = self.forward_dynamics(y_learned, q_learned, dW, t_tensor, self.dt)
                if self.analytical_known:
                    y_analytic = self.forward_dynamics(y_analytic, q_true, dW, t_tensor, self.dt)
                t_scalar += self.dt

        return torch.linspace(0, self.T, self.N + 1).cpu().numpy(), {
            "y": np.stack(y_learned_traj),
            "q": np.stack(q_learned_traj),
            "Y": np.stack(Y_learned_traj),
            "y_true": np.stack(y_true_traj) if self.analytical_known else None,
            "q_true": np.stack(q_true_traj) if self.analytical_known else None,
            "Y_true": np.stack(Y_true_traj) if self.analytical_known else None
        }

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

        losses, losses_Y, losses_dY, losses_dYt, losses_terminal, losses_terminal_gradient, losses_pinn = [], [], [], [], [], [], []
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
        log(f"| lambda_dY (dY loss)       | {self.λ_dY:<25} |")
        log(f"| lambda_dYt (dYt loss)     | {self.λ_dYt:<25} |")
        log(f"| lambda_T (Terminal loss)  | {self.λ_T:<25} |")
        log(f"| lambda_TG (Gradient loss) | {self.λ_TG:<25} |")
        log(f"| lambda_pinn (PINN loss)   | {self.λ_pinn:<25} |")
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
        
        # Dynamic width calculation based on loss components
        max_widths = {
            "epoch": 8,
            "loss": 8,
            "lr": 10,
            "mem": 12,
            "time": 8,
            "eta": 8,
            "status": 20
        }
        loss_components = [
            ("Y loss", self.λ_Y, lambda: np.mean(losses_Y[-K:])),
            ("dY loss", self.λ_dY, lambda: np.mean(losses_dY[-K:])),
            ("dYt loss", self.λ_dYt, lambda: np.mean(losses_dYt[-K:])),
            ("T. loss", self.λ_T, lambda: np.mean(losses_terminal[-K:])),
            ("T.G. loss", self.λ_TG, lambda: np.mean(losses_terminal_gradient[-K:])),
            ("pinn loss", self.λ_pinn, lambda: np.mean(losses_pinn[-K:])),
        ]
        active_losses = [(name, fn) for name, λ, fn in loss_components if λ > 0]

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

        for epoch in range(epochs):

            # Loop over n_paths with a batch size of self.batch_size
            for _ in range(self.n_paths // self.batch_size):
                optimizer.zero_grad()
                t_paths, W_paths = self.fetch_minibatch(self.batch_size)
                loss = self(t_paths, W_paths)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                losses_Y.append(self.Y_loss)
                losses_dY.append(self.dY_loss)
                losses_dYt.append(self.dYt_loss)
                losses_terminal.append(self.terminal_loss)
                losses_terminal_gradient.append(self.terminal_gradient_loss)
                losses_pinn.append(self.pinn_loss)

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
                    header_parts = [f"{'Epoch':>{max_widths['epoch']}}", f"{'Total loss':>10}"]
                    for name, _ in active_losses:
                        header_parts.append(f"{name:>{max_widths[name]}}")
                    header_parts += [
                        f"{'LR':>{max_widths['lr']}}",
                        f"{'Memory [MB]':>{max_widths['mem']}}",
                        f"{'Time [s]':>{max_widths['time']}}",
                        f"{'ETA':>{max_widths['eta']}}",
                        f"{'Status':<{max_widths['status']}}",
                    ]
                    log(" | ".join(header_parts))
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
                log(" | ".join(row_parts))
                start_time = time.time()

        if "last" in self.save:
            try:
                torch.save(self.state_dict(), save_path + ".pth")
            except Exception as e:
                log(f"Error saving last model: {e}")
            status = "Model saved"
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
            log(" | ".join(row_parts))
        log("-" * width)
        log(f"Training completed. Lowest loss: {self.lowest_loss:.6f}. Total time: {time.time() - init_time:.2f} seconds")
        log(f"Model saved to {save_path}.pth")

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
            plt.savefig(f"{save_dir}/loss.png", dpi=300, bbox_inches='tight')
        if plot:
            plt.show()

        return losses