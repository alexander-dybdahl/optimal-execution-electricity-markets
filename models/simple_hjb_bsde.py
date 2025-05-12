import torch
import numpy as np
from core.fbsnn import FBSNN
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class SimpleHJB(FBSNN):
    def __init__(self, args, model_cfg):
        # Override the default architecture to be simpler but with proper initialization
        if args.architecture == "Default":
            model_cfg["Y_layers"] = [64, 64, 1]  # Slightly wider network
            model_cfg["q_layers"] = [64, 64, 1]
            model_cfg["activation"] = "Tanh"  # Use Tanh for better gradient stability
        
        super().__init__(args, model_cfg)
        self.sigma_x = model_cfg["sigma"]
        self.G = model_cfg["G"]
        
        # Initialize networks with smaller weights
        for net in [self.Y_net, self.q_net]:
            for param in net.parameters():
                if len(param.shape) > 1:  # weights only, not biases
                    torch.nn.init.xavier_uniform_(param, gain=0.5)

    def generator(self, y, q):
        x = y[:, 0:1]
        return 0.5 * (q**2 + x**2)  # Quadratic cost

    def terminal_cost(self, y):
        x_T = y[:, 0]
        return self.G * x_T**2

    def mu(self, t, y, q):
        return q

    def sigma(self, t, y):
        batch_size = y.shape[0]
        σ = torch.zeros(batch_size, 1, 1, device=self.device)
        σ[:, 0, 0] = self.sigma_x
        return σ

    def forward(self):
        """Override forward to add gradient clipping and stability measures"""
        batch_size = self.batch_size
        
        # Generate initial states with better coverage
        x0_fixed = torch.linspace(-2.0, 2.0, batch_size, device=self.device)
        y0_batch = x0_fixed.unsqueeze(-1).clone().detach().requires_grad_(True)
        
        t = torch.zeros(batch_size, 1, device=self.device)
        Y0 = self.Y_net(t, y0_batch)

        # Initial value regularization
        K_0 = self.K_analytic(0.0)
        phi_0 = self.phi_analytic(0.0)
        Y0_target = phi_0 + K_0 * y0_batch.squeeze(-1)**2
        Y0_loss = torch.mean((Y0.squeeze(-1) - Y0_target)**2)

        # Compute initial gradients
        dY0 = torch.autograd.grad(
            outputs=Y0,
            inputs=y0_batch,
            grad_outputs=torch.ones_like(Y0),
            create_graph=True,
            retain_graph=True
        )[0]

        total_residual_loss = 0.0
        prev_Y = Y0
        prev_dY = dY0
        prev_y = y0_batch

        for _ in range(self.N):
            q0 = self.q_net(t, prev_y)
            
            # Basic control regularization
            K_t = self.K_analytic(t.mean().item())
            q_target = -K_t * prev_y.squeeze(-1)
            q_reg_loss = torch.mean((q0.squeeze(-1) - q_target)**2)
            
            dW = torch.randn(batch_size, self.dim_W, device=self.device) * self.dt**0.5
            y1 = self.forward_dynamics(prev_y, q0, dW, t, self.dt)
            y1 = y1.clone().detach().requires_grad_(True)
            
            σ0 = self.sigma(t, prev_y)
            z0 = torch.bmm(σ0, prev_dY.unsqueeze(-1)).squeeze(-1)

            t = t + self.dt

            Y1 = self.Y_net(t, y1)
            dY1 = torch.autograd.grad(
                outputs=Y1,
                inputs=y1,
                grad_outputs=torch.ones_like(Y1),
                create_graph=True,
                retain_graph=True
            )[0]
            
            f = self.generator(prev_y, q0)
            Y1_tilde = prev_Y - f * self.dt + (z0 * dW).sum(dim=1, keepdim=True)

            # Basic residual loss
            residual = Y1 - Y1_tilde
            residual_loss = torch.mean(residual**2)
            total_residual_loss = total_residual_loss + residual_loss + 0.1 * q_reg_loss

            # Update for next iteration
            prev_Y = Y1
            prev_dY = dY1
            prev_y = y1

        # Terminal loss
        terminal = self.terminal_cost(y1)
        terminal_loss = torch.mean((Y1 - terminal)**2)
        
        # Simple loss combination
        total_loss = terminal_loss + total_residual_loss + Y0_loss
        return total_loss

    def simulate_paths(self, n_paths=1000, batch_size=256, seed=42, y0_single=None):
        torch.manual_seed(seed)
        self.eval()

        # Generate different initial states for simulation
        if y0_single is None:
            x0_values = torch.linspace(-2.0, 2.0, n_paths, device=self.device).unsqueeze(-1)
            y0_single = x0_values

        # Ensure we process at least one batch
        n_batches = max(1, n_paths // batch_size)
        actual_batch_size = min(batch_size, n_paths)

        all_q, all_Y, all_y = [], [], []

        for batch_idx in range(n_batches):
            start_idx = batch_idx * actual_batch_size
            end_idx = min((batch_idx + 1) * actual_batch_size, n_paths)
            current_batch_size = end_idx - start_idx
            
            y = y0_single[start_idx:end_idx]
            t = torch.zeros(current_batch_size, 1, device=self.device)
            
            q_traj, Y_traj, y_traj = [], [], []

            for i in range(self.N):
                t_input = t.clone()
                q = self.q_net(t_input, y)
                dW = torch.randn(current_batch_size, self.dim_W, device=self.device) * self.dt**0.5
                y = self.forward_dynamics(y, q, dW, t, self.dt)
                Y = self.Y_net(t, y)

                t += self.dt

                q_traj.append(q.detach().cpu().numpy())
                Y_traj.append(Y.detach().cpu().numpy())
                y_traj.append(y.detach().cpu().numpy())

            all_q.append(np.stack(q_traj))
            all_Y.append(np.stack(Y_traj))
            all_y.append(np.stack(y_traj))

        timesteps = np.linspace(0, self.T, self.N)

        # Concatenate all batches
        results = {
            "q": np.concatenate(all_q, axis=1)[:, :n_paths],
            "Y": np.concatenate(all_Y, axis=1)[:, :n_paths],
            "final_y": np.concatenate(all_y, axis=1)[:, :n_paths]
        }

        return timesteps, results

    def K_analytic(self, t):
        """Analytical solution to Riccati equation"""
        G = self.G
        A = ((G + 1) / (G - 1)) * np.exp(2 * (self.T - t))
        return (A + 1) / (A - 1)

    def optimal_control_analytic(self, t, x):
        """Optimal q(t, x) = -K(t) * x"""
        K_t = self.K_analytic(t).reshape(-1, 1)           # shape: (T, 1)
        return -K_t * x  # shape: (T, N_paths)

    def phi_analytic(self, t):
        """Analytical phi(t) from backward integral of K"""
        sigma = self.sigma_x
        G = self.G
        def A(s): return (G + 1) / (G - 1) * np.exp(2 * (self.T - s))
        A_t = A(t)
        A_T = A(self.T)
        log_expr = (A_t / A_T) * ((A_T - 1)**2 / (A_t - 1)**2)
        phi = -0.5 * sigma**2 * np.log(log_expr)
        return phi

    def optimal_cost_analytic(self, t, x):
        """Analytical cost-to-go Y(t) = phi(t) + K(t) * x^2"""
        K_t = self.K_analytic(t).reshape(-1, 1)  # (T, 1)
        phi_t = self.phi_analytic(t).reshape(-1, 1)  # (T, 1)
        return phi_t + K_t * x**2  # shape (T, N)

    def plot_approx_vs_analytic(self, results, timesteps):
        approx_q = results["q"]              # shape: (T, N_paths)
        x_vals = results["final_y"][:, :, 0] # shape: (T, N_paths)
        Y_vals = results["Y"]                # shape: (T, N_paths, 1)

        with torch.no_grad():
            true_q = self.optimal_control_analytic(timesteps, x_vals)          # shape: (T, N)
            true_Y = self.optimal_cost_analytic(timesteps, x_vals)             # shape (T, N)

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        # --- Subplot 1: Learned q(t) paths ---
        colors = cm.get_cmap("tab10", approx_q.shape[1])  # Get a colormap with enough distinct colors

        for i in range(approx_q.shape[1]):
            axs[0, 0].plot(timesteps, approx_q[:, i], color=colors(i), alpha=0.6, label=f"Learned {i+1}")
            axs[0, 0].plot(timesteps, true_q[:, i], linestyle="--", color=colors(i), label=f"Analytical {i+1}")

        axs[0, 0].set_title("Control $q(t)$: Learned vs Analytical")
        axs[0, 0].set_xlabel("Time $t$")
        axs[0, 0].set_ylabel("$q(t)$")
        axs[0, 0].grid(True)
        axs[0, 0].legend(ncol=2, fontsize=8)

        # --- Subplot 2: Absolute Difference ---
        diff = (approx_q.squeeze(-1) - true_q)  # (T, N_paths)
        for i in range(diff.shape[1]):
            axs[0, 1].plot(timesteps, diff[:, i], label=f"Diff Path {i+1}")
        axs[0, 1].set_title("Difference: Learned $-$ Analytical")
        axs[0, 1].set_xlabel("Time $t$")
        axs[0, 1].set_ylabel("$q(t) - q^*(t)$")
        axs[0, 1].grid(True)

        # --- Subplot 3: Y(t) paths ---
        for i in range(Y_vals.shape[1]):
            axs[1, 0].plot(timesteps, Y_vals[:, i, 0], color=colors(i), alpha=0.6, label=f"Learned {i+1}")
            axs[1, 0].plot(timesteps, true_Y[:, i], linestyle="--", color=colors(i), label=f"Analytical {i+1}")
        axs[1, 0].set_title("Cost-to-Go $Y(t)$")
        axs[1, 0].set_xlabel("Time $t$")
        axs[1, 0].set_ylabel("Y(t)")
        axs[1, 0].grid(True)
        axs[1, 0].legend(ncol=2, fontsize=8)

        # --- Subplot 4: x(t) paths ---
        for i in range(x_vals.shape[1]):
            axs[1, 1].plot(timesteps, x_vals[:, i])
        axs[1, 1].set_title("State $x(t)$")
        axs[1, 1].set_xlabel("Time $t$")
        axs[1, 1].set_ylabel("x(t)")
        axs[1, 1].grid(True)


        plt.tight_layout()
        plt.show()
