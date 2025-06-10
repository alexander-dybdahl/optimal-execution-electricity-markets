import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

# Define the PINN model
class ValueFunctionNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # Ensure input_dim = 1 + dim
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, t, y):
        inp = torch.cat([t, y], dim=1)  # Shape should be (batch_size, 1 + dim)
        return self.net(inp)

# HJB loss from stochastic control problem
def hjb_residual(model, dynamics, t, y, create_graph=True):
    t = t.detach().requires_grad_(True)
    y = y.detach().requires_grad_(True)
    Y = model(t, y)

    dY_dt = torch.autograd.grad(Y, t, grad_outputs=torch.ones_like(Y), retain_graph=True, create_graph=create_graph)[0]
    dY_dy = torch.autograd.grad(Y, y, grad_outputs=torch.ones_like(Y), retain_graph=True, create_graph=create_graph)[0]

    q = dynamics.optimal_control(t, y, Y, create_graph=create_graph)
    f = dynamics.generator(y, q)
    mu = dynamics.mu(t, y, q)
    drift_term = (mu * dY_dy).sum(dim=1, keepdim=True)

    batch_size, dim = y.shape
    hess_diag = torch.zeros(batch_size, dim, device=y.device)
    for i in range(dim):
        grad_i = torch.autograd.grad(dY_dy[:, i], y, grad_outputs=torch.ones_like(dY_dy[:, i]), retain_graph=True, create_graph=create_graph)[0]
        hess_diag[:, i] = grad_i[:, i]

    sigma = dynamics.sigma(t, y)
    sigma_sigmaT = torch.bmm(sigma, sigma.transpose(1, 2))
    trace_term = (sigma_sigmaT[:, range(dim), range(dim)] * hess_diag).sum(dim=1, keepdim=True)
    diffusion_term = 0.5 * trace_term

    residual = dY_dt + drift_term + diffusion_term + f
    return residual.pow(2).mean()

# Updated training function using forward simulation with terminal and PINN losses
def train_pinn(model, dynamics, model_cfg, device, n_epochs=1000, lr=1e-3, batch_size=512):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []

    dim = model_cfg["dim"]
    T = model_cfg["T"]
    N = model_cfg.get("N", 20)
    dt = T / N
    dynamics.N = N
    dynamics.dt = dt

    for epoch in range(n_epochs):
        torch.manual_seed(epoch)

        # Sample Brownian increments
        dW = torch.randn(batch_size, N, dynamics.dim_W, device=device) * (dt ** 0.5)
        t_paths = torch.zeros(batch_size, N + 1, 1, device=device)
        W_paths = torch.zeros(batch_size, N + 1, dynamics.dim_W, device=device)

        for n in range(1, N + 1):
            t_paths[:, n, :] = t_paths[:, n - 1, :] + dt
            W_paths[:, n, :] = W_paths[:, n - 1, :] + dW[:, n - 1, :]

        y0 = dynamics.y0.repeat(batch_size, 1).to(device)
        y = y0.clone()
        t = t_paths[:, 0, :]
        W = W_paths[:, 0, :]

        Y = model(t, y)
        dY = torch.autograd.grad(Y, y, grad_outputs=torch.ones_like(Y), create_graph=True, retain_graph=True)[0]

        Y_loss = 0.0

        for n in range(N):
            t1 = t_paths[:, n + 1, :]
            W1 = W_paths[:, n + 1, :]
            Sigma = dynamics.sigma(t, y)
            Z = torch.bmm(Sigma.transpose(1, 2), dY.unsqueeze(-1)).squeeze(-1)
            q = dynamics.optimal_control(t, y, Y)
            y1 = dynamics.forward_dynamics(y, q, W1 - W, t, dt)
            Y1 = model(t1, y1)

            f = dynamics.generator(y, q)
            Y1_tilde = Y - f * dt + (Z * (W1 - W)).sum(dim=1, keepdim=True)
            Y_loss += (Y1 - Y1_tilde).pow(2).mean()

            t, W, y, Y = t1, W1, y1, Y1
            dY = torch.autograd.grad(Y, y, grad_outputs=torch.ones_like(Y), create_graph=True, retain_graph=True)[0]

        terminal_supervision = ((Y - dynamics.terminal_cost(y)) ** 2).mean()

        # Additional physics-informed interior loss (PINN)
        n_phys = batch_size * N
        t_phys = torch.rand(n_phys, 1, device=device) * T
        y_phys = torch.randn(n_phys, dim, device=device) * 2

        pinn_loss = hjb_residual(model, dynamics, t_phys, y_phys)

        total_loss = Y_loss + terminal_supervision + pinn_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        losses.append(total_loss.item())
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Total Loss = {total_loss.item():.5f}, Y Loss = {Y_loss.item():.5f}, Terminal = {terminal_supervision.item():.5f}, PINN = {pinn_loss.item():.5f}")

    return losses

def simulate_and_plot_paths(model, dynamics, model_cfg, device, n_sim=5, seed=42):
    torch.manual_seed(seed)

    N = dynamics.N
    T = dynamics.T
    dt = T / N
    dim = model_cfg["dim"]
    dim_W = dynamics.dim_W

    y0 = dynamics.y0.repeat(n_sim, 1).to(device)
    t_scalar = 0.0

    # Containers
    y_traj = []
    Y_traj = []
    q_traj = []

    y_true_traj, Y_true_traj, q_true_traj = [], [], []

    y = y0.clone()
    y_true = y0.clone()

    for step in range(N + 1):
        t_tensor = torch.full((n_sim, 1), t_scalar, device=device)

        Y = model(t_tensor, y)
        dY = torch.autograd.grad(Y, y, grad_outputs=torch.ones_like(Y), retain_graph=True, create_graph=False)[0]
        q = dynamics.optimal_control(t_tensor, y, Y, create_graph=False)

        y_traj.append(y.detach().cpu().numpy())
        Y_traj.append(Y.detach().cpu().numpy())
        q_traj.append(q.detach().cpu().numpy())

        if dynamics.analytical_known:
            Y_true = dynamics.value_function_analytic(t_tensor, y_true)
            q_true = dynamics.optimal_control(t_tensor, y_true, Y_true, create_graph=False)

            y_true_traj.append(y_true.detach().cpu().numpy())
            Y_true_traj.append(Y_true.detach().cpu().numpy())
            q_true_traj.append(q_true.detach().cpu().numpy())

        # simulate next step
        if step < N:
            dW = torch.randn(n_sim, dim_W, device=device) * dt**0.5
            y = dynamics.forward_dynamics(y, q, dW, t_tensor, dt)
            if dynamics.analytical_known:
                y_true = dynamics.forward_dynamics(y_true, q_true, dW, t_tensor, dt)
            t_scalar += dt

    time_grid = np.linspace(0, T, N + 1)

    # === Plot Value Function ===
    plt.figure(figsize=(12, 5))
    for i in range(n_sim):
        plt.plot(time_grid, [Y_traj[n][i, 0] for n in range(N + 1)], label=f'PINN Sim {i}')
        if dynamics.analytical_known:
            plt.plot(time_grid, [Y_true_traj[n][i, 0] for n in range(N + 1)], '--', label=f'Analytic Sim {i}')
    plt.xlabel('Time')
    plt.ylabel('Value Function V(t, y)')
    plt.title('Learned vs Analytic Value Function')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === Plot State Trajectories (only first coordinate) ===
    plt.figure(figsize=(12, 5))
    for i in range(n_sim):
        plt.plot(time_grid, [y_traj[n][i, 0] for n in range(N + 1)], label=f'PINN Sim {i}')
        if dynamics.analytical_known:
            plt.plot(time_grid, [y_true_traj[n][i, 0] for n in range(N + 1)], '--', label=f'Analytic Sim {i}')
    plt.xlabel('Time')
    plt.ylabel('State y[0]')
    plt.title('Simulated State Trajectories')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
