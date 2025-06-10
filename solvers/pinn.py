import torch
import torch.nn as nn
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

# Terminal loss for the value function
def terminal_loss(model, dynamics, T, dim, device, n_samples=1024):
    y = torch.randn(n_samples, dim, device=device) * 2
    t = torch.full((n_samples, 1), fill_value=T, device=device)  # FIXED
    Y = model(t, y)
    Y_true = dynamics.terminal_cost(y)
    return ((Y - Y_true) ** 2).mean()

# Generate synthetic training data
def generate_training_data(n_samples, dim, T, device):
    t = torch.rand(n_samples, 1, device=device) * T
    y = torch.randn(n_samples, dim, device=device) * 2
    return t, y

# Training loop
def train_pinn(model, dynamics, model_cfg, device, n_epochs=1000, lr=1e-3, n_samples=2048):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []

    dim = model_cfg["dim"]
    T = model_cfg["T"]

    for epoch in range(n_epochs):
        t, y = generate_training_data(n_samples, dim, T, device)
        loss = hjb_residual(model, dynamics, t, y) + terminal_loss(model, dynamics, T, dim, device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.5f}")

    return losses

# Plot at t=0
def plot_value_comparison(model, dynamics, model_cfg, device):
    model.eval()
    dim = model_cfg["dim"]
    T = model_cfg["T"]
    y_range = torch.linspace(-3, 3, 200, device=device).view(-1, 1)
    y_base = torch.zeros(y_range.size(0), dim, device=device)
    y_base[:, 0:1] = y_range  # vary only first coordinate

    t_vals = torch.zeros_like(y_range)
    with torch.no_grad():
        Y_pinn = model(t_vals, y_base)
        Y_true = dynamics.value_function_analytic(t_vals, y_base)

    y_vals_cpu = y_range.cpu().numpy()
    Y_pinn_cpu = Y_pinn.cpu().numpy()
    Y_true_cpu = Y_true.cpu().numpy()

    plt.plot(y_vals_cpu, Y_true_cpu, label='Analytic')
    plt.plot(y_vals_cpu, Y_pinn_cpu, '--', label='PINN')
    plt.xlabel('y[0]')
    plt.ylabel('V(0, y)')
    plt.legend()
    plt.title('Value Function at t=0')
    plt.grid(True)
    plt.show()

# Plot across time
def plot_value_comparison_over_time(model, dynamics, model_cfg, device, time_slices=[0.0, 0.25, 0.5, 0.75, 1.0]):
    model.eval()
    dim = model_cfg["dim"]
    y_range = torch.linspace(-3, 3, 200, device=device).view(-1, 1)
    y_base = torch.zeros(y_range.size(0), dim, device=device)
    y_base[:, 0:1] = y_range

    plt.figure(figsize=(12, 8))
    for t_scalar in time_slices:
        t_tensor = torch.full_like(y_range, fill_value=t_scalar, device=device)
        with torch.no_grad():
            V_pinn = model(t_tensor, y_base)
            V_true = dynamics.value_function_analytic(t_tensor, y_base)

        plt.plot(y_range.cpu().numpy(), V_true.cpu().numpy(), label=f'Analytic t={t_scalar:.2f}')
        plt.plot(y_range.cpu().numpy(), V_pinn.cpu().numpy(), '--', label=f'PINN t={t_scalar:.2f}')

    plt.xlabel("y[0]")
    plt.ylabel("V(t, y)")
    plt.title("Comparison of PINN and Analytic Value Function over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
