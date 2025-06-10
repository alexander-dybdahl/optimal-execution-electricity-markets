import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Define the PINN model
class ValueFunctionNN(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, t, y):
        inp = torch.cat([t, y], dim=1)
        return self.net(inp)

# HJB loss from stochastic control problem
def hjb_residual(model, dynamics, t, y, sigma, create_graph=True):
    t.requires_grad_(True)
    y.requires_grad_(True)
    Y = model(t, y)

    dY_dy = torch.autograd.grad(Y, y, grad_outputs=torch.ones_like(Y),
                                retain_graph=True, create_graph=create_graph)[0]
    dY_dt = torch.autograd.grad(Y, t, grad_outputs=torch.ones_like(Y),
                                retain_graph=True, create_graph=create_graph)[0]
    d2Y_dyy = torch.autograd.grad(dY_dy, y, grad_outputs=torch.ones_like(dY_dy),
                                  retain_graph=True, create_graph=create_graph)[0]

    q = -0.5 * dY_dy
    generator = q**2 + y**2
    sigma_sq = (sigma ** 2).view(-1, 1)
    diffusion_term = 0.5 * sigma_sq * d2Y_dyy
    H = generator + q * dY_dy
    residual = dY_dt + H + diffusion_term
    return residual.pow(2).mean()

# Terminal loss for the value function
def terminal_loss(model, dynamics, T, device, n_samples=1024):
    y = torch.randn(n_samples, 1, device=device) * 2
    t = torch.full_like(y, fill_value=T)
    Y = model(t, y)
    Y_true = dynamics.terminal_cost(y)
    return ((Y - Y_true) ** 2).mean()

# Generate synthetic training data
def generate_training_data(n_samples, T, device):
    t = torch.rand(n_samples, 1, device=device) * T
    y = torch.randn(n_samples, 1, device=device) * 2
    return t, y

# Training loop
def train_pinn(model, dynamics, T, device, n_epochs=1000, lr=1e-3, n_samples=2048):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(n_epochs):
        t, y = generate_training_data(n_samples, T, device)
        sigma = dynamics.sigma(t, y)[:, 0, 0]
        loss = hjb_residual(model, dynamics, t, y, sigma) + terminal_loss(model, dynamics, T, device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.5f}")

    return losses

# Visualize comparison
def plot_value_comparison(model, dynamics, T, device):
    y_vals = torch.linspace(-3, 3, 100, device=device).view(-1, 1)
    t_vals = torch.zeros_like(y_vals)

    model.eval()
    with torch.no_grad():
        Y_pinn = model(t_vals, y_vals)
        Y_true = dynamics.value_function_analytic(t_vals, y_vals)

    y_vals_cpu = y_vals.cpu().numpy()
    Y_pinn_cpu = Y_pinn.cpu().numpy()
    Y_true_cpu = Y_true.cpu().numpy()

    plt.plot(y_vals_cpu, Y_true_cpu, label='Analytic')
    plt.plot(y_vals_cpu, Y_pinn_cpu, '--', label='PINN')
    plt.xlabel('y')
    plt.ylabel('V(0, y)')
    plt.legend()
    plt.title('Value Function at t=0')
    plt.grid(True)
    plt.show()
