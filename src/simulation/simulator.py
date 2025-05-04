import torch
import numpy as np

from config import dt, T, N, dim, dim_w, mu_P, gamma, rho, sigma_P, sigma_D, sigma_B, xi, device
from model.hjb import generator

def simulate_all(model, n_paths=1000, batch_size=256, y0=None, y0_single=None, seed=42):
    torch.manual_seed(seed)
    model.eval()

    assert y0 is not None, "Initial state y0 must be provided"

    n_batches = n_paths // batch_size
    n_steps = N

    q_trajectories = []
    y_trajectories = []
    Y_trajectories = []
    terminal_stats = {"X": [], "D": [], "B": [], "I": []}

    for _ in range(n_batches):
        y = torch.zeros(batch_size, dim, device=device)
        if y0_single is not None:
            y = y0_single.repeat(batch_size, 1).to(device)

        X, P, D, B = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4]
        t = torch.zeros(batch_size, 1, device=device)

        # Initialize state
        X[:, 0] = y0[0, 0]
        P[:, 0] = y0[0, 1]
        D[:, 0] = y0[0, 2]
        B[:, 0] = y0[0, 3]

        q_traj = []
        Y = model.Y0.repeat(batch_size, 1)
        Y_traj = []
        y_traj = []

        for _ in range(n_steps):
            t_input = t.clone()
            q = model.q_net(t_input, y).squeeze(-1)
            z = model.z_net(t_input, y)
            f = generator(y, q.unsqueeze(-1))

            dW = torch.randn(batch_size, dim_w, device=device) * np.sqrt(dt)
            dWP, dWT, dWB = dW[:, 0:1], dW[:, 1:2], dW[:, 2:3]

            dX = q.unsqueeze(-1) * dt
            dP = (mu_P + gamma * q.unsqueeze(-1)) * dt + sigma_P(T, t_input) * dWP
            dD = rho * sigma_D(T, t_input) * dWP + np.sqrt(1 - rho**2) * sigma_D(T, t_input) * dWT
            dB = sigma_B(T, t_input) * dWB

            X = X + dX
            P = P + dP
            D = D + dD
            B = B + dB
            y = torch.cat([X, P, D, B], dim=1)

            Y = Y - f * dt + (z * dW).sum(dim=1, keepdim=True)

            t += dt
            q_traj.append(q.detach().cpu().numpy())
            Y_traj.append(Y.detach().cpu().numpy())
            y_traj.append(y.detach().cpu().numpy())

        q_trajectories.append(np.stack(q_traj, axis=0))
        Y_trajectories.append(np.stack(Y_traj, axis=0))
        y_trajectories.append(np.stack(y_traj, axis=0))

        I = X - D + xi
        terminal_stats["X"].append(X.detach().cpu())
        terminal_stats["D"].append(D.detach().cpu())
        terminal_stats["B"].append(B.detach().cpu())
        terminal_stats["I"].append(I.detach().cpu())

    results = {
        "q": np.concatenate(q_trajectories, axis=1),
        "Y": np.concatenate(Y_trajectories, axis=1),
        "final_y": np.concatenate(y_trajectories, axis=1),
        "X_T": torch.cat(terminal_stats["X"]).squeeze().numpy(),
        "D_T": torch.cat(terminal_stats["D"]).squeeze().numpy(),
        "B_T": torch.cat(terminal_stats["B"]).squeeze().numpy(),
        "I_T": torch.cat(terminal_stats["I"]).squeeze().numpy()
    }

    return results
