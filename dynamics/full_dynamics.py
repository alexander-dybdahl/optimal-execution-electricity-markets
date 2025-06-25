import torch
import numpy as np
import torch.nn as nn
from scipy.integrate import quad

from dynamics.dynamics import Dynamics


class FullDynamics(Dynamics):
    def __init__(self, dynamics_cfg, device="cpu"):
        super().__init__(dynamics_cfg=dynamics_cfg, device=device)
        self.sigma_P = dynamics_cfg["sigma_P"] # volatility for price
        self.alpha_P = dynamics_cfg["alpha_P"] # time dependent volatility for price
        self.beta_P = dynamics_cfg["beta_P"]   # constant volatility for price
        self.sigma_D = dynamics_cfg["sigma_D"] # volatility for demand
        self.alpha_D = dynamics_cfg["alpha_D"] # time dependent volatility for demand
        self.beta_D = dynamics_cfg["beta_D"]   # constant volatility for demand
        self.rho = dynamics_cfg["rho"]         # correlation between price and demand noise
        self.psi = dynamics_cfg["psi"]         # bid-ask spread
        self.gamma = dynamics_cfg["gamma"]     # temp impact
        self.nu = dynamics_cfg["nu"]           # perm impact
        self.eta = dynamics_cfg["eta"]         # terminal penalty
        self.mu_D = dynamics_cfg["mu_D"]       # drift for D
        self.mu_P = dynamics_cfg["mu_P"]       # drift for P

    def generator(self, y, q):
        P = y[:, 1:2]
        temporary_impact = self.gamma * q
        bid_ask_spread = torch.sign(q) * self.psi
        execution_price = P + bid_ask_spread + temporary_impact
        return q * execution_price

    def terminal_cost(self, y):
        D = y[:, 2:3]
        X = y[:, 0:1]
        return 0.5 * self.eta * (D - X)**2

    def mu(self, t, y, q):
        X = y[:, 0:1]
        P = y[:, 1:2]
        D = y[:, 2:3]

        dX = q
        dP = self.mu_P + self.nu * q # drift + permanent impact
        dD = self.mu_D * torch.ones_like(D)

        return torch.cat([dX, dP, dD], dim=1)

    def sigma(self, t, y):
        batch = y.shape[0]
        t_scaled = t / self.T  # normalize time to [0,1]

        sigma_P_t = torch.sqrt(self.alpha_P * t_scaled**2 + self.beta_P).squeeze(-1)
        sigma_D_t = torch.sqrt(self.alpha_D * t_scaled**2 + self.beta_D).squeeze(-1)

        Sigma = torch.zeros(batch, 3, 2, device=self.device)

        Sigma[:, 1, 0] = sigma_P_t                     # dP_t
        Sigma[:, 2, 0] = self.rho * sigma_D_t          # dD_t w/ correlated Brownian
        Sigma[:, 2, 1] = torch.sqrt(torch.tensor(1 - self.rho**2, device=self.device)) * sigma_D_t  # orthogonal noise for D

        return Sigma

    def optimal_control(self, t, y, dY_dy, smooth=True, eps=1.0):
        dY_dX = dY_dy[:, 0:1]
        dY_dP = dY_dy[:, 1:2]
        P = y[:, 1:2]

        # Define Lambda = P + dV/dX + nu * dV/dP
        Lambda = P + dY_dX + self.nu * dY_dP
        psi = self.psi  # half-spread, scalar or tensor
        gamma = self.gamma  # temporary impact

        if psi == 0:
            # If psi is zero, the control is simply the optimal control without bid-ask spread
            q = -Lambda / (2 * gamma)
            return q

        if smooth:
            self.eps = nn.Parameter(torch.tensor(eps))
            sigmoid = lambda x: torch.sigmoid(x / self.eps)

            # Smooth control transitions
            q_pos = -(Lambda + psi) / (2 * gamma)
            q_neg = -(Lambda - psi) / (2 * gamma)

            q = q_pos * sigmoid(Lambda - psi) + q_neg * sigmoid(-Lambda - psi)
        else:
            # Piecewise (non-smooth) definition
            q_pos = -(Lambda + psi) / (2 * gamma)
            q_neg = -(Lambda - psi) / (2 * gamma)

            q = torch.where(
                Lambda > psi, q_pos,
                torch.where(
                    Lambda < -psi, q_neg,
                    torch.zeros_like(Lambda)
                )
            )
        return q

    def optimal_control_analytic(self, t, y):
        X = y[:, 0:1]
        P = y[:, 1:2]
        D = y[:, 2:3]
        tau = self.T - t
        return (self.eta * (self.mu_D * tau + D - X) - P) / ((self.eta + self.nu) * tau + 2 * self.gamma)

    def value_function_analytic(self, t_tensor, y_tensor):
        t = t_tensor.detach().cpu().numpy() if isinstance(t_tensor, torch.Tensor) else t_tensor
        y = y_tensor.detach().cpu().numpy() if isinstance(y_tensor, torch.Tensor) else y_tensor

        X = y[:, 0:1]
        P = y[:, 1:2]
        D = y[:, 2:3]

        T = self.T
        eta = self.eta
        nu = self.nu
        gamma = self.gamma
        mu = self.mu_D
        rho = self.rho

        # Time-dependent sigmas
        def sigma_P2(s): return self.alpha_P * s**2 + self.beta_P
        def sigma_D2(s): return self.alpha_D * s**2 + self.beta_D
        def sigma_P(s): return np.sqrt(sigma_P2(s))
        def sigma_D(s): return np.sqrt(sigma_D2(s))

        # Coefficients
        def A(s): return eta * (0.5 * nu * s + gamma) / ((eta + nu) * s + 2 * gamma)
        def B(s): return -0.5 * s / ((eta + nu) * s + 2 * gamma)
        def F(s): return eta * s / ((eta + nu) * s + 2 * gamma)

        V = np.zeros((y.shape[0], 1))
        for i in range(y.shape[0]):
            tau_i = T - t[i]  # scalar
            X_i, P_i, D_i = X[i], P[i], D[i]

            def integrand(s):
                return B(s) * sigma_P2(s) + A(s) * sigma_D2(s) + rho * F(s) * sigma_P(s) * sigma_D(s)

            K_i, _ = quad(integrand, 0, tau_i)

            A_tau = A(tau_i)
            B_tau = B(tau_i)
            F_tau = F(tau_i)
            G = 2 * mu * tau_i * A_tau
            H = -2 * eta * mu * tau_i * B_tau

            z = D_i - X_i
            V[i] = A_tau * z**2 + B_tau * P_i**2 + F_tau * z * P_i + G * z + H * P_i + K_i

        return torch.tensor(V, dtype=torch.float32, device=t_tensor.device if isinstance(t_tensor, torch.Tensor) else "cpu")
