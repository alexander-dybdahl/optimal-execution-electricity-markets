import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import matplotlib.cm as cm

from dynamics.dynamics import Dynamics


class FullDynamics(Dynamics):
    def __init__(self, dynamics_cfg, device="cpu"):
        super().__init__(dynamics_cfg=dynamics_cfg, device=device)
        self.alpha_P = dynamics_cfg["alpha_P"]
        self.beta_P = dynamics_cfg["beta_P"]
        self.alpha_D = dynamics_cfg["alpha_D"]
        self.beta_D = dynamics_cfg["beta_D"]
        self.rho = dynamics_cfg["rho"]
        self.psi = dynamics_cfg["psi"]         # bid-ask spread
        self.gamma = dynamics_cfg["gamma"]     # temp impact
        self.nu = dynamics_cfg["nu"]           # perm impact
        self.eta = dynamics_cfg["eta"]         # terminal penalty
        self.mu_D = dynamics_cfg["mu_D"]       # drift for D
        self.mu_P = dynamics_cfg["mu_P"]       # drift for D

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

        sigma_P_t = torch.sqrt(self.alpha_P * t_scaled**2 + self.beta_P)
        sigma_D_t = torch.sqrt(self.alpha_D * t_scaled**2 + self.beta_D)

        Sigma = torch.zeros(batch, 3, 2, device=self.device)

        Sigma[:, 1, 0] = sigma_P_t
        Sigma[:, 2, 0] = self.rho * sigma_D_t
        Sigma[:, 2, 1] = torch.sqrt(1 - self.rho**2) * sigma_D_t

        return Sigma

    def optimal_control(self, t, y, dY_dy, smooth=True, eps=1):
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

    def value_function_analytic(self, t, y):
        X = y[:, 0:1]
        P = y[:, 1:2]
        D = y[:, 2:3]

        T = self.T
        eta = self.eta
        nu = self.nu
        gamma = self.gamma
        mu = self.mu_D
        sigma_p = self.sigma_P
        sigma_d = self.sigma_D
        rho = self.rho

        tau = T - t
        denom = (eta + nu) * tau + 2 * gamma

        # Coefficient functions
        A = eta * (0.5 * nu * tau + gamma) / denom
        B = -0.5 * tau / denom
        F = eta * tau / denom
        G = 2 * mu * tau * A
        H = -2 * eta * mu * tau * B

        # Constant K(t)
        log_term = gamma * (sigma_p**2 + sigma_d**2 * eta**2 - 2 * rho * sigma_p * sigma_d * eta) / (eta + nu)**2
        log_expr = 1 + ((eta + nu) * tau) / (2 * gamma)
        K1 = log_term * torch.log(log_expr)

        K2 = (sigma_d**2 * eta * nu + 2 * rho * sigma_p * sigma_d * eta - sigma_p**2) / (2 * (eta + nu)) * tau
        K3 = eta * mu**2 * tau**2 * (0.5 * nu * tau + gamma) / denom
        K = K1 + K2 + K3

        z = D - X
        V = A * z**2 + B * P**2 + F * z * P + G * z + H * P + K
        return V