import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

from dynamics.dynamics import Dynamics


class SimpleDynamics(Dynamics):
    def __init__(self, dynamics_cfg, device="cpu"):
        super().__init__(dynamics_cfg=dynamics_cfg, device=device)
        self.sigma_y = dynamics_cfg["sigma"]
        self.G = dynamics_cfg["G"]

    def generator(self, y, q):
        return q**2 + y**2

    def terminal_cost(self, y):
        return self.G * y**2

    def mu(self, t, y, q):
        return q

    def sigma(self, t, y):
        batch_size = y.shape[0]
        Sigma = torch.zeros(batch_size, 1, 1, device=self.device)
        Sigma[:, 0, 0] = self.sigma_y
        return Sigma
    
    def optimal_control(self, t, y, dY_dy):
        q = - 0.5 * dY_dy
        return q

    def K_analytic(self, t):
        """Analytical solution to Riccati equation"""
        G = self.G
        T = torch.tensor(self.T, device=self.device)
        A = (G + 1) * torch.exp(2 * (T - t))
        return (A + (G - 1)) / (A - (G - 1))

    def optimal_control_analytic(self, t, y):
        """Optimal q(t, y) = -K(t) * y"""
        K_t = self.K_analytic(t)
        return -K_t * y

    def phi_analytic(self, t):
        """Analytical phi(t) from backward integral of K"""
        sigma = self.sigma_y
        G = self.G
        T = torch.tensor(self.T, device=self.device)
        def A(s): return (G + 1) * torch.exp(2 * (T - s))
        A_t = A(t)
        A_T = A(self.T)
        log_expr = (A_T / A_t) * ((A_t - (G - 1))**2 / (A_T - (G - 1))**2)
        phi = 0.5 * sigma**2 * torch.log(log_expr)
        return phi

    def value_function_analytic(self, t, y):
        """Analytical cost-to-go Y(t) = phi(t) + K(t) * y^2"""
        K_t = self.K_analytic(t)
        phi_t = self.phi_analytic(t)
        return phi_t + K_t * y**2