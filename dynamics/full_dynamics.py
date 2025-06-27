import torch
import numpy as np
import torch.nn as nn
from scipy.integrate import quad
from scipy.interpolate import interp1d

from dynamics.dynamics import Dynamics


class FullDynamics(Dynamics):
    def __init__(self, dynamics_cfg, device="cpu"):
        super().__init__(dynamics_cfg=dynamics_cfg, device=device)
        
        self.mu_D = dynamics_cfg["mu_D"]       # drift for D
        self.mu_P = dynamics_cfg["mu_P"]       # drift for P
        
        self.time_dep_vol = dynamics_cfg["time_dep_vol"]  # boolean for time-dependent volatility
        self.low_vol = dynamics_cfg["low_vol"]  # boolean for low volatility regime

        self.alpha_P = dynamics_cfg["alpha_P"] # time dependent volatility for price
        self.beta_P = dynamics_cfg["beta_P"]   # constant volatility for price

        self.alpha_D = dynamics_cfg["alpha_D"] # time dependent volatility for demand
        self.beta_D = dynamics_cfg["beta_D"]   # constant volatility for demand

        if self.low_vol:
            self.sigma_P = np.sqrt(self.beta_P)
            self.sigma_D = np.sqrt(self.beta_D)
        else:
            self.sigma_P = np.sqrt(self.alpha_P + self.beta_P)
            self.sigma_D = np.sqrt(self.alpha_D + self.beta_D)

        self.rho = dynamics_cfg["rho"]         # correlation between price and demand noise

        self.psi = dynamics_cfg["psi"]         # bid-ask spread
        self.psi_end = self.psi                # bid-ask spread
        self.gamma = dynamics_cfg["gamma"]     # temp impact
        self.gamma_end = self.gamma            # temp impact
        self.nu = dynamics_cfg["nu"]           # perm impact
        self.nu_end = self.nu                  # perm impact

        self.eps = nn.Parameter(torch.tensor(dynamics_cfg["eps"]))

        self.eta = dynamics_cfg["eta"]         # terminal penalty

        self.use_exact = dynamics_cfg["analytical_known"]  # whether to plot exact solution
        if self.use_exact:
            self.K_interp = None  # placeholder for K interpolator
            self.precompute_K_table()  # precompute the K(τ) interpolation

    def anneal(self, epoch, total_epochs):
        """Linearly anneal psi, gamma, and nu from start to target values."""
        progress = min(epoch / total_epochs, 1.0)
        self.psi_start = 0.1
        self.gamma_start = 0.1
        self.nu_start = 0.1
        
        self.psi = (1 - progress) * self.psi_start + progress * self.psi_end
        self.gamma = (1 - progress) * self.gamma_start + progress * self.gamma_end
        self.nu = (1 - progress) * self.nu_start + progress * self.nu_end

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
        if self.time_dep_vol:
            return self.sigma_time_dep(t, y)
        else:
            return self.sigma_const(t, y)

    def sigma_time_dep(self, t, y):
        batch = y.shape[0]
        t_scaled = t / self.T  # normalize time to [0,1]

        sigma_P_t = torch.sqrt(self.alpha_P * t_scaled + self.beta_P).squeeze(-1)
        sigma_D_t = torch.sqrt(self.alpha_D * t_scaled + self.beta_D).squeeze(-1)

        Sigma = torch.zeros(batch, 3, 2, device=self.device)

        Sigma[:, 1, 0] = sigma_P_t                     # dP_t
        Sigma[:, 2, 0] = self.rho * sigma_D_t          # dD_t w/ correlated Brownian
        Sigma[:, 2, 1] = torch.sqrt(torch.tensor(1 - self.rho**2, device=self.device)) * sigma_D_t  # orthogonal noise for D

        return Sigma

    def sigma_const(self, t, y):
        batch = y.shape[0]

        Sigma = torch.zeros(batch, 3, 2, device=self.device)
        Sigma[:, 1, 0] = self.sigma_P                          # dP = ... dW1
        Sigma[:, 2, 0] = self.rho * self.sigma_D               # dD = ... dW1
        Sigma[:, 2, 1] = (1 - self.rho**2)**0.5 * self.sigma_D # dD = ... dW2
        return Sigma

    def optimal_control(self, t, y, dY_dy, smooth=True):
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
        if self.time_dep_vol and self.use_exact:
            return self.value_function_analytic_time_dep(t, y)
        else:
            return self.value_function_analytic_const(t, y)

    def precompute_K_table(self, num_points=500):
        T = self.T
        eta = self.eta
        nu = self.nu
        gamma = self.gamma
        rho = self.rho

        # Time-dependent volatilities
        def sigma_P2(s): return self.alpha_P * s**2 + self.beta_P
        def sigma_D2(s): return self.alpha_D * s**2 + self.beta_D
        def sigma_P(s): return np.sqrt(sigma_P2(s))
        def sigma_D(s): return np.sqrt(sigma_D2(s))

        # Coefficient functions
        def A(s): return eta * (0.5 * nu * s + gamma) / ((eta + nu) * s + 2 * gamma)
        def B(s): return -0.5 * s / ((eta + nu) * s + 2 * gamma)
        def F(s): return eta * s / ((eta + nu) * s + 2 * gamma)

        def integrand(s):
            return B(s) * sigma_P2(s) + A(s) * sigma_D2(s) + rho * F(s) * sigma_P(s) * sigma_D(s)

        tau_grid = np.linspace(0, T, num_points)
        K_values = np.array([quad(integrand, 0, tau)[0] for tau in tau_grid])

        # Save interpolator
        self.K_interp = interp1d(tau_grid, K_values, kind="cubic", fill_value="extrapolate")

    def value_function_analytic_time_dep(self, t_tensor, y_tensor):
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

        tau = T - t.flatten()  # shape: (batch,)
        tau = np.clip(tau, 0.0, T)  # ensure valid range

        # Precompute coefficients
        def A(s): return eta * (0.5 * nu * s + gamma) / ((eta + nu) * s + 2 * gamma)
        def B(s): return -0.5 * s / ((eta + nu) * s + 2 * gamma)
        def F(s): return eta * s / ((eta + nu) * s + 2 * gamma)

        A_tau = A(tau).reshape(-1, 1)
        B_tau = B(tau).reshape(-1, 1)
        F_tau = F(tau).reshape(-1, 1)
        G = (2 * mu * tau).reshape(-1, 1) * A_tau
        H = (-2 * eta * mu * tau).reshape(-1, 1) * B_tau

        z = D - X
        K = self.K_interp(tau).reshape(-1, 1)

        V = A_tau * z**2 + B_tau * P**2 + F_tau * z * P + G * z + H * P + K
        return torch.tensor(V, dtype=torch.float32, device=t_tensor.device if isinstance(t_tensor, torch.Tensor) else "cpu")

    def value_function_analytic_const(self, t, y):
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