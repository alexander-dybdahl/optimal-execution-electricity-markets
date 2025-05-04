import torch
from config import sigma_P, sigma_D, sigma_B, T, rho, gamma, mu_P
from model.hjb import generator, terminal_cost
from core.base_bsde import BaseDeepBSDE

class HJMDeepBSDE(BaseDeepBSDE):
    def __init__(self, y0, xi, batch_size):
        super().__init__(y0, xi, batch_size)

    def generator(self, y, q):
        return generator(y, q)

    def terminal_cost(self, y):
        return terminal_cost(y, self.xi)

    def mu(self, t, y, q):
        # y = [X, P, D, B]
        dX = q
        dP = mu_P + gamma * q
        dD = torch.zeros_like(dP)  # purely diffusive
        dB = torch.zeros_like(dP)
        return torch.cat([dX, dP, dD, dB], dim=1)  # (batch, dim)

    def sigma(self, t, y):
        batch_size = t.shape[0]
        σ = torch.zeros(batch_size, 4, 3, device=t.device)  # (batch, dim, dW_dim)

        σ[:, 1, 0] = sigma_P(T, t).squeeze()                     # dWP affects P
        σ[:, 2, 0] = rho * sigma_D(T, t).squeeze()               # dWP affects D
        σ[:, 2, 1] = (1 - rho**2)**0.5 * sigma_D(T, t).squeeze() # dWT affects D
        σ[:, 3, 2] = sigma_B(T, t).squeeze()                     # dWB affects B

        return σ
