import torch

from dynamics.dynamics import Dynamics


class HJBDynamics(Dynamics):
    def __init__(self, args, model_cfg):
        super().__init__(args, model_cfg)
        self.xi = torch.tensor(model_cfg["xi"], device=self.device)
        self.gamma = model_cfg["gamma"]
        self.eta = model_cfg["eta"]
        self.mu_P = model_cfg["mu_P"]
        self.rho = model_cfg["rho"]
        self.vol_P = model_cfg["vol_P"]
        self.vol_D = model_cfg["vol_D"]
        self.vol_B = model_cfg["vol_B"]
        self.dim_W = model_cfg["dim_W"]

    def psi(self, q):
        return self.gamma * q

    def phi(self, q):
        return self.eta * q

    def sigma_P(self, t):
        return self.vol_P * torch.ones_like(t, device=self.device)

    def sigma_D(self, t):
        return self.vol_D * torch.ones_like(t, device=self.device) # (self.T - t) / self.T

    def sigma_B(self, t):
        return self.vol_B * torch.ones_like(t, device=self.device) # (self.T - t) / self.T

    def generator(self, y, q):
        P = y[:, 1:2]
        sign_q = torch.sign(q)
        P_exec = P + sign_q * self.psi(q) + self.phi(q)
        return -q * P_exec

    def terminal_cost(self, y):
        X = y[:, 0]
        D = y[:, 2]
        B = y[:, 3]
        I = X - D + self.xi
        I_plus = torch.clamp(I, min=0.0)
        I_minus = torch.clamp(-I, min=0.0)
        return I_plus * B - I_minus * B

    def mu(self, t, y, q):
        dX = q
        dP = self.mu_P + self.gamma * q
        dD = torch.zeros_like(dP)
        dB = torch.zeros_like(dP)
        return torch.cat([dX, dP, dD, dB], dim=1)

    def sigma(self, t, y):
        batch_size = t.shape[0]
        Sigma = torch.zeros(batch_size, 4, 3, device=self.device)
        Sigma[:, 1, 0] = self.sigma_P(t).squeeze()
        Sigma[:, 2, 0] = self.rho * self.sigma_D(t).squeeze()
        Sigma[:, 2, 1] = (1 - self.rho**2) ** 0.5 * self.sigma_D(t).squeeze()
        Sigma[:, 3, 2] = self.sigma_B(t).squeeze()
        return Sigma
    