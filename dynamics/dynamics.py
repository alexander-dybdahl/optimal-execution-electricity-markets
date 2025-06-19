from abc import ABC, abstractmethod

import torch
import torch.distributed as dist


class Dynamics(ABC):
    def __init__(self, args, model_cfg):
        # TODO: Check if need super init
        # super().__init__()
        # System & Execution Settings
        self.is_distributed = dist.is_initialized()
        self.device = args.device_set
        self.world_size = dist.get_world_size() if self.is_distributed else 1
        self.is_main = not self.is_distributed or dist.get_rank() == 0

        # Time Discretization
        self.t0 = 0.0
        self.T = model_cfg["T"]
        self.N = model_cfg["N"]
        self.dt = model_cfg["dt"]

        # Problem Setup
        self.dim = model_cfg["dim"]  # state space dimension
        self.dim_W = model_cfg["dim_W"]  # Brownian motion dimension
        self.y0 = torch.tensor(
            [model_cfg["y0"]], device=self.device, requires_grad=True
        )
        self.analytical_known = model_cfg["analytical_known"]

    @abstractmethod
    def generator(self, y, q):
        pass

    @abstractmethod
    def terminal_cost(self, y):
        pass

    def terminal_cost_grad(self, y):
        return torch.autograd.grad(
            outputs=self.terminal_cost(y),
            inputs=y,
            grad_outputs=torch.ones_like(self.terminal_cost(y)),
            create_graph=True,
            retain_graph=True,
        )[0]

    @abstractmethod
    def mu(self, t, y, q):
        pass  # shape: (batch, dim)

    @abstractmethod
    def sigma(self, t, y):
        pass  # shape: (batch, dim, dW_dim)

    def forward_dynamics(self, y, q, dW, t, dt):
        mu = self.mu(t, y, q)  # shape: (batch, dim)
        Sigma = self.sigma(t, y)  # shape: (batch, dim, dW_dim)
        diffusion = torch.bmm(Sigma, dW.unsqueeze(-1)).squeeze(
            -1
        )  # shape: (batch, dim)
        return y + mu * dt + diffusion  # shape: (batch, dim)    @abstractmethod

    @abstractmethod
    def optimal_control(self, t, y, dY_dy):
        pass

    @abstractmethod
    def optimal_control_analytic(self, t, y):
        pass

    @abstractmethod
    def value_function_analytic(self, t, y):
        pass

    def generate_paths(self, batch_size, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        dW = torch.randn(batch_size, self.N, self.dim_W, device=self.device) * (
            self.dt**0.5
        )

        W = torch.cat(
            [
                torch.zeros(batch_size, 1, self.dim_W, device=self.device),
                torch.cumsum(dW, dim=1),
            ],
            dim=1,
        )  # (batch_size, N+1, dim_W)

        t = (
            torch.linspace(0, self.T, self.N + 1, device=self.device)
            .view(1, -1, 1)
            .repeat(batch_size, 1, 1)
        )
        return t, dW, W
