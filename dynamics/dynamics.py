from abc import ABC, abstractmethod

import numpy as np
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
        self.dim = model_cfg["dim"]        # state space dimension
        self.dim_W = model_cfg["dim_W"]    # Brownian motion dimension
        self.y0 = torch.tensor([model_cfg["y0"]], device=self.device, requires_grad=True)
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
        pass                                # shape: (batch, dim)

    @abstractmethod
    def sigma(self, t, y):
        pass                                # shape: (batch, dim, dW_dim)

    def forward_dynamics(self, y, q, dW, t, dt):
        mu = self.mu(t, y, q)               # shape: (batch, dim)
        Sigma = self.sigma(t, y)            # shape: (batch, dim, dW_dim)
        diffusion = torch.bmm(Sigma, dW.unsqueeze(-1)).squeeze(
            -1
    )                                       # shape: (batch, dim)
        return y + mu * dt + diffusion      # shape: (batch, dim)

    @abstractmethod
    def optimal_control(self, t, y, Y):
        pass

    @abstractmethod
    def optimal_control_analytic(self, t, y):
        pass

    @abstractmethod
    def value_function_analytic(self, t, y):
        pass

    def simulate_paths(self, agent, n_sim=5, seed=42, y0_single=None):
        torch.manual_seed(seed)

        y0 = (
            y0_single.repeat(n_sim, 1)
            if y0_single is not None
            else self.y0.repeat(n_sim, 1)
        )
        t_scalar = 0.0

        # Initialize both
        y_agent = y0.clone()
        y_analytical = y0.clone()

        Y_agent_traj = []
        y_agent_traj = []
        q_agent_traj = []
        if self.analytical_known:
            Y_analytical_traj = []
            y_analytical_traj = []
            q_analytical_traj = []

        for step in range(self.N + 1):
            t_tensor = torch.full((n_sim, 1), t_scalar, device=self.device)

            # Predict Y and compute control
            # TODO: Check that this is on the correct device
            Y_agent = agent.predict(t_tensor, y_agent)
            q_agent = self.optimal_control(
                t_tensor, y_agent, Y_agent, create_graph=False
            )
            if self.analytical_known:
                Y_analytical = self.value_function_analytic(t_tensor, y_analytical)
                q_analytical = self.optimal_control(
                    t_tensor, y_analytical, Y_analytical, create_graph=False
                )

            # Save states and controls
            Y_agent_traj.append(Y_agent.detach().cpu().numpy())
            y_agent_traj.append(y_agent.detach().cpu().numpy())
            q_agent_traj.append(q_agent.detach().cpu().numpy())

            if self.analytical_known:
                y_analytical_traj.append(y_analytical.detach().cpu().numpy())
                q_analytical_traj.append(q_analytical.detach().cpu().numpy())
                Y_analytical_traj.append(Y_analytical.detach().cpu().numpy())

            if step < self.N:
                dW = torch.randn(n_sim, self.dim_W, device=self.device) * self.dt ** 0.5
                y_agent = self.forward_dynamics(y_agent, q_agent, dW, t_tensor, self.dt)
                if self.analytical_known:
                    y_analytical = self.forward_dynamics(y_analytical, q_analytical, dW, t_tensor, self.dt)
                t_scalar += self.dt

        return torch.linspace(0, self.T, self.N + 1).cpu().numpy(), {
            "y_learned": np.stack(y_agent_traj),
            "q_learned": np.stack(q_agent_traj),
            "Y_learned": np.stack(Y_agent_traj),
            "y_analytical": np.stack(y_analytical_traj) if self.analytical_known else None,
            "q_analytical": np.stack(q_analytical_traj) if self.analytical_known else None,
            "Y_analytical": np.stack(Y_analytical_traj) if self.analytical_known else None
        }
    