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

    def simulate_paths(self, agent, n_sim=5, seed=42, y0_single=None):
        t, dW, _ = self.generate_paths(n_sim, seed=seed)
        
        t0 = t[:, 0, :]
        y0 = (
            y0_single.repeat(n_sim, 1)
            if y0_single is not None
            else self.y0.repeat(n_sim, 1)
        )
        y0_agent = y0.clone()
        Y0_agent = agent.predict_Y_initial(y0_agent)
        
        if self.analytical_known:
            y0_analytical = y0.clone()
            Y0_analytical = self.value_function_analytic(t0, y0_analytical)
            
        # Storage for trajectories
        Y_agent_traj = [Y0_agent.detach().cpu().numpy()]
        y_agent_traj = [y0_agent.detach().cpu().numpy()]
        q_agent_traj = []
        if self.analytical_known:
            Y_analytical_traj = [Y0_analytical.detach().cpu().numpy()]
            y_analytical_traj = [y0_analytical.detach().cpu().numpy()]
            q_analytical_traj = []
            
        for n in range(self.N):
            t1 = t[:, n + 1, :]

            q_agent = agent.predict(t0, y0_agent)
            y1_agent = self.forward_dynamics(y0_agent, q_agent, dW[:, n, :], t0, t1 - t0)
            Y1_agent = agent.predict_Y_next(t0, y0_agent, t1 - t0, y1_agent - y0_agent, Y0_agent)
            
            if self.analytical_known:
                q_analytical = self.optimal_control_analytic(t0, y0_analytical)
                y1_analytical = self.forward_dynamics(y0_analytical, q_analytical, dW[:, n, :], t0, t1 - t0)
                Y1_analytical = self.value_function_analytic(t1, y1_analytical)

                y0_analytical, Y0_analytical = y1_analytical, Y1_analytical
                
                Y_analytical_traj.append(Y0_analytical.detach().cpu().numpy())
                y_analytical_traj.append(y0_analytical.detach().cpu().numpy())
                q_analytical_traj.append(q_analytical.detach().cpu().numpy())

            t0, y0_agent, Y0_agent = t1, y1_agent, Y1_agent            
            
            Y_agent_traj.append(Y0_agent.detach().cpu().numpy())
            y_agent_traj.append(y0_agent.detach().cpu().numpy())
            q_agent_traj.append(q_agent.detach().cpu().numpy())

        Y_agent_traj = np.stack(Y_agent_traj)
        y_agent_traj = np.stack(y_agent_traj)
        q_agent_traj = np.stack(q_agent_traj)

        if self.analytical_known:
            Y_analytical_traj = np.stack(Y_analytical_traj)
            y_analytical_traj = np.stack(y_analytical_traj)
            q_analytical_traj = np.stack(q_analytical_traj)
        else:
            Y_analytical_traj = None
            y_analytical_traj = None
            q_analytical_traj = None

        return torch.linspace(0, self.T, self.N + 1).cpu().numpy(), {
            "y_learned": y_agent_traj,
            "q_learned": q_agent_traj,
            "Y_learned": Y_agent_traj,
            "y_analytical": y_analytical_traj,
            "q_analytical": q_analytical_traj,
            "Y_analytical": Y_analytical_traj,
        }
