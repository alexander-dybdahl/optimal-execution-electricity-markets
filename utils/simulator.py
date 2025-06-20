
import numpy as np
import torch

def simulate_paths(dynamics, agent, n_sim=5, seed=42, y0_single=None):
    t, dW, _ = dynamics.generate_paths(n_sim, seed=seed)
    
    t0 = t[:, 0, :]
    y0 = (
        y0_single.repeat(n_sim, 1)
        if y0_single is not None
        else dynamics.y0.repeat(n_sim, 1)
    )
    y0_agent = y0.clone()
    
    agent_predicts_Y = hasattr(agent, "predict_Y_initial") and hasattr(agent, "predict_Y_next")
    if agent_predicts_Y:
        Y0_agent = agent.predict_Y_initial(y0_agent)
    
    if dynamics.analytical_known:
        y0_analytical = y0.clone()
        Y0_analytical = dynamics.value_function_analytic(t0, y0_analytical)
        
    # Storage for trajectories
    q_agent_traj = []
    y_agent_traj = [y0_agent.detach().cpu().numpy()]
    if agent_predicts_Y:
        Y_agent_traj = [Y0_agent.detach().cpu().numpy()]
    
    if dynamics.analytical_known:
        q_analytical_traj = []
        y_analytical_traj = [y0_analytical.detach().cpu().numpy()]
        Y_analytical_traj = [Y0_analytical.detach().cpu().numpy()]
        
    for n in range(dynamics.N):
        t1 = t[:, n + 1, :]

        q_agent = agent.predict(t0, y0_agent)
        y1_agent = dynamics.forward_dynamics(y0_agent, q_agent, dW[:, n, :], t0, t1 - t0)
        if agent_predicts_Y:
            Y1_agent = agent.predict_Y_next(t0, y0_agent, t1 - t0, y1_agent - y0_agent, Y0_agent)
        
        if dynamics.analytical_known:
            q_analytical = dynamics.optimal_control_analytic(t0, y0_analytical)
            y1_analytical = dynamics.forward_dynamics(y0_analytical, q_analytical, dW[:, n, :], t0, t1 - t0)
            Y1_analytical = dynamics.value_function_analytic(t1, y1_analytical)
            
            Y_analytical_traj.append(Y1_analytical.detach().cpu().numpy())
            q_analytical_traj.append(q_analytical.detach().cpu().numpy())
            y_analytical_traj.append(y1_analytical.detach().cpu().numpy())

            y0_analytical, Y0_analytical = y1_analytical, Y1_analytical
        
        y_agent_traj.append(y1_agent.detach().cpu().numpy())
        q_agent_traj.append(q_agent.detach().cpu().numpy())
        if agent_predicts_Y:
            Y_agent_traj.append(Y1_agent.detach().cpu().numpy())

        t0, y0_agent = t1, y1_agent
        if agent_predicts_Y:
            Y0_agent = Y1_agent

    q_agent_traj = np.stack(q_agent_traj)
    y_agent_traj = np.stack(y_agent_traj)
    if agent_predicts_Y:
        Y_agent_traj = np.stack(Y_agent_traj)
    else:
        Y_agent_traj = None

    if dynamics.analytical_known:
        q_analytical_traj = np.stack(q_analytical_traj)
        y_analytical_traj = np.stack(y_analytical_traj)
        Y_analytical_traj = np.stack(Y_analytical_traj)
    else:
        q_analytical_traj = None
        y_analytical_traj = None
        Y_analytical_traj = None

    return torch.linspace(0, dynamics.T, dynamics.N + 1).cpu().numpy(), {
        "q_learned": q_agent_traj,
        "y_learned": y_agent_traj,
        "Y_learned": Y_agent_traj,
        "q_analytical": q_analytical_traj,
        "y_analytical": y_analytical_traj,
        "Y_analytical": Y_analytical_traj,
    }


def compute_cost_objective(dynamics, q_traj, y_traj):
    dt = dynamics.dt
    running_cost = 0.0
    for n in range(dynamics.N):
        y_n = y_traj[n]      # (batch_size, state_dim)
        q_n = q_traj[n]      # (batch_size, control_dim)
        f_n = dynamics.generator(y_n, q_n)  # (batch_size, 1) or (batch_size,)
        running_cost += f_n * dt                  # accumulate per batch

    terminal_cost = dynamics.terminal_cost(y_traj[-1])  # (batch_size, 1) or (batch_size,)
    cost_objective = running_cost + terminal_cost
    return cost_objective