
import numpy as np
import torch
import logging


def simulate_paths(dynamics, agent, n_sim, seed=None, y0_single=None, analytical=False):
    t, dW, _ = dynamics.generate_paths(n_sim, seed=seed)
    
    t0 = t[0 , :, :]
    y0 = (
        y0_single.repeat(n_sim, 1)
        if y0_single is not None
        else dynamics.y0.repeat(n_sim, 1)
    )
    y0_agent = y0.clone()
    
    agent_predicts_Y_method1 = hasattr(agent, "predict_Y")
    agent_predicts_Y_method2 = hasattr(agent, "predict_Y_initial") and hasattr(agent, "predict_Y_next")
    if agent_predicts_Y_method1:
        Y0_agent = agent.predict_Y(t0, y0_agent)
    elif agent_predicts_Y_method2:
        Y0_agent = agent.predict_Y_initial(y0_agent)

    if dynamics.analytical_known and analytical:
        y0_analytical = y0.clone()
        Y0_analytical = dynamics.value_function_analytic(t0, y0_analytical)
        
    # Storage for trajectories
    q_agent_traj = []
    y_agent_traj = [y0_agent.detach().cpu().numpy()]
    if agent_predicts_Y_method1 or agent_predicts_Y_method2:
        Y_agent_traj = [Y0_agent.detach().cpu().numpy()]
    
    if dynamics.analytical_known and analytical:
        q_analytical_traj = []
        y_analytical_traj = [y0_analytical.detach().cpu().numpy()]
        Y_analytical_traj = [Y0_analytical.detach().cpu().numpy()]
        
    for n in range(dynamics.N):
        t1 = t[n + 1, :, :]

        q_agent = agent.predict(t0, y0_agent)
        y1_agent = dynamics.forward_dynamics(y0_agent, q_agent, dW[n, :, :], t0, t1 - t0)
        if agent_predicts_Y_method1:
            Y1_agent = agent.predict_Y(t1, y1_agent)
        elif agent_predicts_Y_method2:
            Y1_agent = agent.predict_Y_next(t0, y0_agent, t1 - t0, y1_agent - y0_agent, Y0_agent)
        
        if dynamics.analytical_known and analytical:
            q_analytical = dynamics.optimal_control_analytic(t0, y0_analytical)
            y1_analytical = dynamics.forward_dynamics(y0_analytical, q_analytical, dW[n, :, :], t0, t1 - t0)
            Y1_analytical = dynamics.value_function_analytic(t1, y1_analytical)
            
            Y_analytical_traj.append(Y1_analytical.detach().cpu().numpy())
            q_analytical_traj.append(q_analytical.detach().cpu().numpy())
            y_analytical_traj.append(y1_analytical.detach().cpu().numpy())

            y0_analytical, Y0_analytical = y1_analytical, Y1_analytical
        
        y_agent_traj.append(y1_agent.detach().cpu().numpy())
        q_agent_traj.append(q_agent.detach().cpu().numpy())
        if agent_predicts_Y_method2 or agent_predicts_Y_method1:
            Y_agent_traj.append(Y1_agent.detach().cpu().numpy())

        t0, y0_agent = t1, y1_agent
        if agent_predicts_Y_method2:
            Y0_agent = Y1_agent

    q_agent_traj = np.stack(q_agent_traj)
    y_agent_traj = np.stack(y_agent_traj)
    if agent_predicts_Y_method2 or agent_predicts_Y_method1:
        Y_agent_traj = np.stack(Y_agent_traj)
    else:
        Y_agent_traj = None

    if dynamics.analytical_known and analytical:
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


def compute_cost_objective(dynamics, q_traj, y_traj, terminal_cost=True):
    dt = dynamics.dt
    running_cost = 0.0
    for n in range(dynamics.N):
        y_n = y_traj[n]      # (batch_size, state_dim)
        q_n = q_traj[n]      # (batch_size, control_dim)
        f_n = dynamics.generator(y_n, q_n)  # (batch_size, 1) or (batch_size,)
        running_cost += f_n * dt                  # accumulate per batch

    if not terminal_cost:
        return running_cost
    
    terminal_cost = dynamics.terminal_cost(y_traj[-1])  # (batch_size, 1) or (batch_size,)
    return running_cost + terminal_cost

def simulate_paths_batched(dynamics, agent, n_sim, max_batch_size, seed=None, cost_objective=False, analytical=False):
    """
    Simulate paths in batches to handle large numbers of simulations.
    Uses seed + batch_idx pattern to replicate distributed training behavior.
    Memory-efficient: processes costs per batch but stores all trajectories.
    
    Args:
        dynamics: Dynamics object
        agent: Agent to simulate
        n_sim: Total number of simulations
        max_batch_size: Maximum batch size for memory management
        seed: Base seed for reproducibility
        cost_objective: Whether to compute cost objective (True/False)
        analytical: Whether to compute analytical solutions
    
    Returns:
        timesteps: Time steps array
        final_results: Dictionary of trajectory results
        all_costs_concat: Concatenated costs (if cost_objective=True, else None)
    """
    n_batches = (n_sim + max_batch_size - 1) // max_batch_size
    all_results = {}
    all_costs = [] if cost_objective else None
    timesteps = None
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * max_batch_size
        end_idx = min((batch_idx + 1) * max_batch_size, n_sim)
        batch_size = end_idx - start_idx
        
        # Use seed + batch_idx to replicate distributed training pattern
        # This matches the behavior where rank i uses seed + i
        batch_seed = seed + batch_idx if seed is not None else None
        
        logging.info(f"  Processing batch {batch_idx + 1}/{n_batches} (sims {start_idx}-{end_idx-1}) with seed {batch_seed}")
        
        # Simulate this batch
        batch_timesteps, batch_results = simulate_paths(
            dynamics=dynamics,
            agent=agent,
            n_sim=batch_size,
            seed=batch_seed,
            analytical=analytical
        )
        
        if timesteps is None:
            timesteps = batch_timesteps
        
        # Compute cost objective for this batch if requested
        if cost_objective:
            batch_cost_objective = compute_cost_objective(
                dynamics=dynamics,
                q_traj=torch.from_numpy(batch_results["q_learned"]).to(dynamics.device),
                y_traj=torch.from_numpy(batch_results["y_learned"]).to(dynamics.device),
                terminal_cost=True
            )
            
            # Store costs as numpy array
            batch_costs_numpy = batch_cost_objective.detach().cpu().numpy()
            all_costs.append(batch_costs_numpy)
            
            # Clear batch tensor to free GPU memory
            del batch_cost_objective
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Store trajectories - concatenate numpy arrays along simulation dimension (axis 1)
        for key, value in batch_results.items():
            if value is not None:
                if key not in all_results:
                    all_results[key] = []
                all_results[key].append(value)
    
    # Concatenate all batches for trajectories
    final_results = {}
    for key, value_list in all_results.items():
        if value_list and value_list[0] is not None:
            # Concatenate along simulation axis (axis 1)
            final_results[key] = np.concatenate(value_list, axis=1)
        else:
            final_results[key] = None
    
    # Concatenate all costs if computed
    all_costs_concat = np.concatenate(all_costs) if cost_objective else None
    
    return timesteps, final_results, all_costs_concat