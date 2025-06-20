import json
import os
from argparse import ArgumentParser
from utils.logger import Logger

import numpy as np
import torch

from agents.deepagent import DeepAgent
from agents.timeweightedagent import TimeWeightedAgent
from dynamics.aid_dynamics import AidDynamics
from dynamics.hjb_dynamics import HJBDynamics
from dynamics.simple_dynamics import SimpleDynamics
from core.solver import Solver
from utils.load_config import load_config, load_dynamics_config
from utils.plots import plot_approx_vs_analytic, plot_approx_vs_analytic_expectation, plot_terminal_histogram 
from utils.simulator import simulate_paths, compute_cost_objective
from utils.tools import str2bool


def main():
    eval_cfg = load_config(path="config/eval_config.json")
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default=eval_cfg["device"], help="Device to use for training (cpu or cuda)")
    parser.add_argument("--save_dir", type=str, default=eval_cfg["save_dir"], help="Path to save the evaluation results")
    parser.add_argument("--model_dir", type=str, default=eval_cfg["model_dir"], help="Path to the saved model")
    parser.add_argument("--dynamics_path", type=str, default=eval_cfg["dynamics_path"], help="Path to the dynamics configuration file")
    parser.add_argument("--best", type=str2bool, nargs='?', const=True, default=eval_cfg["best"], help="Load the model using the best model found during training")
    parser.add_argument("--verbose", type=str2bool, nargs='?', const=True, default=eval_cfg["verbose"], help="Print training progress")
    parser.add_argument("--n_simulations", type=int, default=eval_cfg["n_simulations"], help="Number of simulations to run")
    args = parser.parse_args()

    device = torch.device(args.device)

    save_dir = f"{args.save_dir}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir + "/imgs", exist_ok=True)

    logger = Logger(save_dir=save_dir, is_main=True, verbose=args.verbose, filename="evaluation.log", overwrite=True)

    if torch.cuda.is_available() and args.device != "cuda":
        logger.log("Warning: CUDA is available but the config file does not set device to cuda.") 
    
    dynamics_cfg = load_dynamics_config(args.dynamics_path)
    dynamics = AidDynamics(dynamics_cfg=dynamics_cfg, device=device)
    
    # Load the model
    train_cfg_path = os.path.join(args.model_dir, "train_config.json")
    train_cfg = load_config(train_cfg_path)
    model = DeepAgent(dynamics=dynamics, model_cfg=train_cfg, device=device)
    model.to(device)
    
    model_path = os.path.join(args.model_dir, "model")
    load_path = model_path + "_best.pth" if args.best else model_path + ".pth"
    model.load_state_dict(torch.load(load_path, map_location=device))
    logger.log("Model loaded successfully.")

    # import warnings
    # warnings.filterwarnings("ignore", message="Attempting to run cuBLAS, but there was no current CUDA context!")

    # Save configuration files
    eval_cfg_path = os.path.join(save_dir, "eval_config.json")
    dynamics_cfg_path = os.path.join(save_dir, "dynamics_config.json")
    eval_cfg = vars(args).copy()
    with open(eval_cfg_path, 'w') as f:
        json.dump(eval_cfg, f, indent=4)
    with open(dynamics_cfg_path, 'w') as f:
        json.dump(dynamics_cfg, f, indent=4)

    # Evaluate
    seed = np.random.randint(0, 1000)
    timesteps, results = simulate_paths(
        dynamics=dynamics,
        agent=model,
        n_sim=args.n_simulations,
        seed=seed
    )
    cost_objective_deepagent = compute_cost_objective(
        dynamics=dynamics,
        q_traj=results["q_learned"], 
        y_traj=results["y_learned"]
    )
    logger.log(f"Cost objective of DeepAgent: {cost_objective_deepagent.mean().item():.4f}")
    
    timesteps, results = simulate_paths(
        dynamics=dynamics,
        agent=TimeWeightedAgent(dynamics=dynamics),
        n_sim=args.n_simulations,
        seed=seed
    )
    cost_objective_timeweigthedagent = compute_cost_objective(
        dynamics=dynamics,
        q_traj=results["q_learned"], 
        y_traj=results["y_learned"]
    )
    logger.log(f"Cost objective of TimeWeightedAgent: {cost_objective_timeweigthedagent.mean().item():.4f}")

    # plot_approx_vs_analytic(results=results, timesteps=timesteps, validation=validation, plot=args.plot, save_dir=save_dir)
    # plot_approx_vs_analytic_expectation(results=results, timesteps=timesteps, plot=args.plot, save_dir=save_dir)
    # plot_terminal_histogram(results=results, dynamics=dynamics, plot=args.plot, save_dir=save_dir)

if __name__ == "__main__":
    main()
