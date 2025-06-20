import json
import os
from argparse import ArgumentParser
from utils.logger import Logger

import numpy as np
import torch

from agents.deepagent import DeepAgent, load_deepagent
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
    parser.add_argument("--plot", type=str2bool, nargs='?', const=True, default=eval_cfg["plot"], help="Plot after training")
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
    model = load_deepagent(dynamics, train_cfg, device, args.model_dir, args.best)
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
    solver = Solver(dynamics=dynamics, seed=seed, n_sim=args.n_simulations)
    solver.evaluate_agent(agent=model, agent_name="deepagent")
    solver.evaluate_agent(agent=TimeWeightedAgent(dynamics=dynamics), agent_name="timeweightedagent", analytical=False)
    logger.log(solver.costs)
    logger.log(f"Evaluation completed with seed {seed}.")
    solver.plot_traj(plot=args.plot, save_dir=save_dir)

if __name__ == "__main__":
    main()
