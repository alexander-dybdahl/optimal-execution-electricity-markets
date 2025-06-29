import json
import os
from argparse import ArgumentParser
from utils.logger import Logger

import numpy as np
import torch

from agents.analyticalagent import AnalyticalAgent
from agents.deepagent import DeepAgent
from agents.immediateagent import ImmediateAgent
from agents.timeweightedagent import TimeWeightedAgent
from dynamics import create_dynamics
from core.solver import Solver
from utils.load_config import load_config, load_dynamics_config
from utils.tools import str2bool


def main():
    eval_cfg = load_config(path="config/eval_config.json")
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default=eval_cfg["device"], help="Device to use for training (cpu or cuda)")
    parser.add_argument("--save_dir", type=str, default=eval_cfg["save_dir"], help="Path to save the evaluation results")
    parser.add_argument("--model_dir", type=str, default=eval_cfg["model_dir"], help="Path to the saved model")
    parser.add_argument("--dynamics_path", type=str, default=eval_cfg["dynamics_path"], help="Path to the dynamics configuration file")
    parser.add_argument("--use_model_dynamics", type=str2bool, nargs='?', const=True, default=eval_cfg["use_model_dynamics"], help="Use dynamics config from model folder if it exists")
    parser.add_argument("--best", type=str2bool, nargs='?', const=True, default=eval_cfg["best"], help="Load the model using the best model found during training")
    parser.add_argument("--verbose", type=str2bool, nargs='?', const=True, default=eval_cfg["verbose"], help="Print training progress")
    parser.add_argument("--plot", type=str2bool, nargs='?', const=True, default=eval_cfg["plot"], help="Plot after training")
    parser.add_argument("--plot_individual_trajectories", type=str2bool, nargs='?', const=True, default=eval_cfg.get("plot_individual_trajectories", False), help="Plot individual trajectories")
    parser.add_argument("--n_traj", type=int, default=eval_cfg.get("n_traj", 5), help="Number of individual trajectories to plot")
    parser.add_argument("--plot_control_trajectories", type=str2bool, nargs='?', const=True, default=eval_cfg["plot_control_trajectories"], help="Plot control trajectories")
    parser.add_argument("--plot_trading_comparison", type=str2bool, nargs='?', const=True, default=eval_cfg["plot_trading_comparison"], help="Plot trading comparison")
    parser.add_argument("--plot_risk_metrics", type=str2bool, nargs='?', const=True, default=eval_cfg["plot_risk_metrics"], help="Plot risk metrics")
    parser.add_argument("--plot_controls", type=str2bool, nargs='?', const=True, default=eval_cfg["plot_controls"], help="Plot controls")
    parser.add_argument("--n_simulations", type=int, default=eval_cfg["n_simulations"], help="Number of simulations to run")
    parser.add_argument("--seed", type=int, default=eval_cfg["seed"], help="Seed to use for evaluation")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Use model_dir if save_dir is None or empty
    save_dir = args.save_dir if args.save_dir else args.model_dir
    save_dir = f"{save_dir}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir + "/imgs", exist_ok=True)

    logger = Logger(save_dir=save_dir, is_main=True, verbose=args.verbose, filename="evaluation.log", overwrite=True)

    if torch.cuda.is_available() and args.device != "cuda":
        logger.log("Warning: CUDA is available but the config file does not set device to cuda.") 
    
    # Determine which dynamics config to use
    dynamics_path = args.dynamics_path
    if args.use_model_dynamics:
        model_dynamics_path = os.path.join(args.model_dir, "dynamics_config.json")
        if os.path.exists(model_dynamics_path):
            dynamics_path = model_dynamics_path
            logger.log(f"Using dynamics config from model folder: {model_dynamics_path}")
        else:
            logger.log(f"Model dynamics config not found at {model_dynamics_path}, using default: {dynamics_path}")
    
    dynamics_cfg = load_dynamics_config(dynamics_path)
    dynamics = create_dynamics(dynamics_cfg=dynamics_cfg, device=device)
    
    # Load the model
    train_cfg_path = os.path.join(args.model_dir, "train_config.json")
    train_cfg = load_config(train_cfg_path)

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
    solver = Solver(dynamics=dynamics, seed=args.seed, n_sim=args.n_simulations)
    logger.log("Starting evaluation.")
    solver.evaluate_agent(agent=AnalyticalAgent(dynamics=dynamics), agent_name="AnalyticalAgent")
    solver.evaluate_agent(agent=DeepAgent.load_from_checkpoint(dynamics=dynamics, model_cfg=train_cfg, device=device, model_dir=args.model_dir, best=args.best), agent_name="DeepAgent")
    solver.evaluate_agent(agent=TimeWeightedAgent(dynamics=dynamics), agent_name="TimeWeightedAgent")
    # solver.evaluate_agent(agent=ImmediateAgent(dynamics=dynamics), agent_name="ImmediateAgent")
    logger.log(f"Evaluation completed with seed {args.seed}.")
    
    # Display risk metrics in console
    solver.display_risk_metrics()
    
    # Generate plots
    # Always plot expectation trajectories (mean ± std)
    solver.plot_trajectories_expectation(plot=args.plot, save_dir=save_dir)
    
    # Plot individual trajectories if requested
    if args.plot_individual_trajectories:
        solver.plot_trajectories_individual(n_traj=args.n_traj, plot=args.plot, save_dir=save_dir)
    
    solver.plot_cost_histograms(plot=args.plot, save_dir=save_dir)
    if args.plot_trading_comparison:
        solver.plot_detailed_trading_trajectories(plot=args.plot, save_dir=save_dir)
        solver.plot_trading_heatmap(plot=args.plot, save_dir=save_dir)
        solver.plot_terminal_cost_analysis(plot=args.plot, save_dir=save_dir)
    if args.plot_risk_metrics:
        solver.plot_risk_metrics(plot=args.plot, save_dir=save_dir)
        solver.plot_risk_comparison_radar(plot=args.plot, save_dir=save_dir)
    if args.plot_controls:
        solver.plot_control_histograms(plot=args.plot, save_dir=save_dir)
    solver.generate_comparison_report(save_dir=save_dir)

if __name__ == "__main__":
    main()
