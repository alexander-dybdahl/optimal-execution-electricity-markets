import json
import os
from argparse import ArgumentParser
from utils.logger import Logger

import torch

from agents.analyticalagent import AnalyticalAgent
from agents.hybridagent import HybridAgent
from agents.deepagent import DeepAgent
from agents.timeweightedagent import TimeWeightedAgent
from agents.immediateagent import ImmediateAgent
from agents.startendagent import StartEndAgent
from agents.ktimesagent import KTimesAgent
from dynamics import create_dynamics
from core.solver import Solver
from utils.load_config import load_config, load_dynamics_config
from utils.tools import str2bool
from utils.plots import plot_training_losses

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
    parser.add_argument("--plot_trading_heatmap", type=str2bool, nargs='?', const=True, default=eval_cfg["plot_trading_heatmap"], help="Plot trading heatmap comparison")
    parser.add_argument("--plot_terminal_cost_analysis", type=str2bool, nargs='?', const=True, default=eval_cfg["plot_terminal_cost_analysis"], help="Plot terminal cost analysis")
    parser.add_argument("--plot_risk_metrics", type=str2bool, nargs='?', const=True, default=eval_cfg["plot_risk_metrics"], help="Plot risk metrics")
    parser.add_argument("--plot_controls", type=str2bool, nargs='?', const=True, default=eval_cfg["plot_controls"], help="Plot controls")
    parser.add_argument("--plot_losses", type=str2bool, nargs='?', const=True, default=eval_cfg.get("plot_losses", False), help="Plot training losses from CSV file")
    parser.add_argument("--plot_learned_vs_analytical", type=str2bool, nargs='?', const=True, default=eval_cfg.get("plot_learned_vs_analytical", False), help="Plot learned vs analytical Q and Y comparison across all dimensions")
    parser.add_argument("--comparison", type=str2bool, nargs='?', const=True, default=eval_cfg["comparison"], help="Generate comparison report")
    parser.add_argument("--n_simulations", type=int, default=eval_cfg["n_simulations"], help="Number of simulations to run")
    parser.add_argument("--seed", type=int, default=eval_cfg["seed"], help="Seed to use for evaluation")
    parser.add_argument("--max_batch_size", type=int, default=eval_cfg.get("max_batch_size", None), help="Maximum batch size for simulation (if None, uses training batch_size_per_rank)")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Use model_dir if save_dir is None or empty
    save_dir = args.save_dir if args.save_dir else args.model_dir
    save_dir = f"{save_dir}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir + "/imgs", exist_ok=True)

    logger = Logger(save_dir=save_dir, is_main=True, verbose=args.verbose, filename="evaluation.log", overwrite=False)

    if torch.cuda.is_available() and args.device != "cuda":
        logger.log("Warning: CUDA is available but the config file does not set device to cuda.") 
    
    try:
        # Check if model directory exists
        if not os.path.exists(args.model_dir):
            raise FileNotFoundError(f"Model directory does not exist: {args.model_dir}")
        
        # Determine which dynamics config to use
        dynamics_path = args.dynamics_path
        if args.use_model_dynamics:
            model_dynamics_path = os.path.join(args.model_dir, "dynamics_config.json")
            if os.path.exists(model_dynamics_path):
                dynamics_path = model_dynamics_path
                logger.log(f"Using dynamics config from model folder: {model_dynamics_path}")
            else:
                logger.log(f"Model dynamics config not found at {model_dynamics_path}, using default: {dynamics_path}")
        
        # Check if dynamics config file exists
        if not os.path.exists(dynamics_path):
            raise FileNotFoundError(f"Dynamics config file does not exist: {dynamics_path}")
        
        dynamics_cfg = load_dynamics_config(dynamics_path)
        
        # Log the dynamics configuration in a nice format
        logger.log("-" * 60)
        logger.log("DYNAMICS CONFIGURATION:")
        logger.log("-" * 60)
        for key, value in dynamics_cfg.items():
            if isinstance(value, dict):
                logger.log(f"{key}:")
                for subkey, subvalue in value.items():
                    logger.log(f"  {subkey}: {subvalue}")
            elif isinstance(value, list):
                logger.log(f"{key}: {value}")
            else:
                logger.log(f"{key}: {value}")
        logger.log("-" * 60)
        
    except Exception as e:
        logger.log(f"Error loading dynamics config: {e}")
        raise
    
    try:
        dynamics = create_dynamics(dynamics_cfg=dynamics_cfg, device=device)
    except Exception as e:
        logger.log(f"Error creating dynamics: {e}")
        raise
    
    try:
        # Load the model configuration
        train_cfg_path = os.path.join(args.model_dir, "train_config.json")
        if not os.path.exists(train_cfg_path):
            raise FileNotFoundError(f"Training config file does not exist: {train_cfg_path}")
        
        train_cfg = load_config(train_cfg_path)
        logger.log(f"Loaded training config from: {train_cfg_path}")
        
    except Exception as e:
        logger.log(f"Error loading training config: {e}")
        raise

    # import warnings
    # warnings.filterwarnings("ignore", message="Attempting to run cuBLAS, but there was no current CUDA context!")

    try:
        # Save configuration files
        eval_cfg_path = os.path.join(save_dir, "eval_config.json")
        dynamics_cfg_path = os.path.join(save_dir, "dynamics_config.json")
        eval_cfg = vars(args).copy()
        with open(eval_cfg_path, 'w') as f:
            json.dump(eval_cfg, f, indent=4)
        with open(dynamics_cfg_path, 'w') as f:
            json.dump(dynamics_cfg, f, indent=4)
        logger.log(f"Saved configuration files to: {save_dir}")
        
    except Exception as e:
        logger.log(f"Error saving configuration files: {e}")
        raise

    try:
        # Load the trained model
        logger.log(f"Loading trained model from: {args.model_dir}")
        logger.log(f"Using {'best' if args.best else 'latest'} model checkpoint")
        
        model = DeepAgent.load_from_checkpoint(
            dynamics=dynamics, 
            model_cfg=train_cfg, 
            device=device, 
            model_dir=args.model_dir, 
            best=args.best
        )
        logger.log("Successfully loaded trained model")
        
        # Plot training losses if requested
        if args.plot_losses:
            logger.log("Plotting training losses from CSV file...")
            plot_training_losses(args.model_dir, save_dir, plot=args.plot)
            logger.log("Training losses plot completed")
        
    except Exception as e:
        logger.log(f"Error loading trained model: {e}")
        logger.log(f"Make sure the model directory contains the required checkpoint files:")
        logger.log(f"  - {'model_best.pth' if args.best else 'model.pth'}")
        logger.log(f"  - train_config.json")
        raise

    try:
        # Determine max_batch_size - use training batch_size_per_rank if not specified
        max_batch_size = args.max_batch_size
        if max_batch_size is None:
            max_batch_size = train_cfg["batch_size_per_rank"]
            logger.log(f"Using training batch_size_per_rank as max_batch_size: {max_batch_size}")
        else:
            logger.log(f"Using specified max_batch_size: {max_batch_size}")

        # Evaluate
        solver = Solver(dynamics=dynamics, seed=args.seed, n_sim=args.n_simulations, max_batch_size=max_batch_size)
        logger.log(f"Starting evaluation with {args.n_simulations} simulations and seed {args.seed}")
        
        solver.evaluate_agent(agent=AnalyticalAgent(dynamics=dynamics), agent_name="Analytical")
        # solver.evaluate_agent(agent=HybridAgent(dynamics=dynamics), agent_name="Hybrid")
        solver.evaluate_agent(agent=model, agent_name="NN")
        solver.evaluate_agent(agent=TimeWeightedAgent(dynamics=dynamics), agent_name="TWAP")
        # solver.evaluate_agent(agent=ImmediateAgent(dynamics=dynamics), agent_name="Immediate")
        # solver.evaluate_agent(agent=StartEndAgent(dynamics=dynamics), agent_name="StartEnd")
        # solver.evaluate_agent(agent=KTimesAgent(dynamics=dynamics, K=5), agent_name="KTimes_6")
        # solver.evaluate_agent(agent=KTimesAgent(dynamics=dynamics, K=10), agent_name="KTimes_12")
        
        logger.log(f"Evaluation completed successfully")
        
    except Exception as e:
        logger.log(f"Error during evaluation: {e}")
        raise
    
    try:
        # Display risk metrics in console
        solver.display_risk_metrics()
        logger.log("Risk metrics displayed successfully")
        
    except Exception as e:
        logger.log(f"Error displaying risk metrics: {e}")
        # Continue with plotting even if risk metrics fail
    
    try:
        # Generate plots
        logger.log("Starting plot generation...")
        
        # Always plot expectation trajectories (mean ± std)
        solver.plot_trajectories_expectation(plot=args.plot, save_dir=save_dir)
        logger.log("Generated expectation trajectories plot")
        
        # Plot individual trajectories if requested
        if args.plot_individual_trajectories:
            solver.plot_trajectories_individual(n_traj=args.n_traj, plot=args.plot, save_dir=save_dir)
            logger.log(f"Generated individual trajectories plot ({args.n_traj} trajectories)")
        
        solver.plot_cost_histograms(plot=args.plot, save_dir=save_dir)
        logger.log("Generated cost histograms")
        
        if args.plot_trading_comparison:
            solver.plot_detailed_trading_trajectories(plot=args.plot, save_dir=save_dir)
            logger.log("Generated trading comparison plots")

        if args.plot_terminal_cost_analysis:
            solver.plot_terminal_cost_analysis(plot=args.plot, save_dir=save_dir)
            logger.log("Generated terminal cost analysis")
        
        if args.plot_trading_heatmap:
            solver.plot_trading_heatmap(plot=args.plot, save_dir=save_dir)
            logger.log("Generated terminal heatmap plot")
            
        if args.plot_risk_metrics:
            solver.plot_risk_metrics(plot=args.plot, save_dir=save_dir)
            solver.plot_risk_comparison_radar(plot=args.plot, save_dir=save_dir)
            logger.log("Generated risk metrics plots")
            
        if args.plot_controls:
            solver.plot_control_histograms(plot=args.plot, save_dir=save_dir)
            logger.log("Generated control histograms")
            
        if args.plot_learned_vs_analytical:
            solver.plot_learned_vs_analytical_comparison(plot=args.plot, save_dir=save_dir)
            logger.log("Generated learned vs analytical Q and Y comparison")
            
        if args.comparison:
            solver.generate_comparison_report(save_dir=save_dir)
            logger.log("Generated comparison report")
        
        logger.log(f"All plots and reports saved to: {save_dir}")
        
    except Exception as e:
        logger.log(f"Error during plot generation: {e}")
        raise

if __name__ == "__main__":
    main()
