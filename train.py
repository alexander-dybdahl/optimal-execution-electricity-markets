import json
import os
from argparse import ArgumentParser
from utils.logger import Logger

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from agents.deepagent import DeepAgent
from dynamics import create_dynamics
from core.solver import Solver
from utils.load_config import load_dynamics_config, load_train_config
from utils.plots import plot_approx_vs_analytic, plot_approx_vs_analytic_expectation, plot_terminal_histogram 
from utils.simulator import simulate_paths
from utils.tools import str2bool


def main():
    train_cfg = load_train_config(path="config/train_config.json")
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default=train_cfg["device"], help="Device to use for training (cpu or cuda)")
    parser.add_argument("--parallel", type=str2bool, nargs='?', const=True, default=train_cfg["parallel"], help="Use data parallelism")
    parser.add_argument("--save_dir", type=str, default=train_cfg["save_dir"], help="Path to save the model")
    parser.add_argument("--dynamics_path", type=str, default=train_cfg["dynamics_path"], help="Path to the dynamics configuration file")
    parser.add_argument("--train", type=str2bool, nargs='?', const=True, default=train_cfg["train"], help="Train the model")
    parser.add_argument("--save", nargs="+", default=train_cfg["save"], help="Model saving strategy: choose from 'best', 'every'")
    parser.add_argument("--load_if_exists", type=str2bool, nargs='?', const=True, default=train_cfg["load_if_exists"], help="Load model if it exists")
    parser.add_argument("--resume", type=str2bool, nargs='?', const=True, default=train_cfg["resume"], help="Resume training with optimizer/scheduler state")
    parser.add_argument("--reset_lr", type=str2bool, nargs='?', const=True, default=train_cfg["reset_lr"], help="Reset the learning rate to the initial value")
    parser.add_argument("--reset_best", type=str2bool, nargs='?', const=True, default=train_cfg["reset_best"], help="Reset the best loss to initial value")
    parser.add_argument("--epochs", type=int, default=train_cfg["epochs"], help="Number of training epochs")
    parser.add_argument("--K", type=int, default=train_cfg["K"], help="Epochs between evaluations of the model")
    parser.add_argument("--batch_size", type=int, default=train_cfg["batch_size"], help="Batch size for training")
    parser.add_argument("--save_n", type=int, default=train_cfg["save_n"], help="If 'every' is selected, save every n epochs")
    parser.add_argument("--plot_n", type=int, default=train_cfg["plot_n"], help="Save plot every n epochs if plot_n is not None")
    parser.add_argument("--best", type=str2bool, nargs='?', const=True, default=train_cfg["best"], help="Run the model using the best model found during training")
    parser.add_argument("--verbose", type=str2bool, nargs='?', const=True, default=train_cfg["verbose"], help="Print training progress")
    parser.add_argument("--plot", type=str2bool, nargs='?', const=True, default=train_cfg["plot"], help="Plot after training")
    parser.add_argument("--plot_loss", type=str2bool, nargs='?', const=True, default=train_cfg["plot_loss"], help="Plot loss after training")
    parser.add_argument("--n_simulations", type=int, default=train_cfg["n_simulations"], help="Number of simulations to run")
    parser.add_argument("--rescale_y0", type=str2bool, nargs='?', const=True, default=train_cfg["rescale_y0"], help="Rescale input in y0_net forward by y0")
    parser.add_argument("--input_bn", type=str2bool, nargs='?', const=True, default=train_cfg["input_bn"], help="Use batch normalization on input")
    parser.add_argument("--affine", type=str2bool, nargs='?', const=True, default=train_cfg["affine"], help="Use affine transformation in batch normalization")
    parser.add_argument("--strong_grad_output", type=str2bool, nargs='?', const=True, default=train_cfg["strong_grad_output"], help="Use strong gradient output in initial network")
    parser.add_argument("--scale_output", type=float, default=train_cfg["scale_output"], help="How much to scale output in initial network")
    parser.add_argument("--careful_init", type=str2bool, nargs='?', const=True, default=train_cfg["careful_init"], help="Use careful initialization for the neural network weights")
    parser.add_argument("--detach_control", type=str2bool, nargs='?', const=True, default=train_cfg["detach_control"], help="Detach control from the network output")
    parser.add_argument("--network_type", type=str, default=train_cfg["network_type"], help="Type of network to use for the agent (Y or dY)")
    parser.add_argument("--supervised", type=str2bool, default=train_cfg["supervised"], help="Use supervised learning using analytical solution")
    parser.add_argument("--architecture", type=str, default=train_cfg["architecture"], help="Neural network architecture to use")
    parser.add_argument("--activation", type=str, default=train_cfg["activation"], help="Activation function to use")
    parser.add_argument("--Y0_layers", type=int, nargs="+", default=train_cfg["Y0_layers"], help="List of hidden layer sizes for the Y0 network")
    parser.add_argument("--Y_layers", type=int, nargs="+", default=train_cfg["Y_layers"], help="List of hidden layer sizes for the neural network")
    parser.add_argument("--subnet_type", type=str, default=train_cfg.get("subnet_type", "FC"), help="Type of subnet to use (FC, Resnet, or NAISnet) for SeparateSubnets and LSTMWithSubnets")
    parser.add_argument("--lstm_layers", type=int, nargs="+", default=train_cfg.get("lstm_layers", [64, 64]), help="Sizes for LSTM layers in LSTMWithSubnets")
    parser.add_argument("--lstm_type", type=str, default=train_cfg.get("lstm_type", "LSTM"), help="Type of LSTM to use in LSTMWithSubnets")
    parser.add_argument("--adaptive", type=str2bool, nargs='?', const=True, default=train_cfg["adaptive"], help="Use adaptive learning rate")
    parser.add_argument("--adaptive_factor", type=float, default=train_cfg["adaptive_factor"], help="Adaptive factor")
    parser.add_argument("--lr", type=float, default=train_cfg["lr"], help="Learning rate for the optimizer")
    parser.add_argument("--annealing", type=str2bool, nargs='?', const=True, default=train_cfg["annealing"], help="Use annealing for psi, gamma, and nu")
    parser.add_argument("--adaptive_loss", type=str2bool, nargs='?', const=True, default=train_cfg["adaptive_loss"], help="Use adaptive loss function")
    parser.add_argument("--lambda_Y0", type=float, default=train_cfg["lambda_Y0"], help="Weight for the Y0 term in the loss function")
    parser.add_argument("--lambda_Y", type=float, default=train_cfg["lambda_Y"], help="Weight for the Y term in the loss function")
    parser.add_argument("--lambda_dY", type=float, default=train_cfg["lambda_dY"], help="Weight for the dY term in the loss function")
    parser.add_argument("--lambda_dYt", type=float, default=train_cfg["lambda_dYt"], help="Weight for the dYt term in the loss function")
    parser.add_argument("--lambda_T", type=float, default=train_cfg["lambda_T"], help="Weight for the terminal term in the loss function")
    parser.add_argument("--lambda_TG", type=float, default=train_cfg["lambda_TG"], help="Weight for the terminal gradient term in the loss function")
    parser.add_argument("--lambda_pinn", type=float, default=train_cfg["lambda_pinn"], help="Weight for the PINN term in the loss function")
    parser.add_argument("--lambda_reg", type=float, default=train_cfg["lambda_reg"], help="Weight for the regularization term in the loss function")
    parser.add_argument("--lambda_cost", type=float, default=train_cfg["lambda_cost"], help="Weight for the cost term in the loss function")
    parser.add_argument("--sobol_points", type=str2bool, nargs='?', const=True, default=train_cfg["sobol_points"], help="Use Sobol points for training")
    parser.add_argument("--use_linear_approx", type=str2bool, nargs='?', const=True, default=train_cfg["use_linear_approx"], help="Use linear loss approximation beyond the threshold")
    parser.add_argument("--loss_threshold", type=float, default=train_cfg["loss_threshold"], help="Threshold to linearly approximate the loss")
    parser.add_argument("--second_order_taylor", type=str2bool, nargs='?', const=True, default=train_cfg["second_order_taylor"], help="Use second order Taylor approximation for Y reconstruction")
    args = parser.parse_args()

    if args.parallel:
        env_rank = int(os.environ.get("RANK", 0))
        env_local_rank = int(os.environ["LOCAL_RANK"])
        env_world_size = int(os.environ.get("WORLD_SIZE", 1))
        env_master_addr = os.environ.get("MASTER_ADDR", "localhost")
        env_master_port = os.environ.get("MASTER_PORT", "23456")

        if args.device == "cuda":
            torch.cuda.set_device(env_local_rank)

        backend = "nccl" if args.device == "cuda" else "gloo"
        dist.init_process_group(backend=backend,
                                world_size=env_world_size,
                                rank=env_rank)
        is_distributed = dist.is_initialized()
        is_main = env_rank == 0
        args.global_rank = env_rank
        args.batch_size_per_rank = args.batch_size // env_world_size
    else:
        is_distributed = False
        is_main = True
        args.global_rank = 0
        args.batch_size_per_rank = args.batch_size
    
    device = torch.device(args.device)

    if args.architecture.lower() == "lstmwithsubnets":
        save_dir = f"{args.save_dir}_{args.architecture.lower()}_{args.lstm_type.lower()}_{args.subnet_type.lower()}_{args.activation}"
    elif args.architecture.lower() == "separatesubnets":
        save_dir = f"{args.save_dir}_{args.architecture.lower()}_{args.subnet_type.lower()}_{args.activation}"
    else:
        save_dir = f"{args.save_dir}_{args.architecture.lower()}_{args.activation}"
    save_path = os.path.join(save_dir, "model")
    
    if is_main:
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(save_dir + "/imgs", exist_ok=True)

    # If loading an existing model and training.log exists, append to it instead of overwriting
    training_log_path = os.path.join(save_dir, "training.log")
    overwrite_log = args.train and not (args.load_if_exists and os.path.exists(training_log_path))
    
    logger = Logger(save_dir=save_dir, is_main=is_main, verbose=args.verbose, filename="training.log", overwrite=overwrite_log)

    # Log whether we're starting fresh or appending
    if args.load_if_exists and os.path.exists(training_log_path) and not overwrite_log:
        logger.log("=" * 80)
        logger.log("RESUMING TRAINING - APPENDING TO EXISTING LOG")
        logger.log("=" * 80)

    if torch.cuda.is_available() and args.device != "cuda":
        logger.log("Warning: CUDA is available but the config file does not set device to cuda.") 
    
    if is_distributed:
        logger.log(f"Distributed training setup: RANK: {args.global_rank}, WORLD_SIZE={env_world_size}, MASTER_ADDR={env_master_addr}, MASTER_PORT={env_master_port}")
        logger.log(f"Running on device: {device}, Global rank: {args.global_rank}, Distributed: {is_distributed}, Main process: {is_main}", override=True)
    else:
        logger.log(f"Running on device: {device}, Parallel training disabled")

    try:
        # Try to load dynamics config from save_dir if it exists, otherwise use args.dynamics_path
        dynamics_cfg_path = os.path.join(save_dir, "dynamics_config.json")
        if os.path.exists(dynamics_cfg_path):
            logger.log(f"Loading dynamics config from {dynamics_cfg_path}")
            dynamics_cfg = load_dynamics_config(dynamics_cfg_path)
        else:
            logger.log(f"Loading dynamics config from {args.dynamics_path}")
            dynamics_cfg = load_dynamics_config(args.dynamics_path)
        
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
        
    dynamics = create_dynamics(dynamics_cfg=dynamics_cfg, device=device)
    
    train_cfg = vars(args).copy()
    
    # Initialize variables for training
    start_epoch = 1
    optimizer_state = None
    scheduler_state = None
    
    # Create a new model
    model = DeepAgent(dynamics=dynamics, model_cfg=train_cfg, device=device)
    model.to(device)
    
    # Check if we should load an existing model
    if args.load_if_exists:
        logger.log(f"Attempting to load model from {save_dir}")
        try:
            # Use the instance method to load the checkpoint
            if args.resume:
                # If resuming, get optimizer and scheduler state too
                optimizer_state, scheduler_state, start_epoch = model.load_checkpoint(
                    model_dir=save_dir,
                    best=args.best
                )
                logger.log(f"Loaded model checkpoint from epoch {start_epoch-1} with training state")
            else:
                # Otherwise just load the model state
                optimizer_state, _, _ = model.load_checkpoint(
                    model_dir=save_dir,
                    best=args.best
                )
                # Reset start_epoch to 1 since we're not resuming training
                start_epoch = 1
                logger.log(f"Loaded model checkpoint (model state only)")
        except Exception as e:
            logger.log(f"No valid model found in {save_dir}, starting from scratch.")
    else:
        logger.log("Not loading any model, starting from scratch.")

    import warnings
    warnings.filterwarnings("ignore", message="Attempting to run cuBLAS, but there was no current CUDA context!")

    # Save configuration files and train
    if args.train:
        if is_main:
            train_cfg_path = os.path.join(save_dir, "train_config.json")
            dynamics_cfg_path = os.path.join(save_dir, "dynamics_config.json")
            with open(train_cfg_path, 'w') as f:
                json.dump(train_cfg, f, indent=4)
            with open(dynamics_cfg_path, 'w') as f:
                json.dump(dynamics_cfg, f, indent=4)
        
        # Wrap in DDP if applicable
        if args.parallel:
            if is_distributed:
                logger.log("Applying DDP for parallel training.")
                model = DDP(model, device_ids=[env_local_rank] if args.device == "cuda" else None)
            else:
                logger.log("Warning: Parallel training is enabled but not running in a distributed environment. DDP will not be applied.")
        
        call_model = model.module if isinstance(model, DDP) else model
        call_model.train_model(
            epochs=args.epochs, 
            K=args.K, 
            lr=args.lr, 
            verbose=args.verbose, 
            plot=args.plot_loss, 
            adaptive=args.adaptive, 
            save_dir=save_dir, 
            logger=logger,
            start_epoch=start_epoch,
            optimizer_state=optimizer_state,
            scheduler_state=scheduler_state
        )

    # Evaluate and plot only on main
    if is_main:
        call_model = model.module if isinstance(model, DDP) else model
        validation = call_model.validation
        timesteps, results = simulate_paths(dynamics=dynamics, agent=call_model, n_sim=args.n_simulations, seed=np.random.randint(0, 1000))
        plot_approx_vs_analytic(results=results, timesteps=timesteps, validation=validation, plot=args.plot, save_dir=save_dir)
        plot_approx_vs_analytic_expectation(results=results, timesteps=timesteps, plot=args.plot, save_dir=save_dir)
        plot_terminal_histogram(results=results, dynamics=dynamics, plot=args.plot, save_dir=save_dir)

    # Sync & cleanup
    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
