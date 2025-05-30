import numpy as np
import torch
import torch.distributed as dist
import os
from argparse import ArgumentParser
from utils.load_config import load_model_config, load_run_config
from utils.tools import str2bool
from models.hjb_bsde import HJB
from models.simple_hjb_bsde import SimpleHJB
from models.aid_bdse import AidIntradayLQ
from torch.nn.parallel import DistributedDataParallel as DDP
import multiprocessing as mp


def main():
    run_cfg = load_run_config(path="config/run_config.json")
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=run_cfg["epochs"], help="Number of training epochs")
    parser.add_argument("--K", type=int, default=run_cfg["K"], help="Epochs between evaluations of the model")
    parser.add_argument("--lr", type=float, default=run_cfg["lr"], help="Learning rate for the optimizer")
    parser.add_argument("--save_path", type=str, default=run_cfg["save_path"], help="Path to save the model")
    parser.add_argument("--n_paths", type=int, default=run_cfg["n_paths"], help="Number of paths to simulate")
    parser.add_argument("--batch_size", type=int, default=run_cfg["batch_size"], help="Batch size for training")
    parser.add_argument("--n_simulations", type=int, default=run_cfg["n_simulations"], help="Number of simulations to run")
    parser.add_argument("--device", type=str, default=run_cfg["device"], help="Device to use for training (cpu or cuda)")
    parser.add_argument("--model_config", type=str, default=run_cfg["config_path"], help="Path to the model configuration file")
    parser.add_argument("--architecture", type=str, default=run_cfg["architecture"], help="Neural network architecture to use")
    parser.add_argument("--activation", type=str, default=run_cfg["activation"], help="Activation function to use")
    parser.add_argument("--adaptive", type=str2bool, nargs='?', const=True, default=run_cfg["adaptive"], help="Use adaptive learning rate")
    parser.add_argument("--parallel", type=str2bool, nargs='?', const=True, default=run_cfg["parallel"], help="Use data parallelism")
    parser.add_argument("--verbose", type=str2bool, nargs='?', const=True, default=run_cfg["verbose"], help="Print training progress")
    parser.add_argument("--plot", type=str2bool, nargs='?', const=True, default=run_cfg["plot"], help="Plot after training")
    parser.add_argument("--plot_loss", type=str2bool, nargs='?', const=True, default=run_cfg["plot_loss"], help="Plot loss after training")
    parser.add_argument("--save", nargs="+", default=run_cfg["save"], help="Model saving strategy: choose from 'best', 'every'")
    parser.add_argument("--save_n", type=int, default=run_cfg["save_n"], help="If 'every' is selected, save every n epochs")
    parser.add_argument("--load_if_exists", type=str2bool, nargs='?', const=True, default=run_cfg["load_if_exists"], help="Load model if it exists")
    parser.add_argument("--train", type=str2bool, nargs='?', const=True, default=run_cfg["train"], help="Train the model")
    parser.add_argument("--best", type=str2bool, nargs='?', const=True, default=run_cfg["best"], help="Run the model using the best model found during training")
    parser.add_argument("--lambda_Y", type=float, default=run_cfg["lambda_Y"], help="Weight for the Y term in the loss function")
    parser.add_argument("--lambda_T", type=float, default=run_cfg["lambda_T"], help="Weight for the terminal term in the loss function")
    parser.add_argument("--lambda_TG", type=float, default=run_cfg["lambda_TG"], help="Weight for the terminal gradient term in the loss function")
    parser.add_argument("--supervised", type=str2bool, default=run_cfg["supervised"], help="Use supervised learning using analytical solution")
    args = parser.parse_args()
    args.Y_layers = run_cfg["Y_layers"]

    if args.parallel:
        env_rank = os.environ.get("RANK", 0)
        env_world_size = os.environ.get("WORLD_SIZE", 1)
        env_master_addr = os.environ.get("MASTER_ADDR", "localhost")
        env_master_port = os.environ.get("MASTER_PORT", "23456")

        local_rank = int(env_rank)
        backend = "nccl" if args.device == "cuda" else "gloo"
        dist.init_process_group(backend=backend,
                                world_size=int(env_world_size),
                                rank=local_rank,
                                init_method=f"tcp://{env_master_addr}:{env_master_port}?use_libuv=0")
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        is_distributed = True
        is_main = dist.get_rank() == 0
    else:
        local_rank = 0
        device = torch.device(args.device)
        is_distributed = False
        is_main = True
    
    if is_main:
        print(f"Distributed training setup: RANK={env_rank}, WORLD_SIZE={env_world_size}, MASTER_ADDR={env_master_addr}, MASTER_PORT={env_master_port}")
        print(f"Running on device: {device}, Local rank: {local_rank}, Distributed: {is_distributed}, Main process: {is_main}")

    model_cfg = load_model_config(args.model_config)
    model = AidIntradayLQ(args, model_cfg).to(device)

    save_dir = f"{args.save_path}_{args.architecture}_{args.activation}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "model")

    if args.load_if_exists:
        load_path = save_path + "_best.pth" if args.best else save_path + ".pth"
        try:
            state_dict = torch.load(load_path, map_location=device)
            model.load_state_dict(state_dict)
            if is_main:
                print("Model loaded successfully.")
        except FileNotFoundError:
            if is_main:
                print(f"No model found in {load_path}, starting from scratch.")
    else:
        if is_main:
            print("Not loading any model, starting from scratch.")

    import warnings
    warnings.filterwarnings("ignore", message="Attempting to run cuBLAS, but there was no current CUDA context!")
    
    # Wrap in DDP if applicable
    if args.parallel and is_distributed:
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)

    # Train
    if args.train:
        call_model = model.module if isinstance(model, DDP) else model
        call_model.train_model(epochs=args.epochs, K=args.K, lr=args.lr, verbose=args.verbose, plot=args.plot_loss, adaptive=args.adaptive, save_dir=save_dir if is_main else None)

    # Evaluate and plot only on main
    if is_main:
        call_model = model.module if isinstance(model, DDP) else model
        timesteps, results = call_model.simulate_paths(n_sim=args.n_simulations, seed=np.random.randint(0, 1000))
        call_model.plot_approx_vs_analytic(results, timesteps, plot=args.plot, save_dir=save_dir)

        timesteps, results = call_model.simulate_paths(n_sim=1000, seed=np.random.randint(0, 1000))
        call_model.plot_approx_vs_analytic_expectation(results, timesteps, plot=args.plot, save_dir=save_dir)
        call_model.plot_terminal_histogram(results, plot=args.plot, save_dir=save_dir)

    # Sync & cleanup
    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
