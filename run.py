import numpy as np
import torch
import os
from argparse import ArgumentParser
from utils.load_config import load_model_config, load_run_config
from utils.tools import str2bool
from models.hjb_bsde import HJB
from models.simple_hjb_bsde import SimpleHJB
from models.aid_bdse import AidIntradayLQ

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

    model_cfg = load_model_config(args.model_config)

    model = AidIntradayLQ(args, model_cfg)
    save_dir = f"{run_cfg['save_path']}_{args.architecture}_{args.activation}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "model")

    import warnings
    warnings.filterwarnings("ignore", message="Attempting to run cuBLAS, but there was no current CUDA context!")

    if args.load_if_exists:
        try:
            load_path = save_path + "_best" if args.best else save_path
            model.load_state_dict(torch.load(load_path + ".pth", map_location=run_cfg["device"]))
            print("Model loaded successfully.")
        except FileNotFoundError:
            print(f"No model found in {save_path}, starting training from scratch.")

    if args.train:
        model.train_model(epochs=args.epochs, K=args.K, lr=args.lr, verbose=args.verbose, plot=args.plot_loss, adaptive=args.adaptive, save_dir=save_dir)
    
    timesteps, results = model.simulate_paths(n_sim=args.n_simulations, seed=np.random.randint(0, 1000))
    model.plot_approx_vs_analytic(results, timesteps, plot=args.plot, save_dir=save_dir)
    
    timesteps, results = model.simulate_paths(n_sim=1000, seed=np.random.randint(0, 1000))
    model.plot_approx_vs_analytic_expectation(results, timesteps, plot=args.plot, save_dir=save_dir)
    model.plot_terminal_histogram(results, plot=args.plot, save_dir=save_dir)

if __name__ == "__main__":
    main()