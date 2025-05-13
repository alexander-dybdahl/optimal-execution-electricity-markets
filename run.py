import numpy as np
import torch
from argparse import ArgumentParser
from utils.load_config import load_model_config, load_run_config
from utils.plots import plot_all_diagnostics
from utils.tools import str2bool
from models.hjb_bsde import HJB
from models.simple_hjb_bsde import SimpleHJB

def main():
    run_cfg = load_run_config(path="config/run_config.json")
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=run_cfg["epochs"], help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=run_cfg["lr"], help="Learning rate for the optimizer")
    parser.add_argument("--save_path", type=str, default=run_cfg["save_path"], help="Path to save the model")
    parser.add_argument("--n_paths", type=int, default=run_cfg["n_paths"], help="Number of paths to simulate")
    parser.add_argument("--batch_size", type=int, default=run_cfg["batch_size"], help="Batch size for training")
    parser.add_argument("--n_simulations", type=int, default=run_cfg["n_simulations"], help="Number of simulations to run")
    parser.add_argument("--sim_batch_size", type=int, default=run_cfg["sim_batch_size"], help="Batch size for simulation")
    parser.add_argument("--device", type=str, default=run_cfg["device"], help="Device to use for training (cpu or cuda)")
    parser.add_argument("--model_config", type=str, default=run_cfg["config_path"], help="Path to the model configuration file")
    parser.add_argument("--architecture", type=str, default=run_cfg["architecture"], help="Neural network architecture to use")
    parser.add_argument("--activation", type=str, default=run_cfg["activation"], help="Activation function to use")
    parser.add_argument("--verbose", type=str2bool, nargs='?', const=True, default=run_cfg["verbose"], help="Print training progress")    
    parser.add_argument("--save", nargs="+", default=run_cfg["save"], help="Model saving strategy: choose from 'best', 'every'")
    parser.add_argument("--save_n", type=int, default=run_cfg["save_n"], help="If 'every' is selected, save every n epochs")
    parser.add_argument("--load_if_exists", type=str2bool, nargs='?', const=True, default=run_cfg["load_if_exists"], help="Load model if it exists")
    parser.add_argument("--train", type=str2bool, nargs='?', const=True, default=run_cfg["train"], help="Train the model")

    args = parser.parse_args()

    model_cfg = load_model_config(args.model_config)

    # model = HJB(args, model_cfg)
    model = SimpleHJB(args, model_cfg)

    if args.load_if_exists:
        try:
            model.load_state_dict(torch.load(run_cfg["save_path"] + ".pth", map_location=run_cfg["device"]))
            print("Model loaded successfully.")
        except FileNotFoundError:
            print("No model found, starting training from scratch.")

    if args.train:
        model.train_model(epochs=args.epochs, lr=args.lr, save_path=args.save_path, verbose=args.verbose, plot=False)

    timesteps, results = model.simulate_paths(n_paths=args.n_simulations, batch_size=args.sim_batch_size, seed=np.random.randint(0, 1000))
    model.plot_approx_vs_analytic(results, timesteps)

if __name__ == "__main__":
    main()