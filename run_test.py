import json
import os
from argparse import ArgumentParser
from utils.logger import Logger

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from solvers.fbsnn import FBSNN
from dynamics.aid_dynamics import AidDynamics
from dynamics.hjb_dynamics import HJBDynamics
from utils.load_config import load_model_config, load_run_config
from utils.tools import str2bool

from dynamics.simple_dynamics import SimpleDynamics
from solvers.pinn_fbsnn import ValueFunctionNN
from solvers.pinn_fbsnn import train_pinn, simulate_and_plot_paths

def main():

    run_cfg = load_run_config(path="config/run_config.json")
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default=run_cfg["device"], help="Device to use for training (cpu or cuda)")
    parser.add_argument("--parallel", type=str2bool, nargs='?', const=True, default=run_cfg["parallel"], help="Use data parallelism")
    parser.add_argument("--epochs", type=int, default=run_cfg["epochs"], help="Number of training epochs")
    parser.add_argument("--K", type=int, default=run_cfg["K"], help="Epochs between evaluations of the model")
    parser.add_argument("--batch_size", type=int, default=run_cfg["batch_size"], help="Batch size for training")
    parser.add_argument("--supervised", type=str2bool, default=run_cfg["supervised"], help="Use supervised learning using analytical solution")
    parser.add_argument("--architecture", type=str, default=run_cfg["architecture"], help="Neural network architecture to use")
    parser.add_argument("--activation", type=str, default=run_cfg["activation"], help="Activation function to use")
    parser.add_argument("--Y_layers", type=int, nargs="+", default=run_cfg["Y_layers"], help="List of hidden layer sizes for the neural network")
    parser.add_argument("--adaptive", type=str2bool, nargs='?', const=True, default=run_cfg["adaptive"], help="Use adaptive learning rate")
    parser.add_argument("--adaptive_factor", type=float, default=run_cfg["adaptive_factor"], help="Adaptive factor")
    parser.add_argument("--lr", type=float, default=run_cfg["lr"], help="Learning rate for the optimizer")
    parser.add_argument("--lambda_Y", type=float, default=run_cfg["lambda_Y"], help="Weight for the Y term in the loss function")
    parser.add_argument("--lambda_dY", type=float, default=run_cfg["lambda_dY"], help="Weight for the dY term in the loss function")
    parser.add_argument("--lambda_dYt", type=float, default=run_cfg["lambda_dYt"], help="Weight for the dYt term in the loss function")
    parser.add_argument("--lambda_T", type=float, default=run_cfg["lambda_T"], help="Weight for the terminal term in the loss function")
    parser.add_argument("--lambda_TG", type=float, default=run_cfg["lambda_TG"], help="Weight for the terminal gradient term in the loss function")
    parser.add_argument("--lambda_pinn", type=float, default=run_cfg["lambda_pinn"], help="Weight for the PINN term in the loss function")
    parser.add_argument("--train", type=str2bool, nargs='?', const=True, default=run_cfg["train"], help="Train the model")
    parser.add_argument("--load_if_exists", type=str2bool, nargs='?', const=True, default=run_cfg["load_if_exists"], help="Load model if it exists")
    parser.add_argument("--save_path", type=str, default=run_cfg["save_path"], help="Path to save the model")
    parser.add_argument("--model_config", type=str, default=run_cfg["config_path"], help="Path to the model configuration file")
    parser.add_argument("--save", nargs="+", default=run_cfg["save"], help="Model saving strategy: choose from 'best', 'every'")
    parser.add_argument("--save_n", type=int, default=run_cfg["save_n"], help="If 'every' is selected, save every n epochs")
    parser.add_argument("--plot_n", type=int, default=run_cfg["plot_n"], help="Save plot every n epochs if plot_n is not None")
    parser.add_argument("--best", type=str2bool, nargs='?', const=True, default=run_cfg["best"], help="Run the model using the best model found during training")
    parser.add_argument("--verbose", type=str2bool, nargs='?', const=True, default=run_cfg["verbose"], help="Print training progress")
    parser.add_argument("--plot", type=str2bool, nargs='?', const=True, default=run_cfg["plot"], help="Plot after training")
    parser.add_argument("--plot_loss", type=str2bool, nargs='?', const=True, default=run_cfg["plot_loss"], help="Plot loss after training")
    parser.add_argument("--n_simulations", type=int, default=run_cfg["n_simulations"], help="Number of simulations to run")
    args = parser.parse_args()
    model_cfg = load_model_config(args.model_config)

    device = torch.device(args.device)
    args.global_rank = 0
    args.device_set = device
    args.batch_size_per_rank = args.batch_size

    dynamics = AidDynamics(args, model_cfg)

    model = ValueFunctionNN(input_dim=model_cfg["dim"] + 1)
    losses = train_pinn(model, dynamics, model_cfg, device=args.device, n_epochs=args.epochs)
    simulate_and_plot_paths(model, dynamics, model_cfg, device=args.device, n_simulations=args.n_simulations)

if __name__ == "__main__":
    main()