import numpy as np
import torch


class Solver:
    def __init__(self, dynamics, args):
        self.dynamics = dynamics
        self.device = args.device_set
