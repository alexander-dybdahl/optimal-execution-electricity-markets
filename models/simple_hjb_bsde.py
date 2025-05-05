import torch
import numpy as np
from core.base_bsde import BaseDeepBSDE

class SimpleHJB(BaseDeepBSDE):
    def __init__(self, args, model_cfg):
        super().__init__(args, model_cfg)

    def generator(self, y, q):
        pass

    def terminal_cost(self, y):
        pass

    def mu(self, t, y, q):
        pass

    def sigma(self, t, y):
        pass