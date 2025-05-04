import torch
from config import gamma, eta, c_prod

# ========== Impact Functions ==========
def psi(q):
    return gamma * q  # permanent spread impact

def phi(q):
    return eta * q ** 2  # temporary execution cost

# ========== Generator ==========
def generator(y, q):
    P = y[:, 1:2]  # mid-price
    sign_q = torch.sign(q)
    P_exec = P + sign_q * psi(q) + phi(q)
    return -q * P_exec  # shape: (batch, 1)

# ========== Terminal Cost ==========
def terminal_cost(y, xi):
    X = y[:, 0]
    D = y[:, 2]
    B = y[:, 3]
    I = X - D + xi
    I_plus = torch.clamp(I, min=0.0)
    I_minus = torch.clamp(-I, min=0.0)
    return 100 * abs(I) #-c_prod(xi) + I_plus * B - I_minus * B
