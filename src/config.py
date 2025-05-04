import torch

# === Global Configuration ===
T = 1.0
N = 40
dt = T / N
dim = 4
dim_w = 3
batch_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Model Parameters ===
mu_P = 0.00
gamma = 0.2
eta = 0.1
def c_prod(xi): return 45.0 + 0.0 * xi
xi = 0.0
rho = -0.5

# === Initial Condition ===
y0 = torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=device)

# === Diffusion Coefficients ===
def sigma_P(T, t): return 0.2 * t / T
def sigma_D(T, t): return 0.1 * (T - t) / T
def sigma_B(T, t): return 0.1 * (T - t) / T
