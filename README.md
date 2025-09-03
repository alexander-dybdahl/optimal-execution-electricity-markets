# Optimal Execution in Electricity Markets using Deep BSDEs

This repository implements a Deep BSDE-based control solver for optimal execution problems in electricity markets. The solver learns a closed-loop feedback control policy for a stochastic system whose dynamics are influenced by control through the drift term. The method is based on stochastic optimal control, combining the Hamilton–Jacobi–Bellman (HJB) equation with its equivalent reformulation as a system of Forward–Backward Stochastic Differential Equations (FBSDEs). Training is guided by a physics-informed loss and simulation-based control cost, and is implemented in PyTorch for efficient experimentation.

## Problem Formulation

We consider the stochastic control problem of minimizing expected total cost:

$$
\min_{q \in \mathcal{A}} \ \mathbb{E} \left[ \int_0^T f(t, y_t, q_t) \, dt + h(y_T) \right],
$$

subject to the stochastic dynamics:

$$
dy_t = b(t, q_t)\, dt + \sigma(t)\, dW_t, \quad y_0 = y(0),
$$

where:
- $` y_t \in \mathbb{R}^n `$ is the state vector (e.g., price, inventory, forecast error),
- $` q_t \in \mathbb{R} `$ is the control process (trading rate),
- $` b(t, q) `$ is the drift depending directly on control $` q `$,
- $` \sigma(t) `$ is the time-dependent volatility matrix,
- $` W_t \in \mathbb{R}^m `$ is a Brownian motion.

The value function $` V(t, y) `$ satisfies the HJB equation. Under sufficient regularity, this equation admits a probabilistic representation as a solution to a coupled FBSDE system. The control $` q^\theta(t, y) `$ is obtained from the gradient of the neural approximation $` Y^\theta \approx V `$ by minimizing the Hamiltonian in feedback form.

## Methodology

The core approach combines:
- Deep neural network approximation of $` V(t, y) `$ using an NAIS-LSTM + NAISNet architecture
- Closed-loop control via $` q^\theta(t, y) `$ derived from minimizing the Hamiltonian
- PINN-based residual losses enforcing the HJB equation structure
- Forward-backward simulation of trajectories using the learned policy

The solver supports time-dependent volatilities, control-dependent drift, and volume uncertainty. The full objective includes:
- BSDE alignment loss
- Terminal state constraints
- Control cost penalty
- (Optional) HJB residual enforcement

## Project Structure

```
.
├── agents/                      # Execution agents (analytical, deep, immediate)
│   ├── analyticalagent.py
│   ├── deepagent.py
│   ├── immediateagent.py
│   └── timeweightedagent.py
│
├── config/                     # Config files (training, evaluation, dynamics)
│   ├── train_config.json
│   ├── eval_config.json
│   └── dynamics_configs/
│       ├── full_config.json
│       ├── full_config_SP01.json
│       └── ...
│
├── core/
│   └── solver.py               # Evaluation, simulation, comparison, and plotting 
│
├── dynamics/                   # Problem-specific dynamics classes
│   ├── full_dynamics.py
│   ├── simple_dynamics.py
│   └── aid_dynamics.py
│
├── utils/                      # Utility modules
│   ├── nnet.py
│   ├── plots.py
│   ├── simulator.py
│   └── ...
│
├── saved_models/               # Trained model checkpoints
├── saved_evaluations/          # Evaluation outputs
├── train.py                    # Training entry point
├── evaluate.py                 # Evaluation entry point
├── slurm_job.sh                # SLURM job script for HPC
├── pyproject.toml              # Poetry environment file
└── README.md
```

## Configuration

Training is configured through `config/train_config.json`, e.g.:

```
{
  "device": "cuda",
  "architecture": "naislstm",
  "activation": "GELU",
  "epochs": 100000,
  "batch_size": 256,
  "lr": 1e-2,
  "network_type": "dY",
  "lambda_cost": 1e4,
  "lambda_T": 1,
  "lambda_Y": 1,
  ...
}
```

Evaluation is configured via `config/eval_config.json`, e.g.:

```
{
  "device": "cuda",
  "model_dir": "saved_models/full_naislstm_GELU",
  "dynamics_path": "config/dynamics_configs/full_config.json",
  "n_simulations": 256,
  "plot_controls": true,
  "plot_trading_comparison": true
}
```

The model-specific problem configuration is stored in `dynamics_configs/full_config.json`, specifying initial conditions, volatility structure, impact parameters, etc.

## Running the Code

### Training

```
python train.py --config config/train_config.json
```

### Evaluation

```
python evaluate.py --config config/eval_config.json
```

### On HPC (NTNU Idun)

Load environment and submit job:

```
module load Python/3.11.5-GCCcore-13.2.0
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
poetry install
source $(poetry env info --path)/bin/activate
sbatch slurm_job.sh
```

Monitor queue:

```
squeue -u <username>
```

View live logs:

```
less +F slurm-<jobid>.out
```

Cancel job:

```
scancel <jobid>
```

## Output

Evaluation produces:
- Optimal control paths $` q(t) `$
- Value process $` Y(t) `$
- Trajectory distributions (inventory, prices, demand, imbalance)
- Terminal cost histograms and validation errors
- Comparison against analytical benchmarks (if available)

## Interactive Use

To run experiments in a Jupyter notebook:

```python
from agents.deepagent import DeepAgent
from utils.load_config import load_train_config

cfg = load_train_config("config/train_config.json")
agent = DeepAgent(cfg)
agent.simulate(n_paths=512)
agent.plot_controls()
```

## Notes

- This solver operates in discrete time using a fixed number of time steps $` N `$, and both the control and dynamics are evaluated on this grid.
- The HJB formulation is derived in continuous time, but solved numerically using discretized FBSDEs.
- The control policy is fully closed-loop, enabling generalization to unseen states via neural approximation.
- The method supports Sobol sampling, second-order Taylor expansion, and careful initialization heuristics for stability.

## References

1. Han, Jiequn; Jentzen, Arnulf; E, Weinan (2018). Solving high-dimensional partial differential equations using deep learning. Proceedings of the National Academy of Sciences 115(34), 8505–8510. DOI: 10.1073/pnas.1718942115. Preprint: arXiv:1707.02568

2. Pereira, Marcus A.; Wang, Ziyi; Exarchos, Ioannis; Theodorou, Evangelos A. (2019). Learning Deep Stochastic Optimal Control Policies using Forward-Backward SDEs. arXiv:1902.03986

3. Raissi, Maziar; Perdikaris, Paris; Karniadakis, George E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics 378, 686–707. DOI: 10.1016/j.jcp.2018.10.045
