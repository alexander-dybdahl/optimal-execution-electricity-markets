# Optimal Execution in Electricity Markets with Deep BSDEs

This project implements a Deep BSDE (Backward Stochastic Differential Equation) solver for modeling optimal execution strategies in electricity markets. It is written in PyTorch and structured for modularity, simulation, and visual diagnostics.

---

## Project Structure

```
optimal-execution-electricity-markets/
├── config/              # JSON configurations
│   ├── hjb_config.json
│   └── run_config.json
├── core/                # Base BSDE logic
│   └── base_bsde.py
├── models/              # HJB model and saved weights
│   ├── hjb_bsde.py
│   └── hjb_model.pth
├── notebooks/           # Jupyter notebooks (optional use)
├── old/                 # Archived experiments/code
├── utils/               # Helper utilities
│   ├── load_config.py
│   ├── plots.py
│   └── tools.py
├── run.py               # Main entry point
├── README.md
├── pyproject.toml       # Poetry environment
└── poetry.lock
```

---

## Installation

Install dependencies with [Poetry](https://python-poetry.org/):

```bash
git clone https://github.com/your-username/optimal-execution-electricity-markets.git
cd optimal-execution-electricity-markets
poetry install
poetry shell
```

---

## Features

- Deep BSDE training and simulation
- Modular architecture for multiple models
- JSON-based configuration
- CLI interface via argparse
- Jupyter notebook compatibility
- Plotting utilities for diagnostics

---

## Configuration

Two config files are used:

- `config/hjb_config.json` – model-specific settings
- `config/run_config.json` – runtime settings

Example `hjb_config.json`:
```json
{
  "T": 1.0,
  "N": 40,
  "dim": 4,
  "dim_W": 3,
  "y0": [0.0, 0.0, 0.0, 0.0],
  "xi": 0.0,
  "gamma": 0.2,
  "eta": 0.1,
  "mu_P": 0.0,
  "rho": -0.5,
  "vol_P": 0.2,
  "vol_D": 0.1,
  "vol_B": 0.1,
  "dt": 0.025
}
```

Example `run_config.json`:
```json
{
  "epochs": 1000,
  "lr": 1e-3,
  "save_path": "models/hjb_model.pth",
  "n_paths": 1000,
  "batch_size": 256,
  "device": "cuda",
  "train": true,
  "load_if_exists": true,
  "verbose": true
}
```

---

## Running

Run the training and simulation:

```bash
python run.py
```

Override config defaults with command-line arguments:

```bash
python run.py --epochs 500 --train True --verbose False
```

See all options:

```bash
python run.py --help
```

## Running distributed computation

On Idun (HPC cluster on NTNU) start a VS Code server with the following environment setup:

```bash
module load Python/3.11.5-GCCcore-13.2.0
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
poetry install
```

Add your project directory:

/cluster/home/<ntnu_username>

Run the training and simulation:

```bash
sbatch slurm_job.sh
```

It will then queue the job, to view all your jobs running or in a queue (this includes the job where you are running VS Code - don't stop this job):

```bash
squeue -u <ntnu_username>
```

Override distributed config defaults in the slurm_job.sh file. These settings might be the reason for why your job is in the queue for a long time.

To cancel a job with a specific PID number (this can be found using the above):

```bash
scancel <PID>
```

To view the log of your job live when it is running:

```bash
less +F slurm-<PID>.out
```

This log can also be found in the project directory, but this can be delayed.

## Output

The script produces:

- Trained model (`models/hjb_model.pth`)
- Simulation of optimal execution paths
- Visual plots using `matplotlib`:
  - Trading rate `q(t)`
  - Value function `Y(t)`
  - Terminal states: `X(T)`, `D(T)`, `B(T)`, `I(T)`

---

## Jupyter Usage

To explore interactively:

```python
from models.hjb_bsde import HJBDeepBSDE
from utils.load_config import load_model_config, load_run_config

cfg = load_model_config("config/hjb_config.json")
run_cfg = load_run_config()
model = HJBDeepBSDE(run_cfg, cfg)
```

Then run `simulate_paths` and plot diagnostics.

---

## Notes

- Run scripts from the **project root** (where `run.py` lives).
- Use `poetry run` if you don't want to enter the virtual environment manually.
