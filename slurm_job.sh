#!/bin/bash
#SBATCH --job-name="optimal-execution-electricity-markets"
#SBATCH --partition=GPUQ
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=1000G

# Load modules
module purge
module load Python/3.11.5-GCCcore-13.2.0

# Activate poetry environment
source $(poetry env info --path)/bin/activate

# Set rendezvous config
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500
export OMP_NUM_THREADS=4

# Launch torchrun
srun \
  python \
  run.py \
  --device cuda \
  --parallel True \
  --supervised False \
  --load_if_exists False \
  --epochs 10000
