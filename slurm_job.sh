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

echo "Running on $SLURM_JOB_NUM_NODES nodes with $SLURM_NTASKS_PER_NODE tasks per node."

# Launch torchrun
torchrun \
  --nproc-per-node=$SLURM_NTASKS_PER_NODE \
  --nnodes=$SLURM_JOB_NUM_NODES \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  poetry run run.py \
  --parallel True \
  --supervised False \
  --load_if_exists False \
  --epochs 10000
