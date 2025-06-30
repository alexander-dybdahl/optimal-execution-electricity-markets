#!/bin/bash

# Load modules
module purge
module load Python/3.11.5-GCCcore-13.2.0

# Activate poetry environment
source $(poetry env info --path)/bin/activate

# Set rendezvous config
export NNODES=1
export NPROC_PER_NODE=4
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export OMP_NUM_THREADS=4

# Launch torchrun
torchrun \
  --nnodes=$NNODES \
  --nproc-per-node=$NPROC_PER_NODE \
  --rdzv_id=$(uuidgen) \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  train.py \
  --parallel True \
  --batch_size 16384 \
  --resume True \
  --reset_best True \
  --reset_lr True \
  --lr 1e-3 \
  --save_dir "saved_models/eta10_x0_gamma0.01_lowvol_psi5_scheeps1" \
  --seed 42

# Note: If stopped, remember to kill the job with:
# pkill -u $USER -f python
# lsof -i :29500
# kill -9 <PID>