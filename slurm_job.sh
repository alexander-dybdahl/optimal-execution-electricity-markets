#!/bin/bash
#SBATCH --job-name="optimal-execution-electricity-markets"
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4

source "/cluster/home/christdy/.cache/pypoetry/virtualenvs/optimal-execution-electricity-markets-PWlYJeun-py3.11/bin/activate"

export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NTASKS_PER_NODE * SLURM_NNODES))

echo $MASTER_ADDR
echo $MASTER_PORT
echo $WORLD_SIZE

poetry run torchrun \
  --nproc-per-node=$SLURM_NTASKS_PER_NODE \
  --nnodes=$SLURM_JOB_NUM_NODES \
  # --node_rank=$SLURM_NODEID \
  # --master-addr=$MASTER_ADDR \
  # --master-port=$MASTER_PORT \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  run.py \
  --parallel True \
  --supervised False \
  --load_if_exists True \
  --epochs 10000