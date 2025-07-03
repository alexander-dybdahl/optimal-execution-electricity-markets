#!/bin/bash

python evaluate.py \
  --seed 100 \
  --agents "Analytical" "NN" \
  --model_dir "saved_models/evaluation/psi0_naislstm_GELU"
