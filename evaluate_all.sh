#!/bin/bash

python evaluate.py \
  --seed 42 \
  --agents "Analytical" "NN" "TWAP" \
  --model_dir "saved_models/evaluation_all/psi0_naislstm_GELU"
