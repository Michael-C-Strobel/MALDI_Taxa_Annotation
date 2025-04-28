#!/bin/bash

# Enable exit on error
set -e

cd ../../../../

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ./ml_maldi_nn_conda_env/

current_dir=$(pwd)

cd bin/ml

python3 bare_inference.py   --model "CLIP_Transformer" \
                            --version 1 \
                            --dataset "DIRAMS-A" \
                            --target "species" \
                            --split_type "species" \
                            --inference_set "test"
