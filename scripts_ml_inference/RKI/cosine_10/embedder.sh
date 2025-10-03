#!/bin/bash

# Enable exit on error
set -e

cd ../../../

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ./ml_maldi_nn_conda_env/

current_dir=$(pwd)

cd bin/ml

VERSION=11

python3 bare_inference.py   --model "cosine_10" \
                            --dataset "RKI" \
                            --target "genera" \
                            --split_type "species" \
                            --inference_set "all" \
                            --run_for_score \
                            --new_paths