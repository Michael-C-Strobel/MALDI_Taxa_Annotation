#!/bin/bash

# Enable exit on error
set -e

cd ../../../../

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ./ml_maldi_nn_conda_env/

current_dir=$(pwd)

cd bin/ml

python3 bare_inference.py   --model "Prototyical_Transformer" \
                            --version 10 \
                            --dataset "DIRAMS-A" \
                            --target "genera" \
                            --split_type "genera" \
                            --inference_set "train"
