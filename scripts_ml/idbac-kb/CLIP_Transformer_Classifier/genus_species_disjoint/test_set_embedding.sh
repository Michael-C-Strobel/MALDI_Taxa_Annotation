#!/bin/bash

# Enable exit on error
set -e

cd ../../../../

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ./ml_maldi_nn_conda_env/

current_dir=$(pwd)

cd bin/ml

python3 bare_inference.py   --model "CLIP_Transformer_Classifier" \
                            --checkpoint_path "./lightning_logs/CLIP_Transformer_Classifier/version_1//checkpoints/epoch=329-step=6270.ckpt" \
                            --dataset "IDBac" \
                            --target "genera" \
                            --split_type "species" \
                            --inference_set "test"
