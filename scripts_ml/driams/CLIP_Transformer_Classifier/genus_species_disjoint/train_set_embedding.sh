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
                            --checkpoint_path "./lightning_logs_DRIAMS_A/CLIP_Transformer_Classifier/version_8/checkpoints/epoch=1173-step=5870.ckpt" \
                            --dataset "DIRAMS-A" \
                            --target "genera" \
                            --split_type "species" \
                            --inference_set "train"
