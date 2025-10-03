#!/bin/bash

# Enable exit on error
set -e

cd ../../../../

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ./ml_maldi_nn_conda_env/

current_dir=$(pwd)

cd bin/ml

python3 bare_inference.py   --model "MaldiTransformerWrapper" \
                            --dataset "DRIAMS-A" \
                            --target "genera" \
                            --split_type "species" \
                            --inference_set "test" \
                            --run_for_score \
                            --new_paths \
                            --checkpoint_path "/data/nas-gpu/wang/mstro016/SourceCode/16s_sim_pred/bin/ml/MALDI-Transformer_Reproduction/k=0/malditrfvanilla_M_200_0.15_0.01_0.0005/version_1/checkpoints/epoch=5783-step=480000.ckpt" \
                            --maldi_nn_preprocessing \
                            -k 0