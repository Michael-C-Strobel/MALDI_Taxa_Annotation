#!/bin/bash

# Enable exit on error
set -e

cd ../../../

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ./ml_maldi_nn_conda_env/

current_dir=$(pwd)

cd bin/ml

VERSION=0

for k in {0..6}; do
    python3 bare_inference.py   --model "CLIP_Transformer" \
                                --dataset "RKI" \
                                --target "genera" \
                                --split_type "species" \
                                --inference_set "all" \
                                --run_for_score \
                                --new_paths \
                                --checkpoint_path "/data/nas-gpu/wang/mstro016/SourceCode/16s_sim_pred/bin/ml/lightning_logs_DRIAMS_A_for_score/genera/species/k=${k}/CLIP_Transformer/version_${VERSION}/checkpoints/best-checkpoint.ckpt"
done