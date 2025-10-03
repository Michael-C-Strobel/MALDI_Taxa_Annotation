#!/bin/bash

set -e

cd ../../../../
pwd
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ./ml_maldi_nn_conda_env/

cd bin/ml

python3 abstracted_eval_malditransformer.py --model_path '/data/nas-gpu/wang/mstro016/SourceCode/16s_sim_pred/bin/ml/MALDI-Transformer_Reproduction/k=0/malditrfvanilla_M_200_0.15_0.01_0.0005/version_1/checkpoints/epoch=6024-step=500000.ckpt'\
                                         --data_path '../../data/driams/processed_data/MaldiTransformer_Genus_Labels/species/maldi_transformer_fold_0.h5torch'