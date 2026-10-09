#!/bin/bash

# Exit on any failure
set -e

cd ../..

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ./ml_maldi_nn_conda_env/

cd bin

# Symlink the raw download to the raw folder
ROOT_FOR_MALDITRANSFORMER="../data/driams/raw/MaldiTransformer/DRIAMS_Root/"
mkdir -p "../data/driams/raw/MaldiTransformer/"
echo "Attempting to link from $(realpath ../data/driams/downloads/DRIAMS-A) "
echo "to $(realpath $ROOT_FOR_MALDITRANSFORMER/DRIAMS-A)"

mkdir -p $ROOT_FOR_MALDITRANSFORMER


if [ ! -L "$ROOT_FOR_MALDITRANSFORMER/DRIAMS-A" ]; then
    ln -s "$(realpath ../data/driams/downloads/DRIAMS-A)" "$(realpath $ROOT_FOR_MALDITRANSFORMER/DRIAMS-A)"
fi

# python3 process_MALDI_Transformer_data.py \
#             --input_raw_path $ROOT_FOR_MALDITRANSFORMER \
#             --output_dir "../data/driams/processed_data/MaldiTransformer"


# # To generate splits compatable with the MaldiTransformer format 
# # Species
# python3 resplit_maldi_transformer_data.py \
#             --input_h5torch_path ../data/driams/processed_data/MaldiTransformer/MaldiTransformer_peaks.h5torch \
#             --split_dir ../data/driams/processed_data/species/ \
#             --output_h5torch_dir ../data/driams/processed_data/MaldiTransformer_Genus_Labels/species \
#             --metadata_path ../data/driams/preprocessing/merged_metadata.csv \
#             --accessions



# python3 convert_maldi_transformer_to_pt.py \
#             --input_h5torch_path ../data/driams/processed_data/MaldiTransformer/MaldiTransformer_peaks.h5torch \
#             --output_pt_dir ../data/driams/processed_data/MaldiTransformer_Genus_Labels/species/spectra/all

# Genus
python3 resplit_maldi_transformer_data.py \
            --input_h5torch_path ../data/driams/processed_data/MaldiTransformer/MaldiTransformer_peaks.h5torch \
            --split_dir ../data/driams/processed_data/genera/ \
            --output_h5torch_dir ../data/driams/processed_data/MaldiTransformer_Genus_Labels/genera \
            --metadata_path ../data/driams/preprocessing/merged_metadata.csv \
            --accessions

python3 convert_maldi_transformer_to_pt.py \
            --input_h5torch_path ../data/driams/processed_data/MaldiTransformer/MaldiTransformer_peaks.h5torch \
            --output_pt_dir ../data/driams/processed_data/MaldiTransformer_Genus_Labels/genera/spectra/all