#!/bin/bash

# Exit on any failure
set -e

cd ../..

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ./ml_maldi_nn_conda_env/

cd bin

INPUT_MZML_DIR="../data/RKI/processed/mzML"
METADATA_CSV="../data/RKI/processed/rki_metadata.csv"
OUTPUT_DIR="../data/RKI/processed/MaldiTransformer"

mkdir -p "$OUTPUT_DIR"

python3 process_MALDI_Transformer_data.py \
	--dataset rki \
	--input_mzml_path "$INPUT_MZML_DIR" \
	--metadata_csv "$METADATA_CSV" \
	--output_dir "$OUTPUT_DIR"

python3 convert_maldi_transformer_to_pt.py \
    --input_h5torch_path "$OUTPUT_DIR/MaldiTransformer_peaks.h5torch" \
    --output_pt_dir "$OUTPUT_DIR/spectra/all"