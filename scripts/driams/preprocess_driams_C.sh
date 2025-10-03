#!/bin/bash

# Exit on any failure
set -e

cd ../..

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ./conda_env

cd bin

# DRIAMS-C
mkdir -p ../data/driams-C/ preprocessing

echo "Generating metdata file for DRIAMS-C"
python3 driams/generate_metadata_file.py --input_csv "../data/driams/downloads/DRIAMS-C/id/**/*_clean.csv" \
                                         --output_file "../data/driams-C/preprocessing/merged_metadata.csv"

echo "Processing DRIAMS-C raw data"
python3 process_data.py --input_file "../data/driams/downloads/DRIAMS-C/raw/**/*.txt" \
                        --driams_csv "../data/driams-C/preprocessing/merged_metadata.csv" \
                        --output_mzML_dir "../data/driams-C/raw/converted_to_mzml/driams-C" \
                        --output_dir "../data/driams-C/preprocessing/baseline_corrected/driams"


echo "Merging JSON spectra for DRIAMS-C"
python3 driams/merge_json_files.py --json_files "../data/driams-C/preprocessing/baseline_corrected/driams.json" \
                                   --output_file "../data/driams-C/preprocessing/baseline_corrected.json"