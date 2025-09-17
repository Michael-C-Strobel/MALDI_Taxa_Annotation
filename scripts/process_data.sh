#!/bin/bash

# Enable exit on error
set -e

cd ..

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ./conda_env

current_dir=$(pwd)

cd bin

python3 process_data.py --input_file "../data/idbac_db/raw/spectra.json" \
                        --output_mzML_dir "../data/idbac_db/raw/converted_to_mzml/" \
                        --output_dir "../data/idbac_db/preprocessing/baseline_corrected/"

# Label Generation

echo "This script will use your github user.email to get an API key for the NCBI API. Use ctrl+C to abort"
sleep 1

# email=$(git config user.email) # Will use your email for the API key

# python3 collect_fasta_files.py --input_csv "../data/idbac_db/raw/db.csv" \
#                                 --output_csv "../data/idbac_db/raw/ammended_db.csv" \
#                                 --output_dir "../data/idbac_db/raw/fasta_files/"  \
#                                 --email $email

# # # Blast
# echo "Running BLAST"
# cd $current_dir/scripts/
# pwd
# bash all_pairs_blast.sh


#### THIS IS NOW DONE IN DATASET.PY
# # Convert to ML pipeline-ready format
# conda activate ../ml_conda_env
# cd $current_dir
# cd bin
# python3 ml/preprocess.py  --input_dir "../data/idbac_db/preprocessing/" \
#                         --output_dir "../data/idbac_db/processed_data/" \
#                         --metadata_file "../data/idbac_db/raw/ammended_db.csv"