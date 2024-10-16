#!/bin/bash

cd ..

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ./conda_env

cd bin

# python3 process_data.py --json_input "../data/idbac_db/raw/spectra.json" \
#                         --output_mzML_dir "../data/idbac_db/raw/converted_to_mzml/" \
#                         --output_dir "../data/idbac_db/preprocessing/baseline_corrected/"

# Label Generation

echo "This script will use your github user.email to get an API key for the NCBI API. Use ctrl+C to abort"
sleep 1

email=$(git config user.email) # Will use your email for the API key

# python3 collect_fasta_files.py --input_csv "../data/idbac_db/raw/db.csv" \
#                                 --output_dir "../data/idbac_db/raw/fasta_files/"  \
#                                 --email $email

python3 pairwise_blast.py --input_dir "../data/idbac_db/raw/fasta_files/" \
                          --output_dir "../data/idbac_db/raw/blast_results/" \
                          --email $email