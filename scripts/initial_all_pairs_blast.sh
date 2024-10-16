#!/bin/bash

BLASTN="../Downloads/ncbi-blast-2.16.0+/bin/blastn"
MAKEBLASTDB="../Downloads/ncbi-blast-2.16.0+/bin/makeblastdb"

# Directory containing your FASTA files
FASTA_DIR="../data/idbac_db/raw/fasta_files"
DB_DIR="../data/idbac_db/raw/db"

# Output directory for BLAST results
OUTPUT_DIR="../data/idbac_db/preprocessing/blast_results"

# Make sure the output directory exists
mkdir -p "$OUTPUT_DIR"

# Loop over all FASTA files
for query in "$FASTA_DIR"/*.fasta; do

    bn=$(basename "$query")
    _db="$DB_DIR/$bn.db"
    # Create a BLAST database for each FASTA file
    $MAKEBLASTDB -in "$query" -dbtype nucl -out $_db -title "$(basename "$query" .fasta)"
    
    # Compare this FASTA file against every other one
    for subject in "$FASTA_DIR"/*.fasta; do
        if [ "$query" != "$subject" ]; then
            # Run blastn comparison
            $BLASTN -query "$query" -db "$_db" \
                    -out "$OUTPUT_DIR/$(basename "$query" .fasta)_vs_$(basename "$subject" .fasta).txt" \
                    -outfmt 6 \
                    -max_target_seqs 1000000 \
                    -num_threads 4
        fi
    done
done
