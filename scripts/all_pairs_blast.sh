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
mkdir -p "$DB_DIR"  # Ensure the DB directory exists

# Create a single BLAST database from all FASTA files
combined_db="$DB_DIR/combined_db"

# Cat all FASTA files together
cat "$FASTA_DIR"/*.fasta > "$combined_db.fasta"

$MAKEBLASTDB -in "$combined_db.fasta" -dbtype nucl -out "$combined_db" -title "Combined FASTA Database"


# Count the number of FASTA files
total_files=$(ls -1 "$FASTA_DIR"/*.fasta | wc -l)
counter=0


# Perform pairwise comparisons against the combined database
for query in "$FASTA_DIR"/*.fasta; do
    # Run blastn comparison for each FASTA file against the combined database
                $BLASTN -query "$query" -db "$combined_db" \
                    -out "$OUTPUT_DIR/$(basename "$query" .fasta).txt" \
                    -outfmt 6 \
                    -max_target_seqs $total_files \
                    -num_threads 4

    # Increment the counter
    counter=$((counter + 1))

    # Print progress
    echo "Started comparison $counter of $total_files: $(basename "$query") vs combined_db"

done

# Wait for all blastn processes to finish
wait
