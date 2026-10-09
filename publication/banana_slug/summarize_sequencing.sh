#!/usr/bin/env bash
# Rename any generic sequencing_summary files by their barcode and run the summary script.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SEQ_DIR="$SCRIPT_DIR/data/sequencing_summaries"

# Rename any files that still have the default "sequencing_summary" name
for f in "$SEQ_DIR"/sequencing_summary*.tsv; do
    [ -f "$f" ] || continue
    sample=$(awk -F'\t' 'NR==2 {print $NF}' "$f")
    dest="$SEQ_DIR/${sample}.sequencing_summary.tsv"
    if [ "$f" != "$dest" ]; then
        if [ -f "$dest" ]; then
            echo "Skipping $f (duplicate of $sample, $dest already exists)"
            continue
        fi
        mv "$f" "$dest"
        echo "Renamed: $(basename "$f") -> $(basename "$dest")"
    fi
done

python3 "$SCRIPT_DIR/summarize_sequencing.py"
