#!/usr/bin/env python3
"""
Generate consensus 16S rRNA sequences from multi-copy FASTA files.

For each sample, aligns all 16S copies with MUSCLE, then calls a
majority-rule consensus (most frequent base at each column; ties
broken alphabetically; columns that are majority-gap are excluded).
"""

import argparse
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create consensus 16S sequences from multi-copy FASTAs."
    )
    parser.add_argument(
        "-i", "--input-dir", required=True,
        help="Directory containing per-sample 16S FASTA files (*.16S.fasta)."
    )
    parser.add_argument(
        "-o", "--output-dir", required=True,
        help="Directory for consensus FASTA output files."
    )
    parser.add_argument(
        "--muscle", default="muscle",
        help="Path to MUSCLE executable (default: muscle)."
    )
    parser.add_argument(
        "--min-copies", type=int, default=1,
        help="Minimum number of 16S copies required to produce a consensus (default: 1)."
    )
    return parser.parse_args()


def align_sequences(fasta_path, muscle_bin):
    """Run MUSCLE alignment on a multi-sequence FASTA and return aligned records."""
    with tempfile.NamedTemporaryFile(suffix=".afa", delete=False) as tmp:
        aln_path = tmp.name

    try:
        cmd = [muscle_bin, "-align", str(fasta_path), "-output", aln_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  MUSCLE error: {result.stderr.strip()}", file=sys.stderr)
            return None
        records = list(SeqIO.parse(aln_path, "fasta"))
        return records
    finally:
        if os.path.exists(aln_path):
            os.unlink(aln_path)


def make_consensus(aligned_records, threshold=0.5):
    """
    Build a majority-rule consensus from aligned sequences.

    At each column, the most frequent non-gap character is chosen if it
    appears in more than `threshold` fraction of sequences. Columns that
    are majority-gap are dropped (not included in the consensus).
    """
    if not aligned_records:
        return ""

    aln_len = len(aligned_records[0].seq)
    n_seqs = len(aligned_records)
    consensus = []

    for i in range(aln_len):
        column = [str(rec.seq[i]).upper() for rec in aligned_records]
        counts = Counter(column)

        gap_count = counts.get("-", 0) + counts.get(".", 0)
        if gap_count / n_seqs > threshold:
            continue

        base_counts = {b: c for b, c in counts.items() if b not in ("-", ".")}
        if not base_counts:
            continue

        best_base = max(base_counts, key=lambda b: (base_counts[b], -ord(b)))
        consensus.append(best_base)

    return "".join(consensus)


def process_sample(fasta_path, output_dir, muscle_bin, min_copies):
    """Process a single sample FASTA file and write its consensus."""
    sample_name = Path(fasta_path).name.replace(".16S.fasta", "")
    records = list(SeqIO.parse(fasta_path, "fasta"))
    n_copies = len(records)

    if n_copies == 0:
        print(f"  {sample_name}: 0 copies — skipped (empty file)")
        return False

    if n_copies < min_copies:
        print(f"  {sample_name}: {n_copies} copies — skipped (below --min-copies {min_copies})")
        return False

    if n_copies == 1:
        consensus_seq = str(records[0].seq)
        print(f"  {sample_name}: 1 copy — used directly as consensus")
    else:
        print(f"  {sample_name}: {n_copies} copies — aligning with MUSCLE...")
        aligned = align_sequences(fasta_path, muscle_bin)
        if aligned is None:
            print(f"  {sample_name}: alignment failed — skipped", file=sys.stderr)
            return False
        consensus_seq = make_consensus(aligned)
        print(f"  {sample_name}: consensus length = {len(consensus_seq)} bp")

    out_record = SeqRecord(
        Seq(consensus_seq),
        id=f"{sample_name}_16S_consensus",
        description=f"consensus of {n_copies} 16S rRNA gene copies",
    )
    out_path = Path(output_dir) / f"{sample_name}.consensus_16S.fasta"
    SeqIO.write([out_record], str(out_path), "fasta")
    return True


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fasta_files = sorted(input_dir.glob("*.16S.fasta"))
    if not fasta_files:
        print(f"Error: no *.16S.fasta files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(fasta_files)} sample(s) in {input_dir}")
    print(f"Output directory: {output_dir}\n")

    success = 0
    skipped = 0
    for fasta in fasta_files:
        if process_sample(fasta, output_dir, args.muscle, args.min_copies):
            success += 1
        else:
            skipped += 1

    print(f"\nDone: {success} consensus sequence(s) written, {skipped} sample(s) skipped.")


if __name__ == "__main__":
    main()
