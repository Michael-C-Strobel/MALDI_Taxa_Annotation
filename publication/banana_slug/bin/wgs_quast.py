#!/usr/bin/env python3
"""
Compute assembly QC metrics for WGS assemblies.

Calculates the same core metrics as QUAST (N50, L50, GC%, contig counts,
total length, etc.) using Biopython — no external tool required.
"""

import argparse
import csv
import sys
from pathlib import Path

from Bio import SeqIO


def parse_args():
    parser = argparse.ArgumentParser(description="Assembly QC metrics for WGS FASTAs.")
    parser.add_argument("-i", "--input-dir", required=True, help="Directory with *.fasta assemblies.")
    parser.add_argument("-o", "--output-dir", required=True, help="Output directory.")
    parser.add_argument("--min-contig", type=int, default=500, help="Min contig length to include (default: 500).")
    return parser.parse_args()


def compute_metrics(fasta_path, min_contig):
    """Compute assembly metrics for a single FASTA."""
    records = list(SeqIO.parse(str(fasta_path), "fasta"))

    all_lengths = sorted([len(r.seq) for r in records], reverse=True)
    filtered_lengths = sorted([l for l in all_lengths if l >= min_contig], reverse=True)

    total_all = sum(all_lengths)
    total_filtered = sum(filtered_lengths)

    # GC content
    gc = 0
    n_count = 0
    total_bases = 0
    for r in records:
        seq = str(r.seq).upper()
        gc += seq.count("G") + seq.count("C")
        n_count += seq.count("N")
        total_bases += len(seq)
    gc_pct = 100 * gc / (total_bases - n_count) if (total_bases - n_count) > 0 else 0

    # N50, L50, N75, L75
    def calc_nx(lengths, total, x=50):
        target = total * x / 100
        cumsum = 0
        for i, l in enumerate(lengths):
            cumsum += l
            if cumsum >= target:
                return l, i + 1
        return 0, 0

    n50, l50 = calc_nx(filtered_lengths, total_filtered, 50)
    n75, l75 = calc_nx(filtered_lengths, total_filtered, 75)

    # N's per 100 kbp
    n_per_100k = round(n_count * 100000 / total_all, 2) if total_all > 0 else 0

    return {
        "n_contigs_total": len(all_lengths),
        "n_contigs": len(filtered_lengths),
        "total_length_bp": total_all,
        "total_length_filtered": total_filtered,
        "largest_contig": all_lengths[0] if all_lengths else 0,
        "gc_pct": f"{gc_pct:.2f}",
        "n50": n50,
        "n75": n75,
        "l50": l50,
        "l75": l75,
        "n_per_100kbp": n_per_100k,
    }


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fastas = sorted(input_dir.glob("*.fasta"))
    if not fastas:
        print(f"Error: no *.fasta files in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(fastas)} assembly(ies) in {input_dir}")
    print(f"Min contig length: {args.min_contig}")
    print(f"Output: {output_dir}\n")

    all_metrics = []
    for fasta in fastas:
        sample = fasta.stem
        metrics = compute_metrics(fasta, args.min_contig)
        metrics["sample"] = sample
        all_metrics.append(metrics)

        print(f"  {sample}: {metrics['n_contigs_total']} contigs, "
              f"{metrics['total_length_bp']:,} bp, N50={metrics['n50']:,}, "
              f"GC={metrics['gc_pct']}%")

    # Write summary
    fields = ["sample", "n_contigs_total", "n_contigs", "total_length_bp",
              "total_length_filtered", "largest_contig", "gc_pct",
              "n50", "n75", "l50", "l75", "n_per_100kbp"]
    summary_path = output_dir / "summary.tsv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for m in all_metrics:
            writer.writerow(m)

    print(f"\nSummary: {summary_path}")
    print(f"Done: {len(all_metrics)} sample(s) processed.")


if __name__ == "__main__":
    main()
