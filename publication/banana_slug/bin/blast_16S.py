#!/usr/bin/env python3
"""
Query consensus 16S sequences against NCBI BLAST and save filtered results.

Uses Biopython's NCBIWWW.qblast to run remote blastn queries against NCBI's
16S ribosomal RNA database. Results are filtered by percent identity and
saved as TSV files.
"""

import argparse
import os
import sys
import time
from pathlib import Path

from Bio import SeqIO
from Bio.Blast import NCBIWWW, NCBIXML


def parse_args():
    parser = argparse.ArgumentParser(
        description="BLAST consensus 16S sequences against NCBI and filter by identity."
    )
    parser.add_argument(
        "-i", "--input-dir", required=True,
        help="Directory containing consensus 16S FASTA files (*.consensus_16S.fasta)."
    )
    parser.add_argument(
        "-o", "--output-dir", required=True,
        help="Directory for BLAST results."
    )
    parser.add_argument(
        "--db", default="nt",
        help="NCBI database to search (default: nt)."
    )
    parser.add_argument(
        "--min-identity", type=float, default=95.0,
        help="Minimum percent identity to keep (default: 95.0)."
    )
    parser.add_argument(
        "--max-hits", type=int, default=500,
        help="Maximum number of hits to return per query (default: 500)."
    )
    parser.add_argument(
        "--evalue", type=float, default=1e-5,
        help="E-value threshold (default: 1e-5)."
    )
    parser.add_argument(
        "--delay", type=float, default=5.0,
        help="Seconds to wait between BLAST queries to respect NCBI rate limits (default: 5)."
    )
    return parser.parse_args()


TSV_HEADER = "\t".join([
    "sample",
    "hit_accession",
    "hit_description",
    "percent_identity",
    "alignment_length",
    "query_coverage_pct",
    "mismatches",
    "gap_opens",
    "evalue",
    "bit_score",
]) + "\n"


def run_blast(fasta_path, db, max_hits, evalue):
    """Submit a remote BLAST query and return the XML result string."""
    record = next(SeqIO.parse(str(fasta_path), "fasta"))
    query_seq = str(record.seq)

    result_handle = NCBIWWW.qblast(
        program="blastn",
        database=db,
        sequence=query_seq,
        hitlist_size=max_hits,
        expect=evalue,
        megablast=True,
    )
    xml_string = result_handle.read()
    result_handle.close()
    return xml_string, len(query_seq)


def parse_and_filter(xml_string, sample_name, query_length, min_identity):
    """Parse BLAST XML and return rows that pass the identity filter."""
    from io import StringIO
    blast_record = NCBIXML.read(StringIO(xml_string))

    rows = []
    for alignment in blast_record.alignments:
        for hsp in alignment.hsps:
            pct_identity = (hsp.identities / hsp.align_length) * 100
            if pct_identity < min_identity:
                continue

            query_cov = (hsp.align_length / query_length) * 100

            accession = alignment.accession
            description = alignment.hit_def
            # Truncate very long descriptions
            if len(description) > 200:
                description = description[:197] + "..."

            rows.append({
                "sample": sample_name,
                "hit_accession": accession,
                "hit_description": description,
                "percent_identity": f"{pct_identity:.2f}",
                "alignment_length": str(hsp.align_length),
                "query_coverage_pct": f"{query_cov:.1f}",
                "mismatches": str(hsp.align_length - hsp.identities - hsp.gaps),
                "gap_opens": str(hsp.gaps),
                "evalue": f"{hsp.expect:.2e}",
                "bit_score": f"{hsp.bits:.1f}",
            })
    return rows


def write_sample_tsv(rows, output_path):
    """Write filtered BLAST results for a single sample."""
    with open(output_path, "w") as f:
        f.write(TSV_HEADER)
        for row in rows:
            f.write("\t".join(row.values()) + "\n")


def process_sample(fasta_path, output_dir, db, max_hits, evalue, min_identity):
    """Run BLAST for one sample, save XML and filtered TSV."""
    sample_name = Path(fasta_path).name.replace(".consensus_16S.fasta", "")

    xml_dir = output_dir / "xml"
    xml_dir.mkdir(parents=True, exist_ok=True)

    print(f"  {sample_name}: submitting BLAST query...", flush=True)
    try:
        xml_string, query_length = run_blast(fasta_path, db, max_hits, evalue)
    except Exception as e:
        print(f"  {sample_name}: BLAST failed — {e}", file=sys.stderr)
        return []

    # Save raw XML
    xml_path = xml_dir / f"{sample_name}.blast.xml"
    with open(xml_path, "w") as f:
        f.write(xml_string)

    # Parse and filter
    rows = parse_and_filter(xml_string, sample_name, query_length, min_identity)
    print(f"  {sample_name}: {len(rows)} hit(s) >= {min_identity}% identity")

    # Save per-sample TSV
    tsv_path = output_dir / f"{sample_name}.blast.tsv"
    write_sample_tsv(rows, tsv_path)

    return rows


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fasta_files = sorted(input_dir.glob("*.consensus_16S.fasta"))
    if not fasta_files:
        print(f"Error: no *.consensus_16S.fasta files in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(fasta_files)} consensus sequence(s) in {input_dir}")
    print(f"Database: {args.db}")
    print(f"Min identity: {args.min_identity}%")
    print(f"Output directory: {output_dir}\n")

    all_rows = []
    for i, fasta in enumerate(fasta_files):
        rows = process_sample(
            fasta, output_dir, args.db,
            args.max_hits, args.evalue, args.min_identity,
        )
        all_rows.extend(rows)

        # Rate-limit between queries (not after the last one)
        if i < len(fasta_files) - 1:
            print(f"  waiting {args.delay}s before next query...", flush=True)
            time.sleep(args.delay)

    # Write combined results
    combined_path = output_dir / "all_results.tsv"
    with open(combined_path, "w") as f:
        f.write(TSV_HEADER)
        for row in all_rows:
            f.write("\t".join(row.values()) + "\n")

    print(f"\nDone: {len(all_rows)} total hit(s) across {len(fasta_files)} sample(s).")
    print(f"Combined results: {combined_path}")


if __name__ == "__main__":
    main()
