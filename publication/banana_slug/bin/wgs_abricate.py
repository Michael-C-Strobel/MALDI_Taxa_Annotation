#!/usr/bin/env python3
"""
Screen WGS assemblies for AMR and virulence genes using abricate.
"""

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Screen assemblies with abricate.")
    parser.add_argument("-i", "--input-dir", required=True, help="Directory with *.fasta assemblies.")
    parser.add_argument("-o", "--output-dir", required=True, help="Output directory.")
    parser.add_argument("--databases", default="ncbi,card,resfinder,vfdb",
                        help="Comma-separated abricate databases (default: ncbi,card,resfinder,vfdb).")
    parser.add_argument("--minid", type=float, default=80, help="Min identity %% (default: 80).")
    parser.add_argument("--mincov", type=float, default=60, help="Min coverage %% (default: 60).")
    return parser.parse_args()


def setup_databases():
    """Ensure abricate databases are set up."""
    result = subprocess.run(["abricate", "--list"], capture_output=True, text=True)
    if result.returncode != 0:
        print("Running abricate --setupdb...", flush=True)
        subprocess.run(["abricate", "--setupdb"], capture_output=True, text=True)


def run_abricate(fasta_path, db, minid, mincov):
    """Run abricate on a single FASTA with a single database."""
    cmd = [
        "abricate",
        "--db", db,
        "--minid", str(minid),
        "--mincov", str(mincov),
        str(fasta_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return result.stdout


def parse_abricate_output(output_text):
    """Parse abricate tab-separated output into list of dicts.

    Handles both old (0.7, 12 columns) and new (1.0+, 15 columns) formats.
    """
    rows = []
    for line in output_text.strip().split("\n"):
        if not line.strip() or line.startswith("#") or line.startswith("Using ") or line.startswith("Processing") or line.startswith("Found "):
            continue
        fields = line.split("\t")
        if len(fields) >= 12:
            # Old format (0.7): FILE SEQ START END GENE COVERAGE MAP GAPS %COV %ID DB ACC
            # New format (1.0+): FILE SEQ START END STRAND GENE COVERAGE MAP GAPS %COV %ID DB ACC PRODUCT RESISTANCE
            if len(fields) >= 15:
                gene = fields[5]
                pct_cov = fields[9]
                pct_id = fields[10]
                database = fields[11]
                accession = fields[12]
                product = fields[13]
                resistance = fields[14]
            else:
                gene = fields[4]
                pct_cov = fields[8]
                pct_id = fields[9]
                database = fields[10]
                accession = fields[11]
                product = ""
                resistance = ""
            rows.append({
                "file": fields[0],
                "sequence": fields[1],
                "start": fields[2],
                "end": fields[3],
                "gene": gene,
                "pct_coverage": pct_cov,
                "pct_identity": pct_id,
                "database": database,
                "accession": accession,
                "product": product,
                "resistance": resistance,
            })
    return rows


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    databases = [d.strip() for d in args.databases.split(",")]

    fastas = sorted(input_dir.glob("*.fasta"))
    if not fastas:
        print(f"Error: no *.fasta files in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(fastas)} assembly(ies) in {input_dir}")
    print(f"Databases: {', '.join(databases)}")
    print(f"Output: {output_dir}\n")

    setup_databases()

    # Run abricate for each database
    all_amr_hits = []  # ncbi + card + resfinder
    all_vf_hits = []   # vfdb

    for db in databases:
        db_dir = output_dir / db
        db_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [{db}]")

        db_all_rows = []
        for fasta in fastas:
            sample = fasta.stem
            output_text = run_abricate(fasta, db, args.minid, args.mincov)
            if output_text is None:
                print(f"    {sample}: FAILED")
                continue

            rows = parse_abricate_output(output_text)
            # Tag with sample name
            for r in rows:
                r["sample"] = sample

            db_all_rows.extend(rows)
            n = len(rows)
            genes = ", ".join(sorted(set(r["gene"] for r in rows)))[:80] if rows else "none"
            print(f"    {sample}: {n} hit(s) — {genes}")

            # Save per-sample raw output
            raw_path = db_dir / f"{sample}.tsv"
            with open(raw_path, "w") as f:
                f.write(output_text)

        # Save combined per-database hits
        if db_all_rows:
            combined_path = db_dir / "all_hits.tsv"
            fields = ["sample"] + [k for k in db_all_rows[0].keys() if k != "sample"]
            with open(combined_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
                writer.writeheader()
                for row in db_all_rows:
                    writer.writerow(row)

        # Collect for summaries
        if db in ("ncbi", "card", "resfinder"):
            all_amr_hits.extend(db_all_rows)
        elif db == "vfdb":
            all_vf_hits.extend(db_all_rows)

    # Write AMR summary (per sample: gene list, count)
    _write_summary(all_amr_hits, output_dir / "amr_summary.tsv", fastas, "amr")
    _write_summary(all_vf_hits, output_dir / "virulence_summary.tsv", fastas, "virulence")

    print(f"\nAMR summary:       {output_dir / 'amr_summary.tsv'}")
    print(f"Virulence summary: {output_dir / 'virulence_summary.tsv'}")
    print("Done.")


def _write_summary(hits, output_path, fastas, label):
    """Write a per-sample summary of hits."""
    sample_genes = defaultdict(set)
    sample_details = defaultdict(list)
    for h in hits:
        sample_genes[h["sample"]].add(h["gene"])
        sample_details[h["sample"]].append(h)

    fields = ["sample", f"n_{label}_genes", f"{label}_genes"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for fasta in fastas:
            sample = fasta.stem
            genes = sorted(sample_genes.get(sample, set()))
            writer.writerow({
                "sample": sample,
                f"n_{label}_genes": len(genes),
                f"{label}_genes": "; ".join(genes) if genes else "none",
            })


if __name__ == "__main__":
    main()
