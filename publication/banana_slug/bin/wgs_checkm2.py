#!/usr/bin/env python3
"""
Run CheckM2 on WGS assemblies to assess completeness and contamination.
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run CheckM2 on WGS assemblies.")
    parser.add_argument("-i", "--input-dir", required=True, help="Directory with *.fasta assemblies.")
    parser.add_argument("-o", "--output-dir", required=True, help="Output directory for CheckM2 results.")
    parser.add_argument("--db", default=None, help="Path to CheckM2 database directory.")
    parser.add_argument("--download-db", action="store_true", help="Download CheckM2 database if missing.")
    parser.add_argument("--threads", type=int, default=4, help="Threads (default: 4).")
    return parser.parse_args()


def download_database(db_path):
    """Download CheckM2 database."""
    db_path = Path(db_path)
    db_path.mkdir(parents=True, exist_ok=True)
    print(f"Downloading CheckM2 database to {db_path} (~3.5 GB)...")
    cmd = ["checkm2", "database", "--download", "--path", str(db_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Database download failed: {result.stderr.strip()[:300]}", file=sys.stderr)
        sys.exit(1)
    print("Database download complete.")


def run_checkm2(input_dir, output_dir, db_path, threads):
    """Run CheckM2 predict on all assemblies."""
    cmd = [
        "checkm2", "predict",
        "--input", str(input_dir),
        "--output-directory", str(output_dir / "checkm2_raw"),
        "-x", "fasta",
        "--threads", str(threads),
        "--force",
    ]
    if db_path:
        cmd.extend(["--database_path", str(db_path)])

    print("Running CheckM2 predict...", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"CheckM2 error: {result.stderr.strip()[:500]}", file=sys.stderr)
        return False
    return True


def parse_and_summarize(output_dir):
    """Parse CheckM2 output and add QC flags."""
    raw_report = output_dir / "checkm2_raw" / "quality_report.tsv"
    if not raw_report.exists():
        print(f"Error: {raw_report} not found", file=sys.stderr)
        return []

    rows = []
    with open(raw_report) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            completeness = float(row.get("Completeness", 0))
            contamination = float(row.get("Contamination", 0))

            if completeness >= 90 and contamination < 5:
                qc_flag = "PASS"
            elif completeness >= 50 and contamination < 10:
                qc_flag = "WARN"
            else:
                qc_flag = "FAIL"

            rows.append({
                "sample": row.get("Name", ""),
                "completeness": f"{completeness:.1f}",
                "contamination": f"{contamination:.1f}",
                "completeness_model": row.get("Completeness_Model_Used", ""),
                "coding_density": row.get("Coding_Density", ""),
                "contig_n50": row.get("Contig_N50", ""),
                "genome_size": row.get("Genome_Size", ""),
                "gc_content": row.get("GC_Content", ""),
                "total_coding_seqs": row.get("Total_Coding_Sequences", ""),
                "qc_flag": qc_flag,
            })
    return rows


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve database path
    db_path = None
    if args.db:
        db_path = Path(args.db)
        # Look for the actual .dmnd file inside the db directory
        dmnd_files = list(db_path.rglob("*.dmnd"))
        if dmnd_files:
            db_path = dmnd_files[0]
        elif args.download_db:
            download_database(args.db)
            dmnd_files = list(Path(args.db).rglob("*.dmnd"))
            if dmnd_files:
                db_path = dmnd_files[0]
        elif not dmnd_files:
            print(f"No CheckM2 database found at {args.db}.")
            if args.download_db:
                download_database(args.db)
                dmnd_files = list(Path(args.db).rglob("*.dmnd"))
                if dmnd_files:
                    db_path = dmnd_files[0]
            else:
                print("Use --download-db to download it, or set CHECKM2DB env var.")
                sys.exit(1)

    fastas = sorted(input_dir.glob("*.fasta"))
    print(f"Found {len(fastas)} assembly(ies) in {input_dir}")
    print(f"Output: {output_dir}\n")

    if not run_checkm2(input_dir, output_dir, db_path, args.threads):
        sys.exit(1)

    rows = parse_and_summarize(output_dir)
    if not rows:
        sys.exit(1)

    # Print results
    for row in sorted(rows, key=lambda r: r["sample"]):
        flag = row["qc_flag"]
        marker = " ***" if flag != "PASS" else ""
        print(f"  {row['sample']}: completeness={row['completeness']}%, "
              f"contamination={row['contamination']}%, {flag}{marker}")

    # Write summary
    fields = list(rows[0].keys())
    summary_path = output_dir / "summary.tsv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r["sample"]):
            writer.writerow(row)

    print(f"\nSummary: {summary_path}")
    n_pass = sum(1 for r in rows if r["qc_flag"] == "PASS")
    n_warn = sum(1 for r in rows if r["qc_flag"] == "WARN")
    n_fail = sum(1 for r in rows if r["qc_flag"] == "FAIL")
    print(f"Done: {n_pass} PASS, {n_warn} WARN, {n_fail} FAIL.")


if __name__ == "__main__":
    main()
