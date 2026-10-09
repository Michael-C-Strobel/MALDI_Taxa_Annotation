#!/usr/bin/env python3
"""
Run MLST typing on WGS assemblies.
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run MLST on WGS assemblies.")
    parser.add_argument("-i", "--input-dir", required=True, help="Directory with *.fasta assemblies.")
    parser.add_argument("-o", "--output-dir", required=True, help="Output directory.")
    parser.add_argument("--scheme", default=None, help="Force a specific MLST scheme (default: auto-detect).")
    parser.add_argument("--threads", type=int, default=4, help="Threads (default: 4).")
    return parser.parse_args()


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
    print(f"Output: {output_dir}\n")

    # Run mlst on all files at once
    cmd = ["mlst", "--threads", str(args.threads)]
    if args.scheme:
        cmd.extend(["--scheme", args.scheme])
    cmd.extend(str(f) for f in fastas)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"MLST error: {result.stderr.strip()[:300]}", file=sys.stderr)
        sys.exit(1)

    # Save raw output
    raw_path = output_dir / "mlst_raw.tsv"
    with open(raw_path, "w") as f:
        f.write(result.stdout)

    # Parse and clean
    rows = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        fields = line.split("\t")
        sample = Path(fields[0]).stem
        scheme = fields[1] if len(fields) > 1 else "-"
        st = fields[2] if len(fields) > 2 else "-"
        alleles = "\t".join(fields[3:]) if len(fields) > 3 else ""

        rows.append({
            "sample": sample,
            "scheme": scheme,
            "st": st,
            "alleles": alleles,
        })

        st_display = f"ST-{st}" if st != "-" else "no match"
        print(f"  {sample}: {scheme} {st_display}")

    # Write summary
    summary_path = output_dir / "summary.tsv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample", "scheme", "st", "alleles"], delimiter="\t")
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r["sample"]):
            writer.writerow(row)

    print(f"\nSummary: {summary_path}")
    typed = sum(1 for r in rows if r["st"] != "-")
    print(f"Done: {typed}/{len(rows)} sample(s) typed.")


if __name__ == "__main__":
    main()
