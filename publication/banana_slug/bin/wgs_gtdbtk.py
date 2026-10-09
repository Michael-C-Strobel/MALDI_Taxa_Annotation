#!/usr/bin/env python3
"""
Run GTDB-tk classify_wf on WGS assemblies for genus/species identification.
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run GTDB-tk classify_wf on WGS assemblies.")
    parser.add_argument("-i", "--input-dir", required=True, help="Directory with *.fasta assemblies.")
    parser.add_argument("-o", "--output-dir", required=True, help="Output directory for GTDB-tk results.")
    parser.add_argument("--db", default=None, help="Path to GTDB-tk reference data (overrides GTDBTK_DATA_PATH).")
    parser.add_argument("--download-db", action="store_true", help="Download GTDB-tk reference data if missing.")
    parser.add_argument("--threads", type=int, default=4, help="Threads (default: 4).")
    parser.add_argument("--pplacer-threads", type=int, default=None,
                        help="Threads for pplacer (default: same as --threads; reduce if memory-limited).")
    return parser.parse_args()


def download_database(db_path):
    """Download GTDB-tk reference data."""
    db_path = Path(db_path)
    db_path.mkdir(parents=True, exist_ok=True)
    print(f"Downloading GTDB-tk reference data to {db_path} (~85 GB)...")
    print("This will take a while on the first run.")
    cmd = ["download-db.sh", str(db_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Database download failed: {result.stderr.strip()[:500]}", file=sys.stderr)
        sys.exit(1)
    print("Database download complete.")


def check_db(db_path):
    """Verify the GTDB-tk reference data directory looks valid."""
    db = Path(db_path)
    # GTDB-tk expects subdirectories like taxonomy/, markers/, etc.
    expected = ["taxonomy", "markers", "fastani", "masks"]
    found = [d for d in expected if (db / d).is_dir()]
    return len(found) >= 2


def run_classify_wf(input_dir, output_dir, threads, pplacer_threads):
    """Run gtdbtk classify_wf."""
    raw_dir = output_dir / "gtdbtk_raw"
    cmd = [
        "gtdbtk", "classify_wf",
        "--genome_dir", str(input_dir),
        "--out_dir", str(raw_dir),
        "--extension", "fasta",
        "--cpus", str(threads),
    ]
    if pplacer_threads:
        cmd.extend(["--pplacer_cpus", str(pplacer_threads)])

    print("Running GTDB-tk classify_wf...", flush=True)
    print(f"  Command: {' '.join(cmd)}\n", flush=True)
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print("GTDB-tk classify_wf failed.", file=sys.stderr)
        return False
    return True


def parse_summary(output_dir):
    """Parse GTDB-tk summary TSV(s) and extract genus/species assignments."""
    raw_dir = output_dir / "gtdbtk_raw"
    rows = []

    # GTDB-tk produces separate files for bacteria and archaea
    for summary_name in ["gtdbtk.bac120.summary.tsv", "gtdbtk.ar53.summary.tsv"]:
        summary_path = raw_dir / summary_name
        if not summary_path.exists():
            continue

        domain = "Bacteria" if "bac120" in summary_name else "Archaea"

        with open(summary_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                classification = row.get("classification", "")
                # Parse the GTDB taxonomy string: d__;p__;c__;o__;f__;g__;s__
                taxa = {}
                for level in classification.split(";"):
                    level = level.strip()
                    if "__" in level:
                        prefix, name = level.split("__", 1)
                        taxa[prefix] = name

                genus = taxa.get("g", "")
                species = taxa.get("s", "")
                # Species field is full binomial (e.g. "Streptomyces coelicolor");
                # extract the specific epithet
                species_epithet = ""
                if species and " " in species:
                    species_epithet = species.split(" ", 1)[1]
                elif species:
                    species_epithet = species

                rows.append({
                    "sample": row.get("user_genome", ""),
                    "domain": domain,
                    "classification": classification,
                    "genus": genus,
                    "species": species,
                    "species_epithet": species_epithet,
                    "fastani_reference": row.get("fastani_reference", ""),
                    "fastani_ani": row.get("fastani_ani", ""),
                    "fastani_af": row.get("fastani_af", ""),
                    "closest_placement_reference": row.get("closest_placement_reference", ""),
                    "classification_method": row.get("classification_method", ""),
                    "note": row.get("note", ""),
                    "warnings": row.get("warnings", ""),
                })
    return rows


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle database path
    if args.db:
        db_path = Path(args.db)
        if not check_db(db_path):
            if args.download_db:
                download_database(db_path)
            else:
                print(f"GTDB-tk reference data not found at {db_path}.", file=sys.stderr)
                print("Use --download-db to download, or set GTDBTK_DATA_PATH.", file=sys.stderr)
                sys.exit(1)
        os.environ["GTDBTK_DATA_PATH"] = str(db_path)
    elif not os.environ.get("GTDBTK_DATA_PATH"):
        print("Error: GTDB-tk reference data path not set.", file=sys.stderr)
        print("Provide --db <path> or set GTDBTK_DATA_PATH.", file=sys.stderr)
        sys.exit(1)

    fastas = sorted(input_dir.glob("*.fasta"))
    if not fastas:
        print(f"Error: no *.fasta files in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(fastas)} assembly(ies) in {input_dir}")
    print(f"Output: {output_dir}")
    print(f"GTDBTK_DATA_PATH: {os.environ.get('GTDBTK_DATA_PATH', 'N/A')}\n")

    pplacer_threads = args.pplacer_threads or args.threads
    if not run_classify_wf(input_dir, output_dir, args.threads, pplacer_threads):
        sys.exit(1)

    rows = parse_summary(output_dir)
    if not rows:
        print("Error: no GTDB-tk classification results found.", file=sys.stderr)
        sys.exit(1)

    # Print results
    for row in sorted(rows, key=lambda r: r["sample"]):
        genus = row["genus"] or "unclassified"
        species = row["species"] or "unclassified"
        method = row["classification_method"]
        ani = row["fastani_ani"]
        ani_str = f", ANI={ani}%" if ani else ""
        print(f"  {row['sample']}: {genus} / {species} ({method}{ani_str})")

    # Write summary
    fields = ["sample", "domain", "genus", "species", "species_epithet",
              "classification", "classification_method",
              "fastani_reference", "fastani_ani", "fastani_af",
              "closest_placement_reference", "note", "warnings"]
    summary_path = output_dir / "summary.tsv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r["sample"]):
            writer.writerow(row)

    print(f"\nSummary: {summary_path}")
    n_species = sum(1 for r in rows if r["species"])
    n_genus = sum(1 for r in rows if r["genus"])
    print(f"Done: {n_genus}/{len(rows)} assigned genus, {n_species}/{len(rows)} assigned species.")


if __name__ == "__main__":
    main()
