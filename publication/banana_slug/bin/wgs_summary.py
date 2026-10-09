#!/usr/bin/env python3
"""
Merge all WGS analysis summaries into a master table with quality tiers.
"""

import argparse
import csv
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Merge WGS analysis summaries.")
    parser.add_argument("--quast-dir", default=None, help="QUAST output dir.")
    parser.add_argument("--checkm2-dir", default=None, help="CheckM2 output dir.")
    parser.add_argument("--abricate-dir", default=None, help="Abricate output dir.")
    parser.add_argument("--mlst-dir", default=None, help="MLST output dir.")
    parser.add_argument("--gtdbtk-dir", default=None, help="GTDB-tk output dir.")
    parser.add_argument("--qc-16s", default=None, help="Path to 16S QC report TSV.")
    parser.add_argument("-o", "--output-dir", required=True, help="Output directory.")
    return parser.parse_args()


def load_tsv(path, key_field="sample"):
    """Load a TSV into a dict keyed by sample name."""
    data = {}
    if path and Path(path).exists():
        with open(path) as f:
            for row in csv.DictReader(f, delimiter="\t"):
                data[row[key_field]] = row
    return data


def assign_quality_tier(quast, checkm2, qc_16s):
    """Assign a quality tier based on available metrics."""
    completeness = float(checkm2.get("completeness", 0)) if checkm2 else 0
    contamination = float(checkm2.get("contamination", 100)) if checkm2 else 100
    status_16s = qc_16s.get("status", "") if qc_16s else ""
    n_contigs = int(quast.get("n_contigs_total", 9999)) if quast else 9999

    if contamination >= 10 or status_16s == "MULTI_ORGANISM":
        return "LOW"
    if completeness < 50:
        return "FAIL"
    if completeness >= 90 and contamination < 5 and n_contigs <= 50:
        return "HIGH"
    return "MEDIUM"


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all summaries
    quast = load_tsv(Path(args.quast_dir) / "summary.tsv") if args.quast_dir else {}
    checkm2 = load_tsv(Path(args.checkm2_dir) / "summary.tsv") if args.checkm2_dir else {}
    amr = load_tsv(Path(args.abricate_dir) / "amr_summary.tsv") if args.abricate_dir else {}
    vf = load_tsv(Path(args.abricate_dir) / "virulence_summary.tsv") if args.abricate_dir else {}
    mlst = load_tsv(Path(args.mlst_dir) / "summary.tsv") if args.mlst_dir else {}
    gtdbtk = load_tsv(Path(args.gtdbtk_dir) / "summary.tsv") if args.gtdbtk_dir else {}
    qc_16s = load_tsv(args.qc_16s) if args.qc_16s else {}

    # Collect all sample names
    all_samples = sorted(set(
        list(quast.keys()) + list(checkm2.keys()) + list(amr.keys()) +
        list(mlst.keys()) + list(gtdbtk.keys()) + list(qc_16s.keys())
    ))

    if not all_samples:
        print("Error: no data found in any summary file.", file=sys.stderr)
        sys.exit(1)

    print(f"Merging data for {len(all_samples)} sample(s)\n")

    fields = [
        "sample",
        "gtdbtk_genus", "gtdbtk_species", "gtdbtk_classification_method",
        "n_contigs", "total_length_bp", "largest_contig", "n50", "gc_pct",
        "completeness", "contamination", "checkm2_qc",
        "mlst_scheme", "mlst_st",
        "n_amr_genes", "amr_genes",
        "n_virulence_genes", "virulence_genes",
        "16s_status", "quality_tier",
    ]

    rows = []
    for sample in all_samples:
        q = quast.get(sample, {})
        c = checkm2.get(sample, {})
        a = amr.get(sample, {})
        v = vf.get(sample, {})
        m = mlst.get(sample, {})
        g = gtdbtk.get(sample, {})
        s = qc_16s.get(sample, {})

        tier = assign_quality_tier(q, c, s)

        row = {
            "sample": sample,
            "gtdbtk_genus": g.get("genus", "N/A"),
            "gtdbtk_species": g.get("species", "N/A"),
            "gtdbtk_classification_method": g.get("classification_method", "N/A"),
            "n_contigs": q.get("n_contigs_total", "N/A"),
            "total_length_bp": q.get("total_length_bp", "N/A"),
            "largest_contig": q.get("largest_contig", "N/A"),
            "n50": q.get("n50", "N/A"),
            "gc_pct": q.get("gc_pct", "N/A"),
            "completeness": c.get("completeness", "N/A"),
            "contamination": c.get("contamination", "N/A"),
            "checkm2_qc": c.get("qc_flag", "N/A"),
            "mlst_scheme": m.get("scheme", "N/A"),
            "mlst_st": m.get("st", "N/A"),
            "n_amr_genes": a.get("n_amr_genes", "0"),
            "amr_genes": a.get("amr_genes", "none"),
            "n_virulence_genes": v.get("n_virulence_genes", "0"),
            "virulence_genes": v.get("virulence_genes", "none"),
            "16s_status": s.get("status", "N/A"),
            "quality_tier": tier,
        }
        rows.append(row)

        marker = ""
        if tier in ("LOW", "FAIL"):
            marker = f" *** {tier} ***"
        gtdbtk_id = row["gtdbtk_species"] if row["gtdbtk_species"] != "N/A" else row["gtdbtk_genus"]
        print(f"  {sample}: tier={tier}, taxonomy={gtdbtk_id}, "
              f"completeness={row['completeness']}%, "
              f"contamination={row['contamination']}%{marker}")

    # Write master summary
    master_path = output_dir / "master_summary.tsv"
    with open(master_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Write flagged samples
    flagged = [r for r in rows if r["quality_tier"] in ("LOW", "FAIL")]
    flagged_path = output_dir / "flagged_samples.tsv"
    with open(flagged_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in flagged:
            writer.writerow(row)

    print(f"\nMaster summary:  {master_path}")
    print(f"Flagged samples: {flagged_path} ({len(flagged)} sample(s))")

    n_high = sum(1 for r in rows if r["quality_tier"] == "HIGH")
    n_med = sum(1 for r in rows if r["quality_tier"] == "MEDIUM")
    n_low = sum(1 for r in rows if r["quality_tier"] == "LOW")
    n_fail = sum(1 for r in rows if r["quality_tier"] == "FAIL")
    print(f"Quality: {n_high} HIGH, {n_med} MEDIUM, {n_low} LOW, {n_fail} FAIL.")


if __name__ == "__main__":
    main()
