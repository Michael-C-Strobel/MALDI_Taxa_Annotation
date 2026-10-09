#!/usr/bin/env python3
"""
Analyze BLAST results for consensus 16S sequences.

Uses ete3's NCBITaxa to resolve and harmonize genus/species names from
BLAST hit descriptions against the NCBI taxonomy. Reports the most
common genus and all genera at configurable genus- and species-level
identity thresholds.
"""

import argparse
import csv
import json
import os
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

from ete3 import NCBITaxa


TAXDUMP_URL = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze 16S BLAST results at genus and species identity thresholds."
    )
    parser.add_argument(
        "-i", "--input-dir", required=True,
        help="Directory containing per-sample BLAST TSV files (*.blast.tsv)."
    )
    parser.add_argument(
        "-o", "--output-dir", required=True,
        help="Directory for analysis output files."
    )
    parser.add_argument(
        "--genus-threshold", type=float, default=95.0,
        help="Minimum percent identity for genus-level calls (default: 95.0)."
    )
    parser.add_argument(
        "--species-threshold", type=float, default=97.0,
        help="Minimum percent identity for species-level calls (default: 97.0)."
    )
    parser.add_argument(
        "--taxdump", default=None,
        help="Path to taxdump.tar.gz for ete3 NCBITaxa initialization. "
             "If omitted and the DB doesn't exist, it will be downloaded."
    )
    return parser.parse_args()


def init_ncbi_taxa(taxdump_path=None):
    """Initialize ete3 NCBITaxa, downloading the taxdump if needed."""
    db_path = os.path.join(os.path.expanduser("~"), ".etetoolkit", "taxa.sqlite")

    if os.path.exists(db_path):
        return NCBITaxa()

    if taxdump_path and os.path.exists(taxdump_path):
        print(f"Initializing taxonomy DB from {taxdump_path}...")
        return NCBITaxa(taxdump_file=taxdump_path)

    print("Taxonomy DB not found. Downloading taxdump from NCBI...")
    tmp_path = "/tmp/taxdump.tar.gz"
    urllib.request.urlretrieve(TAXDUMP_URL, tmp_path)
    print("Initializing taxonomy DB...")
    return NCBITaxa(taxdump_file=tmp_path)


def extract_organism_name(description):
    """
    Extract a candidate organism name from a BLAST hit description.

    Returns a list of candidates to try, from most specific to least.
    """
    text = description.strip()
    # Handle multiple descriptions separated by '>'
    text = text.split(">")[0].strip()

    text = re.sub(r"^[Uu]ncultured\s+", "", text)
    text = re.sub(r"^[Cc]andidatus\s+", "", text)

    words = text.split()
    if not words:
        return []

    candidates = []
    # Try "Genus species" if the second word looks like an epithet
    if (len(words) >= 2
            and words[1].islower()
            and words[1].isalpha()
            and words[1] not in ("sp", "cf", "aff", "str", "gene", "strain",
                                 "chromosome", "genomic", "plasmid", "partial",
                                 "complete", "contig", "scaffold")):
        candidates.append(f"{words[0]} {words[1]}")

    # Fall back to genus only
    candidates.append(words[0])
    return candidates


def resolve_taxonomy(description, ncbi, cache, failed_names, sample_name):
    """
    Resolve genus and species from a BLAST hit description using ete3.

    Returns (genus, species) with NCBI-standardized names.
    Returns (None, None) if unresolvable, and records the failure in failed_names.
    """
    candidates = extract_organism_name(description)

    for candidate in candidates:
        if candidate in cache:
            if cache[candidate] != (None, None):
                return cache[candidate]
            continue

        taxids = ncbi.get_name_translator([candidate])
        if not taxids:
            cache[candidate] = (None, None)
            continue

        tid = list(taxids.values())[0][0]
        lineage = ncbi.get_lineage(tid)
        ranks = ncbi.get_rank(lineage)
        names = ncbi.get_taxid_translator(lineage)

        genus_tids = [t for t, r in ranks.items() if r == "genus"]
        species_tids = [t for t, r in ranks.items() if r == "species"]

        genus = names[genus_tids[0]] if genus_tids else None
        species = names[species_tids[0]] if species_tids else None

        result = (genus, species)
        cache[candidate] = result

        if genus is not None:
            return result

    # All candidates failed — record the failure
    tried = [c for c in candidates if c]
    key = tried[0] if tried else description[:80]
    if key not in failed_names:
        failed_names[key] = {
            "candidates_tried": tried,
            "example_description": description[:200],
            "samples": set(),
            "n_occurrences": 0,
        }
    failed_names[key]["n_occurrences"] += 1
    failed_names[key]["samples"].add(sample_name)

    return (None, None)


def load_blast_tsv(tsv_path):
    """Load a per-sample BLAST TSV and return list of row dicts."""
    rows = []
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            row["percent_identity"] = float(row["percent_identity"])
            rows.append(row)
    return rows


def analyze_sample(rows, sample_name, ncbi, cache, failed_names, genus_threshold, species_threshold):
    """
    Analyze BLAST hits for one sample at both thresholds.

    Returns a dict with genus-level and species-level results.
    """
    result = {
        "genus_level": {"genera_counts": Counter(), "species_counts": Counter(), "n_hits": 0},
        "species_level": {"genera_counts": Counter(), "species_counts": Counter(), "n_hits": 0},
        "n_total_hits": len(rows),
        "n_failed_names": 0,
    }

    for row in rows:
        pct_id = row["percent_identity"]
        genus, species = resolve_taxonomy(
            row["hit_description"], ncbi, cache, failed_names, sample_name,
        )

        if genus is None:
            result["n_failed_names"] += 1
            continue

        if pct_id >= genus_threshold:
            result["genus_level"]["genera_counts"][genus] += 1
            result["genus_level"]["n_hits"] += 1
            if species:
                result["genus_level"]["species_counts"][species] += 1

        if pct_id >= species_threshold:
            result["species_level"]["genera_counts"][genus] += 1
            result["species_level"]["n_hits"] += 1
            if species:
                result["species_level"]["species_counts"][species] += 1

    # Derive summary fields
    for level in ("genus_level", "species_level"):
        gc = result[level]["genera_counts"]
        n = result[level]["n_hits"]
        if gc:
            top_g, top_g_count = gc.most_common(1)[0]
            result[level]["top_genus"] = top_g
            result[level]["top_genus_pct"] = 100 * top_g_count / n if n else 0
        else:
            result[level]["top_genus"] = None
            result[level]["top_genus_pct"] = 0
        result[level]["all_genera"] = [g for g, _ in gc.most_common()]

        sc = result[level]["species_counts"]
        if sc:
            top_s, top_s_count = sc.most_common(1)[0]
            result[level]["top_species"] = top_s
            result[level]["top_species_pct"] = 100 * top_s_count / n if n else 0
        else:
            result[level]["top_species"] = None
            result[level]["top_species_pct"] = 0
        result[level]["all_species"] = [s for s, _ in sc.most_common()]

    return result


def write_summary_tsv(all_results, output_path, genus_threshold, species_threshold):
    """Write the per-sample summary table."""
    gt = f"{genus_threshold:.0f}"
    st = f"{species_threshold:.0f}"
    fields = [
        "sample",
        "n_total_hits",
        "n_failed_names",
        f"top_genus_ge{gt}pct",
        f"top_genus_pct_of_hits_ge{gt}pct",
        f"all_genera_ge{gt}pct",
        f"n_genus_hits_ge{gt}pct",
        f"top_genus_ge{st}pct",
        f"top_species_ge{st}pct",
        f"top_species_pct_of_hits_ge{st}pct",
        f"all_genera_ge{st}pct",
        f"n_species_hits_ge{st}pct",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()

        for sample, res in sorted(all_results.items()):
            gl = res["genus_level"]
            sl = res["species_level"]
            writer.writerow({
                fields[0]: sample,
                fields[1]: res["n_total_hits"],
                fields[2]: res["n_failed_names"],
                fields[3]: gl["top_genus"] or "N/A",
                fields[4]: f"{gl['top_genus_pct']:.1f}" if gl["top_genus"] else "N/A",
                fields[5]: "; ".join(gl["all_genera"]) or "N/A",
                fields[6]: gl["n_hits"],
                fields[7]: sl["top_genus"] or "N/A",
                fields[8]: sl["top_species"] or "N/A",
                fields[9]: f"{sl['top_species_pct']:.1f}" if sl["top_species"] else "N/A",
                fields[10]: "; ".join(sl["all_genera"]) or "N/A",
                fields[11]: sl["n_hits"],
            })


def write_detail_tsv(all_results, output_path, genus_threshold, species_threshold):
    """Write a detailed per-sample breakdown with genus and species counts."""
    fields = [
        "sample", "threshold_level", "threshold_pct",
        "taxon_rank", "taxon", "hit_count", "rank",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()

        for sample, res in sorted(all_results.items()):
            for level_name, threshold in [("genus", genus_threshold), ("species", species_threshold)]:
                level_key = f"{level_name}_level"
                # Genus breakdown at this threshold
                for rank, (genus, count) in enumerate(res[level_key]["genera_counts"].most_common(), 1):
                    writer.writerow({
                        "sample": sample,
                        "threshold_level": level_name,
                        "threshold_pct": f"{threshold:.1f}",
                        "taxon_rank": "genus",
                        "taxon": genus,
                        "hit_count": count,
                        "rank": rank,
                    })
                # Species breakdown at this threshold
                for rank, (species, count) in enumerate(res[level_key]["species_counts"].most_common(), 1):
                    writer.writerow({
                        "sample": sample,
                        "threshold_level": level_name,
                        "threshold_pct": f"{threshold:.1f}",
                        "taxon_rank": "species",
                        "taxon": species,
                        "hit_count": count,
                        "rank": rank,
                    })


def write_failed_names_json(failed_names, output_path):
    """Write failed name mappings to JSON."""
    # Convert sets to sorted lists for JSON serialization
    serializable = {}
    for name, info in sorted(failed_names.items()):
        serializable[name] = {
            "candidates_tried": info["candidates_tried"],
            "example_description": info["example_description"],
            "samples": sorted(info["samples"]),
            "n_occurrences": info["n_occurrences"],
        }

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tsv_files = sorted(input_dir.glob("*.blast.tsv"))
    tsv_files = [f for f in tsv_files if f.name != "all_results.tsv"]

    if not tsv_files:
        print(f"Error: no *.blast.tsv files in {input_dir}", file=sys.stderr)
        sys.exit(1)

    ncbi = init_ncbi_taxa(args.taxdump)
    cache = {}
    failed_names = {}

    print(f"\nFound {len(tsv_files)} sample result(s) in {input_dir}")
    print(f"Genus threshold:   >= {args.genus_threshold}% identity")
    print(f"Species threshold: >= {args.species_threshold}% identity")
    print(f"Output directory:  {output_dir}\n")

    all_results = {}
    for tsv_path in tsv_files:
        sample = tsv_path.stem.replace(".blast", "")
        rows = load_blast_tsv(tsv_path)
        res = analyze_sample(
            rows, sample, ncbi, cache, failed_names,
            args.genus_threshold, args.species_threshold,
        )
        all_results[sample] = res

        gl = res["genus_level"]
        sl = res["species_level"]
        failed = res["n_failed_names"]
        total = res["n_total_hits"]
        print(f"  {sample}: ({failed}/{total} hits failed name resolution)")
        print(f"    genus level  (>={args.genus_threshold}%):  top={gl['top_genus'] or 'N/A'}, "
              f"{len(gl['all_genera'])} genera from {gl['n_hits']} hits")
        print(f"    species level (>={args.species_threshold}%): top={sl['top_species'] or sl['top_genus'] or 'N/A'}, "
              f"{len(sl['all_genera'])} genera from {sl['n_hits']} hits")

    summary_path = output_dir / "summary.tsv"
    write_summary_tsv(all_results, summary_path, args.genus_threshold, args.species_threshold)
    print(f"\nSummary: {summary_path}")

    detail_path = output_dir / "detail.tsv"
    write_detail_tsv(all_results, detail_path, args.genus_threshold, args.species_threshold)
    print(f"Detail:  {detail_path}")

    failed_path = output_dir / "failed_name_mappings.json"
    write_failed_names_json(failed_names, failed_path)
    print(f"Failed:  {failed_path} ({len(failed_names)} unique name(s))")

    print(f"\nDone: {len(all_results)} sample(s) analyzed.")


if __name__ == "__main__":
    main()
