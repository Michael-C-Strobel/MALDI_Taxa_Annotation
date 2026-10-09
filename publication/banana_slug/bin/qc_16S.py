#!/usr/bin/env python3
"""
Quality-check 16S rRNA gene copies and detect multi-organism assemblies.

For each sample:
  - Reports sequence lengths and flags outliers
  - Checks for ambiguous bases and canonical 16S start
  - Aligns copies with MUSCLE and computes alignment-based pairwise identities
  - Clusters copies by identity threshold to detect mixed assemblies
  - Writes per-cluster FASTA files for downstream use
  - Optionally BLASTs a representative from each cluster for identification

Output structure:
  <output-dir>/clusters/  — per-cluster FASTA files (<sample>.cluster_<N>.fasta)
  <output-dir>/metrics/   — QC report TSV, cluster details TSV, full JSON
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from io import StringIO
from pathlib import Path

from Bio import SeqIO
from Bio.Blast import NCBIWWW, NCBIXML


CANONICAL_16S_START = "AGAGTTTGAT"


def parse_args():
    parser = argparse.ArgumentParser(
        description="QC 16S copies and detect multi-organism assemblies."
    )
    parser.add_argument(
        "-i", "--input-dir", required=True,
        help="Directory containing per-sample 16S FASTA files (*.16S.fasta)."
    )
    parser.add_argument(
        "-o", "--output-dir", required=True,
        help="Base output directory (clusters/ and metrics/ subdirs will be created)."
    )
    parser.add_argument(
        "--muscle", default="muscle",
        help="Path to MUSCLE executable (default: muscle)."
    )
    parser.add_argument(
        "--identity-threshold", type=float, default=90.0,
        help="Min pairwise identity (%%) to consider copies from the same organism (default: 90.0)."
    )
    parser.add_argument(
        "--blast-reps", action="store_true", default=False,
        help="BLAST a representative from each cluster against NCBI nt (slow, requires internet)."
    )
    parser.add_argument(
        "--blast-delay", type=float, default=5.0,
        help="Seconds between BLAST queries (default: 5)."
    )
    return parser.parse_args()


def align_sequences_muscle(records, muscle_bin):
    """Align sequences with MUSCLE and return aligned records."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as tmp_in:
        SeqIO.write(records, tmp_in, "fasta")
        in_path = tmp_in.name

    with tempfile.NamedTemporaryFile(suffix=".afa", delete=False) as tmp_out:
        out_path = tmp_out.name

    try:
        result = subprocess.run(
            [muscle_bin, "-align", in_path, "-output", out_path],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"    MUSCLE error: {result.stderr.strip()}", file=sys.stderr)
            return None
        return list(SeqIO.parse(out_path, "fasta"))
    finally:
        for p in (in_path, out_path):
            if os.path.exists(p):
                os.unlink(p)


def pairwise_identity_from_alignment(aligned_records):
    """
    Compute pairwise identities from a multiple sequence alignment.

    Identity = matches / (aligned columns excluding double-gap positions).
    """
    ids = {}
    seqs = [(r.id, str(r.seq)) for r in aligned_records]
    for i in range(len(seqs)):
        for j in range(i + 1, len(seqs)):
            name_i, seq_i = seqs[i]
            name_j, seq_j = seqs[j]
            matches = 0
            compared = 0
            for a, b in zip(seq_i, seq_j):
                if a == "-" and b == "-":
                    continue
                compared += 1
                if a == b:
                    matches += 1
            pct = 100 * matches / compared if compared > 0 else 0
            ids[(name_i, name_j)] = pct
            ids[(name_j, name_i)] = pct
    return ids


def cluster_by_identity(record_ids, pairwise_ids, threshold):
    """
    Single-linkage clustering: two sequences are in the same cluster
    if any pair between their clusters exceeds the threshold.
    """
    clusters = {rid: i for i, rid in enumerate(record_ids)}

    for (id_i, id_j), pct in pairwise_ids.items():
        if id_i >= id_j:
            continue
        if pct >= threshold:
            ci, cj = clusters[id_i], clusters[id_j]
            if ci != cj:
                target = min(ci, cj)
                source = max(ci, cj)
                for k in clusters:
                    if clusters[k] == source:
                        clusters[k] = target

    unique = sorted(set(clusters.values()))
    remap = {old: new + 1 for new, old in enumerate(unique)}
    return {rid: remap[cid] for rid, cid in clusters.items()}


def blast_sequence(seq_str, max_hits=3):
    """BLAST a single sequence against NCBI nt and return top hits."""
    try:
        result = NCBIWWW.qblast(
            "blastn", "nt", seq_str,
            hitlist_size=max_hits, expect=1e-5, megablast=True,
        )
        xml = result.read()
        result.close()
        blast_record = NCBIXML.read(StringIO(xml))

        hits = []
        for aln in blast_record.alignments[:max_hits]:
            hsp = aln.hsps[0]
            pct_id = 100 * hsp.identities / hsp.align_length
            qcov = 100 * hsp.align_length / len(seq_str)
            hits.append({
                "accession": aln.accession,
                "description": aln.hit_def[:120],
                "pct_identity": round(pct_id, 1),
                "query_coverage": round(qcov, 1),
            })
        return hits
    except Exception as e:
        return [{"accession": "ERROR", "description": str(e),
                 "pct_identity": 0, "query_coverage": 0}]


def parse_contig(description):
    """Extract contig number from record description."""
    parts = description.split("_")
    try:
        idx = parts.index("contig")
        return parts[idx + 1]
    except (ValueError, IndexError):
        return "unknown"


def parse_direction(description):
    """Extract DIR+ or DIR- from record description."""
    if "DIR+" in description:
        return "+"
    elif "DIR-" in description:
        return "-"
    return "?"


def analyze_sample(fasta_path, muscle_bin, identity_threshold, do_blast, blast_delay, blast_count):
    """Run full QC on a single sample. Returns a result dict."""
    sample = Path(fasta_path).stem.replace(".16S", "")
    records = list(SeqIO.parse(str(fasta_path), "fasta"))
    n_copies = len(records)

    result = {
        "sample": sample,
        "n_copies": n_copies,
        "lengths": [],
        "length_range": "",
        "n_ambiguous": 0,
        "canonical_start": True,
        "n_clusters": 0,
        "status": "",
        "clusters": [],
        "min_pairwise_identity": None,
    }

    if n_copies == 0:
        result["status"] = "EMPTY"
        return result

    # Basic sequence checks
    lengths = []
    n_ambig = 0
    all_canonical = True
    for r in records:
        seq = str(r.seq).upper()
        lengths.append(len(seq))
        if sum(1 for c in seq if c not in "ACGT") > 0:
            n_ambig += 1
        if not seq.startswith(CANONICAL_16S_START):
            all_canonical = False

    result["lengths"] = lengths
    result["length_range"] = f"{min(lengths)}-{max(lengths)}"
    result["n_ambiguous"] = n_ambig
    result["canonical_start"] = all_canonical

    if n_copies == 1:
        result["n_clusters"] = 1
        result["status"] = "OK_SINGLE"
        result["clusters"] = [{
            "cluster_id": 1,
            "n_copies": 1,
            "member_ids": [records[0].id],
            "contigs": [parse_contig(records[0].description)],
            "directions": [parse_direction(records[0].description)],
            "mean_length": lengths[0],
            "representative_id": records[0].id,
            "blast_hits": [],
        }]
        if do_blast:
            print(f"    cluster 1: BLASTing...", flush=True)
            result["clusters"][0]["blast_hits"] = blast_sequence(str(records[0].seq))
            blast_count[0] += 1
        return result

    # Align and compute pairwise identities
    aligned = align_sequences_muscle(records, muscle_bin)
    if aligned is None:
        result["status"] = "ALIGN_FAILED"
        return result

    pw_ids = pairwise_identity_from_alignment(aligned)

    if pw_ids:
        result["min_pairwise_identity"] = round(min(pw_ids.values()), 1)

    # Cluster
    record_ids = [r.id for r in records]
    clusters = cluster_by_identity(record_ids, pw_ids, identity_threshold)
    n_clusters = len(set(clusters.values()))
    result["n_clusters"] = n_clusters
    result["status"] = "OK" if n_clusters == 1 else "MULTI_ORGANISM"

    # Build per-cluster info with member IDs
    cluster_members = defaultdict(list)
    for rid, cid in clusters.items():
        rec = next(r for r in records if r.id == rid)
        cluster_members[cid].append(rec)

    cluster_details = []
    for cid in sorted(cluster_members):
        members = cluster_members[cid]
        contigs = [parse_contig(r.description) for r in members]
        directions = [parse_direction(r.description) for r in members]
        mlens = [len(r.seq) for r in members]
        rep = members[0]

        info = {
            "cluster_id": cid,
            "n_copies": len(members),
            "member_ids": [r.id for r in members],
            "contigs": sorted(set(contigs)),
            "directions": directions,
            "mean_length": round(sum(mlens) / len(mlens), 0),
            "representative_id": rep.id,
            "blast_hits": [],
        }

        if do_blast:
            if blast_count[0] > 0:
                time.sleep(blast_delay)
            print(f"    cluster {cid} ({len(members)} copies): BLASTing...", flush=True)
            info["blast_hits"] = blast_sequence(str(rep.seq))
            blast_count[0] += 1

        cluster_details.append(info)

    result["clusters"] = cluster_details
    return result


# --- Output writers ---

def write_qc_report(all_results, output_path):
    """Write per-sample QC summary to metrics/."""
    fields = [
        "sample", "n_copies", "length_range", "canonical_start",
        "n_ambiguous_seqs", "min_pairwise_identity",
        "n_clusters", "status",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for res in all_results:
            writer.writerow({
                "sample": res["sample"],
                "n_copies": res["n_copies"],
                "length_range": res["length_range"],
                "canonical_start": res["canonical_start"],
                "n_ambiguous_seqs": res["n_ambiguous"],
                "min_pairwise_identity": res["min_pairwise_identity"]
                    if res["min_pairwise_identity"] is not None else "N/A",
                "n_clusters": res["n_clusters"],
                "status": res["status"],
            })


def write_cluster_details(all_results, output_path):
    """Write per-cluster detail table to metrics/."""
    fields = [
        "sample", "cluster_id", "n_copies", "member_ids", "contigs",
        "mean_length", "top_blast_hit", "blast_pct_identity", "blast_query_coverage",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for res in all_results:
            for cl in res["clusters"]:
                top_hit = cl["blast_hits"][0] if cl["blast_hits"] else {}
                writer.writerow({
                    "sample": res["sample"],
                    "cluster_id": cl["cluster_id"],
                    "n_copies": cl["n_copies"],
                    "member_ids": "; ".join(cl["member_ids"]),
                    "contigs": "; ".join(str(c) for c in cl["contigs"]),
                    "mean_length": int(cl["mean_length"]),
                    "top_blast_hit": top_hit.get("description", "N/A"),
                    "blast_pct_identity": top_hit.get("pct_identity", "N/A"),
                    "blast_query_coverage": top_hit.get("query_coverage", "N/A"),
                })


def write_full_json(all_results, output_path):
    """Write full results as JSON to metrics/."""
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)


def write_cluster_fastas(all_results, cluster_dir, input_dir):
    """Write per-cluster FASTA files to clusters/."""
    cluster_dir.mkdir(parents=True, exist_ok=True)
    for res in all_results:
        sample = res["sample"]
        if res["n_copies"] == 0:
            continue

        records = {r.id: r for r in SeqIO.parse(
            str(input_dir / f"{sample}.16S.fasta"), "fasta")}

        for cl in res["clusters"]:
            cid = cl["cluster_id"]
            members = [records[mid] for mid in cl["member_ids"]]
            out_path = cluster_dir / f"{sample}.cluster_{cid}.fasta"
            SeqIO.write(members, str(out_path), "fasta")


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    cluster_dir = output_dir / "clusters"
    metrics_dir = output_dir / "metrics"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    fasta_files = sorted(input_dir.glob("*.16S.fasta"))
    if not fasta_files:
        print(f"Error: no *.16S.fasta files in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(fasta_files)} sample(s) in {input_dir}")
    print(f"Identity threshold: {args.identity_threshold}%")
    print(f"BLAST representatives: {args.blast_reps}")
    print(f"Cluster FASTAs: {cluster_dir}")
    print(f"QC metrics:     {metrics_dir}\n")

    blast_count = [0]
    all_results = []
    n_flagged = 0

    for fasta in fasta_files:
        sample = fasta.stem.replace(".16S", "")
        print(f"  {sample}:", end="", flush=True)

        res = analyze_sample(
            fasta, args.muscle, args.identity_threshold,
            args.blast_reps, args.blast_delay, blast_count,
        )
        all_results.append(res)

        flag = ""
        if res["status"] == "MULTI_ORGANISM":
            flag = " *** FLAGGED ***"
            n_flagged += 1
        elif res["status"] == "EMPTY":
            flag = " (empty)"

        copies_str = f"{res['n_copies']} copies"
        cluster_str = f"{res['n_clusters']} cluster(s)"
        min_id = res["min_pairwise_identity"]
        id_str = f"min_id={min_id}%" if min_id is not None else ""

        print(f" {copies_str}, {cluster_str}, {id_str}{flag}")

        if res["clusters"] and args.blast_reps:
            for cl in res["clusters"]:
                hits = cl["blast_hits"]
                if hits:
                    h = hits[0]
                    print(f"      cluster {cl['cluster_id']}: {h['description'][:70]} "
                          f"({h['pct_identity']}% id, {h['query_coverage']}% cov)")

    # Write outputs
    write_cluster_fastas(all_results, cluster_dir, input_dir)
    print(f"\nCluster FASTAs: {cluster_dir}/")

    report_path = metrics_dir / "qc_report.tsv"
    write_qc_report(all_results, report_path)
    print(f"QC report:      {report_path}")

    cluster_path = metrics_dir / "qc_clusters.tsv"
    write_cluster_details(all_results, cluster_path)
    print(f"Cluster details: {cluster_path}")

    json_path = metrics_dir / "qc_full.json"
    write_full_json(all_results, json_path)
    print(f"Full JSON:      {json_path}")

    print(f"\nDone: {len(all_results)} sample(s), {n_flagged} flagged as multi-organism.")


if __name__ == "__main__":
    main()
