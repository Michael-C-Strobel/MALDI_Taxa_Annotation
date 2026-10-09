#!/usr/bin/env python3
"""Summarize ONT sequencing stats per sample and cross-reference with assembly metrics."""

import csv
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEQ_DIR = os.path.join(BASE_DIR, "data", "sequencing_summaries")
ASM_METRICS = os.path.join(BASE_DIR, "data", "Assembly Metrics.tsv")
OUT_FILE = os.path.join(SEQ_DIR, "summarized", "summary.tsv")


def load_assembly_sizes(path):
    sizes = {}
    contigs = {}
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["Assembly"]:
                sizes[row["Assembly"]] = int(row["Total length"])
                contigs[row["Assembly"]] = int(row["# contigs"])
    return sizes, contigs


def summarize_sample(path):
    lengths = []
    quals = []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            lengths.append(int(row["sequence_length_template"]))
            quals.append(float(row["mean_qscore_template"]))

    lengths.sort()
    n = len(lengths)
    total_bp = sum(lengths)
    median_len = lengths[n // 2]
    mean_q = sum(quals) / n

    cumsum = 0
    n50 = 0
    for l in reversed(lengths):
        cumsum += l
        if cumsum >= total_bp / 2:
            n50 = l
            break

    return {
        "read_count": n,
        "total_bp": total_bp,
        "median_read_length": median_len,
        "read_n50": n50,
        "mean_qscore": round(mean_q, 1),
    }


def main():
    asm_sizes, asm_contigs = load_assembly_sizes(ASM_METRICS)

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    fields = [
        "sample",
        "read_count",
        "total_yield_mb",
        "median_read_length",
        "read_n50",
        "mean_qscore",
        "assembly_size_mb",
        "assembly_contigs",
        "coverage_x",
    ]

    rows = []
    for fname in sorted(os.listdir(SEQ_DIR)):
        if not fname.endswith(".tsv"):
            continue
        sample = fname.split(".")[0]
        path = os.path.join(SEQ_DIR, fname)
        stats = summarize_sample(path)

        asm_size = asm_sizes.get(sample, 0)
        asm_ctg = asm_contigs.get(sample, 0)
        coverage = stats["total_bp"] / asm_size if asm_size else 0

        rows.append({
            "sample": sample,
            "read_count": stats["read_count"],
            "total_yield_mb": round(stats["total_bp"] / 1e6, 1),
            "median_read_length": stats["median_read_length"],
            "read_n50": stats["read_n50"],
            "mean_qscore": stats["mean_qscore"],
            "assembly_size_mb": round(asm_size / 1e6, 1),
            "assembly_contigs": asm_ctg,
            "coverage_x": round(coverage, 1),
        })

    with open(OUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} samples to {OUT_FILE}")
    for row in rows:
        print(f"  {row['sample']}: {row['read_count']:,} reads, {row['total_yield_mb']} Mb yield, {row['coverage_x']}x coverage")


if __name__ == "__main__":
    main()
