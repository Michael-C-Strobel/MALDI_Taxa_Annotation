#!/usr/bin/env python3
"""Check whether each 16S sequence is found in its corresponding WGS assembly."""

import os
import sys

DATA_DIR = "data"
DIR_16S = os.path.join(DATA_DIR, "16S")
DIR_WGS = os.path.join(DATA_DIR, "WGS")

COMPLEMENT = str.maketrans("ATCGatcg", "TAGCtagc")


def reverse_complement(seq):
    return seq.translate(COMPLEMENT)[::-1]


def read_fasta(path):
    """Yield (header, sequence) tuples from a FASTA file."""
    header, parts = None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(parts).upper()
                header, parts = line[1:], []
            else:
                parts.append(line)
    if header is not None:
        yield header, "".join(parts).upper()


def main():
    samples_16s = sorted(f for f in os.listdir(DIR_16S) if f.endswith(".fasta"))

    all_pass = True
    for fname_16s in samples_16s:
        sample_id = fname_16s.split(".")[0]
        fname_wgs = f"{sample_id}.fasta"
        path_wgs = os.path.join(DIR_WGS, fname_wgs)

        if not os.path.exists(path_wgs):
            print(f"[SKIP] {sample_id}: no WGS file found")
            continue

        # Load all WGS contigs into one dict
        wgs_contigs = dict(read_fasta(path_wgs))

        # Check each 16S sequence
        queries = list(read_fasta(os.path.join(DIR_16S, fname_16s)))
        hits, misses = 0, 0

        for qheader, qseq in queries:
            rc_qseq = reverse_complement(qseq)
            found = False
            match_contig = None
            match_strand = None
            match_pct = None

            for contig_name, contig_seq in wgs_contigs.items():
                if qseq in contig_seq:
                    found = True
                    match_contig = contig_name
                    match_strand = "+"
                    match_pct = 100.0
                    break
                if rc_qseq in contig_seq:
                    found = True
                    match_contig = contig_name
                    match_strand = "-"
                    match_pct = 100.0
                    break

            if found:
                hits += 1
                print(f"  [MATCH]  {sample_id} | {qheader[:60]} | {match_strand} strand | {match_contig[:40]}")
            else:
                # Try allowing a small mismatch — find best substring match
                best_pct = 0.0
                best_contig = None
                best_strand = None
                qlen = len(qseq)
                for contig_name, contig_seq in wgs_contigs.items():
                    for strand_label, search_seq in [("+", qseq), ("-", rc_qseq)]:
                        # Slide along the contig
                        for i in range(len(contig_seq) - qlen + 1):
                            window = contig_seq[i:i + qlen]
                            matches = sum(a == b for a, b in zip(search_seq, window))
                            pct = 100.0 * matches / qlen
                            if pct > best_pct:
                                best_pct = pct
                                best_contig = contig_name
                                best_strand = strand_label
                            if pct >= 99.0:
                                break
                        if best_pct >= 99.0:
                            break
                    if best_pct >= 99.0:
                        break

                if best_pct >= 99.0:
                    hits += 1
                    print(f"  [~MATCH] {sample_id} | {qheader[:60]} | {best_strand} strand | {best_pct:.1f}% | {best_contig[:40]}")
                else:
                    misses += 1
                    all_pass = False
                    print(f"  [MISS]   {sample_id} | {qheader[:60]} | best {best_pct:.1f}% in {best_contig[:40] if best_contig else 'N/A'}")

        total = hits + misses
        status = "OK" if misses == 0 else "MISMATCH"
        print(f"[{status}] {sample_id}: {hits}/{total} 16S seqs found in WGS\n")

    if all_pass:
        print("=== ALL SAMPLES CONSISTENT ===")
    else:
        print("=== SOME MISMATCHES FOUND ===")


if __name__ == "__main__":
    main()
