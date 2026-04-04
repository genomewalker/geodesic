#!/usr/bin/env python3
"""
Normalize a geodesic TSV (accession, taxonomy, file) to canonical 10-rank format.

Rules (mirrors normalize_taxonomy() in src/pipeline.cpp):
  - Canonical ranks: d__, l__, k__, p__, c__, o__, f__, g__, s__, S__
  - 7-rank GTDB (missing l__/k__): insert l__/k__ derived from d__, add S__
  - 9-rank GTDB (has l__/k__, missing S__): add S__
  - Empty s__ stubs: replaced with accession-derived species stem
  - S__ always derived from s__ (stripped of prefix)
  - Non-d__ taxonomy (e.g. "Unclassified Bacteria"): synthetic per-accession taxonomy

Usage:
  normalize-taxonomy-tsv.py <input.tsv> <output.tsv>
"""

import sys
import re

RANKS = ["d__", "l__", "k__", "p__", "c__", "o__", "f__", "g__", "s__", "S__"]
RANK_SET = set(RANKS)


def species_stem(acc: str) -> str:
    if acc.startswith("RS_") or acc.startswith("GB_"):
        acc = acc[3:]
    if acc.startswith("GCF_") or acc.startswith("GCA_"):
        dot = acc.rfind(".")
        if dot != -1:
            return acc[:dot]
    return acc


def normalize(taxonomy: str, accession: str) -> str:
    stem = species_stem(accession)

    # Non-GTDB (doesn't start with d__)
    if len(taxonomy) < 3 or not taxonomy.startswith("d__"):
        return (f"d__Unclassified;l__Unclassified;k__Unclassified;"
                f"p__Unclassified;c__Unclassified;o__Unclassified;"
                f"f__Unclassified;g__Unclassified;s__{stem};S__{stem}")

    # Parse existing tokens into prefix → full token
    rank_map = {}
    for token in taxonomy.split(";"):
        if len(token) >= 3 and token[1] == "_" and token[2] == "_":
            rank_map[token[:3]] = token

    # Propagate parent → child for missing/empty ranks
    def propagate(child: str, parent: str):
        if child not in rank_map or rank_map[child] == child:
            if parent in rank_map:
                rank_map[child] = child + rank_map[parent][3:]

    propagate("l__", "d__")
    propagate("k__", "l__")
    propagate("p__", "k__")
    propagate("c__", "p__")
    propagate("o__", "c__")
    propagate("f__", "o__")
    propagate("g__", "f__")

    # s__: empty stub → accession stem
    if "s__" not in rank_map or rank_map["s__"] == "s__":
        rank_map["s__"] = "s__" + stem

    # S__ always from s__
    rank_map["S__"] = "S__" + rank_map["s__"][3:]

    return ";".join(rank_map.get(r, r) for r in RANKS)


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.tsv> <output.tsv>", file=sys.stderr)
        sys.exit(1)

    in_path, out_path = sys.argv[1], sys.argv[2]
    n_7rank = n_9rank = n_10rank = n_synthetic = 0

    with open(in_path, "r", buffering=1 << 20) as fin, \
         open(out_path, "w", buffering=1 << 20) as fout:

        header = fin.readline()
        fout.write(header)

        for i, line in enumerate(fin, 1):
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) < 3:
                fout.write(line)
                continue

            acc, tax, path = parts
            n_ranks = tax.count(";") + 1 if tax else 0

            norm = normalize(tax, acc)
            fout.write(f"{acc}\t{norm}\t{path}\n")

            if n_ranks == 7:
                n_7rank += 1
            elif n_ranks == 9:
                n_9rank += 1
            elif n_ranks == 10:
                n_10rank += 1
            else:
                n_synthetic += 1

            if i % 500_000 == 0:
                print(f"  {i:,} rows processed...", file=sys.stderr)

    total = n_7rank + n_9rank + n_10rank + n_synthetic
    print(f"Done: {total:,} rows", file=sys.stderr)
    print(f"  7-rank (old GTDB, l__/k__ added): {n_7rank:,}", file=sys.stderr)
    print(f"  9-rank (l__/k__ present, S__ added): {n_9rank:,}", file=sys.stderr)
    print(f"  10-rank (already canonical): {n_10rank:,}", file=sys.stderr)
    print(f"  synthetic (non-d__, unclassified): {n_synthetic:,}", file=sys.stderr)


if __name__ == "__main__":
    main()
