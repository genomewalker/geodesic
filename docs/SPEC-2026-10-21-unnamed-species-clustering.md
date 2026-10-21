# Spec: cluster genomes without a species name by ANI instead of making each its own species

Status as of 2026-10-21: proposed, not implemented, no Codex stream launched. Companion:
`$REPOS/genopack/docs/SPEC-2026-10-21-unresolved-ranks.md` (stream A, stop inventing names; step C, apply the
overrides). This stream (B) must merge before any pack built with stream A is used.

## Why

`group_by_taxonomy` (`src/pipeline.cpp:56-80`) groups genomes by the full taxonomy string. A genome GTDB-Tk could not
place in a species carries `s__<accession>` (genopack fills it), so it forms its own group: the log calls them "natural
singletons" (`:69-71`, detected by `has_accession_species`, `src/taxonomy/species.cpp:16`). In the r232 catalogue that
is 500,846 of 1,313,026 reps, none ever compared with each other. The de novo clustering run on 2026-10-21
(`/projects/caeg/scratch/kbd606/tmp/selfnamed_clust/s2/`, `s2.py`) puts them in 173,057 species.

Once genopack stream A lands, the same genomes carry a bare `s__` instead. Grouping by string would then put every
unnamed genome of a genus into one taxon and dereplicate across species, which is worse. So this stream is a
prerequisite of stream A.

## Stream B: geodesic `feat/unnamed-species-clustering`

Repo `$REPOS/geodesic`, branch from `main`. Port the S2 rules (`selfnamed_clust/s2/s2.py`) into geodesic.

B1. Unnamed = species rank bare (`s__` / `S__` empty) or `has_accession_species` true (old packs). Named genomes are
grouped exactly as today.

B2. Anchors first. For each unnamed genome, compare with the named species reps of its family (skani, below). A
genome with ANI >= that rep's radius and max(AF_ref, AF_query) >= 50 joins that species (GTDB-Tk's own rule, top
hit among AF-passing refs). Radii: `gtdbtk_db/radii/gtdb_radii.tsv`, given with a new option `--species-radii`.
Without the option, radius = 95.

B3. Species clustering of the rest, GTDB rule, blocked by family and then a residual pass over centroids per phylum,
as `s2.py`:
- edge = ANI >= 95 and max(AF_ref, AF_query) >= 50;
- order: (existing rep first, quality tier HQ < MQ < LQ, -completeness, n_contigs, NCBI accession first, accession);
- greedy: a genome is a centroid iff no earlier centroid has an edge to it; members join the best (ANI, AF) centroid;
- per-rep radius = min(97, max(95, highest ANI with AF >= 50 to another final rep or a named rep)); a member below
  its rep's radius moves to the closest rep it passes, else becomes a singleton with its own radius;
- final check: no two final reps share an edge (searched pairs; cross-phylum pairs are not searched, log it).

B4. Each cluster becomes one `Taxon`, then dereplication per taxon runs unchanged. Placeholder name per cluster as S3
(`selfnamed_clust/s3/s3.py`): `s__<Genus> sp<9 digits>`, digits from the rep's NCBI accession, else
900000000 + sha1(rep accession) mod 10^8 with linear probing; names unique and absent from the named species. Genus =
majority over members with an assigned genus. A cluster with no member that has an assigned genus still forms a
`Taxon` (dereplication runs within it), but its `g__`/`s__`/`S__` stay bare and it gets no name and no row in
`species_assignments.tsv`; log the count (user 2026-10-21: the 22,417 genus-unresolved clusters are held out of
naming; naming them is a separate user decision). Write `species_assignments.tsv` (accession, cluster, rep, species,
genus, radius, ani_to_rep, af_to_rep, source anchor|cluster) next to the run outputs, in the 2-column-compatible form
`genopack taxonomy patch` reads, so the names can be written back to the pack.

B5. ANI engine: the skani binary (`skani triangle -E -s 90 --min-af 0` per block, `skani dist` for re-checks; ANI and
AF as in `s2/s2.sbatch`), run as a subprocess with a new option `--skani-bin` and genomes extracted from the pack per
block. This is new: `src/core/skani.hpp` is not skani but forwards to genopack's FracMinHash containment ANI
(`genopack::compute_ani`), which `geodesic ani` uses. Measured 2026-10-21: skani triangle on 35k genomes 72 s /
31.8 GB vs `geodesic ani` 1,468 s / 68.7 GB, and `geodesic ani` at ANI >= 95 + AF >= 0.5 missed 344 of 3,302 skani
edges on 1,189 genomes (`docs/SPEC-2026-10-21-ani-prefilter-stream-fmhg.md`). Switch to `compute_ani` only after
that spec's gates show edge parity.

B6. Option `--unnamed-species cluster|singleton` (default `cluster`); `singleton` keeps today's behaviour.

B gates (dandycomp chaos node or Slurm, 32 threads):
- `--unnamed-species singleton` output identical to `main` on a test set;
- parity with S2: on 3 families from `selfnamed_clust/s2/lists/` (one heavy, e.g. family 815 Bacteroidaceae, 15,379
  genomes; one mid; one with < 100 genomes), cluster membership equal to `s2/s2_clusters.tsv` restricted to them
  (report the adjusted Rand index and every differing genome; target ARI >= 0.99, differences explained by skani
  version 0.2.2 vs the build's);
- B2 on the 140 step 1 anchors (`selfnamed_clust/s2/anchors.tsv`): all join the same GTDB species;
- a test pack with bare `s__` (stream A) and the same pack with `s__<accession>` give the same clusters;
- wall time and peak RSS of the Bacteroidaceae family (S2 reference: 76 s, 17.3 GB on a chaos node, 48 threads);
- report table: case, wall, peak RSS, clusters, ARI, exit code.

## Not in scope

Species rescoring (S6(a)) and re-derep of the existing catalogue (S6(b), the user's call). Cross-phylum comparison.
