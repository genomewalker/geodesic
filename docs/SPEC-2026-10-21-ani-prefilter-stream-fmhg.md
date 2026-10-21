# Spec: faster `geodesic ani` (marker prefilter, streaming sketches, per-genome FMH sketches in the pack)

Status as of 2026-10-21: proposed, not implemented. Two Codex streams: A (genopack) then B (geodesic).
Items 1-2 (B1, B2) do not depend on A and can start at once; B3 needs A merged.

## Why

Measured 2026-10-21 on dandycomp08fl, 32 threads, random GTDB r232 reps from phylum 1239
(bench inputs in `/projects/caeg/scratch/kbd606/tmp/geoani_bench_gcBUVy/`: `acc_5000`, `acc_35000`, `acc`
= family 4881362, 1,189 genomes; matching skani sketch lists `sk_5000.list`, `sk_35000.list`):

| genomes | `geodesic ani` | skani triangle on existing sketches |
|---|---|---|
| 1,189 | 250 s (pair merge ~4 s) | 4.6 s |
| 5,000 | 450 s, 10.4 GB | 6 s, 4.5 GB |
| 35,000 | 1,468 s, 68.7 GB (~717 s load + sketch, ~751 s pairs) | 72 s, 31.8 GB |

The per-pair merge is already near the hardware limit (`genopack/src/skani.cpp:229-256`, ~0.6 ns per sketch
element). The cost is structural:
1. exhaustive all-pairs, no screen (`src/run_ani.cpp:115-127`): 612M merges at 35k for ~159k pairs at ANI >= 94;
2. all FASTA bytes held in RAM before sketching (`src/run_ani.cpp:65-71`), most of the 68.7 GB peak;
3. sketches rebuilt every run (`src/run_ani.cpp:73-82`): ~12 min of the 35k run.

Out of scope: the ANI/AF definition (containment, `skani.cpp:258-273`) stays as is; output columns and values
must not change except where a screen drops pairs below its threshold.

## Stream A: genopack, per-genome FracMinHash section `FMHG`

Repo `$REPOS/genopack`, branch `feat/fmhg-sketch` from `main` (the checkout is on
`feat/taxonomy-partition-contiguous` with uncommitted files; work in a worktree, do not touch that checkout).

A1. Section `SEC_FMHG = "FMHG"` in `include/genopack/format.hpp`, next to `SEC_FMHR` (`:32`). New
`include/genopack/fmhg.hpp` + `src/fmhg.cpp`, modelled on `fmhr.hpp`/`fmhr.cpp`:
- header: magic, k, c, n_genomes, offsets (64 bytes, `static_assert`);
- entries keyed by genome_id (sorted, binary-searchable): genome_id u32, n_hashes u32, hashes_offset u64;
- payload: concatenated sorted, deduplicated u64 hashes, exactly `build_sketch(acc, fasta, k, c).hashes`
  (`src/skani.cpp:84`), so a stored sketch is bit-identical to one built from the sequence;
- writer spills to a temp file like `FmhrWriter` (peak RAM = one batch, not the whole section);
- reader: zero-copy `std::span<const uint64_t>` per genome_id from the mmap; `has(genome_id)`, `k()`, `c()`.

A2. Producer: `genopack reindex --fmh-sketch [--fmh-k 21] [--fmh-c 125]` (`cli/main.cpp:1326` `cmd_reindex`),
sketching from the archive's own sequences in shard batches, threaded. Re-running replaces the section.
Register in `include/genopack/derivation.hpp` as derived from `SEC_SHRD`, like `SEC_FMHR` (`:37`).

A3. Merge / subset / rm: `subset` copies the kept genomes' FMHG rows, alongside its SKCH/QUAL/KMRX copying
(`src/subset.cpp:620-719`); `merge` drops FMHG with a warning naming `reindex --fmh-sketch` (add to the list at
`src/merger.cpp:191` handling); tombstoned genomes are skipped by readers.

A4. Storage cost is a user decision, not the stream's: ~8 B x genome_length / c (about 256 KB for a 4 Mb genome
at c = 125; ~25 % of a 2-bit packed genome; ~0.13 TB for the 500k-rep `reps_pack`). The section is opt-in;
nothing builds it by default.

A gates: the build and `ctest` in the build dir pass; new test: for 50 genomes of a test pack, FMHG span == `build_sketch`
output (k 21, c 125) element for element; `genopack verify` passes on a reindexed copy; subset round trip keeps
the rows.

## Stream B: geodesic `ani`

Repo `$REPOS/geodesic`, branch `perf/ani-prefilter-stream` from `main`. Touches `src/run_ani.cpp`,
`src/config.cpp`/`config.hpp` (options), `wiki/ANI-Computation.md` (numbers after the change).

B1. Marker screen (skani-style), new option `--screen FLOAT` (ANI %, default 80, `0` = exhaustive, today's
behaviour):
- markers = sketch hashes with `h % M == 0`, `M = 8` (density 1/(c x M) = 1/1000, skani's marker density),
  taken from the existing sorted sketch (no rehash);
- inverted index marker -> sorted genome indices (build by sorting (hash, idx) pairs; skip markers seen in more
  than `--max-marker-df` genomes, default 0 = no cap);
- per query, accumulate shared-marker counts over its markers (dense u32 counter array per thread, reset by
  touched list); a ref is a candidate if `shared / min(nm_q, nm_r) >= (screen/100)^k`; only candidates get the
  full `compute_ani` merge;
- self all-pairs: only pairs `qi < ri`, no visiting and skipping the other half (today `:122-123`).

B2. Streaming sketches: sketch inside the `visit_shard_batches` callback (`:67-70`), threaded over the batch,
move each FASTA in, release it after sketching; never hold all FASTA bytes. Peak RAM ~ all sketches + one batch.
Same `build_sketch` call, so sketches are identical.

B3 (after A merged): if the pack has FMHG with matching k and c, take sketches from it for every genome that
has a row and build only the missing ones from sequence; log `from pack N, built M`. `--no-pack-sketch` forces
B2. A k or c mismatch logs a warning and falls back.

B gates (all on dandycomp08fl or a Slurm node, 32 threads, outputs sorted before compare):
- `--screen 0` output byte-identical to current `main` on `acc` (1,189);
- `--screen 80` on `acc`: identical to `--screen 0` for every pair with ani >= 90; print recall at ani >= 85;
- `acc_35000` with `--screen 80 --min-af 0.27`: wall time and peak RSS (python `resource.getrusage`
  RUSAGE_CHILDREN; `/usr/bin/time` is missing on chaos nodes), target < 300 s and < 15 GB; pairs at ani >= 94
  identical to the 2026-10-21 run (`geo_35000.tsv`, made with `--min-af 0.27`, screen off);
- B3: same three gates with FMHG present give identical output, load phase in seconds.
- Report a table: case, wall, peak RSS, pairs, recall; one line per gate, exit codes.

## Not in these streams

Result-buffer over-reservation (`run_ani.cpp:110`, ~1.5 GB reserved per thread at 35k) and `pow` before the AF
filter (`skani.cpp:272`) are separate small fixes; do them only if B1 rewrites those lines anyway.
