# Derep output (`.gpd` archive)

When `geodesic derep --pack <gpk> --emit-gpd <path>` is given, geodesic writes a **Geodesic Derep Archive** (`.gpd`). The file is the durable record of a derep run: rep set, rep→cluster membership, rep-only embeddings, and a fingerprint of the source genopack archive that lets a reader detect when the underlying pack has drifted.

The reader is `genopack::DerepView`. Geodesic is the only writer.

This page is the byte-level spec; for the consumer-side API see the [genopack format / API docs](https://github.com/genomewalker/genopack).

---

## Output TSV files

Every derep run writes six tab-separated files under the output directory, prefixed
with `-p <prefix>` (all written unconditionally, `src/io/results_writer.cpp:219-226`).
These are the human-readable companions to the binary `.gpd`.

### `<prefix>_derep_genomes.tsv`

Every input genome mapped to its representative. One row per genome.

| Column | Description |
|--------|-------------|
| `accession` | Genome accession |
| `taxonomy` | Taxon lineage string |
| `representative` | `1` if this genome is a representative, `0` if a clustered member |
| `cluster_rep` | Accession of the representative this genome maps to (its own when it is a rep) |
| `nn_dist` | Angular distance from the member to its assigned representative (`0` for reps) |
| `sketch_fill` | OPH sketch occupancy fraction of the genome (`1.0` default) |

### `<prefix>_results.tsv`

Per-taxon summary. One row per taxon.

| Column | Description |
|--------|-------------|
| `taxonomy` | Taxon lineage string |
| `method` | Selection method: `geodesic`, `geodesic-self-rep`, `singleton`, or `fixed` |
| `n_genomes` | Input genomes in the taxon |
| `n_genomes_derep` | Representatives retained |
| `communities` | Number of communities detected in the taxon |
| `weight` | Reserved; always emitted as `NA` (`results_writer.cpp:159`) |

### `<prefix>_stats.tsv`

Per-taxon pipeline counts and MST diagnostics. One row per non-failed taxon.

| Column | Description |
|--------|-------------|
| `taxonomy` | Taxon lineage string |
| `method` | Selection method (as above) |
| `n_input` | Input genomes |
| `n_preflight_excluded` | Genomes dropped in preflight (before embedding) |
| `n_quality_floor_excluded` | Genomes dropped by the quality floor (e.g. `--skip-lq`, `--min-completeness`) |
| `n_outliers_excluded` | Outliers removed from representative selection |
| `n_outliers_retained` | Outliers flagged but still eligible as reps |
| `n_failed` | Genomes that failed to resolve/embed |
| `n_embedded` | Genomes embedded = `n_input − n_preflight_excluded − n_failed` |
| `n_representatives` | Representatives retained |
| `rep_fraction` | `n_representatives / n_input` |
| `mst_p90_edge` | 90th-percentile MST edge length (angular distance) |
| `mst_true_max` | Largest raw MST edge (before bridge-conditioning) |
| `ani_threshold_used` | ANI threshold applied for this taxon |
| `n_outliers_fragmented` | Outliers whose `flag_reason` includes `fragmented` |
| `n_outliers_size` | Outliers whose `flag_reason` includes `size_outlier` (non-fragmented) |
| `n_outliers_distance` | Remaining outliers (nn/distance-driven) |

### `<prefix>_diversity_stats.tsv`

Per-taxon coverage (member→rep) and diversity (rep↔rep) ANI metrics. One row per taxon with diversity stats.

| Column | Description |
|--------|-------------|
| `taxonomy` | Taxon lineage string |
| `method` | Selection method |
| `n_genomes` | Input genomes |
| `n_representatives` | Representatives retained |
| `reduction_ratio` | Fraction of genomes removed = `1 − n_representatives / n_genomes` |
| `runtime_seconds` | Per-taxon processing time |
| `coverage_mean_ani` | Mean ANI of members to their assigned representative |
| `coverage_min_ani` | Robust worst-case coverage ANI (p5 ANI, from p95 distance) |
| `coverage_max_ani` | Best-covered ANI (p95 ANI, from p5 distance) |
| `coverage_below_99` | Count of member→rep pairs with ANI < 99% |
| `coverage_below_98` | Count of member→rep pairs with ANI < 98% |
| `coverage_below_97` | Count of member→rep pairs with ANI < 97% |
| `coverage_below_95` | Count of member→rep pairs with ANI < 95% |
| `diversity_mean_ani` | Mean pairwise ANI among representatives |
| `diversity_min_ani` | ANI of the most divergent rep pair (p95 distance) |
| `diversity_max_ani` | ANI of the most similar rep pair (p5 distance) |
| `diversity_ani_range` | `diversity_max_ani − diversity_min_ani` |
| `diversity_n_pairs` | Number of rep–rep pairs compared |
| `n_outliers_excluded` | Outliers removed from selection |
| `n_outliers_retained` | Outliers flagged but retained |

### `<prefix>_failed.tsv`

Genomes that could not be embedded/clustered. One row per failure.

| Column | Description |
|--------|-------------|
| `accession` | Genome accession |
| `taxonomy` | Taxon lineage string |
| `file` | Source file path (if known) |
| `reason` | Failure reason, e.g. `accession not found …` (`NA` when unset) |

Note: genomes with a resolvable accession but no sketch ("sketch not found …") are
**not** listed here — they are kept as self-representatives.

### `<prefix>_outliers.tsv`

All flagged outlier candidates. One row per flagged genome. The per-genome signals
reuse the definitions in [Outlier detection](Outlier-Detection.md) and
[Contamination detection](Contamination.md).

| Column | Description |
|--------|-------------|
| `taxonomy` | Taxon lineage string |
| `accession` | Genome accession |
| `category` | `misassigned` (nn/distance), `low_quality` (`fragmented:pre_filter`), or `contaminated` (GUNC) |
| `nn_outlier` | Boolean: `isolation_score` exceeds the taxon threshold |
| `isolation_score` | Mean angular distance to the k nearest neighbours |
| `kmer_div_zscore` | Occupied-OPH-bins-per-kbp z-score (informational only) |
| `genome_size_zscore` | Z-score of genome size within the taxon |
| `centroid_distance` | Angular distance from the taxon centroid |
| `anomaly_score` | Currently equal to `isolation_score` |
| `genome_length_bp` | Genome length in base pairs |
| `n_contigs` | Contig count |
| `margin_to_threshold` | `isolation_score − threshold` (positive = above threshold) |
| `flag_reason` | Raw flag string (e.g. `nn_outlier`, `nn_outlier+size_outlier`, `:fragmented`) |
| `excluded` | Boolean: `1` = removed from selection, `0` = flagged only, still eligible |

---

## File layout

A `.gpd` is a **single file**, not a directory:

```
[GpdFileHeader               64 B]
[Section blobs ...] (HDR · ASTR · ASOF · ARMP · RTBL · G2RM · EMBD)
[TOC                                ]
[GpdTailLocator             16 B   ]
```

All multi-byte integers are little-endian; section payloads are 8-byte aligned (zero padding). Sections may be zstd-compressed (`flags & 1`); the reader transparently decompresses on first access.

### `GpdFileHeader` — 64 bytes (offset 0)

| Offset | Size | Field | Description |
|---|---|---|---|
| 0  | 4 B  | `magic`        | `'GPDF'` (`0x46445047`) |
| 4  | 2 B  | `format_major` | 1 |
| 6  | 2 B  | `format_minor` | 0 |
| 8  | 8 B  | `toc_offset`   | Byte offset of TOC section |
| 16 | 8 B  | `toc_size`     | TOC byte size |
| 24 | 40 B | _reserved_     | Zero-padded |

### `GpdTailLocator` — 16 bytes (last bytes of file)

| Offset | Size | Field | Description |
|---|---|---|---|
| 0  | 8 B | `toc_offset` | Duplicate of header `toc_offset` (corruption check) |
| 8  | 4 B | `magic`      | `'GPDT'` (`0x54445047`) |
| 12 | 4 B | `crc32`      | crc32 of TOC descriptor array (same value as `crc32_of_following_descs` in the TOC payload header) |

### Section magics

| Magic | Name | Description |
|---|---|---|
| `GHDR` | HDR  | Identity, params, source-part fingerprints (always uncompressed) |
| `GAST` | ASTR | Concatenated accession string pool, ASCIIbetically sorted |
| `GASO` | ASOF | `uint32_t offsets[n_genomes+1]` boundaries into ASTR |
| `GARM` | ARMP | Optional open-addressed accession→ordinal hash map |
| `GRTB` | RTBL | Rep table (one entry per representative) |
| `G2RM` | G2RM | `uint32_t rep_id[n_genomes]` (sentinels: `0xFFFFFFFE` unclustered, `0xFFFFFFFF` tombstoned) |
| `GEMB` | EMBD | Rep-only embedding matrix (default f16 × dim, typically 256) |
| `GTOC` | TOC  | Section descriptor table |

### TOC section

```c
struct GpdSectionDesc {           // 56 bytes
    uint32_t type;                // GPD_SEC_*
    uint32_t flags;               // bit0 = zstd-compressed payload
    uint64_t file_offset;         // absolute byte offset to section payload
    uint64_t compressed_size;
    uint64_t uncompressed_size;
    uint64_t section_id;          // unique within file, monotonic, starts at 1
    uint64_t reserved[2];
};

// TOC payload layout:
//   uint32_t magic = 'GTOC'
//   uint32_t n_sections
//   uint32_t crc32_of_following_descs   // crc32 over the descriptor array bytes only
//   uint32_t pad
//   GpdSectionDesc descs[n_sections]
```

---

## HDR section (always present, always uncompressed)

```c
struct GpdHeader {
    uint32_t magic;            // 'GPDH' = 0x48445047
    uint16_t format_major;     // 1
    uint16_t format_minor;     // 0
    uint64_t created_at_unix;
    uint8_t  run_id[16];       // UUID v4 — unique per derep run
    uint16_t n_parts;          // source pack parts at derep time
    uint16_t embedding_dim;    // typically 256
    uint8_t  embedding_dtype;  // 0=f32, 1=f16
    uint8_t  pad0[3];
    uint64_t n_genomes;        // total genomes covered (= sum of part live_counts)
    uint64_t n_reps;
    uint64_t n_unclustered;
    // followed by:
    //   GpdSourcePart parts[n_parts];
    //   GpdDerepParams params;
};

struct GpdSourcePart {           // 48 bytes
    uint8_t  archive_uuid[16];   // from genopack archive header
    uint64_t generation;
    uint64_t n_genomes_total;
    uint64_t n_genomes_live;
    uint64_t accession_set_hash; // xxh3-64 of '\n'-joined sorted live accessions
};

struct GpdDerepParams {
    uint8_t  n_kmer_sizes;
    uint8_t  kmer_sizes[7];      // up to 7; tail zero-padded
    uint32_t sketch_size;
    uint64_t sig1_seed;          // = --seed   (default 42)
    uint64_t sig2_seed;          // = --seed+1 (default 43)
    float    jaccard_thresh;
    uint16_t geodesic_ver_len;
    uint8_t  pad1[2];
    char     geodesic_ver[];     // not null-terminated; padded to 8
};
```

### `accession_set_hash`

The operational identity of a source part: take all **live** (non-tombstoned) accessions, sort ASCIIbetically, join with `\n` (no trailing newline), hash with xxh3-64. Geodesic computes this from the `ArchiveSetReader` at derep time; `DerepView::check(pack)` recomputes it from the live pack to validate. Both implementations must produce identical bytes for the same live set — be careful about UTF-8 normalisation, trailing newlines, and locale-dependent sorting.

---

## ASTR section — accession string pool

Concatenated accession strings, sorted ASCIIbetically, no separators (offsets give boundaries). Includes all genomes live at derep time (reps + members + unclustered). zstd-compressed when `flags & 1`.

## ASOF section — accession offsets

```c
uint32_t magic = 'GASO';
uint32_t n_genomes;
uint64_t pad;
uint32_t offsets[n_genomes + 1];   // offsets[i+1] - offsets[i] = length of accession i
```

`acc_string(ord)` = `string_view(astr + offsets[ord], offsets[ord+1] - offsets[ord])`.

`acc_to_ord(s)`: binary search over `[0, n_genomes)` comparing against `acc_string(mid)`. ~22 comparisons for 5M genomes.

## ARMP section — accession hash map (optional but recommended)

Open-addressed hash table for O(1) `acc_to_ord`. Geodesic emits this when `cfg.emit_armp = true` (default).

```c
uint32_t magic = 'GARM';
uint32_t n_buckets;       // power of two, load factor target 0.7
uint32_t hash_seed;       // reserved; currently always 0 (XXH3_64bits is called unseeded)
uint32_t pad;
struct GpdArmpEntry {
    uint64_t hash;        // high 64 bits of XXH3_128bits(accession)
    uint32_t ordinal;     // index into ASOF; 0xFFFFFFFF = empty bucket
    uint32_t pad;
} entries[n_buckets];
```

Lookup: bucket = `hash & (n_buckets - 1)`. Linear probe on collision. Verify by comparing the accession string at the candidate ordinal. If absent, reader falls back to ASOF binary search.

## RTBL section — rep table

```c
uint32_t magic = 'GRTB';
uint32_t n_reps;
uint64_t pad;
struct GpdRepEntry {           // 24 bytes
    uint32_t rep_acc_ord;      // index into ASOF (the rep is itself a genome)
    uint32_t cluster_size;     // including the rep itself; ≥ 1
    uint64_t source_locator;   // (part_idx<<48)|local_genome_id at derep time, advisory
    uint16_t sketch_kmer;      // which k-mer size produced the winning sketch
    uint8_t  flags;            // bit0=has_embedding (must be 1 in v1)
    uint8_t  pad;
    uint32_t reserved;         // reserved (zero)
} entries[n_reps];
```

Reps are sorted by `rep_acc_ord` ascending so `rep_id ↔ rep_acc_ord` is strictly monotonic and embedding-row order matches rep-id order for cache locality in cosine-similarity sweeps.

## G2RM section — genome→rep map

```c
uint32_t magic = 'G2RM';
uint32_t n_genomes;
uint64_t pad;
uint32_t rep_id[n_genomes];    // sentinels: 0xFFFFFFFE = unclustered
                               //            0xFFFFFFFF = tombstoned-at-derep-time
```

Indexed by genome ordinal (= index into ASOF). For genome `i`, `rep_id[i]` is the index into RTBL; for a rep, `rep_id[rep.rep_acc_ord] == self_rep_id`.

## EMBD section — embeddings (rep-only)

```c
uint32_t magic = 'GEMB';
uint16_t dim;                  // typically 256
uint8_t  dtype;                // 0=f32, 1=f16
uint8_t  pad0;
uint32_t n_reps;
uint32_t pad1;
// followed by: dtype_bytes(dtype) * dim * n_reps  raw matrix
//   row i corresponds to rep_id i (RTBL ordering)
```

Default dtype is **f16** (~270 MB for 546k × 256). Loss is acceptable for cosine search; mapping pipelines that need f32 can request via `--emit-gpd-embedding-dtype f32` at write time (when exposed; currently fixed at f16 from geodesic).

---

## Staleness detection (reader side)

`genopack::DerepView::check(pack)` returns one of:

| Level | Trigger |
|---|---|
| `Valid`                      | All parts: same `accession_set_hash`, `archive_uuid`, `generation` |
| `LayoutChangedSameLiveSet`   | Same live accession set, but UUID/generation differs (e.g. pack was repacked) |
| `StaleNewGenomes`            | Pack contains accessions absent from the .gpd |
| `StaleTombstones`            | .gpd contains accessions no longer live in the pack |
| `Mismatch`                   | Structural difference (missing part, etc.) |

Mapping pipelines should treat anything other than `Valid` (and possibly `LayoutChangedSameLiveSet`) as a signal to re-run derep.

---

## Versioning

`format_major` bumps for incompatible changes. `format_minor` bumps for additive changes (e.g. new optional sections). Readers must reject unknown `format_major` and accept higher `format_minor` (ignoring unknown sections).

The format is committed from `1.0`. Pre-1.0 geodesic releases may bump `format_minor` aggressively as new optional sections are added.
