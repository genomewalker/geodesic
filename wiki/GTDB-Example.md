# Full GTDB Workflow: Archive Creation to Dereplication

This walkthrough covers the complete pipeline for GTDB r232 (or any large NCBI/GTDB
collection): building a genopack archive, normalising taxonomy, running geodesic
dereplication, and interpreting the output.

---

## Prerequisites

- `genopack` and `geodesic` binaries on `$PATH`
- GTDB genome assemblies (FASTA, one file per genome)
- GTDB metadata TSV (`bac120_metadata_r232.tsv` / `ar53_metadata_r232.tsv`) — for taxonomy and quality scores
- NCBI taxdump (optional but recommended for 10-rank normalisation): `nodes.dmp` + `names.dmp`

---

## 1. Prepare the input TSV

genopack `build` requires a tab-separated file with at least three columns:

```
accession   file_path   taxonomy
```

Additional optional columns: `completeness`, `contamination` (CheckM2 values, 0–100).

### From GTDB metadata

```bash
# Extract accession, file path, and GTDB lineage from the metadata TSV
# Assumes assemblies are at /data/genomes/{accession}_genomic.fna.gz
awk -F'\t' 'NR>1 {
    gsub(/^GB_|^RS_/, "", $1)          # strip GB_/RS_ prefix
    print $1 "\t" "/data/genomes/" $1 "_genomic.fna.gz" "\t" $17 "\t" $2 "\t" $3
}' bac120_metadata_r232.tsv > genomes_raw.tsv
# Columns: accession  file_path  gtdb_taxonomy  completeness  contamination
```

> Adjust column indices to your metadata version. `$17` is the GTDB lineage in r232;
> `$2` and `$3` are CheckM2 completeness and contamination.

---

## 2. Normalise taxonomy

GTDB lineages (`d__;p__;c__;o__;f__;g__;s__`) have 7 ranks. genopack's taxonomy
normaliser expands them to 10 canonical ranks (domain → phylum → class → order →
family → tribe → genus → species → strain → accession), filling missing ranks from
NCBI taxdump where available.

```bash
genopack taxonomy normalize \
    -i genomes_raw.tsv \
    -o genomes_norm.tsv \
    --ncbi-taxdump /data/taxdump/
```

The output `genomes_norm.tsv` has the same column order with the taxonomy column
replaced by the normalised 10-rank lineage. Inspect a few lines:

```bash
head -3 genomes_norm.tsv | cut -f3
# d__Bacteria;p__Pseudomonadota;c__Gammaproteobacteria;o__Enterobacterales;...
```

If you don't have NCBI taxdump, omit `--ncbi-taxdump`; ranks not present in GTDB
will be filled with synthetic placeholders (`t__`, `st__`).

---

## 3. Build the genopack archive

```bash
genopack build \
    -i genomes_norm.tsv \
    -o gtdb_r232.gpk \
    -t 48 \
    -z 6 \
    --sketch-kmers 16,21 \
    --taxon-rank g
```

Key flags:

| Flag | Value | Why |
|------|-------|-----|
| `-t 48` | 48 I/O threads | Tune to your node (I/O-bound; more threads = faster for NFS) |
| `-z 6` | zstd level 6 | Good compression/speed trade-off for genome FASTA |
| `--sketch-kmers 16,21` | two k-mer sizes | k=16 for broad genus-level ANI, k=21 for species-level; geodesic auto-picks per taxon |
| `--taxon-rank g` | genus | Groups genomes into per-genus shards — critical for NFS read locality during derep |
| `--cidx` | (omit for GTDB) | Opt in to building the CIDX contig index; off by default. Adds ~20 min for GTDB scale. |

For a full GTDB r232 (~5.2 M genomes) this takes roughly **4–6 hours** on a 48-thread
NFS node at ~250 MB/s. The resulting archive is ~1.5–2 TB compressed.

### Multipart build (distributed)

For very large collections or when source genomes are spread across multiple nodes,
partition first and build parts in parallel:

```bash
# Split into 8 parts, balanced at genus rank
genopack taxonomy partition \
    -i genomes_norm.tsv \
    -n 8 \
    -o parts/ \
    -r g

# Build each part on a separate node (submit as array job)
for i in $(seq 0 7); do
    genopack build \
        -i parts/part_${i}.tsv \
        -o /scratch/part_${i}.gpk \
        -t 24 -z 6 --sketch-kmers 16,21 --taxon-rank g &
done
wait

# Merge into a single multipart set (or use the parts directory directly)
# geodesic accepts a directory of part_*.gpk archives transparently
ls /scratch/part_*.gpk -d   # each is a directory itself
```

---

## 4. Verify the archive

```bash
genopack verify gtdb_r232.gpk
genopack stat gtdb_r232.gpk
```

`stat` reports genome count, section sizes, SKCH layout, and sketch preload cost —
useful for capacity planning before running geodesic.

---

## 5. Prepare the accession list

geodesic takes a plain accession list (one per line). You can generate it from the
normalised TSV or directly from the archive:

```bash
cut -f1 genomes_norm.tsv > accessions.txt
# or equivalently from the archive:
# genopack stat gtdb_r232.gpk --list-accessions > accessions.txt
```

Optionally filter by quality before dereplication:

```bash
awk -F'\t' '$4 >= 90 && $5 <= 5' genomes_norm.tsv | cut -f1 > accessions_hq.txt
```

---

## 6. Run dereplication

```bash
geodesic derep \
    -g accessions.txt \
    --pack gtdb_r232.gpk \
    --threads 24 \
    -p gtdb_r232 \
    --emit-gpd gtdb_r232.gpd \
    --lock-output gtdb_r232.lock
```

With CheckM2 quality scores for quality-weighted representative selection:

```bash
geodesic derep \
    -g accessions.txt \
    --pack gtdb_r232.gpk \
    --threads 24 \
    -p gtdb_r232 \
    --checkm2 checkm2_quality.tsv \
    --emit-gpd gtdb_r232.gpd \
    --lock-output gtdb_r232.lock
```

The CheckM2 TSV should have columns `accession`, `completeness`, `contamination`
(standard `quality_report.tsv` output works directly).

**Expected runtime and memory** (GTDB r232, 5.2 M genomes, 24 threads):

| Metric | Value |
|--------|-------|
| Wall time | ~65 min |
| Peak RSS | ~63 GB |
| Representatives | ~886k (83% reduction) |

---

## 7. Output files

| File | Contents |
|------|----------|
| `gtdb_r232_results.tsv` | Per-taxon summary: `taxonomy`, `method`, `n_genomes`, `n_genomes_derep`, `communities`, `weight` |
| `gtdb_r232_derep_genomes.tsv` | Per-genome: `accession`, `taxonomy`, `representative` (0/1), `cluster_rep`, `nn_dist`, `sketch_fill` |
| `gtdb_r232_diversity_stats.tsv` | Per-taxon coverage and diversity metrics |
| `gtdb_r232_stats.tsv` | Per-taxon pipeline stats (MST edges, ANI threshold, outlier counts) |
| `gtdb_r232_outliers.tsv` | Contamination candidates flagged by the MAD-based estimator |
| `gtdb_r232_failed.tsv` | Unresolvable accessions (`accession`, `taxonomy`, `file`, `reason`). Genomes that resolve in the pack but lack a sketch are kept as self-representative singletons, not failed |
| `gtdb_r232.gpd` | Geodesic Derep Archive — machine-readable rep set consumed by `genopack::DerepView` |
| `gtdb_r232.lock` | JSON provenance file; input to `geodesic update` for incremental re-runs |

See [Derep Output](Derep-Output) for full column descriptions.

---

## Scale example (GTDB r232)

A full end-to-end run over all of GTDB r232, built as a 10-part multipack.

**Input:** 9,530,982 genomes across 695,904 taxa.

| Step | Layout | Resources | Time | Peak RAM |
|------|--------|-----------|------|----------|
| Build | per part, 10-way array | 24 CPU, 80 GB | ~5.4–7.8 h | 40.7 GB |
| Check `--recompute` | per part, 10-way array | 24 CPU, 40 GB | ~1.9–4.2 h | 24.2 GB |
| Derep | single node | 24 CPU, 192 GB | 7.5 h | 128 GB |

**Result:** 1,300,340 representatives — 7.33× reduction (13.6% retained), 0 failures.

**Storage:** 5.77 TB pack vs 8.6 TB gzipped FASTA (1.5×), ~27 TB uncompressed (~4.7×).
The pack also holds sketches, QUAL, GSTX, taxonomy, and indexes.

---

## 8. (Optional) Validate the sketches and spot-check ANI

Before trusting a large run, confirm that the pack's OPH sketches track exact ANI
on a sample of pairs:

```bash
geodesic validate-ani \
    -g accessions.txt \
    --pack gtdb_r232.gpk \
    -n 2000 \
    -o gtdb_r232_ani_validation.tsv \
    -t 24
```

Inspect the per-k error columns (`err_k16`, `err_k21`, …) — near the 95% ANI
threshold, dense-sketch error should sit well below 0.1 ANI points. To compute
exact pairwise FracMinHash ANI for a specific set of representatives (e.g. to QC a
cluster), use `geodesic ani`:

```bash
geodesic ani --ql cluster_members.txt --pack gtdb_r232.gpk -t 24 -o cluster_ani.tsv
```

See [ANI Computation](ANI-Computation) for full details on both subcommands.

---

## 9. Incremental update (new release)

When a new GTDB release adds genomes, add them to the archive and re-run only
affected taxa — no full re-derep required:

```bash
# 1. Add new genomes to the archive
genopack add gtdb_r232.gpk -i new_genomes_norm.tsv

# 2. Rebuild the sketch index for the new entries
genopack reindex gtdb_r232.gpk --skch --sketch-kmers 16,21 --skch-threads 24

# 3. Update the accession list
cat accessions.txt new_accessions.txt | sort -u > accessions_r233.txt

# 4. Run incremental derep — only taxa with new members are recomputed
geodesic update \
    -g accessions_r233.txt \
    --lock gtdb_r232.lock \
    --pack gtdb_r232.gpk \
    --threads 24 \
    --geodf-output gtdb_r233.geodf \
    --lock-output gtdb_r233.lock
```

---

## 10. Using the derep archive downstream

```cpp
#include <genopack/derep_view.hpp>

genopack::DerepView derep;
derep.open("gtdb_r232.gpd");

// Check freshness against the current pack
genopack::ArchiveSetReader pack("gtdb_r232.gpk");
auto staleness = derep.check(pack);
// DerepStaleness::Valid → safe to use

// Look up a genome
auto status = derep.status_for_accession("GCA_000008085.1");
if (status.kind == genopack::RepStatus::Kind::Representative) {
    std::vector<float> emb(derep.embedding_dim());
    derep.embedding_for_rep(status.rep_id, emb);
    // emb is the 256-dim f32 embedding of this representative
}

auto stats = derep.stats();
// stats.n_reps, stats.n_genomes_indexed, stats.n_unclustered
```

---

## Tips and common pitfalls

**Taxonomy normalisation is required for per-taxon dereplication.** geodesic reads
taxonomy from the pack's TAXN section; if genomes lack a lineage string the entire
collection is treated as a single taxon, which defeats the per-taxon algorithm and
will be very slow.

**Sketch preload memory scales with sketch budget, not genome count.** The default
sketch budget is 64 GB; geodesic loads as many genus-level sketches as fit, then
processes in waves. You can lower it via the `GEODESIC_BUCKET_RAM_GB` environment
variable (or `GEODESIC_BUCKET_RAM_MB` for finer control) if RAM is limited, e.g.
`GEODESIC_BUCKET_RAM_GB=32 geodesic derep …`.

**`--taxon-rank g` during build is critical for NFS performance.** Genomes from the
same genus end up in the same shard file. Geodesic reads sketches genus by genus,
so without grouping each taxon scatters across hundreds of shards — catastrophic for
random-access NFS latency.

**Use `--checkm2` for MAG collections.** Without quality scores, representative
selection uses sketch completeness as a proxy. For cultured isolates this is fine;
for MAGs with highly variable completeness the CheckM2 scores produce meaningfully
better representatives.
