# ANI computation (`ani` and `validate-ani`)

Beyond dereplication, geodesic ships two ANI utilities that read genomes
directly from a genopack archive:

- **`geodesic ani`** — fast all-pairs FracMinHash ANI over genomes stored in a
  pack, computed entirely in memory with no FASTA extraction to disk.
- **`geodesic validate-ani`** — accuracy harness that compares the OPH-based
  Jaccard ANI estimates stored in the pack against geodesic's own FracMinHash
  ANI, used to verify that the cheap OPH sketches track exact ANI closely enough
  for dereplication.

Both subcommands load raw FASTA sequences from the pack with the same in-memory
shard reader used by `derep`, so they never touch a temp directory or
materialise genomes on disk.

---

## `geodesic ani`

Computes pairwise FracMinHash ANI between genomes in a pack.

### What it does

1. **Load sequences in-memory.** The union of `--ql` and `--rl` accessions is
   read from the pack via the shard-batch reader. FASTA bytes stay in RAM; no
   extraction step.
2. **Build FracMinHash sketches.** For each genome, every position is encoded as
   a canonical k-mer (2 bits/base, forward vs. reverse-complement minimum) and
   hashed with an ntHash-style rolling hash. A k-mer is kept in the sketch when
   `hash % c == 0` — Lemire fast-divisibility selection — giving a sketch whose
   size scales with `genome_length / c`. This is the same compression model as
   skani's `-c` parameter.
3. **All-pairs intersection.** Each sketch is a sorted set of 64-bit hashes;
   ANI is computed from the size of the sorted-set intersection (AVX2-accelerated
   merge) between query and reference sketches. Alignment fraction (`af`) and the
   two directional containments (`c_ab`, `c_ba`) fall out of the same
   intersection.
4. **Write TSV.** One row per surviving pair.

When `--rl` is omitted, the reference list defaults to the query list and
geodesic computes self all-pairs: each unordered pair is emitted once
(`query < ref`), self-pairs skipped.

### Usage

All-pairs ANI within a single collection:

```bash
geodesic ani \
    --ql accessions.txt \
    --pack mydb.gpk \
    -o ani_results.tsv \
    -t 24
```

Query genomes against a separate reference set:

```bash
geodesic ani \
    --ql queries.txt \
    --rl references.txt \
    --pack mydb.gpk \
    --min-af 0.15 \
    -o ani_results.tsv \
    -t 24
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--ql` | required | Query accession list (one per line; `#` comments allowed) |
| `--rl` | -- | Reference accession list. Empty = same as `--ql` (self all-pairs) |
| `--pack` | required | genopack archive (single `.gpk` or directory of `part_*.gpk`) |
| `-o, --output` | `ani_results.tsv` | Output TSV path |
| `-t, --threads` | 4 | Threads for parallel sketch building and pair computation |
| `--min-af` | 0 | Minimum alignment fraction to report a pair |
| `--ani-k` | 21 | k-mer size |
| `-c, --compression` | 125 | Compression factor: keep k-mer if `hash % c == 0` (matches skani `-c`) |

### Output

A TSV with a header row and one row per pair that passes the `--min-af` filter:

| Column | Description |
|--------|-------------|
| `query` | Query accession |
| `ref` | Reference accession |
| `ani` | Estimated average nucleotide identity (%) |
| `af` | Alignment fraction (fraction of the smaller sketch shared) |
| `c_ab` | Containment of A in B (fraction of query sketch found in reference) |
| `c_ba` | Containment of B in A (fraction of reference sketch found in query) |

Numeric columns are written with four decimals.

```
query           ref             ani      af       c_ab     c_ba
GCA_000005845   GCA_000008865   97.4213  0.8821   0.8702   0.8940
GCA_000005845   GCA_000019425   95.1077  0.6634   0.6512   0.6755
```

Accessions present in the lists but absent from the pack are silently skipped
(a count of loaded vs. requested FASTAs is logged).

---

## `geodesic validate-ani`

Validates the cheap OPH Jaccard ANI estimates stored in a pack against
geodesic's own FracMinHash ANI, treated as ground truth.

### Why it exists

Dereplication relies on OPH (one-permutation hash) sketches for both the
spectral embedding (Phase 2) and the final sketch-space certification (Phase 8).
OPH Jaccard is a cheap, fixed-size approximation; `validate-ani` quantifies how
well it tracks exact FracMinHash ANI across a real collection, per stored k-mer
size. The error columns it reports are the empirical version of the OPH error
bounds discussed in the [Algorithm reference](ALGORITHM).

### What it does

1. **Sample pairs.** Draw `-n` random unordered pairs (default 500) from the
   accession list using `--seed`; duplicates and self-pairs are removed.
2. **Load stored OPH sketches.** For every accession touched by a sampled pair,
   read its OPH signatures at **all** k-mer sizes present in the pack's SKCH
   section (e.g. 16, 21, 31) in a single archive pass.
3. **Build FracMinHash reference.** Load the same genomes' FASTAs from the pack
   and build FracMinHash sketches (the same `ani`-subcommand machinery), then run
   `compute_ani()` to get the reference ANI (`ani_geo`).
4. **Estimate OPH ANI.** For each stored k, compute the b-bit-corrected OPH
   Jaccard `J`, then convert to ANI via the Mash formula
   `ANI = (2J/(1+J))^(1/k) × 100`, clamped to `[70, 100]`.
5. **Write TSV.** One row per pair, with per-k Jaccard, ANI estimate, and the
   signed error against `ani_geo`.

The pack must contain OPH sketches: if the SKCH section is empty the command
fails with a request to rebuild the pack with `--sketch-kmers`.

### Usage

```bash
geodesic validate-ani \
    -g accessions.txt \
    --pack mydb.gpk \
    -n 1000 \
    -o ani_validation.tsv \
    -t 24
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-g, --genomes` | required | Accession list (one per line; `#` comments allowed) |
| `--pack` | required | genopack archive (single `.gpk` or directory of `part_*.gpk`) |
| `-n, --pairs` | 500 | Number of random pairs to evaluate |
| `-o, --output` | `ani_validation.tsv` | Output TSV path |
| `--seed` | 42 | RNG seed for pair sampling |
| `--geodesic-sketch-size` | 10000 | Sketch size (bins) used for OPH Jaccard; clamped to the size stored in the pack |
| `-t, --threads` | 4 | Threads for FracMinHash sketch building |

### Output

The column set adapts to the k-mer sizes stored in the pack. For a pack built
with `--sketch-kmers 16,21,31` the header is:

```
query  ref  ani_geo  j_k16 ani_est_k16 err_k16  j_k21 ani_est_k21 err_k21  j_k31 ani_est_k31 err_k31  fill_query  fill_ref
```

| Column | Description |
|--------|-------------|
| `query`, `ref` | The sampled accession pair |
| `ani_geo` | FracMinHash ANI from `compute_ani()` — the reference value |
| `j_k<N>` | b-bit-corrected OPH Jaccard at k-mer size *N* |
| `ani_est_k<N>` | OPH ANI estimate at *N* via the Mash formula |
| `err_k<N>` | `ani_est_k<N> − ani_geo` — signed error in ANI points |
| `fill_query`, `fill_ref` | OPH bin fill fraction (real bins / m) for query and reference, taken from the first stored k |

Pairs whose reference ANI could not be computed (e.g. a FASTA missing from the
pack) are skipped and counted in the log. Low `fill_*` values flag sparse
assemblies, where OPH variance — and hence `err_*` — is expected to be larger
(see the [fill fraction](ALGORITHM#fill-fraction) discussion).

### Interpreting the results

The `err_k<N>` distribution shows how each k-mer size tracks exact ANI:

- Near the default 95% ANI dereplication threshold, dense-sketch OPH error is
  typically well below 0.1 ANI points.
- `k=16` keeps signal at lower ANI (broad genera) but compresses high-ANI Jaccard;
  `k=31` spreads clonal (>99% ANI) pairs apart but loses signal below ~90% ANI.
  `k=21` is the default species-level compromise.
- A right-tail of large `|err|` concentrated at low `fill_*` is the expected
  signature of incomplete assemblies, not a model failure — this is exactly the
  regime where geodesic switches to containment-based corrections during derep.

---

## OPH vs FracMinHash: why two sketches?

| | OPH (in-pack) | FracMinHash (`ani`) |
|---|---|---|
| Size | Fixed `m` bins (default 10,000) | Scales with `genome_length / c` |
| Cost | Precomputed, stored in pack; one-pass load | Built on demand from FASTA |
| Used by | derep embedding + certification | `ani` subcommand, `validate-ani` reference |
| Strength | Constant memory, fast Jaccard, occupancy bitmask for containment | Resolution scales with genome size; closer to alignment ANI |

Dereplication uses OPH because its fixed footprint makes 5M-genome runs
tractable and the per-bin occupancy bitmask gives containment for free.
`validate-ani` exists precisely to confirm that this cheaper sketch stays within
tolerance of the FracMinHash ANI that `geodesic ani` reports.

See the [Algorithm reference](ALGORITHM) for the full OPH derivation and the
k-mer/ANI accuracy tables.

---

## Performance benchmark

> **Provenance:** reproduce with `bench/run_ani_bench.sh <pack> <accessions.txt>` against a
> clean build. Numbers below are from 200 *E. coli* GTDB r232 genomes (96–100% ANI range),
> 8 threads, NFS-backed storage. All sketch-based methods underestimate vs fastANI by
> 0.09–0.13 ANI points on average; this is expected and not a calibration error (fastANI uses
> fragmented alignment; sketch methods use k-mer Jaccard).

### Speed

| Method | Version | Pre-build | Query (19,900 pairs) | ms / pair |
|--------|---------|-----------|----------------------|-----------|
| fastANI | 1.34 | none | ~50 min | ~15 |
| skani | 0.2.2 | 2.1 s sketch | 80.5 s | 4.0 |
| **FracMinHash (GPK)** | — | 37.2 s build¹ | **4.3 s** | **0.3** |
| OPH k=31 (stored) | — | pre-built | <1 s | <0.1 |
| OPH k=21 (stored) | — | pre-built | <1 s | <0.1 |

¹ genopack build cost is amortised — it also compresses sequences, builds the sketch, and
writes GCOV/FCOV sections used by contamination detection.

### Accuracy vs fastANI (12,611 matched pairs)

| Method | RMSE (ANI pts) | MAE | Bias |
|--------|---------------|-----|------|
| skani 0.2.2 | 0.177 | 0.140 | −0.088 |
| **FracMinHash (GPK)** | **0.198** | 0.161 | −0.127 |
| OPH k=31 | 0.338 | 0.274 | −0.250 |
| OPH k=21 | 0.246 | 0.197 | +0.141 |

**Interpretation.** FracMinHash from a pre-built GPK is 13× faster per pair than skani
and comparable in accuracy (0.198 vs 0.177 RMSE). The stored OPH sketches (k=21) are an
order of magnitude faster still but carry ~0.25 ANI points more error — sufficient for
dereplication clustering but not for reporting exact ANI values. `validate-ani` exists
to quantify this error on your specific collection.

### Mash formula note

The Mash ANI formula used throughout geodesic and validate-ani is:

$$\mathrm{ANI} = \left(\frac{2J}{1+J}\right)^{1/k} \times 100$$

where $J$ is the (b-bit-corrected) OPH Jaccard similarity. The `2J/(1+J)` term converts
union-based Jaccard to the equivalent containment-symmetric Jaccard under the Mash model;
it is the correct transformation for sketch-based estimators and matches the skani and
Mash papers. All ANI threshold conversions in geodesic use $q = (\mathrm{ANI}/100)^k$ (ANI
as a fraction, 0–1) to avoid unit errors at the certification step.
