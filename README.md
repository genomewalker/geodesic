# geodesic

**spherical genome embeddings for diverse representative selection**

`geodesic` selects a diverse set of representative genomes per species for reference-based short-read mapping. Unlike traditional dereplication tools that minimise redundancy, `geodesic` maximises the diversity of the retained set.

## Algorithm

1. **Sketch** — sketch each genome with two independent One-Permutation Hashing (OPH) signatures (k=21, m=10000 bins, seeds 42 and 1337). Each bin holds the minimum hash among all k-mers mapping to that bin. Property: P[sig_A[t] == sig_B[t]] = J(A,B). Averaging two signatures halves Jaccard estimation variance. The bitmask of occupied bins (pre-densification) is stored for containment estimation.

2. **Embed** — project all genomes onto a unit sphere via Nyström spectral embedding with four robustness improvements:
   - *Stratified anchors*: anchors are sampled stratified by fill fraction f_i = n_real_bins/m across 5 quantile strata, ensuring sparse MAGs and complete genomes are equally represented as landmarks.
   - *Averaged kernel*: both anchor-anchor and genome-anchor similarities use the dual-sketch average K[i,j] = (J₁(i,j) + J₂(i,j)) / 2, reducing Jaccard estimation variance.
   - *Containment blend*: when either genome has f_i < 0.2, Jaccard is blended with bin co-occupancy max(C(A→B), C(B→A)) where C(A→B) = |bins(A) ∩ bins(B)| / |bins(A)|, weighted by alpha = 1 - f_i/0.2. Applied consistently in both the anchor Gram matrix and genome-to-anchor extension.
   - *Regularisation*: symmetric Laplacian normalisation (D^{-1/2} K D^{-1/2}) removes hub-anchor bias; Tikhonov diagonal loading (K += λI, λ = 0.01 × max(mean diagonal, 1e-4)) prevents near-zero eigenvalues from blowing up the Nyström inversion.

   The Gram matrix is eigendecomposed; all genomes are projected onto the top d eigenvectors via the Nyström extension and L2-normalised. d is auto-selected to capture ≥95% of variance. Dot products on the unit sphere approximate the normalised kernel similarity, not raw Jaccard — borderline threshold decisions use exact OPH Jaccard (Phase 7).

3. **Index** — build an HNSW nearest-neighbour index on the sphere for sub-linear candidate retrieval.

4. **Score** — compute isolation scores (mean angular distance to k=10 nearest neighbours) as a proxy for genome representativeness. Derive the diversity threshold from the observed NN distance distribution: `diversity_threshold = min(NN_P95, angular_distance(user_ANI_threshold))`.

5. **Select** — run quality-weighted Farthest Point Sampling (FPS) on the unit sphere. FPS is a greedy 2-approximation to the k-center problem: each step adds the genome farthest from its nearest representative, weighted by quality and sqrt(genome_size / median_genome_size). Stops when every genome is within `diversity_threshold` of some representative.

6. **Merge** — coalesce representatives within `min_rep_distance` via Union-Find: reps that landed too close due to quality/size weighting are collapsed to one.

7. **Verify** — for non-representatives with embedding distance in [θ·(1−ε), θ) where ε = min(3/√d, 0.3), check the top-3 nearest representatives by embedding similarity using dual-sketch averaged OPH Jaccard. Promote only if no checked representative is within θ in raw OPH space.

ANI thresholds are defined from raw Jaccard via the Mash formula: `ANI = (2J / (1+J))^(1/k) × 100`. The embedding is used as a search surrogate; borderline cases are resolved in raw OPH Jaccard space.

## Build

**Dependencies** (resolved automatically via CMake FetchContent):
- CLI11, spdlog, BS::thread_pool, hnswlib, Catch2, rapidgzip, Eigen3

**System dependencies:**
- DuckDB (searched in conda env or system paths)
- C++20 compiler with AVX2 support

```bash
git clone https://github.com/genomewalker/geodesic
cd geodesic
mkdir build && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Binary at `build/geodesic`. RPATH is embedded — no `LD_LIBRARY_PATH` needed.

## Usage

```bash
geodesic derep -t genomes.tsv --threads 24 -p my_run
```

### Input

Tab-separated file with header:

```
accession	taxonomy	file
GCA_000001405	d__Bacteria;...;s__Escherichia coli	/path/to/genome.fna.gz
```

- `taxonomy`: GTDB-style semicolon-separated lineage (`d__;p__;c__;o__;f__;g__;s__`)
- `file`: absolute path to FASTA (plain or gzip)

Optional CheckM2 quality file (completeness/contamination):

```bash
geodesic derep -t genomes.tsv --checkm2 quality.tsv --threads 24 -p my_run
```

### Output

Results written to `<prefix>/`:
- `<prefix>_derep_genomes.tsv` — all genomes with representative assignment and ANI to rep
- `<prefix>_results.tsv` — per-taxon summary (n_genomes, n_reps, method, runtime)
- `<prefix>_diversity_stats.tsv` — coverage and diversity metrics per taxon
- `<prefix>.db` — DuckDB database with full results

### Key options

| Option | Default | Description |
|--------|---------|-------------|
| `--threads` | 1 | Total CPU threads |
| `-w,--workers` | 1 | Parallel taxa workers (workers × threads = total cores used) |
| `--ani-threshold` | 95.0 | ANI threshold (%) — acts as cap; actual threshold inferred from data |
| `--geodesic-dim` | 256 | Nyström embedding dimension (auto-selected if variance-based) |
| `--geodesic-kmer-size` | 21 | k-mer size for OPH sketching |
| `--geodesic-sketch-size` | 10000 | OPH sketch size (bins) |
| `--geodesic-auto-calibrate` | on | Auto-select tier (k/dim/sketch) from random ANI sample |
| `--geodesic-calibration-pairs` | 50 | Pairs sampled for tier selection |
| `--checkm2` | — | CheckM2 TSV for quality-weighted selection |
| `-z` | 2.0 | Z-score threshold for contamination detection |
| `--nystrom-diagonal-loading` | 0.01 | Tikhonov regularization fraction for Nyström Gram matrix |
| `--nystrom-degree-normalize` | on | Symmetric Laplacian normalization of Gram matrix |

### Resuming

`geodesic` writes progress to DuckDB as it runs. Resume an interrupted run by pointing to the same database:

```bash
geodesic derep -t genomes.tsv -d my_run/my_run.db --threads 24 -p my_run
```

Already-completed taxa are skipped automatically.

## Performance

Benchmarked on GTDB r220 (5.2M genomes, 130k taxa, 24 workers × 4 threads on 96-core node):

| Scale | Genomes | Representatives | Runtime |
|-------|---------|-----------------|---------|
| Full GTDB r220 | 5,195,094 | ~520,000 | ~6.5 h |
| S. enterica (single taxon) | 367,440 | 9,415 | 18.5 min (24t) |
| E. coli (single taxon) | 233,000 | ~15,000 | ~22 min |

Coverage: 99.19% of S. enterica genomes within 99% ANI of their nearest representative. Mean coverage ANI 99.99%. See [wiki/ALGORITHM.md](wiki/ALGORITHM.md) for full algorithm documentation.

## License

MIT
