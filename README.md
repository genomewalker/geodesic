# geodesic

**spherical genome embeddings for diverse representative selection**

`geodesic` selects a constellation of representative genomes per species that maximises nucleotide diversity for reference-based short-read mapping. Unlike traditional dereplication tools that minimise redundancy, `geodesic` maximises the diversity of the retained set.

## Algorithm

1. **Embed** — sketch each genome with One-Permutation Hashing (OPH, k=21, m=10000 bins), then project via CountSketch to the unit sphere S²⁵⁵. The embedding preserves Jaccard similarity: E[dot(u,v)] = J(A,B).
2. **Index** — build an HNSW nearest-neighbour index on the sphere for fast candidate retrieval.
3. **Score** — compute isolation scores (mean distance to k nearest neighbours) as a proxy for genome representativeness.
4. **Select** — run Farthest Point Sampling (FPS) on the sphere, weighted by isolation score. This is a greedy approximation to the Thomson problem: distribute k charges on a sphere to maximise mutual repulsion.
5. **Merge** — coalesce representatives within `min_rep_distance` via Union-Find (electrostatic merge).
6. **Verify** — run skani only on borderline pairs where the embedding Jaccard estimate is within 3/√dim of the ANI threshold.

ANI is derived calibration-free from the embedding: `ANI = (2J / (1+J))^(1/k) × 100`, where `J ≈ cos(π × angular_distance)` (Mash formula).

Auto-calibration samples random genome pairs to set the diversity threshold from the observed ANI range, scaling further by genome size heterogeneity (open pangenome detection).

## Build

**Dependencies** (resolved automatically via CMake FetchContent):
- CLI11, spdlog, BS::thread_pool, hnswlib, Catch2, rapidgzip

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
geodesic -t genomes.tsv --threads 24 -p my_run
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
geodesic -t genomes.tsv --checkm2 quality.tsv --threads 24 -p my_run
```

### Output

Results written to `<prefix>/`:
- `<prefix>_derep.tsv` — all genomes with representative assignment and ANI to rep
- `<prefix>_results.tsv` — per-taxon summary (n_genomes, n_reps, method, runtime)
- `<prefix>_diversity_stats.tsv` — coverage and diversity metrics per taxon
- `<prefix>.db` — DuckDB database with full results

### Key options

| Option | Default | Description |
|--------|---------|-------------|
| `--threads` | 1 | Total CPU threads |
| `--ani-threshold` | 95.0 | ANI threshold (%) for redundancy |
| `--geodesic-dim` | 256 | Embedding dimension |
| `--geodesic-kmer-size` | 21 | k-mer size for OPH sketching |
| `--geodesic-sketch-size` | 10000 | OPH sketch size (bins) |
| `--geodesic-diversity-threshold` | 0.02 | Min angular distance gain to add a rep |
| `--geodesic-auto-calibrate` | on | Calibrate threshold from random ANI sample |
| `--geodesic-calibration-pairs` | 50 | Pairs sampled for auto-calibration |
| `--checkm2` | — | CheckM2 TSV for quality-weighted selection |
| `-z` | 2.0 | Z-score threshold for contamination detection |

### Resuming

`geodesic` writes progress to DuckDB as it runs. Resume an interrupted run by pointing to the same database:

```bash
geodesic -t genomes.tsv -d my_run/my_run.db --threads 24 -p my_run
```

Already-completed taxa are skipped automatically.

## Performance

Benchmarked on GTDB r220 (5.2M genomes, 130k taxa, 24 threads):

| Species | Genomes | Representatives | Runtime |
|---------|---------|-----------------|---------|
| S. enterica | 367,440 | ~2,500 | ~16 min |
| E. coli | 233,000 | ~15,000 | ~22 min |
| Tiny taxa (2–10 genomes) | — | — | <0.5s |

Coverage (mean ANI of each genome to nearest representative): >99% for all major clades.

## License

MIT
