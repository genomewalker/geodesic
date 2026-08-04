# geodesic

θ-cover dereplication on a genopack archive. Every genome is within θ of at least one rep.

[Algorithm](https://genomewalker.github.io/geodesic/) · [Wiki](https://github.com/genomewalker/geodesic/wiki)

## Build

C++20, AVX2. Dependencies fetched via FetchContent (CLI11, spdlog, hnswlib, Eigen3, Catch2, rapidgzip).

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

## Usage

```bash
geodesic derep -g accessions.txt --pack mydb.gpk -p run1 --threads 16
geodesic ani   --ql queries.txt  --pack mydb.gpk -t 16 -o ani.tsv
geodesic update -g new.txt --lock run1.lock --pack mydb.gpk --threads 16
```

`--ani-threshold` (default 95.0), `--seed` (default 42), `--threads`, `--tmp-dir`, `--checkm2`, `--gunc-scores`.

`-g`: one accession per line, `#` comments ignored. `--pack`: `.gpk` file or directory of `part_*.gpk` parts.

### Output

| File | Contents |
|------|----------|
| `<prefix>_derep_genomes.tsv` | accession, taxonomy, representative, cluster_rep, nn_dist, sketch_fill; with `--pack`, also quality_tier, contam_D, is_singleton (self-describing output) |
| `<prefix>_results.tsv` | per-taxon: method, n_genomes, n_genomes_derep, communities, weight |
| `<prefix>_stats.tsv` | per-taxon: preflight/quality counts, θ used |
| `<prefix>_diversity_stats.tsv` | per-taxon: diversity ANI range, n_pairs, outliers excluded/retained |
| `<prefix>_outliers.tsv` | flagged candidates: category, nn_outlier, isolation_score, kmer_div_zscore, flag_reason, excluded |
| `<prefix>_failed.tsv` | accession, taxonomy, file, reason (resolve/embed failures; sketch-less-but-resolvable genomes are kept as self-reps) |

`--emit-gpd <path>` writes the derep archive (`.gpd`) with per-genome cluster and embedding data (see [Derep-Output](wiki/Derep-Output.md)). `--grd-output`, `--geodf-output`, `--lock-output` are needed for distributed/incremental runs.

## Distributed

```bash
geodesic scatter -g genomes.txt --pack mydb.gpk -n 4 -o dist/ --threads 16
parallel -j4 < dist/run.sh
geodesic gather -d dist/ -o dist/merged.grd -p merged
```

## Algorithm

1. Two OPH signatures per genome (k=21, m=10,000 by default).
2. Nyström extension onto the unit sphere; symmetric Laplacian + Tikhonov regularisation.
3. HNSW index on the sphere.
4. Isolation scores from k_iso = max(10, min(20, ⌊log₂n⌋)) neighbours; θ = min(θ_ANI, max(θ_MST, θ_ANI/4)), where θ_MST is the MST max edge with outlier bridges excluded and capped at the closest cross-cluster pair; outliers by MAD Z-score.
5. Greedy FPS θ-cover; fitness = d·√(L/L_m)·(0.5+0.5·q̂), q̂ = clamp(quality/100, 0, 1).
6. Union-Find coalescence within d_min; borderline non-reps rechecked by OPH Jaccard.
7. Each non-rep vs its rep by Jaccard and containment.

ANI = (2J/(1+J))^(1/k) × 100.

## License

MIT
