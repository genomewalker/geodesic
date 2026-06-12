# geodesic

θ-cover dereplification of genomes from a genopack archive. Every input genome ends up within ANI threshold of at least one representative.

[Algorithm](https://genomewalker.github.io/geodesic/) · [Wiki](https://github.com/genomewalker/geodesic/wiki)

## Build

C++20, AVX2. Dependencies fetched via FetchContent (CLI11, spdlog, hnswlib, Eigen3, Catch2, rapidgzip).

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

## Usage

```bash
geodesic derep -g accessions.txt --pack mydb.gpk -p run1 -t 16
geodesic ani   --ql queries.txt  --pack mydb.gpk -t 16 -o ani.tsv
geodesic update -g new.txt --lock run1.lock --pack mydb.gpk -t 16
```

Key flags: `--ani-threshold` (default 95.0), `--seed` (default 42, deterministic), `--threads`, `--tmp-dir`, `--checkm2`, `--gunc-scores`.

`-g` accepts one accession per line (`#` comments ignored). `--pack` is a `.gpk` file or directory of `part_*.gpk` parts.

### Output files

| File | Contents |
|------|----------|
| `<prefix>_derep_genomes.tsv` | per-genome: accession, taxon, rep, cluster_rep, nn_dist, sketch_fill |
| `<prefix>_results.tsv` | per-taxon: method, n_genomes, n_derep, communities |
| `<prefix>_stats.tsv` | per-taxon pipeline counts and ANI used |
| `<prefix>_failed.tsv` | genomes that failed sketching or embedding |

Pass `--grd-output` / `--geodf-output` / `--lock-output` for distributed or incremental runs.

## Distributed

```bash
geodesic scatter -g genomes.txt --pack mydb.gpk -n 4 -o dist/ -t 16
parallel -j4 < dist/run.sh
geodesic gather -d dist/ -o dist/merged.grd -p merged
```

## Algorithm

1. Two OPH signatures per genome (k=21, m=10,000 by default).
2. Nyström extension onto the unit sphere; symmetric Laplacian + Tikhonov regularisation.
3. HNSW index on the sphere.
4. Isolation scores from k_iso = max(10, min(20, ⌊log₂n⌋)) neighbours; θ = longest MST edge; outliers by MAD Z-score.
5. Greedy FPS θ-cover; fitness = d·√(L/L_m); stops when all genomes are covered.
6. Union-Find coalescence within d_min; borderline non-reps rechecked by OPH Jaccard.
7. Full coverage pass: each non-rep vs its rep by Jaccard and containment.

ANI = (2J/(1+J))^(1/k) × 100.

## License

MIT
