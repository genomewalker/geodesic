# geodesic

Selects representative genomes per taxon from a genopack archive. Output is a θ-cover: every input genome is within ANI threshold of at least one rep.

[Algorithm](https://genomewalker.github.io/geodesic/) · [Wiki](https://github.com/genomewalker/geodesic/wiki)

## Algorithm

1. Two independent OPH signatures per genome (pre-computed in the pack at multiple k). Default k=21, m=10,000.
2. Nyström extension onto the unit sphere from a landmark anchor set. Symmetric Laplacian normalisation; Tikhonov ridge regularisation.
3. HNSW index on the unit sphere.
4. Isolation scores from k_iso = max(10, min(20, ⌊log₂ n⌋)) neighbours. θ = longest MST edge in the k-NN graph at k_stable. Outliers flagged by MAD Z-score on per-component isolation scores.
5. Greedy FPS θ-cover. Fitness = d_i · √(L_i/L_m). Terminates when every genome is within θ of a rep.
6. Union-Find coalescence of reps within d_min.
7. Borderline non-reps (d ∈ [θ(1−ε), θ)) rechecked by OPH Jaccard against top-3 nearest reps.
8. Full coverage pass: every non-rep vs its assigned rep by OPH Jaccard. τ = q/(2−q), q = (ANI/100)^k. Sketch-asymmetric pairs also checked by containment.

ANI = (2J/(1+J))^(1/k) × 100.

## Build

Requires C++20 with AVX2. Dependencies (CLI11, spdlog, BS::thread_pool, hnswlib, Catch2, rapidgzip, Eigen3) fetched via CMake FetchContent.

```bash
git clone https://github.com/genomewalker/geodesic
cd geodesic
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Binary at `build/geodesic`.

## Usage

```bash
geodesic derep -g genomes.txt --pack mydb.gpk --threads 24 -p my_run
```

`-g`: one accession per line, `#` comments ignored. `--pack`: a `.gpk` directory or a directory of `part_*.gpk` parts. Taxonomy comes from the pack's TAXN section.

### Output

| File | Contents |
|------|----------|
| `<prefix>_derep_genomes.tsv` | accession, taxonomy, representative, cluster_rep, nn_dist, sketch_fill |
| `<prefix>_results.tsv` | per-taxon summary: method, n_genomes, n_genomes_derep, communities, weight |
| `<prefix>_diversity_stats.tsv` | coverage and diversity metrics per taxon |
| `<prefix>_stats.tsv` | per-taxon pipeline stats: preflight/quality/outlier counts, MST edges, ANI used |
| `<prefix>_outliers.tsv` | isolation score outliers |
| `<prefix>_failed.tsv` | genomes that failed sketching/embedding (reason column) |
| `<path>.grd` | `--grd-output`: sketches + embeddings + indexes; required for `gather` |
| `<path>.gpd` | `--emit-gpd`: derep archive consumed by `genopack::DerepView` |
| `<path>.geodf` | `--geodf-output`: binary results (rep set + embeddings); input to `update` |
| `<path>.lock` | `--lock-output`: provenance JSON; input to `geodesic update` |

### Options

```
-g, --genomes       accession list (required)
--pack              genopack archive (required)
-o, --out-dir       output directory (default: .)
-p, --prefix        output file prefix
--tmp-dir           temp directory (default: .)
--references        accessions forced to be reps
--fixed-taxa        pre-assigned reps; skip selection for those taxa
--checkm2           CheckM2 TSV for quality-weighted FPS
--gunc-scores       GUNC TSV; exclude chimeric assemblies
--threads           CPU threads (default: 1)
--seed              RNG seed; sig1=seed, sig2=seed+1 (default: 42)
--ani-threshold     ANI cap % (default: 95.0)
-z, --z-threshold   MAD Z-score cutoff for outlier flagging (default: 2.0)
--geodesic-kmer-size   k for OPH sketching (default: 21)
--geodesic-sketch-size OPH bins (default: 10000)
--geodesic-dim      Nyström embedding dimension (default: 256)
--k-cap-max         max K_cap retry when k-NN graph fails to connect (default: 256)
--copy-reps         copy rep FASTAs to --out-dir
-v / -q / --debug   verbosity
```

Results are bit-identical across runs with the same `--seed` and input order (Eigen pinned to 1 thread at startup).

## Pairwise ANI (`geodesic ani`)

```bash
geodesic ani --ql accessions.txt --pack mydb.gpk -t 24 -o ani_results.tsv
geodesic ani --ql queries.txt --rl references.txt --pack mydb.gpk -t 24 -o ani_results.tsv
```

Output columns: `query`, `ref`, `ani`, `af`, `c_ab`, `c_ba`. `--rl` omitted → self all-pairs. Options: `--ani-k` (default 21), `-c/--compression` (default 125), `--min-af`.

`geodesic validate-ani` compares pack OPH Jaccard ANI against FracMinHash ANI per k-mer size.

## Distributed mode

```bash
geodesic scatter -g genomes.txt --pack mydb.gpk -n 4 -o dist/ --threads 24
bash dist/run.sh           # or: parallel -j4 < dist/run.sh
geodesic gather -d dist/ -o dist/merged.grd -p merged
```

Each worker runs `geodesic derep` independently on its partition. `scatter --rank` sets the partitioning rank (g/f/s; default g). `gather` merges GRD archives and TSV outputs.

## Incremental updates

```bash
geodesic update -g new_genomes.txt --lock prev_run.lock \
    --pack mydb.gpk \
    --geodf-output updated.geodf --lock-output updated.lock \
    --threads 24
```

Diffs accessions against the prior GEODF, re-runs only taxa that gained members, copies unchanged taxa from the prior result.

## License

MIT
