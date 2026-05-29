# geodesic

**spherical genome embeddings for diverse representative selection**

`geodesic` selects representative genomes per species for reference-based short-read mapping. It maximises diversity in the retained set rather than minimising redundancy: every genome in the collection is within a target ANI of at least one representative.

[Algorithm visualisation](https://genomewalker.github.io/geodesic/) · [Wiki](https://github.com/genomewalker/geodesic/wiki)

## Algorithm

1. **Sketch** -- read two independent [One-Permutation Hash (OPH)](https://proceedings.neurips.cc/paper_files/paper/2012/file/eaa32c96f620053cf442ad32258076b9-Paper.pdf) signatures per genome from the genopack archive. The archive stores pre-computed OPH signatures at multiple k-mer sizes (e.g. $k \in \{16, 21, 31\}$, $m=10{,}000$ bins per size, seeds `--seed` (default 42) for sig1 and `--seed`+1 (default 43) for sig2). Auto-calibration selects the appropriate k and m from a small sample of genome pairs; the default tier for 95–99% ANI taxa uses $k=21$, $m=10{,}000$ bins. Each bin holds the minimum hash of all k-mers mapping to it, giving $\Pr[\mathrm{sig}_A[t] = \mathrm{sig}_B[t]] \approx J(A,B)$ (equality holds before densification; after densification, filled bins introduce correlation). Averaging two independent signatures halves Jaccard estimation variance. The per-bin occupancy bitmask enables containment estimation for sparse assemblies.

2. **Embed** -- place all genomes in a low-dimensional similarity space using a small set of landmark genomes (anchors):
   - *Anchor selection*: anchors are drawn across the full range of assembly completeness so sparse MAGs and complete genomes are equally represented as landmarks.
   - *Stable similarities*: anchor Gram matrix uses the average Jaccard from two independent OPH sketches, reducing sketch noise.
   - *Regularised spectral map*: symmetric Laplacian normalisation removes hub-anchor bias; light Tikhonov ridge prevents eigenvalue blow-up; any indefinite shift is repaired in-place.
   - *Project all genomes*: each genome is projected by its similarity to the anchors via Nyström extension, then L2-normalised onto the unit sphere.

   The embedding dimension is auto-selected to capture ≥95% of anchor variance. Borderline coverage decisions are re-checked with direct OPH Jaccard (Phase 7).

3. **Index** -- build an [HNSW](https://arxiv.org/abs/1603.09320) nearest-neighbour index on the sphere for sub-linear candidate retrieval.

4. **Score** -- compute isolation scores (mean angular distance to $k_\mathrm{iso} = \max(10, \min(20, \lfloor\log_2 n\rfloor))$ nearest neighbours) and build the minimum spanning tree of the k-NN graph via Kruskal's algorithm. The edge budget $K_\mathrm{cap}$ scales with taxon size (64 / 128 / 256 for $n \leq 5{,}000$ / $50{,}000$ / larger) and is determined by a two-phase adaptive scan. Phase A sweeps k-NN columns from $k=1$ to $K_\mathrm{cap}$ via DSU, recording $k_\mathrm{conn}$: the first $k$ at which the core graph becomes connected (−1 if never). Phase B probes a ladder $\{1,2,3,4,6,8,12,16,24,32,48,64\}$ starting from $k_\mathrm{conn}$, picking the smallest $k$ where the bottleneck (MST max edge) is within 3% of the $K_\mathrm{cap}$ reference value -- this is $k_\mathrm{stable}$. Using $k_\mathrm{conn}$ alone is insufficient because the first-connection edge is often a brittle bridge; the probe identifies where the bottleneck has stabilised. The longest MST edge at $k_\mathrm{stable}$ sets the diversity threshold $\theta$: the minimum inter-strain scale at which the proximity graph becomes connected. Flag contamination candidates using a MAD-based robust estimator (median + $z \cdot 1.4826 \cdot \mathrm{MAD}$) on the per-component isolation score distribution, resistant to the long right tail introduced by contaminated genomes. When the k-NN graph fails to connect at $K_\mathrm{cap}=64$, retries at 128 and 256 with full HNSW requery and raised `ef_search` (configurable via `--k-cap-max`).

5. **Select** -- [Farthest Point Sampling (FPS)](https://en.wikipedia.org/wiki/Farthest-first_traversal) on the unit sphere, a greedy $\theta$-cover: each step adds the genome farthest from its nearest representative, with fitness $= d_i \cdot \sqrt{L_i/L_m}$ where $d_i$ is distance to nearest rep and $L_i/L_m$ is genome size relative to median. Quality serves as tie-breaker only (not multiplied into fitness). When CheckM2 scores are provided via `--checkm2`, quality = completeness − 5 × contamination. Without CheckM2, quality = (filled bins / sketch size) × 100 (sketch completeness). Stops when every genome is within $\theta$ of some representative.

6. **Merge** -- coalesce representatives within $d_{\mathrm{min}}$ via Union-Find.

7. **Verify** -- for non-representatives with embedding distance in $[\theta(1-\varepsilon),\,\theta)$, check the top-3 nearest representatives using exact dual-sketch OPH Jaccard. Promote only if no representative is within $\theta$ in sketch space.

8. **Certify** -- universal sketch-space coverage pass: every non-representative is verified against its assigned representative by direct OPH Jaccard. Certification threshold $\tau = q/(2-q)$ where $q = ({\rm ANI}/100)^k$ (ANI as a fraction, e.g. 0.95 for 95%). Any genome failing this check is promoted to a representative. OPH estimation error depends on real-bin occupancy and Jaccard; near the default 95% ANI threshold with dense sketches it is typically well below 0.1 ANI points, but sparse genomes are less stable. Sketch-asymmetric pairs (MAG vs. complete genome, $n_\mathrm{real,small}/n_\mathrm{real,large} \leq 0.85$ by occupied OPH bins) are additionally checked by directional containment: the fraction of the small genome's real bins that match the large genome must exceed $q = ({\rm ANI}/100)^k$.

ANI thresholds are derived from Jaccard via the Mash formula: $\mathrm{ANI} = \left(\frac{2J}{1+J}\right)^{1/k} \times 100$.

## Build

**Dependencies** (resolved automatically via CMake FetchContent):
- CLI11, spdlog, BS::thread_pool, hnswlib, Catch2, rapidgzip, Eigen3

**System dependencies:**
- C++20 compiler with AVX2 support

```bash
git clone https://github.com/genomewalker/geodesic
cd geodesic
mkdir build && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Binary at `build/geodesic`. RPATH is embedded -- no `LD_LIBRARY_PATH` needed.

## Usage

```bash
geodesic derep -g genomes.txt --pack mydb.gpk --threads 24 -p my_run
```

### Input

`--genomes` and `--pack` are both required. Taxonomy is read from the pack's TAXN section — no taxonomy column needed.

**Accession list** (`--genomes <file>`): one accession per line, `#` comments allowed.

```
# E. coli collection
GCA_000001405
GCA_000005845
```

**genopack archive** (`--pack <dir>`): a single `.gpk` archive directory or a directory containing one or more `part_*.gpk` parts (multipart set). Genome sequences, OPH sketches, and taxonomy are all read from the pack — no FASTA files needed.

### Output

Results written to the working directory (or `--out-dir`). TSV outputs are always written; binary archives are opt-in.

| File | When | Contents |
|---|---|---|
| `<prefix>_derep_genomes.tsv` | always | `accession`, `taxonomy`, `representative` (0/1), `cluster_rep`, `nn_dist`, `sketch_fill` — one row per genome |
| `<prefix>_results.tsv`       | always | Per-taxon summary (`taxonomy`, `method`, `n_genomes`, `n_genomes_derep`, `communities`, `weight`) |
| `<prefix>_diversity_stats.tsv` | always | Coverage and diversity metrics per taxon |
| `<prefix>_stats.tsv`         | always | Per-taxon pipeline stats (preflight/quality/outlier counts, MST edges, ANI threshold used) |
| `<prefix>_outliers.tsv`      | always | Flagged anomalous genomes (isolation score outliers) |
| `<prefix>_failed.tsv`        | always | Genomes that failed sketching/embedding (`accession`, `taxonomy`, `file`, `reason`) |
| `<path>.grd`                 | `--grd-output` | GRD archive: per-genome OPH sketches + Nyström embeddings + indexes (TIDX/ACCX/STRT). Required for `gather` merging across distributed shards. |
| `<path>.gpd`                 | `--emit-gpd`   | Geodesic Derep Archive consumed by `genopack::DerepView` (see [Derep Output](wiki/Derep-Output)). Requires `--pack`. |
| `<path>.geodf`               | `--geodf-output` | Binary results file (rep set + per-genome embeddings); referenced by the `--lock-output` JSON. |
| `<path>.lock` (JSON)         | `--lock-output` | Provenance file recording rep set, parameters, and the `.geodf` path; input to `geodesic update --lock`. |

### Key options

**Inputs / outputs**

| Option | Default | Description |
|--------|---------|-------------|
| `-g, --genomes` | required | Accession list (one per line; taxonomy read from pack TAXN section) |
| `--pack` | required | genopack archive (single `.gpk` directory or directory of `part_*.gpk` parts) |
| `-o, --out-dir` | `.` | Parent directory for all outputs; TSV files go to `<out-dir>/<prefix>/`; also used for FASTA copies when `--copy-reps` is set |
| `-p, --prefix` | -- | Prefix for `<prefix>_*.tsv` outputs |
| `--tmp-dir` | `.` | Temporary directory |
| `--references` | -- | File of accessions (one per line) to always include as representatives |
| `--fixed-taxa` | -- | File with fixed representative assignments (skip selection for those taxa) |
| `--grd-output` | -- | Path for `.grd` archive (sketches + embeddings); required for `gather` |
| `--emit-gpd` | -- | Path for `.gpd` derep archive (requires `--pack`); empty string = `<out-dir>/<prefix>.gpd` |
| `--geodf-output` | -- | Path for `.geodf` binary results file |
| `--lock-output` | -- | Path for the JSON lock file consumed by `geodesic update` |
| `--copy-reps` | off | Copy representative FASTA files into `--out-dir` (requires `--out-dir`) |
| `--checkm2` | -- | CheckM2 TSV for quality-weighted selection |
| `--gunc-scores` | -- | GUNC TSV to exclude chimeric assemblies from selection |
| `-v, --verbose` / `-q, --quiet` / `--debug` | info | Logging verbosity |
| `--keep-intermediates` | off | Keep intermediate files in `--tmp-dir` |

**Compute / parallelism**

| Option | Default | Description |
|--------|---------|-------------|
| `--threads` | 1 | Total CPU threads |
| `-w, --workers` | 1 | Worker pool (advanced; overrides `total_budget = workers * threads`) |
| `--io-threads` | 0 (auto) | Max concurrent NFS file readers (`0` = auto = `--threads`) |
| `--seed` | 42 | Master RNG seed; OPH sig1 uses `seed`, sig2 uses `seed`+1 (43 by default) |

**Algorithm**

| Option | Default | Description |
|--------|---------|-------------|
| `--ani-threshold` | 95.0 | ANI threshold (%) — acts as cap; actual threshold inferred from data |
| `-z, --z-threshold` | 2.0 | Z-score threshold for contamination/outlier flagging |
| `--geodesic-dim` | 256 | Nyström embedding dimension |
| `--geodesic-kmer-size` | 21 | k-mer size for OPH sketching |
| `--geodesic-sketch-size` | 10000 | OPH sketch size (bins) |
| `--geodesic-syncmer-s` | 0 | Open-syncmer prefilter `s` (0 = disabled) |
| `--geodesic-diversity-threshold` | 0.02 | Minimum FPS step gain (relative) before stopping |
| `--geodesic-max-rep-fraction` | 0.2 | Hard cap on rep set size as fraction of `n` |
| `--k-cap-max` | 256 | Max `K_cap` retry value when k-NN graph fails to connect at 64 |
| `--nystrom-diagonal-loading` | 0.01 | Tikhonov regularisation fraction |
| `--nystrom-degree-normalize / --no-nystrom-degree-normalize` | on | Symmetric Laplacian normalisation of Gram matrix |

**Determinism.** geodesic pins Eigen to a single thread (`Eigen::setNbThreads(1)`) at startup and derives every per-phase RNG seed from `--seed`, so a fixed `--seed` (and identical input ordering) yields bit-identical derep results across runs and machines.

## Pairwise ANI (`geodesic ani`)

`geodesic ani` computes all-pairs FracMinHash ANI between genomes stored in a genopack archive. It loads raw FASTA sequences from the pack in memory — no FASTA extraction to disk — builds FracMinHash sketches (k-mer kept when `hash % c == 0`, the same compression model as skani's `-c`), and computes ANI from AVX2-accelerated sorted-set intersections.

```bash
# All-pairs ANI within a collection
geodesic ani --ql accessions.txt --pack mydb.gpk -t 24 -o ani_results.tsv

# Query set against a separate reference set
geodesic ani --ql queries.txt --rl references.txt --pack mydb.gpk -t 24 -o ani_results.tsv
```

When `--rl` is omitted it defaults to `--ql`, computing self all-pairs (each unordered pair once). Output is a TSV: `query`, `ref`, `ani`, `af` (alignment fraction), `c_ab`, `c_ba` (directional containments). Key options: `--ani-k` (k-mer size, default 21), `-c, --compression` (default 125), `--min-af` (minimum alignment fraction filter), `-t` (threads).

A companion `geodesic validate-ani` compares the cheap OPH Jaccard ANI estimates stored in the pack against this FracMinHash ANI, per stored k-mer size — useful for confirming the derep sketches stay within tolerance. See the [ANI Computation](https://github.com/genomewalker/geodesic/wiki/ANI-Computation) wiki page for the full CLI reference, output schemas, and the OPH-vs-FracMinHash accuracy context.

### Distributed mode

For large collections across multiple nodes, use scatter/gather:

```bash
# 1. Partition input across N workers
geodesic scatter -g genomes.txt --pack mydb.gpk -n 4 -o dist/ --threads 24

# 2. Run each partition (via sbatch, ssh, GNU parallel, etc.)
bash dist/run.sh                    # sequential
parallel -j4 < dist/run.sh         # local parallel
sbatch --array=0-3 --wrap 'sed -n "$((SLURM_ARRAY_TASK_ID+1))p" dist/run.sh | bash'

# 3. Merge shard results
geodesic gather -d dist/ -o dist/merged.grd -p merged
```

Each worker runs an independent `geodesic derep` on its partition. Partitions share no state. The `gather` step merges GRD archives and TSV result files.

### Scatter/Gather options

| Option | Default | Description |
|--------|---------|-------------|
| `scatter -g, --genomes` | required | Accession list (one per line) |
| `scatter --pack` | required | genopack archive (taxonomy resolved from TAXN section) |
| `scatter -n, --partitions` | required | Number of partitions |
| `scatter -o, --output-dir` | required | Output directory for partition files and worker script |
| `scatter --rank` | `g` | Taxonomy rank for LPT partitioning (`g`=genus, `f`=family, `s`=species) |
| `scatter --tmp-dir` | scatter dir | Temporary directory for workers |
| `scatter --threads` | 4 | Threads per worker (baked into generated `run.sh`) |
| `gather -d, --shard-dir` | required | Directory containing shard results |
| `gather -o, --output` | required | Output path for merged GRD file |
| `gather -p, --prefix` | `merged` | Prefix for merged TSV files |

See [Distributed Mode](wiki/Distributed-Mode.md) for the full workflow.

### Incremental updates

`geodesic update` re-runs only taxa that gained new members since the previous run. Add new genomes to the pack with `genopack reindex`, then point `update` at the grown pack and an updated accession list:

```bash
geodesic update -g new_genomes.txt --lock prev_run.lock \
    --pack mydb.gpk \
    --geodf-output updated.geodf --lock-output updated.lock \
    --threads 24
```

`--lock` takes the JSON lock file written by `--lock-output`. `update` diffs accessions in the new list against the prior GEODF, identifies which taxa gained members (taxonomy resolved from pack TAXN), and re-runs only those taxa. Unchanged taxa are copied from the prior GEODF.

## Coverage guarantee

Phase 8 OPH certification guarantees every genome is within the ANI threshold
of its assigned representative in sketch space. Near the default 95% ANI
threshold, OPH estimation error is typically well below 0.1 ANI points for
dense assemblies. Sparse genomes (MAGs with few contigs) switch to
containment-based ANI estimation when the number of occupied OPH bins is
below the sketch size.

## License

MIT
