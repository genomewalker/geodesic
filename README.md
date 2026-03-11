# geodesic

**spherical genome embeddings for diverse representative selection**

`geodesic` selects a diverse set of representative genomes per species for reference-based short-read mapping. Unlike traditional dereplication tools that minimise redundancy, `geodesic` maximises the diversity of the retained set, ensuring every genome in the collection is within a target ANI of at least one representative.

[Algorithm visualisation](https://genomewalker.github.io/geodesic/) · [Wiki](https://github.com/genomewalker/geodesic/wiki)

## Algorithm

1. **Sketch** — compute two independent [One-Permutation Hash (OPH)](https://files.ifi.uzh.ch/dbtg/sdbs13/T17.0.pdf) signatures per genome ($k=21$, $m=10{,}000$ bins, seeds 42 and 1337). Each bin holds the minimum hash of all k-mers mapping to it, giving $\Pr[\mathrm{sig}_A[t] = \mathrm{sig}_B[t]] \approx J(A,B)$ (equality holds before densification; after densification, filled bins introduce correlation). Averaging two independent signatures halves Jaccard estimation variance. The per-bin occupancy bitmask enables containment estimation for sparse assemblies.

2. **Embed** — place all genomes in a low-dimensional similarity space using a small set of landmark genomes (anchors):
   - *Anchor selection*: anchors are drawn across the full range of assembly completeness so sparse MAGs and complete genomes are equally represented as landmarks.
   - *Stable similarities*: anchor Gram matrix uses the average Jaccard from two independent OPH sketches, reducing sketch noise.
   - *Regularised spectral map*: symmetric Laplacian normalisation removes hub-anchor bias; light Tikhonov ridge prevents eigenvalue blow-up; any indefinite shift is repaired in-place.
   - *Project all genomes*: each genome is projected by its similarity to the anchors via Nyström extension, then L2-normalised onto the unit sphere.

   The embedding dimension is auto-selected to capture ≥95% of anchor variance. Borderline coverage decisions are re-checked with direct OPH Jaccard (Phase 7).

3. **Index** — build an [HNSW](https://arxiv.org/abs/1603.09320) nearest-neighbour index on the sphere for sub-linear candidate retrieval.

4. **Score** — compute isolation scores (mean angular distance to $k_\text{iso}=10$ nearest neighbours) and build the minimum spanning tree of the k-NN graph via Kruskal's algorithm. The edge budget $K_\text{cap} = \min(64, n-1)$ is determined by a two-phase adaptive scan. Phase A sweeps k-NN columns from $k=1$ to $K_\text{cap}$ via DSU, recording $k_\text{conn}$: the first $k$ at which the core graph becomes connected (−1 if never). Phase B probes a ladder $\{1,2,3,4,6,8,12,16,24,32,48,64\}$ starting from $k_\text{conn}$, picking the smallest $k$ where the bottleneck (MST max edge) is within 3% of the $K_\text{cap}$ reference value — this is $k_\text{stable}$. Using $k_\text{conn}$ alone is insufficient because the first-connection edge is often a brittle bridge; the probe identifies where the bottleneck has stabilised. The longest MST edge at $k_\text{stable}$ sets the diversity threshold $\theta$: the minimum inter-strain scale at which the proximity graph becomes connected. This is inferred from the data, not a fixed parameter. Flag contamination candidates using a FastMCD robust estimator (minimum-variance $h$-subset, $h = \lceil 0.75n \rceil$) on the isolation score distribution, resistant to the long right tail introduced by contaminated genomes. When the k-NN graph fails to connect at $K_\text{cap}=64$, retries at 128 and 256 with full HNSW requery and raised ef\_search (configurable via \texttt{--k-cap-max}).

5. **Select** — quality-weighted [Farthest Point Sampling (FPS)](https://en.wikipedia.org/wiki/Farthest-first_traversal) on the unit sphere, a greedy $\theta$-cover: each step adds the genome farthest from its nearest representative, with fitness weighted by $q_i/100 \cdot \sqrt{L_i/L_m}$ (CheckM2 quality × genome size). Stops when every genome is within $\theta$ of some representative. Quality weighting and batch processing mean the formal Gonzalez (1985) 2-approximation bound for unweighted k-center does not apply; coverage is evaluated empirically.

6. **Merge** — coalesce representatives within $d_{\mathrm{min}}$ via Union-Find.

7. **Verify** — for non-representatives with embedding distance in $[\theta(1-\varepsilon),\,\theta)$, check the top-3 nearest representatives using exact dual-sketch OPH Jaccard. Promote only if no representative is within $\theta$ in sketch space.

8. **Certify** — universal sketch-space coverage pass: every non-representative is verified against its assigned representative by direct OPH Jaccard. Certification threshold $\tau = q/(2-q)$ where $q = \mathrm{ANI}^k$. Any genome failing this check is promoted to a representative, providing an explicit sketch-space coverage guarantee independent of Nyström approximation error. OPH estimation error depends on real-bin occupancy and Jaccard; near the default 95% ANI threshold with dense sketches it is typically well below 0.1 ANI points, but sparse genomes are less stable. Sketch-asymmetric pairs (MAG vs. complete genome, $n_\text{real,small}/n_\text{real,large} < 0.5$ by occupied OPH bins) are additionally checked by directional containment: the fraction of the small genome's real bins that match the large genome must exceed $q = \mathrm{ANI}^k$.

ANI thresholds are derived from Jaccard via the Mash formula: $\mathrm{ANI} = \left(\frac{2J}{1+J}\right)^{1/k} \times 100$.

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

### Output

Results written to the working directory (or `--out-dir`):
- `<prefix>_derep_genomes.tsv` — all genomes with representative assignment and ANI to rep
- `<prefix>_results.tsv` — per-taxon summary (n_genomes, n_reps, method, runtime)
- `<prefix>_diversity_stats.tsv` — coverage and diversity metrics per taxon
- `<prefix>_contamination.tsv` — flagged anomalous genomes
- `<prefix>.db` — DuckDB database with full results

### Key options

| Option | Default | Description |
|--------|---------|-------------|
| `--threads` | 1 | Total CPU threads |
| `--ani-threshold` | 95.0 | ANI threshold (%) — acts as cap; actual threshold inferred from data |
| `--geodesic-dim` | 256 | Nyström embedding dimension |
| `--geodesic-kmer-size` | 21 | k-mer size for OPH sketching |
| `--geodesic-sketch-size` | 10000 | OPH sketch size (bins) |
| `--geodesic-auto-calibrate` | on | Auto-select embedding tier from random ANI sample |
| `--geodesic-calibration-pairs` | 50 | Pairs sampled for tier selection |
| `--checkm2` | — | CheckM2 TSV for quality-weighted selection |
| `--gunc-scores` | — | GUNC TSV to exclude chimeric assemblies from selection |
| `-z` | 2.0 | Z-score threshold for contamination detection |
| `--nystrom-diagonal-loading` | 0.01 | Tikhonov regularisation fraction |
| `--nystrom-degree-normalize` | on | Symmetric Laplacian normalisation of Gram matrix |
| `--embedding-db` | — | Persistent embedding store for incremental updates |
| `--incremental` | off | Reuse existing embeddings, embed only new genomes |

### Resuming

`geodesic` writes progress to DuckDB as it runs. Resume an interrupted run by pointing to the same database:

```bash
geodesic derep -t genomes.tsv -d my_run.db --threads 24 -p my_run
```

Already-completed taxa are skipped automatically.

## Performance

| Species | Genomes | Representatives | Reduction | Runtime | Cov. mean ANI | Cov. min ANI | Threads |
|---------|---------|-----------------|-----------|---------|---------------|--------------|---------|
| *E. coli* | 233,166 | 1,359 | 99.4% | 15 min | 99.96% | 99.91% | 24 |
| *S. enterica* | 367,440 | 982 | 99.7% | 18 min | 99.99% | 99.96% | 24 |

Coverage: Phase 8 OPH certification guarantees every genome is within the ANI threshold of its assigned representative in sketch space. Near the default 95% ANI threshold, OPH estimation error is typically well below 0.1 ANI points for dense assemblies.

## License

MIT
