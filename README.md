# geodesic

**spherical genome embeddings for diverse representative selection**

`geodesic` selects a diverse set of representative genomes per species for reference-based short-read mapping. Unlike traditional dereplication tools that minimise redundancy, `geodesic` maximises the diversity of the retained set, ensuring every genome in the collection is within a target ANI of at least one representative.

[Algorithm visualisation](https://genomewalker.github.io/geodesic/) · [Wiki](https://github.com/genomewalker/geodesic/wiki)

## Algorithm

1. **Sketch** — compute two independent [One-Permutation Hash (OPH)](https://papers.nips.cc/paper/2012/hash/eaa32c96f620053cf442ad32258076b9-Abstract.html) signatures per genome ($k=21$, $m=10{,}000$ bins, seeds 42 and 1337). Each bin holds the minimum hash of all k-mers mapping to it, giving $\Pr[\mathrm{sig}_A[t] = \mathrm{sig}_B[t]] = J(A,B)$. Averaging two independent signatures halves Jaccard estimation variance. The per-bin occupancy bitmask enables containment estimation for sparse assemblies.

2. **Embed** — project all genomes onto a unit sphere via [Nyström spectral embedding](https://en.wikipedia.org/wiki/Nystr%C3%B6m_method) with four robustness improvements:
   - *Stratified anchors*: anchors stratified by fill fraction $f_i = n_\text{real}/m$ across 5 quantile strata, ensuring sparse MAGs and complete genomes are equally represented as landmarks.
   - *Averaged kernel*: anchor Gram matrix uses dual-sketch average $K[i,j] = (J_1+J_2)/2$, halving Jaccard estimation variance.
   - *Containment blend*: when $f_i < 0.2$, Jaccard is blended with bin co-occupancy $\max(C(A{\to}B),\,C(B{\to}A))$ where $C(A{\to}B) = |B_A \cap B_B|/|B_A|$, weighted by $\alpha = 1 - f_i/0.2$.
   - *Regularisation*: symmetric Laplacian normalisation ($D^{-1/2} K D^{-1/2}$) removes hub-anchor bias; Tikhonov regularisation ($\lambda = 0.01 \cdot \max(\bar{K}_\text{diag}, 10^{-4})$) prevents eigenvalue blow-up.

   The Gram matrix is eigendecomposed; all genomes are projected onto the top $d$ eigenvectors via Nyström extension and L2-normalised. $d$ is auto-selected to capture ≥95% of variance. Borderline decisions use exact OPH Jaccard (Phase 7).

3. **Index** — build an [HNSW](https://arxiv.org/abs/1603.09320) nearest-neighbour index on the sphere for sub-linear candidate retrieval.

4. **Score** — compute isolation scores (mean angular distance to $k=10$ nearest neighbours). Flag anomalous genomes when isolation score $> \mu + z\sigma$. Derive the diversity threshold from the NN distribution: $\theta = \min(\mathrm{NN}_{P95},\,\theta_\mathrm{ANI})$.

5. **Select** — quality-weighted [Farthest Point Sampling (FPS)](https://en.wikipedia.org/wiki/Farthest-first_traversal) on the unit sphere. FPS is a greedy 2-approximation to the k-center problem: each step adds the genome farthest from its nearest representative, with fitness weighted by $q_i/100 \cdot \sqrt{L_i/L_m}$ (CheckM2 quality × genome size). Stops when every genome is within $\theta$ of some representative.

6. **Merge** — coalesce representatives within $d_\text{min}$ via Union-Find.

7. **Verify** — for non-representatives with embedding distance in $[\theta(1-\varepsilon),\,\theta)$, check the top-3 nearest representatives using exact dual-sketch OPH Jaccard. Promote only if no representative is within $\theta$ in sketch space.

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
| `--chimera-zscore` | 3.0 | K-mer diversity z-score threshold for chimera flagging |
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

| Scale | Genomes | Representatives | Runtime | Threads |
|-------|---------|-----------------|---------|---------|
| *E. coli* | 233,166 | 12,500 | 14 min | 24 |
| *S. enterica* | 367,440 | 4,252 | 23 min | 24 |
| Full GTDB r226 | 5,195,094 | ~206,000 | ~6.8 h | 96 |

Coverage validation on *E. coli* (233k genomes, exact skani ANI): **100% of genomes within 95% ANI of their nearest representative**. Minimum coverage ANI: 99.93%. Mean: 99.98%.

## License

MIT
