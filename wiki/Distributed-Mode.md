# Distributed mode

geodesic supports distributed execution across multiple nodes via a scatter/gather pattern. Each worker processes an independent partition with no shared state or coordination.

---

## Architecture

```
genomes.tsv ──→ scatter ──→ part_0.tsv ──→ worker 0 ──→ shard_0.grd
                           part_1.tsv ──→ worker 1 ──→ shard_1.grd
                           part_2.tsv ──→ worker 2 ──→ shard_2.grd
                           ...
                                        gather ──→ merged.grd + merged TSVs
```

Workers are fully independent: no TCP, no shared memory, no NFS locking. Each writes its own GRD shard and TSV result files. The `gather` step merges everything into a single output.

---

## Scatter

Partitions the input by taxonomy using LPT (Longest Processing Time) bin-packing to balance genome counts across partitions. Taxonomy is read from the pack's TAXN section, so the input is a plain accession list:

```bash
geodesic scatter -g genomes.txt --pack mydb.gpk -n 4 -o dist/ --rank g --threads 24
```

| Option | Default | Description |
|--------|---------|-------------|
| `-g, --genomes` | required | Accession list (one per line; taxonomy read from pack TAXN section) |
| `-n, --partitions` | required | Number of partitions (typically = number of worker nodes) |
| `-o, --output-dir` | required | Output directory for partition files and worker script |
| `--pack` | -- | genopack archive path (passed through to worker commands) |
| `--rank` | `g` | Taxonomy rank for grouping (`g`=genus, `f`=family, `s`=species) |
| `--threads` | 4 | Threads per worker (baked into the generated `run.sh`) |
| `--tmp-dir` | scatter output dir | Temporary directory for workers |

Outputs:
- `part_N.tsv` -- per-partition genome lists
- `run.sh` -- executable script with one `geodesic derep` command per partition
- `manifest.tsv` -- machine-readable partition metadata

---

## Running workers

The generated `run.sh` contains self-contained commands. Run them however you like:

```bash
# Sequential
bash dist/run.sh

# GNU parallel (4 at a time)
parallel -j4 < dist/run.sh

# SLURM array job
sbatch --array=0-3 --wrap 'sed -n "$((SLURM_ARRAY_TASK_ID+1))p" dist/run.sh | bash'

# SSH to cluster nodes
i=0; while read cmd; do ssh node$i "$cmd" & ((i++)); done < dist/run.sh
```

Each worker runs a standard `geodesic derep` and produces a `.grd` shard plus TSV result files.

---

## Gather

Merges all shard results into unified outputs:

```bash
geodesic gather -d dist/ -o dist/merged.grd -p merged
```

| Option | Default | Description |
|--------|---------|-------------|
| `-d` | -- | Directory containing shard results |
| `-o` | -- | Output path for merged GRD file |
| `-p` | `merged` | Prefix for merged TSV files |

The GRD merge copies compressed per-taxon sections as-is (no decompress/recompress), renumbers taxon ordinals, and rebuilds global indexes (TIDX, ACCX, STRT).

Merged TSV files: `_derep_genomes.tsv`, `_results.tsv`, `_stats.tsv`, `_diversity_stats.tsv`, `_outliers.tsv`.
