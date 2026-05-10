# Changelog

## 1.0.0 — 2026-05-10

Initial stable release.

### Highlights

**Thread model**
- Replaced OpenMP with BS::thread\_pool throughout (21 call sites). `--threads N` now spawns exactly N OS threads instead of 2N, giving predictable CPU accounting on HPC nodes.

**Memory**
- `mmap` regions in genopack now call `MADV_DONTNEED` after use, releasing pages back to the OS immediately. Peak RSS on the full GTDB r232 run dropped from 521 GB to 63 GB at 24 threads.
- Wave-level memory management: `malloc_trim` is called after each sketch wave to return freed heap pages to the OS between phases.

**CLI**
- `--genomes` accepts a plain accession list; `--pack` accepts a `.gpk` archive. Both replace the old `--input` TSV interface.
- Per-taxon automatic k selection: geodesic picks the best available k-mer size from the pack index rather than requiring a fixed global k.

**Performance (GTDB r232, 5,238,926 genomes)**

| Metric | Value |
|---|---|
| Representatives | 886,507 (83.1% reduction) |
| Runtime | ~65 min at 24 threads |
| Peak RSS | ~63 GB at 24 threads |
| Throughput | ~1,340 genomes/sec |

**Other**
- Probe always uses the smallest available k (cache-hot path) instead of the largest.
- `PreloadedPackReader` falls through to disk when probe size is smaller than the stored size.
- `derep_genomes.tsv` output now includes `cluster_rep` and `nn_dist` columns.
- Wave-level dominant\_k uses `avail_ks.front()` for consistency with the probe path.
