# Changelog

## 1.1.0 — 2026-07-18

### Dereplication quality gating

- **`--skip-lq`** hard-excludes LQ genomes from the candidate pool using the pack's `quality_tier_u8` (genopack's completeness-only tier). Without it, LQ genomes are still *included* — quality only downweights representative selection via a bounded `0.5 + 0.5·q` FPS factor, so an LQ singleton or best-in-cluster genome can still be a representative (128,758 of the 1,300,606 ungated GTDB r232 representatives are LQ). With `--skip-lq`, the corrected tier holds out 535,855 LQ genomes and the derep returns 1,164,985 representatives, more complete than the ungated set (mean `comp_eff` 0.777 → 0.837).
- **`--min-completeness` (alias `--min-cr`)** gates on intrinsic `completeness_effective` (`marker → aamer_core → post_decontam`), not pangenome breadth, so finished isolates in diverse genera are not penalised.
- **Continuous quality score for representative ranking.** When the pack carries a genopack QUAL section (the default for `derep --pack`), representatives are ranked by genopack's completeness-only `quality_score`, overriding the CheckM2 `completeness − 5×contamination` form (which remains the fallback).
- **`--resume`**: per-wave checkpoints for crash recovery.
- Genomes absent from the SKCH section are kept as self-representatives rather than dropped.
- Fixed `quality_tier_for_accession` ID aliasing across multipart archives.

### Documentation

- Corrected the quality/contamination wiki to the decoupled model (Contamination.md, ALGORITHM.md, Outlier-Detection.md): completeness-only tier, three reported D/S/G contamination channels, current GTDB r232 tier counts (HQ 55.16% / MQ 39.22% / LQ 5.62%), and removed QUAL columns that no longer exist in the `check` output.

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
