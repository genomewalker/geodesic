# ANI shard I/O and CPU threads

Status as of 2026-10-22: readers only `pread`; decoding and sketching run on the `-t` pool. On 815 Bacteroidaceae genomes (`-t 32`, default 8 readers) the pack ANI took 22.3 s warm vs 46.8 s for skani on sketches, and 75.7 s cold vs 83.3 s for skani cold (dandycomp02fl, fincore < 1 GB cached), with output identical to main.

`geodesic ani --io-threads R` controls concurrent compressed-shard readers,
independently of `-t`. Explicit values are capped at 64 and at the number of
requested shards, so `-t 32 --io-threads 16` runs sixteen readers. Zero chooses
`max(1, min(8, t/4))` readers.

Readers perform only allocation and `pread`, traversing contiguous bands of
requested shards across archives. A separate pool of exactly `t` workers opens
shards, decodes individual genomes, and builds sketches. OpenMP teams are disabled
inside these workers, including explicit teams requested by library code.
Waiting I/O readers consume no compute slots; decode and sketch work together
never exceed `t` workers. The pair phase subsequently uses the full CPU budget.

A shared semaphore bounds all compressed buffers, including reads in progress,
queued shards, and shards still referenced by workers, to `2R`. A compressed
buffer is released after its last genome finishes. At most `t` decoded genomes
are live, rather than one decoded batch per reader; per-worker decoder caches
also contribute to memory. The bound scales with shard size, and explicit high
reader counts can increase RSS. Missing accessions are ignored and duplicate
requests retain their caller indices. Reader and callback errors are joined and
reported to the caller.

The pack adapters expose borrowed file descriptors and shard offsets from
public genopack APIs. The pack must outlive the visit. Genopack is unchanged.
The existing batch visitors and derep FASTA extraction retain their behavior.
ANI logs read+sketch, index, pairs, and write phase wall times.

Measured defaults and RSS results are recorded in the round-3 checks report.
