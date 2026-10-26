# ANI shard I/O and CPU threads

`geodesic ani --io-threads R` controls concurrent compressed-shard readers,
independently of `-t`. Explicit values are capped at 64 and at the number of
requested shards, so `-t 32 --io-threads 8` runs eight readers. Zero chooses
16 readers whatever `-t` is.

Readers perform only allocation and `pread`, traversing contiguous bands of
requested shards across archives. Each shard read covers the header, the genome
directory and the ranges `ShardReader` touches for the requested genomes: the
dictionary, the codec reference blobs, and each requested blob with its
checkpoints. Ranges are merged across gaps of at most 1 MiB. A separate pool of exactly `t` workers opens
shards, decodes individual genomes, and builds sketches. OpenMP teams are disabled
inside these workers, including explicit teams requested by library code.
Waiting I/O readers consume no compute slots; decode and sketch work together
never exceed `t` workers. The pair phase subsequently uses the full CPU budget.

A shared semaphore bounds all compressed buffers, including reads in progress,
queued shards, and shards still referenced by workers, to `2R`. Each buffer is
allocated at full shard size but only its read ranges are touched. A compressed
buffer is released after its last genome finishes. At most `t` decoded genomes
are live, rather than one decoded batch per reader; per-worker decoder caches
also contribute to memory. The bound scales with shard size, and explicit high
reader counts can increase RSS. Missing accessions are ignored and duplicate
requests retain their caller indices. Reader and callback errors are joined and
reported to the caller.

The pack adapters expose borrowed file descriptors and shard offsets from
public genopack APIs. The pack must outlive the visit.
The existing batch visitors and derep FASTA extraction retain their behavior.
ANI logs read+sketch, index, pairs, and write phase wall times.
