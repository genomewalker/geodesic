# ANI reader threads

`geodesic ani --io-threads R` controls concurrent shard readers within the total
CPU budget `-t`. Zero (the default) selects `max(1, min(4, t/8))`; explicit values
are capped at `max(1, t/8)`. At eight CPUs or fewer, one reader uses the full
budget for sketching, with no CPUs reserved for waiting readers.

Readers traverse contiguous bands of requested archive/shard IDs. Each reader
reuses its CPU share for decoding and sketching in successive OpenMP regions.
Genopack explicitly requests up to eight decode threads, so each concurrent
reader needs at least eight CPUs; below eight CPUs decoding runs serially.
Sketching uses `floor(t/R)` threads per reader. These teams never nest.

Each callback finishes sketching before its reader advances. Consumed FASTAs
are released immediately; at most R decoded batches and 2R compressed prefetch
buffers are retained, along with per-thread decoder caches. The automatic
four-reader cap reduces compressed-buffer memory compared with eight readers.
With multiple readers, their threads exit after read+sketch and release their
decoder caches.
Pair computation then uses the full `-t` budget. Output values and pair
orientation are unchanged.

The log reports wall seconds for read+sketch, index, pairs, and write.
The optional derep FASTA extraction call sites retain their existing reader;
sharing this CPU-budget design there requires restructuring their callbacks.
