# ANI reader threads

`geodesic ani --io-threads R` controls concurrent shard readers within the total
CPU budget `-t`. Zero (the default) selects `max(1, min(8, t/4))`; explicit values
are capped at `max(1, t-1)`. With `-t 1`, reading and sketching are serial.

Readers traverse contiguous bands of requested archive/shard IDs. A single
OpenMP team executes reader and sketch tasks; nested genopack decoding is
serialized. Readers wait for their batch's sketch tasks before advancing, so
at most R decoded batches are retained, alongside genopack's compressed prefetch
buffers. Pair computation then uses the full `-t` budget. Output values and
pair orientation are unchanged.

The log reports wall seconds for read+sketch, index, pairs, and write.
The optional derep FASTA extraction call sites retain their existing reader;
sharing this CPU-budget design there requires restructuring their callbacks.
