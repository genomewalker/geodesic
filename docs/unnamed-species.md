# Clustering genomes without a species name

`geodesic derep` clusters unnamed genomes by default. This includes bare `s__`
(and species-only `S__`) ranks and older accession-filled species ranks. A strain
`S__` does not override an existing `s__` species.

```sh
geodesic derep --pack reps_pack -g genomes.list -o output --threads 32 \
  --skani-bin /path/to/skani --species-radii gtdb_radii.tsv
```

The clustering stage uses the skani executable, independently of geodesic's
sketch-based ANI implementation. It extracts FASTA files from the pack into
private directories under `--tmp-dir`. Each block is removed after use.

Unnamed genomes first join their best named reference in the same family when
ANI meets that reference's radius and either alignment fraction reaches 50%.
`--species-radii` accepts GTDB's tab-separated species, representative accession,
and radius file; omitted radii default to 95. The best AF-passing hit is tested
once, even if a lower-ranked hit would pass its own radius.

The remaining genomes undergo greedy clustering within families, followed by a
centroid pass within each phylum. An edge requires ANI >= 95 and maximum AF >=
50%. Existing representatives precede other genomes, followed by quality tier,
completeness, contig count, NCBI accession status, and accession. Members join
the centroid with the best ANI, then AF, then accession. Radii are bounded to
95–97 ANI using competing representatives. Fresh skani distances recheck
membership; failing members move to a passing representative or become
singletons. Final representatives are checked for remaining species edges.
Cross-phylum comparisons are deliberately absent and logged.

Each resulting cluster is a separate dereplication taxon. Names use the majority
assigned genus and representative accession, with deterministic SHA1-based
fallbacks and collision handling. Clusters without an assigned genus retain
bare genus, species and strain ranks and are excluded from naming.

`species_assignments.tsv` accompanies the run outputs. Columns are `accession`,
`species`, `taxonomy`, `cluster`, `rep`, `genus`, `radius`, `ani_to_rep`,
`af_to_rep`, and `source` (`anchor` or `cluster`). The third column contains the
full taxonomy because `genopack taxonomy patch` reads that column in wider
TSV files. It can be passed directly as `--patch`. Unnamed clusters without an
assigned genus have no rows in this file.

Use `--unnamed-species singleton` to retain the previous taxonomy-string
grouping behavior. This option also avoids invoking skani.

### Matched S2 ordering and external GTDB anchors

With `--unnamed-species cluster`, `--ordering-tsv FILE` reads a header TSV keyed
by `accession`. Its required columns are `quality_tier` (`HQ`, `MQ`, `LQ`),
`completeness_effective`, `n_contigs`, and `ncbi` (0 or 1). Listed genomes use
these values verbatim, including effective completeness above 1; unlisted genomes
use pack metadata. Representatives sort by prior representative status, tier,
descending effective completeness, contig count, descending NCBI flag, accession.
A prior representative can be replaced by a better tier, or the same tier with
at least 0.05 greater effective completeness.

`--anchor-skani-db DIR --species-radii FILE` searches an external skani sketch
database instead of packed named representatives. `--anchor-skani-bin BIN`
(default `skani`) selects the search executable independently of `--skani-bin`,
which continues to run clustering triangles and distances. The search command is
`skani search -d DIR --ql LIST -t THREADS --min-af 0`.
The radii TSV has species, reference accession (optionally prefixed `RS_`/`GB_`),
and ANI radius columns. Unknown references and hits with both AF values below 50
are ignored. Among remaining hits, the greatest (ANI, maximum AF) wins; an exact
tie retains the first hit. Only that reference's radius decides anchoring.
Accepted genomes leave clustering and receive its GTDB species. AF-passing hits
also supply the GTDB neighbour ANI used for de novo representative radii.
`anchor_hits.tsv` records the top AF-passing evidence and `anchor_search_seconds.txt`
records search wall time separately from extraction and clustering. Singleton mode
without these options retains its existing behavior.
