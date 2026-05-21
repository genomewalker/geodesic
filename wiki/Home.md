# geodesic

A species is not a single genome but a cloud of strains spread across sequence space. Short-read mappers have a practical mapping floor near 92% ANI: strains without a close reference lose coverage and produce biased variant calls. `geodesic` selects a small set of representatives that collectively covers the full strain diversity of a species, ensuring every genome in the collection is within a target ANI of at least one reference.

It does this by sketching genomes with OPH, placing them in an approximate similarity space via Nyström spectral embedding, greedily selecting the most complementary representatives with Farthest Point Sampling, and then verifying coverage back in sketch space. The coverage stopping threshold is inferred from the data (the MST bottleneck of the within-taxon k-NN graph), so broad strain clouds get more representatives and tight clonal populations get fewer. After selection, every non-representative is re-verified by direct OPH Jaccard (with a directional containment check for sparse MAGs), providing an explicit sketch-space coverage guarantee independent of embedding approximation error. Contaminated and chimeric assemblies are flagged before selection using isolation scores and optionally CheckM2 or GUNC, preventing them from anchoring reads from neither parent organism.

## Quick start

```bash
# Dereplicate a collection, reading sequences, sketches, and taxonomy from a pack
geodesic derep -g genomes.txt --pack mydb.gpk --threads 24 -p my_run

# All-pairs FracMinHash ANI over genomes in a pack (no FASTA extraction to disk)
geodesic ani --ql genomes.txt --pack mydb.gpk -t 24 -o ani_results.tsv

# Check that the pack's OPH sketches track exact ANI
geodesic validate-ani -g genomes.txt --pack mydb.gpk -n 1000 -o ani_validation.tsv
```

## Subcommands

| Command | Purpose |
|---------|---------|
| `derep` | Select diverse representatives per taxon (the main pipeline) |
| `ani` | All-pairs FracMinHash ANI from a pack, in memory — see [ANI Computation](ANI-Computation) |
| `validate-ani` | Compare in-pack OPH ANI estimates to FracMinHash ANI — see [ANI Computation](ANI-Computation) |
| `scatter` / `gather` | Distributed derep across nodes — see [Distributed Mode](Distributed-Mode) |
| `update` | Incremental re-derep of taxa that gained members |

## Pages

| Page | Contents |
|------|----------|
| [Background and Motivation](Background-and-Motivation) | The pan-genome framing; tiling sequence space; the Thomson problem analogy |
| [Algorithm](ALGORITHM) | Full algorithm reference: OPH, Nyström spectral embedding, HNSW, Farthest Point Sampling, ANI chain, parameter choices |
| [ANI Computation](ANI-Computation) | The `ani` and `validate-ani` subcommands: in-pack FracMinHash ANI, OPH-vs-FracMinHash accuracy validation, CLI reference |
| [Outlier Detection](Outlier-Detection) | NN-outlier flagging, k-mer diversity z-score, CheckM2 and GUNC integration |
| [Distributed Mode](Distributed-Mode) | Scatter/gather for multi-node execution |
| [Derep Output](Derep-Output) | Byte-level spec of the `.gpd` Geodesic Derep Archive |
