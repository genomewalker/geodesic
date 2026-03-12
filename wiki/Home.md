# geodesic

A species is not a single genome but a cloud of strains spread across sequence space. Short-read mappers have a practical mapping floor near 92% ANI: strains without a close reference lose coverage and produce biased variant calls. `geodesic` selects a small set of representatives that collectively covers the full strain diversity of a species, ensuring every genome in the collection is within a target ANI of at least one reference.

It does this by sketching genomes with OPH, placing them in an approximate similarity space via Nyström spectral embedding, greedily selecting the most complementary representatives with Farthest Point Sampling, and then verifying coverage back in sketch space. The coverage stopping threshold is inferred from the data (the MST bottleneck of the within-taxon k-NN graph), so broad strain clouds get more representatives and tight clonal populations get fewer. After selection, every non-representative is re-verified by direct OPH Jaccard (with a directional containment check for sparse MAGs), providing an explicit sketch-space coverage guarantee independent of embedding approximation error. Contaminated and chimeric assemblies are flagged before selection using isolation scores and optionally CheckM2 or GUNC, preventing them from anchoring reads from neither parent organism.

## Pages

| Page | Contents |
|------|----------|
| [Background and Motivation](Background-and-Motivation) | The pan-genome framing; tiling sequence space; the Thomson problem analogy |
| [Algorithm](ALGORITHM) | Full algorithm reference: OPH, Nyström spectral embedding, HNSW, Farthest Point Sampling, ANI chain, parameter choices |
| [Contamination Detection](Contamination) | NN-outlier flagging, k-mer diversity z-score, CheckM2 and GUNC integration |
