# Background and Motivation

## The Problem: Tiling Sequence Space

A bacterial species does not have a single genome. It has a [pan-genome](https://doi.org/10.1073/pnas.0506758102) (Tettelin et al. 2005): the union of all sequences present in any member of the species. Different strains carry different gene content, different alleles of shared genes, and different patterns of indels and substitutions. For short-read mapping to capture variant diversity, the reference set must *tile* the sequence space of the pan-genome at a resolution finer than the read length.

Concretely: short-read mappers have a practical [mapping floor near 92% ANI](https://instrain.readthedocs.io/en/latest/important_concepts.html#b-ensure-that-genomes-aren-t-too-distinct-from-one-another); below that, reads from divergent loci fail to align entirely. Because ANI is a genome-wide average, a reference at 95% ANI still has regions divergent enough to lose reads; a 3% buffer is needed to keep coverage uniform. A strain with no reference within 95% ANI will have systematically missing coverage and biased variant calls.

The target is to ensure that **every genome in the collection is within some ANI threshold of at least one representative**. This is the [k-center problem](https://en.wikipedia.org/wiki/Metric_k-center): choose $k$ centres to minimise the maximum distance from any point to its nearest centre.

## Pan-Genome Openness

Species vary enormously in how broadly their genomes are distributed across sequence space. *Escherichia coli* is the canonical example of an open pan-genome: two strains can differ by 30% or more of their gene content while sharing 16S rRNA identity above 97%. Accessory genome variation (virulence factors, antibiotic resistance genes, mobile elements) is clinically relevant but invisible to a single-reference mapping strategy.

This openness directly determines how many representatives are needed. geodesic measures it from the data: after building the HNSW k-NN graph, it constructs the minimum spanning tree of the non-outlier genomes and takes the longest MST edge as the FPS stopping threshold. That edge is the minimum angular distance at which the proximity graph becomes connected: the natural inter-strain scale of the taxon. Open pan-genomes have long MST bottleneck edges and produce more representatives; tight clonal populations have short edges and need few. No fixed parameter is required; the right number of representatives falls out of the structure of the data.

## The Thomson Problem Analogy

The representative selection problem can be stated geometrically. Embed all genomes as points on a high-dimensional unit sphere $S^{d-1}$, where the angular distance between two points corresponds to genomic dissimilarity. The goal is to select $k$ points that best cover the sphere, ensuring every genome falls within some angular radius of its nearest representative.

The [Thomson problem](https://en.wikipedia.org/wiki/Thomson_problem) offers a useful intuition: $N$ charges on a sphere, repelling each other, spread to maximise their mutual separation. The analogy captures the repulsion principle: each new representative should be placed as far as possible from those already selected. [Farthest Point Sampling](https://doi.org/10.1016/0304-3975(85)90224-5) (Gonzalez 1985) implements exactly this greedy rule and is a 2-approximation to the k-center objective: the maximum coverage radius is at most twice optimal.

Note: the Thomson problem minimises total electrostatic energy ($\sum_{i \neq j} 1/r_{ij}$), not the k-center objective. The two problems have different solutions in general; the analogy holds as intuition, not as a mathematical equivalence.

## Scale and the Sphere Approach

The practical motivation was scale. A full-scale run spans 5.2 million genome assemblies collected from NCBI, [SPIRE](https://spire.embl.de), [MGnify](https://www.ebi.ac.uk/metagenomics) and other repositories, mapped to the [GTDB](https://gtdb.ecogenomic.org) r226 taxonomy across 130,000 species-level taxa. For a species with $n = 100{,}000$ members, $O(n^2)$ pairwise ANI comparisons are infeasible.

The sphere approach avoids all-vs-all pairwise comparisons. Each genome is sketched once and projected to a unit vector via [Nyström spectral embedding](https://en.wikipedia.org/wiki/Nystr%C3%B6m_method): the anchor Gram matrix is built from Jaccard similarities among a small anchor set ($n_\text{anchors} \ll n$), eigendecomposed once, and each remaining genome is projected into the same spectral basis at $O(n_\text{anchors})$ cost. Total comparison cost is $O(n_\text{anchors}^2 + n \cdot n_\text{anchors})$, linear in $n$ for fixed anchor count, replacing the $O(n^2)$ all-pairs baseline. Representative selection then operates on precomputed embedding vectors via [HNSW](https://doi.org/10.1109/TPAMI.2018.2889473) (Malkov & Yashunin 2018), with OPH-sketch verification only for borderline pairs near the ANI threshold.

The key technical insight is that OPH signatures encode k-mer set structure in a form amenable to Nyström spectral embedding: dual-sketch averaged Jaccard similarities give the anchor Gram matrix, and the Mash distance formula relates Jaccard to ANI, completing the pipeline from raw sequence to a quantitative diversity score with no training data.
