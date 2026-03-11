# geodesic: Algorithm Reference

**Spherical genome embeddings for diverse representative selection**

---

## Overview

`geodesic` selects a diverse set of representative genomes per taxon for reference-based short-read mapping. It uses OPH sketching, Nyström spectral embedding, farthest-point sampling, Union-Find merge, and OPH-sketch verification to retain diversity while keeping every genome within an ANI threshold of its nearest representative (measured in OPH-sketch Jaccard space, not exact ANI).

The pipeline has seven phases:

```
Sketch → Embed → Index → Score → Select → Merge → Verify
```

---

## Phase 1: OPH Sketching

### One-Permutation Hashing

For each genome, geodesic computes a [One-Permutation Hash (OPH)](https://papers.nips.cc/paper/2012/hash/eaa32c96f620053cf442ad32258076b9-Abstract.html) signature of $m = 10{,}000$ bins using k-mers of length $k = 21$.

**Canonical k-mer selection.** For each position in the genome, both the forward k-mer and its reverse complement are encoded as a 64-bit integer (2 bits per base, A=0/C=1/G=2/T=3). The canonical k-mer is the lexicographic minimum of the two encodings, selected by a branchless comparison:

```c
fwd = ((fwd << 2) | base) & k_mask
rev = (rev >> 2) | ((3 ^ base) << rev_shift)
canonical = fwd ^ ((fwd ^ rev) & -(uint64_t)(fwd > rev))   // = min(fwd, rev)
```

Ambiguous bases (N and other non-ACGT characters) reset the rolling state, discarding the current k-mer window.

**OPH hash.** A single 64-bit hash is computed from the canonical encoding and the sketch seed using a WyHash-based mixing step:

```
h = wymix(canonical XOR (seed + P0),  canonical XOR P1)
    where P0 = 0xa0761d6478bd642f, P1 = 0xe7037ed1a0b428db  (WyHash v4 constants)
    wymix(a, b) = lo64(a*b) XOR hi64(a*b)
```

sig1 uses seed = 42, sig2 uses seed = 1337. Both the bin index and the per-bin value are derived from this one hash:

$$t = \left\lfloor \frac{h \cdot m}{2^{64}} \right\rfloor, \qquad \text{sig}[t] = \min\!\left(\text{sig}[t],\ h \gg 32\right)$$

The stored uint32 value is truncated to uint16 at storage time (retaining bits 32–47 of $h$). A single hash call per k-mer determines both bin index and comparison value.

**Densification.** After scanning all k-mers, empty bins are filled by nearest-neighbour propagation, following the OPH densification scheme of Li & König (2012):

```
Forward:  if sig[t] = EMPTY and sig[t-1] ≠ EMPTY: sig[t] = SplitMix64(sig[t-1] XOR t)
Backward: if sig[t] = EMPTY and sig[t+1] ≠ EMPTY: sig[t] = SplitMix64(sig[t+1] XOR t)
```

The specific mixing function (SplitMix64 with a bin-index XOR) is a local implementation choice. Densified bins carry no independent information about k-mer overlap — their values are deterministic functions of real neighbours. The collision probability below applies only to real (pre-densification) bins.

**Jaccard property.** For any bin $t$ that is real in at least one genome:

$$\Pr\!\left[\text{sig}_A[t] = \text{sig}_B[t]\right] \approx J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

(Equality holds before densification; after densification, filled bins introduce correlation and the identity is approximate.)

where $A$ and $B$ are the sets of distinct canonical k-mers. Over $m$ bins, the fraction of collisions is an unbiased estimator of $J$ with approximate variance:

$$\operatorname{Var}(\hat{J}) \approx \frac{J(1-J)}{m_\text{real}}$$

where $m_\text{real}$ is the number of bins that are real in at least one of $A$ or $B$. For complete bacterial genomes ($m_\text{real} \approx m = 10{,}000$), standard error is $< 0.01$ at $J = 0.3$. For incomplete MAGs with $m_\text{real} \ll m$, variance is higher.

### OPH vs Bottom-k MinHash

[Bottom-k MinHash](https://en.wikipedia.org/wiki/MinHash) keeps the $k$ smallest hash values across the entire genome — a one-pass streaming algorithm with fixed sketch size $k$. OPH partitions the hash space into $m$ bins and keeps one value per bin. OPH is preferred here for two reasons:

1. **Single pass, fixed memory**: one pass fills all $m$ bins regardless of genome content.
2. **Free bitmask**: the per-bin occupancy bitmask (bit $t = 1$ iff bin $t$ has a real k-mer) is obtained at no extra cost. This bitmask is used for bin co-occupancy estimation in Phase 2 and for fill-fraction computation.

The weakness of OPH is heteroskedastic variance: genomes with few real bins have higher estimation error. The dual-sketch strategy below partly addresses this.

### Dual OPH Sketches

Two independent OPH signatures (sig1, sig2) are computed per genome using seeds 42 and 1337. Because different seeds produce different WyHash outputs, the two signatures have completely different bin assignments — they are independent sketches, not two values for the same bins.

The anchor Gram matrix uses the dual-sketch average (see Phase 2):

$$K[i,j] = \frac{J_1(\text{anchor}_i, \text{anchor}_j) + J_2(\text{anchor}_i, \text{anchor}_j)}{2}$$

For two independent $J$ estimators, averaging halves the variance:

$$\operatorname{Var}\!\left(\frac{J_1 + J_2}{2}\right) = \frac{J(1-J)}{2\, m_\text{real}}$$

**16-bit storage.** Each per-bin value (a uint32) is stored as a uint16 (low 16 bits), halving RAM: $10{,}000 \text{ bins} \times 2 \text{ bytes} = 20\ \text{KB}$ per genome. Truncation increases the probability of false matches. The [b-bit MinHash bias correction](https://dl.acm.org/doi/10.1145/1989323.1989399) (Li & König 2011) corrects the raw collision fraction:

$$\hat{J}_\text{corr} = \max\!\left(0,\ \frac{\hat{J}_\text{raw} - 2^{-16}}{1 - 2^{-16}}\right)$$

**Lazy sig2 materialisation.** sig2 is not computed for all $n$ genomes at sketch time. It is materialised on demand only for the $\sim$512 anchor genomes (needed for the dual-sketch Gram matrix) and for borderline verification candidates (Phase 7). Non-anchor Nyström extension uses sig1 only.

**Buffer cache.** The decompressed FASTA buffer for each genome (up to a 2.5 GB aggregate per taxon) is held in memory after the sig1 sketch pass. Anchor sig2 materialisation reads from this in-memory buffer rather than the NFS mount, avoiding a second disk read for anchor genomes.

### Fill Fraction

The fill fraction $f_i = n_{\text{real},i} / m$ is the fraction of bins with at least one real k-mer before densification. Under a Poisson occupancy model for the bin-assignment process:

$$\mathbb{E}[f_i] \approx 1 - e^{-|G_i|/m}$$

where $|G_i|$ is the number of distinct canonical k-mers. Fill fraction saturates quickly for complete bacterial genomes ($|G| \sim 10^6$ k-mers, $m = 10{,}000$ → $f \approx 1$). Its practical role is identifying highly incomplete assemblies ($f \ll 0.2$) where OPH variance is elevated and containment-based corrections are needed.

---

## Phase 2: Nyström Spectral Embedding

### Motivation

Exact pairwise Jaccard over $n$ genomes requires $O(n^2 m)$ operations — infeasible for $n = 10^5$. The embedding uses a [Nyström-style](https://en.wikipedia.org/wiki/Nystr%C3%B6m_method) heuristic with ridge regularisation (Williams & Seeger 2001), not a formal Mercer kernel approximation: pure OPH Jaccard is used throughout the anchor kernel (the earlier containment blend was removed for PSD correctness), but Laplacian normalisation and dual-sketch asymmetry (see below) still break strict Nyström coherence. The method approximates the $n \times n$ similarity matrix from a small anchor subset of size $p \ll n$. The dominant cost becomes $O(n \cdot p \cdot m)$ for the genome-to-anchor similarities.

### Anchor Sampling

The anchor count is $p = \min(n,\ \max(200,\ 2 \cdot d_\text{cfg}))$, where $d_\text{cfg}$ is the configured maximum embedding dimension (default 256, see `--geodesic-dim`). The actual embedding dimension $d$ is auto-selected from the anchor eigenspectrum (see Eigendecomposition below) and may be less than $d_\text{cfg}$.

Genomes are stratified by fill fraction $f_i$ into $Q = 5$ quantile strata, and an equal number of anchors is drawn from each stratum by Fisher-Yates shuffle. Stratification ensures the anchor Gram matrix covers the full range of genome completeness — without it, uniform sampling over-represents the abundant complete genomes and the eigenvectors become insensitive to MAG geometry.

### Anchor Gram Matrix

The $p \times p$ anchor Gram matrix $K_\text{raw}$ is computed using **dual-sketch averaged Jaccard** (both sig1 and sig2 are materialised for all $p$ anchors):

$$K_\text{raw}[i,j] = \frac{J_1(\text{anchor}_i, \text{anchor}_j) + J_2(\text{anchor}_i, \text{anchor}_j)}{2}$$

**Bin co-occupancy blend for sparse anchors.** When either anchor in a pair has $f_i < 0.2$ (fewer than 2,000 of 10,000 bins occupied, a heuristic threshold), raw Jaccard underestimates similarity because most of both genomes' k-mers land in the same few bins. A bin co-occupancy statistic is blended in:

$$C_\text{occ}(A \to B) = \frac{\text{popcount}(\text{mask}_A \mathbin{\&} \text{mask}_B)}{n_{\text{real},A}}$$

This approximates the conditional probability that a bin occupied in $A$ is also occupied in $B$. It is not a rigorous containment estimator because it conflates true k-mer overlap with $B$'s overall bin occupancy rate. In the fully saturated regime ($f \approx 1$), $C_\text{occ} \approx 1$ regardless of similarity and carries no signal. The correction is only applied for $f_i < 0.2$, where sparse bins correlate with true k-mer overlap.

The blended kernel:

$$K_\text{blend}[i,j] = (1-\alpha)\, K_\text{raw}[i,j] + \alpha \cdot \max\!\left(C_\text{occ}(i \to j),\ C_\text{occ}(j \to i)\right)$$

with $\alpha = \max(\alpha_i, \alpha_j)$ and $\alpha_i = \max(0,\ 1 - f_i/0.2)$ (linear ramp: $\alpha_i = 1$ at $f_i = 0$, $\alpha_i = 0$ at $f_i = 0.2$). The threshold 0.2 and the linear schedule are engineering heuristics, not derived from theory. After blending, $K_\text{blend}$ is no longer guaranteed positive semi-definite; near-zero eigenvalues are floored at $10^{-10}$ before inversion.

### Gram Matrix Regularisation

Two steps are applied to $K_\text{blend}$ before eigendecomposition.

**Symmetric Laplacian normalisation.** Hub anchors with high similarity to many others inflate their row sums and distort the eigenspectrum. Degree-normalising removes this bias:

$$K_\text{norm}[i,j] = \frac{K_\text{blend}[i,j]}{\sqrt{d_i \cdot d_j}}, \qquad d_i = \sum_j K_\text{blend}[i,j]$$

equivalently $K_\text{norm} = D^{-1/2} K_\text{blend} D^{-1/2}$. After this step, dot products in embedding space approximate a normalised-graph similarity, not raw k-mer Jaccard. The exact-sketch verification step (Phase 7) corrects borderline threshold decisions back to raw Jaccard space.

**Tikhonov diagonal loading.** Near-zero eigenvalues blow up the Nyström projection $W = U \Lambda^{-1/2}$. A ridge is added:

$$K_\text{reg} = K_\text{norm} + \lambda I, \qquad \lambda = 0.01 \cdot \max\!\left(\overline{K}_\text{diag},\ 10^{-4}\right)$$

where $\overline{K}_\text{diag}$ is the mean diagonal of $K_\text{norm}$. The floor on $\lambda$ ensures meaningful regularisation even when degree normalisation compresses the diagonal toward zero.

### Nyström Extension

The anchor Gram matrix $K_\text{reg}$ ($p \times p$, symmetric) is eigendecomposed:

$$K_\text{reg} = U \Lambda U^\top \qquad \text{[SelfAdjointEigenSolver]}$$

The embedding dimension $d$ is auto-selected as the smallest $d'$ such that the top $d'$ eigenvalues explain $\geq 95\%$ of total non-negative variance:

$$d = \min\!\left\{ d' :\ \frac{\displaystyle\sum_{i=p-d'+1}^{p} \lambda_i}{\displaystyle\sum_i \max(\lambda_i, 0)} \geq 0.95 \right\}$$

The projection matrix is:

$$W = U_{[:,\,\text{top-}d]} \cdot \operatorname{diag}\!\left(\lambda_{\text{top-}d}^{-1/2}\right) \qquad [p \times d]$$

Each genome $G_i$ is mapped to a $d$-dimensional unit vector. For anchors, the embedding vector is read directly from $K_\text{reg}$ (row $i$, already degree-normalised). For non-anchor genomes, sig1 is used to compute the genome-to-anchor similarity vector $\mathbf{k}_G$, which is then degree-normalised and projected:

$$k_G[a] = J_1(G_i, \text{anchor}_a)$$

$$\mathbf{k}_G \leftarrow \mathbf{k}_G \odot \mathbf{d}_\text{anchor}^{-1/2} \,/\, \sqrt{\textstyle\sum_a k_G[a]} \quad \text{[degree normalisation]}$$

$$\tilde{\mathbf{e}}(G_i) = W^\top \mathbf{k}_G, \qquad \mathbf{e}(G_i) = \tilde{\mathbf{e}}(G_i) \,/\, \|\tilde{\mathbf{e}}(G_i)\|_2$$

Non-anchors use sig1 only for this step — sig2 is not materialised for them.

> **Design note — dual-sketch asymmetry.** The anchor Gram matrix $K$ uses dual-sketch averaged Jaccard (lower variance), while the extension vector $\mathbf{k}_G$ uses sig1 only. This means the kernel function changes between the anchor block (training) and the cross block (test), which formally breaks standard Nyström coherence (which assumes one consistent kernel $K(x,y)$ throughout). The asymmetry is an engineering trade-off: materialising sig2 for all $n$ non-anchor genomes would double the sketching cost. In practice, the anchor block dominates the spectral approximation quality, and Phase 8 universal certification compensates for any miscoverage arising from the noisier extension. This embedding is therefore a **heuristic** rather than a formal Nyström approximation.

---

## Phase 3: HNSW Index

An [HNSW (Hierarchical Navigable Small World)](https://arxiv.org/abs/1603.09320) index (Malkov & Yashunin 2018) is built over all $n$ unit-sphere embeddings using inner product as the metric. Default parameters: $M = 48$, ef\_construction $= 400$, ef\_search $= 200$. HNSW provides approximate nearest-neighbour queries in $O(\log n)$ amortised time; recall is high but not 100%.

The index serves two purposes:
- Computing isolation scores (Phase 4)
- Finding too-close representative pairs for merging (Phase 6)

---

## Phase 4: Isolation Scores and Diversity Threshold

### Isolation Score

For each genome $G_i$, the isolation score is the mean angular distance to its $k = 10$ nearest neighbours (self excluded):

$$\text{isolation}(G_i) = \frac{1}{k} \sum_{j \in kNN(G_i)} \frac{\arccos(\mathbf{e}_i \cdot \mathbf{e}_j)}{\pi}$$

Higher isolation = more separated from neighbours = stronger candidate for a representative.

### Diversity Threshold

The diversity threshold $\theta$ controls when FPS stops. It is set to the minimum of:

1. The 95th percentile of all pairwise nearest-neighbour angular distances ($\text{NN}_{P95}$ — typical separation in the taxon)
2. The ANI threshold converted to angular distance via the Mash formula (see §ANI from Jaccard)

$$\theta = \min\!\left(\text{NN}_{P95},\ \frac{\arccos(J_\text{ANI})}{\pi}\right)$$

For clonal taxa with a tight distribution, $\text{NN}_{P95}$ is small and drives selection of fewer representatives. For diverse taxa, the user ANI threshold is the binding constraint.

---

## Phase 5: Farthest Point Sampling

[Farthest Point Sampling (FPS)](https://en.wikipedia.org/wiki/Farthest-first_traversal) selects representatives greedily: each step adds the genome farthest from all current representatives, forming a greedy $\theta$-cover. For unweighted FPS with fixed $k$ on a metric space, Gonzalez (1985) showed a 2-approximation to the k-center problem. That bound does not apply here: quality weighting, batch processing ($B=16$), and threshold-based stopping (variable $k$) violate the Gonzalez conditions. Coverage is evaluated empirically.

**Quality and size weighting.** Each genome is assigned a fitness score:

$$\text{fitness}(G_i) = \text{dist}(G_i) \cdot \frac{\text{quality}_i}{100} \cdot \sqrt{\frac{\text{size}_i}{\text{median\_size}}}$$

where $\text{quality} = \text{completeness} - 5 \times \text{contamination}$ (a heuristic combining CheckM2 completeness and contamination estimates, commonly used in genome quality filtering but not a CheckM2-defined metric). This weighting biases selection toward high-quality complete genomes.

**Algorithm:**
1. Start with the genome maximising $\text{isolation} \times \text{quality} \times \text{size}$ as the first representative
2. Track $\text{max\_sim}[j] = \max_\text{reps} \, \mathbf{e}_\text{rep} \cdot \mathbf{e}_j$ (max dot product to any current representative)
3. Repeat: compute fitness for all uncovered genomes, add the top-$B = 16$ candidates (partial sort) in a single parallel pass, then update $\text{max\_sim}$ for all $n$ genomes
4. Terminate when all non-excluded genomes satisfy $\text{dist\_to\_nearest\_rep} < \theta$

Batching $B = 16$ candidates per iteration fuses 16 distance updates into one parallel pass over $n$ genomes, reducing OpenMP synchronisation overhead. Covered genomes are removed from the active set after each batch, reducing per-iteration work from $O(n)$ to $O(n_\text{uncovered})$.

---

## Phase 6: Union-Find Merge

Quality and size weighting can push representatives toward dense high-quality regions, leaving some representative pairs closer than intended. Representatives with embedding distance below $d_\text{min}$ are merged via [Union-Find](https://en.wikipedia.org/wiki/Disjoint-set_data_structure): for each such pair $(\text{rep}_i, \text{rep}_j)$, they are merged into one cluster with the surviving representative chosen by $\text{quality} \times \text{size}$.

Merge candidates are found via HNSW search over the representative set, avoiding an $O(r^2)$ brute-force scan. Because HNSW is approximate, a small fraction of too-close pairs may be missed.

$d_\text{min}$ is set to $\min(\text{NN}_{P5},\ \theta / 2)$, keeping it below the diversity threshold.

---

## Phase 7: Borderline Verification

### Approximation Error

Nyström embedding introduces geometric error. The implementation uses $\varepsilon = \min(3/\sqrt{d},\ 0.3)$ as an empirical error tolerance (not a derived bound; for $d = 100$, $\varepsilon = 0.3$). A genome at embedding distance in

$$\left[\theta(1-\varepsilon),\ \theta\right)$$

is borderline covered. The multiplicative form is a practical heuristic: the true approximation error depends on the specific taxon geometry, not just $d$. Borderline cases are resolved by a direct sketch comparison.

### OPH Sketch Jaccard Check

For each borderline-covered genome $G_i$:

1. Find the top-3 closest representatives by embedding dot product (linear scan over all current representatives). Three is a heuristic: if the embedding ranking is wrong, the true covering representative may not be among the top 3.
2. For each candidate representative $R_k$, compute the dual-sketch averaged OPH Jaccard (sig2 is materialised on demand for both $G_i$ and all current representatives):
$$J_\text{dual} = \frac{J(\text{sig1}_i,\ \text{sig1}_{R_k}) + J(\text{sig2}_i,\ \text{sig2}_{R_k})}{2}$$
3. Convert: $d_\text{sketch} = \arccos\!\left(\operatorname{clamp}(J_\text{dual}, 0, 1)\right) / \pi$
4. Promote $G_i$ to representative only if all checked representatives satisfy $d_\text{sketch} \geq \theta$.
5. Batch-update $\text{max\_sim}$ for all promoted genomes.

This check uses OPH sketch Jaccard ($m = 10{,}000$ bin comparison), not exact ANI. It is more accurate than the embedding approximation but remains a sketch estimator with variance $J(1-J) / (2\, m_\text{real})$.

---

## ANI from Jaccard

[Average Nucleotide Identity (ANI)](https://en.wikipedia.org/wiki/Average_nucleotide_identity) is related to k-mer Jaccard via the [Mash formula](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-016-0997-x) (Ondov et al. 2016). Under a Poisson substitution model with per-base substitution rate $r = 1 - \text{ANI}/100$, the probability that a specific k-mer is shared between two equal-length genomes is approximately $(1-r)^k = p^k$ where $p = \text{ANI}/100$. For equal-size genomes, k-mer Jaccard relates to this probability as:

$$J \approx \frac{p^k}{2 - p^k}, \qquad p = \frac{\text{ANI}}{100}$$

Inverting:

$$\text{ANI} = \left(\frac{2J}{1 + J}\right)^{1/k} \times 100$$

This approximation assumes equal genome sizes, negligible indels, and independence of substitutions across sites. It degrades for highly divergent sequences or assemblies with large size differences.

The angular distance on the unit sphere is:

$$d_\text{angular}(A, B) = \frac{\arccos(\mathbf{e}_A \cdot \mathbf{e}_B)}{\pi} \in [0, 1]$$

After Nyström embedding **without** degree normalisation, $\mathbf{e}_A \cdot \mathbf{e}_B \approx J(A, B)$, so the angular threshold corresponding to an ANI cutoff is:

$$\theta_\text{ANI} = \frac{\arccos(J_\text{threshold})}{\pi}$$

After degree normalisation, dot products approximate a normalised-graph similarity, not raw Jaccard. FPS operates in this normalised space; Phase 7 corrects borderline threshold decisions back to sketch Jaccard space.

---

## Complexity

| Phase | Operation | Complexity |
|-------|-----------|------------|
| Sketch | OPH per genome (rolling hash) | $O(L)$ where $L$ = genome length |
| Embed anchors | Gram matrix $K$ | $O(p^2 m)$ |
| Embed all | Nyström extension (sig1 only) | $O(npm)$ |
| Eigendecomp | $K_\text{reg}$ ($p \times p$) | $O(p^3)$ |
| Index | HNSW build | $O(nM \log n)$ |
| Score | kNN isolation | $O(n \log n)$ |
| Select | FPS batched ($B=16$) | $O(nrd/B)$ where $r$ = n\_reps |
| Merge | HNSW search + Union-Find | $O(r \log r)$ |
| Verify | OPH sketch Jaccard | $O(n_\text{borderline} \cdot m)$ |

Typical values for a large taxon: $p \approx 512$, $m = 10{,}000$, $d \approx 64$–$256$. Embedding (Nyström extension) dominates for taxa with $n > 10{,}000$; FPS dominates for medium-size taxa.

---

## Implementation Notes

- **SIMD**: AVX2 is used in the OPH sketching inner loop (`scan_valid_run`, 32 bytes/cycle), in the anchor-slab Gram matrix loop (`_mm256_cmpeq_epi16`, 16 uint16 comparisons per instruction), and in FPS distance update loops.
- **OpenMP**: parallel OPH sketching, parallel Gram matrix rows, parallel FPS fitness/update loops, parallel HNSW isolation queries.
- **DuckDB**: all results persisted incrementally; interrupted runs resume by skipping completed taxa.
- **Anchor slab**: anchor signatures ($p \times m \times 2$ bytes $\approx 10\ \text{MB}$ for $p = 512$, $m = 10{,}000$) are packed into a contiguous aligned buffer for cache-friendly Gram matrix computation.
- **Producer-consumer I/O**: genome decompression is overlapped with k-mer computation via a bounded semaphore, capping simultaneous NFS file opens.
- **NFS robustness**: `embed_genome()` retries failed reads up to 3 times with 500 ms / 1 s backoff. Permanently failed genomes are recorded in the `jobs_failed` table.

---

## References

- Ondov et al. (2016) *Mash: fast genome and metagenome distance estimation using MinHash*. Genome Biology. [doi:10.1186/s13059-016-0997-x](https://doi.org/10.1186/s13059-016-0997-x)
- Li, P. & König, A.C. (2011) *b-Bit Minwise Hashing*. Proceedings of WWW 2011. [doi:10.1145/1989323.1989399](https://doi.org/10.1145/1989323.1989399)
- Li, P. & König, A.C. (2012) *One Permutation Hashing*. Advances in Neural Information Processing Systems (NIPS). [link](https://papers.nips.cc/paper/2012/hash/eaa32c96f620053cf442ad32258076b9-Abstract.html)
- Williams, C.K.I. & Seeger, M. (2001) *Using the Nyström Method to Speed Up Kernel Machines*. NIPS. [link](https://proceedings.neurips.cc/paper/2000/file/19de10adbaa1b2ee13f77f679fa1483a-Paper.pdf)
- Gonzalez, T.F. (1985) *Clustering to minimize the maximum intercluster distance*. Theoretical Computer Science 38:293–306.
- Malkov, Y.A. & Yashunin, D.A. (2018) *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs*. IEEE TPAMI 42(4):824–836. [doi:10.1109/TPAMI.2018.2889473](https://doi.org/10.1109/TPAMI.2018.2889473)
- Chowdhury et al. (2023) *CheckM2: a rapid, scalable and accurate tool for assessing microbial genome quality using machine learning*. Nature Methods. [doi:10.1038/s41592-023-01940-w](https://doi.org/10.1038/s41592-023-01940-w)
