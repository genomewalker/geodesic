# geodesic: Algorithm Reference

**Spherical genome embeddings for diverse representative selection**

---

## Overview

`geodesic` selects a diverse set of representative genomes per taxon for reference-based short-read mapping. It uses OPH sketching, Nyström embedding, farthest-point sampling, Union-Find merge, and direct OPH verification to retain diversity while keeping every genome within an ANI threshold of some representative.

The pipeline has seven phases:

```
Sketch → Embed → Index → Score → Select → Merge → Verify
```

---

## Phase 1: OPH Sketching

### One-Permutation Hashing

For each genome, geodesic computes a One-Permutation Hash (OPH) signature of m = 10,000 bins using k-mers of length k = 21.

**Canonical k-mer hashing.** For each position in the genome, both the forward 21-mer and its reverse complement are hashed. The canonical hash is:

```
h_canonical = min(hash(kmer_fwd), hash(kmer_rev))
```

using rolling WyMix hashing for O(1) update per base:

```
fwd = ((fwd << 2) | base) & k_mask
rev = (rev >> 2) | ((3 - base) << rev_shift)
canonical = fwd XOR ((fwd XOR rev) & -(fwd > rev))
```

**Bin assignment.** Each canonical hash h is assigned to a bin t in {0, ..., m-1} via:

```
t = floor(h * m / 2^64)      [multiplication-based modulo, bias-free]
```

The per-bin value stored is the high-32 bits of the hashed canonical:

```
sig[t] = min(sig[t], high32(WyMix(canonical, seed)))
```

**Densification.** Empty bins are filled by deterministic propagation from neighbours:

1. Forward pass: if `sig[t]` is empty and `sig[t-1]` is real, set `sig[t] = mix(sig[t-1] XOR t)`
2. Backward pass: if `sig[t]` is empty and `sig[t+1]` is real, set `sig[t] = mix(sig[t+1] XOR t)`

where `mix` is a 64-bit finalizer (SplitMix64). Nearest-neighbour densification preserves the unbiasedness property conditionally on the densified OPH construction (Li & König 2011).

**Jaccard property.** After densification, for any bin t:

```
P[sig_A[t] = sig_B[t]] = J(A, B) = |A ∩ B| / |A ∪ B|
```

where A, B are the sets of distinct canonical k-mers. Over m bins, the empirical collision fraction is an unbiased estimator of J with variance:

```
Var(J_hat) = J(1-J) / m_eff
```

where m_eff <= m is the number of real (pre-densification) bins. For complete genomes with m_eff ~= m = 10,000, standard error is < 0.01 at J = 0.3. For MAGs with m_eff << m, variance is higher because densified bins carry no independent signal about the true set intersection.

### OPH vs Bottom-k MinHash

OPH is chosen over bottom-k MinHash for three reasons:

1. **Single pass**: all m bins are filled in one scan. Bottom-k requires maintaining a sorted heap.
2. **Fixed memory**: m bins regardless of genome size. Bottom-k sketch size adapts to k-mer diversity.
3. **Free byproduct**: the real-bin bitmask `mask_A` (bit t = 1 iff bin t has a real k-mer) is obtained at no extra cost and enables containment and fill-fraction estimation (see Phase 2).

The statistical weakness of OPH -- heteroskedastic variance driven by m_eff -- is partly corrected by the dual-sketch strategy below.

### Dual OPH Sketches

Two independent OPH signatures are computed per genome using different hash seeds (42 and 1337). The Gram matrix entry used for Nyström is:

```
K[i,j] = (J_1(i,j) + J_2(i,j)) / 2
```

For two independent estimators of J, averaging reduces variance by 1/2:

```
Var((J_1 + J_2)/2) = J(1-J) / (2 * m_eff)
```

The benefit is largest for MAGs where m_eff is small. For complete genomes where m_eff ~= 10,000, the improvement is modest (1/sqrt(2) in standard deviation) but adds robustness to hash collisions and outlier bins.

The two sketches use independent seeds for both the bin-assignment hash and the within-bin comparison hash. Averaging two PSD collision matrices stays PSD.

Each OPH value is truncated to 16 bits (`uint16_t`) for storage, halving RAM versus `uint32_t`. A b = 16 bit bias correction is applied when comparing:

```
J_corrected = max(0, (J_raw - 2^{-b}) / (1 - 2^{-b}))     [b = 16]
```

**Lazy sig2 materialisation.** The second OPH sketch (seed 1337) is not computed for all n genomes during the sketch phase. Only the ~512 anchor genomes used for Nyström need sig2 for the Gram matrix. Sig2 is materialised on demand for anchors and for borderline verification candidates (Phase 7), avoiding a full NFS re-read and sketch pass for n - 512 genomes.

**Buffer cache.** The decompressed FASTA buffer for each genome (up to a 2.5 GB aggregate budget) is retained in memory after the sig1 sketch pass. Anchor sig2 materialisation reads from this cache rather than the NFS mount, eliminating the re-read latency for anchor genomes entirely.

### Fill Fraction and Completeness Proxy

The fill fraction f_i = n_real_bins_i / m (fraction of bins with at least one real k-mer before densification) serves as a proxy for genome completeness. Under a Poisson occupancy model:

```
E[f_i] ~= 1 - exp(-|G_i| / m)
```

where |G_i| is the number of distinct canonical k-mers in genome G_i. For complete bacterial genomes (|G| ~= 10^6 k-mers, m = 10,000), f ~= 1. Even at 1% completeness (|G| ~= 10^4 k-mers), f ~= 0.63.

f_i is not a direct completeness estimator for typical bacterial genomes (bins saturate quickly). Its value lies in identifying the extreme tail of highly incomplete, short assemblies where OPH variance is highest.

---

## Phase 2: Nyström Spectral Embedding

### Motivation

Exact pairwise Jaccard over n genomes requires O(n^2 * m) computations -- infeasible for n = 10^5. The Nyström method approximates the n*n Gram matrix K from a small anchor subset, enabling O(n*m + p^2*m) work where p << n is the number of anchors.

### Anchor Sampling

A subset of p anchors is selected from the n genomes. The auto-selected count is:

```
p = min(n, max(200, 2 * d))
```

where d is the target embedding dimension.

Genomes are stratified by fill fraction f_i into Q = 5 quantile strata. An equal number of anchors is drawn from each stratum by Fisher-Yates shuffle. This ensures the anchor Gram matrix covers the full spectrum of genome completeness, preventing the eigenvectors from being dominated by the geometry of complete genomes alone.

### Anchor Gram Matrix

The p*p anchor Gram matrix K_raw is computed as:

```
K_raw[i,j] = (J_1(anchor_i, anchor_j) + J_2(anchor_i, anchor_j)) / 2
```

When either genome in a pair has fill fraction f_i < 0.2 (fewer than 2,000 of 10,000 bins are real), Jaccard is blended with a bin co-occupancy statistic:

```
C_occ(A in B) = popcount(mask_A AND mask_B) / n_real_bins_A
```

This statistic measures what fraction of A's occupied bins are also occupied by B. It is not a rigorous k-mer containment estimator (C(A in B) = |A ∩ B|/|A|), because under the Poisson model:

```
E[C_occ(A in B)] = P(bin_t real in B | bin_t real in A)
```

which confounds true overlap with B's total bin occupancy rate. In the saturated-bin regime (complete bacterial genomes), all bins are occupied and C_occ ~= 1 regardless of true similarity -- it carries no signal. The correction is only applied when f_i < 0.2, where bins are sparse enough that co-occupancy correlates with true overlap.

The blended kernel value is:

```
K_blend[i,j] = (1 - alpha) * K_raw[i,j] + alpha * max(C_occ(i in j), C_occ(j in i))
```

with alpha = max(alpha_i, alpha_j) where alpha_i = max(0, 1 - f_i / 0.2). For f_i >= 0.2, alpha_i = 0 and raw Jaccard is used unmodified.

After containment blending, K_blend is no longer guaranteed to be positive semi-definite. The implementation floors selected eigenvalues at 1e-10 before inversion.

### Gram Matrix Regularisation

Two regularisation steps are applied to K_blend before eigendecomposition.

**Symmetric Laplacian normalisation.** Hub anchors -- those similar to many other anchors -- have inflated row sums and bias the eigenspectrum. Normalising by anchor degree d_i = sum_j K[i,j] corrects this:

```
K_norm[i,j] = K_blend[i,j] / sqrt(d_i * d_j)
```

equivalently K_norm = D^{-1/2} K_blend D^{-1/2} where D = diag(d_1, ..., d_p).

After this transformation, the embedding approximates community structure in the similarity graph, not raw pairwise Jaccard. Downstream dot products dot(e_A, e_B) approximate a normalised-graph similarity, not J(A,B) directly. The exact-Jaccard borderline verification step (Phase 7) corrects threshold decisions back to raw Jaccard space.

**Tikhonov diagonal loading.** Small or near-zero eigenvalues blow up the inverse-square-root W = U * diag(lambda^{-1/2}) during the Nyström projection. A ridge is added:

```
K_reg = K_norm + lambda * I       where  lambda = 0.01 * max(mean_diagonal(K_norm), 1e-4)
```

The floor on lambda ensures meaningful regularisation even when degree normalisation compresses the diagonal toward zero in high-connectivity anchor sets.

### Nyström Extension

Each genome G_i is mapped to a d-dimensional unit vector via:

```
k(G_i) = [similarity(G_i, anchor_1), ..., similarity(G_i, anchor_p)]
e_tilde(G_i) = W^T * k(G_i)
e(G_i) = e_tilde(G_i) / ||e_tilde(G_i)||_2
```

For anchor genomes k(anchor_i) = K_reg[row i] (exact; already normalised). For non-anchors, each entry is the dual-sketch Jaccard (with containment blend if sparse), then degree-normalised.

For non-anchor genome G_i, the raw similarity vector k_raw[a] uses the dual-sketch averaged Jaccard to anchor a, with the same containment blend applied when f_i < 0.2 or f_anchor < 0.2. If degree normalisation is enabled, k[a] = k_raw[a] / sqrt(d_i * d_anchor_a) where d_i = sum_a k_raw[a].

The Nyström approximation gives:

```
K ~= K_{n,p} * K_{p,p}^{-1} * K_{p,n}
```

where K_{n,p}[i,a] = K(G_i, anchor_a). On the unit sphere, dot(e_A, e_B) approximates the normalised Gram matrix entry.

### Eigendecomposition and Dimension Selection

The regularised matrix K_reg (p*p, symmetric) is eigendecomposed:

```
K_reg = U Lambda U^T     [SelfAdjointEigenSolver]
```

The embedding dimension d is auto-selected as the minimum d such that the top-d eigenvalues capture >= 95% of the total non-negative eigenvalue sum:

```
d = min{ d' : sum_{i=n-d'+1}^{n} lambda_i / sum_i max(lambda_i, 0) >= 0.95 }
```

The projection matrix is:

```
W = U[:, top-d] * diag(lambda_{top-d}^{-1/2})     [p * d]
```

---

## Phase 3: HNSW Index

An HNSW (Hierarchical Navigable Small World) index is built over the n unit-sphere embeddings using the inner product metric. Default parameters: M = 48, ef_construction = 400, ef_search = 200.

The index is used for:
- Computing isolation scores (Phase 4)
- Electrostatic merge during representative selection (Phase 6)

---

## Phase 4: Isolation Scores and Diversity Threshold

### Isolation Score

For each genome G_i, the isolation score is the mean angular distance to its k = 10 nearest neighbours:

```
isolation(G_i) = (1/k) * sum_{j in kNN(G_i)} arccos(dot(e_i, e_j)) / pi
```

Higher isolation score = more isolated = better candidate for a representative.

### Diversity Threshold

The diversity threshold theta controls when FPS stops. It is set to the minimum of:

1. The 95th percentile of all nearest-neighbour angular distances (typical separation in the taxon)
2. The user ANI threshold converted to angular distance via the Mash formula

```
theta = min(NN_P95, arccos(J_ANI_threshold) / pi)
```

For clonal taxa (tight cluster), NN_P95 is small -- few representatives selected. For diverse taxa, NN_P95 >= ANI angular threshold -- the ANI threshold is the binding constraint.

---

## Phase 5: Farthest Point Sampling

### Quality-Weighted FPS

FPS selects representatives greedily: at each step, the unrepresented genome with maximum distance to any current representative is added.

The fitness score combines coverage distance with genome quality and size:

```
fitness(G_i) = dist_to_nearest_rep(G_i) * (quality_i / 100) * sqrt(size_i / median_size)
```

where quality = completeness - 5 * contamination (CheckM2 scale, 0-100).

Algorithm:
1. Start with the genome maximising isolation * quality * size as the first representative
2. Update max_sim_to_rep[j] = max(max_sim_to_rep[j], dot(e_rep, e_j)) for all j
3. Repeat: select top-B = 16 candidates (partial sort for efficiency), add those above threshold theta
4. Terminate when all non-excluded genomes have dist_to_nearest_rep < theta

FPS is a greedy 2-approximation to the k-center problem: minimise the maximum distance from any genome to its nearest representative. Formally, if OPT_k is the optimal k-center radius, the FPS solution has radius <= 2 * OPT_k (Gonzalez 1985).

In each iteration, the top-B = 16 candidates are added as representatives simultaneously. Their distance updates are fused into a single parallel pass over all n genomes, reducing OpenMP launch overhead by 16x and improving cache reuse.

Covered genomes (dist < theta) are removed from the active set after each iteration. For clonal taxa where most genomes are quickly covered, this reduces per-iteration work from O(n) to O(n_uncovered) -- roughly 10x speedup in practice.

---

## Phase 6: Electrostatic Merge

Representatives that landed too close together (due to quality * size weighting pushing them toward high-quality dense regions) are merged via Union-Find:

For each representative pair with dist(rep_i, rep_j) < min_rep_distance, they are merged into one. The surviving representative is chosen by quality * size.

In Nyström mode: merging candidates are found via HNSW search (parallel over representatives, serial Union-Find), avoiding the O(r^2) brute-force needed for r representatives.

`min_rep_distance` is set to `min(NN_P5, theta / 2)`, ensuring it stays below the diversity threshold and only genuinely over-proximate reps are merged.

---

## Phase 7: Borderline Verification

### Approximation Error

The Nyström approximation can introduce angular error on the order of 3/sqrt(d). The implementation uses epsilon = min(3/sqrt(d), 0.3) as a cap (at d = 64, 3/sqrt(64) = 0.375, capped at 0.3).

A genome with embedding distance in:

```
[theta * (1 - epsilon),  theta)
```

is borderline covered: the embedding says it is within the threshold of some representative, but the approximation error may push the true similarity below the threshold.

### OPH Exact Jaccard Check

For each borderline-covered genome G_i:

1. Find the top-K = 3 current representatives by embedding dot product (linear scan over all reps).
2. For each candidate representative R_k, compute dual-sketch averaged OPH Jaccard:
   `J_dual = (refine_jaccard(sig1_i, sig1_Rk) + refine_jaccard(sig2_i, sig2_Rk)) / 2`
3. Convert to angular distance: `d_exact = arccos(clamp(J_dual, 0, 1)) / pi`
4. Promote G_i only if all checked representatives satisfy `d_exact >= theta` (none truly covers it).
5. After all promotions, update max-similarity bookkeeping in one batched parallel pass.

The threshold theta is defined via the Mash formula from the user's ANI threshold, which is a Jaccard-based quantity. The borderline check uses the same metric as the threshold definition, not the normalised embedding space used by FPS.

---

## ANI from Jaccard

Average Nucleotide Identity (ANI) is related to Jaccard similarity of k-mer sets via the Mash formula (Ondov et al. 2016):

```
ANI = (2J / (1 + J))^{1/k} * 100
```

This assumes random i.i.d. substitutions (Jukes-Cantor), similar genome sizes, and negligible indels relative to substitutions.

Under these assumptions, the probability that a k-mer is shared between two genomes at ANI p is p^k, and Jaccard of k-mer sets equals this probability:

```
J ~= p^k / (2 - p^k)     [Mash formula, solved for J given ANI]
ANI ~= (2J / (1+J))^{1/k} * 100     [inverted]
```

The angular distance on the unit sphere (after embedding) is:

```
d_angular(A, B) = arccos(dot(e_A, e_B)) / pi  in [0, 1]
```

Since dot(e_A, e_B) ~= J(A, B) (after Nyström embedding, before degree normalisation):

```
theta_ANI = arccos(J_threshold) / pi
```

After degree normalisation, this relationship is a surrogate: FPS operates on normalised-kernel space, and the exact OPH check (Phase 7) corrects borderline cases back to true Jaccard space.

---

## Robustness Improvements

| # | Improvement | Where | Problem solved |
|---|-------------|-------|----------------|
| 1 | Stratified anchor sampling (5 strata by f_i) | Phase 2 | MAGs under-represented in uniform sampling -- biased eigenvectors |
| 2 | Symmetric Laplacian normalisation | Phase 2 | Hub anchors dominate spectrum -- distorted embedding geometry |
| 3 | Tikhonov diagonal loading (lambda = 0.01) | Phase 2 | Near-zero eigenvalues blow up W = U * Lambda^{-1/2} -- numerical instability |
| 4 | Bin co-occupancy blend for sparse anchors (f_i < 0.2) | Phase 2 | OPH Jaccard underestimates similarity for sparse MAGs -- k-mer void bias |
| 5 | Dual OPH sketches (seeds 42 + 1337), averaged kernel | Phases 1, 2 | High-variance Jaccard estimates for low-m_eff genomes -- noisy Gram matrix |
| 6 | FPS borderline refinement: top-K=3 reps, dual-sketch OPH, epsilon = min(3/sqrt(d), 0.3) | Phase 7 | Nyström approximation error misclassifies borderline genomes as covered |

---

## Complexity

| Phase | Operation | Complexity |
|-------|-----------|------------|
| Sketch | OPH per genome | O(L*k) where L = genome length |
| Embed anchors | Gram matrix K | O(p^2 * m) |
| Embed all | Nyström extension | O(n*p*m) |
| Eigendecomp | K_reg (p*p) | O(p^3) |
| Index | HNSW build | O(n * log n) |
| Score | kNN isolation | O(n * log n) |
| Select | FPS (batched) | O(n*r*d) where r = n_reps |
| Merge | Union-Find | O(r * log n) |
| Verify | OPH exact Jaccard | O(n_borderline * m) |

For GTDB r220: n = 5.2M, p ~= 512, m = 10,000, d ~= 64-256, r ~= 2,500 per large taxon. Embedding dominates for giant taxa; FPS dominates for medium taxa.

---

## Implementation Notes

- **SIMD**: AVX2 dot products (8 floats/cycle) accelerate isolation, FPS updates, and Gram matrix build. AVX2 equality comparison (`_mm256_cmpeq_epi16`) is used in the anchor-slab Gram matrix loop, processing 16 uint16 bins per cycle.
- **OpenMP**: parallel OPH sketching, parallel Gram matrix rows, parallel FPS fitness/update loops, parallel HNSW isolation.
- **DuckDB**: all results persisted incrementally; interrupted runs resume automatically.
- **16-bit OPH storage**: halves RAM vs uint32 with b-bit bias correction; 10,000 bins * 16 bits = 20 KB per genome.
- **Anchor slab**: anchor signatures (p * m * 2 bytes = ~10 MB for p=512, m=10,000) are packed into a contiguous aligned buffer for cache-friendly Gram matrix computation.
- **Lazy sig2**: the second OPH sketch (seed 1337) is materialised only for anchors and borderline candidates on demand, not for all n genomes. Saves one full NFS read + sketch pass per non-anchor genome.
- **Buffer cache**: decompressed FASTA buffers (up to 2.5 GB aggregate) are held in memory after the sig1 sketch pass and consumed by anchor sig2 materialisation. After Nyström completes, the cache is freed.
- **NFS robustness**: `embed_genome()` retries failed reads up to 3 times with 500 ms/1 s backoff. Permanently failed genomes are recorded in the `jobs_failed` database table with their error reason.
- **Producer-consumer I/O**: genome decompression is overlapped with k-mer computation via a bounded semaphore, capping simultaneous NFS file opens to avoid metadata-op saturation.

---

## References

- Ondov et al. (2016) *Mash: fast genome and metagenome distance estimation using MinHash*. Genome Biology.
- Li & König (2011) *b-Bit Minwise Hashing*. VLDB.
- Gonzalez (1985) *Clustering to minimize the maximum intercluster distance*. TCS.
- Malkov & Yashunin (2018) *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs*. TPAMI.
