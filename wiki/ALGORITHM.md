# geodesic — Algorithm Reference

**Spherical genome embeddings for diverse representative selection**

---

## Overview

`geodesic` selects a diverse set of representative genomes per taxon for reference-based short-read mapping. It uses OPH sketching, Nyström embedding, farthest-point sampling, Union-Find merge, and direct OPH verification to retain diversity while keeping every genome within an ANI threshold of some representative.

The pipeline has seven phases:

```
Sketch → Embed → Index → Score → Select → Merge → Verify
```

---

## Phase 1 — OPH Sketching

### One-Permutation Hashing

For each genome, geodesic computes a **One-Permutation Hash (OPH)** signature of m = 10,000 bins using k-mers of length k = 21.

**Canonical k-mer hashing.** For each position in the genome, both the forward 21-mer and its reverse complement are hashed. The canonical hash is:

```
h_canonical = min(hash(kmer_fwd), hash(kmer_rev))
```

using rolling Murmur-style hashing (WyMix) for O(1) update per base:

```
fwd ← ((fwd << 2) | base) & k_mask
rev ← (rev >> 2) | ((3 - base) << rev_shift)
canonical ← fwd XOR ((fwd XOR rev) & -(fwd > rev))
```

**Bin assignment.** Each canonical hash h is assigned to a bin t ∈ {0, …, m−1} via:

```
t = floor(h × m / 2^64)      [multiplication-based modulo, bias-free]
```

The per-bin value stored is the high-32 bits of the hashed canonical:

```
sig[t] ← min(sig[t], high32(WyMix(canonical, seed)))
```

**Densification.** Many bins may be unoccupied, especially for incomplete assemblies (MAGs). Empty bins are filled by deterministic propagation from neighbours:

1. Forward pass: if `sig[t]` is empty and `sig[t-1]` is real, set `sig[t] = mix(sig[t-1] XOR t)`
2. Backward pass: if `sig[t]` is empty and `sig[t+1]` is real, set `sig[t] = mix(sig[t+1] XOR t)`

where `mix` is a 64-bit finalizer (SplitMix64). This "nearest-neighbor" densification preserves the unbiasedness property conditionally on the densified OPH construction (Li & König 2011).

**Jaccard property.** After densification, for any bin t:

```
P[sig_A[t] = sig_B[t]] = J(A, B) = |A ∩ B| / |A ∪ B|
```

where A, B are the sets of distinct canonical k-mers. Over m bins, the empirical collision fraction is an unbiased estimator of J with variance:

```
Var(Ĵ) = J(1-J) / m_eff
```

where m_eff ≤ m is the number of *informative* (real) bins. For complete genomes with m_eff ≈ m = 10,000, standard error is < 0.01 at J = 0.3. **For MAGs with m_eff << m, variance is substantially higher** — the densified bins carry no independent signal about the true set intersection.

### OPH vs. Bottom-k MinHash

OPH is chosen over bottom-k MinHash for three reasons:

1. **Single pass**: all m bins are filled in one scan. Bottom-k requires maintaining a sorted heap.
2. **Fixed memory**: m bins regardless of genome size. Bottom-k sketch size adapts to k-mer diversity.
3. **Free byproduct**: the real-bin bitmask `mask_A` (bit t = 1 iff bin t has a real k-mer) is obtained at no extra cost and enables containment and fill-fraction estimation (see Phase 2).

The statistical weakness of OPH — heteroskedastic variance driven by m_eff — is partly corrected by the dual-sketch strategy below.

### Dual OPH Sketches

Two independent OPH signatures are computed per genome using different hash seeds (42 and 1337). The Gram matrix entry used for Nyström is:

```
K[i,j] = (Ĵ₁(i,j) + Ĵ₂(i,j)) / 2
```

**Variance reduction.** For two independent estimators of J, averaging reduces variance by 1/2:

```
Var((Ĵ₁ + Ĵ₂)/2) = J(1-J) / (2·m_eff)
```

The benefit is largest for MAGs where m_eff is small and each individual sketch is noisy. For complete genomes where m_eff ≈ 10,000, the improvement is modest (1/√2 in standard deviation) but adds robustness to hash collisions and outlier bins.

**Independence.** The two sketches use independent seeds for both the bin-assignment hash and the within-bin comparison hash (the seed enters through `WyMix(canonical, seed)`). Averaging two PSD collision matrices stays PSD.

**Storage.** Each OPH value is truncated to 16 bits (`uint16_t`) for storage, halving RAM versus `uint32_t`. A b = 16 bit bias correction is applied when comparing:

```
Ĵ_corrected = max(0, (Ĵ_raw - 2^{-b}) / (1 - 2^{-b}))     [b = 16]
```

### Fill Fraction and Completeness Proxy

The fill fraction f_i = n_real_bins_i / m (fraction of bins with at least one real k-mer before densification) serves as a proxy for genome completeness. Under a Poisson occupancy model:

```
E[f_i] ≈ 1 - exp(-|G_i| / m)
```

where |G_i| is the number of distinct canonical k-mers in genome G_i. For complete bacterial genomes (|G| ≈ 10^6 k-mers, m = 10,000), f ≈ 1 − e^{−100} ≈ 1. For MAGs with 10% completeness (|G| ≈ 10^5 k-mers), f ≈ 1 − e^{−10} ≈ 1. Even at 1% completeness (|G| ≈ 10^4 k-mers), f ≈ 1 − e^{−1} ≈ 0.63.

This means **f_i is not a direct completeness estimator** for typical bacterial genomes (bins saturate quickly). Its value lies in identifying the extreme tail of highly incomplete, very short assemblies where OPH variance is highest.

---

## Phase 2 — Nyström Spectral Embedding

### Motivation

Exact pairwise Jaccard over n genomes requires O(n²m) computations — infeasible for n = 10^5. The Nyström method approximates the n×n Gram matrix K from a small anchor subset, enabling O(nm + p²m) work where p << n is the number of anchors.

### Step 2a — Stratified Anchor Sampling

A subset of p anchors is selected from the n genomes. The auto-selected count is:

```
p = min(n, max(200, 2·d))
```

where d is the target embedding dimension.

**Stratified sampling.** Rather than uniform random sampling (which under-represents MAGs in taxa mixing complete and partial assemblies), genomes are stratified by fill fraction f_i into Q = 5 quantile strata. An equal number of anchors is drawn from each stratum by Fisher-Yates shuffle within the stratum.

This ensures the anchor Gram matrix covers the full spectrum of genome completeness, preventing the eigenvectors from being dominated by the geometry of complete genomes alone.

### Step 2b — Anchor Gram Matrix Construction

The p×p anchor Gram matrix K_raw is computed as:

```
K_raw[i,j] = (Ĵ₁(anchor_i, anchor_j) + Ĵ₂(anchor_i, anchor_j)) / 2
```

**Containment blend for sparse genome pairs.** When either genome in a pair has fill fraction f_i < 0.2 (i.e., fewer than 2,000 of 10,000 bins are real), Jaccard is blended with a bin co-occupancy statistic. This blend is applied consistently in both the anchor Gram matrix and the genome-to-anchor Nyström extension:

```
Ĉ_occ(A in B) = popcount(mask_A AND mask_B) / n_real_bins_A
```

This statistic measures "what fraction of A's occupied bins are also occupied by B" — a proxy for directional similarity in bin-space. It is **not** a rigorous k-mer containment estimator (C(A⊆B) = |A∩B|/|A|), because under the Poisson model:

```
E[Ĉ_occ(A in B)] = P(bin_t real in B | bin_t real in A)
```

which confounds true overlap with B's total bin occupancy rate (larger B inflates this even when |A∩B| = 0). **In the saturated-bin regime** (complete bacterial genomes), all bins are occupied and Ĉ_occ ≈ 1 regardless of true similarity — it carries no signal. The correction is therefore only applied when f_i < 0.2, where bins are sparse enough that co-occupancy correlates with true overlap.

The blended kernel value is:

```
K_blend[i,j] = (1 - α)·K_raw[i,j] + α·max(Ĉ_occ(i in j), Ĉ_occ(j in i))
```

with alpha = max(α_i, α_j) where α_i = max(0, 1 − f_i / 0.2). For f_i ≥ 0.2, α_i = 0 and raw Jaccard is used unmodified.

> **Caveat.** After containment blending, K_blend is no longer guaranteed to be positive semi-definite (PSD). The implementation floors selected eigenvalues at 1e-10 before inversion. The embedding is still geometrically valid but may sacrifice a small fraction of explained variance.

### Step 2c — Gram Matrix Regularisation

Two regularisation steps are applied to K_blend before eigendecomposition.

**Symmetric Laplacian normalisation (degree normalisation).** Hub anchors — those similar to many other anchors — have inflated row sums and bias the eigenspectrum. Normalising by anchor degree d_i = Σ_j K[i,j] corrects this:

```
K_norm[i,j] = K_blend[i,j] / sqrt(d_i · d_j)
```

equivalently K_norm = D^{−1/2} K_blend D^{−1/2} where D = diag(d_1, …, d_p).

After this transformation, the embedding approximates **community structure** in the similarity graph, not raw pairwise Jaccard. Downstream dot products dot(e_A, e_B) approximate a normalised-graph similarity, not J(A,B) directly. This is why an exact-Jaccard borderline verification step (Phase 7) is required for threshold decisions.

**Tikhonov diagonal loading.** Small or near-zero eigenvalues blow up the inverse-square-root W = U·diag(λ^{−1/2}) during the Nyström projection. A ridge is added:

```
K_reg = K_norm + λ·I       where  λ = 0.01 · max(mean_diagonal(K_norm), 1e-4)
```

The floor ensures meaningful regularisation even when degree normalisation compresses the diagonal toward zero in high-connectivity anchor sets. It prevents numerical instabilities in W = U·diag(λ^{−1/2}).

**Non-anchor Nyström extension.** For non-anchor genome G_i, the raw similarity vector k_raw[a] uses the dual-sketch averaged Jaccard to anchor a, with the same containment blend applied when f_i < 0.2 or f_anchor < 0.2. If degree normalisation is enabled, k[a] = k_raw[a] / sqrt(d_i · d_anchor_a) where d_i = Σ_a k_raw[a]. For anchor genomes, k(anchor_i) is the corresponding row of K_reg (already normalised).

### Step 2d — Eigendecomposition and Dimension Selection

The regularised matrix K_reg (p×p, symmetric) is eigendecomposed:

```
K_reg = U Λ U^T     [SelfAdjointEigenSolver, exact for symmetric matrices]
```

The embedding dimension d is auto-selected as the minimum d such that the top-d eigenvalues capture ≥ 95% of the total (non-negative) eigenvalue sum:

```
d = min{ d' : (Σ_{i=n-d'+1}^{n} λ_i) / (Σ_i max(λ_i, 0)) ≥ 0.95 }
```

The projection matrix is:

```
W = U[:, top-d] · diag(λ_{top-d}^{-1/2})     [p × d]
```

### Step 2e — Nyström Extension

Each genome G_i is mapped to a d-dimensional unit vector via:

```
k(G_i) = [similarity(G_i, anchor_1), …, similarity(G_i, anchor_p)]   [p-vector, degree-normalised]
ẽ(G_i) = W^T · k(G_i)                                                  [raw d-vector]
e(G_i) = ẽ(G_i) / ‖ẽ(G_i)‖₂                                          [unit sphere projection]
```

For anchor genomes k(anchor_i) = K_reg[row i] (exact; already normalised). For non-anchors, each entry is the dual-sketch Jaccard (with containment blend if sparse), then degree-normalised. See Step 2c for the k_G normalisation formula.

**Theoretical guarantee.** The Nyström approximation gives:

```
K ≈ K_{n,p} · K_{p,p}^{-1} · K_{p,n}
```

where K_{n,p}[i,a] = K(G_i, anchor_a). On the unit sphere, dot(e_A, e_B) ≈ normalised Gram matrix entry — an approximation to the normalised Jaccard-based similarity, not raw Jaccard.

---

## Phase 3 — HNSW Index

An HNSW (Hierarchical Navigable Small World) index is built over the n unit-sphere embeddings using the inner product metric. Default parameters: M = 48, ef_construction = 400, ef_search = 200.

The index is used for:
- Computing isolation scores (Phase 4)
- Electrostatic merge during representative selection (Phase 6)

---

## Phase 4 — Isolation Scores and Diversity Threshold

### Isolation Score

For each genome G_i, the isolation score is the mean angular distance to its k = 10 nearest neighbours:

```
isolation(G_i) = (1/k) Σ_{j ∈ kNN(G_i)} arccos(dot(e_i, e_j)) / π
```

Higher isolation score = more isolated = better candidate for a representative (the genome is in a sparse region of the sphere).

### Diversity Threshold (Data-Driven)

The diversity threshold θ controls when FPS stops. It is set to the minimum of:

1. **Empirical P95 of NN distances**: the 95th percentile of all nearest-neighbour angular distances, capturing the typical separation in the taxon.
2. **User ANI threshold** converted to angular distance via the Mash formula (see ANI section below).

```
θ = min(NN_P95, arccos(J_ANI_threshold) / π)
```

This ensures θ is grounded in the observed data geometry, not just a fixed user parameter. For clonal taxa (tight cluster), NN_P95 is small → few representatives. For diverse taxa, NN_P95 ≥ ANI angular threshold → ANI threshold is the binding constraint.

---

## Phase 5 — Farthest Point Sampling (FPS)

### Quality-Weighted FPS

FPS selects representatives greedily: at each step, the unrepresented genome with maximum distance to any current representative is added.

The **fitness score** combines coverage distance with genome quality and size:

```
fitness(G_i) = dist_to_nearest_rep(G_i) · (quality_i / 100) · sqrt(size_i / median_size)
```

where quality = completeness − 5 × contamination (CheckM2 scale, 0–100).

**Algorithm:**
1. Start with the genome maximising isolation × quality × size as the first representative
2. Update max_sim_to_rep[j] = max(max_sim_to_rep[j], dot(e_rep, e_j)) for all j
3. Repeat: select top-B = 16 candidates (partial sort for efficiency), add those above threshold θ
4. Terminate when all non-excluded genomes have dist_to_nearest_rep < θ

**k-center interpretation.** FPS is a greedy 2-approximation to the k-center problem: minimise the maximum distance from any genome to its nearest representative. Formally, if OPT_k is the optimal k-center radius, the FPS solution has radius ≤ 2·OPT_k (Gonzalez 1985).

**Batched FPS.** In each iteration, the top-B = 16 candidates are all added as representatives simultaneously. Their distance updates are fused into a single parallel pass over all n genomes, reducing OpenMP launch overhead by 16× and improving cache reuse.

**Active set compression.** Covered genomes (dist < θ) are removed from the active set after each iteration. For clonal taxa where most genomes are quickly covered, this reduces per-iteration work from O(n) to O(n_uncovered) — a ~10× speedup in practice.

---

## Phase 6 — Electrostatic Merge (Union-Find)

Representatives that landed too close together (due to quality × size weighting pushing them toward high-quality dense regions) are merged via Union-Find:

For each representative pair with dist(rep_i, rep_j) < min_rep_distance, they are merged into one. The surviving representative is chosen by quality × size.

In Nyström mode: merging candidates are found via HNSW search (parallel over representatives, serial Union-Find), avoiding the O(r²) brute-force needed for r representatives.

`min_rep_distance` is set to `min(NN_P5, θ / 2)`, ensuring it stays below the diversity threshold and only genuinely too-close reps are merged.

---

## Phase 7 — Borderline Verification

### Embedding Approximation Error

The Nyström approximation can introduce angular error on the order of 3/√d. The implementation uses ε = min(3/√d, 0.3) as a cap (at d = 64, 3/√64 = 0.375 → capped at 0.3).

A genome with embedding distance in:

```
[θ·(1 − ε),  θ)
```

is "borderline covered": the embedding says it is within the threshold of some representative, but the approximation error may be large enough that the true similarity is below the threshold.

### OPH Exact Jaccard Check

For each borderline-covered genome G_i:

1. Find the top-K = 3 current representatives by embedding dot product (linear scan over all reps).
2. For each candidate representative R_k, compute dual-sketch averaged OPH Jaccard:
   `Ĵ_dual = (refine_jaccard(sig1_i, sig1_Rk) + refine_jaccard(sig2_i, sig2_Rk)) / 2`
3. Convert to angular distance: `d_exact = arccos(clamp(Ĵ_dual, 0, 1)) / π`
4. Promote G_i only if **all** checked representatives satisfy `d_exact ≥ θ` (none truly covers it).
5. After all promotions, update max-similarity bookkeeping in one batched parallel pass.

> **Why raw OPH Jaccard, not normalised-kernel dot product?** The threshold θ is defined via the Mash formula from the user's ANI threshold, which is a Jaccard-based quantity. The borderline check must use the same metric as the threshold definition, not the normalised embedding space used by FPS.

---

## ANI Derivation from Jaccard (Mash Formula)

Average Nucleotide Identity (ANI) is related to Jaccard similarity of k-mer sets via the Mash formula (Ondov et al. 2016):

```
Mash distance D = −(1/k) ln(2J / (1 + J))
ANI = (2J / (1 + J))^{1/k} × 100
```

This is a model-based approximation assuming:
- Random i.i.d. substitutions (Jukes-Cantor model)
- Similar genome sizes (equal k-mer set sizes)
- Negligible insertion/deletion relative to substitutions
- No contamination

Under these assumptions, the probability that a k-mer is shared between two genomes at ANI p is:

```
P[kmer shared] = p^k = (ANI/100)^k
```

and Jaccard of k-mer sets equals this probability (assuming uniform coverage and no repeats):

```
J ≈ p^k / (2 - p^k)     [Mash formula, solved for J given ANI]
ANI ≈ (2J / (1+J))^{1/k} × 100     [inverted]
```

The angular distance on the unit sphere (after embedding) is:

```
d_angular(A, B) = arccos(dot(e_A, e_B)) / π  ∈ [0, 1]
```

Since dot(e_A, e_B) ≈ J(A, B) (after Nyström embedding, before degree normalisation), we have:

```
θ_ANI = arccos(J_threshold) / π
```

as the angular distance corresponding to the user's ANI threshold. After degree normalisation, this relationship is a surrogate: FPS operates on normalised-kernel space, and the exact OPH check (Phase 7) corrects borderline cases back to true Jaccard space.

---

## Summary of Robustness Improvements

| # | Improvement | Where | Problem solved |
|---|-------------|-------|----------------|
| 1 | Stratified anchor sampling (5 strata by f_i) | Phase 2a | MAGs under-represented in uniform sampling → biased eigenvectors |
| 2 | Symmetric Laplacian normalisation | Phase 2c | Hub anchors dominate spectrum → distorted embedding geometry |
| 3 | Tikhonov diagonal loading (λ = 0.01) | Phase 2c | Near-zero eigenvalues blow up W = U·Λ^{−1/2} → numerical instability |
| 4 | Bin co-occupancy blend for sparse anchors (f_i < 0.2) | Phase 2b | OPH Jaccard underestimates similarity for sparse MAGs → k-mer void bias |
| 5 | Dual OPH sketches (seeds 42 + 1337), averaged kernel | Phases 1, 2b | High-variance Jaccard estimates for low-m_eff genomes → noisy Gram matrix |
| 6 | FPS borderline refinement: top-K=3 reps, dual-sketch OPH, ε = min(3/√d, 0.3) | Phase 7 | Nyström approximation error misclassifies borderline genomes as covered; single-rep check risks wrong nearest rep |

---

## Complexity

| Phase | Operation | Complexity |
|-------|-----------|------------|
| Sketch | OPH per genome | O(L·k) where L = genome length |
| Embed anchors | Gram matrix K | O(p²·m) |
| Embed all | Nyström extension | O(n·p·m) |
| Eigendecomp | K_reg (p×p) | O(p³) |
| Index | HNSW build | O(n·log n) |
| Score | kNN isolation | O(n·log n) |
| Select | FPS (batched) | O(n·r·d) where r = n_reps |
| Merge | Union-Find | O(r·log n) |
| Verify | OPH exact Jaccard | O(n_borderline·m) |

For GTDB r220: n = 5.2M, p ≈ 512, m = 10,000, d ≈ 64–256, r ≈ 2,500 per large taxon. Embedding dominates for giant taxa; FPS dominates for medium taxa.

---

## Implementation Notes

- **SIMD**: AVX2 dot products (8 floats/cycle) accelerate isolation, FPS updates, and Gram matrix build
- **OpenMP**: Parallel OPH sketching, parallel Gram matrix rows, parallel FPS fitness/update loops, parallel HNSW isolation
- **DuckDB**: All results persisted incrementally; interrupted runs resume automatically
- **Producer-consumer I/O**: genome decompression overlapped with k-mer computation via bounded queue
- **16-bit OPH storage**: halves RAM vs uint32 with b-bit bias correction; 10,000 bins × 16 bits = 20 KB per genome

---

## References

- Ondov et al. (2016) *Mash: fast genome and metagenome distance estimation using MinHash*. Genome Biology.
- Li & König (2011) *b-Bit Minwise Hashing*. VLDB.
- Gonzalez (1985) *Clustering to minimize the maximum intercluster distance*. TCS.
- Zhang et al. (2021) *A study of HNSW for vector search*. arXiv.
- Malkov & Yashunin (2018) *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs*. TPAMI.
