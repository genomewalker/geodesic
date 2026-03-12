# geodesic: Algorithm Reference

**Spherical genome embeddings for diverse representative selection**

---

## Overview

`geodesic` selects a diverse set of representative genomes per taxon for reference-based short-read mapping. It uses OPH sketching, Nyström spectral embedding, farthest-point sampling, Union-Find merge, and OPH-sketch verification to retain diversity while keeping every genome within an ANI threshold of its nearest representative.

The pipeline has eight phases:

```
Sketch → Embed → Index → Score → Select → Merge → Verify → Certify
```

For taxa with exactly 2 genomes, the full pipeline is skipped; see [n=2 fast path](#n2-fast-path).

---

## Phase 1: OPH sketching

### One-permutation hashing

For each genome, geodesic computes a [One-Permutation Hash (OPH)](https://papers.nips.cc/paper/2012/hash/eaa32c96f620053cf442ad32258076b9-Abstract.html) signature of $m = 10{,}000$ bins using k-mers of length $k = 21$.

**Canonical k-mer selection.** For each position in the genome, both the forward k-mer and its reverse complement are encoded as a 64-bit integer (2 bits per base, A=0/C=1/G=2/T=3). The canonical k-mer is the lexicographic minimum of the two encodings, selected by a branchless comparison:

```c
fwd = ((fwd << 2) | base) & k_mask
rev = (rev >> 2) | ((3 ^ base) << rev_shift)
canonical = fwd ^ ((fwd ^ rev) & -(uint64_t)(fwd > rev))   // = min(fwd, rev)
```

Ambiguous bases (N and other non-ACGT characters) reset the rolling state, discarding the current k-mer window.

**OPH hash.** A single 64-bit hash is computed from the canonical encoding and the sketch seed using a [WyHash](https://github.com/wangyi-fudan/wyhash)-based mixing step:

```
h = wymix(canonical XOR (seed + P0),  canonical XOR P1)
    where P0 = 0xa0761d6478bd642f, P1 = 0xe7037ed1a0b428db  (WyHash v4 constants)
    wymix(a, b) = lo64(a*b) XOR hi64(a*b)
```

sig1 uses seed = 42, sig2 uses seed = 1337. Both the bin index and the per-bin value are derived from this one hash:

$$
t = \left\lfloor \frac{h \cdot m}{2^{64}} \right\rfloor, \qquad \text{sig}[t] = \min\!\left(\text{sig}[t],\ h \gg 32\right)
$$

The stored uint32 value is truncated to uint16 at storage time (retaining bits 32–47 of $h$). A single hash call per k-mer determines both bin index and comparison value.

**Densification.** After scanning all k-mers, empty bins are filled by nearest-neighbour propagation, following the OPH densification scheme of Li & König (2012):

```
Forward:  if sig[t] = EMPTY and sig[t-1] ≠ EMPTY: sig[t] = SplitMix64(sig[t-1] XOR t)
Backward: if sig[t] = EMPTY and sig[t+1] ≠ EMPTY: sig[t] = SplitMix64(sig[t+1] XOR t)
```

Densified bins carry no independent information about k-mer overlap; their values are deterministic functions of real neighbours. The collision probability below applies only to real (pre-densification) bins.

**Jaccard property.** For any bin $t$ that is real in at least one genome:

$$
\Pr\!\left[\text{sig}_A[t] = \text{sig}_B[t]\right] = J(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

where $A$ and $B$ are the sets of distinct canonical k-mers. Over $m$ bins, the fraction of collisions is an unbiased estimator of $J$ with approximate variance:

$$
\text{Var}(\hat{J}) \approx \frac{J(1-J)}{m_\text{real}}
$$

where $m_\text{real}$ is the number of bins that are real in at least one of $A$ or $B$.

### OPH vs bottom-k MinHash

[Bottom-k MinHash](https://en.wikipedia.org/wiki/MinHash) keeps the $k$ smallest hash values across the entire genome. OPH partitions the hash space into $m$ bins and keeps one value per bin. OPH is preferred because one pass fills all $m$ bins, and the per-bin occupancy bitmask (used for sparse-anchor correction in Phase 2) is obtained at no extra cost.

### Dual OPH sketches

Two independent OPH signatures (sig1, sig2) are computed per genome using seeds 42 and 1337. The anchor Gram matrix uses dual-sketch averaged Jaccard (see Phase 2):

$$
K[i,j] = \frac{J_1(\text{anchor}_i, \text{anchor}_j) + J_2(\text{anchor}_i, \text{anchor}_j)}{2}
$$

For two independent $J$ estimators, averaging halves the variance:

$$
\text{Var}\!\left(\frac{J_1 + J_2}{2}\right) = \frac{J(1-J)}{2\, m_\text{real}}
$$

**16-bit storage.** Each per-bin value is stored as a uint16, halving RAM to $20\ \text{KB}$ per sketch per genome. Truncation increases false-match probability; the [b-bit MinHash bias correction](https://dl.acm.org/doi/10.1145/1989323.1989399) (Li & König 2011) corrects this:

$$
\hat{J}_\text{corr} = \max\!\left(0,\ \frac{\hat{J}_\text{raw} - 2^{-16}}{1 - 2^{-16}}\right)
$$

**Lazy sig2 materialisation.** sig2 is materialised on demand only for anchor genomes and borderline verification candidates. Non-anchor Nyström extension uses sig1 only. Decompressed FASTA buffers are cached in memory so anchor sig2 materialisation avoids a second NFS read.

### Fill fraction

The fill fraction $f_i = n_{\text{real},i} / m$ is the fraction of bins with at least one real k-mer before densification:

$$
\mathbb{E}[f_i] \approx 1 - e^{-|G_i|/m}
$$

For complete bacterial genomes ($|G| \sim 10^6$ k-mers, $m = 10{,}000$), $f \approx 1$. Highly incomplete assemblies ($f \ll 0.2$) have elevated OPH variance and trigger containment-based corrections in Phase 2.

### K-mer size and OPH accuracy

The default is $k = 21$. The k-mer size determines how quickly Jaccard drops with ANI (Mash model: $q = \text{ANI}^k$, $J = q/(2-q)$):

| ANI  | k=16  | k=21  | k=31  |
|------|-------|-------|-------|
| 85%  | 0.039 | 0.017 | 0.003 |
| 90%  | 0.102 | 0.058 | 0.019 |
| 95%  | 0.282 | 0.205 | 0.114 |
| 99%  | 0.741 | 0.680 | 0.578 |

At 95–99% ANI (typical bacterial species), $k=21$ gives $J \in [0.205, 0.680]$, sufficient for reliable ranking. Use `--sketch-k 16` for highly diverse taxa (ANI < 95%) where $k=21$ gives $J < 0.02$ and loses signal; use `--sketch-k 31` for clonal taxa (ANI > 99%) to spread compressed $J$ values apart.

The OPH direct Jaccard estimator (used in Phase 7) has standard deviation $\sqrt{J(1-J)/m}$, giving sub-0.2% ANI error at $m=10{,}000$ across the 90–99% ANI range:

| ANI  | $J$ (k=21) | OPH $\sigma_J$ | ANI error (1σ) |
|------|-----------|----------------|----------------|
| 90%  | 0.058     | 0.00234        | 0.16%          |
| 95%  | 0.205     | 0.00404        | 0.074%         |
| 99%  | 0.680     | 0.00466        | 0.019%         |

---

## Phase 2: Nyström spectral embedding

### Motivation

Exact pairwise Jaccard over $n$ genomes requires $O(n^2 m)$ operations, infeasible for $n = 10^5$. The [Nyström method](https://en.wikipedia.org/wiki/Nystr%C3%B6m_method) (Williams & Seeger 2001) approximates the $n \times n$ kernel matrix from a small anchor subset of size $p \ll n$. The dominant cost becomes $O(n \cdot p \cdot m)$ for the genome-to-anchor similarities.

### Anchor sampling

The anchor count is $p = \min(n,\ \max(200,\ 2 \cdot d_\text{cfg}))$, where $d_\text{cfg}$ is the configured maximum embedding dimension (default 256, `--geodesic-dim`). The actual embedding dimension $d$ is auto-selected from the anchor eigenspectrum (see below) and may be less than $d_\text{cfg}$.

Genomes are stratified by fill fraction $f_i$ into $Q = 5$ quantile strata, and an equal number of anchors is drawn from each stratum by Fisher-Yates shuffle. Stratification ensures the anchor Gram matrix covers the full range of genome completeness.

### Anchor Gram matrix

The $p \times p$ anchor Gram matrix $K_\text{raw}$ is computed using dual-sketch averaged Jaccard:

$$
K_\text{raw}[i,j] = \frac{J_1(\text{anchor}_i, \text{anchor}_j) + J_2(\text{anchor}_i, \text{anchor}_j)}{2}
$$

**Bin co-occupancy blend for sparse anchors.** When either anchor has $f_i < 0.2$, a bin co-occupancy statistic is blended in to correct Jaccard underestimation:

$$
C_\text{occ}(A \to B) = \frac{|\text{mask}_A \cap \text{mask}_B|}{n_{\text{real},A}}
$$

where $|\text{mask}_A \cap \text{mask}_B|$ is the number of bins occupied in both $A$ and $B$. The blended kernel:

$$
K_\text{blend}[i,j] = (1-\alpha)\, K_\text{raw}[i,j] + \alpha \cdot \max\!\left(C_\text{occ}(i \to j),\ C_\text{occ}(j \to i)\right)
$$

with $\alpha_i = \max(0,\ 1 - f_i/0.2)$ (linear ramp from 1 at $f_i=0$ to 0 at $f_i=0.2$), $\alpha = \max(\alpha_i, \alpha_j)$.

### Gram matrix regularisation

**Symmetric Laplacian normalisation** removes hub-anchor bias:

$$
K_\text{norm}[i,j] = \frac{K_\text{blend}[i,j]}{\sqrt{d_i \cdot d_j}}, \qquad d_i = \sum_j K_\text{blend}[i,j]
$$

equivalently $K_\text{norm} = D^{-1/2} K_\text{blend} D^{-1/2}$. After this step, dot products approximate a normalised-graph similarity, not raw Jaccard. Phase 7 corrects borderline decisions back to raw sketch Jaccard space.

**Tikhonov ridge** prevents near-zero eigenvalues from blowing up the projection:

$$
K_\text{reg} = K_\text{norm} + \lambda I, \qquad \lambda = 0.01 \cdot \max\!\left(\overline{K}_\text{diag},\ 10^{-4}\right)
$$

### Nyström extension

The anchor Gram matrix $K_\text{reg}$ is eigendecomposed:

$$
K_\text{reg} = U \Lambda U^\top \qquad \text{[SelfAdjointEigenSolver]}
$$

The embedding dimension $d$ is auto-selected as the smallest $d'$ such that the top $d'$ eigenvalues explain at least 95% of total non-negative variance:

$$
d = \min\lbrace d' : \frac{\sum_{i=p-d'+1}^{p} \lambda_i}{\sum_i \max(\lambda_i, 0)} \geq 0.95 \rbrace
$$

The projection matrix $W = U_d \cdot \text{diag}(\lambda_d^{-1/2})$, where $U_d$ and $\lambda_d$ are the top-$d$ eigenvectors and eigenvalues, maps genomes to $d$-dimensional unit vectors. For non-anchors:

$$
k_G[a] = J_1(G_i, \text{anchor}_a), \quad
\tilde{\mathbf{e}}(G_i) = W^\top \tilde{k}_G, \quad
\mathbf{e}(G_i) = \tilde{\mathbf{e}}(G_i) \,/\, \|\tilde{\mathbf{e}}(G_i)\|_2
$$

where $\tilde{k}_G$ is $k_G$ after degree normalisation. Non-anchors use sig1 only.

### Embedding dimension

The default maximum is $d_\text{cfg} = 256$. Empirical MAE vs ANI on a 159-genome cross-species validation:

| d   | MAE (90–95% ANI) | MAE (95–99% ANI) | Build time |
|-----|-----------------|-----------------|------------|
| 64  | 4.1%            | 1.8%            | 1×         |
| 128 | 2.7%            | 1.1%            | 1.5×       |
| 256 | 2.1%            | 0.73%           | 2.8×       |
| 512 | 2.0%            | 0.68%           | 5.2×       |

Beyond $d=256$, accuracy improves by $< 0.1\%$ while cost doubles. The embedding provides approximate nearest-neighbour ranking; Phase 7 OPH sketch comparison handles the remaining error.

---

## Phase 3: HNSW index

An [HNSW](https://arxiv.org/abs/1603.09320) index (Malkov & Yashunin 2018) is built over all $n$ unit-sphere embeddings using inner product as the metric. Default parameters: $M = 16$, ef\_construction $= 64$, ef\_search $= 50$.

The index serves two purposes:
- Computing isolation scores (Phase 4): finding the $k_\text{iso}=10$ nearest neighbours of each genome; collecting up to $K_\text{cap}$ edges per genome for the adaptive MST threshold derivation
- Finding too-close representative pairs for merging (Phase 6)

For $n \leq 50$ genomes, HNSW overhead exceeds $O(n^2)$ brute-force dot products; the brute-force path is used instead.

**Reduced ef\_search for isolation scores.** Only approximate nearest-neighbour ordering is needed; ef\_search is set to $\max(50,\, \min(200,\, n/100))$ during the isolation pass, then restored. This cuts HNSW query time without affecting representative quality.

---

## Phase 4: Isolation scores and diversity threshold

### Isolation score

For each genome $G_i$, the isolation score is the mean angular distance to its $k_\text{iso} = 10$ nearest neighbours:

$$
\text{isolation}(G_i) = \frac{1}{k_\text{iso}} \sum_{j \in k_\text{iso}\text{NN}(G_i)} \frac{\arccos(\mathbf{e}_i \cdot \mathbf{e}_j)}{\pi}
$$

Higher isolation = more separated from neighbours = stronger candidate for a representative.

### Diversity threshold

The diversity threshold $\theta$ controls when FPS terminates. It is derived from the k-NN graph of the taxon, capped at the user ANI threshold:

$$
\theta = \min\!\left(\theta_\text{MST},\ \frac{\arccos(J_\text{ANI})}{\pi}\right)
$$

**$\theta_\text{MST}$: MST max-edge threshold.** After the isolation-score pass, k-NN edges are collected (genomic outliers with isolation score $> \mu + 2\sigma$ excluded) and [Kruskal's algorithm](https://en.wikipedia.org/wiki/Kruskal%27s_algorithm) builds the minimum spanning tree of the remaining genomes. The longest MST edge $\theta_\text{MST}$ is the minimum angular distance at which the k-NN proximity graph becomes connected: the natural inter-strain scale of the taxon.

**Kruskal's construction.** The k-NN edges are sorted in ascending order of angular distance. Union-Find processes them greedily, adding each edge only if it connects two previously disconnected components. The algorithm terminates as soon as a single component spans all non-outlier genomes; the edge that triggered this merge is $\theta_\text{MST}$ by construction.

**Adaptive $k$ selection.** Isolation scoring uses a fixed $k_\text{iso}$ neighbours. MST edge collection uses a two-phase adaptive scan with budget $K_\text{cap} = \min(64, n-1)$.

**Phase A — DSU connectivity scan.** The k-NN edges are added column by column, incrementing $k$ from 1 to $K_\text{cap}$. A Union-Find structure tracks component membership. The scan halts at the first $k$ for which the core k-NN graph (outliers excluded) becomes fully connected; this value is recorded as $k_\text{conn}$. If no $k \leq K_\text{cap}$ achieves connectivity (e.g., a taxon with genuine phylogenetic sub-lineages), $k_\text{conn} = -1$.

**Phase B — Bottleneck stability probe.** Starting from $k_\text{conn}$ (or $K_\text{cap}$ if $k_\text{conn} = -1$), the bottleneck $B(k)$ (MST max edge at a given $k$) is evaluated at a probe ladder $\{1,2,3,4,6,8,12,16,24,32,48,64\}$ plus $K_\text{cap}$ as a reference value. The smallest $k$ in the ladder for which $B(k)$ is within 3% of $B(K_\text{cap})$ is taken as $k_\text{stable}$. Using $k_\text{conn}$ alone is insufficient: the first edge that connects the graph is often a brittle long-range bridge, giving an artificially elevated $B(k_\text{conn})$. The probe identifies the point at which the bottleneck has stabilised and further neighbours no longer change the MST max edge materially.

The MST is then constructed at $k_\text{stable}$ edges per genome. For $k_\text{conn} = -1$ taxa, $k_\text{stable} = K_\text{cap}$ and per-component thresholds are used. Isolation scoring is unaffected.

**Observed values.** *E. coli* 200 genomes: $k_\text{conn}=8$, $k_\text{stable}=24$, $K_\text{cap}=64$. *S. enterica* 2,000 genomes: $k_\text{conn}=-1$, $k_\text{stable}=64$.

For clonal taxa (tight NN distribution), $\theta_\text{MST}$ is small and drives more representatives. For diverse taxa, the ANI cap is the binding constraint. Outlier genomes are excluded from the MST to prevent contaminated assemblies from inflating $\theta_\text{MST}$ via long bridge edges.

**Instability detection.** Three flags are computed after MST construction:

| Flag | Condition | Severity | Interpretation |
|------|-----------|----------|----------------|
| `low_pair_count` | fewer than 20 non-outlier genomes | warning | MST built on too few points; threshold unreliable |
| `high_gap_ratio` | $\theta_\text{MST} / \text{NN}_{P95} > 5$ | warning | one long bridge edge dominates; MST may conflate sub-populations |
| `disconnected_mst` | MST has $> 1$ component at $k_\text{iso}$ (not at adaptive $k_\text{stable}$) | warning | k-NN graph not connected; threshold is a lower bound |

`high_gap_ratio` uses $\text{NN}_{P95}$ as denominator. Clonal taxa (e.g. *E. coli*, *S. enterica*) have $\text{NN}_{P50} \approx 0.003$ (intra-clone distances), which would produce false alarms when the MST bridge correctly captures an inter-pathotype gap; $\text{NN}_{P95}$ represents the upper bound of normal within-population variation and is a more robust reference.

`disconnected_mst` is only raised when the graph remains disconnected even at the adaptive $k_\text{stable}$ (i.e., $k_\text{conn} = -1$). A graph that is disconnected at $k_\text{iso}$ but connects before $K_\text{cap}$ logs an info message instead.

When any warning flag is set, the MST threshold is used as-is. Override with `--geodesic-diversity-threshold` if the inferred threshold is unsuitable.

When $\theta_\text{MST}$ is unavailable (small-$n$ brute-force path), $\text{NN}_{P95}$ is used directly.

---

## Phase 5: Farthest point sampling

[Farthest point sampling (FPS)](https://en.wikipedia.org/wiki/Farthest-first_traversal) selects representatives greedily: each step adds the uncovered genome with the highest fitness score. For unweighted FPS on a metric space, this gives a 2-approximation to the k-center problem (Gonzalez 1985).

**Fitness score.** Each genome is weighted by assembly quality and size to bias selection toward complete, uncontaminated assemblies:

$$
\text{fitness}_i = (1 - s_i) \cdot \frac{q_i}{100} \cdot \sqrt{\frac{L_i}{L_m}}
$$

where $s_i = \max_{r \in R} \mathbf{e}_r \cdot \mathbf{e}_i$ is the dot-product similarity to the nearest current representative, $q_i$ is the CheckM2 quality score ($\text{completeness} - 5 \times \text{contamination}$; defaults to 100 when unavailable), $L_i$ is genome length, and $L_m$ is the taxon median genome length. The formal 2-approximation guarantee does not carry over to this weighted variant; coverage is evaluated empirically.

**Algorithm:**
1. Seed: select the genome maximising $\text{isolation} \times (\text{quality}/100) \times \sqrt{L_i / L_\text{med}}$ as the first representative
2. Maintain $s_j$ for all active (uncovered) genomes after each representative is added
3. Each round: partial-sort the active set by fitness; promote the top-$B = 16$ genomes to representatives
4. Remove newly covered genomes ($(1 - s_i) < \theta$) from the active set
5. Terminate when the active set is empty, or the top candidate's angular distance $\arccos(s_i)/\pi < \theta$

Batching $B = 16$ candidates fuses 16 distance updates into one parallel pass, reducing OpenMP synchronisation overhead.

---

## Phase 6: Union-Find merge

Quality weighting can place two representatives closer than intended. Representatives with embedding distance below $d_\text{min}$ are merged via [Union-Find](https://en.wikipedia.org/wiki/Disjoint-set_data_structure): the pair is collapsed to the survivor with higher $\text{quality} \times \text{size}$.

Merge candidates are found via HNSW search over the representative set. $d_\text{min} = \min(\text{NN}_{P5},\ \theta / 2)$.

---

## Phase 7: Borderline verification

### Approximation error

Nyström embedding introduces geometric error. The implementation uses $\varepsilon = \min(3/\sqrt{d},\ 0.3)$ as an empirical error tolerance. A genome at embedding distance in

$$
\left[\theta(1-\varepsilon),\ \theta\right)
$$

is borderline covered and verified by a direct sketch comparison.

### OPH sketch Jaccard check

For each borderline-covered genome $G_i$:

1. Find the top-3 closest representatives by embedding dot product.
2. For each candidate representative $R_k$, compute dual-sketch averaged OPH Jaccard:

$$
J_\text{dual} = \frac{J(\text{sig1}_i,\ \text{sig1}_{R_k}) + J(\text{sig2}_i,\ \text{sig2}_{R_k})}{2}
$$

3. Convert: $d_\text{sketch} = \arccos\!\left(\min(1, \max(0, J_\text{dual}))\right) / \pi$
4. Promote $G_i$ to representative only if all checked representatives satisfy $d_\text{sketch} \geq \theta$.

This uses OPH sketch Jaccard ($m = 10{,}000$ bins), not exact ANI, with variance $J(1-J) / (2\, m_\text{real})$.

---

## Phase 8: Universal OPH certification

Phase 7 (borderline verification) only checks genomes near the embedding coverage boundary. Nyström approximation error is not uniform; it is larger for sparse genomes (MAGs, incomplete assemblies) whose k-mer sets differ substantially from the anchor sample. A genome that appears covered in embedding space may be genuinely uncovered in OPH sketch space.

Phase 8 runs a universal coverage check over every non-representative genome.

**J_cert threshold.** The OPH Jaccard equivalent of the user ANI threshold:

$$
q = \text{ANI}^k, \qquad J_\text{cert} = \frac{q}{2 - q}
$$

For the default threshold of 95% ANI with $k = 21$: $q \approx 0.341$, $J_\text{cert} \approx 0.212$.

**Algorithm.** For each non-representative genome $G_i$ (excluding contamination-excluded genomes):

1. **Fast path**: compute OPH Jaccard between $G_i$ and its assigned representative. If $J \geq J_\text{cert}$, genome is certified; continue.
2. **Exhaustive scan**: if the fast path fails, compute OPH Jaccard against every representative. Reassign $G_i$ to the closest representative with $J \geq J_\text{cert}$.
3. **Repair queue**: if no representative passes the threshold, $G_i$ is added to the repair queue and promoted to a new representative.

The outer loop is parallelised with OpenMP (`schedule(dynamic, 256)`); each thread maintains a local repair queue merged after the barrier.

**Coverage guarantee.** After Phase 8, every non-representative genome is within $J_\text{cert}$ OPH Jaccard of at least one representative, independent of Nyström approximation error. Because $J_\text{cert}$ is derived directly from the user ANI threshold via the Mash formula, this is an explicit sketch-space coverage guarantee. The remaining uncertainty is OPH estimation variance: $\sigma_J = \sqrt{J(1-J)/m} \approx 0.004$ at 95% ANI, corresponding to $< 0.1\%$ ANI error near the default threshold with dense assemblies. Sparse genomes (low $m_\text{real}$) have higher variance and looser guarantees.

---

## n=2 fast path

For taxa with exactly 2 genomes, the full pipeline is unnecessary. The outcome depends only on whether the two genomes are similar enough to collapse:

- If OPH Jaccard(A, B) → ANI $\geq$ ani_threshold: select the genome with higher quality score as the sole representative.
- Otherwise: both are representatives.

The OPH direct Jaccard at $m=10{,}000$ has sub-0.2% ANI error (see Phase 1 table), making this comparison reliable. For $n=1$ taxa, the single genome is trivially the representative.

---

## ANI from Jaccard

[ANI](https://en.wikipedia.org/wiki/Average_nucleotide_identity) is related to k-mer Jaccard via the [Mash formula](https://doi.org/10.1186/s13059-016-0997-x) (Ondov et al. 2016). Under a Poisson substitution model with per-base rate $r = 1 - p$ (ANI $= p \times 100\%$), the expected Jaccard for equal-size genomes is:

$$
J \approx \frac{p^k}{2 - p^k}, \qquad p = \frac{\text{ANI}}{100}
$$

Inverting exactly:

$$
\text{ANI} = \left(\frac{2J}{1 + J}\right)^{1/k} \times 100
$$

The model assumes equal genome sizes, i.i.d. substitutions, and negligible indels. It degrades for highly divergent sequences (ANI < 90%) where indels dominate.

The angular distance threshold corresponding to an ANI cutoff (before degree normalisation):

$$
\theta_\text{ANI} = \frac{\arccos(J_\text{threshold})}{\pi}
$$

After degree normalisation, dot products approximate a normalised-graph similarity rather than raw Jaccard. Phase 7 corrects borderline decisions back to sketch Jaccard space.

---

## Complexity

| Phase | Operation | Complexity |
|-------|-----------|------------|
| Sketch | OPH per genome (rolling hash) | $O(L)$ where $L$ = genome length |
| Embed anchors | Gram matrix $K$ | $O(p^2 m)$ |
| Embed all | Nyström extension (sig1 only) | $O(npm)$ |
| Eigendecomp | $K_\text{reg}$ ($p \times p$) | $O(p^3)$ |
| Index | HNSW build | $O(nM \log n)$ |
| Score | kNN isolation ($k_\text{iso}$) + adaptive DSU scan + MST (Kruskal, $k_\text{stable} \leq K_\text{cap}$) | $O(n \log n)$ |
| Select | FPS batched ($B=16$) | $O(nrd/B)$ where $r$ = number of reps |
| Merge | HNSW search + Union-Find | $O(r \log r)$ |
| Verify | OPH sketch Jaccard | $O(n_\text{borderline} \cdot m)$ |
| Certify | Universal OPH coverage check | $O(nm)$ fast path; $O(n \cdot r \cdot m)$ worst case |

Typical values: $p \approx 512$, $m = 10{,}000$, $d \approx 64$–$256$. Embedding (Nyström extension) dominates for $n > 10{,}000$; FPS dominates for medium-size taxa.

---

## Implementation notes

- **SIMD**: AVX2 in the OPH inner loop (32 bytes/cycle), anchor-slab Gram matrix (`_mm256_cmpeq_epi16`), and FPS update loops.
- **OpenMP**: parallel OPH sketching, Gram matrix rows, FPS fitness/update loops, HNSW isolation queries.
- **DuckDB**: all results persisted incrementally; interrupted runs resume by skipping completed taxa (`SELECT 1 FROM results WHERE taxonomy = ?`).
- **Anchor slab**: anchor signatures ($p \times m \times 2$ bytes $\approx 10\ \text{MB}$ for $p=512$) packed into a contiguous aligned buffer for cache-friendly Gram matrix computation.
- **Producer-consumer I/O**: genome decompression overlapped with k-mer computation via bounded semaphore.
- **NFS robustness**: `embed_genome()` retries failed reads up to 3 times with 500 ms / 1 s backoff; permanently failed genomes recorded in the `jobs_failed` table.

---

## References

- Ondov et al. (2016) *Mash: fast genome and metagenome distance estimation using MinHash*. Genome Biology. [doi:10.1186/s13059-016-0997-x](https://doi.org/10.1186/s13059-016-0997-x)
- Li, P. & König, A.C. (2011) *b-Bit Minwise Hashing*. WWW 2011. [doi:10.1145/1989323.1989399](https://doi.org/10.1145/1989323.1989399)
- Li, P. & König, A.C. (2012) *One Permutation Hashing*. NIPS. [link](https://papers.nips.cc/paper/2012/hash/eaa32c96f620053cf442ad32258076b9-Abstract.html)
- Williams, C.K.I. & Seeger, M. (2001) *Using the Nyström Method to Speed Up Kernel Machines*. NIPS. [link](https://proceedings.neurips.cc/paper/2000/file/19de10adbaa1b2ee13f77f679fa1483a-Paper.pdf)
- Gonzalez, T.F. (1985) *Clustering to minimize the maximum intercluster distance*. Theoretical Computer Science 38:293–306.
- Malkov, Y.A. & Yashunin, D.A. (2018) *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs*. IEEE TPAMI 42(4):824–836. [doi:10.1109/TPAMI.2018.2889473](https://doi.org/10.1109/TPAMI.2018.2889473)
- Chowdhury et al. (2023) *CheckM2: a rapid, scalable and accurate tool for assessing microbial genome quality using machine learning*. Nature Methods. [doi:10.1038/s41592-023-01940-w](https://doi.org/10.1038/s41592-023-01940-w)
