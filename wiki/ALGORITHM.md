# geodesic: Algorithm Reference

**Spherical genome embeddings for diverse representative selection**

---

## Overview

`geodesic` selects a diverse set of representative genomes per taxon for reference-based short-read mapping. All-pairs comparison is infeasible at scale — a species like *E. coli* has 200,000+ assemblies. geodesic embeds all genomes on a unit sphere from their sketches, greedily picks the most spread-out subset by Farthest Point Sampling, then runs a sketch-space certification pass so every genome lands within the ANI threshold of at least one representative.

The pipeline has eight phases:

```
Sketch → Embed → Index → Score → Select → Merge → Verify → Certify
  (1)      (2)    (3)     (4)     (5)      (6)     (7)      (8)
```

| Phase | Name | What it does |
|-------|------|-------------|
| 1 | Sketch | Read pre-computed OPH signatures from the genopack archive (stored at multiple k-mer sizes; auto-calibration selects k and m per taxon) |
| 2 | Embed | Project all genomes onto a unit sphere using a small anchor subset (Nyström) |
| 3 | Index | Build an approximate nearest-neighbour index (HNSW) over the sphere |
| 4 | Score | Measure how isolated each genome is; infer the taxon's natural diversity scale from an MST |
| 5 | Select | Greedily pick the most spread-out genomes with quality weighting (Farthest Point Sampling) |
| 6 | Merge | Collapse representative pairs that are too close via Union-Find |
| 7 | Verify | Re-check borderline genomes near the coverage boundary with exact sketch Jaccard |
| 8 | Certify | Universal coverage pass: every genome is certified against its representative in sketch space |

For taxa with 20 or fewer genomes, the full pipeline is replaced by a brute-force all-pairs cover; see [Tiny-taxon fast path](#tiny-taxon-fast-path-n--20).

---

## Phase 1: OPH sketching

### One-permutation hashing

For each genome, geodesic reads a pre-computed [One-Permutation Hash (OPH)](https://papers.nips.cc/paper/2012/hash/eaa32c96f620053cf442ad32258076b9-Abstract.html) signature from the genopack archive. The archive stores OPH signatures at multiple k-mer sizes (e.g. $k \in \{16, 21, 31\}$, $m = 10{,}000$ bins per size). Auto-calibration selects the appropriate k and m from a small sample of genome pairs; the default tier for 95–99% ANI taxa uses $m = 10{,}000$ and $k = 21$. The calibration pre-probe uses 500 bins to estimate the nearest-neighbour distance distribution before committing to the full sketch size.

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

sig1 uses seed = `--seed` (default 42), sig2 uses seed = `--seed`+1 (default 43). Both the bin index and the per-bin value are derived from this one hash:

$$
t = \left\lfloor \frac{h \cdot m}{2^{64}} \right\rfloor, \qquad \mathrm{sig}[t] = \min\!\left(\mathrm{sig}[t],\ \mathrm{hi32}(h)\right)
$$

The stored uint32 value is truncated to uint16 at storage time (retaining bits 32–47 of $h$).

**Densification.** After scanning all k-mers, empty bins are filled by nearest-neighbour propagation, following the OPH densification scheme of Li, Owen & Zhang (2012):

```
Forward:  if sig[t] = EMPTY and sig[t-1] ≠ EMPTY: sig[t] = SplitMix64(sig[t-1] XOR t)
Backward: if sig[t] = EMPTY and sig[t+1] ≠ EMPTY: sig[t] = SplitMix64(sig[t+1] XOR t)
```

Densified bins carry no independent information about k-mer overlap; their values are deterministic functions of real neighbours. The collision probability below applies only to real (pre-densification) bins.

**Jaccard property.** For any bin $t$ that is real in at least one genome:

$$
\Pr\!\left[\mathrm{sig}_A[t] = \mathrm{sig}_B[t]\right] = J(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

where $A$ and $B$ are the sets of distinct canonical k-mers. Over $m$ bins, the fraction of collisions is an unbiased estimator of $J$ with approximate variance:

$$
\mathrm{Var}(\hat{J}) \approx \frac{J(1-J)}{m_{\mathrm{real}}}
$$

where $m_{\mathrm{real}}$ is the number of bins that are real in at least one of $A$ or $B$.

### OPH vs bottom-k MinHash

[Bottom-k MinHash](https://en.wikipedia.org/wiki/MinHash) keeps the $k$ smallest hash values across the entire genome. OPH partitions the hash space into $m$ bins and keeps one value per bin. OPH is preferred because one pass fills all $m$ bins, and the per-bin occupancy bitmask (used for sparse-anchor correction in Phase 2) is obtained at no extra cost.

### Dual OPH sketches

Two independent OPH signatures (sig1, sig2) are read per genome; sig1 was computed with seed `--seed` (default 42), sig2 computed with seed `--seed`+1 (default 43). The anchor Gram matrix uses dual-sketch averaged Jaccard (see Phase 2):

$$
K[i,j] = \frac{J_1(\mathrm{anchor}_i, \mathrm{anchor}_j) + J_2(\mathrm{anchor}_i, \mathrm{anchor}_j)}{2}
$$

For two independent $J$ estimators, averaging halves the variance:

$$
\mathrm{Var}\!\left(\frac{J_1 + J_2}{2}\right) = \frac{J(1-J)}{2\, m_{\mathrm{real}}}
$$

**16-bit storage.** Each per-bin value is stored as a uint16, halving RAM to $20\ \mathrm{KB}$ per sketch per genome. Truncation increases false-match probability; the [b-bit MinHash bias correction](https://dl.acm.org/doi/10.1145/1989323.1989399) (Li & König 2011) corrects this:

$$
\hat{J}_{\mathrm{corr}} = \max\!\left(0,\ \frac{\hat{J}_{\mathrm{raw}} - 2^{-16}}{1 - 2^{-16}}\right)
$$

**sig2 loading.** Both sig1 and sig2 are loaded eagerly for all genomes at startup from the genopack archive's SKCH section. sig2 is used for the anchor Gram matrix (dual-sketch averaging) and then freed before Phase 5; Phase 7 and Phase 8 use sig1 only.

### Fill fraction

The fill fraction $f_i = n_{\mathrm{real},i} / m$ is the fraction of bins with at least one real k-mer before densification:

$$
\mathbb{E}[f_i] \approx 1 - e^{-|G_i|/m}
$$

For complete bacterial genomes ($|G| \sim 10^6$ k-mers, $m = 10{,}000$), $f \approx 1$. Highly incomplete assemblies ($f \ll 0.2$) have elevated OPH variance and trigger containment-based corrections in Phase 2. At the 500-bin calibration pre-probe tier, the same assemblies still satisfy $f \approx 1$.

### K-mer size and OPH accuracy

The default is $k = 21$. The k-mer size determines how quickly Jaccard drops with ANI (Mash model, with $p = \mathrm{ANI}/100$: $q = p^k$, $J = q/(2-q)$):

| ANI  | k=16  | k=21  | k=31  |
|------|-------|-------|-------|
| 85%  | 0.039 | 0.017 | 0.003 |
| 90%  | 0.102 | 0.058 | 0.019 |
| 95%  | 0.282 | 0.205 | 0.114 |
| 99%  | 0.741 | 0.680 | 0.578 |

At 95–99% ANI (typical bacterial species), $k=21$ gives $J \in [0.205, 0.680]$, sufficient for reliable ranking. Use `--geodesic-kmer-size 16` for highly diverse taxa (ANI < 95%) where $k=21$ gives $J < 0.02$ and loses signal; use `--geodesic-kmer-size 31` for clonal taxa (ANI > 99%) to spread compressed $J$ values apart.

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

The anchor count is $p = \min(n,\ \max(200,\ 2 \cdot d_{\mathrm{cfg}}))$, where $d_{\mathrm{cfg}}$ is the configured maximum embedding dimension (default 256, `--geodesic-dim`). The actual embedding dimension $d$ is auto-selected from the anchor eigenspectrum (see below) and may be less than $d_{\mathrm{cfg}}$.

Genomes are split into a **dense pool** ($f_i \geq 0.85$, shuffled uniformly) and a **sparse pool** (sparser assemblies, sorted by descending fill). Anchors are drawn from the dense pool first, then from the sparse pool in fill order. This ensures the anchor Gram matrix covers the full range of genome completeness, including MAGs with few occupied bins.

### Anchor Gram matrix

The $p \times p$ anchor Gram matrix $K_{\mathrm{raw}}$ is computed using dual-sketch averaged Jaccard:

$$
K_{\mathrm{raw}}[i,j] = \frac{J_1(\mathrm{anchor}_i, \mathrm{anchor}_j) + J_2(\mathrm{anchor}_i, \mathrm{anchor}_j)}{2}
$$

The raw Gram matrix $K_{\mathrm{raw}}$ uses pure Jaccard for all anchor pairs.
Sparse-anchor containment blending was considered but removed: the blended kernel
$(1-\alpha)J + \alpha\,C_{\mathrm{occ}}$ is not positive semi-definite, which breaks
the spectral decomposition required in the next step. Containment corrections are
applied at Phase 8 (certification) where no PSD constraint is needed.

### Gram matrix regularisation

**Symmetric Laplacian normalisation** removes hub-anchor bias:

$$
K_{\mathrm{norm}}[i,j] = \frac{K_{\mathrm{raw}}[i,j]}{\sqrt{d_i \cdot d_j}}, \qquad d_i = \sum_j K_{\mathrm{raw}}[i,j]
$$

equivalently $K_{\mathrm{norm}} = D^{-1/2} K_{\mathrm{raw}} D^{-1/2}$. After this step, dot products approximate a normalised-graph similarity, not raw Jaccard. Phase 7 corrects borderline decisions back to raw sketch Jaccard space.

**Tikhonov ridge** prevents near-zero eigenvalues from blowing up the projection:

$$
K_{\mathrm{reg}} = K_{\mathrm{norm}} + \lambda I, \qquad \lambda = 0.01 \cdot \max\!\left(\overline{K}_{\mathrm{diag}},\ 10^{-4}\right)
$$

### Nyström extension

The anchor Gram matrix $K_{\mathrm{reg}}$ is eigendecomposed:

$$
K_{\mathrm{reg}} = U \Lambda U^\top \qquad \mathrm{[SelfAdjointEigenSolver]}
$$

The embedding dimension $d$ is auto-selected as the smallest $d'$ such that the top $d'$ eigenvalues explain at least 95% of total non-negative variance:

$$
d = \min\lbrace d' : \frac{\sum_{i=p-d'+1}^{p} \lambda_i}{\sum_i \max(\lambda_i, 0)} \geq 0.95 \rbrace
$$

The projection matrix $W = U_d \cdot \mathrm{diag}(\lambda_d^{-1/2})$, where $U_d$ and $\lambda_d$ are the top-$d$ eigenvectors and eigenvalues, maps genomes to $d$-dimensional unit vectors. For non-anchors:

$$
k_G[a] = J_1(G_i, \mathrm{anchor}_a), \quad
\tilde{\mathbf{e}}(G_i) = W^\top \tilde{k}_G, \quad
\mathbf{e}(G_i) = \tilde{\mathbf{e}}(G_i) \,/\, \|\tilde{\mathbf{e}}(G_i)\|_2
$$

where $\tilde{k}_G$ is $k_G$ after degree normalisation. Non-anchors use sig1 only.

### Embedding dimension

The default maximum is $d_{\mathrm{cfg}} = 256$. Empirical MAE vs ANI on a 159-genome cross-species validation:

| d   | MAE (90–95% ANI) | MAE (95–99% ANI) | Build time |
|-----|-----------------|-----------------|------------|
| 64  | 4.1%            | 1.8%            | 1×         |
| 128 | 2.7%            | 1.1%            | 1.5×       |
| 256 | 2.1%            | 0.73%           | 2.8×       |
| 512 | 2.0%            | 0.68%           | 5.2×       |

Beyond $d=256$, accuracy improves by $< 0.1\%$ while cost doubles. The embedding provides approximate nearest-neighbour ranking; Phase 7 OPH sketch comparison handles the remaining error.

---

## Phase 3: HNSW index

A [Hierarchical Navigable Small World (HNSW)](https://arxiv.org/abs/1603.09320) index (Malkov & Yashunin 2018) is built over all $n$ unit-sphere embeddings using inner product as the metric. HNSW is a graph-based approximate nearest-neighbour structure that supports sub-linear query time by navigating a layered proximity graph from coarse to fine resolution. Default parameters: $M = 48$, ef\_construction $= 400$, ef\_search $= 50$ at build; the isolation pass raises it to $\max(2\,K_{\mathrm{cap}},\, \min(200,\, n/100))$ during querying, and it is raised further at each $K_{\mathrm{cap}}$ retry level.

The index serves two purposes:
- Computing isolation scores (Phase 4): finding the $k_{\mathrm{iso}} = \max(10, \min(20, \lfloor\log_2 n\rfloor))$ nearest neighbours of each genome; collecting up to $K_{\mathrm{cap}}$ edges per genome for the adaptive MST threshold derivation
- Finding too-close representative pairs for merging (Phase 6)

For $n \leq 50$ genomes, HNSW overhead exceeds $O(n^2)$ brute-force dot products; the brute-force path is used instead.

**Reduced ef\_search for isolation scores.** Only approximate nearest-neighbour ordering is needed; ef\_search is set to $\max(2 K_{\mathrm{cap}},\, \min(200,\, n/100))$ during the isolation pass, then restored. This cuts HNSW query time without affecting representative quality.

---

## Phase 4: Isolation scores and diversity threshold

### Isolation score

For each genome $G_i$, the isolation score is the mean angular distance to its $k_{\mathrm{iso}} = \max(10, \min(20, \lfloor\log_2 n\rfloor))$ nearest neighbours ($k_{\mathrm{iso}} \in [10, 20]$, scales with taxon size):

$$
\mathrm{isolation}(G_i) = \frac{1}{k_{\mathrm{iso}}} \sum_{j \in k_{\mathrm{iso}}\mathrm{NN}(G_i)} \frac{\arccos(\mathbf{e}_i \cdot \mathbf{e}_j)}{\pi}
$$

Higher isolation = more separated from neighbours = stronger candidate for a representative.

### Diversity threshold

The diversity threshold $\theta$ controls when FPS terminates. It is derived from the k-NN graph of the taxon, capped at the user ANI threshold:

$$
\theta = \min\!\left(\theta_{\mathrm{ANI}},\ \max\!\left(\theta_{\mathrm{MST}},\ \frac{\theta_{\mathrm{ANI}}}{4}\right)\right),
\qquad \theta_{\mathrm{ANI}} = \frac{\arccos(J_{\mathrm{ANI}})}{\pi}
$$

The $\theta_{\mathrm{ANI}}/4$ floor prevents clonal databases (where all $K$ nearest neighbours are near-duplicates) from driving $\theta_{\mathrm{MST}}$ far below the biological ANI scale. By the triangle inequality, this floor also ensures the coverage-preserving merge property.

**$\theta_{\mathrm{MST}}$: MST max-edge threshold.** After the isolation-score pass, k-NN edges are collected (genomic outliers with isolation score exceeding a $\bar{x} + 2\sigma$ threshold excluded from the MST — note this is a simpler mean+2σ pre-filter, not the MAD threshold used for final outlier classification) and [Kruskal's algorithm](https://en.wikipedia.org/wiki/Kruskal%27s_algorithm) builds the minimum spanning tree of the remaining genomes. The longest MST edge $\theta_{\mathrm{MST}}$ is the minimum angular distance at which the k-NN proximity graph becomes connected: the natural inter-strain scale of the taxon.

**Kruskal's construction.** The k-NN edges are sorted in ascending order of angular distance. Union-Find processes them greedily, adding each edge only if it connects two previously disconnected components. The algorithm terminates as soon as a single component spans all non-outlier genomes; the edge that triggered this merge is $\theta_{\mathrm{MST}}$ by construction.

**Adaptive $k$ selection.** Isolation scoring uses a fixed $k_{\mathrm{iso}}$ neighbours. MST edge collection uses a two-phase adaptive scan with budget $K_{\mathrm{cap}}$ scaled by taxon size: 64 for $n \leq 5{,}000$, 128 for $5{,}000 < n \leq 50{,}000$, 256 for $n > 50{,}000$ (further capped by `--k-cap-max`).

**Phase A -- DSU connectivity scan.** The k-NN edges are added column by column, incrementing $k$ from 1 to $K_{\mathrm{cap}}$. A [Disjoint Set Union (DSU)](https://en.wikipedia.org/wiki/Disjoint-set_data_structure) structure (also called Union-Find) tracks component membership -- each genome starts in its own component, and merging two sets takes near-constant amortised time. The scan halts at the first $k$ for which the core k-NN graph (outliers excluded) becomes fully connected; this value is recorded as $k_{\mathrm{conn}}$. If no $k \leq K_{\mathrm{cap}}$ achieves connectivity (e.g., a taxon with genuine phylogenetic sub-lineages), $k_{\mathrm{conn}} = -1$.

**Phase B -- Bottleneck stability probe.** Starting from $k_{\mathrm{conn}}$ (or $K_{\mathrm{cap}}$ if $k_{\mathrm{conn}} = -1$), the bottleneck $B(k)$ (MST max edge at a given $k$) is evaluated at a probe ladder $\{1,2,3,4,6,8,12,16,24,32,48,64\}$ plus $K_{\mathrm{cap}}$ as a reference value. The smallest $k$ in the ladder for which $B(k)$ is within 3% of $B(K_{\mathrm{cap}})$ is taken as $k_{\mathrm{stable}}$. Using $k_{\mathrm{conn}}$ alone is insufficient: the first edge that connects the graph is often a brittle long-range bridge, giving an artificially elevated $B(k_{\mathrm{conn}})$. The probe identifies the point at which the bottleneck has stabilised and further neighbours no longer change the MST max edge materially.

The MST is then constructed at $k_{\mathrm{stable}}$ edges per genome. For $k_{\mathrm{conn}} = -1$ taxa, $k_{\mathrm{stable}} = K_{\mathrm{cap}}$ and per-component thresholds are used. Isolation scoring is unaffected.

**Observed values.** *E. coli* 200 genomes: $k_{\mathrm{conn}}=8$, $k_{\mathrm{stable}}=24$, $K_{\mathrm{cap}}=64$. *S. enterica* 2,000 genomes: $k_{\mathrm{conn}}=-1$, $k_{\mathrm{stable}}=64$.

For clonal taxa (tight NN distribution), $\theta_{\mathrm{MST}}$ is small and drives more representatives. For diverse taxa, the ANI cap is the binding constraint. Outlier genomes are excluded from the MST to prevent contaminated assemblies from inflating $\theta_{\mathrm{MST}}$ via long bridge edges.

**Instability detection.** Three flags are computed after MST construction:

| Flag | Condition | Severity | Interpretation |
|------|-----------|----------|----------------|
| `low_pair_count` | fewer than 20 non-outlier genomes | warning | MST built on too few points; threshold unreliable |
| `pathological_bridge` | smallest MST component $\leq 10$ genomes AND $\leq 0.5\%$ of non-outliers, AND the terminal merge is isolated (few other MST edges near the same scale) | warning | an anomalous genome (e.g. misassigned singleton) is bridging two distinct taxa; the bridge MST edge is not a real inter-strain scale |
| `disconnected_mst` | MST has $> 1$ component at $K_{\mathrm{cap}}$ (i.e., $k_{\mathrm{conn}} = -1$) | warning | k-NN graph not connected; threshold is a lower bound |

`pathological_bridge` distinguishes anomalous-singleton bridges from genuine multi-scale population structure (e.g. *S. enterica* serovars), which also produce a large MST bridge but have substantial components on both sides and many inter-group edges near the same scale. The ratio $\theta_{\mathrm{MST}} / \mathrm{NN}_{P95}$ is logged as a diagnostic when it exceeds 5, but is not itself a warning flag — high ratios are expected and correct for taxa with genuine pathotype gaps.

`disconnected_mst` is only raised when the graph remains disconnected even at the adaptive $k_{\mathrm{stable}}$ (i.e., $k_{\mathrm{conn}} = -1$). A graph that is disconnected at $k_{\mathrm{iso}}$ but connects before $K_{\mathrm{cap}}$ logs an info message instead.

When any warning flag is set, the MST threshold is used as-is. The threshold is always inferred from the data (ANI cap + MST bottleneck); it is not a user-tunable parameter.

When $\theta_{\mathrm{MST}}$ is unavailable (small-$n$ brute-force path), $\mathrm{NN}_{P95}$ is used directly.

**K_cap retry and bridge diagnostic.** When the k-NN graph fails to connect at $K_{\mathrm{cap}} = 64$, the algorithm retries at $K_{\mathrm{cap}} = 128$ then 256, rebuilding the full HNSW index with raised `ef_search` at each level (maximum configurable via `--k-cap-max`). HNSW is an approximate nearest-neighbour (ANN) structure -- it can miss true neighbours, especially in high-dimensional or poorly-connected graphs. If the graph still does not connect after retries, 50 cross-component genome pairs are sampled and their OPH Jaccard computed directly to diagnose whether the disconnection is a retrieval failure or genuine biology:

| Outcome | Condition | Interpretation |
|---------|-----------|----------------|
| `ANN_RECALL_GAP` | any sampled pair $J \geq J_{\mathrm{cert}}$ | HNSW missed a true neighbour; the gap is a retrieval artefact, not a real phylogenetic break |
| `POSSIBLE_GAP` | any sampled pair $J \geq J_{80}$ (80% ANI equivalent) | pairs are similar but below the coverage threshold; could be genuine phylogenetic sub-structure or marginal ANN recall |
| `NO_BRIDGE_FOUND` | no sampled pair passes either threshold | taxon genuinely contains multiple well-separated sub-populations at the current ANI scale |

Per-component thresholds are used for FPS when the graph remains disconnected.

---

## Phase 5: Farthest point sampling

[Farthest point sampling (FPS)](https://en.wikipedia.org/wiki/Farthest-first_traversal) selects representatives greedily: each step adds the uncovered genome with the highest fitness score. For unweighted FPS on a metric space, this gives a 2-approximation to the k-center problem (Gonzalez 1985).

**Fitness score.** Each genome is scored by distance, size, and a bounded quality factor:

$$
\mathrm{fitness}_i = d_i \cdot \sqrt{\frac{L_i}{L_m}} \cdot \left(0.5 + 0.5\,\hat{q}_i\right), \qquad \hat{q}_i = \mathrm{clamp}(q_i / 100,\ 0,\ 1)
$$

where $d_i = \sqrt{2(1 - s_i)}$ is the angular distance proxy to the nearest representative (monotonic in true angular distance), $L_i$ is genome length, and $L_m$ is the taxon median genome length. The quality factor lies in $[0.5, 1.0]$, so it shifts which genome is chosen for a region toward higher quality without dominating the distance term. Coverage and the stopping test use raw similarity, not fitness, so quality changes which representatives are picked but not how many or their ANI spread. Quality also breaks exact fitness ties (higher $q_i$ wins).

**Quality score** $q_i$, in priority order (`taxon_processor.cpp:24-38`):
- **From the pack QUAL section** (default for `derep --pack`): genopack's per-genome `quality_score`, which is **completeness-only** (equal to `completeness_effective`; `pack_reader.hpp:94-101`). This overrides the forms below whenever the pack carries QUAL — the standard case — so contamination does not enter representative ranking.
- **With CheckM2** and no pack QUAL score (`--checkm2`): $q_i = \mathrm{completeness} - 5 \times \mathrm{contamination}$
- **Otherwise**: $q_i = (n_{\mathrm{real\_bins}} / \mathrm{sketch\_size}) \times 100$ (sketch completeness)

The sketch-completeness proxy is the fraction of OPH bins filled. It does not anti-correlate with isolation, so it avoids the parabolic fitness of a centrality-based proxy.

**Algorithm:**
1. Seed: select the genome maximising $\mathrm{isolation} \times \sqrt{L_i / L_{\mathrm{med}}}$ as the first representative (quality breaks ties)
2. Maintain $s_j$ for all active (uncovered) genomes after each representative is added
3. Each round: partial-sort the active set by fitness; promote the top-$B = 16$ genomes to representatives
4. Remove newly covered genomes ($(1 - s_i) < \theta$) from the active set
5. Terminate when the active set is empty, or the top candidate's angular distance $\arccos(s_i)/\pi < \theta$

Batching $B = 16$ candidates fuses their distance updates into one parallel pass, reducing thread-pool synchronisation overhead.

---

## Phase 6: Union-Find merge

Quality weighting can place two representatives closer than intended. Representatives with embedding distance below $d_{\mathrm{min}}$ are merged via [Union-Find](https://en.wikipedia.org/wiki/Disjoint-set_data_structure): the pair is collapsed to the survivor that was selected earlier by FPS (lower index in the representative list), preserving the FPS diversity ordering.

Merge candidates are found via HNSW search over the representative set. $d_{\mathrm{min}} = \min(\mathrm{NN}_{P5},\ \theta / 4)$.

---

## Phase 7: Borderline verification

### Approximation error

Nyström embedding introduces geometric error. The implementation uses $\varepsilon = \min(1.5/\sqrt{d},\ 0.3)$ as an empirical error tolerance. A genome at embedding distance in

$$
\left[\theta(1-\varepsilon),\ \theta\right)
$$

is borderline covered and verified by a direct sketch comparison.

### OPH sketch Jaccard check

For each borderline-covered genome $G_i$:

1. Find the top-3 closest representatives by embedding dot product.
2. For each candidate representative $R_k$, compute single-sketch OPH Jaccard (sig1 only — sig2 was freed after Phase 2):

$$
J_{\mathrm{verify}} = J(\mathrm{sig1}_i,\ \mathrm{sig1}_{R_k})
$$

3. Convert: $d_{\mathrm{sketch}} = \arccos\!\left(\min(1, \max(0, J_{\mathrm{verify}}))\right) / \pi$
4. Promote $G_i$ to representative only if all checked representatives satisfy $d_{\mathrm{sketch}} \geq \theta$.

This uses OPH sketch Jaccard (with $m$ bins as configured for the taxon tier, typically 10,000), not exact ANI, with variance $J(1-J) / m_{\mathrm{real}}$.

---

## Phase 8: Universal OPH certification

Phase 7 (borderline verification) only checks genomes near the embedding coverage boundary. Nyström approximation error is not uniform; it is larger for sparse genomes (MAGs, incomplete assemblies) whose k-mer sets differ substantially from the anchor sample. A genome that appears covered in embedding space may be genuinely uncovered in OPH sketch space.

Phase 8 runs a universal coverage check over every non-representative genome.

**Certification thresholds.** Two thresholds are derived from the user ANI parameter $p = \mathrm{ANI}/100$ and k-mer size $k$:

$$
q_{\mathrm{cert}} = p^k, \qquad J_{\mathrm{cert}} = \frac{q_{\mathrm{cert}}}{2 - q_{\mathrm{cert}}}
$$

At 95% ANI with $k = 21$: $q_{\mathrm{cert}} = 0.95^{21} \approx 0.3406$, $J_{\mathrm{cert}} = q/(2-q) \approx 0.2052$.

$J_{\mathrm{cert}}$ is used for the symmetric Jaccard arm (equal-size genomes). $q_{\mathrm{cert}}$ is used directly for the containment arm (sparse genomes), where the genome size asymmetry means Jaccard underestimates similarity.

**Two-arm `oph_certified` function.** For a pair $(G_i, G_j)$, the function returns `true` if either arm passes:

- **Arm 1 (symmetric Jaccard):** $J(G_i, G_j) \geq J_{\mathrm{cert}}$ (sig1 only).
- **Arm 2 (directional containment):** triggered when $n_{\mathrm{real,small}} / n_{\mathrm{real,large}} \leq 0.85$ (one genome occupies ≤85% of the OPH bins of the other — catches most MAG vs. complete-genome pairs). Computes the containment fraction $C = n_{\mathrm{match}} / n_{\mathrm{real,small}}$ and requires $C \geq q_{\mathrm{cert}}$.

**Algorithm.** For each non-representative genome $G_i$ (excluding contamination-excluded genomes):

1. **Fast path**: run `oph_certified(G_i, R_{\mathrm{assigned}})` against the currently assigned representative. If it passes (either arm), $G_i$ is certified; continue.
2. **Exhaustive scan**: if the fast path fails, run `oph_certified(G_i, R_k)` against every representative $R_k$. Reassign $G_i$ to the representative with the highest $J_{\mathrm{dual}}$ among those that pass. Using either arm ensures that a MAG correctly covered by containment is not incorrectly sent to the repair queue.
3. **Repair queue**: if no representative passes either arm, $G_i$ is promoted to a new representative.

The outer loop is parallelised with a BS thread pool; each thread maintains a local repair queue merged after the barrier.

**Coverage guarantee.** After Phase 8, every non-representative genome satisfies `oph_certified` against at least one representative, independent of Nyström approximation error. For symmetric pairs this is a sketch-space Jaccard guarantee; for asymmetric pairs it is a directional containment guarantee. The remaining uncertainty is OPH estimation variance: at 95% ANI with $m = 10{,}000$ bins (typical tier), $\sigma_J \approx 0.004$ for dense assemblies; sparse genomes (low $m_{\mathrm{real}}$) have higher variance and looser sketch-space guarantees.

---

## Tiny-taxon fast path ($n \leq 20$)

For taxa with 20 or fewer genomes, the full Nyström embedding pipeline is skipped. A brute-force all-pairs OPH Jaccard cover is run instead:

- Compute all $\binom{n}{2}$ pairwise Jaccard values directly.
- Run a greedy cover: iteratively pick the genome that covers the most uncovered genomes within the ANI threshold, using the genome with higher quality score as tie-breaker.
- Any genome not within ANI threshold of any representative becomes its own representative.

For $n=1$, the single genome is trivially the representative.

---

## ANI from Jaccard

[ANI](https://en.wikipedia.org/wiki/Average_nucleotide_identity) is related to k-mer Jaccard via the [Mash formula](https://doi.org/10.1186/s13059-016-0997-x) (Ondov et al. 2016). Under a Poisson substitution model with per-base rate $r = 1 - p$ (ANI $= p \times 100\%$), the expected Jaccard for equal-size genomes is:

$$
J \approx \frac{p^k}{2 - p^k}, \qquad p = \frac{\mathrm{ANI}}{100}
$$

Inverting exactly:

$$
\mathrm{ANI} = \left(\frac{2J}{1 + J}\right)^{1/k} \times 100
$$

The model assumes equal genome sizes, i.i.d. substitutions, and negligible indels. It degrades for highly divergent sequences (ANI < 90%) where indels dominate.

The angular distance threshold corresponding to an ANI cutoff (before degree normalisation):

$$
\theta_{\mathrm{ANI}} = \frac{\arccos(J_{\mathrm{threshold}})}{\pi}
$$

After degree normalisation, dot products approximate a normalised-graph similarity rather than raw Jaccard. Phase 7 corrects borderline decisions back to sketch Jaccard space.

### K-mer size pre-probe

Before loading all $n$ sketches, geodesic runs a lightweight pre-probe (`probe_taxon_kmer`) to determine which stored k-mer size best matches the taxon's diversity:

1. Sample up to $\min(300,\ n/5)$ genomes at uniform stride.
2. Load their OPH sketches at the largest available k (e.g. $k=31$) using only 500 bins.
3. Compute all-pairs OPH Jaccard dissimilarity $d_{ij} = 1 - J_{ij}$ among the probe set; record each genome's nearest-neighbour distance $d_i^{\mathrm{nn}} = \min_{j \ne i} d_{ij}$.
4. Compute $p_5$ and $p_{95}$ of the NN distribution. Apply the selection rule:

| $p_{95}$ | selected k | taxon regime |
|----------|-----------|--------------|
| $< 0.002$ | 31 | clonal / near-identical strains |
| $< 0.010$ | 21 | moderate intra-species diversity |
| $\geq 0.010$ | 16 | diverse or cross-species taxa |

Safety: if $k = 16$ was selected but $p_5 < 0.010$ (mixed clonal + diverse), falls back to $k = 21$.

The thresholds follow from the OPH Jaccard ↔ ANI table above: at $k = 31$, $d = 0.010$ corresponds to $J = 0.990$, which via the Mash formula gives ANI $\approx 99.97\%$ — the boundary between clonal and moderately diverse taxa. At $k = 31$, $d = 0.002$ ($J = 0.998$) is the near-identical threshold ($\mathrm{ANI} \gtrsim 99.99\%$).

If the taxon has fewer than 20 genomes, or the archive stores only one k, the probe is skipped and the configured default ($k = 21$) is used.

After Phase 4 (Score), geodesic may also call `maybe_reselect_k` if the post-embedding $p_{95}$ nearest-neighbour distance warrants a different k, triggering a full re-embedding with the new k.

---

## Complexity

| Phase | Operation | Complexity |
|-------|-----------|------------|
| Sketch | OPH per genome (rolling hash) | $O(L)$ where $L$ = genome length |
| Embed anchors | Gram matrix $K$ | $O(p^2 m)$ |
| Embed all | Nyström extension (sig1 only) | $O(npm)$ |
| Eigendecomp | $K_{\mathrm{reg}}$ ($p \times p$) | $O(p^3)$ |
| Index | HNSW build | $O(nM \log n)$ |
| Score | kNN isolation ($k_{\mathrm{iso}}$) + adaptive DSU scan + MST (Kruskal, $k_{\mathrm{stable}} \leq K_{\mathrm{cap}}$) | $O(n \log n)$ |
| Select | FPS batched ($B=16$) | $O(nrd/B)$ where $r$ = number of reps |
| Merge | HNSW search + Union-Find | $O(r \log r)$ |
| Verify | OPH sketch Jaccard | $O(n_{\mathrm{borderline}} \cdot m)$ |
| Certify | Universal OPH coverage check | $O(nm)$ fast path; $O(n \cdot r \cdot m)$ worst case |

Typical values: $p \approx 512$, $m \in \{5{,}000,\, 10{,}000,\, 20{,}000\}$ (tier-dependent), $d \approx 64$–$512$. Embedding (Nyström extension) dominates for $n > 10{,}000$; FPS dominates for medium-size taxa.

---

## Implementation notes

- **SIMD**: AVX2 in the OPH inner loop (32 bytes/cycle), anchor-slab Gram matrix (`_mm256_cmpeq_epi16`), and FPS update loops.
- **Thread pool (BS::thread_pool)**: parallel Gram matrix rows, FPS fitness/update loops, HNSW isolation queries, Phase 8 certification. OPH sketching uses a separate producer-consumer pipeline.
- **GEODF**: results written to flat binary file; interrupted runs resume via GEODF crash recovery.
- **Anchor slab**: anchor signatures ($p \times m \times 2$ bytes, e.g. $512 \times 4{,}096 \times 2 = 4\ \mathrm{MB}$ at the medium tier or $512 \times 10{,}000 \times 2 \approx 10\ \mathrm{MB}$ at the typical tier) packed into a contiguous aligned buffer for cache-friendly Gram matrix computation.
- **Producer-consumer I/O**: genome decompression overlapped with k-mer computation via bounded semaphore.
- **Binary result writer**: taxon results are written to the GEODF flat binary (and the optional GRD archive for visualization) via positioned `pwrite` at computed offsets. There is no SQLite database or background batching thread in the tree.
- **NFS robustness**: the GEODF/GRD writers retry a failed `pwrite` on `ENOSPC`/`EIO` with exponential backoff — `retry_secs` starts at 5 s and doubles, capped at 300 s; any other error throws immediately.

---

## References

- Ondov et al. (2016) *Mash: fast genome and metagenome distance estimation using MinHash*. Genome Biology. [doi:10.1186/s13059-016-0997-x](https://doi.org/10.1186/s13059-016-0997-x)
- Li, P. & König, A.C. (2011) *b-Bit Minwise Hashing*. WWW 2011. [doi:10.1145/1989323.1989399](https://doi.org/10.1145/1989323.1989399)
- Li, P., Owen, A. & Zhang, C.H. (2012) *One Permutation Hashing*. NIPS. [link](https://proceedings.neurips.cc/paper_files/paper/2012/file/eaa32c96f620053cf442ad32258076b9-Paper.pdf)
- Williams, C.K.I. & Seeger, M. (2001) *Using the Nyström Method to Speed Up Kernel Machines*. NIPS. [link](https://proceedings.neurips.cc/paper/2000/file/19de10adbaa1b2ee13f77f679fa1483a-Paper.pdf)
- Gonzalez, T.F. (1985) *Clustering to minimize the maximum intercluster distance*. Theoretical Computer Science 38:293–306.
- Malkov, Y.A. & Yashunin, D.A. (2018) *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs*. IEEE TPAMI 42(4):824–836. [doi:10.1109/TPAMI.2018.2889473](https://doi.org/10.1109/TPAMI.2018.2889473)
- Chklovski, A., Parks, D.H., Woodcroft, B.J. & Tyson, G.W. (2023) *CheckM2: a rapid, scalable and accurate tool for assessing microbial genome quality using machine learning*. Nature Methods. [doi:10.1038/s41592-023-01940-w](https://doi.org/10.1038/s41592-023-01940-w)
