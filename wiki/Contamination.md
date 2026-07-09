# Contamination detection

geodesic uses two complementary contamination tracks: an **embedding-based outlier detector** that
runs inside the dereplication pipeline, and the **genopack QUAL suite** — a battery of k-mer-derived
quality signals computed during archive construction that scales to tens of millions of genomes
without any marker-gene database.

The QUAL scoring methods (completeness, contamination aggregate, quality score, tiers) are
specified in the genopack [Quality Scoring](https://genomewalker.github.io/genopack/quality/) docs;
this page covers how geodesic consumes them plus its own embedding-based outlier track.

---

## Why contamination matters for representative selection

A contaminated or chimeric assembly contains sequence from multiple lineages and looks
artificially "diverse": its k-mer composition reflects both parents, pushing it far from other
taxon members in embedding space. If selected as a representative, reads from neither lineage
map cleanly — species abundance is smeared across clades, lineage-specific loci lose coverage,
and variant calls become unreliable. Diversity-maximising selection (FPS) actively *favours*
such assemblies because they appear to fill underrepresented parts of sequence space.
Contamination filtering protects the biological meaning of the representative set.

---

## Track 1: Embedding-based outlier detection (per-taxon, in-pipeline)

Six per-genome signals are computed for every taxon during the dereplication run:

| Signal | Description |
|--------|-------------|
| `isolation_score` | Mean angular distance to the $k$ nearest neighbours ($k = \max(10,\, \min(20,\, \lfloor\log_2 n\rfloor))$); high = isolated = anomalous |
| `centroid_distance` | Angular distance from the species centroid |
| `genome_size_zscore` | Z-score of genome size relative to taxon distribution |
| `kmer_div_zscore` | GUNC clade-separation score (CSS) when GUNC output is available (`--gunc-scores`); 0 otherwise. A chimeric assembly scores high CSS because its marker genes span multiple clades. The field name is a historical artefact — it does not store a k-mer density z-score. |
| `anomaly_score` | Currently equal to `isolation_score`; reserved for future composite scoring |
| `nn_outlier` | Boolean: `isolation_score` exceeds the taxon threshold — primary exclusion criterion |

A genome is excluded from representative selection when `nn_outlier = TRUE`. The threshold
uses the MAD estimator (50% breakdown point), taking the max of a per-MST-component and
global threshold:

$$
\text{threshold} = \tilde{\mu} + z \cdot 1.4826 \cdot \text{MAD}, \qquad \text{MAD} = \text{median}(|x_i - \tilde{\mu}|)
$$

$z$ is configurable via `--z-threshold` (default 2.0). When MAD = 0 (all isolation scores
identical in a component) the code falls back to IQR. Ordinary mean/SD are never used because
contaminated genomes form a long right tail.

Excluded genomes retain their cluster assignment in `_derep_genomes.tsv`; they are mapped to
the nearest clean representative rather than dropped entirely.

---

## Track 2: genopack QUAL suite (at-scale, marker-gene-free)

The genopack archive stores a QUAL section for every genome produced by `geodesic check`.
Unlike CheckM2 and GUNC — which require a marker-gene database and scale roughly linearly with
genome count — the QUAL signals are derived entirely from k-mer statistics and run as part of
the archive construction pipeline. At 9.3 M genomes (GTDB r232) the QUAL section achieves
**100 % coverage with zero null values**.

### Completeness signals

| Column | Description |
|--------|-------------|
| `genome_fill` | Total occupied FracMinHash bins / expected fill for a complete genome of this taxon's median size |
| `completeness_cluster_relative` | Occupied OPH bins relative to the taxon cluster median; 1.0 = median-sized genome for this taxon |
| `completeness_sketch_fill` | OPH bin occupancy fraction (k-mer sketch saturation) |
| `completeness_fragmentation` | Assembly fragmentation penalty: penalises high contig count relative to genome size |
| `completeness_post_decontam` | Effective completeness after removing contaminated k-mer windows |
| `completeness_aamer_core` | AAMER-based core k-mer completeness (genus-level conserved k-mers) |
| `completeness_aamer_family_core` | AAMER-based completeness using family-level conserved k-mers |

`completeness_cluster_relative` compares a genome's occupied-bin count against the median of
its GTDB species cluster. It reflects pangenome breadth relative to the cluster, not intrinsic
completeness: a finished isolate in a diverse genus reports a low value because the cluster
median carries more accessory content, not because sequence is missing. geodesic does not gate
on it directly — the intrinsic gate (`--min-completeness`, below) uses genus single-copy-core
recovery, and folds `completeness_cluster_relative` in only as a soft corroborator when the
core signal also reads incomplete.

`sketch_fill` measures OPH bin saturation — whether enough total sequence is present to fill
all bins. A genome with 1.5 Mb of repetitive or contaminant sequence can reach
`sketch_fill = 1.0` while being incomplete, so it is not a completeness gate either.

### Contamination signals

| Column | Description |
|--------|-------------|
| `fmh_contamination` | FracMinHash-based contamination fraction: k-mers mapping to out-of-taxon references |
| `contamination_leakage` | Cross-taxon k-mer leakage fraction in the OPH sketch; penalises genomes with k-mers pulled from a neighbouring taxon |
| `contamination_tnf_excess` | Tetranucleotide frequency excess: excess non-taxon TNF signal, flags chimeric assemblies |
| `contamination_contig_outlier` | Fraction of contigs whose k-mer profiles are outliers relative to the genome's own distribution |
| `contamination_contig_outlier_adj` | Contig outlier fraction adjusted for assembly size |
| `contamination_cross_genus` | k-mer signal from cross-genus contamination |
| `contamination_contig_split` | Fraction of contigs that appear split (two incompatible k-mer pools within a single contig) |
| `contamination_duplication` | Excess k-mer duplication relative to taxon expectation |
| `contamination_mixture` | Fraction of windows best explained by a two-genome mixture model |
| `contamination_spe` | Single-pass entropy contamination estimate |
| `contamination_rho_outlier` | Rank-correlation outlier score across the k-mer distribution |

### Genome-coherence signals

| Column | Description |
|--------|-------------|
| `chromosome_skew_closure` | GC/AT skew closure metric for complete/circular chromosomes |
| `chargaff_parity` | Deviation from Chargaff's second parity rule; violations flag inter-strand contamination |
| `self_coherence` | Internal k-mer self-consistency score |
| `spectral_gap` | Spectral gap in the k-mer hash density spectrum; large gaps indicate compositional discontinuities |
| `scale_kink` | Scale-space kink in the hash density spectrum; detects chimeric junctions |
| `leakage_residual` | Residual leakage signal after contamination removal |

### Quality tier

All signals feed into a single `quality_tier` (LQ / MQ / HQ) and a MIMAG-compatible
`mimag_tier` output. The composite score used for the tier:

$$
q = c_{\text{eff}} \times 100 - 5 \times \text{contamination\_leakage} \times 100
$$

where $c_{\text{eff}} = \text{completeness\_post\_decontam}$ when not NaN, else
`completeness_cluster_relative`.

### GTDB r232 results (9.3 M genomes)

| Tier | Count | % |
|------|------:|--:|
| HQ | 4,522,037 | 48.7 % |
| MQ | 4,544,766 | 49.0 % |
| LQ | 210,790 | 2.3 % |
| **Total** | **9,277,593** | **100 %** |

Zero genomes with missing quality data. CheckM2 on 1,067 genomes took 26 min on 48 cores;
extrapolated to 9.3 M that is ~14,500 CPU-hours. genopack QUAL runs during archive construction
with no additional wall-clock cost.

---

## Integrating QUAL into dereplication

### `--skip-lq`

Excludes all genomes with `quality_tier = LQ` from FPS representative selection.
Genomes without a QUAL record pass through (the three-state rule: LQ → skip, HQ/MQ → keep,
unknown → keep).

### `--min-completeness FLOAT` (alias `--min-cr`)

Excludes genomes whose intrinsic completeness is below the given threshold (0–1), regardless
of tier. The intrinsic estimate is genus single-copy/prevalence-core recovery
(`completeness_aamer_core`), with post-decontam bp-retention (`completeness_post_decontam`) as
fallback; `completeness_cluster_relative` corroborates only when the core signal also reads
incomplete. This is genome completeness, not pangenome breadth — it does not penalise finished
isolates in diverse genera.

```bash
geodesic derep \
    --skip-lq \
    --min-completeness 0.5 \
    ...
```

Use it to gate MQ genomes that are biologically incomplete but pass `--skip-lq` because their
k-mers are taxonomically clean. Genomes with no QUAL record are never excluded. `--min-cr` is a
deprecated alias for the same gate.

---

## Optional integrations: CheckM2 and GUNC

These external tools provide independent quality estimates and can be used alongside — not
instead of — the QUAL-based gates above.

### CheckM2

When CheckM2 quality estimates are available (`--checkm2`), quality enters the fitness function
as a bounded factor $\hat{q} = \mathrm{clamp}(q/100,\ 0,\ 1)$, with
$q = \text{completeness} - 5 \times \text{contamination}$:

$$
\text{fitness}_i = d_i \cdot \sqrt{L_i / L_m} \cdot (0.5 + 0.5\,\hat{q}_i)
$$

The factor lies in $[0.5, 1.0]$; coverage and the stopping test use raw similarity, so it
shifts which genome is chosen for a region toward higher quality without changing the number of
representatives or their ANI spread. A genome with 10 % CheckM2 contamination loses 50 quality
points and is deprioritised.

### GUNC

Pass GUNC output with `--gunc-scores gunc_output.tsv`. Genomes with `pass.GUNC = False` are
excluded from representative selection. GUNC is more reliable than k-mer methods for subtle
contamination involving phylogenetically diverse marker gene sets. Practical limitation: GUNC
does not scale to millions of genomes; use it for targeted validation of representative sets.

---

## Output

`_outliers.tsv` contains all flagged candidates:

```
taxonomy  accession  category  nn_outlier  isolation_score  kmer_div_zscore
genome_size_zscore  centroid_distance  anomaly_score  genome_length_bp
n_contigs  margin_to_threshold  flag_reason  excluded
```

All genomes still appear in `_derep_genomes.tsv` assigned to their nearest representative;
contamination detection only affects selection eligibility, not assignment.
