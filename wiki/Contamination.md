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
| `kmer_div_zscore` | K-mer diversity z-score: occupied OPH bins per kbp ($n_{\text{real}}/\text{kbp}$) relative to the taxon mean. Informational only — computed and stored, never used as a flagging criterion. A chimeric assembly tends to score high (k-mers from both parents inflate occupied bins per kbp). |
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
the archive construction pipeline. At 9.53 M genomes (GTDB r232) the QUAL section achieves
**100 % coverage with zero null values**.

### Completeness signals

| Column | Description |
|--------|-------------|
| `completeness_marker` | Present / expected single-copy genus markers (CheckM2-aligned, genus-calibrated); the **primary** intrinsic estimate, fraction-tracking (declines with fragmentation). NA without a `--markers` panel |
| `completeness_aamer_core` | AAMER genus prevalence-core completeness (amino-acid 8-mers); first fallback, presence-saturating near 1.0 |
| `completeness_post_decontam` | bp retained after the contig contamination scan; last-resort fallback |
| `completeness_cluster_relative` (TSV alias `pangenome_fraction`) | Occupied aamer content relative to the genus **pangenome** median — accessory breadth, **not** intrinsic completeness; admitted only as a soft corroborator |
| `completeness_effective` | The single value fed to tier/score: `marker → aamer_core → post_decontam`, with `cluster_relative` folded in only when the intrinsic signal also reads incomplete |

`completeness_cluster_relative` compares a genome's occupied-bin count against the median of
its GTDB species cluster. It reflects pangenome breadth relative to the cluster, not intrinsic
completeness: a finished isolate in a diverse genus reports a low value because the cluster
median carries more accessory content, not because sequence is missing. geodesic does not gate
on it directly — the intrinsic gate (`--min-completeness`, below) uses genus single-copy-core
recovery, and folds `completeness_cluster_relative` in only as a soft corroborator when the
core signal also reads incomplete.

### Contamination signals

Contamination is **reported, never a discard gate** (see [Quality tier](#quality-tier)).
The raw axes collapse into three near-independent **channels** consumed by dereplication as a
D → S → G tiebreak; the raw axes are emitted alongside for inspection.

| Column | Description |
|--------|-------------|
| `contam_D` | **Channel D** — calibrated single-copy-core duplication (`contamination_duplication` / `core_dup_mass` mapped to CheckM2 units); the only CheckM2-calibrated channel and the only contamination signal that touches the tier (caps HQ → MQ at ≥ 0.05) |
| `contam_S` | **Channel S** — `fmh_contamination`, FracMinHash k-mer minority mass |
| `contam_G` | **Channel G** — median of the present geometry axes (`contamination_rho_outlier`, `contamination_spe`, `contamination_contig_outlier_adj`, `contamination_tnf_minor`), collapsing the one correlated TNF/GCOV signal to a single vote |
| `contam_score` | Noisy-OR union of the present channels — **display only**, not a gate |
| `channels_fired` | Count of channels over threshold (geometry counts only when ≥ 2 of its four axes agree) |
| `contamination_contig_outlier[_adj]`, `contamination_spe`, `contamination_rho_outlier`, `contamination_tnf_minor` | Raw geometry axes behind channel G (Hotelling $T^2$/SPE outlier fraction, rank-correlation outlier, TNF-GMM minority mass) |
| `contamination_leakage`, `contamination_tnf_excess`, `contamination_cross_genus`, `contamination_contig_split` | Reported diagnostics only — **not** gates. leakage and tnf_excess are dropped from the channels (mathematically dead / untrusted); `cross_genus` is a ranker (2.8% PPV as a demotion rule), never a veto |

### Quality tier

The `quality_tier` (LQ / MQ / HQ) is **completeness-only** — contamination is decoupled
from it entirely (genopack commit `61e8a84`). A genome is LQ solely on genuine incompleteness
(`comp_eff < 0.50`); MQ up to 0.90; HQ at ≥ 0.90 unless the single CheckM2-calibrated
duplication channel caps it to MQ (`D ≥ 0.05`) or completeness rests only on the saturating
`aamer_core` fallback. `comp_eff` is the first available of `completeness_marker` →
`completeness_aamer_core` → `completeness_post_decontam`. The contamination channels (D/S/G,
below) are reported for dereplication's D → S → G tiebreak, never a discard. The exact rule
chain is in the genopack [Quality Scoring](https://genomewalker.github.io/genopack/quality/)
docs and not restated here to avoid drift.

### GTDB r232 results (9.53 M genomes)

| Tier | Count | % |
|------|------:|--:|
| HQ | 5,257,430 | 55.16 % |
| MQ | 3,737,697 | 39.22 % |
| LQ | 535,855 | 5.62 % |
| **Total** | **9,530,982** | **100 %** |

Zero genomes with missing quality data, and `completeness_marker` populated for ~99% of
genomes — every genome is routed through the marker stage when `--markers` is supplied
(genopack commit `ae70b1b`), rather than only the ~34% flagged by pass-A. The LQ floor is
genuine incompleteness only: decoupling contamination from the tier removed 362,935 wrongful
contamination demotions that had scored 6.9% PPV against CheckM2. genopack QUAL runs during
archive construction with no additional wall-clock cost.

---

## Integrating QUAL into dereplication

### `--skip-lq`

Excludes all genomes with `quality_tier = LQ` from FPS representative selection.
Genomes without a QUAL record pass through (the three-state rule: LQ → skip, HQ/MQ → keep,
unknown → keep).

### `--min-completeness FLOAT` (alias `--min-cr`)

Excludes genomes whose intrinsic completeness is below the given threshold (0–1), regardless
of tier. The intrinsic estimate is `completeness_effective`: single-copy marker recovery
(`completeness_marker`) first, then genus prevalence-core (`completeness_aamer_core`), then
post-decontam bp-retention (`completeness_post_decontam`); `completeness_cluster_relative`
corroborates only when the intrinsic signal also reads incomplete. This is genome completeness,
not pangenome breadth — it does not penalise finished isolates in diverse genera.

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
