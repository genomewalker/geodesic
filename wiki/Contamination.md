# Contamination detection

geodesic identifies potentially contaminated and chimeric assemblies before representative selection. Contaminated genomes receive a fitness score of zero, excluding them from being chosen as representatives while still assigning them to the nearest representative in the output.

---

## Why this matters

A contaminated or chimeric assembly contains sequence from multiple lineages and looks artificially "diverse": its k-mer composition reflects both parents, pushing it far from other taxon members in embedding space. If selected as a representative, reads from neither lineage map cleanly: species abundance is smeared across clades, lineage-specific loci lose coverage, and variant calls become unreliable. Worse, diversity-maximising selection actively favours such assemblies because they appear to fill underrepresented parts of sequence space. Contamination filtering protects the biological meaning of the representative set.

---

## Detection signals

Six per-genome signals are computed and stored in the `contamination_candidates` table:

| Signal | Description |
|--------|-------------|
| `isolation_score` | Mean angular distance to the $k=10$ nearest neighbours in embedding space (high = isolated = anomalous) |
| `centroid_distance` | Angular distance from the species centroid (mean embedding vector, renormalised to unit length) |
| `genome_size_zscore` | Z-score of genome size relative to the taxon distribution |
| `kmer_div_zscore` | K-mer diversity z-score: occupied OPH bins per kbp relative to the taxon mean. Informational; see below. |
| `anomaly_score` | Currently equal to `isolation_score`; field reserved for future composite scoring |
| `nn_outlier` | Boolean flag: `isolation_score` exceeds the taxon threshold (primary exclusion criterion) |

---

## Flagging criterion

A genome is excluded from representative selection when `nn_outlier = TRUE`. The threshold is derived from a [Median Absolute Deviation (MAD)](https://en.wikipedia.org/wiki/Median_absolute_deviation)-based robust estimate of the per-component isolation score distribution. MAD has a breakdown point of 50%: up to half of genomes can be contaminated without biasing the estimator.

$$
\text{threshold} = \tilde{\mu} + z \cdot 1.4826 \cdot \mathrm{MAD}
$$

where $\tilde{\mu}$ is the component median isolation score, $\mathrm{MAD} = \text{median}(|x_i - \tilde{\mu}|)$, and $z$ is configurable via `--z-threshold` (default 2.0). The factor 1.4826 makes MAD consistent with standard deviation for normal distributions. Ordinary mean and SD are not used because contaminated genomes form a long right tail in the isolation score distribution; including them in the estimator inflates $\sigma$ and raises the threshold, masking the very outliers we want to detect.

Genomes with `isolation_score > threshold` have anomalously large mean distance to their nearest neighbours in embedding space, the primary signal of taxonomic misassignment or cross-species contamination. Their fitness is set to zero: they cannot be selected as representatives but remain in the output assigned to their nearest representative.

---

## K-mer diversity z-score

The `kmer_div_zscore` is a population-aware signal intended to detect chimeric assemblies. A chimeric assembly (two organisms stitched together) contains k-mers from both genomes, resulting in more occupied OPH bins per kilobase than any single-organism genome in the taxon.

For each genome $G_i$, let $r_i$ be occupied OPH bins per kbp:

$$
r_i = \frac{n_{\text{real},i}}{L_i / 1000}
$$

The z-score relative to the taxon distribution:

$$
z_i = \frac{r_i - \bar{r}}{s_r}
$$

This signal is computed and stored for analysis. It is not currently used as a flagging criterion.

---

## CheckM2 integration

When CheckM2 quality estimates are available (`--checkm2`), contamination enters directly through the fitness function:

$$
q = \text{completeness} - 5 \times \text{contamination}
$$

$$
\text{fitness}_i = d_i \cdot \frac{q_i}{100} \cdot \sqrt{\frac{L_i}{L_m}}
$$

where $d_i$ is the distance to the nearest current representative, $q_i$ is the CheckM2 quality score (defaults to 100 when unavailable), $L_i$ is genome length, and $L_m$ is the taxon median genome length.

A genome with 10% CheckM2 contamination loses 50 quality points. This typically suppresses contaminated genomes without fully excluding them; they remain in the candidate pool but rank far below clean assemblies. The embedding-based `nn_outlier` flag is the fallback when CheckM2 scores are unavailable.

---

## GUNC integration

[GUNC](https://doi.org/10.1186/s13059-021-02393-0) (Orakov et al. 2021) detects chimeric assemblies using phylogenetically diverse marker genes. A genome is chimeric if its marker genes span multiple clades inconsistently. GUNC is more reliable than k-mer-based approaches for subtle contamination.

Pass GUNC output with `--gunc-scores gunc_output.tsv`. Genomes with `pass.GUNC = False` are excluded from representative selection.

---

## Output

The `_contamination.tsv` file contains all flagged candidates with columns:

```
taxonomy  accession  nn_outlier  isolation_score  kmer_div_zscore  genome_size_zscore  centroid_distance  anomaly_score  genome_length_bp  n_contigs  margin_to_threshold  flag_reason
```

All genomes still appear in `_derep_genomes.tsv` assigned to their nearest representative; contamination detection only affects selection eligibility, not assignment.
