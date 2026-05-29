# Contamination detection

geodesic identifies potentially contaminated and chimeric assemblies before representative selection. Contaminated genomes receive a fitness score of zero, excluding them from being chosen as representatives while still assigning them to the nearest representative in the output.

---

## Why this matters

A contaminated or chimeric assembly contains sequence from multiple lineages and looks artificially "diverse": its k-mer composition reflects both parents, pushing it far from other taxon members in embedding space. If selected as a representative, reads from neither lineage map cleanly: species abundance is smeared across clades, lineage-specific loci lose coverage, and variant calls become unreliable. Worse, diversity-maximising selection actively favours such assemblies because they appear to fill underrepresented parts of sequence space. Contamination filtering protects the biological meaning of the representative set.

---

## Detection signals

Six per-genome signals are computed and stored in the outlier candidates list:

| Signal | Description |
|--------|-------------|
| `isolation_score` | Mean angular distance to the $k$ nearest neighbours ($k = \max(10,\, \min(20,\, \lfloor\log_2 n\rfloor))$); high = isolated = anomalous |
| `centroid_distance` | Angular distance from the species centroid (mean embedding vector, renormalised to unit length) |
| `genome_size_zscore` | Z-score of genome size relative to the taxon distribution |
| `kmer_div_zscore` | K-mer diversity z-score: occupied OPH bins per kbp relative to the taxon mean. Informational; see below. |
| `anomaly_score` | Currently equal to `isolation_score`; field reserved for future composite scoring |
| `nn_outlier` | Boolean flag: `isolation_score` exceeds the taxon threshold (primary exclusion criterion) |

---

## Flagging criterion

A genome is excluded from representative selection when `nn_outlier = TRUE`. The threshold is `max(per_component_thr, global_thr)` — both computed with the MAD estimator (50% breakdown point):

$$
\text{threshold} = \tilde{\mu} + z \cdot 1.4826 \cdot \text{MAD}, \qquad \text{MAD} = \text{median}(|x_i - \tilde{\mu}|)
$$

The **per-component** threshold uses only genomes in the same MST component; the **global** threshold uses all non-excluded genomes. Taking the max means a genome must exceed both its local and global distributions. $z$ is configurable via `--z-threshold` (default 2.0). When MAD=0 (all isolation scores identical in a component), the code falls back to IQR. Ordinary mean/SD are not used because contaminated genomes form a long right tail.

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
\text{fitness}_i = d_i \cdot \sqrt{\frac{L_i}{L_m}}
$$

where $d_i$ is the distance to the nearest current representative, $L_i$ is genome length, and $L_m$ is the taxon median genome length. The quality score $q_i$ acts as a **tie-breaker only** — when two candidates have equal fitness, the one with higher quality score is preferred. This keeps the primary selection signal (diversity) independent of quality noise.

A genome with 10% CheckM2 contamination loses 50 quality points, making it less likely to win ties. The embedding-based `nn_outlier` flag remains the primary contamination filter; CheckM2 quality only modulates selection among clean candidates.

---

## GUNC integration

[GUNC](https://doi.org/10.1186/s13059-021-02393-0) (Orakov et al. 2021) detects chimeric assemblies using phylogenetically diverse marker genes. A genome is chimeric if its marker genes span multiple clades inconsistently. GUNC is more reliable than k-mer-based approaches for subtle contamination.

Pass GUNC output with `--gunc-scores gunc_output.tsv`. Genomes with `pass.GUNC = False` are excluded from representative selection.

---

## Output

The `_outliers.tsv` file contains all flagged candidates with columns:

```
taxonomy  accession  category  nn_outlier  isolation_score  kmer_div_zscore  genome_size_zscore  centroid_distance  anomaly_score  genome_length_bp  n_contigs  margin_to_threshold  flag_reason  excluded
```

All genomes still appear in `_derep_genomes.tsv` assigned to their nearest representative; contamination detection only affects selection eligibility, not assignment.
