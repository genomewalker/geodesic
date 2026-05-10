#pragma once
#include "core/types.hpp"
namespace BS { class thread_pool; }
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <future>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace derep { struct IPackReader; }

namespace derep {

// Aligned allocator for SIMD-friendly memory layout
template <typename T, size_t Alignment = 64>
struct AlignedAllocator {
    using value_type = T;

    T* allocate(std::size_t n) {
        void* p = nullptr;
        if (posix_memalign(&p, Alignment, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(p);
    }

    void deallocate(T* p, std::size_t) noexcept {
        free(p);
    }

    template <typename U>
    struct rebind { using other = AlignedAllocator<U, Alignment>; };
};

// Lightweight non-owning view over a contiguous array — replaces per-genome
// std::vector members in GenomeEmbedding to eliminate thousands of small heap
// allocations that glibc retains in the brk arena after free.
template<typename T>
struct SpanView {
    const T* ptr = nullptr;
    uint32_t n   = 0;
    bool     empty()                    const noexcept { return n == 0 || ptr == nullptr; }
    uint32_t size()                     const noexcept { return n; }
    const T* data()                     const noexcept { return ptr; }
    const T* begin()                    const noexcept { return ptr; }
    const T* end()                      const noexcept { return ptr + n; }
    const T& operator[](size_t i)       const noexcept { return ptr[i]; }
};

// Structure-of-Arrays (SoA) embedding storage for cache-friendly SIMD access.
// Also owns the flat sketch arrays (sigs_flat / sigs2_flat / masks_flat) so that
// sketch data for an entire taxon lives in ONE large mmap'd allocation — returned
// to the OS immediately when the GeodesicDerep object is destroyed, instead of
// leaving thousands of 20 KB heap fragments that glibc never reclaims.
struct SoAStore {
    size_t n = 0;
    size_t dim = 512;
    std::vector<float, AlignedAllocator<float, 64>> data;  // size n*dim, row-major

    // Metadata arrays (SoA)
    std::vector<uint64_t> genome_ids;
    std::vector<float> isolation_scores;
    std::vector<float> quality_scores;
    std::vector<uint64_t> genome_sizes;
    std::vector<std::string> accessions;

    // Flat sketch storage — one large allocation per taxon (replaces per-genome vectors).
    uint32_t sketch_size_flat = 0;
    uint32_t mask_words_flat  = 0;
    std::vector<uint16_t> sigs_flat;   // n × sketch_size_flat, row-major
    std::vector<uint16_t> sigs2_flat;  // n × sketch_size_flat, row-major
    std::vector<uint64_t> masks_flat;  // n × mask_words_flat,  row-major

    void resize(size_t count, size_t dimension) {
        n = count;
        dim = dimension;
        data.resize(n * dim);
        genome_ids.resize(n);
        isolation_scores.resize(n);
        quality_scores.resize(n);
        genome_sizes.resize(n);
        accessions.resize(n);
    }

    void init_sketches(uint32_t sk, uint32_t mw) {
        sketch_size_flat = sk;
        mask_words_flat  = mw;
        sigs_flat.assign(n * sk, 0xFFFFu);
        sigs2_flat.assign(n * sk, 0xFFFFu);
        masks_flat.assign(n * mw, 0u);
    }

    // Compact flat sketch rows to match a valid[] mask, then shrink_to_fit.
    void compact_sketches(const std::vector<uint8_t>& valid, size_t new_n) {
        size_t dst = 0;
        for (size_t src = 0; src < n; ++src) {
            if (!valid[src]) continue;
            if (dst != src) {
                std::copy_n(sigs_flat.data()  + src * sketch_size_flat, sketch_size_flat,
                            sigs_flat.data()  + dst * sketch_size_flat);
                std::copy_n(sigs2_flat.data() + src * sketch_size_flat, sketch_size_flat,
                            sigs2_flat.data() + dst * sketch_size_flat);
                std::copy_n(masks_flat.data() + src * mask_words_flat,  mask_words_flat,
                            masks_flat.data() + dst * mask_words_flat);
            }
            ++dst;
        }
        sigs_flat.resize(new_n * sketch_size_flat);  sigs_flat.shrink_to_fit();
        sigs2_flat.resize(new_n * sketch_size_flat); sigs2_flat.shrink_to_fit();
        masks_flat.resize(new_n * mask_words_flat);  masks_flat.shrink_to_fit();
    }

    bool has_sketch_data() const noexcept { return sketch_size_flat > 0 && !sigs_flat.empty(); }

          uint16_t* sig(size_t i)        noexcept { return sigs_flat.data()  + i * sketch_size_flat; }
    const uint16_t* sig(size_t i)  const noexcept { return sigs_flat.data()  + i * sketch_size_flat; }
          uint16_t* sig2(size_t i)       noexcept { return sigs2_flat.data() + i * sketch_size_flat; }
    const uint16_t* sig2(size_t i) const noexcept { return sigs2_flat.data() + i * sketch_size_flat; }
          uint64_t* mask(size_t i)       noexcept { return masks_flat.data() + i * mask_words_flat; }
    const uint64_t* mask(size_t i) const noexcept { return masks_flat.data() + i * mask_words_flat; }

    SpanView<uint16_t> sig_span(size_t i)  const noexcept { return {sig(i),  sketch_size_flat}; }
    SpanView<uint16_t> sig2_span(size_t i) const noexcept { return {sig2(i), sketch_size_flat}; }
    SpanView<uint64_t> mask_span(size_t i) const noexcept { return {mask(i), mask_words_flat}; }

    // Expand or shrink embedding dimension in-place (copies existing data row-by-row).
    void resize_dim(size_t new_dim) {
        if (new_dim == dim) return;
        std::vector<float, AlignedAllocator<float, 64>> new_data(n * new_dim, 0.0f);
        const size_t copy_dim = std::min(dim, new_dim);
        for (size_t i = 0; i < n; ++i)
            std::copy(data.data() + i * dim,
                      data.data() + i * dim + copy_dim,
                      new_data.data() + i * new_dim);
        data = std::move(new_data);
        dim = new_dim;
    }

    float* row(size_t i) { return data.data() + i * dim; }
    const float* row(size_t i) const { return data.data() + i * dim; }
};

// Genome embedding on unit sphere (dimension configurable)
struct GenomeEmbedding {
    uint64_t genome_id;
    std::vector<float> vector;        // CountSketch unit vector (dim=256)
    // oph_sig / oph_sig2 / real_bins_mask live in SoAStore flat arrays (not here).
    float isolation_score;            // Mean distance to k nearest neighbors
    float quality_score;              // completeness - 5*contamination (0-100)
    uint64_t genome_size;
    std::string accession;            // genopack archive accession (unique id)
    uint32_t n_real_bins = 0;         // non-empty OPH bins before densification
    uint32_t n_contigs = 0;           // Number of sequences (FASTA '>' headers)
};

// Calibration model: embedding_distance → [ANI_lower, ANI_upper]
// Uses monotonic quantile regression with conformal safety margins
class ANICalibrator {
public:
    struct Bounds {
        double lower;  // Conservative lower bound on ANI
        double upper;  // Conservative upper bound on ANI
    };

    // Fit on (embedding_distance, true_ANI) pairs
    void fit(const std::vector<std::pair<double, double>>& samples);

    // Predict ANI bounds for given embedding distance
    Bounds predict(double embedding_distance) const;

    // Inverse: find distance threshold where upper bound = target ANI
    double inverse_upper(double target_ani) const;

    // Inverse: find distance threshold where lower bound = target ANI
    double inverse_lower(double target_ani) const;

    // Coverage probability guarantee
    double coverage_probability() const { return coverage_prob_; }

    bool is_fitted() const { return fitted_; }

private:
    bool fitted_ = false;
    double coverage_prob_ = 0.95;

    // Monotonic quantile curves (distance → ANI bounds)
    std::vector<double> distance_grid_;
    std::vector<double> ani_lower_curve_;
    std::vector<double> ani_upper_curve_;

    // Safety margins from conformal calibration
    double lower_margin_ = 0.02;
    double upper_margin_ = 0.02;
};

// SIMD dot product (AVX2) - declared here, defined in cpp
float dot_product_simd(const float* a, const float* b, size_t dim);

// Cosine similarity using SIMD (no acos - faster for comparisons)
inline float cosine_similarity_simd(const float* a, const float* b, size_t dim) {
    return dot_product_simd(a, b, dim);  // Vectors are normalized
}

// GEODESIC: Genome Embedding + On-Demand Edge Synthesis with Indexed Clustering
// A physics-inspired approach to genome dereplication
class GeodesicDerep {
public:
    struct Config {
        // Embedding parameters (tuned for high-ANI accuracy)
        int embedding_dim = 256;     // Higher dim preserves more sketch information
        int sketch_size = 10000;     // Large sketch for accurate Jaccard at high ANI
        int kmer_size = 21;          // Larger k is more discriminative at >95% ANI
        int syncmer_s = 0;           // 0 = disabled, >0 = open-syncmer OPH prefilter

        // ANI threshold for redundancy
        double ani_threshold = 0.95;

        // HNSW index parameters
        int hnsw_m = 48;             // Higher M for better recall
        int hnsw_ef_construction = 400;
        int hnsw_ef_search = 200;

        // Parallelism
        int threads = 4;
        BS::thread_pool* pool = nullptr;  // if non-null, used instead of OMP
        // Max concurrent NFS file readers during genome embedding.
        // 0 = auto: threads (total budget for this taxon caps NFS readers).
        int io_threads = 0;

        // Calibration
        int calibration_samples = 500;

        // Isolation score
        int isolation_k = 10;  // k nearest neighbors for isolation
        int k_cap_max = 256;   // Max K_cap for adaptive retry on disconnected k-NN

        // FPS stopping criteria (derived from learned embedding↔ANI model)
        float diversity_threshold = 0.02f;    // Stop when diversity gain < this
        float min_rep_distance = 0.025f;      // Min distance between reps (electrostatic merge threshold)
        float max_rep_fraction = 0.2f;        // At most this fraction as reps

        // Nyström spectral embedding (always active for n > SMALL_N_THRESHOLD)
        // -1  = auto: n_anchors = min(n, max(200, 2 * embedding_dim))
        // >0  = explicit anchor count
        int nystrom_anchors = -1;

        // Target fraction of Gram matrix variance captured by the embedding.
        // Auto-selects embedding dimension d as minimum to explain >= this fraction.
        float nystrom_min_variance = 0.95f;

        // Tikhonov regularization (fraction of mean diagonal)
        float nystrom_diagonal_loading = 0.01f;
        // Symmetric Laplacian normalization of Gram matrix
        bool nystrom_degree_normalize = true;

        // Internal: set by apply_nystrom_embeddings(). After L2 normalisation,
        // dot(e_A,e_B) ≈ J(A,B)/captured_variance.
        float nystrom_captured_variance = 1.0f;

        // Master RNG seed. All sub-seeds are derived from this:
        //   sig1 (OPH sketch):          seed
        //   sig2 (OPH sketch, sig2≠sig1): seed + 1
        //   HNSW construction:          seed
        //   Nyström anchor sampling:    seed
        //   diversity pair sampling:    seed
        uint64_t seed = 42;
    };

    explicit GeodesicDerep(Config cfg);
    ~GeodesicDerep();

    // GPK sketch-accelerated build: load pre-computed OPH sketches (sig + sig2) from
    // the SKCH section of a V4 genopack archive. Zero decompression, zero re-sketching.
    // quality_scores: accession → quality (completeness - 5*contamination)
    void build_index_from_gpk_sketches(
        const std::vector<std::string>& accessions,
        IPackReader& gpk,
        const std::unordered_map<std::string, double>& quality_scores = {});

    // Get representative genome IDs (after select_representatives)
    std::vector<uint64_t> get_representative_ids() const { return last_representative_ids_; }

    // Exclude accessions from being selected as representatives (sets quality score to 0).
    // Call after build_index_from_gpk_sketches and detect_outlier_candidates,
    // before select_representatives.
    void exclude_from_reps(const std::unordered_set<std::string>& accessions);

    // Compute ad-hoc quality scores for genomes without CheckM2 data.
    // Uses centrality (inverse isolation) and kmer density as proxy for assembly quality.
    // Call after compute_isolation_scores(), before select_representatives.
    void compute_adhoc_quality_scores();

    // Pre-seed accessions as representatives before FPS runs.
    // Call after build_index_from_gpk_sketches and before select_representatives.
    void set_pinned_representatives(const std::unordered_set<std::string>& accessions);

    // Phase 4: Select representatives with lazy certified ANI
    std::vector<SimilarityEdge> select_representatives();

    // Optional: invoked once at the start of the serial cert_reps pruning phase.
    // Pipeline uses this to release most of the taxon's thread budget back to the
    // scheduler while the serial loop runs — boosts aggregate occupancy without
    // blocking correctness (the current taxon continues on its own thread).
    void set_on_serial_phase(std::function<void()> cb) { on_serial_phase_ = std::move(cb); }

    // Get all embeddings
    const std::vector<GenomeEmbedding>& embeddings() const { return embeddings_; }

    // Get per-genome component IDs (set by compute_isolation_scores)
    const std::vector<int>& component_ids() const { return component_ids_; }

    // Exact Jaccard from OPH signatures with b-bit bias correction.
    // Works for both uint16_t (stored) and uint32_t (in-memory OPH path).
    template<typename T>
    static double refine_jaccard(const std::vector<T>& sig_a, const std::vector<T>& sig_b) {
        if (sig_a.empty() || sig_b.empty()) return 0.0;
        const size_t m = std::min(sig_a.size(), sig_b.size());
        if (m == 0) return 0.0;
        size_t matches = 0;
        for (size_t t = 0; t < m; ++t)
            if (sig_a[t] == sig_b[t]) ++matches;
        const double j_raw = static_cast<double>(matches) / static_cast<double>(m);
        if constexpr (sizeof(T) <= 2) {
            // b-bit bias correction: J_true ≈ (J_obs - 2^-b) / (1 - 2^-b), b=16
            constexpr double inv_2b = 1.0 / 65536.0;
            return std::max(0.0, (j_raw - inv_2b) / (1.0 - inv_2b));
        }
        return j_raw;
    }

    // ANI from exact Jaccard via Mash formula (calibration-free)
    static double jaccard_to_ani(double J, int kmer_size);

    // Unified per-pair ANI fraction [0,1].
    // Uses Jaccard ANI when both sketches are dense (≥70% fill) and similarly sized
    // (fill ratio ≥0.7); otherwise uses containment ANI from sparser to denser.
    // J_hint: pre-computed Jaccard estimate for the dense path (<0 = re-scan raw bins).
    // 0xFFFF bins are densification sentinels (not real k-mer evidence).
    static float score_pair(const uint16_t* sa, const uint16_t* sb, uint32_t S,
                            uint32_t nr_a, uint32_t nr_b, int k, double J_hint = -1.0) {
        const double dense_thr = 0.7 * static_cast<double>(S);
        const bool both_dense = static_cast<double>(nr_a) >= dense_thr &&
                                static_cast<double>(nr_b) >= dense_thr;
        const bool similar_density = both_dense && nr_a > 0 && nr_b > 0 &&
            static_cast<double>(std::min(nr_a, nr_b)) >=
            0.7 * static_cast<double>(std::max(nr_a, nr_b));

        if (similar_density) {
            double J = J_hint;
            if (J < 0.0) {
                size_t matches = 0, total = 0;
                for (uint32_t t = 0; t < S; ++t) {
                    if (sa[t] == 0xFFFFu && sb[t] == 0xFFFFu) continue;
                    ++total;
                    if (sa[t] == sb[t]) ++matches;
                }
                J = total > 0 ? static_cast<double>(matches) / total : 0.0;
            }
            if (J <= 0.0) return 0.0f;
            return static_cast<float>(std::pow(2.0 * J / (1.0 + J), 1.0 / k));
        }

        // Containment: orient from sparser (smaller nr) to denser
        const uint16_t* sq = (nr_a <= nr_b) ? sa : sb;
        const uint16_t* sr = (nr_a <= nr_b) ? sb : sa;
        const uint32_t  nr_q = std::min(nr_a, nr_b);
        if (nr_q == 0) {
            if (J_hint > 0.0)
                return static_cast<float>(std::pow(2.0 * J_hint / (1.0 + J_hint), 1.0 / k));
            return 0.0f;
        }
        size_t matches = 0;
        for (uint32_t t = 0; t < S; ++t)
            if (sq[t] != 0xFFFFu && sq[t] == sr[t]) ++matches;
        double c = static_cast<double>(matches) / static_cast<double>(nr_q);
        return static_cast<float>(std::pow(std::clamp(c, 0.0, 1.0), 1.0 / k));
    }

    // Contamination detection: returns genome IDs with anomalous embedding patterns
    struct OutlierCandidate {
        uint64_t genome_id;
        float centroid_distance;    // Distance from species centroid (informational)
        float isolation_score;      // Mean distance to k-NN
        float anomaly_score;        // isolation_score (repurposed field)
        float genome_size_zscore;   // Z-score of genome size within taxon
        bool nn_outlier;            // isolation_score > 90% ANI threshold (primary: misassigned)
        float kmer_div_zscore = 0.0f; // k-mer diversity z-score (n_real_bins/kbp vs population; informational)
        float margin_to_threshold = 0.0f; // isolation_score - nn_threshold (positive = above threshold)
        std::string flag_reason;    // "nn_outlier", "size_outlier", or "nn_outlier+size_outlier"
        std::string accession;      // genopack archive accession
        uint32_t n_contigs = 0;
        uint64_t genome_length_bp = 0;
        bool excluded = true;   // false = flagged only, still participates in rep selection
    };
    std::vector<OutlierCandidate> detect_outlier_candidates(
        float z_threshold = 2.0f);

    // NN distance distribution from HNSW.
    struct NNDistStats {
        double p5;
        double p50;
        double p95;
        // Bridge-size-conditioned trimmed maximum MST edge weight.
        // Excludes outlier bridges (min component side ≤ ceil(sqrt(n))) that connect
        // singleton/pair outliers to the main mass. Falls back to true max when all
        // edges have tiny bridge sides (very small taxa). Zero if unavailable.
        double mst_max_edge = 0.0;
        double mst_true_max = 0.0;       // Actual maximum MST edge (for tail_ratio diagnostic)
        double mst_w2 = 0.0;             // second-largest MST edge (penultimate Kruskal merge)
        double tail_ratio = 0.0;         // mst_true_max / mst_max_edge; >2 signals heavy-tail suppression
        uint32_t bridge_min_side = 0;    // smaller component at the final MST merge
        int k_conn   = -1;  // smallest k where k-NN graph connects (-1 = never within K_cap)
        int k_stable = -1;  // k chosen by bottleneck stability probe (smallest k within 3% of B(K_cap))
        int k_cap    = 0;   // K_cap used as HNSW query budget
        // Instability flags: when set, mst_max_edge may not reflect intra-species scale.
        bool low_pair_count        = false;  // < 20 non-outlier genomes in MST
        bool pathological_bridge   = false;  // tiny-side AND isolated terminal merge
        bool disconnected_mst      = false;  // k-NN graph has > 1 component at K_cap
        bool nystrom_taxon_applied = false;  // per-taxon Nyström re-embedding succeeded
    };

    // Phase 3: Compute isolation scores AND return NN distance stats in one HNSW pass.
    // Replaces the old void compute_isolation_scores() + separate compute_nn_distance_stats().
    NNDistStats compute_isolation_scores();

    // Update thresholds after build_index_from_gpk_sketches (allows data-driven calibration).
    void set_min_rep_distance(float d) { cfg_.min_rep_distance = d; }
    void set_diversity_threshold(float d) { cfg_.diversity_threshold = d; }
    void set_nystrom_applied(bool v) { nystrom_applied_ = v; }
    float nystrom_scaled_j_floor() const { return nystrom_scaled_j_floor_; }
    bool nystrom_oph_sphere_applied() const { return nystrom_oph_sphere_applied_; }

    // Returns (accession, genome_length_bp) for all embedded genomes after
    // build_index_from_gpk_sketches. Used to persist genome_length to the DB.
    std::vector<std::pair<std::string, uint64_t>> get_genome_sizes() const;

    // Adaptive k-selection: pick the k that best matches this taxon's NN-distance diversity.
    // If the GPK has that k and it differs from current cfg_.kmer_size, re-embeds everything
    // and returns true. Otherwise returns false (caller reuses existing embeddings/HNSW).
    bool maybe_reselect_k(const NNDistStats& stats,
                          const std::unordered_map<std::string, double>& quality_scores);

    // Select best k based on P95 NN distance (clonal→31, moderate→21, diverse→16).
    static int select_best_k_for_diversity(float p95_nn_dist);

    // Genomes that were absent or unreadable during build_index_from_gpk_sketches.
    // Each entry: (accession, error_reason). Caller should record these in jobs_failed.
    const std::vector<std::pair<std::string, std::string>>& failed_reads() const {
        return failed_reads_;
    }

    // Diversity statistics computed from embeddings (no skani needed)
    struct DiversityMetrics {
        // Coverage: embedding distance from each genome to nearest representative
        double coverage_mean_dist = 0.0;
        double coverage_p5_dist  = 0.0;  // 5th percentile (best-covered genomes)
        double coverage_p95_dist = 0.0;  // 95th percentile (robust worst-case; ignores top 5% outliers)
        int coverage_below_99 = 0;  // Estimated ANI < 99%
        int coverage_below_98 = 0;
        int coverage_below_97 = 0;
        int coverage_below_95 = 0;

        // Diversity: pairwise embedding distance among representatives
        double diversity_mean_dist = 0.0;
        double diversity_p5_dist  = 0.0;
        double diversity_p95_dist = 0.0;
        int diversity_n_pairs = 0;
    };

    // Compute diversity metrics from embeddings (uses calibrated distance→ANI model)
    DiversityMetrics compute_diversity_metrics(
        const std::vector<uint64_t>& representative_ids) const;

private:
    std::function<void()> on_serial_phase_;
    Config cfg_;
    int runtime_dim_ = 0;  // Actual embedding dim after Nystrom (may differ from cfg_.embedding_dim)
    std::vector<GenomeEmbedding> embeddings_;
    SoAStore store_;  // SoA layout for SIMD-friendly access
    ANICalibrator calibrator_;

    // (Projection matrix removed: now uses OPH + CountSketch)

    // HNSW index (forward declaration to avoid header dependency)
    struct HNSWIndex;
    std::unique_ptr<HNSWIndex> index_;

    // Per-genome component label from K_cap Kruskal (-1 = MST outlier).
    // Set by compute_isolation_scores(), updated by detect_outlier_candidates()
    // when a genuine bimodal taxon is detected (Otsu split on isolation scores).
    std::vector<int> component_ids_;

    // Smaller side of the final MST merge (bridge_min_side from NNDistStats).
    // Stored as member so detect_outlier_candidates() can access it.
    uint32_t bridge_min_side_ = 0;

    // True when detect_outlier_candidates() detected a genuine bimodal taxon
    // (bridge_min_side_/n > 5%). Signals FPS to reset max_sim_to_rep to
    // same-cluster-only values after pre-seeding, preventing cross-cluster
    // coverage from suppressing within-cluster FPS diversity selection.
    bool genuine_bimodal_ = false;

    // Last selected representatives (for incremental workflows)
    std::vector<uint64_t> last_representative_ids_;

    // Pinned representative paths (pre-seeded before FPS)
    std::unordered_set<std::string> pinned_rep_paths_;

    // Whether Nyström embedding was applied (false → exact Jaccard FPS for small n)
    bool nystrom_applied_ = false;
    bool nystrom_taxon_applied_ = false;  // set by apply_nystrom_taxon on success
    bool nystrom_multicomp_applied_ = false;  // set by apply_nystrom_multicomp on success
    float nystrom_multicomp_s_max_ = 0.0f;   // max bridge Jaccard used in multi-component embedding
    bool nystrom_percomp_applied_ = false;    // set by apply_nystrom_percomp (ANN recall gap path)
    float nystrom_scaled_j_floor_ = 0.0f;    // > 0 when kernel-scaled Nyström active (ANN recall gap)
    bool nystrom_oph_sphere_applied_ = false; // OPH token sphere used (ANN recall gap path)

    // Set by maybe_reselect_k before calling build_index_from_gpk_sketches to
    // suppress the internal k pre-probe (which would otherwise override the
    // embedding-space k selection with an OPH-space probe result).
    bool kmer_size_locked_ = false;

    // Canonical lookup: genome_id → row index in embeddings_/store_.
    // Rebuilt after every sort of embeddings_ so that genome_ids (which are opaque
    // identifiers after sorting) are never used as direct array indices.
    std::unordered_map<uint64_t, size_t> gid_to_row_;

    // Genomes that were absent or unreadable during build: (accession, reason).
    // Written from OMP threads (mutex-protected); read by taxon_processor.
    std::vector<std::pair<std::string, std::string>> failed_reads_;
    std::mutex failed_reads_mutex_;

    // Pack reader retained for adaptive k re-embedding via maybe_reselect_k().
    // Set by build_index_from_gpk_sketches(), null otherwise.
    IPackReader* gpk_reader_ = nullptr;
    std::vector<std::string> gpk_accessions_;  // index-parallel to embeddings_

    // Nyström spectral embedding: replace placeholder vectors with data-adapted
    // projections onto the top eigenvectors of the OPH Jaccard kernel.
    void apply_nystrom_embeddings();

    // Re-run Nyström for a subset of rows (bridge-connected components) with bridge
    // endpoints as pinned anchors. Called after bridge detection when HNSW is
    // disconnected. Writes to embeddings_[row].vector and store_.row(row) in-place.
    // Returns true on success; false if numerically degenerate (caller keeps global coords).
    bool apply_nystrom_taxon(const std::vector<uint32_t>& taxon_rows,
                             const std::vector<uint32_t>& forced_anchor_rows,
                             float j_floor = 0.0f);

    // Per-component local Nyström stitched via bridge Jaccards into a global unit-sphere
    // embedding ("multi-component"). comp_labels[li] is the HNSW component label for taxon_ei[li].
    // Returns false (→ caller falls back to apply_nystrom_taxon) when s_max >= cos_diversity
    // (ANN recall gap: components are within diversity distance, exact Jaccard FPS is correct).
    bool apply_nystrom_multicomp(const std::vector<uint32_t>& taxon_ei,
                                    const std::vector<uint32_t>& forced_ei,
                                    const std::vector<std::pair<uint32_t,uint32_t>>& bridge_pairs_ei,
                                    const std::vector<float>& bridge_jaccards,
                                    const std::vector<int>& comp_labels);

    // ANN recall gap path: per-component independent Nyström (no planet block).
    // Component c occupies dims [c*d_per, (c+1)*d_per]; cross-component dot = 0
    // so FPS runs independently within each component. Phase 3 uses exact Jaccard
    // (nystrom_applied_ reset after FPS) to prune cross-component redundancy.
    bool apply_nystrom_percomp(const std::vector<uint32_t>& taxon_ei,
                               const std::vector<uint32_t>& forced_ei,
                               const std::vector<int>& comp_labels);

    // ANN recall gap fallback: direct OPH token sphere embedding.
    // φ(x)[b] = (1/√m) Σ_t sign(hash(t, x_t, b)).  E[⟨φ(x),φ(y)⟩] = Jaccard(x,y),
    // approximation error σ ≈ √(K(1-K)/m) ≈ 0.005 for m=10000.
    // Sets nystrom_oph_sphere_applied_=true; cos_diversity threshold unchanged.
    bool apply_oph_sphere(const std::vector<uint32_t>& taxon_ei);

    // SoA copy, Nyström embedding, HNSW build. Called from build_index_from_gpk_sketches.
    void finalize_embeddings_();

    // Brute-force O(n²) isolation scores for small n (no HNSW needed)
    void compute_isolation_scores_brute();

    // Sample up to ~300 genomes, compute pairwise Jaccard on 500 bins, return
    // select_best_k_for_diversity(p95_nn).  Returns 0 if probe inconclusive.
    int probe_kmer_size_(const std::vector<std::string>& accessions,
                         IPackReader& gpk) const;

    // Angular distance between two embeddings (works with any dimension)
    static float angular_distance(const std::vector<float>& a,
                                  const std::vector<float>& b);
    static float angular_distance(const float* a, const float* b, size_t d);

};

} // namespace derep
