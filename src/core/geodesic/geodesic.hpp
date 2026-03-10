#pragma once
#include "core/types.hpp"
#include <array>
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <memory>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace derep {

namespace db { class EmbeddingStore; }  // Forward declaration

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

// Structure-of-Arrays (SoA) embedding storage for cache-friendly SIMD access
struct SoAStore {
    size_t n = 0;
    size_t dim = 512;
    std::vector<float, AlignedAllocator<float, 64>> data;  // size n*dim, row-major

    // Metadata arrays (SoA)
    std::vector<uint64_t> genome_ids;
    std::vector<float> isolation_scores;
    std::vector<float> quality_scores;
    std::vector<uint64_t> genome_sizes;
    std::vector<std::filesystem::path> paths;

    void resize(size_t count, size_t dimension) {
        n = count;
        dim = dimension;
        data.resize(n * dim);
        genome_ids.resize(n);
        isolation_scores.resize(n);
        quality_scores.resize(n);
        genome_sizes.resize(n);
        paths.resize(n);
    }

    float* row(size_t i) { return data.data() + i * dim; }
    const float* row(size_t i) const { return data.data() + i * dim; }
};

// Genome embedding on unit sphere (dimension configurable)
struct GenomeEmbedding {
    uint64_t genome_id;
    std::vector<float> vector;        // CountSketch unit vector (dim=256)
    std::vector<uint16_t> oph_sig;         // 16-bit OPH signature (2× RAM vs uint32; bias-corrected Jaccard)
    std::vector<uint16_t> oph_sig2;        // Second OPH sketch (seed=1337) for variance reduction
    std::vector<uint64_t> real_bins_mask;  // Bitmask: bit t=1 iff bin t has a real k-mer (pre-densification)
    float isolation_score;            // Mean distance to k nearest neighbors
    float quality_score;              // completeness - 5*contamination (0-100)
    uint64_t genome_size;
    std::filesystem::path path;
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
        // Max concurrent NFS file readers during genome embedding.
        // 0 = auto: threads (total budget for this taxon caps NFS readers).
        int io_threads = 0;

        // Calibration
        int calibration_samples = 500;

        // Isolation score
        int isolation_k = 10;  // k nearest neighbors for isolation

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
    };

    explicit GeodesicDerep(Config cfg);
    ~GeodesicDerep();

    // Auto-calibrate embedding parameters from random genome sample
    struct CalibratedParams {
        int kmer_size;
        int embedding_dim;
        int sketch_size;
    };
    static CalibratedParams auto_calibrate(
        const std::vector<std::filesystem::path>& genomes,
        int sample_pairs = 50,
        int threads = 4);

    // Phase 1-2: Embed all genomes and build spatial index
    // quality_scores: path.string() → quality (completeness - 5*contamination)
    // emb_store/taxonomy: if provided, CountSketch vectors are saved asynchronously
    //   (background thread overlapping Nyström/HNSW/FPS) for future warm runs.
    void build_index(const std::vector<std::filesystem::path>& genomes,
                     const std::unordered_map<std::string, double>& quality_scores = {},
                     db::EmbeddingStore* emb_store = nullptr,
                     const std::string& taxonomy = "");

    // Incremental build: load existing embeddings from store, embed only missing genomes
    // Returns number of newly embedded genomes
    size_t build_index_incremental(
        const std::vector<std::filesystem::path>& genomes,
        db::EmbeddingStore& store,
        const std::string& taxonomy,
        const std::unordered_map<std::string, double>& quality_scores = {});

    // Async variant: uses pre-Nyström float snapshot, streams in batches to cap RAM overhead.
    void save_embeddings_async(db::EmbeddingStore& store, const std::string& taxonomy,
                               std::vector<std::vector<float>>&& vec_snap);

    // Get representative genome IDs (after select_representatives)
    std::vector<uint64_t> get_representative_ids() const { return last_representative_ids_; }

    // Exclude paths from being selected as representatives (sets quality score to 0).
    // Call after build_index and detect_contamination, before select_representatives.
    void exclude_from_reps(const std::unordered_set<std::string>& paths);

    // Pre-seed paths as representatives before FPS runs.
    // Call after build_index and before select_representatives.
    void set_pinned_representatives(const std::unordered_set<std::string>& paths);

    // Phase 4: Select representatives with lazy certified ANI
    std::vector<SimilarityEdge> select_representatives();

    // Get all embeddings
    const std::vector<GenomeEmbedding>& embeddings() const { return embeddings_; }

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

    // Contamination detection: returns genome IDs with anomalous embedding patterns
    struct ContaminationCandidate {
        uint64_t genome_id;
        float centroid_distance;    // Distance from species centroid (informational)
        float isolation_score;      // Mean distance to k-NN
        float anomaly_score;        // isolation_score (repurposed field)
        float genome_size_zscore;   // Z-score of genome size within taxon
        bool nn_outlier;            // isolation_score > 90% ANI threshold (primary: misassigned)
        float kmer_div_zscore = 0.0f; // k-mer diversity z-score (n_real_bins/kbp vs population; informational)
        float margin_to_threshold = 0.0f; // isolation_score - nn_threshold (positive = above threshold)
        std::string flag_reason;    // "nn_outlier", "size_outlier", or "nn_outlier+size_outlier"
        std::filesystem::path path;
    };
    std::vector<ContaminationCandidate> detect_contamination_candidates(
        float z_threshold = 2.0f) const;

    // NN distance distribution from HNSW.
    struct NNDistStats {
        double p5;
        double p50;
        double p95;
        // Maximum edge in the minimum spanning tree of the k-NN graph.
        // The minimum θ at which the k-NN graph becomes connected — the
        // binary_search_filter analog from the original graph-based pipeline.
        // Zero if unavailable (small-n brute-force path).
        double mst_max_edge = 0.0;
        double mst_w2 = 0.0;             // second-largest MST edge (penultimate Kruskal merge)
        uint32_t bridge_min_side = 0;    // smaller component at the final MST merge
        int k_conn = -1;           // smallest tested k where graph connects (-1 = never)
        int k_cap  = 0;            // K_cap used for the final threshold
        double drift_base = 0.0;   // mst_max(k_mst)/mst_max(K_cap) - 1 (>0.05 = unstable)
        // Instability flags: when set, mst_max_edge may not reflect intra-species scale.
        bool low_pair_count      = false;  // < 20 non-outlier genomes in MST
        bool pathological_bridge = false;  // tiny-side AND isolated terminal merge
        bool disconnected_mst    = false;  // MST has > 1 component at K_cap (truly disconnected)
        bool threshold_unstable  = false;  // |drift_base| > 5%
    };

    // Phase 3: Compute isolation scores AND return NN distance stats in one HNSW pass.
    // Replaces the old void compute_isolation_scores() + separate compute_nn_distance_stats().
    NNDistStats compute_isolation_scores();

    // Update thresholds after build_index (allows data-driven calibration).
    void set_min_rep_distance(float d) { cfg_.min_rep_distance = d; }
    void set_diversity_threshold(float d) { cfg_.diversity_threshold = d; }

    // Returns (path, genome_length_bp) for all embedded genomes after build_index.
    // Used to persist genome_length to the DB (OPH sketch computes it; input TSV doesn't have it).
    std::vector<std::pair<std::filesystem::path, uint64_t>> get_genome_sizes() const;

    // Genomes that permanently failed to read (all retries exhausted) during build_index.
    // Each entry: (file_path, error_reason). Caller should record these in jobs_failed.
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
    // Set by compute_isolation_scores(), read by detect_contamination_candidates().
    std::vector<int> component_ids_;

    // Last selected representatives (for incremental workflows)
    std::vector<uint64_t> last_representative_ids_;

    // Path to accession mapping (for incremental store integration)
    std::unordered_map<std::string, std::string> path_to_accession_;

    // Pinned representative paths (pre-seeded before FPS)
    std::unordered_set<std::string> pinned_rep_paths_;

    // Async embedding save (started in build_index, joined in destructor)
    std::future<void> async_save_future_;

    // Whether Nyström embedding was applied (false → exact Jaccard FPS for small n)
    bool nystrom_applied_ = false;

    // Canonical lookup: genome_id → row index in embeddings_/store_.
    // Rebuilt after every sort of embeddings_ so that genome_ids (which are opaque
    // identifiers after sorting) are never used as direct array indices.
    std::unordered_map<uint64_t, size_t> gid_to_row_;

    // Genomes that failed to read after retries: (file_path, reason).
    // Written from OMP threads (mutex-protected); read by taxon_processor after build_index.
    std::vector<std::pair<std::string, std::string>> failed_reads_;
    std::mutex failed_reads_mutex_;

    // Decompressed FASTA buffers cached during embed_genome() for reuse in
    // materialize_sig2_for_indices() — avoids NFS re-read for anchor sig2 sketching.
    // Indexed by genome id (== embeddings_ index). Freed after Nyström step.
    static constexpr size_t kBufCacheLimitBytes = 2'500'000'000ULL;
    std::vector<std::vector<char>> buf_cache_;
    std::atomic<size_t> buf_cache_bytes_{0};

    // Generate embedding for single genome (reads from NFS)
    GenomeEmbedding embed_genome(const std::filesystem::path& path, uint64_t id);

    // Generate embedding from pre-decompressed FASTA buffer (producer-consumer path)
    GenomeEmbedding embed_genome_from_buffer(const char* data, size_t len,
                                              uint64_t id, const std::filesystem::path& path);

    // Nyström spectral embedding: replace placeholder vectors with data-adapted
    // projections onto the top eigenvectors of the OPH Jaccard kernel.
    void apply_nystrom_embeddings();

    // Lazily materialize oph_sig2 for a subset of genome indices by re-reading
    // their FASTA files. Used for anchors (Gram matrix) and borderline candidates
    // (Phase 7 verification). Skips indices that already have sig2 or lack a path.
    void materialize_sig2_for_indices(const std::vector<size_t>& indices);

    // Brute-force O(n²) isolation scores for small n (no HNSW needed)
    void compute_isolation_scores_brute();

    // Angular distance between two embeddings (works with any dimension)
    static float angular_distance(const std::vector<float>& a,
                                  const std::vector<float>& b);
};

} // namespace derep
