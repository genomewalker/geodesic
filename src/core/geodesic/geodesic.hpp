#pragma once
#include "core/types.hpp"
#include <array>
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <random>
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
struct EmbeddingStore {
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
    std::vector<uint64_t> oph_sig;    // Raw OPH signature for exact Jaccard refinement
    float isolation_score;            // Mean distance to k nearest neighbors
    float quality_score;              // completeness - 5*contamination (0-100)
    uint64_t genome_size;
    std::filesystem::path path;
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
        int syncmer_s = 8;

        // ANI threshold for redundancy
        double ani_threshold = 0.95;

        // HNSW index parameters
        int hnsw_m = 48;             // Higher M for better recall
        int hnsw_ef_construction = 400;
        int hnsw_ef_search = 200;

        // Parallelism
        int threads = 4;

        // Calibration
        int calibration_samples = 500;

        // Isolation score
        int isolation_k = 10;  // k nearest neighbors for isolation

        // FPS stopping criteria (derived from learned embedding↔ANI model)
        float diversity_threshold = 0.02f;    // Stop when diversity gain < this
        float min_rep_distance = 0.025f;      // Min distance between reps (electrostatic merge threshold)
        float max_rep_fraction = 0.2f;        // At most this fraction as reps
    };

    explicit GeodesicDerep(Config cfg);
    ~GeodesicDerep();

    // Auto-calibrate parameters based on ANI span from random sample
    // Returns recommended params + learned embedding↔ANI relationship
    struct CalibratedParams {
        int kmer_size;
        int embedding_dim;
        int sketch_size;
        float diversity_threshold;
        float min_rep_distance;     // Derived from target min ANI divergence
        double ani_min;
        double ani_max;
        double ani_spread;
        double ani_threshold;   // Derived FPS threshold: ani_min + 0.5 * ani_spread (as fraction 0-1)
        // Linear model: ANI ≈ slope * embedding_dist + intercept
        double ani_slope;
        double ani_intercept;
        // Genome size heterogeneity: std(sizes)/mean(sizes) across sampled genomes.
        // High CV indicates open pangenome → benefit from more representatives.
        float size_cv = 0.0f;
    };
    static CalibratedParams auto_calibrate(
        const std::vector<std::filesystem::path>& genomes,
        int sample_pairs = 50,
        int threads = 4);

    // Phase 1-2: Embed all genomes and build spatial index
    // quality_scores: path.string() → quality (completeness - 5*contamination)
    void build_index(const std::vector<std::filesystem::path>& genomes,
                     const std::unordered_map<std::string, double>& quality_scores = {});

    // Incremental build: load existing embeddings from store, embed only missing genomes
    // Returns number of newly embedded genomes
    size_t build_index_incremental(
        const std::vector<std::filesystem::path>& genomes,
        db::EmbeddingStore& store,
        const std::string& taxonomy,
        const std::unordered_map<std::string, double>& quality_scores = {});

    // Save embeddings to store (call after build_index or compute_isolation_scores)
    void save_embeddings_to_store(db::EmbeddingStore& store, const std::string& taxonomy);

    // Get representative genome IDs (after select_representatives)
    std::vector<uint64_t> get_representative_ids() const { return last_representative_ids_; }

    // Phase 3: Compute isolation scores (gravitational potential)
    void compute_isolation_scores();

    // Exclude paths from being selected as representatives (sets quality score to 0).
    // Call after build_index and detect_contamination, before select_representatives.
    void exclude_from_reps(const std::unordered_set<std::string>& paths);

    // Phase 4: Select representatives with lazy certified ANI
    std::vector<SimilarityEdge> select_representatives();

    // Calibration: fit ANI bounds from embedding distances
    void calibrate(const std::vector<std::filesystem::path>& sample_genomes);

    // Get all embeddings
    const std::vector<GenomeEmbedding>& embeddings() const { return embeddings_; }

    // Statistics
    size_t total_skani_calls() const { return skani_calls_; }
    size_t total_certified_redundant() const { return certified_redundant_; }
    size_t total_certified_unique() const { return certified_unique_; }

    // Exact Jaccard from raw OPH signatures: J = #{t: sig_A[t]==sig_B[t]} / M
    static double refine_jaccard(const std::vector<uint64_t>& sig_a,
                                 const std::vector<uint64_t>& sig_b);

    // ANI from exact Jaccard via Mash formula (calibration-free)
    static double jaccard_to_ani(double J, int kmer_size);

    // Contamination detection: returns genome IDs with anomalous embedding patterns
    // Contaminated genomes often appear as outliers (far from centroid) or
    // have inconsistent neighbor patterns (low isolation despite being outliers)
    struct ContaminationCandidate {
        uint64_t genome_id;
        float centroid_distance;    // Distance from species centroid
        float isolation_score;      // Mean distance to k-NN
        float anomaly_score;        // Combined anomaly metric
        std::filesystem::path path;
    };
    std::vector<ContaminationCandidate> detect_contamination_candidates(
        float z_threshold = 2.0f) const;

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
    std::vector<GenomeEmbedding> embeddings_;
    EmbeddingStore store_;  // SoA layout for SIMD-friendly access
    ANICalibrator calibrator_;

    // (Projection matrix removed: now uses OPH + CountSketch)

    // HNSW index (forward declaration to avoid header dependency)
    struct HNSWIndex;
    std::unique_ptr<HNSWIndex> index_;

    // Statistics
    mutable std::atomic<size_t> skani_calls_{0};
    mutable std::atomic<size_t> certified_redundant_{0};
    mutable std::atomic<size_t> certified_unique_{0};

    // Last selected representatives (for incremental workflows)
    std::vector<uint64_t> last_representative_ids_;

    // Path to accession mapping (for incremental store integration)
    std::unordered_map<std::string, std::string> path_to_accession_;

    // Generate embedding for single genome
    GenomeEmbedding embed_genome(const std::filesystem::path& path, uint64_t id);

    // CountSketch projection of OPH signature → unit embedding vector.
    // E[dot(embed(A), embed(B))] ≈ J(A,B)  →  calibration-free ANI formula.
    std::vector<float> countsketch_project(const std::vector<uint64_t>& oph_signature) const;

    // Brute-force O(n²) isolation scores for small n (no HNSW needed)
    void compute_isolation_scores_brute();

    // Find covering representatives for a query genome
    // Returns rep IDs that may cover query (need skani verification)
    std::vector<uint64_t> find_candidate_covers(
        uint64_t query_id,
        const std::vector<uint64_t>& current_reps);

    // Compute exact ANI between two genomes
    double compute_exact_ani(const std::filesystem::path& a,
                            const std::filesystem::path& b);

    // Angular distance between two embeddings (works with any dimension)
    static float angular_distance(const std::vector<float>& a,
                                  const std::vector<float>& b);
};

} // namespace derep
