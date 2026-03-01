#include "core/geodesic/geodesic.hpp"
#include "core/sketch/minhash.hpp"
#include "core/progress.hpp"
#include "core/logging.hpp"
#include "db/embedding_store.hpp"
#include <hnswlib/hnswlib.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <future>
#include <numeric>
#include <queue>
#include <random>
#include <thread>
#include <unordered_set>

// SIMD support detection
#if defined(__AVX2__)
#include <immintrin.h>
#define GEODESIC_USE_AVX2 1
#else
#define GEODESIC_USE_AVX2 0
#endif

// OpenMP support
#if defined(_OPENMP)
#include <omp.h>
#define GEODESIC_USE_OMP 1
#else
#define GEODESIC_USE_OMP 0
#endif

namespace derep {

// Skip HNSW build and IsolationForest for small taxa — brute-force is faster
static constexpr size_t SMALL_N_THRESHOLD = 50;

// AVX2 optimized dot product (unaligned loads — works with any allocation)
#if GEODESIC_USE_AVX2
float dot_product_simd(const float* a, const float* b, size_t dim) {
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;

    // Process 32 floats per iteration (4 AVX2 registers)
    for (; i + 32 <= dim; i += 32) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        __m256 va2 = _mm256_loadu_ps(a + i + 16);
        __m256 vb2 = _mm256_loadu_ps(b + i + 16);
        __m256 va3 = _mm256_loadu_ps(a + i + 24);
        __m256 vb3 = _mm256_loadu_ps(b + i + 24);

        sum = _mm256_fmadd_ps(va0, vb0, sum);
        sum = _mm256_fmadd_ps(va1, vb1, sum);
        sum = _mm256_fmadd_ps(va2, vb2, sum);
        sum = _mm256_fmadd_ps(va3, vb3, sum);
    }

    // Process remaining 8-float blocks
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    // Horizontal sum
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 s = _mm_add_ps(hi, lo);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    float result = _mm_cvtss_f32(s);

    // Tail elements (if dim not multiple of 8)
    for (; i < dim; ++i) {
        result += a[i] * b[i];
    }

    return result;
}
#else
// Scalar fallback
float dot_product_simd(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}
#endif

// Real HNSW index using hnswlib for O(log n) nearest neighbor search
// Uses inner product space (for normalized vectors, maximizing IP = minimizing angular distance)
struct GeodesicDerep::HNSWIndex {
    size_t dim = 256;  // Set dynamically during build
    std::unique_ptr<hnswlib::InnerProductSpace> space;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index;
    std::vector<GenomeEmbedding>* embeddings;
    int ef_search;

    void build(std::vector<GenomeEmbedding>& embs, int M, int ef_construction) {
        embeddings = &embs;
        if (embs.empty()) return;

        dim = embs[0].vector.size();
        space = std::make_unique<hnswlib::InnerProductSpace>(dim);
        index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
            space.get(), embs.size(), M, ef_construction);

        // Add all embeddings to the index (parallel)
        if (!embs.empty()) {
            index->addPoint(embs[0].vector.data(), 0);
#if GEODESIC_USE_OMP
            #pragma omp parallel for schedule(dynamic, 1000)
#endif
            for (size_t i = 1; i < embs.size(); ++i) {
                index->addPoint(embs[i].vector.data(), i);
            }
        }
    }

    // KNN search - returns (genome_id, angular_distance) pairs
    std::vector<std::pair<uint64_t, float>> search(
        const std::vector<float>& query,
        size_t k,
        const std::unordered_set<uint64_t>* filter = nullptr) const {

        if (!index) return {};
        index->setEf(std::max(ef_search, static_cast<int>(k)));

        // hnswlib returns (distance, label) where distance = 1 - inner_product for IP space
        auto result = index->searchKnn(query.data(), k);

        std::vector<std::pair<uint64_t, float>> out;
        out.reserve(result.size());

        while (!result.empty()) {
            auto [ip_dist, label] = result.top();
            result.pop();

            if (filter && filter->find(label) == filter->end()) continue;

            // Convert IP distance to angular distance
            // IP distance in hnswlib = 1 - inner_product
            // For normalized vectors: inner_product = cos(angle)
            // angular_distance = arccos(inner_product) / pi
            float inner_product = 1.0f - ip_dist;
            inner_product = std::clamp(inner_product, -1.0f, 1.0f);
            float angular_dist = std::acos(inner_product) / static_cast<float>(M_PI);

            out.emplace_back(label, angular_dist);
        }

        // Reverse to get ascending order (closest first)
        std::reverse(out.begin(), out.end());
        return out;
    }

    // Radius search - fallback to brute force (HNSW doesn't support radius natively)
    std::vector<std::pair<uint64_t, float>> search_radius(
        const std::vector<float>& query,
        float radius,
        const std::unordered_set<uint64_t>* filter = nullptr) const {

        std::vector<std::pair<uint64_t, float>> result;
        for (const auto& emb : *embeddings) {
            if (filter && filter->find(emb.genome_id) == filter->end()) continue;
            float dist = GeodesicDerep::angular_distance(query, emb.vector);
            if (dist <= radius) {
                result.emplace_back(emb.genome_id, dist);
            }
        }

        std::sort(result.begin(), result.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });
        return result;
    }
};

// ANICalibrator implementation

void ANICalibrator::fit(const std::vector<std::pair<double, double>>& samples) {
    if (samples.size() < 10) {
        spdlog::warn("ANICalibrator: insufficient samples ({}), using defaults", samples.size());
        fitted_ = false;
        return;
    }

    // Sort by distance
    auto sorted = samples;
    std::sort(sorted.begin(), sorted.end());

    // Build distance grid
    size_t grid_size = std::min(size_t{100}, samples.size());
    distance_grid_.resize(grid_size);
    ani_lower_curve_.resize(grid_size);
    ani_upper_curve_.resize(grid_size);

    for (size_t i = 0; i < grid_size; ++i) {
        size_t idx = i * sorted.size() / grid_size;
        distance_grid_[i] = sorted[idx].first;

        // Find all samples near this distance
        std::vector<double> nearby_anis;
        double dist_window = (i < grid_size - 1) ?
            (distance_grid_[i + 1] - distance_grid_[i]) * 0.5 :
            (distance_grid_[i] - distance_grid_[i - 1]) * 0.5;

        for (const auto& [d, ani] : sorted) {
            if (std::abs(d - distance_grid_[i]) <= dist_window) {
                nearby_anis.push_back(ani);
            }
        }

        if (nearby_anis.empty()) {
            nearby_anis.push_back(sorted[idx].second);
        }

        std::sort(nearby_anis.begin(), nearby_anis.end());

        // 5th and 95th percentile
        size_t lower_idx = nearby_anis.size() * 5 / 100;
        size_t upper_idx = nearby_anis.size() * 95 / 100;
        upper_idx = std::min(upper_idx, nearby_anis.size() - 1);

        ani_lower_curve_[i] = nearby_anis[lower_idx] - lower_margin_;
        ani_upper_curve_[i] = nearby_anis[upper_idx] + upper_margin_;
    }

    // Enforce monotonicity (lower should decrease, upper should decrease with distance)
    for (size_t i = 1; i < grid_size; ++i) {
        ani_lower_curve_[i] = std::min(ani_lower_curve_[i], ani_lower_curve_[i - 1]);
        ani_upper_curve_[i] = std::min(ani_upper_curve_[i], ani_upper_curve_[i - 1]);
    }

    fitted_ = true;
    spdlog::info("ANICalibrator: fitted on {} samples, grid_size={}", samples.size(), grid_size);
}

ANICalibrator::Bounds ANICalibrator::predict(double embedding_distance) const {
    if (!fitted_ || distance_grid_.empty()) {
        // Calibration-free formula: ANI = (2J/(1+J))^(1/k), J ≈ cos(π*d)
        // Derived from Mash chain: 2J/(1+J) = exp(-k*(1-ANI)), k=21
        double approx_ani = 1.0;
        if (embedding_distance > 0.0 && embedding_distance < 0.5) {
            double cos_sim = std::cos(embedding_distance * M_PI);
            if (cos_sim > 0.0) {
                double ratio = 2.0 * cos_sim / (1.0 + cos_sim);
                approx_ani = std::pow(ratio, 1.0 / 21.0);
            }
        } else if (embedding_distance >= 0.5) {
            approx_ani = 0.7;
        }
        return {approx_ani - 0.015, std::min(1.0, approx_ani + 0.015)};
    }

    // Binary search for position in grid
    auto it = std::lower_bound(distance_grid_.begin(), distance_grid_.end(), embedding_distance);

    if (it == distance_grid_.begin()) {
        return {ani_lower_curve_.front(), ani_upper_curve_.front()};
    }
    if (it == distance_grid_.end()) {
        return {ani_lower_curve_.back(), ani_upper_curve_.back()};
    }

    // Linear interpolation
    size_t idx = it - distance_grid_.begin();
    double t = (embedding_distance - distance_grid_[idx - 1]) /
               (distance_grid_[idx] - distance_grid_[idx - 1]);

    double lower = ani_lower_curve_[idx - 1] + t * (ani_lower_curve_[idx] - ani_lower_curve_[idx - 1]);
    double upper = ani_upper_curve_[idx - 1] + t * (ani_upper_curve_[idx] - ani_upper_curve_[idx - 1]);

    return {lower, upper};
}

double ANICalibrator::inverse_upper(double target_ani) const {
    if (!fitted_ || distance_grid_.empty()) {
        // Inverse of ANI = (2J/(1+J))^(1/k): J = ANI^k / (2 - ANI^k), k=21
        if (target_ani >= 1.0) return 0.0;
        if (target_ani <= 0.7) return 0.5;
        double ak = std::pow(target_ani, 21.0);
        double cos_sim = std::min(1.0, ak / (2.0 - ak));
        return std::acos(cos_sim) / M_PI;
    }

    // Binary search on upper curve
    for (size_t i = 0; i < ani_upper_curve_.size(); ++i) {
        if (ani_upper_curve_[i] < target_ani) {
            if (i == 0) return 0.0;
            // Interpolate
            double t = (target_ani - ani_upper_curve_[i]) /
                       (ani_upper_curve_[i - 1] - ani_upper_curve_[i]);
            return distance_grid_[i] - t * (distance_grid_[i] - distance_grid_[i - 1]);
        }
    }

    return distance_grid_.back();
}

double ANICalibrator::inverse_lower(double target_ani) const {
    if (!fitted_ || distance_grid_.empty()) {
        // Inverse of ANI = (2J/(1+J))^(1/k), with margin for conservative bound, k=21
        double adjusted_ani = std::min(target_ani + 0.01, 1.0);
        if (adjusted_ani >= 1.0) return 0.0;
        if (adjusted_ani <= 0.7) return 0.5;
        double ak = std::pow(adjusted_ani, 21.0);
        double cos_sim = std::min(1.0, ak / (2.0 - ak));
        return std::acos(cos_sim) / M_PI;
    }

    for (size_t i = 0; i < ani_lower_curve_.size(); ++i) {
        if (ani_lower_curve_[i] < target_ani) {
            if (i == 0) return 0.0;
            double t = (target_ani - ani_lower_curve_[i]) /
                       (ani_lower_curve_[i - 1] - ani_lower_curve_[i]);
            return distance_grid_[i] - t * (distance_grid_[i] - distance_grid_[i - 1]);
        }
    }

    return distance_grid_.back();
}

// Auto-calibrate embedding parameters from ANI span
GeodesicDerep::CalibratedParams GeodesicDerep::auto_calibrate(
    const std::vector<std::filesystem::path>& genomes,
    int sample_pairs,
    int threads) {

    // Helper: embed a set of genome indices and return index→embedding map.
    // Uses OPH+CountSketch projection: E[dot(u,v)] = J(A,B).
    auto embed_sample = [&](const std::vector<size_t>& indices,
                             int kmer_size, int sketch_size, int embedding_dim,
                             std::unordered_map<size_t, uint64_t>* out_sizes = nullptr)
        -> std::unordered_map<size_t, std::vector<float>>
    {
        MinHasher hasher({ .kmer_size = kmer_size, .sketch_size = sketch_size, .seed = 42 });

        // OPH+CountSketch projection: E[dot(u,v)] = J(A,B)
        auto cs_project = [&](const std::vector<uint64_t>& oph_sig) -> std::vector<float> {
            auto mix = [](uint64_t x) -> uint64_t {
                x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
                x ^= x >> 27; x *= 0x94d049bb133111ebULL;
                x ^= x >> 31; return x;
            };
            constexpr uint64_t P1 = 0x9e3779b97f4a7c15ULL;
            constexpr uint64_t P2 = 0x6c62272e07bb0142ULL;
            const uint64_t EMPTY = std::numeric_limits<uint64_t>::max();
            const uint64_t dim64 = static_cast<uint64_t>(embedding_dim);
            std::vector<float> emb(embedding_dim, 0.0f);
            for (int t = 0; t < static_cast<int>(oph_sig.size()); ++t) {
                if (oph_sig[t] == EMPTY) continue;
                uint64_t h1 = mix(oph_sig[t] + static_cast<uint64_t>(t) * P1);
                uint64_t h2 = mix(oph_sig[t] + static_cast<uint64_t>(t) * P2);
                emb[static_cast<int>(h1 % dim64)] += (h2 & 1) ? 1.0f : -1.0f;
            }
            float norm = dot_product_simd(emb.data(), emb.data(), embedding_dim);
            norm = std::sqrt(norm);
            if (norm > 1e-10f)
                for (float& v : emb) v /= norm;
            return emb;
        };

        // Parallel sketch + project
        std::unordered_map<size_t, std::vector<float>> cache;
        cache.reserve(indices.size());
        std::mutex mtx;
        size_t nw = std::min(static_cast<size_t>(threads), indices.size());
        size_t chunk = (indices.size() + nw - 1) / nw;
        std::vector<std::future<void>> futs;
        for (size_t t = 0; t < nw; ++t) {
            size_t lo = t * chunk, hi = std::min(lo + chunk, indices.size());
            if (lo >= hi) break;
            futs.push_back(std::async(std::launch::async, [&, lo, hi]() {
                for (size_t pos = lo; pos < hi; ++pos) {
                    size_t idx = indices[pos];
                    auto oph = hasher.sketch_oph(genomes[idx], sketch_size);
                    auto emb = cs_project(oph.signature);
                    std::lock_guard lock(mtx);
                    cache[idx] = std::move(emb);
                    if (out_sizes) (*out_sizes)[idx] = oph.genome_length;
                }
            }));
        }
        for (auto& f : futs) f.get();
        return cache;
    };

    // Helper: compute angular distances for a set of index pairs given a cache.
    auto compute_distances = [](
        const std::vector<std::pair<size_t, size_t>>& pairs,
        const std::unordered_map<size_t, std::vector<float>>& cache) -> std::vector<double>
    {
        std::vector<double> dists;
        dists.reserve(pairs.size());
        for (const auto& [a, b] : pairs) {
            auto ia = cache.find(a), ib = cache.find(b);
            if (ia == cache.end() || ib == cache.end()) continue;
            const auto& ea = ia->second;
            const auto& eb = ib->second;
            float dot = dot_product_simd(ea.data(), eb.data(), ea.size());
            dot = std::max(-1.0f, std::min(1.0f, dot));
            dists.push_back(std::acos(dot) / M_PI);
        }
        return dists;
    };

    // Default params returned on degenerate input
    auto defaults = []() -> CalibratedParams {
        return { 21, 256, 10000, 0.02f, 0.01f, 100.0, 100.0, 0.0, 21.0, 0.0 };
    };

    CalibratedParams params = defaults();

    if (genomes.size() < 2) return params;

    // ----------------------------------------------------------------
    // Step 1: sample random pairs + unique genome indices
    // ----------------------------------------------------------------
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<size_t> udist(0, genomes.size() - 1);
    std::vector<std::pair<size_t, size_t>> pairs;
    std::unordered_set<size_t> seen;
    for (int tries = 0; tries < sample_pairs * 4 && (int)pairs.size() < sample_pairs; ++tries) {
        size_t a = udist(rng), b = udist(rng);
        if (a != b) { pairs.emplace_back(a, b); seen.insert(a); seen.insert(b); }
    }
    if (pairs.empty()) return params;

    std::vector<size_t> idx_vec(seen.begin(), seen.end());

    // ----------------------------------------------------------------
    // Step 2: first pass with default params (k=21, sketch=10k, dim=256)
    //         to measure spread and choose final tier
    // ----------------------------------------------------------------
    const int k0 = 21, s0 = 10000, d0 = 256;
    std::unordered_map<size_t, uint64_t> size_map;
    auto cache0 = embed_sample(idx_vec, k0, s0, d0, &size_map);
    auto dists0  = compute_distances(pairs, cache0);

    // Genome size coefficient of variation: proxy for open pangenome diversity
    float size_cv = 0.0f;
    if (size_map.size() >= 2) {
        double sum = 0.0, sum2 = 0.0;
        for (const auto& [idx, sz] : size_map) {
            sum  += static_cast<double>(sz);
            sum2 += static_cast<double>(sz) * static_cast<double>(sz);
        }
        double n   = static_cast<double>(size_map.size());
        double mean = sum / n;
        if (mean > 0.0) {
            double var = sum2 / n - mean * mean;
            size_cv = static_cast<float>(std::sqrt(std::max(0.0, var)) / mean);
        }
    }

    if (dists0.empty()) return params;

    std::vector<double> sorted0 = dists0;
    std::sort(sorted0.begin(), sorted0.end());
    size_t n0 = sorted0.size();
    double p5_0  = sorted0[n0 * 5  / 100];
    double p95_0 = sorted0[n0 * 95 / 100];
    double spread = p95_0 - p5_0;

    // Choose tier from first-pass spread
    int kmer, sketch_size, embedding_dim;
    if (spread < 0.05) {
        // Very clonal (≳99% ANI): use high-resolution embedding
        kmer = 31; sketch_size = 20000; embedding_dim = 512;
    } else if (spread < 0.20) {
        // Typical (95–99% ANI)
        kmer = 21; sketch_size = 10000; embedding_dim = 256;
    } else {
        // Very diverse (<95% ANI)
        kmer = 16; sketch_size = 5000;  embedding_dim = 128;
    }

    // ----------------------------------------------------------------
    // Step 3: second pass only if tier changed
    // ----------------------------------------------------------------
    std::vector<double> final_dists;
    if (kmer == k0 && sketch_size == s0 && embedding_dim == d0) {
        final_dists = dists0;
    } else {
        auto cache1 = embed_sample(idx_vec, kmer, sketch_size, embedding_dim);
        final_dists  = compute_distances(pairs, cache1);
    }

    if (final_dists.empty()) {
        params.kmer_size = kmer; params.embedding_dim = embedding_dim;
        params.sketch_size = sketch_size;
        return params;
    }

    // ----------------------------------------------------------------
    // Step 4: derive thresholds from final distance distribution
    // ----------------------------------------------------------------
    std::vector<double> sorted_f = final_dists;
    std::sort(sorted_f.begin(), sorted_f.end());
    size_t nf = sorted_f.size();
    double p5  = sorted_f[nf * 5  / 100];
    double p50 = sorted_f[nf * 50 / 100];
    double p95 = sorted_f[nf * 95 / 100];

    // Infer ANI range via OPH+CountSketch chain: ANI = (2J/(1+J))^(1/k), J ≈ cos(π*d)
    auto dist_to_ani = [&](double d) -> double {
        double c = std::cos(M_PI * d);
        if (c <= 0.0) return 0.0;
        double ratio = 2.0 * c / (1.0 + c);
        return std::pow(ratio, 1.0 / kmer) * 100.0;
    };
    params.ani_max    = dist_to_ani(p5);
    params.ani_min    = dist_to_ani(p95);
    params.ani_spread = params.ani_max - params.ani_min;
    // FPS threshold: midpoint of detected ANI range (as fraction), clamped to [0.90, 0.9999]
    params.ani_threshold = std::max(0.90, std::min(0.9999,
        (params.ani_min + 0.5 * params.ani_spread) / 100.0));

    params.kmer_size        = kmer;
    params.embedding_dim    = embedding_dim;
    params.sketch_size      = sketch_size;

    // diversity_threshold: FPS stops when every genome is within ani_threshold of some rep.
    // Derived from Mash chain: J = q/(2-q), q = exp(-k*(1-ANI)), dist = acos(J)/π
    // This is calibration-free and matches the semantic intent of ani_threshold.
    {
        double q = std::exp(-static_cast<double>(kmer) * (1.0 - params.ani_threshold));
        double j = q / (2.0 - q);
        params.diversity_threshold = static_cast<float>(std::acos(j) / M_PI);
    }
    // min_rep_distance: don't merge reps that are genuinely distinct (p5 of observed distances)
    params.min_rep_distance = static_cast<float>(p5);
    if (params.min_rep_distance >= params.diversity_threshold)
        params.min_rep_distance = params.diversity_threshold * 0.5f;
    params.ani_slope     = static_cast<double>(kmer);
    params.ani_intercept = 0.0;
    params.size_cv       = size_cv;

    if (is_verbose()) {
        spdlog::info("GEODESIC: Calibration — {} pairs, spread={:.3f}, tier k={}/dim={}/sketch={}, size_cv={:.3f}",
                     nf, spread, kmer, embedding_dim, sketch_size, size_cv);
        spdlog::info("GEODESIC: Distances P5={:.4f}, P50={:.4f}, P95={:.4f}  "
                     "→ ANI {:.1f}%–{:.1f}%  thresholds: diversity={:.4f}, min_rep={:.4f}",
                     p5, p50, p95, params.ani_min, params.ani_max,
                     params.diversity_threshold, params.min_rep_distance);
    }

    return params;
}

// GeodesicDerep implementation

GeodesicDerep::GeodesicDerep(Config cfg)
    : cfg_(std::move(cfg))
    , index_(std::make_unique<HNSWIndex>()) {
}

GeodesicDerep::~GeodesicDerep() = default;

std::vector<float> GeodesicDerep::countsketch_project(
        const std::vector<uint64_t>& oph_signature) const {
    // splitmix64 finalizer for independent hashing of each token
    auto mix = [](uint64_t x) -> uint64_t {
        x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27; x *= 0x94d049bb133111ebULL;
        x ^= x >> 31; return x;
    };
    constexpr uint64_t PRIME1 = 0x9e3779b97f4a7c15ULL;
    constexpr uint64_t PRIME2 = 0x6c62272e07bb0142ULL;
    const uint64_t EMPTY = std::numeric_limits<uint64_t>::max();
    const uint64_t dim64 = static_cast<uint64_t>(cfg_.embedding_dim);

    // Each OPH bin t contributes +1 or -1 to one output dimension.
    // E[dot(u,v)] = fraction of bins where sig_A[t] == sig_B[t] = Jaccard(A,B).
    std::vector<float> emb(cfg_.embedding_dim, 0.0f);
    for (int t = 0; t < static_cast<int>(oph_signature.size()); ++t) {
        if (oph_signature[t] == EMPTY) continue;
        uint64_t h1 = mix(oph_signature[t] + static_cast<uint64_t>(t) * PRIME1);
        uint64_t h2 = mix(oph_signature[t] + static_cast<uint64_t>(t) * PRIME2);
        emb[static_cast<int>(h1 % dim64)] += (h2 & 1) ? 1.0f : -1.0f;
    }

    float norm = dot_product_simd(emb.data(), emb.data(), cfg_.embedding_dim);
    norm = std::sqrt(norm);
    if (norm > 1e-10f)
        for (float& v : emb) v /= norm;
    return emb;
}

float GeodesicDerep::angular_distance(const std::vector<float>& a,
                                       const std::vector<float>& b) {
    // Cosine similarity → angular distance
    float dot = dot_product_simd(a.data(), b.data(), std::min(a.size(), b.size()));
    // Clamp for numerical stability
    dot = std::max(-1.0f, std::min(1.0f, dot));
    // Angular distance in [0, 1] range
    return std::acos(dot) / static_cast<float>(M_PI);
}

double GeodesicDerep::refine_jaccard(const std::vector<uint64_t>& sig_a,
                                      const std::vector<uint64_t>& sig_b) {
    if (sig_a.empty() || sig_b.empty()) return 0.0;
    const size_t m = std::min(sig_a.size(), sig_b.size());
    size_t matches = 0;
    for (size_t t = 0; t < m; ++t)
        if (sig_a[t] == sig_b[t]) ++matches;
    return static_cast<double>(matches) / static_cast<double>(m);
}

double GeodesicDerep::jaccard_to_ani(double J, int kmer_size) {
    if (J <= 0.0) return 70.0;
    if (J >= 1.0) return 100.0;
    double ratio = 2.0 * J / (1.0 + J);
    return std::max(70.0, std::min(100.0, std::pow(ratio, 1.0 / kmer_size) * 100.0));
}

GenomeEmbedding GeodesicDerep::embed_genome(const std::filesystem::path& path, uint64_t id) {
    // Use existing MinHasher
    MinHasher hasher({
        .kmer_size = cfg_.kmer_size,
        .sketch_size = cfg_.sketch_size,
        .seed = 42
    });

    auto oph = hasher.sketch_oph(path, cfg_.sketch_size);

    GenomeEmbedding emb;
    emb.genome_id = id;
    emb.vector = countsketch_project(oph.signature);
    if (oph.genome_length == 0 || emb.vector.empty() ||
        dot_product_simd(emb.vector.data(), emb.vector.data(), emb.vector.size()) < 0.01f) {
        spdlog::warn("GEODESIC: zero embedding for {} (genome_length={}): "
                     "empty or unreadable FASTA — genome excluded from stats",
                     path.filename().string(), oph.genome_length);
    }
    emb.oph_sig = std::move(oph.signature);
    emb.isolation_score = 0.0f;
    emb.quality_score = 50.0f;  // Default, overridden in build_index
    emb.genome_size = oph.genome_length;
    emb.path = path;

    return emb;
}

void GeodesicDerep::build_index(const std::vector<std::filesystem::path>& genomes,
                                 const std::unordered_map<std::string, double>& quality_scores) {
    if (is_verbose()) spdlog::info("GEODESIC: embedding {} genomes (dim={}, k={}, threads={})",
                 genomes.size(), cfg_.embedding_dim, cfg_.kmer_size, cfg_.threads);

#if GEODESIC_USE_AVX2
    if (is_verbose()) spdlog::info("GEODESIC: AVX2 SIMD enabled");
#endif
#if GEODESIC_USE_OMP
    if (is_verbose()) spdlog::info("GEODESIC: OpenMP parallel enabled ({} threads)", cfg_.threads);
    omp_set_num_threads(cfg_.threads);
#endif

    size_t n = genomes.size();
    embeddings_.resize(n);
    store_.resize(n, cfg_.embedding_dim);

    ProgressCounter progress(n, "GEODESIC: embedding");

    // Precompute path strings to avoid repeated .string() conversions in loops
    std::vector<std::string> path_strs(n);
    for (size_t i = 0; i < n; ++i) path_strs[i] = genomes[i].string();

    // Parallel embedding with OpenMP or std::async
#if GEODESIC_USE_OMP
    #pragma omp parallel for schedule(dynamic, 10)
    for (size_t i = 0; i < n; ++i) {
        embeddings_[i] = embed_genome(genomes[i], i);
        auto it = quality_scores.find(path_strs[i]);
        if (it != quality_scores.end()) {
            embeddings_[i].quality_score = static_cast<float>(it->second);
        } else {
            embeddings_[i].quality_score = 50.0f;
        }
        progress.increment();
    }
#else
    auto worker = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            embeddings_[i] = embed_genome(genomes[i], i);
            auto it = quality_scores.find(path_strs[i]);
            if (it != quality_scores.end()) {
                embeddings_[i].quality_score = static_cast<float>(it->second);
            } else {
                embeddings_[i].quality_score = 50.0f;
            }
            progress.increment();
        }
    };

    size_t num_threads = std::min(static_cast<size_t>(cfg_.threads), n);
    if (num_threads <= 1) {
        worker(0, n);
    } else {
        std::vector<std::future<void>> futures;
        size_t chunk = (n + num_threads - 1) / num_threads;
        for (size_t t = 0; t < num_threads; ++t) {
            size_t start = t * chunk;
            size_t end = std::min(start + chunk, n);
            if (start < end) {
                futures.push_back(std::async(std::launch::async, worker, start, end));
            }
        }
        for (auto& f : futures) f.get();
    }
#endif
    progress.finish();

    // Copy embeddings to SoA store for SIMD-friendly access
    for (size_t i = 0; i < n; ++i) {
        store_.genome_ids[i] = embeddings_[i].genome_id;
        store_.isolation_scores[i] = embeddings_[i].isolation_score;
        store_.quality_scores[i] = embeddings_[i].quality_score;
        store_.genome_sizes[i] = embeddings_[i].genome_size;
        store_.paths[i] = embeddings_[i].path;
        std::copy(embeddings_[i].vector.begin(), embeddings_[i].vector.end(), store_.row(i));
    }

    // Build HNSW index — skip for small n (brute-force pairwise is faster)
    if (n > SMALL_N_THRESHOLD) {
        index_->build(embeddings_, cfg_.hnsw_m, cfg_.hnsw_ef_construction);
        index_->ef_search = cfg_.hnsw_ef_search;
        if (is_verbose()) spdlog::info("GEODESIC: HNSW index built ({} embeddings, M={}, ef={})",
                     embeddings_.size(), cfg_.hnsw_m, cfg_.hnsw_ef_construction);
    }
}

void GeodesicDerep::compute_isolation_scores() {
    if (!index_->index) {
        compute_isolation_scores_brute();
        return;
    }
    if (is_verbose()) spdlog::info("GEODESIC: computing isolation scores (k={})", cfg_.isolation_k);

    // For each genome, compute mean distance to k nearest neighbors (parallel: search is thread-safe)
    // Use low ef for isolation queries: precision not needed, just ordering.
    const int saved_ef = index_->ef_search;
    index_->ef_search = 15;

    size_t n_emb = embeddings_.size();
#if GEODESIC_USE_OMP
    #pragma omp parallel for schedule(dynamic, 100) num_threads(cfg_.threads)
#endif
    for (size_t ei = 0; ei < n_emb; ++ei) {
        auto& emb = embeddings_[ei];
        auto neighbors = index_->search(emb.vector, cfg_.isolation_k + 1);

        float total_dist = 0.0f;
        int count = 0;
        for (const auto& [id, dist] : neighbors) {
            if (id != emb.genome_id) {
                total_dist += dist;
                ++count;
            }
        }

        emb.isolation_score = (count > 0) ? total_dist / count : 1.0f;
    }
    index_->ef_search = saved_ef;

    // Compute median genome size for length normalization
    std::vector<uint64_t> sizes;
    sizes.reserve(embeddings_.size());
    for (const auto& e : embeddings_) sizes.push_back(e.genome_size);
    auto mid = sizes.begin() + sizes.size() / 2;
    std::nth_element(sizes.begin(), mid, sizes.end());
    uint64_t median_size = sizes[sizes.size() / 2];

    // Sort by quality×length-weighted isolation (descending)
    // fitness = isolation × (quality/100) × sqrt(size/median)
    // Prefer isolated, high-quality, AND longer genomes
    float median_sz = static_cast<float>(median_size);
    std::sort(embeddings_.begin(), embeddings_.end(),
              [median_sz](const auto& a, const auto& b) {
                  float len_a = std::sqrt(static_cast<float>(a.genome_size) / median_sz);
                  float len_b = std::sqrt(static_cast<float>(b.genome_size) / median_sz);
                  float fitness_a = a.isolation_score * (a.quality_score / 100.0f) * len_a;
                  float fitness_b = b.isolation_score * (b.quality_score / 100.0f) * len_b;
                  return fitness_a > fitness_b;
              });

    // Update SoA store to match sorted embeddings
    size_t n = embeddings_.size();
    for (size_t i = 0; i < n; ++i) {
        store_.genome_ids[i] = embeddings_[i].genome_id;
        store_.isolation_scores[i] = embeddings_[i].isolation_score;
        store_.quality_scores[i] = embeddings_[i].quality_score;
        store_.genome_sizes[i] = embeddings_[i].genome_size;
        store_.paths[i] = embeddings_[i].path;
        std::copy(embeddings_[i].vector.begin(), embeddings_[i].vector.end(), store_.row(i));
    }

    if (is_verbose()) spdlog::info("GEODESIC: isolation scores computed (max={:.4f}, min={:.4f})",
                 embeddings_.front().isolation_score,
                 embeddings_.back().isolation_score);
    if (is_verbose()) spdlog::info("GEODESIC: quality range: {:.1f} - {:.1f}",
                 embeddings_.back().quality_score, embeddings_.front().quality_score);
    if (is_verbose()) {
        auto [sz_min, sz_max] = std::minmax_element(sizes.begin(), sizes.end());
        spdlog::info("GEODESIC: genome size range: {} - {} bp (median={})",
                     *sz_min, *sz_max, median_size);
    }
}

void GeodesicDerep::compute_isolation_scores_brute() {
    size_t n   = embeddings_.size();
    size_t dim = store_.dim;
    int    k   = std::min(cfg_.isolation_k, static_cast<int>(n) - 1);

    for (size_t i = 0; i < n; ++i) {
        const float* vi = store_.row(i);
        std::vector<float> dists;
        dists.reserve(n - 1);
        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            float dot = dot_product_simd(vi, store_.row(j), dim);
            dot = std::clamp(dot, -1.0f, 1.0f);
            dists.push_back(std::acos(dot) / static_cast<float>(M_PI));
        }
        if (k > 0 && k <= static_cast<int>(dists.size())) {
            std::nth_element(dists.begin(), dists.begin() + k, dists.end());
            float sum = 0.0f;
            for (int d = 0; d < k; ++d) sum += dists[d];
            embeddings_[i].isolation_score = sum / static_cast<float>(k);
        } else {
            float sum = 0.0f;
            for (float d : dists) sum += d;
            embeddings_[i].isolation_score = dists.empty() ? 1.0f
                                           : sum / static_cast<float>(dists.size());
        }
    }

    std::vector<uint64_t> sizes;
    sizes.reserve(n);
    for (const auto& e : embeddings_) sizes.push_back(e.genome_size);
    std::nth_element(sizes.begin(), sizes.begin() + n / 2, sizes.end());
    float median_sz = static_cast<float>(sizes[n / 2]);
    if (median_sz == 0.0f) median_sz = 1.0f;

    std::sort(embeddings_.begin(), embeddings_.end(),
              [median_sz](const auto& a, const auto& b) {
                  float la = std::sqrt(static_cast<float>(a.genome_size) / median_sz);
                  float lb = std::sqrt(static_cast<float>(b.genome_size) / median_sz);
                  return a.isolation_score * (a.quality_score / 100.0f) * la >
                         b.isolation_score * (b.quality_score / 100.0f) * lb;
              });

    for (size_t i = 0; i < n; ++i) {
        store_.genome_ids[i]       = embeddings_[i].genome_id;
        store_.isolation_scores[i] = embeddings_[i].isolation_score;
        store_.quality_scores[i]   = embeddings_[i].quality_score;
        store_.genome_sizes[i]     = embeddings_[i].genome_size;
        store_.paths[i]            = embeddings_[i].path;
        std::copy(embeddings_[i].vector.begin(), embeddings_[i].vector.end(), store_.row(i));
    }
}

void GeodesicDerep::calibrate(const std::vector<std::filesystem::path>& sample_genomes) {
    if (sample_genomes.size() < 20) {
        spdlog::warn("GEODESIC: insufficient genomes for calibration ({})", sample_genomes.size());
        return;
    }

    if (is_verbose()) spdlog::info("GEODESIC: calibrating ANI bounds on {} sample pairs", cfg_.calibration_samples);

    // Select random pairs
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<size_t> dist(0, sample_genomes.size() - 1);

    std::vector<std::pair<double, double>> samples;
    samples.reserve(cfg_.calibration_samples);

    for (int i = 0; i < cfg_.calibration_samples; ++i) {
        size_t a = dist(rng);
        size_t b = dist(rng);
        if (a == b) continue;

        // Compute embedding distance
        auto emb_a = embed_genome(sample_genomes[a], a);
        auto emb_b = embed_genome(sample_genomes[b], b);
        double emb_dist = angular_distance(emb_a.vector, emb_b.vector);

        // Compute exact ANI
        double ani = compute_exact_ani(sample_genomes[a], sample_genomes[b]);

        samples.emplace_back(emb_dist, ani);

        if ((i + 1) % 100 == 0) {
            if (is_verbose()) spdlog::info("GEODESIC: calibration progress {}/{}", i + 1, cfg_.calibration_samples);
        }
    }

    calibrator_.fit(samples);
}

double GeodesicDerep::compute_exact_ani(const std::filesystem::path& a,
                                        const std::filesystem::path& b) {
    ++skani_calls_;

    // Use skani for exact ANI
    // Parse field 3 (ANI) from skani dist output
    std::string cmd = "skani dist -q \"" + a.string() + "\" -r \"" + b.string() +
                      "\" 2>/dev/null | tail -1 | cut -f3";

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return 0.0;

    char buffer[128];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    pclose(pipe);

    try {
        return std::stod(result) / 100.0;  // Convert percentage to fraction
    } catch (...) {
        return 0.0;
    }
}

std::vector<uint64_t> GeodesicDerep::find_candidate_covers(
    uint64_t query_id,
    const std::vector<uint64_t>& current_reps) {

    if (current_reps.empty()) return {};

    const auto& query = embeddings_[query_id];

    // Create filter set for reps only
    std::unordered_set<uint64_t> rep_set(current_reps.begin(), current_reps.end());

    // Find safe radius where ANI_upper < threshold
    float safe_radius = static_cast<float>(calibrator_.inverse_upper(cfg_.ani_threshold));

    // Search within safe radius
    auto neighbors = index_->search_radius(query.vector, safe_radius * 1.2f, &rep_set);

    std::vector<uint64_t> candidates;
    for (const auto& [rep_id, dist] : neighbors) {
        auto bounds = calibrator_.predict(dist);

        if (bounds.lower >= cfg_.ani_threshold) {
            // Certified redundant! Return immediately
            ++certified_redundant_;
            return {rep_id};
        }

        if (bounds.upper >= cfg_.ani_threshold) {
            // Uncertain: need verification
            candidates.push_back(rep_id);
        } else {
            // Certified non-redundant (upper < threshold)
            ++certified_unique_;
        }
    }

    return candidates;
}

void GeodesicDerep::exclude_from_reps(const std::unordered_set<std::string>& paths) {
    for (size_t i = 0; i < store_.n; ++i) {
        if (paths.count(store_.paths[i].string()))
            store_.quality_scores[i] = 0.0f;
    }
}

std::vector<SimilarityEdge> GeodesicDerep::select_representatives() {
    if (is_verbose()) spdlog::info("GEODESIC: FPS + Electrostatic Refinement (SIMD+parallel optimized)");

    std::vector<uint64_t> representatives;
    std::vector<SimilarityEdge> edges;
    size_t n = store_.n;
    size_t dim = store_.dim;

    // Track max similarity to any representative (higher = closer = more redundant)
    // Using similarity instead of distance avoids acos in the hot loop
    std::vector<float> max_sim_to_rep(n, -1.0f);
    std::vector<uint64_t> nearest_rep(n, UINT64_MAX);

    // Stopping criteria (converted to similarity: sim = 1 - dist for small angles)
    // diversity_threshold in distance ≈ (1 - diversity_threshold) in similarity
    const float diversity_sim_threshold = 1.0f - cfg_.diversity_threshold;
    const float min_rep_sim = 1.0f - cfg_.min_rep_distance;

    // Precompute length factors using SoA
    std::vector<float> length_factors(n);
    std::vector<uint64_t> sizes(store_.genome_sizes.begin(), store_.genome_sizes.end());
    std::nth_element(sizes.begin(), sizes.begin() + n / 2, sizes.end());
    double median_size = static_cast<double>(sizes[n / 2]);

    for (size_t i = 0; i < n; ++i) {
        length_factors[i] = static_cast<float>(
            std::sqrt(static_cast<double>(store_.genome_sizes[i]) / median_size));
    }

    // Phase 1: Start with most isolated genome (skip excluded/contaminated quality=0)
    size_t first_idx = 0;
    while (first_idx < n && store_.quality_scores[first_idx] == 0.0f)
        ++first_idx;
    if (first_idx == n) first_idx = 0;  // fallback: all excluded (shouldn't happen)
    uint64_t first_rep = store_.genome_ids[first_idx];
    representatives.push_back(first_rep);
    if (is_verbose()) {
        spdlog::info("GEODESIC: First representative (most isolated): genome_{} (isolation={:.4f})",
                     first_rep, store_.isolation_scores[first_idx]);
    }

    // Update similarities to first rep using SIMD
    const float* first_vec = store_.row(first_idx);
#if GEODESIC_USE_OMP
    #pragma omp parallel for
#endif
    for (size_t i = 0; i < n; ++i) {
        float sim = dot_product_simd(first_vec, store_.row(i), dim);
        max_sim_to_rep[i] = sim;
        nearest_rep[i] = first_rep;
    }
    max_sim_to_rep[first_idx] = 1.0f;  // Self-similarity

    // Phase 2: Iteratively add farthest point (maximizing diversity × quality × length)
    while (true) {
        // Find genome with maximum fitness using parallel reduction
        uint64_t farthest_idx = 0;
        float max_fitness = 0.0f;
        float best_sim = 1.0f;

#if GEODESIC_USE_OMP
        #pragma omp parallel
        {
            uint64_t local_best_idx = 0;
            float local_max_fitness = 0.0f;
            float local_best_sim = 1.0f;

            #pragma omp for nowait
            for (size_t i = 0; i < n; ++i) {
                // Distance proxy: 1 - similarity (monotonic with angular distance)
                float dist_proxy = 1.0f - max_sim_to_rep[i];
                float quality_factor = store_.quality_scores[i] / 100.0f;
                float fitness = dist_proxy * quality_factor * length_factors[i];

                if (fitness > local_max_fitness) {
                    local_max_fitness = fitness;
                    local_best_sim = max_sim_to_rep[i];
                    local_best_idx = i;
                }
            }

            #pragma omp critical
            {
                if (local_max_fitness > max_fitness) {
                    max_fitness = local_max_fitness;
                    best_sim = local_best_sim;
                    farthest_idx = local_best_idx;
                }
            }
        }
#else
        for (size_t i = 0; i < n; ++i) {
            float dist_proxy = 1.0f - max_sim_to_rep[i];
            float quality_factor = store_.quality_scores[i] / 100.0f;
            float fitness = dist_proxy * quality_factor * length_factors[i];

            if (fitness > max_fitness) {
                max_fitness = fitness;
                best_sim = max_sim_to_rep[i];
                farthest_idx = i;
            }
        }
#endif

        // Convert similarity back to angular distance for threshold comparison
        // This ensures compatibility with calibrated thresholds
        float clamped_sim = std::max(-1.0f, std::min(1.0f, best_sim));
        float best_dist = std::acos(clamped_sim) / static_cast<float>(M_PI);

        // Check stopping criterion using angular distance
        if (best_dist < cfg_.diversity_threshold) {
            if (is_verbose()) spdlog::info("GEODESIC: Diversity saturated (max_dist={:.4f} < threshold={:.4f})",
                         best_dist, cfg_.diversity_threshold);
            break;
        }

        // Add farthest point as new representative
        uint64_t farthest_id = store_.genome_ids[farthest_idx];
        representatives.push_back(farthest_id);

        if (is_verbose()) {
            if (is_verbose()) spdlog::info("GEODESIC: Rep #{}: genome_{} (dist={:.4f}, qual={:.1f}, len={:.2f}Mb, fitness={:.4f})",
                         representatives.size(), farthest_idx, best_dist,
                         store_.quality_scores[farthest_idx],
                         store_.genome_sizes[farthest_idx] / 1e6, max_fitness);
        }

        // Update similarities using SIMD + parallel
        const float* farthest_vec = store_.row(farthest_idx);
#if GEODESIC_USE_OMP
        #pragma omp parallel for
#endif
        for (size_t i = 0; i < n; ++i) {
            float sim = dot_product_simd(farthest_vec, store_.row(i), dim);
            if (sim > max_sim_to_rep[i]) {
                max_sim_to_rep[i] = sim;
                nearest_rep[i] = farthest_id;
            }
        }
        max_sim_to_rep[farthest_idx] = 1.0f;
    }

    // Phase 3: Electrostatic coalescence - merge close reps using HNSW-accelerated lookup
    // Like charged particles that merge when too close
    if (is_verbose()) spdlog::info("GEODESIC: Electrostatic refinement ({} candidates, HNSW-accelerated)",
                 representatives.size());

    // Build rep index for fast lookup
    std::unordered_map<uint64_t, size_t> rep_to_idx;
    for (size_t i = 0; i < representatives.size(); ++i) {
        rep_to_idx[representatives[i]] = i;
    }

    // Union-Find for efficient merging
    std::vector<size_t> parent(representatives.size());
    std::iota(parent.begin(), parent.end(), 0);

    std::function<size_t(size_t)> find = [&](size_t x) -> size_t {
        if (parent[x] != x) parent[x] = find(parent[x]);
        return parent[x];
    };

    auto unite = [&](size_t x, size_t y) {
        size_t px = find(x), py = find(y);
        if (px != py) parent[px] = py;
    };

    // For each rep, use HNSW to find close neighbors (much faster than O(r²))
    std::unordered_set<uint64_t> rep_set(representatives.begin(), representatives.end());
    const size_t k_neighbors = std::min(representatives.size(), size_t{64});

    for (size_t i = 0; i < representatives.size(); ++i) {
        uint64_t rep_i = representatives[i];

        // HNSW KNN query (returns closest neighbors)
        auto neighbors = index_->search(embeddings_[rep_i].vector, k_neighbors, &rep_set);

        for (const auto& [neighbor_id, dist] : neighbors) {
            if (neighbor_id == rep_i) continue;

            // Convert angular distance to check merge threshold
            if (dist < cfg_.min_rep_distance) {
                auto it = rep_to_idx.find(neighbor_id);
                if (it != rep_to_idx.end()) {
                    unite(i, it->second);
                    spdlog::debug("GEODESIC: Merged genome_{} into genome_{} (dist={:.4f})",
                                 neighbor_id, rep_i, dist);
                }
            }
        }
    }

    // Collect one representative per component (prefer lower index = selected earlier)
    std::vector<uint64_t> refined_reps;
    std::unordered_set<size_t> seen_roots;

    for (size_t i = 0; i < representatives.size(); ++i) {
        size_t root = find(i);
        if (seen_roots.find(root) == seen_roots.end()) {
            seen_roots.insert(root);
            refined_reps.push_back(representatives[i]);
        }
    }

    size_t merged_count = representatives.size() - refined_reps.size();
    if (merged_count > 0) {
        if (is_verbose()) spdlog::info("GEODESIC: Electrostatic coalescence removed {} redundant reps ({} → {})",
                     merged_count, representatives.size(), refined_reps.size());
        representatives = std::move(refined_reps);

        // Recompute nearest_rep for the refined set using parallel SIMD
        rep_set.clear();
        rep_set.insert(representatives.begin(), representatives.end());

#if GEODESIC_USE_OMP
        #pragma omp parallel for
#endif
        for (size_t i = 0; i < n; ++i) {
            if (rep_set.count(store_.genome_ids[i])) {
                max_sim_to_rep[i] = 1.0f;
                nearest_rep[i] = store_.genome_ids[i];
            } else {
                float max_sim = -1.0f;
                uint64_t best_rep = representatives[0];
                const float* vec_i = store_.row(i);

                for (uint64_t r : representatives) {
                    float sim = dot_product_simd(vec_i, store_.row(r), dim);
                    if (sim > max_sim) {
                        max_sim = sim;
                        best_rep = r;
                    }
                }
                max_sim_to_rep[i] = max_sim;
                nearest_rep[i] = best_rep;
            }
        }
    } else {
        representatives = std::move(refined_reps);
    }

    // Phase 4: Build edges (genome → nearest representative)
    // weight_raw = refined Jaccard estimate J ≈ E[dot(u,v)].
    // ANI = (2J/(1+J))^(1/k) — calibration-free, derived from Mash formula.
    std::unordered_set<uint64_t> final_rep_set(representatives.begin(), representatives.end());

    // Build O(1) lookup: genome_id → index (same index into embeddings_)
    std::unordered_map<uint64_t, size_t> id_to_idx;
    id_to_idx.reserve(n);
    for (size_t r = 0; r < n; ++r) id_to_idx[store_.genome_ids[r]] = r;

    // Precompute path strings to avoid repeated .string() conversions
    std::vector<std::string> path_strings(n);
    for (size_t i = 0; i < n; ++i) path_strings[i] = store_.paths[i].string();

    // OPH refinement: for pairs whose 256-dim estimate is near the threshold,
    // refine J via exact bin-counting. SNR of direct OPH comparison with M=10000:
    //   at 90% ANI (J=0.065): SNR=26  vs  CountSketch@256: SNR=1
    //   at 95% ANI (J=0.212): SNR=50  vs  CountSketch@256: SNR=3.5
    // Borderline zone ≈ 3-sigma of the CountSketch estimator: ±3/√dim ≈ ±0.19
    const bool has_oph = !embeddings_.empty() && !embeddings_[0].oph_sig.empty();
    const double q_t = std::exp(-cfg_.kmer_size * (1.0 - cfg_.ani_threshold));
    const double J_threshold = q_t / (2.0 - q_t);
    const double J_margin = 3.0 / std::sqrt(static_cast<double>(cfg_.embedding_dim));

    for (size_t i = 0; i < n; ++i) {
        if (final_rep_set.count(store_.genome_ids[i])) continue;

        auto it = id_to_idx.find(nearest_rep[i]);
        if (it == id_to_idx.end()) {
            spdlog::warn("nearest_rep id {} not found in store for genome {}, skipping edge",
                         nearest_rep[i], path_strings[i]);
            continue;
        }
        const size_t rep_idx = it->second;

        // Start with the fast CountSketch estimate
        double J_est = static_cast<double>(max_sim_to_rep[i]);

        // Refine borderline pairs with exact OPH bin-counting
        if (has_oph && std::abs(J_est - J_threshold) <= J_margin) {
            J_est = refine_jaccard(embeddings_[i].oph_sig, embeddings_[rep_idx].oph_sig);
        }

        SimilarityEdge edge;
        edge.source   = path_strings[i];
        edge.target   = path_strings[rep_idx];
        edge.weight_raw = static_cast<float>(J_est);
        edge.weight     = static_cast<float>(J_est);
        edge.aln_frac   = 1.0;
        edges.push_back(edge);
    }

    if (is_verbose()) spdlog::info("GEODESIC: DONE - {} diverse representatives from {} genomes (SIMD+parallel, NO skani!)",
                 representatives.size(), n);

    return edges;
}

// ---------------------------------------------------------------------------
// Isolation Forest (Liu, Ting & Zhou 2008) for high-dimensional anomaly detection.
// Anomalous points have shorter average path lengths in random isolation trees.
// Score ∈ (0,1]: score > 0.5 = anomalous, score ≈ 0.5 = normal, < 0.5 = dense inlier.
// ---------------------------------------------------------------------------
struct IForest {
    static constexpr int PSI   = 256;  // subsample size per tree
    static constexpr int T     = 100;  // number of trees
    static constexpr int H_LIM = 8;    // max depth = ceil(log2(PSI))

    struct Node {
        int  split_dim = -1;
        float split_val = 0.0f;
        int  left = -1, right = -1;
        int  size = 0;
    };
    struct Tree { std::vector<Node> nodes; };

    std::vector<Tree> trees;
    int   dim  = 0;
    float c_psi = 1.0f;

    // Expected path length for a BST of n points (Eq. 1 in paper)
    static float c(int m) {
        if (m <= 1) return 0.0f;
        return 2.0f * (std::log(static_cast<float>(m - 1)) + 0.5772156649f)
               - 2.0f * (m - 1) / static_cast<float>(m);
    }

    // Build one tree on a subsample of row-major float data (n rows, d cols)
    void build_tree(Tree& tree, const float* data, std::vector<int>& pts,
                    int node_idx, int depth, std::mt19937_64& rng) {
        tree.nodes[node_idx].size = static_cast<int>(pts.size());
        if (static_cast<int>(pts.size()) <= 1 || depth >= H_LIM) return;

        // Random split dimension and value
        int d = static_cast<int>(rng() % static_cast<uint64_t>(dim));
        float mn = data[static_cast<size_t>(pts[0]) * dim + d];
        float mx = mn;
        for (int p : pts) {
            float v = data[static_cast<size_t>(p) * dim + d];
            mn = std::min(mn, v);
            mx = std::max(mx, v);
        }
        if (mn >= mx) return;  // all same value — treat as leaf

        float sv = std::uniform_real_distribution<float>(mn, mx)(rng);

        std::vector<int> lpts, rpts;
        for (int p : pts)
            (data[static_cast<size_t>(p) * dim + d] < sv ? lpts : rpts).push_back(p);

        // Allocate child nodes before recursing (avoids dangling refs after push_back)
        int l_idx = static_cast<int>(tree.nodes.size()); tree.nodes.push_back({});
        int r_idx = static_cast<int>(tree.nodes.size()); tree.nodes.push_back({});
        tree.nodes[node_idx].split_dim = d;
        tree.nodes[node_idx].split_val = sv;
        tree.nodes[node_idx].left  = l_idx;
        tree.nodes[node_idx].right = r_idx;

        build_tree(tree, data, lpts, l_idx, depth + 1, rng);
        build_tree(tree, data, rpts, r_idx, depth + 1, rng);
    }

    void fit(const float* data, int n, int dimension, uint64_t seed = 42) {
        dim   = dimension;
        int psi = std::min(n, PSI);
        c_psi = c(psi);
        trees.resize(T);
        std::mt19937_64 rng(seed);

        // Allocate once outside the loop — reused across all T trees
        std::vector<int> all(n);
        std::iota(all.begin(), all.end(), 0);

        for (int t = 0; t < T; ++t) {
            // Fisher-Yates subsample (partial shuffle, only first psi elements)
            for (int i = 0; i < psi; ++i) {
                int j = i + static_cast<int>(rng() % static_cast<uint64_t>(n - i));
                std::swap(all[i], all[j]);
            }
            std::vector<int> sub(all.begin(), all.begin() + psi);

            // Each tree has at most 2*psi nodes; reserve to prevent reallocation
            auto& tr = trees[t];
            tr.nodes.reserve(2 * psi + 4);
            tr.nodes.push_back({});  // root at index 0
            build_tree(tr, data, sub, 0, 0, rng);
        }
    }

    float score(const float* x) const {
        float total = 0.0f;
        for (const auto& tr : trees) {
            int idx = 0, depth = 0;
            while (true) {
                const Node& nd = tr.nodes[idx];
                if (nd.split_dim == -1 || nd.left == -1) {
                    total += static_cast<float>(depth) + c(nd.size);
                    break;
                }
                idx = (x[nd.split_dim] < nd.split_val) ? nd.left : nd.right;
                ++depth;
            }
        }
        return std::pow(2.0f, -(total / static_cast<float>(T)) / c_psi);
    }
};

std::vector<GeodesicDerep::ContaminationCandidate>
GeodesicDerep::detect_contamination_candidates(float z_threshold) const {
    if (embeddings_.size() <= SMALL_N_THRESHOLD) return {};

    size_t n   = embeddings_.size();
    size_t dim = static_cast<size_t>(cfg_.embedding_dim);

    // --- Fit Isolation Forest on embeddings (store_ is row-major, aligned) ---
    IForest iforest;
    iforest.fit(store_.data.data(), static_cast<int>(n), static_cast<int>(dim));

    // Score every genome: score > threshold → anomaly candidate
    // z_threshold=2.0 → if_threshold=0.55 (standard anomaly cutoff)
    // z_threshold=3.0 → 0.575, z_threshold=1.0 → 0.525
    const float if_threshold = 0.5f + 0.025f * z_threshold;

    // Centroid for informational storage (centroid_distance field)
    std::vector<float> centroid(dim, 0.0f);
    for (const auto& emb : embeddings_)
        for (size_t d = 0; d < dim; ++d)
            centroid[d] += emb.vector[d];
    float cn = static_cast<float>(n);
    for (float& v : centroid) v /= cn;
    float cnorm = dot_product_simd(centroid.data(), centroid.data(), dim);
    cnorm = std::sqrt(cnorm);
    if (cnorm > 1e-10f) for (float& v : centroid) v /= cnorm;

    // Score all genomes in parallel
    std::vector<float> scores(n);
    {
        size_t nw = std::min(static_cast<size_t>(cfg_.threads), n);
        size_t chunk = (n + nw - 1) / nw;
        std::vector<std::future<void>> futs;
        for (size_t t = 0; t < nw; ++t) {
            size_t lo = t * chunk, hi = std::min(lo + chunk, n);
            if (lo >= hi) break;
            futs.push_back(std::async(std::launch::async, [&, lo, hi]() {
                for (size_t i = lo; i < hi; ++i)
                    scores[i] = iforest.score(store_.data.data() + i * dim);
            }));
        }
        for (auto& f : futs) f.get();
    }

    std::vector<ContaminationCandidate> candidates;
    for (size_t i = 0; i < n; ++i) {
        if (scores[i] > if_threshold) {
            ContaminationCandidate c;
            c.genome_id         = embeddings_[i].genome_id;
            c.centroid_distance = angular_distance(embeddings_[i].vector, centroid);
            c.isolation_score   = embeddings_[i].isolation_score;
            c.anomaly_score     = scores[i];
            c.path              = embeddings_[i].path;
            candidates.push_back(c);
        }
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.anomaly_score > b.anomaly_score; });

    if (!candidates.empty())
        if (is_verbose())
            spdlog::info("GEODESIC: Isolation Forest detected {} candidates (score > {:.3f})",
                         candidates.size(), if_threshold);

    return candidates;
}

GeodesicDerep::DiversityMetrics GeodesicDerep::compute_diversity_metrics(
    const std::vector<uint64_t>& representative_ids) const {

    DiversityMetrics metrics;

    if (embeddings_.empty() || representative_ids.empty()) {
        return metrics;
    }

    // Build genome_id -> embedding index map
    std::unordered_map<uint64_t, size_t> id_to_idx;
    for (size_t i = 0; i < embeddings_.size(); ++i) {
        id_to_idx[embeddings_[i].genome_id] = i;
    }

    // Get representative indices
    std::vector<size_t> rep_indices;
    rep_indices.reserve(representative_ids.size());
    for (uint64_t id : representative_ids) {
        auto it = id_to_idx.find(id);
        if (it != id_to_idx.end()) {
            rep_indices.push_back(it->second);
        }
    }

    if (rep_indices.empty()) {
        return metrics;
    }

    // Calibration-free ANI from OPH+CountSketch angular distance.
    // E[dot(u,v)] = J(A,B), Mash chain: ANI = (2J/(1+J))^(1/k), k=21
    auto dist_to_ani = [this](double dist) -> double {
        if (dist <= 0.0) return 1.0;
        if (dist >= 0.5) return 0.7;
        double cos_sim = std::cos(dist * M_PI);
        if (cos_sim <= 0.0) return 0.7;
        double ratio = 2.0 * cos_sim / (1.0 + cos_sim);
        double ani = std::pow(ratio, 1.0 / cfg_.kmer_size);
        return std::max(0.7, std::min(1.0, ani));
    };

    // --- Coverage: for each genome, find distance to nearest representative ---
    std::vector<double> coverage_dists(embeddings_.size());
    double coverage_sum = 0.0;
    double coverage_min = std::numeric_limits<double>::max();
    double coverage_max = 0.0;
    size_t n_valid = 0;

#if GEODESIC_USE_OMP
    #pragma omp parallel for reduction(+:coverage_sum) reduction(min:coverage_min) reduction(max:coverage_max)
#endif
    for (size_t i = 0; i < embeddings_.size(); ++i) {
        const float* query = store_.row(i);
        // Skip zero-norm embeddings (failed sketch: empty/corrupt FASTA)
        if (dot_product_simd(query, query, cfg_.embedding_dim) < 0.01f) {
            coverage_dists[i] = 0.5;  // sentinel; excluded from stats
            continue;
        }
        ++n_valid;

        double min_dist = std::numeric_limits<double>::max();
        for (size_t rep_idx : rep_indices) {
            const float* rep = store_.row(rep_idx);
            float sim = dot_product_simd(query, rep, cfg_.embedding_dim);
            float clamped = std::max(-1.0f, std::min(1.0f, sim));
            float dist = std::acos(clamped) / static_cast<float>(M_PI);
            min_dist = std::min(min_dist, static_cast<double>(dist));
        }

        coverage_dists[i] = min_dist;
        coverage_sum += min_dist;
        coverage_min = std::min(coverage_min, min_dist);
        coverage_max = std::max(coverage_max, min_dist);
    }

    metrics.coverage_mean_dist = n_valid > 0 ? coverage_sum / static_cast<double>(n_valid) : 0.0;

    // Percentile-based coverage: sort only valid distances (exclude sentinel 0.5)
    {
        std::vector<double> valid_dists;
        valid_dists.reserve(n_valid);
        for (size_t i = 0; i < embeddings_.size(); ++i) {
            if (coverage_dists[i] < 0.5 - 1e-6)
                valid_dists.push_back(coverage_dists[i]);
        }
        if (!valid_dists.empty()) {
            std::sort(valid_dists.begin(), valid_dists.end());
            size_t nv = valid_dists.size();
            metrics.coverage_p5_dist  = valid_dists[nv *  5 / 100];
            metrics.coverage_p95_dist = valid_dists[nv * 95 / 100];
        }
    }

    // Count genomes below ANI thresholds (valid only)
    for (size_t i = 0; i < embeddings_.size(); ++i) {
        if (coverage_dists[i] >= 0.5 - 1e-6) continue;
        double ani = dist_to_ani(coverage_dists[i]);
        if (ani < 0.99) metrics.coverage_below_99++;
        if (ani < 0.98) metrics.coverage_below_98++;
        if (ani < 0.97) metrics.coverage_below_97++;
        if (ani < 0.95) metrics.coverage_below_95++;
    }

    // --- Diversity: pairwise distances among representatives ---
    // Sample at most 2000 rep pairs to avoid O(n²) for large rep sets
    if (rep_indices.size() >= 2) {
        std::vector<double> pair_dists;
        const size_t max_pairs = 2000;
        std::mt19937_64 rng(12345);

        if (rep_indices.size() * (rep_indices.size() - 1) / 2 <= max_pairs) {
            // Enumerate all pairs
            for (size_t i = 0; i < rep_indices.size(); ++i) {
                for (size_t j = i + 1; j < rep_indices.size(); ++j) {
                    const float* a = store_.row(rep_indices[i]);
                    const float* b = store_.row(rep_indices[j]);
                    float sim = dot_product_simd(a, b, cfg_.embedding_dim);
                    double dist = std::acos(std::max(-1.0f, std::min(1.0f, sim))) / M_PI;
                    pair_dists.push_back(dist);
                }
            }
        } else {
            // Random sample
            std::uniform_int_distribution<size_t> ud(0, rep_indices.size() - 1);
            pair_dists.reserve(max_pairs);
            for (size_t s = 0; s < max_pairs * 4 && pair_dists.size() < max_pairs; ++s) {
                size_t i = ud(rng), j = ud(rng);
                if (i == j) continue;
                const float* a = store_.row(rep_indices[i]);
                const float* b = store_.row(rep_indices[j]);
                float sim = dot_product_simd(a, b, cfg_.embedding_dim);
                double dist = std::acos(std::max(-1.0f, std::min(1.0f, sim))) / M_PI;
                pair_dists.push_back(dist);
            }
        }

        if (!pair_dists.empty()) {
            std::sort(pair_dists.begin(), pair_dists.end());
            double sum = 0.0;
            for (double d : pair_dists) sum += d;
            size_t np = pair_dists.size();
            metrics.diversity_mean_dist = sum / np;
            metrics.diversity_p5_dist   = pair_dists[np *  5 / 100];
            metrics.diversity_p95_dist  = pair_dists[np * 95 / 100];
            metrics.diversity_n_pairs   = static_cast<int>(np);
        }
    }

    return metrics;
}

// ============================================================================
// Incremental embedding support
// ============================================================================

size_t GeodesicDerep::build_index_incremental(
    const std::vector<std::filesystem::path>& genomes,
    db::EmbeddingStore& store,
    const std::string& taxonomy,
    const std::unordered_map<std::string, double>& quality_scores) {

    if (genomes.empty()) return 0;

    // Build path → accession map (extract from filename)
    path_to_accession_.clear();
    std::vector<std::string> all_accessions;
    all_accessions.reserve(genomes.size());
    for (const auto& p : genomes) {
        std::string acc = p.stem().string();
        // Remove _genomic suffix if present
        if (acc.size() > 8 && acc.substr(acc.size() - 8) == "_genomic") {
            acc = acc.substr(0, acc.size() - 8);
        }
        path_to_accession_[p.string()] = acc;
        all_accessions.push_back(acc);
    }

    // Find which genomes are already embedded
    auto embedded_set = store.get_embedded_set(taxonomy);
    std::vector<std::filesystem::path> missing_genomes;
    missing_genomes.reserve(genomes.size());

    for (const auto& p : genomes) {
        const auto& acc = path_to_accession_[p.string()];
        if (embedded_set.find(acc) == embedded_set.end()) {
            missing_genomes.push_back(p);
        }
    }

    if (is_verbose()) spdlog::info("GEODESIC: {} genomes total, {} already embedded, {} to embed",
                 genomes.size(), embedded_set.size(), missing_genomes.size());

    // Load existing embeddings from store
    auto existing = store.load_embeddings(taxonomy);

    // Embed missing genomes
    size_t newly_embedded = 0;
    if (!missing_genomes.empty()) {
        if (is_verbose()) spdlog::info("GEODESIC: embedding {} new genomes (dim={}, k={}, threads={})",
                     missing_genomes.size(), cfg_.embedding_dim, cfg_.kmer_size, cfg_.threads);

#if GEODESIC_USE_AVX2
        if (is_verbose()) spdlog::info("GEODESIC: AVX2 SIMD enabled");
#endif
#if GEODESIC_USE_OMP
        if (is_verbose()) spdlog::info("GEODESIC: OpenMP parallel enabled ({} threads)", cfg_.threads);
#endif

        std::vector<GenomeEmbedding> new_embeddings(missing_genomes.size());
        ProgressCounter embed_progress(missing_genomes.size(), "GEODESIC: embedding");

#if GEODESIC_USE_OMP
        omp_set_num_threads(cfg_.threads);
        #pragma omp parallel for schedule(dynamic)
#endif
        for (size_t i = 0; i < missing_genomes.size(); ++i) {
            new_embeddings[i] = embed_genome(missing_genomes[i], existing.size() + i);

            // Add quality score
            auto qit = quality_scores.find(missing_genomes[i].string());
            if (qit != quality_scores.end()) {
                new_embeddings[i].quality_score = static_cast<float>(qit->second);
            }

            embed_progress.increment();
        }
        embed_progress.finish();

        // Save new embeddings to store
        std::vector<db::GenomeEmbedding> store_embeddings;
        store_embeddings.reserve(new_embeddings.size());
        for (const auto& emb : new_embeddings) {
            db::GenomeEmbedding se;
            se.accession = path_to_accession_[emb.path.string()];
            se.taxonomy = taxonomy;
            se.file_path = emb.path;
            se.embedding = emb.vector;
            se.isolation_score = emb.isolation_score;
            se.quality_score = emb.quality_score;
            se.genome_size = emb.genome_size;
            store_embeddings.push_back(std::move(se));
        }

        store.insert_embeddings(store_embeddings);
        newly_embedded = new_embeddings.size();

        // Add to internal embeddings list
        for (auto& emb : new_embeddings) {
            embeddings_.push_back(std::move(emb));
        }
    }

    // Convert existing store embeddings to internal format
    for (const auto& se : existing) {
        GenomeEmbedding emb;
        // Find the genome ID - assign based on position
        emb.genome_id = embeddings_.size();
        emb.vector = se.embedding;
        emb.isolation_score = se.isolation_score;
        emb.quality_score = se.quality_score;
        emb.genome_size = se.genome_size;
        emb.path = se.file_path;
        embeddings_.push_back(std::move(emb));
    }

    // Build HNSW index with all embeddings
    size_t n = embeddings_.size();
    if (n == 0) return 0;

    // Resize SoA store
    store_.resize(n, cfg_.embedding_dim);

    // Copy to SoA layout
    for (size_t i = 0; i < n; ++i) {
        std::copy(embeddings_[i].vector.begin(), embeddings_[i].vector.end(), store_.row(i));
        store_.genome_ids[i] = embeddings_[i].genome_id;
        store_.isolation_scores[i] = embeddings_[i].isolation_score;
        store_.quality_scores[i] = embeddings_[i].quality_score;
        store_.genome_sizes[i] = embeddings_[i].genome_size;
        store_.paths[i] = embeddings_[i].path;
    }

    // Build HNSW index (parallel insertion)
    index_ = std::make_unique<HNSWIndex>();
    index_->space = std::make_unique<hnswlib::InnerProductSpace>(cfg_.embedding_dim);
    index_->index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
        index_->space.get(), n, cfg_.hnsw_m, cfg_.hnsw_ef_construction);

    // First point must be added sequentially to initialize the graph
    if (n > 0) {
        index_->index->addPoint(store_.row(0), 0);
    }
    // Remaining points can be added in parallel (hnswlib is thread-safe for addPoint)
#if GEODESIC_USE_OMP
    #pragma omp parallel for schedule(dynamic, 1000)
#endif
    for (size_t i = 1; i < n; ++i) {
        index_->index->addPoint(store_.row(i), i);
    }
    index_->index->setEf(cfg_.hnsw_ef_search);

    if (is_verbose()) spdlog::info("GEODESIC: HNSW index built ({} embeddings, M={}, ef={})",
                 n, cfg_.hnsw_m, cfg_.hnsw_ef_construction);

    // Rebuild store's HNSW index
    store.rebuild_index();

    return newly_embedded;
}

void GeodesicDerep::save_embeddings_to_store(db::EmbeddingStore& store, const std::string& taxonomy) {
    if (embeddings_.empty()) return;

    std::vector<db::GenomeEmbedding> store_embeddings;
    store_embeddings.reserve(embeddings_.size());

    for (const auto& emb : embeddings_) {
        db::GenomeEmbedding se;

        // Get accession from path
        auto it = path_to_accession_.find(emb.path.string());
        if (it != path_to_accession_.end()) {
            se.accession = it->second;
        } else {
            // Extract from filename
            se.accession = emb.path.stem().string();
            if (se.accession.size() > 8 && se.accession.substr(se.accession.size() - 8) == "_genomic") {
                se.accession = se.accession.substr(0, se.accession.size() - 8);
            }
        }

        se.taxonomy = taxonomy;
        se.file_path = emb.path;
        se.embedding = emb.vector;
        se.isolation_score = emb.isolation_score;
        se.quality_score = emb.quality_score;
        se.genome_size = emb.genome_size;
        store_embeddings.push_back(std::move(se));
    }

    store.insert_embeddings(store_embeddings);
    if (is_verbose()) spdlog::info("GEODESIC: Saved {} embeddings to store", store_embeddings.size());
}

} // namespace derep
