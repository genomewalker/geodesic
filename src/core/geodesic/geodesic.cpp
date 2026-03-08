#include "core/geodesic/geodesic.hpp"
#include "core/sketch/minhash.hpp"
#include "core/progress.hpp"
#include "core/logging.hpp"
#include "db/embedding_store.hpp"
#include "io/gz_reader.hpp"
#include <hnswlib/hnswlib.h>
#include <spdlog/spdlog.h>
#include <Eigen/Dense>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <fcntl.h>
#include <fstream>
#include <future>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <thread>
#include <semaphore>
#include <unordered_set>
#include <unistd.h>

// SIMD support detection
#if defined(__AVX2__)
#include <immintrin.h>
#define GEODESIC_USE_AVX2 1
#else
#define GEODESIC_USE_AVX2 0
#endif

#if defined(__FMA__)
#define GEODESIC_USE_FMA 1
#else
#define GEODESIC_USE_FMA 0
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

// AVX2 optimized dot product — 4 independent accumulators to break
// the FMA latency chain (4-cycle latency × 0.5-cycle throughput = 8 FMAs in flight).
#if GEODESIC_USE_AVX2
float dot_product_simd(const float* __restrict__ a, const float* __restrict__ b, size_t dim) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();
    size_t i = 0;

    for (; i + 32 <= dim; i += 32) {
#if GEODESIC_USE_FMA
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),      _mm256_loadu_ps(b + i),      sum0);
        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),   _mm256_loadu_ps(b + i + 8),  sum1);
        sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16),  _mm256_loadu_ps(b + i + 16), sum2);
        sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24),  _mm256_loadu_ps(b + i + 24), sum3);
#else
        sum0 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(a + i),      _mm256_loadu_ps(b + i)),      sum0);
        sum1 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(a + i + 8),  _mm256_loadu_ps(b + i + 8)),  sum1);
        sum2 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16)), sum2);
        sum3 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24)), sum3);
#endif
    }

    // Reduce 4 accumulators to 1
    sum0 = _mm256_add_ps(sum0, sum1);
    sum2 = _mm256_add_ps(sum2, sum3);
    sum0 = _mm256_add_ps(sum0, sum2);

    // Process remaining 8-float blocks
    for (; i + 8 <= dim; i += 8) {
#if GEODESIC_USE_FMA
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), sum0);
#else
        sum0 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)), sum0);
#endif
    }

    // Horizontal sum
    __m128 hi = _mm256_extractf128_ps(sum0, 1);
    __m128 lo = _mm256_castps256_ps128(sum0);
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

// Pointer-based Jaccard estimator for uint16_t OPH signatures.
// Uses AVX2 equality counting when available (~16× faster than scalar).
// b-bit bias correction for 16-bit storage: J_true = (J_raw - 2^-16) / (1 - 2^-16).
static double refine_jaccard_ptr(const uint16_t* __restrict a,
                                  const uint16_t* __restrict b,
                                  size_t m) {
    if (m == 0) return 0.0;
    size_t matches = 0;
#if GEODESIC_USE_AVX2
    const size_t avx_n = m & ~static_cast<size_t>(15);  // round down to multiple of 16
    for (size_t t = 0; t < avx_n; t += 16) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + t));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + t));
        __m256i eq = _mm256_cmpeq_epi16(va, vb);
        // movemask_epi8 yields 1 bit per byte; each uint16_t match → 2 bits set
        matches += static_cast<size_t>(__builtin_popcount(
            static_cast<unsigned>(_mm256_movemask_epi8(eq)))) >> 1;
    }
    for (size_t t = avx_n; t < m; ++t)
        if (a[t] == b[t]) ++matches;
#else
    for (size_t t = 0; t < m; ++t)
        if (a[t] == b[t]) ++matches;
#endif
    const double j_raw = static_cast<double>(matches) / static_cast<double>(m);
    constexpr double inv_2b = 1.0 / 65536.0;
    return std::max(0.0, (j_raw - inv_2b) / (1.0 - inv_2b));
}

// Real HNSW index using hnswlib for O(log n) nearest neighbor search
// Uses inner product space (for normalized vectors, maximizing IP = minimizing angular distance)
struct GeodesicDerep::HNSWIndex {
    size_t dim = 256;  // Set dynamically during build
    std::unique_ptr<hnswlib::InnerProductSpace> space;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index;
    std::vector<GenomeEmbedding>* embeddings;
    int ef_search;

    void build(std::vector<GenomeEmbedding>& embs, int M, int ef_construction, int ef_search_value) {
        embeddings = &embs;
        if (embs.empty()) return;

        ef_search = std::max(1, ef_search_value);
        dim = embs[0].vector.size();
        space = std::make_unique<hnswlib::InnerProductSpace>(dim);
        index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
            space.get(), embs.size(), M, ef_construction);
        index->setEf(ef_search);  // Set once here; never mutated during concurrent search

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

        if (!index || !embeddings || embeddings->empty() || k == 0) return {};

        const size_t total = embeddings->size();
        // When filtering, over-query to ensure we get k valid results
        size_t query_k = std::min(total, k);
        if (filter) query_k = std::min(total, k * 4);

        std::vector<std::pair<uint64_t, float>> out;
        for (;;) {
            auto result = index->searchKnn(query.data(), query_k);

            out.clear();
            out.reserve(result.size());

            while (!result.empty()) {
                auto [ip_dist, label] = result.top();
                result.pop();

                // label is the sequential insertion index, not genome_id
                const size_t idx = static_cast<size_t>(label);
                if (idx >= total) continue;

                const uint64_t genome_id = (*embeddings)[idx].genome_id;
                if (filter && filter->find(genome_id) == filter->end()) continue;

                // Convert IP distance to angular distance
                float inner_product = std::clamp(1.0f - ip_dist, -1.0f, 1.0f);
                float angular_dist = std::acos(inner_product) / static_cast<float>(M_PI);

                out.emplace_back(genome_id, angular_dist);
            }

            std::reverse(out.begin(), out.end());

            // If we got enough results or exhausted the index, stop
            if (!filter || out.size() >= k || query_k == total) break;
            query_k = std::min(total, query_k * 2);
        }

        if (out.size() > k) out.resize(k);
        return out;
    }

    // Radius search - brute force with dot product threshold (no per-element acos)
    std::vector<std::pair<uint64_t, float>> search_radius(
        const std::vector<float>& query,
        float radius,
        const std::unordered_set<uint64_t>* filter = nullptr) const {

        if (!index || !embeddings || embeddings->empty()) return {};
        // Precompute: dist <= radius ↔ dot >= cos(radius * π)
        const float cos_radius = std::cos(radius * static_cast<float>(M_PI));
        const size_t d = query.size();
        std::vector<std::pair<uint64_t, float>> result;
        for (const auto& emb : *embeddings) {
            if (filter && filter->find(emb.genome_id) == filter->end()) continue;
            float dot = dot_product_simd(query.data(), emb.vector.data(), d);
            if (dot >= cos_radius) {
                // Convert to angular distance only for accepted results
                float dist = std::acos(std::clamp(dot, -1.0f, 1.0f)) / static_cast<float>(M_PI);
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

    // Pass 1: fill distance_grid_ so Pass 2 can safely read neighbours
    for (size_t i = 0; i < grid_size; ++i) {
        size_t idx = i * sorted.size() / grid_size;
        distance_grid_[i] = sorted[idx].first;
    }

    // Pass 2: compute ANI curves using fully-populated grid for window width
    for (size_t i = 0; i < grid_size; ++i) {
        size_t idx = i * sorted.size() / grid_size;

        double dist_window;
        if (grid_size == 1) {
            dist_window = 1e-12;
        } else if (i == 0) {
            dist_window = (distance_grid_[1] - distance_grid_[0]) * 0.5;
        } else if (i + 1 == grid_size) {
            dist_window = (distance_grid_[i] - distance_grid_[i - 1]) * 0.5;
        } else {
            dist_window = (distance_grid_[i + 1] - distance_grid_[i - 1]) * 0.25;
        }
        dist_window = std::max(dist_window, 1e-12);

        std::vector<double> nearby_anis;
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
        ani_lower_curve_[i] = std::clamp(ani_lower_curve_[i], 0.0, 1.0);
        ani_upper_curve_[i] = std::clamp(ani_upper_curve_[i], 0.0, 1.0);
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
    double dx = distance_grid_[idx] - distance_grid_[idx - 1];
    if (std::abs(dx) <= 1e-12) {
        return {ani_lower_curve_[idx - 1], ani_upper_curve_[idx - 1]};
    }
    double t = (embedding_distance - distance_grid_[idx - 1]) / dx;

    double lower = ani_lower_curve_[idx - 1] + t * (ani_lower_curve_[idx] - ani_lower_curve_[idx - 1]);
    double upper = ani_upper_curve_[idx - 1] + t * (ani_upper_curve_[idx] - ani_upper_curve_[idx - 1]);

    if (lower > upper) std::swap(lower, upper);
    lower = std::clamp(lower, 0.0, 1.0);
    upper = std::clamp(upper, 0.0, 1.0);

    return {lower, upper};
}

double ANICalibrator::inverse_upper(double target_ani) const {
    if (!fitted_ || distance_grid_.empty()) {
        // Inverse of ANI = (2J/(1+J))^(1/k): J = ANI^k / (2 - ANI^k), k=21
        if (target_ani >= 1.0) return 0.0;
        if (target_ani <= 0.7) return 1.0;
        double ak = std::pow(target_ani, 21.0);
        double cos_sim = std::clamp(ak / (2.0 - ak), -1.0, 1.0);
        return std::acos(cos_sim) / M_PI;
    }

    // Binary search on upper curve
    for (size_t i = 0; i < ani_upper_curve_.size(); ++i) {
        if (ani_upper_curve_[i] < target_ani) {
            if (i == 0) return 0.0;
            double dy = ani_upper_curve_[i - 1] - ani_upper_curve_[i];
            if (std::abs(dy) <= 1e-12) return distance_grid_[i];
            double t = (target_ani - ani_upper_curve_[i]) / dy;
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
        if (adjusted_ani <= 0.7) return 1.0;
        double ak = std::pow(adjusted_ani, 21.0);
        double cos_sim = std::clamp(ak / (2.0 - ak), -1.0, 1.0);
        return std::acos(cos_sim) / M_PI;
    }

    for (size_t i = 0; i < ani_lower_curve_.size(); ++i) {
        if (ani_lower_curve_[i] < target_ani) {
            if (i == 0) return 0.0;
            double dy = ani_lower_curve_[i - 1] - ani_lower_curve_[i];
            if (std::abs(dy) <= 1e-12) return distance_grid_[i];
            double t = (target_ani - ani_lower_curve_[i]) / dy;
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
        MinHasher hasher({ .kmer_size = kmer_size, .sketch_size = sketch_size,
                           .syncmer_s = 0, .seed = 42 });

        // OPH+CountSketch projection: E[dot(u,v)] = J(A,B)
        auto cs_project = [&](const std::vector<uint16_t>& oph_sig) -> std::vector<float> {
            auto mix = [](uint64_t x) -> uint64_t {
                x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
                x ^= x >> 27; x *= 0x94d049bb133111ebULL;
                x ^= x >> 31; return x;
            };
            constexpr uint64_t P1 = 0x9e3779b97f4a7c15ULL;
            constexpr uint64_t P2 = 0x6c62272e07bb0142ULL;
            const uint64_t dim64 = static_cast<uint64_t>(embedding_dim);
            std::vector<float> emb(embedding_dim, 0.0f);
            for (int t = 0; t < static_cast<int>(oph_sig.size()); ++t) {
                uint64_t h1 = mix(static_cast<uint64_t>(oph_sig[t]) + static_cast<uint64_t>(t) * P1);
                uint64_t h2 = mix(static_cast<uint64_t>(oph_sig[t]) + static_cast<uint64_t>(t) * P2);
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
        const size_t requested = (threads > 0) ? static_cast<size_t>(threads) : 1;
        size_t nw = std::min(requested, indices.size());
        if (nw == 0) return cache;
        size_t chunk = (indices.size() + nw - 1) / nw;
        std::vector<std::future<void>> futs;
        for (size_t t = 0; t < nw; ++t) {
            size_t lo = t * chunk, hi = std::min(lo + chunk, indices.size());
            if (lo >= hi) break;
            futs.push_back(std::async(std::launch::async, [&, lo, hi]() {
                for (size_t pos = lo; pos < hi; ++pos) {
                    size_t idx = indices[pos];
                    auto oph = hasher.sketch_oph(genomes[idx], sketch_size);
                    std::vector<uint16_t> sig16(oph.signature.size());
                    for (size_t t = 0; t < sig16.size(); ++t) sig16[t] = static_cast<uint16_t>(oph.signature[t]);
                    auto emb = cs_project(sig16);
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

    // Choose tier from the estimated ANI of the most divergent sampled pair (P95).
    // Using ANI rather than raw distance spread avoids the spread being dominated
    // by within-serovar clonal structure while still placing species like S. enterica
    // (97–99% inter-serovar ANI) in the correct k=21 tier.
    auto dist_to_ani0 = [k0](double d) -> double {
        double c = std::cos(M_PI * d);
        if (c <= 0.0) return 0.0;
        double r = 2.0 * c / (1.0 + c);
        return std::pow(std::max(1e-15, r), 1.0 / k0) * 100.0;
    };
    double ani_p95_sample = dist_to_ani0(p95_0);  // ANI of most divergent sampled pair

    int kmer, sketch_size, embedding_dim;
    if (ani_p95_sample >= 99.0) {
        // Very clonal (all pairs >99% ANI): high k for within-clone discrimination
        kmer = 31; sketch_size = 20000; embedding_dim = 512;
    } else if (ani_p95_sample >= 95.0) {
        // Typical species (95–99% ANI): standard resolution
        kmer = 21; sketch_size = 10000; embedding_dim = 256;
    } else {
        // Very diverse or cross-genus (<95% ANI in sample)
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
    // Auto-calibrated ANI range (used for logging / min_rep_distance only)
    params.ani_threshold = std::max(0.90, std::min(0.9999,
        (params.ani_min + 0.5 * params.ani_spread) / 100.0));

    params.kmer_size        = kmer;
    params.embedding_dim    = embedding_dim;
    params.sketch_size      = sketch_size;

    // diversity_threshold: stored in params but overridden at call site with user's ani_threshold.
    {
        double ak = std::pow(params.ani_threshold, static_cast<double>(kmer));
        double j = std::clamp(ak / (2.0 - ak), 0.0, 1.0);
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
    , runtime_dim_(cfg_.embedding_dim)
    , index_(std::make_unique<HNSWIndex>()) {
}

GeodesicDerep::~GeodesicDerep() {
    if (async_save_future_.valid()) {
        try { async_save_future_.get(); } catch (const std::exception& e) {
            spdlog::error("GEODESIC: async save failed: {}", e.what());
        }
    }
}

float GeodesicDerep::angular_distance(const std::vector<float>& a,
                                       const std::vector<float>& b) {
    const size_t d = std::min(a.size(), b.size());
    if (a.size() != b.size()) spdlog::warn("angular_distance: dim mismatch {} vs {}", a.size(), b.size());
    // Cosine similarity → angular distance
    float dot = dot_product_simd(a.data(), b.data(), d);
    // Clamp for numerical stability
    dot = std::max(-1.0f, std::min(1.0f, dot));
    // Angular distance in [0, 1] range
    return std::acos(dot) / static_cast<float>(M_PI);
}


double GeodesicDerep::jaccard_to_ani(double J, int kmer_size) {
    if (J <= 0.0) return 70.0;
    if (J >= 1.0) return 100.0;
    double ratio = 2.0 * J / (1.0 + J);
    return std::max(70.0, std::min(100.0, std::pow(ratio, 1.0 / kmer_size) * 100.0));
}

// Producer-consumer: N reader threads decompress genomes into a bounded queue;
// compute threads pop items and run the k-mer/OPH computation.
// This overlaps NFS I/O with CPU work and issues posix_fadvise WILLNEED hints.
namespace {

struct PrefetchedItem {
    size_t idx;
    std::vector<char> data;
};

class GenomePrefetcher {
public:
    GenomePrefetcher(const std::vector<std::filesystem::path>& paths,
                     size_t n_readers, size_t queue_cap,
                     size_t byte_budget = 2ULL * 1024 * 1024 * 1024)
        : paths_(paths), cap_(queue_cap), n_readers_(n_readers),
          byte_budget_(byte_budget) {
        for (size_t t = 0; t < n_readers_; ++t)
            readers_.emplace_back([this] { read_loop(); });
    }

    ~GenomePrefetcher() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            cancelled_ = true;
        }
        not_full_.notify_all();
        not_empty_.notify_all();
        for (auto& t : readers_) if (t.joinable()) t.join();
    }

    std::optional<PrefetchedItem> pop() {
        std::unique_lock<std::mutex> lk(mu_);
        not_empty_.wait(lk, [&] { return !queue_.empty() || all_done_; });
        if (queue_.empty()) return std::nullopt;
        auto item = std::move(queue_.front());
        queue_.pop_front();
        queued_bytes_ -= item.data.size();
        not_full_.notify_one();
        return item;
    }

private:
    const std::vector<std::filesystem::path>& paths_;
    size_t cap_;
    size_t n_readers_;
    size_t byte_budget_;
    size_t queued_bytes_{0};
    std::atomic<size_t> next_{0};
    std::mutex mu_;
    std::condition_variable not_empty_, not_full_;
    std::deque<PrefetchedItem> queue_;
    bool cancelled_ = false;
    bool all_done_ = false;
    std::atomic<size_t> finished_{0};
    std::vector<std::thread> readers_;

    void read_loop() {
        while (true) {
            const size_t i = next_.fetch_add(1, std::memory_order_relaxed);
            if (i >= paths_.size()) break;

            // Fadvise lookahead: hint NFS to start fetching a future file now
            const size_t look = i + cap_;
            if (look < paths_.size()) {
                const std::string apath = paths_[look].string();
                int fd = ::open(apath.c_str(), O_RDONLY);
                if (fd >= 0) {
                    ::posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
                    ::posix_fadvise(fd, 0, 0, POSIX_FADV_WILLNEED);
                    ::close(fd);
                }
            }

            std::vector<char> d;
            try {
                d = GzReader::decompress_file(paths_[i].string());
            } catch (const std::exception& ex) {
                spdlog::warn("GenomePrefetcher: read failed for {}: {}", paths_[i].filename().string(), ex.what());
            } catch (...) {
                spdlog::warn("GenomePrefetcher: read failed for {}: unknown error", paths_[i].filename().string());
            }

            std::unique_lock<std::mutex> lk(mu_);
            not_full_.wait(lk, [&] {
                return (queue_.size() < cap_ && queued_bytes_ + d.size() <= byte_budget_) || cancelled_;
            });
            if (cancelled_) break;
            queued_bytes_ += d.size();
            queue_.push_back({i, std::move(d)});
            not_empty_.notify_one();
        }

        if (finished_.fetch_add(1, std::memory_order_acq_rel) + 1 == n_readers_) {
            std::lock_guard<std::mutex> lk(mu_);
            all_done_ = true;
            not_empty_.notify_all();
        }
    }
};

} // namespace

GenomeEmbedding GeodesicDerep::embed_genome(const std::filesystem::path& path, uint64_t id) {
    // Read file once into buffer, then compute both OPH sketches from memory.
    // Retry up to 3 times on failure (NFS transient errors); catching here prevents
    // an uncaught exception from crashing the enclosing OMP parallel region.
    std::vector<char> buf;
    std::string read_error;
    for (int attempt = 0; attempt < 3; ++attempt) {
        try {
            buf = GzReader::decompress_file(path.string());
            read_error.clear();
            break;
        } catch (const std::exception& ex) {
            read_error = ex.what();
            if (attempt < 2)
                std::this_thread::sleep_for(std::chrono::milliseconds(500 * (attempt + 1)));
        } catch (...) {
            read_error = "unknown error";
            break;
        }
    }
    if (!read_error.empty()) {
        spdlog::warn("GEODESIC: cannot read {} after 3 tries: {}",
                     path.filename().string(), read_error);
        std::lock_guard<std::mutex> lock(failed_reads_mutex_);
        failed_reads_.emplace_back(path.string(), read_error);

        GenomeEmbedding emb;
        emb.genome_id = id;
        emb.vector.assign(cfg_.embedding_dim, 0.0f);
        emb.isolation_score = 0.0f;
        emb.quality_score = 0.0f;
        emb.genome_size = 0;
        emb.path = path;
        return emb;
    }

    MinHasher hasher({
        .kmer_size = cfg_.kmer_size,
        .sketch_size = cfg_.sketch_size,
        .syncmer_s = cfg_.syncmer_s,
        .seed = 42
    });
    auto oph = hasher.sketch_oph_with_positions_from_buffer(buf.data(), buf.size(), cfg_.sketch_size);

    GenomeEmbedding emb;
    emb.genome_id = id;
    // Placeholder — overwritten by apply_nystrom_embeddings() in build_index()
    emb.vector.assign(cfg_.embedding_dim, 0.0f);
    if (oph.genome_length == 0) {
        spdlog::warn("GEODESIC: empty genome {} — excluded from stats",
                     path.filename().string());
    }
    // Truncate to 16-bit OPH for long-term storage (bias-corrected in refine_jaccard)
    {
        const size_t m = oph.signature.size();
        emb.oph_sig.resize(m);
        for (size_t t = 0; t < m; ++t)
            emb.oph_sig[t] = static_cast<uint16_t>(oph.signature[t]);
    }
    // oph_sig2 is populated lazily via materialize_sig2_for_indices() for anchors
    // and borderline candidates only, avoiding a 2× sketch overhead for all genomes.
    emb.real_bins_mask = std::move(oph.real_bins_bitmask);
    emb.n_real_bins = static_cast<uint32_t>(oph.n_real_bins);
    emb.n_contigs = static_cast<uint32_t>(oph.n_contigs);
    emb.isolation_score = 0.0f;
    emb.quality_score = 50.0f;  // Default, overridden in build_index
    emb.genome_size = oph.genome_length;
    emb.path = path;

    // Cache buffer for lazy sig2 materialization of anchors (avoids NFS re-read).
    if (!buf_cache_.empty() && id < buf_cache_.size() && !buf.empty()) {
        const size_t nbytes = buf.size();
        const size_t prev = buf_cache_bytes_.fetch_add(nbytes);
        if (prev + nbytes <= kBufCacheLimitBytes) {
            buf_cache_[id] = std::move(buf);
        } else {
            buf_cache_bytes_.fetch_sub(nbytes);
        }
    }

    return emb;
}

GenomeEmbedding GeodesicDerep::embed_genome_from_buffer(
        const char* data, size_t len, uint64_t id, const std::filesystem::path& path) {
    MinHasher hasher({
        .kmer_size = cfg_.kmer_size,
        .sketch_size = cfg_.sketch_size,
        .syncmer_s = cfg_.syncmer_s,
        .seed = 42
    });

    auto oph = hasher.sketch_oph_with_positions_from_buffer(data, len, cfg_.sketch_size);

    GenomeEmbedding emb;
    emb.genome_id = id;
    emb.vector.assign(cfg_.embedding_dim, 0.0f);
    if (oph.genome_length == 0) {
        spdlog::warn("GEODESIC: empty genome {} — excluded from stats",
                     path.filename().string());
    }
    {
        const size_t m = oph.signature.size();
        emb.oph_sig.resize(m);
        for (size_t t = 0; t < m; ++t)
            emb.oph_sig[t] = static_cast<uint16_t>(oph.signature[t]);
    }
    // oph_sig2 populated lazily by materialize_sig2_for_indices() for anchors/borderline only.
    emb.real_bins_mask = std::move(oph.real_bins_bitmask);
    emb.n_real_bins = static_cast<uint32_t>(oph.n_real_bins);
    emb.n_contigs = static_cast<uint32_t>(oph.n_contigs);
    emb.isolation_score = 0.0f;
    emb.quality_score = 50.0f;
    emb.genome_size = oph.genome_length;
    emb.path = path;

    return emb;
}

void GeodesicDerep::materialize_sig2_for_indices(const std::vector<size_t>& indices) {
    if (indices.empty()) return;

    std::vector<size_t> need;
    need.reserve(indices.size());
    for (size_t i : indices) {
        if (i < embeddings_.size() && embeddings_[i].oph_sig2.empty() && !embeddings_[i].path.empty())
            need.push_back(i);
    }
    if (need.empty()) return;

    size_t cache_hits = 0;
    if (!buf_cache_.empty()) {
        for (size_t i : need)
            if (i < buf_cache_.size() && !buf_cache_[i].empty()) ++cache_hits;
    }
    spdlog::info("GEODESIC: materializing sig2 for {} genomes ({} from cache, {} NFS re-reads)",
                 need.size(), cache_hits, need.size() - cache_hits);

    const MinHasher::Config cfg2{
        .kmer_size  = cfg_.kmer_size,
        .sketch_size = cfg_.sketch_size,
        .syncmer_s  = cfg_.syncmer_s,
        .seed       = 1337
    };

#if GEODESIC_USE_OMP
    #pragma omp parallel for schedule(dynamic, 4) num_threads(cfg_.threads)
    for (size_t k = 0; k < need.size(); ++k) {
        const size_t i = need[k];
        try {
            std::vector<char> local_buf;
            const char* data_ptr;
            size_t data_len;
            if (!buf_cache_.empty() && i < buf_cache_.size() && !buf_cache_[i].empty()) {
                data_ptr = buf_cache_[i].data();
                data_len = buf_cache_[i].size();
            } else {
                local_buf = GzReader::decompress_file(embeddings_[i].path.string());
                data_ptr = local_buf.data();
                data_len = local_buf.size();
            }
            MinHasher hasher2(cfg2);
            auto oph2 = hasher2.sketch_oph_with_positions_from_buffer(data_ptr, data_len, cfg_.sketch_size);
            const size_t m2 = oph2.signature.size();
            std::vector<uint16_t> sig2(m2);
            for (size_t t = 0; t < m2; ++t)
                sig2[t] = static_cast<uint16_t>(oph2.signature[t]);
            embeddings_[i].oph_sig2 = std::move(sig2);
        } catch (const std::exception& e) {
            spdlog::warn("GEODESIC: sig2 materialization failed for {}: {}",
                         embeddings_[i].path.filename().string(), e.what());
        }
    }
#else
    for (const size_t i : need) {
        try {
            std::vector<char> local_buf;
            const char* data_ptr;
            size_t data_len;
            if (!buf_cache_.empty() && i < buf_cache_.size() && !buf_cache_[i].empty()) {
                data_ptr = buf_cache_[i].data();
                data_len = buf_cache_[i].size();
            } else {
                local_buf = GzReader::decompress_file(embeddings_[i].path.string());
                data_ptr = local_buf.data();
                data_len = local_buf.size();
            }
            MinHasher hasher2(cfg2);
            auto oph2 = hasher2.sketch_oph_with_positions_from_buffer(data_ptr, data_len, cfg_.sketch_size);
            const size_t m2 = oph2.signature.size();
            embeddings_[i].oph_sig2.resize(m2);
            for (size_t t = 0; t < m2; ++t)
                embeddings_[i].oph_sig2[t] = static_cast<uint16_t>(oph2.signature[t]);
        } catch (const std::exception& e) {
            spdlog::warn("GEODESIC: sig2 materialization failed for {}: {}",
                         embeddings_[i].path.filename().string(), e.what());
        }
    }
#endif
}

void GeodesicDerep::build_index(const std::vector<std::filesystem::path>& genomes,
                                 const std::unordered_map<std::string, double>& quality_scores,
                                 db::EmbeddingStore* emb_store,
                                 const std::string& taxonomy) {
    spdlog::info("GEODESIC: embedding {} genomes (dim={}, k={}, threads={})",
                 genomes.size(), cfg_.embedding_dim, cfg_.kmer_size, cfg_.threads);
    auto t_build_start = std::chrono::steady_clock::now();
    auto t_phase = [&t_build_start](const char* name) {
        auto now = std::chrono::steady_clock::now();
        double s = std::chrono::duration<double>(now - t_build_start).count();
        spdlog::info("  [profile] {:30s} {:7.1f}s (cumulative)", name, s);
    };

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

    // Pre-allocate buffer cache slots; filled by embed_genome() below and consumed by
    // materialize_sig2_for_indices() during Nyström anchor sketching.
    buf_cache_.assign(n, {});
    buf_cache_bytes_.store(0);

    ProgressCounter progress(n, "GEODESIC: embedding");

    // Precompute path strings to avoid repeated .string() conversions in loops
    std::vector<std::string> path_strs(n);
    for (size_t i = 0; i < n; ++i) path_strs[i] = genomes[i].string();

    // Limit concurrent NFS file opens to avoid metadata-op saturation.
    // Each embed_genome() is ~95% NFS I/O; capping simultaneous readers at
    // io_threads gives full bandwidth with far less server contention.
    const int io_threads = (cfg_.io_threads > 0) ? cfg_.io_threads
                                                  : cfg_.threads;

    // Parallel embedding: each thread reads+decompresses its own genome file.
    // Larger NFS reads (4MB chunks in GzReader) + posix_fadvise hints in GzReader
    // reduce round-trips vs the old 256KB reads.
#if GEODESIC_USE_OMP
    {
    std::counting_semaphore<> io_sem(io_threads);
    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < n; ++i) {
        io_sem.acquire();
        try {
            embeddings_[i] = embed_genome(genomes[i], i);
        } catch (const std::exception& e) {
            spdlog::error("GEODESIC: embed_genome exception for {}: {}", genomes[i].filename().string(), e.what());
            embeddings_[i].genome_id = i;
            embeddings_[i].vector.assign(cfg_.embedding_dim, 0.0f);
            embeddings_[i].quality_score = 0.0f;
            embeddings_[i].path = genomes[i];
        } catch (...) {
            spdlog::error("GEODESIC: embed_genome unknown exception for {}", genomes[i].filename().string());
            embeddings_[i].genome_id = i;
            embeddings_[i].vector.assign(cfg_.embedding_dim, 0.0f);
            embeddings_[i].quality_score = 0.0f;
            embeddings_[i].path = genomes[i];
        }
        io_sem.release();
        auto it = quality_scores.find(path_strs[i]);
        if (it != quality_scores.end())
            embeddings_[i].quality_score = static_cast<float>(it->second);
        else if (embeddings_[i].quality_score != 0.0f)
            embeddings_[i].quality_score = 50.0f;
        progress.increment();
    }
    }
#else
    {
    auto worker = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            embeddings_[i] = embed_genome(genomes[i], i);
            auto it = quality_scores.find(path_strs[i]);
            if (it != quality_scores.end())
                embeddings_[i].quality_score = static_cast<float>(it->second);
            else
                embeddings_[i].quality_score = 50.0f;
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
            if (start < end)
                futures.push_back(std::async(std::launch::async, worker, start, end));
        }
        for (auto& f : futures) f.get();
    }
    }
#endif
    progress.finish();
    {
        // Log throughput to distinguish NFS-bound from compute-bound runs.
        // NFS-bound: low MB/s (cold cache); compute-bound: > ~300 MB/s.
        auto t_oph_end = std::chrono::steady_clock::now();
        double oph_s = std::chrono::duration<double>(t_oph_end - t_build_start).count();
        uint64_t total_bases = 0;
        for (size_t i = 0; i < n; ++i) total_bases += embeddings_[i].genome_size;
        double gbp = static_cast<double>(total_bases) / 1e9;
        spdlog::info("  [profile] OPH throughput: {:.0f} MB/s ({:.1f} Gbp, {:.1f}s)",
                     gbp * 1e3 / oph_s, gbp, oph_s);
    }
    t_phase("OPH sketch (embed_genome)");

    // Build canonical ID → row map (identity at this point, but needed after future sorts)
    gid_to_row_.clear();
    gid_to_row_.reserve(n);
    for (size_t i = 0; i < n; ++i)
        gid_to_row_[embeddings_[i].genome_id] = i;

    // Copy embeddings to SoA store for SIMD-friendly access
    for (size_t i = 0; i < n; ++i) {
        store_.genome_ids[i] = embeddings_[i].genome_id;
        store_.isolation_scores[i] = embeddings_[i].isolation_score;
        store_.quality_scores[i] = embeddings_[i].quality_score;
        store_.genome_sizes[i] = embeddings_[i].genome_size;
        store_.paths[i] = embeddings_[i].path;
        std::copy(embeddings_[i].vector.begin(), embeddings_[i].vector.end(), store_.row(i));
    }

    t_phase("SoA copy");

    // Nyström spectral embedding (replaces CountSketch placeholder vectors)
    nystrom_applied_ = false;
    if (n > SMALL_N_THRESHOLD) {
        apply_nystrom_embeddings();  // sets nystrom_applied_ = true, updates store_.data
        t_phase("Nyström embedding (total)");
    }

    // Release buffer cache — anchor sig2 materialization is done.
    buf_cache_ = {};
    buf_cache_bytes_.store(0);

    // Snapshot finalised (post-Nyström) float vectors and launch async save.
    // The background thread reads oph_sig directly from embeddings_ (Nyström never modifies it).
    // The future is joined in ~GeodesicDerep(), so it always finishes before embeddings_ dies.
    if (emb_store && !taxonomy.empty()) {
        std::vector<std::vector<float>> vec_snap(n);
        for (size_t i = 0; i < n; ++i) vec_snap[i] = embeddings_[i].vector;
        async_save_future_ = std::async(std::launch::async,
            [this, emb_store, taxonomy, snap = std::move(vec_snap)]() mutable {
                save_embeddings_async(*emb_store, taxonomy, std::move(snap));
            });
    }

    // Build HNSW index — skip for small n (brute-force pairwise is faster)
    if (n > SMALL_N_THRESHOLD) {
        index_->build(embeddings_, cfg_.hnsw_m, cfg_.hnsw_ef_construction, cfg_.hnsw_ef_search);
        if (is_verbose()) spdlog::info("GEODESIC: HNSW index built ({} embeddings, M={}, ef={})",
                     embeddings_.size(), cfg_.hnsw_m, cfg_.hnsw_ef_construction);
        t_phase("HNSW build");
    }
}

GeodesicDerep::NNDistStats GeodesicDerep::compute_isolation_scores() {
    if (embeddings_.empty()) return {0.0, 0.0, 0.0};
    if (!index_->index) {
        compute_isolation_scores_brute();
        // For small n (no HNSW), compute 1-NN distances brute-force from SoA
        size_t n = store_.n;
        size_t dim = store_.dim;
        std::vector<float> nn_dists(n, 1.0f);
        for (size_t i = 0; i < n; ++i) {
            float min_d = 1.0f;
            for (size_t j = 0; j < n; ++j) {
                if (i == j) continue;
                float dot = dot_product_simd(store_.row(i), store_.row(j), dim);
                dot = std::clamp(dot, -1.0f, 1.0f);
                float d = std::acos(dot) / static_cast<float>(M_PI);
                if (d < min_d) min_d = d;
            }
            nn_dists[i] = min_d;
        }
        std::sort(nn_dists.begin(), nn_dists.end());
        return {
            .p5  = nn_dists[n * 5  / 100],
            .p50 = nn_dists[n * 50 / 100],
            .p95 = nn_dists[n * 95 / 100],
        };
    }
    if (is_verbose()) spdlog::info("GEODESIC: computing isolation scores (k={})", cfg_.isolation_k);

    // Single HNSW pass: compute mean k-NN distance (isolation) AND collect 1-NN
    // distance per genome for threshold calibration — avoids a second HNSW sweep.
    size_t n_emb = embeddings_.size();
    const int saved_ef = index_->ef_search;
    // ef=15 is too low for small-medium taxa (n=50-500): isolation scores become inaccurate,
    // causing a centroid-like genome to appear most-isolated and become rep1. FPS then
    // stops at 1 rep while the brute-force coverage pass shows many genomes at 92-95% ANI.
    // Scale ef with n: min 50 (adequate recall for k=10), cap 200 for speed on large taxa.
    index_->ef_search = std::max(50, std::min(200, static_cast<int>(n_emb / 100)));
    std::vector<float> nn_dists(n_emb, 1.0f);

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
                if (count == 0) nn_dists[ei] = dist;  // first non-self = 1-NN
                total_dist += dist;
                ++count;
            }
        }

        emb.isolation_score = (count > 0) ? total_dist / count : 1.0f;
    }
    index_->ef_search = saved_ef;

    // Compute median genome size for length normalization
    std::vector<uint64_t> sizes;
    sizes.reserve(n_emb);
    for (const auto& e : embeddings_) sizes.push_back(e.genome_size);
    auto mid = sizes.begin() + sizes.size() / 2;
    std::nth_element(sizes.begin(), mid, sizes.end());
    uint64_t median_size = sizes[sizes.size() / 2];

    // Sort by genome_id for a deterministic, stable data layout.
    // Sorting by isolation_score (HNSW-derived) was non-deterministic: parallel HNSW
    // insertion produces different graphs each run → different isolation scores →
    // different sort order → IForest sees different row indices → different contamination
    // counts on identical data. genome_id sort is fully deterministic.
    // FPS Phase 1 now finds the most-isolated genome by argmax scan instead of [0].
    // Permutation sort: GenomeEmbedding structs are large (~80KB each due to OPH
    // signatures). Sort a small index vector, then rearrange with one sequential
    // pass of moves instead of O(n log n) random swaps of 80KB structs.
    {
        std::vector<size_t> perm(n_emb);
        std::iota(perm.begin(), perm.end(), 0);
        std::sort(perm.begin(), perm.end(), [&](size_t a, size_t b) {
            return embeddings_[a].genome_id < embeddings_[b].genome_id;
        });
        std::vector<GenomeEmbedding> sorted;
        sorted.reserve(n_emb);
        for (size_t idx : perm) sorted.push_back(std::move(embeddings_[idx]));
        embeddings_ = std::move(sorted);
    }

    // Rebuild canonical ID → row map after sort
    gid_to_row_.clear();
    gid_to_row_.reserve(n_emb);
    for (size_t i = 0; i < n_emb; ++i)
        gid_to_row_[embeddings_[i].genome_id] = i;

    // Update SoA store to match sorted embeddings (also copies Nyström vectors to store_.data)
    size_t n = n_emb;
    for (size_t i = 0; i < n; ++i) {
        store_.genome_ids[i] = embeddings_[i].genome_id;
        store_.isolation_scores[i] = embeddings_[i].isolation_score;
        store_.quality_scores[i] = embeddings_[i].quality_score;
        store_.genome_sizes[i] = embeddings_[i].genome_size;
        store_.paths[i] = embeddings_[i].path;
        std::copy(embeddings_[i].vector.begin(), embeddings_[i].vector.end(), store_.row(i));
    }

    if (is_verbose()) {
        const auto [it_min, it_max] = std::minmax_element(
            embeddings_.begin(), embeddings_.end(),
            [](const auto& a, const auto& b) { return a.isolation_score < b.isolation_score; });
        spdlog::info("GEODESIC: isolation scores computed (max={:.4f}, min={:.4f})",
                     it_max->isolation_score, it_min->isolation_score);
        const auto [qit_min, qit_max] = std::minmax_element(
            embeddings_.begin(), embeddings_.end(),
            [](const auto& a, const auto& b) { return a.quality_score < b.quality_score; });
        spdlog::info("GEODESIC: quality range: {:.1f} - {:.1f}",
                     qit_min->quality_score, qit_max->quality_score);
    }
    if (is_verbose()) {
        auto [sz_min, sz_max] = std::minmax_element(sizes.begin(), sizes.end());
        spdlog::info("GEODESIC: genome size range: {} - {} bp (median={})",
                     *sz_min, *sz_max, median_size);
    }

    std::sort(nn_dists.begin(), nn_dists.end());
    return {
        .p5  = nn_dists[n * 5  / 100],
        .p50 = nn_dists[n * 50 / 100],
        .p95 = nn_dists[n * 95 / 100],
    };
}

void GeodesicDerep::apply_nystrom_embeddings() {
    const size_t n = embeddings_.size();
    if (n == 0) return;

    auto t_nys_start = std::chrono::steady_clock::now();
    auto t_nys = [&t_nys_start](const char* name) {
        auto now = std::chrono::steady_clock::now();
        double s = std::chrono::duration<double>(now - t_nys_start).count();
        spdlog::info("  [nystrom] {:30s} {:7.1f}s", name, s);
    };

    // Auto n_anchors: -1 → min(n, max(200, 2 * embedding_dim))
    const int anchors_requested = (cfg_.nystrom_anchors < 0)
        ? std::min(static_cast<int>(n), std::max(200, 2 * cfg_.embedding_dim))
        : cfg_.nystrom_anchors;
    const size_t n_anchors = static_cast<size_t>(
        std::min(anchors_requested, static_cast<int>(n)));

    if (is_verbose()) {
        spdlog::info("GEODESIC/Nyström: building spectral embedding "
                     "(n_anchors={}, min_variance={:.0f}%)",
                     n_anchors, cfg_.nystrom_min_variance * 100.0f);
    }

    // ---- Step 1: sample anchor indices (stratified by fill fraction) ----
    std::vector<size_t> anchor_idx;
    anchor_idx.reserve(n_anchors);
    {
        std::mt19937_64 rng(42);
        const bool do_stratify = (n_anchors >= 5 && n >= 5 && cfg_.sketch_size > 0);
        if (do_stratify) {
            static constexpr int N_STRATA = 5;
            // Sort genome indices by fill fraction f_i = n_real_bins_i / sketch_size
            std::vector<std::pair<float, size_t>> fi_sorted(n);
            for (size_t i = 0; i < n; ++i) {
                fi_sorted[i] = {
                    static_cast<float>(embeddings_[i].n_real_bins) / static_cast<float>(cfg_.sketch_size),
                    i
                };
            }
            std::sort(fi_sorted.begin(), fi_sorted.end());

            size_t per_stratum = n_anchors / N_STRATA;
            size_t remainder   = n_anchors % N_STRATA;
            for (int s = 0; s < N_STRATA; ++s) {
                size_t stratum_start = (static_cast<size_t>(s) * n) / N_STRATA;
                size_t stratum_end   = (static_cast<size_t>(s + 1) * n) / N_STRATA;
                size_t k = per_stratum + (static_cast<size_t>(s) < remainder ? 1 : 0);
                k = std::min(k, stratum_end - stratum_start);

                std::vector<size_t> stratum;
                stratum.reserve(stratum_end - stratum_start);
                for (size_t idx = stratum_start; idx < stratum_end; ++idx)
                    stratum.push_back(fi_sorted[idx].second);

                for (size_t t = 0; t < k && !stratum.empty(); ++t) {
                    std::uniform_int_distribution<size_t> dist(t, stratum.size() - 1);
                    std::swap(stratum[t], stratum[dist(rng)]);
                    anchor_idx.push_back(stratum[t]);
                }
            }
            // Pad to n_anchors if strata were too small
            if (anchor_idx.size() < n_anchors) {
                std::unordered_set<size_t> already(anchor_idx.begin(), anchor_idx.end());
                std::vector<size_t> remaining;
                for (size_t i = 0; i < n; ++i)
                    if (!already.count(i)) remaining.push_back(i);
                std::shuffle(remaining.begin(), remaining.end(), rng);
                for (size_t t = 0; t < remaining.size() && anchor_idx.size() < n_anchors; ++t)
                    anchor_idx.push_back(remaining[t]);
            }
        } else {
            // Uniform fallback (original behavior for small n or sketch_size=0)
            anchor_idx.resize(n);
            std::iota(anchor_idx.begin(), anchor_idx.end(), 0);
            for (size_t i = 0; i < n_anchors; ++i) {
                std::uniform_int_distribution<size_t> dist(i, n - 1);
                std::swap(anchor_idx[i], anchor_idx[dist(rng)]);
            }
            anchor_idx.resize(n_anchors);
        }
    }

    // Lazily compute sig2 for anchors: only ~512 genomes re-read from NFS.
    // This keeps dual-sketch accuracy for the anchor Gram (eigendecomposition-sensitive)
    // without computing sig2 for all n genomes during the sketch pass.
    materialize_sig2_for_indices(anchor_idx);

    // ---- Step 2: build anchor Gram matrix K [n_anchors × n_anchors] ----
    // K[i,j] = OPH Jaccard(anchor_i, anchor_j) — symmetric PD kernel.
    // RowMajor: each thread writes its own contiguous row (no false sharing).
    using RowMajorMatXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    // Uninitialized: every element is written — diagonal in the parallel loop,
    // upper triangle in the parallel loop, lower triangle from upper after.
    RowMajorMatXf K_rm(static_cast<int>(n_anchors), static_cast<int>(n_anchors));

    // Improvement 4: Containment blend for MAG anchors.
    static constexpr float kContainmentBlendThresh = 0.2f;
    auto containment_oph = [](const std::vector<uint64_t>& mask_a, uint32_t n_real_a,
                               const std::vector<uint64_t>& mask_b) -> float {
        if (n_real_a == 0 || mask_a.empty() || mask_b.empty()) return 0.0f;
        size_t words = std::min(mask_a.size(), mask_b.size());
        int shared = 0;
        for (size_t w = 0; w < words; ++w)
            shared += __builtin_popcountll(mask_a[w] & mask_b[w]);
        return static_cast<float>(shared) / static_cast<float>(n_real_a);
    };

#if GEODESIC_USE_OMP
    #pragma omp parallel for schedule(dynamic, 4) num_threads(cfg_.threads)
#endif
    for (int i = 0; i < static_cast<int>(n_anchors); ++i) {
        K_rm(i, i) = 1.0f;
        for (int j = i + 1; j < static_cast<int>(n_anchors); ++j) {
            size_t ai = anchor_idx[static_cast<size_t>(i)];
            size_t aj = anchor_idx[static_cast<size_t>(j)];
            float jac = static_cast<float>(
                refine_jaccard(embeddings_[ai].oph_sig, embeddings_[aj].oph_sig));
            // Improvement 5: average two sketches for variance reduction
            if (!embeddings_[ai].oph_sig2.empty() && !embeddings_[aj].oph_sig2.empty()) {
                float jac2 = static_cast<float>(
                    refine_jaccard(embeddings_[ai].oph_sig2, embeddings_[aj].oph_sig2));
                jac = (jac + jac2) * 0.5f;
            }
            // Improvement 4: containment blend for sparse anchors (MAGs)
            float f_i = (cfg_.sketch_size > 0)
                ? static_cast<float>(embeddings_[ai].n_real_bins) / cfg_.sketch_size : 1.0f;
            float f_j = (cfg_.sketch_size > 0)
                ? static_cast<float>(embeddings_[aj].n_real_bins) / cfg_.sketch_size : 1.0f;
            if (f_i < kContainmentBlendThresh || f_j < kContainmentBlendThresh) {
                float cont_ij = containment_oph(embeddings_[ai].real_bins_mask,
                                                 embeddings_[ai].n_real_bins,
                                                 embeddings_[aj].real_bins_mask);
                float cont_ji = containment_oph(embeddings_[aj].real_bins_mask,
                                                 embeddings_[aj].n_real_bins,
                                                 embeddings_[ai].real_bins_mask);
                float alpha_i = std::max(0.0f, 1.0f - f_i / kContainmentBlendThresh);
                float alpha_j = std::max(0.0f, 1.0f - f_j / kContainmentBlendThresh);
                float alpha = std::max(alpha_i, alpha_j);
                jac = (1.0f - alpha) * jac + alpha * std::max(cont_ij, cont_ji);
            }
            K_rm(i, j) = jac;
        }
    }
    // Fill lower triangle from upper (sequential)
    for (int i = 0; i < static_cast<int>(n_anchors); ++i)
        for (int j = 0; j < i; ++j)
            K_rm(i, j) = K_rm(j, i);

    // Improvement 2: Symmetric Laplacian normalization.
    // Reduces hub-anchor bias: K_norm[i,j] = K[i,j] / sqrt(d_i * d_j).
    // d_anchor_inv_sqrt persists in scope for k_G normalization in Step 4.
    Eigen::VectorXf d_anchor_inv_sqrt = Eigen::VectorXf::Ones(static_cast<int>(n_anchors));
    if (cfg_.nystrom_degree_normalize) {
        Eigen::VectorXf d_anchor(static_cast<int>(n_anchors));
        for (int i = 0; i < static_cast<int>(n_anchors); ++i)
            d_anchor(i) = K_rm.row(i).sum();
        d_anchor = d_anchor.cwiseMax(1e-10f);
        d_anchor_inv_sqrt = d_anchor.cwiseSqrt().cwiseInverse();
        for (int i = 0; i < static_cast<int>(n_anchors); ++i)
            for (int j = 0; j < static_cast<int>(n_anchors); ++j)
                K_rm(i, j) *= d_anchor_inv_sqrt(i) * d_anchor_inv_sqrt(j);
    }

    // Improvement 3: Diagonal loading (Tikhonov regularization).
    // Prevents near-zero eigenvalues that blow up W = U * diag(lam^{-1/2}).
    if (cfg_.nystrom_diagonal_loading > 0.0f) {
        float mean_diag = K_rm.diagonal().sum() / static_cast<float>(n_anchors);
        // Floor ensures meaningful regularization even when degree normalization
        // compresses mean_diag toward zero (high-connectivity anchors).
        float lambda_reg = cfg_.nystrom_diagonal_loading * std::max(mean_diag, 1e-4f);
        for (int i = 0; i < static_cast<int>(n_anchors); ++i)
            K_rm(i, i) += lambda_reg;
    }

    t_nys("anchor Gram matrix K");

    Eigen::MatrixXf K = K_rm;  // convert to ColMajor for Eigen's SelfAdjointEigenSolver

    // ---- Step 3: eigendecompose K → top-d eigenvectors + eigenvalues ----
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> solver(K);
    if (solver.info() != Eigen::Success) {
        spdlog::error("GEODESIC/Nyström: eigendecomposition failed — embeddings remain zero vectors");
        return;
    }

    // Auto-select d: fewest top eigenvectors that capture >= nystrom_min_variance.
    // Clamp to non-negative before summing: Gram matrices can have small numerical
    // negative eigenvalues that shrink total_eig and cause captured > 1.0.
    const float total_eig = solver.eigenvalues().cwiseMax(0.0f).sum();
    const int max_d = std::min(cfg_.embedding_dim, static_cast<int>(n_anchors));
    int actual_d = 1;
    if (total_eig > 0.0f) {
        float cumsum = 0.0f;
        for (int i = static_cast<int>(n_anchors) - 1;
             i >= 0 && actual_d <= max_d; --i) {
            cumsum += std::max(0.0f, solver.eigenvalues()(i));
            if (cumsum / total_eig >= cfg_.nystrom_min_variance) break;
            if (actual_d < max_d) ++actual_d;
        }
    } else {
        actual_d = max_d;
    }

    Eigen::MatrixXf U = solver.eigenvectors().rightCols(actual_d); // [n_anchors × d]
    Eigen::VectorXf lam = solver.eigenvalues().tail(actual_d);     // [d]
    lam = lam.cwiseMax(1e-10f);

    // W = U × diag(λ^{-1/2})   [n_anchors × d]
    Eigen::MatrixXf W = U * lam.cwiseSqrt().cwiseInverse().asDiagonal();

    t_nys("eigendecompose K");

    float captured = (total_eig > 0.0f) ? lam.sum() / total_eig : 1.0f;
    if (is_verbose()) {
        spdlog::info("GEODESIC/Nyström: auto-selected d={} capturing {:.1f}% of variance "
                     "(target {:.0f}%, n_anchors={})",
                     actual_d, captured * 100.0f,
                     cfg_.nystrom_min_variance * 100.0f, n_anchors);
    }

    // ---- Step 4: compute Nyström embedding for all genomes (parallel) ----

    // Dense O(1) anchor lookup: anchor_pos[i] = anchor row in K, or -1 if not an anchor.
    // Avoids unordered_map hash lookup in the hot parallel loop.
    std::vector<int32_t> anchor_pos(n, -1);
    for (size_t a = 0; a < n_anchors; ++a)
        anchor_pos[anchor_idx[a]] = static_cast<int32_t>(a);

    // Pack anchor signatures into a contiguous slab for cache-friendly access.
    // Without packing, each of the 512 anchor sigs is a separate heap allocation,
    // causing cache/TLB misses when scanning them per genome. The slab is ~10 MB
    // (512 × 10000 × 2 bytes) — fits in L3 and stays warm across the parallel loop.
    const size_t m_sig = cfg_.sketch_size;
    std::vector<uint16_t> anchor_slab;
    std::vector<float>    anchor_fill_f;
    std::vector<uint32_t> anchor_n_real_v;
    if (m_sig > 0 && !anchor_idx.empty() && !embeddings_[anchor_idx[0]].oph_sig.empty()) {
        anchor_slab.resize(n_anchors * m_sig);
        anchor_fill_f.resize(n_anchors);
        anchor_n_real_v.resize(n_anchors);
        for (size_t a = 0; a < n_anchors; ++a) {
            const size_t ai = anchor_idx[a];
            const auto& sig = embeddings_[ai].oph_sig;
            const size_t copy_m = std::min(sig.size(), m_sig);
            std::memcpy(anchor_slab.data() + a * m_sig, sig.data(), copy_m * sizeof(uint16_t));
            anchor_fill_f[a]   = static_cast<float>(embeddings_[ai].n_real_bins) / static_cast<float>(m_sig);
            anchor_n_real_v[a] = embeddings_[ai].n_real_bins;
        }
    }

    // Update store dim if actual_d differs from configured max
    if (actual_d != runtime_dim_) {
        runtime_dim_ = actual_d;
        store_.resize(n, actual_d);
    }

    // Pre-size embedding vectors once to avoid realloc inside the parallel loop.
    for (size_t i = 0; i < n; ++i)
        embeddings_[i].vector.resize(actual_d);

#if GEODESIC_USE_OMP
    #pragma omp parallel num_threads(cfg_.threads)
    {
        // Thread-local Eigen temporaries: one alloc per thread instead of per genome.
        Eigen::VectorXf k_G(static_cast<int>(n_anchors));
        Eigen::VectorXf e_raw(actual_d);

        #pragma omp for schedule(dynamic, 50)
        for (size_t i = 0; i < n; ++i) {
            int32_t ap = anchor_pos[i];
            if (ap >= 0) {
                // Anchor: k_G is already degree-normalized (read from K after Step 2+3).
                // Do NOT re-normalize below.
                k_G = K.row(ap).transpose();
            } else {
                // Non-anchor: sig1 only, via contiguous anchor slab + AVX2 equality count.
                // Each Jaccard call compares 10000 uint16_t; AVX2 cuts that to 625 vector ops.
                if (embeddings_[i].oph_sig.empty()) continue;  // genome failed to sketch
                const uint16_t* sig_i_ptr = embeddings_[i].oph_sig.data();
                const float f_i = anchor_slab.empty()
                    ? 1.0f
                    : static_cast<float>(embeddings_[i].n_real_bins) / static_cast<float>(m_sig);
                for (size_t a = 0; a < n_anchors; ++a) {
                    float jac;
                    if (!anchor_slab.empty()) {
                        jac = static_cast<float>(
                            refine_jaccard_ptr(sig_i_ptr,
                                              anchor_slab.data() + a * m_sig, m_sig));
                    } else {
                        jac = static_cast<float>(
                            refine_jaccard(embeddings_[i].oph_sig, embeddings_[anchor_idx[a]].oph_sig));
                    }
                    // Improvement 4: containment blend for sparse genomes/anchors (MAGs).
                    // Precomputed fill fraction; falls back to scattered mask access only for MAGs.
                    const float f_a = anchor_slab.empty()
                        ? 1.0f : anchor_fill_f[a];
                    if (f_i < kContainmentBlendThresh || f_a < kContainmentBlendThresh) {
                        const size_t ai = anchor_idx[a];
                        float cont_ia = containment_oph(embeddings_[i].real_bins_mask,
                                                         embeddings_[i].n_real_bins,
                                                         embeddings_[ai].real_bins_mask);
                        float cont_ai = containment_oph(embeddings_[ai].real_bins_mask,
                                                         anchor_n_real_v[a],
                                                         embeddings_[i].real_bins_mask);
                        float alpha_i = std::max(0.0f, 1.0f - f_i / kContainmentBlendThresh);
                        float alpha_a = std::max(0.0f, 1.0f - f_a / kContainmentBlendThresh);
                        float alpha   = std::max(alpha_i, alpha_a);
                        jac = (1.0f - alpha) * jac + alpha * std::max(cont_ia, cont_ai);
                    }
                    k_G(static_cast<int>(a)) = jac;
                }
            }

            // Apply matching degree normalization only for non-anchors.
            // Anchor k_G is read from K which is already degree-normalized.
            if (cfg_.nystrom_degree_normalize && ap < 0) {
                float d_i = k_G.sum();
                if (d_i < 1e-10f) d_i = 1.0f;
                float inv_sqrt_di = 1.0f / std::sqrt(d_i);
                k_G = k_G.cwiseProduct(d_anchor_inv_sqrt) * inv_sqrt_di;
            }

            // ẽ(G) = W^T × k_G, normalized to unit sphere
            e_raw.noalias() = W.transpose() * k_G;
            float norm = e_raw.norm();
            if (norm > 1e-10f) e_raw /= norm;

            std::copy(e_raw.data(), e_raw.data() + actual_d, embeddings_[i].vector.data());
            std::copy(e_raw.data(), e_raw.data() + actual_d, store_.row(i));
        }
    }
#else
    {
        Eigen::VectorXf k_G(static_cast<int>(n_anchors));
        Eigen::VectorXf e_raw(actual_d);
        for (size_t i = 0; i < n; ++i) {
            int32_t ap = anchor_pos[i];
            if (ap >= 0) {
                // Anchor: k_G is already degree-normalized (read from K after Step 2+3).
                // Do NOT re-normalize below.
                k_G = K.row(ap).transpose();
            } else {
                // Non-anchor: anchor slab + AVX2 Jaccard (see OMP path for rationale).
                if (embeddings_[i].oph_sig.empty()) continue;  // genome failed to sketch
                const uint16_t* sig_i_ptr = embeddings_[i].oph_sig.data();
                const float f_i = anchor_slab.empty()
                    ? 1.0f
                    : static_cast<float>(embeddings_[i].n_real_bins) / static_cast<float>(m_sig);
                for (size_t a = 0; a < n_anchors; ++a) {
                    float jac;
                    if (!anchor_slab.empty()) {
                        jac = static_cast<float>(
                            refine_jaccard_ptr(sig_i_ptr,
                                              anchor_slab.data() + a * m_sig, m_sig));
                    } else {
                        jac = static_cast<float>(
                            refine_jaccard(embeddings_[i].oph_sig, embeddings_[anchor_idx[a]].oph_sig));
                    }
                    const float f_a = anchor_slab.empty() ? 1.0f : anchor_fill_f[a];
                    if (f_i < kContainmentBlendThresh || f_a < kContainmentBlendThresh) {
                        const size_t ai = anchor_idx[a];
                        float cont_ia = containment_oph(embeddings_[i].real_bins_mask,
                                                         embeddings_[i].n_real_bins,
                                                         embeddings_[ai].real_bins_mask);
                        float cont_ai = containment_oph(embeddings_[ai].real_bins_mask,
                                                         anchor_n_real_v[a],
                                                         embeddings_[i].real_bins_mask);
                        float alpha_i = std::max(0.0f, 1.0f - f_i / kContainmentBlendThresh);
                        float alpha_a = std::max(0.0f, 1.0f - f_a / kContainmentBlendThresh);
                        float alpha   = std::max(alpha_i, alpha_a);
                        jac = (1.0f - alpha) * jac + alpha * std::max(cont_ia, cont_ai);
                    }
                    k_G(static_cast<int>(a)) = jac;
                }
            }
            // Apply matching degree normalization only for non-anchors.
            // Anchor k_G is read from K which is already degree-normalized.
            if (cfg_.nystrom_degree_normalize && ap < 0) {
                float d_i = k_G.sum();
                if (d_i < 1e-10f) d_i = 1.0f;
                float inv_sqrt_di = 1.0f / std::sqrt(d_i);
                k_G = k_G.cwiseProduct(d_anchor_inv_sqrt) * inv_sqrt_di;
            }
            e_raw.noalias() = W.transpose() * k_G;
            float norm = e_raw.norm();
            if (norm > 1e-10f) e_raw /= norm;
            std::copy(e_raw.data(), e_raw.data() + actual_d, embeddings_[i].vector.data());
            std::copy(e_raw.data(), e_raw.data() + actual_d, store_.row(i));
        }
    }
#endif

    t_nys("Nyström project all genomes");

    cfg_.nystrom_captured_variance = captured;
    nystrom_applied_ = true;
    if (is_verbose()) {
        spdlog::info("GEODESIC/Nyström: spectral embedding complete ({} genomes → {}D sphere)",
                     n, actual_d);
    }
}

void GeodesicDerep::set_pinned_representatives(const std::unordered_set<std::string>& paths) {
    pinned_rep_paths_ = paths;
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

    // Permutation sort: avoid O(n log n) random swaps of large GenomeEmbedding structs.
    {
        std::vector<size_t> perm(n);
        std::iota(perm.begin(), perm.end(), 0);
        std::sort(perm.begin(), perm.end(), [&](size_t a, size_t b) {
            float la = std::sqrt(static_cast<float>(embeddings_[a].genome_size) / median_sz);
            float lb = std::sqrt(static_cast<float>(embeddings_[b].genome_size) / median_sz);
            float fa = embeddings_[a].isolation_score * (embeddings_[a].quality_score / 100.0f) * la;
            float fb = embeddings_[b].isolation_score * (embeddings_[b].quality_score / 100.0f) * lb;
            if (fa != fb) return fa > fb;
            return embeddings_[a].genome_id < embeddings_[b].genome_id;
        });
        std::vector<GenomeEmbedding> sorted;
        sorted.reserve(n);
        for (size_t idx : perm) sorted.push_back(std::move(embeddings_[idx]));
        embeddings_ = std::move(sorted);
    }

    // Rebuild canonical ID → row map after sort
    gid_to_row_.clear();
    gid_to_row_.reserve(n);
    for (size_t i = 0; i < n; ++i)
        gid_to_row_[embeddings_[i].genome_id] = i;

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

    auto qit = gid_to_row_.find(query_id);
    if (qit == gid_to_row_.end()) return {};
    const auto& query = embeddings_[qit->second];

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
    edges.reserve(n);
    size_t dim = store_.dim;
#if GEODESIC_USE_OMP
    // Always enforce thread count — omp_set_num_threads() may not have been called
    // when build_index_incremental runs with zero missing genomes (fully-warm run).
    // Without this, OMP defaults to all system CPUs causing massive per-region overhead.
    omp_set_num_threads(cfg_.threads);
#endif

    // Track max similarity to any representative (higher = closer = more redundant)
    // Using similarity instead of distance avoids acos in the hot loop
    std::vector<float> max_sim_to_rep(n, -1.0f);
    std::vector<uint64_t> nearest_rep(n, UINT64_MAX);

    // Similarity function: exact Jaccard for small n (no Nyström), dot product otherwise.
    auto sim_fn = [&](size_t i, size_t j) -> float {
        if (!nystrom_applied_)
            return static_cast<float>(refine_jaccard(embeddings_[i].oph_sig, embeddings_[j].oph_sig));
        return dot_product_simd(store_.row(i), store_.row(j), dim);
    };

    // Precomputed cosine thresholds: dist < threshold ↔ sim > cos(threshold * π)
    // Avoids acos() in every FPS iteration (hot path).
    const float cos_diversity = std::cos(cfg_.diversity_threshold * static_cast<float>(M_PI));
    const float cos_min_rep   = std::cos(cfg_.min_rep_distance * static_cast<float>(M_PI));

    // Precompute length factors using SoA
    std::vector<float> length_factors(n);
    std::vector<uint64_t> sizes(store_.genome_sizes.begin(), store_.genome_sizes.end());
    std::nth_element(sizes.begin(), sizes.begin() + n / 2, sizes.end());
    double median_size = static_cast<double>(sizes[n / 2]);
    median_size = std::max(1.0, median_size);

    for (size_t i = 0; i < n; ++i) {
        length_factors[i] = static_cast<float>(
            std::sqrt(static_cast<double>(store_.genome_sizes[i]) / median_size));
    }

    // Pre-seed pinned representatives (e.g., GTDB reference genomes)
    if (!pinned_rep_paths_.empty()) {
        for (size_t i = 0; i < n; ++i) {
            if (!pinned_rep_paths_.count(store_.paths[i].string())) continue;
            uint64_t rep_id = store_.genome_ids[i];
            representatives.push_back(rep_id);
            const size_t rep_store_i = i;
#if GEODESIC_USE_OMP
            #pragma omp parallel for
#endif
            for (size_t j = 0; j < n; ++j) {
                float sim = sim_fn(rep_store_i, j);
                if (sim > max_sim_to_rep[j]) {
                    max_sim_to_rep[j] = sim;
                    nearest_rep[j] = rep_id;
                }
            }
            max_sim_to_rep[i] = 1.0f;
        }
    }

    // Phase 1: Start with most isolated genome (skip if pinned reps already seeded).
    // Data is sorted by genome_id (deterministic), not isolation score, so we scan
    // for argmax of the composite isolation-quality-length score.
    if (representatives.empty()) {
        size_t first_idx = 0;
        {
            float best = -1.0f;
            std::vector<uint64_t> tmp_sizes(n);
            for (size_t i = 0; i < n; ++i) tmp_sizes[i] = store_.genome_sizes[i];
            auto mid = tmp_sizes.begin() + static_cast<ptrdiff_t>(n / 2);
            std::nth_element(tmp_sizes.begin(), mid, tmp_sizes.end());
            float median_sz = std::max(1.0f, static_cast<float>(*mid));
            for (size_t i = 0; i < n; ++i) {
                if (store_.quality_scores[i] == 0.0f) continue;
                float len = std::sqrt(static_cast<float>(store_.genome_sizes[i]) / median_sz);
                float score = store_.isolation_scores[i] * (store_.quality_scores[i] / 100.0f) * len;
                if (score > best) { best = score; first_idx = i; }
            }
        }
        if (first_idx == n) first_idx = 0;  // fallback: all excluded (shouldn't happen)
        uint64_t first_rep = store_.genome_ids[first_idx];
        representatives.push_back(first_rep);
        if (is_verbose()) {
            spdlog::info("GEODESIC: First representative (most isolated): genome_{} (isolation={:.4f})",
                         first_rep, store_.isolation_scores[first_idx]);
        }

        // Update similarities to first rep
#if GEODESIC_USE_OMP
        #pragma omp parallel for
#endif
        for (size_t i = 0; i < n; ++i) {
            max_sim_to_rep[i] = sim_fn(first_idx, i);
            nearest_rep[i] = first_rep;
        }
        max_sim_to_rep[first_idx] = 1.0f;  // Self-similarity
    }

    // Active set: indices of genomes not yet covered by any representative.
    // A genome is covered when its nearest-rep distance < diversity_threshold.
    // Scanning only active indices avoids O(n) work per FPS iteration once
    // most genomes are covered — yields ~10x speedup on large, clonal taxa.
    std::vector<size_t> active;
    active.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (store_.quality_scores[i] > 0.0f)
            active.push_back(i);
    }
    // Remove newly covered after each rep addition (lambda reused below)
    auto compact_active = [&]() {
        active.erase(
            std::remove_if(active.begin(), active.end(), [&](size_t i) {
                // dist < diversity_threshold ↔ sim > cos_diversity (no acos)
                return max_sim_to_rep[i] > cos_diversity;
            }),
            active.end());
    };
    compact_active();  // After first rep

    // Deterministic tie-break: primary = higher fitness, secondary = lower genome_id.
    // Epsilon absorbs HNSW approximation noise (~1e-5 in dist_proxy) without masking
    // real fitness differences (which differ by >= 1e-3 for biologically distinct genomes).
    static constexpr float kFitnessTieEps = 1e-5f;
    auto fitness_beats = [&](float new_fit, size_t new_idx, float cur_fit, size_t cur_idx) -> bool {
        if (new_fit > cur_fit + kFitnessTieEps) return true;
        if (new_fit >= cur_fit - kFitnessTieEps)
            return store_.genome_ids[new_idx] < store_.genome_ids[cur_idx];
        return false;
    };

    // Phase 2: Batched FPS — select top-B candidates per round to amortize
    // OpenMP launch overhead and improve cache reuse on the update pass.
    static constexpr size_t FPS_BATCH = 16;

    while (!active.empty()) {
        // Compute fitness for all active members in parallel
        std::vector<std::pair<float, size_t>> fit_active(active.size());
#if GEODESIC_USE_OMP
        #pragma omp parallel for
#endif
        for (size_t ai = 0; ai < active.size(); ++ai) {
            size_t i = active[ai];
            float sim = max_sim_to_rep[i];
            // Fast angular distance proxy: sqrt(2(1-sim)) ≈ acos(sim) for sim near 1.
            // Preserves ranking for FPS candidate selection (monotonic in sim).
            float d_proxy = std::sqrt(2.0f * std::max(0.0f, 1.0f - sim));
            fit_active[ai] = {d_proxy * (store_.quality_scores[i] / 100.0f) * length_factors[i], i};
        }

        // Partial sort: bring top-B to front (O(n) average)
        if (fit_active.empty()) break;
        size_t b = std::min(FPS_BATCH, fit_active.size());
        std::nth_element(fit_active.begin(), fit_active.begin() + b, fit_active.end(),
                         [](const auto& x, const auto& y) { return x.first > y.first; });
        std::sort(fit_active.begin(), fit_active.begin() + b,
                  [](const auto& x, const auto& y) { return x.first > y.first; });

        // Stopping criterion: top candidate covered ↔ sim > cos_diversity
        float top_sim = max_sim_to_rep[fit_active[0].second];
        if (top_sim > cos_diversity) {
            if (is_verbose()) {
                float top_dist = std::acos(std::clamp(top_sim, -1.0f, 1.0f)) / static_cast<float>(M_PI);
                spdlog::info("GEODESIC: Diversity saturated (max_dist={:.4f} < threshold={:.4f})",
                             top_dist, cfg_.diversity_threshold);
            }
            break;
        }

        // Build batch: top-B candidates above threshold
        std::vector<size_t> batch_idx;
        std::vector<uint64_t> batch_gids;
        std::vector<const float*> batch_vecs;
        for (size_t k = 0; k < b; ++k) {
            size_t i = fit_active[k].second;
            // dist < diversity_threshold ↔ sim > cos_diversity
            if (max_sim_to_rep[i] > cos_diversity) continue;
            batch_idx.push_back(i);
            batch_gids.push_back(store_.genome_ids[i]);
            batch_vecs.push_back(store_.row(i));
            representatives.push_back(store_.genome_ids[i]);
        }

        // Update only active (uncovered) genomes + compact.
        // After 50% coverage, each iteration scans only the remainder → ~40-50% less total work.
        for (size_t bi : batch_idx) max_sim_to_rep[bi] = 1.0f;

        const size_t nb = batch_idx.size();
        const size_t n_active = active.size();
#if GEODESIC_USE_OMP
        #pragma omp parallel for schedule(static)
#endif
        for (size_t ai = 0; ai < n_active; ++ai) {
            size_t j = active[ai];
            for (size_t k = 0; k < nb; ++k) {
                float sim = sim_fn(batch_idx[k], j);
                if (sim > max_sim_to_rep[j]) {
                    max_sim_to_rep[j] = sim;
                    nearest_rep[j] = batch_gids[k];
                }
            }
        }
        compact_active();
    }

    // Improvement 6: FPS borderline refinement.
    // Genomes whose embedding distance to nearest rep is in [lo_thresh, diversity_threshold)
    // might be falsely classified as covered due to Nyström approximation error (≈3/√d).
    // For each such genome, check top-K reps by embedding similarity (not just the single
    // nearest — Nyström error can displace the nearest-rep assignment) using dual-sketch
    // averaged OPH Jaccard. Promote only if no top-K rep truly covers it.
    if (nystrom_applied_ && !embeddings_.empty()) {
        const float eps_nystrom = std::min(1.5f / std::sqrt(static_cast<float>(runtime_dim_)), 0.3f);
        const float lo_thresh = cfg_.diversity_threshold * (1.0f - eps_nystrom);
        const float cos_lo = std::cos(lo_thresh * static_cast<float>(M_PI));

        std::unordered_set<uint64_t> rep_set_ids(representatives.begin(), representatives.end());

        // Build list of rep store indices for top-K embedding scan.
        std::vector<size_t> rep_store_idx;
        rep_store_idx.reserve(representatives.size());
        for (size_t i = 0; i < n; ++i)
            if (rep_set_ids.count(store_.genome_ids[i]))
                rep_store_idx.push_back(i);

        // Number of nearby reps to verify with exact OPH (3 balances precision vs cost).
        static constexpr int kBorderlineCheckK = 3;

        // Lazily materialize sig2 for borderline candidates + all current reps.
        // Both sides of each dual-sketch check need sig2; reps are checked against
        // multiple borderline genomes so materializing them all up front is efficient.
        {
            std::vector<size_t> to_materialize;
            to_materialize.insert(to_materialize.end(), rep_store_idx.begin(), rep_store_idx.end());
            for (size_t i = 0; i < n; ++i) {
                if (rep_set_ids.count(store_.genome_ids[i])) continue;
                if (store_.quality_scores[i] == 0.0f) continue;
                float sim  = max_sim_to_rep[i];
                // dist in [lo_thresh, diversity_threshold) ↔ sim in (cos_diversity, cos_lo]
                if (sim <= cos_diversity && sim >= cos_lo)
                    to_materialize.push_back(i);
            }
            // Deduplicate before re-reading NFS
            std::sort(to_materialize.begin(), to_materialize.end());
            to_materialize.erase(std::unique(to_materialize.begin(), to_materialize.end()),
                                 to_materialize.end());
            materialize_sig2_for_indices(to_materialize);
        }

        std::vector<size_t> newly_added;
        for (size_t i = 0; i < n; ++i) {
            if (rep_set_ids.count(store_.genome_ids[i])) continue;
            if (store_.quality_scores[i] == 0.0f) continue;

            float sim  = max_sim_to_rep[i];

            // Only check borderline: dist in [lo_thresh, diversity_threshold)
            // ↔ sim in (cos_diversity, cos_lo]
            if (sim > cos_diversity || sim < cos_lo) continue;

            // Find top-K reps by embedding dot product (linear scan, cheap relative to OPH).
            const float* vi = store_.row(i);
            int k_check = std::min(kBorderlineCheckK, static_cast<int>(rep_store_idx.size()));
            std::vector<std::pair<float, size_t>> top_reps(rep_store_idx.size());
            for (size_t r = 0; r < rep_store_idx.size(); ++r)
                top_reps[r] = {dot_product_simd(vi, store_.row(rep_store_idx[r]), dim), rep_store_idx[r]};
            std::partial_sort(top_reps.begin(), top_reps.begin() + k_check, top_reps.end(),
                              [](const auto& a, const auto& b) { return a.first > b.first; });

            // Verify each top-K rep with dual-sketch averaged OPH Jaccard.
            bool covered = false;
            for (int ki = 0; ki < k_check && !covered; ++ki) {
                size_t rep_emb_i = top_reps[ki].second;
                const auto& sig_a  = embeddings_[i].oph_sig;
                const auto& sig_b  = embeddings_[rep_emb_i].oph_sig;
                if (sig_a.empty() || sig_b.empty()) continue;
                double jac_exact = refine_jaccard(sig_a, sig_b);
                if (!embeddings_[i].oph_sig2.empty() && !embeddings_[rep_emb_i].oph_sig2.empty()) {
                    double jac2 = refine_jaccard(embeddings_[i].oph_sig2, embeddings_[rep_emb_i].oph_sig2);
                    jac_exact = (jac_exact + jac2) * 0.5;
                }
                // dist_exact < diversity_threshold ↔ jac_exact > cos_diversity
                if (static_cast<float>(jac_exact) > cos_diversity) covered = true;
            }

            if (!covered) {
                // No top-K rep truly covers this genome → add as representative.
                representatives.push_back(store_.genome_ids[i]);
                rep_set_ids.insert(store_.genome_ids[i]);
                rep_store_idx.push_back(i);
                newly_added.push_back(i);
                max_sim_to_rep[i] = 1.0f;
            }
        }
        if (!newly_added.empty()) {
            spdlog::info("GEODESIC: Borderline refinement added {} representatives "
                         "(Nyström eps={:.3f}, K={})", newly_added.size(), eps_nystrom, kBorderlineCheckK);
            // Batched similarity update: one parallel pass over all n, inner loop over newly added.
            std::vector<const float*> new_vecs;
            std::vector<uint64_t> new_gids;
            new_vecs.reserve(newly_added.size());
            new_gids.reserve(newly_added.size());
            for (size_t bi : newly_added) {
                new_vecs.push_back(store_.row(bi));
                new_gids.push_back(store_.genome_ids[bi]);
            }
            const size_t nb_new = new_vecs.size();
#if GEODESIC_USE_OMP
            #pragma omp parallel for num_threads(cfg_.threads)
#endif
            for (size_t j = 0; j < n; ++j) {
                const float* vj = store_.row(j);
                for (size_t k = 0; k < nb_new; ++k) {
                    float s = dot_product_simd(new_vecs[k], vj, dim);
                    if (s > max_sim_to_rep[j]) {
                        max_sim_to_rep[j] = s;
                        nearest_rep[j] = new_gids[k];
                    }
                }
            }
        }
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

    // Iterative path-halving Union-Find — no recursion risk for large r.
    auto find = [&](size_t x) -> size_t {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];  // path halving
            x = parent[x];
        }
        return x;
    };

    auto unite = [&](size_t x, size_t y) {
        size_t px = find(x), py = find(y);
        if (px != py) parent[px] = py;
    };

    std::unordered_set<uint64_t> rep_set(representatives.begin(), representatives.end());

    if (!nystrom_applied_) {
        // Exact mode (small n): brute-force O(r²) Jaccard
        std::unordered_map<uint64_t, size_t> gid_to_idx;
        gid_to_idx.reserve(n);
        for (size_t i = 0; i < n; ++i)
            gid_to_idx[store_.genome_ids[i]] = i;

        for (size_t i = 0; i < representatives.size(); ++i) {
            size_t ri = gid_to_idx[representatives[i]];
            for (size_t j = i + 1; j < representatives.size(); ++j) {
                size_t rj = gid_to_idx[representatives[j]];
                double jac = refine_jaccard(embeddings_[ri].oph_sig, embeddings_[rj].oph_sig);
                // dist < min_rep_distance ↔ jac > cos_min_rep
                if (static_cast<float>(jac) > cos_min_rep) {
                    unite(i, j);
                    spdlog::debug("GEODESIC: Merged rep {} into rep {} (jaccard={:.4f})", j, i, jac);
                }
            }
        }
    } else {
        // Nyström mode (large n): parallel HNSW searches then serial union-find.
        // index_->search() with a read-only filter set is thread-safe.
        const size_t k_neighbors = std::min(representatives.size(), size_t{64});
        std::vector<std::pair<size_t, size_t>> merge_pairs;

#if GEODESIC_USE_OMP
        std::mutex merge_mutex;
        #pragma omp parallel
        {
            std::vector<std::pair<size_t, size_t>> local_pairs;
            #pragma omp for nowait
            for (size_t i = 0; i < representatives.size(); ++i) {
                uint64_t rep_i = representatives[i];
                auto row_it = gid_to_row_.find(rep_i);
                if (row_it == gid_to_row_.end()) continue;
                auto neighbors = index_->search(embeddings_[row_it->second].vector, k_neighbors, &rep_set);
                for (const auto& [neighbor_id, dist] : neighbors) {
                    if (neighbor_id == rep_i || dist >= cfg_.min_rep_distance) continue;
                    auto it = rep_to_idx.find(neighbor_id);
                    if (it != rep_to_idx.end()) {
                        size_t j = it->second;
                        if (i < j) local_pairs.emplace_back(i, j);
                    }
                }
            }
            std::lock_guard<std::mutex> lk(merge_mutex);
            merge_pairs.insert(merge_pairs.end(), local_pairs.begin(), local_pairs.end());
        }
#else
        for (size_t i = 0; i < representatives.size(); ++i) {
            uint64_t rep_i = representatives[i];
            auto row_it = gid_to_row_.find(rep_i);
            if (row_it == gid_to_row_.end()) continue;
            auto neighbors = index_->search(embeddings_[row_it->second].vector, k_neighbors, &rep_set);
            for (const auto& [neighbor_id, dist] : neighbors) {
                if (neighbor_id == rep_i || dist >= cfg_.min_rep_distance) continue;
                auto it = rep_to_idx.find(neighbor_id);
                if (it != rep_to_idx.end()) {
                    size_t j = it->second;
                    if (i < j) merge_pairs.emplace_back(i, j);
                }
            }
        }
#endif
        for (const auto& [x, y] : merge_pairs) {
            unite(x, y);
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

        // Identify merged-away rep IDs so we only recompute affected genomes.
        std::unordered_set<uint64_t> refined_set(refined_reps.begin(), refined_reps.end());
        std::unordered_set<uint64_t> merged_away;
        for (uint64_t r : representatives)
            if (!refined_set.count(r)) merged_away.insert(r);

        representatives = std::move(refined_reps);

        rep_set.clear();
        rep_set.insert(representatives.begin(), representatives.end());

        // Selective recompute: only genomes whose nearest_rep was merged away.
        // Genomes assigned to surviving reps keep their current assignment.
#if GEODESIC_USE_OMP
        #pragma omp parallel for
#endif
        for (size_t i = 0; i < n; ++i) {
            if (rep_set.count(store_.genome_ids[i])) {
                max_sim_to_rep[i] = 1.0f;
                nearest_rep[i] = store_.genome_ids[i];
            } else if (merged_away.count(nearest_rep[i])) {
                float max_sim = -1.0f;
                uint64_t best_rep = representatives[0];

                for (uint64_t r : representatives) {
                    auto rit = gid_to_row_.find(r);
                    if (rit == gid_to_row_.end()) continue;
                    float sim = sim_fn(i, rit->second);
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
    const double J_margin = 3.0 / std::sqrt(static_cast<double>(runtime_dim_));

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
        J_est = std::clamp(J_est, 0.0, 1.0);

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

    // Join async DB save (launched in build_index after Nyström). The save does not read
    // oph_sig2 (lazy design: sig2 is not persisted), so there is no sig2 race. Joining
    // ensures the DB write completes before we return and is safe to do here.
    if (async_save_future_.valid()) {
        try { async_save_future_.get(); } catch (const std::exception& e) {
            spdlog::error("GEODESIC: async save failed in select_representatives: {}", e.what());
        }
    }
    // Free sig2 for the few genomes that had it materialized (anchors + borderline candidates).
    for (auto& emb : embeddings_) {
        std::vector<uint16_t>().swap(emb.oph_sig2);
    }

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
        if (c_psi <= 0.0f) return 0.5f;
        return std::pow(2.0f, -(total / static_cast<float>(T)) / c_psi);
    }
};

std::vector<GeodesicDerep::ContaminationCandidate>
GeodesicDerep::detect_contamination_candidates(float z_threshold) const {
    if (embeddings_.size() <= SMALL_N_THRESHOLD) return {};

    size_t n   = embeddings_.size();
    size_t dim = static_cast<size_t>(runtime_dim_);

    // Isolation score z-score for nn_outlier: Welford online mean/std over all genomes.
    // Flagging: nn_outlier = isolation_score > mean + z_threshold * std.
    // This is population-relative: a genome is flagged if its k-NN distances are anomalously
    // high relative to the taxon's own distribution, regardless of absolute ANI scale.
    double iso_mean = 0.0, iso_m2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double v = embeddings_[i].isolation_score;
        const double delta = v - iso_mean;
        iso_mean += delta / static_cast<double>(i + 1);
        iso_m2   += delta * (v - iso_mean);
    }
    const double iso_std = (n > 1) ? std::sqrt(iso_m2 / static_cast<double>(n - 1)) : 1.0;
    const float nn_threshold = static_cast<float>(iso_mean + z_threshold * iso_std);

    // Genome size z-score: informational only, NOT used as a flagging criterion.
    // MAGs are naturally smaller (incomplete assemblies) and would always appear as size
    // outliers. Genome size does not reliably indicate contamination.
    double size_mean = 0.0, size_m2 = 0.0;
    size_t size_n = 0;
    for (size_t i = 0; i < n; ++i) {
        if (store_.genome_sizes[i] == 0) continue;
        const double v = static_cast<double>(store_.genome_sizes[i]);
        ++size_n;
        const double delta = v - size_mean;
        size_mean += delta / static_cast<double>(size_n);
        size_m2   += delta * (v - size_mean);
    }
    const double std_sz = (size_n > 1) ? std::sqrt(size_m2 / static_cast<double>(size_n - 1)) : 1.0;

    // k-mer diversity z-score (n_real_bins / kbp): informational, NOT used for flagging.
    // Chimeric assemblies have k-mers from two organisms → anomalously high diversity per bp.
    double kmer_div_mean = 0.0, kmer_div_m2 = 0.0;
    size_t kmer_div_n = 0;
    for (size_t i = 0; i < n; ++i) {
        if (embeddings_[i].genome_size == 0) continue;
        const double div = static_cast<double>(embeddings_[i].n_real_bins)
                         / (embeddings_[i].genome_size / 1000.0);
        ++kmer_div_n;
        const double delta = div - kmer_div_mean;
        kmer_div_mean += delta / static_cast<double>(kmer_div_n);
        kmer_div_m2   += delta * (div - kmer_div_mean);
    }
    const double kmer_div_std = (kmer_div_n > 1)
        ? std::sqrt(kmer_div_m2 / static_cast<double>(kmer_div_n - 1)) : 1.0;

    // Centroid for informational centroid_distance field only
    std::vector<float> centroid(dim, 0.0f);
    for (const auto& emb : embeddings_)
        for (size_t d = 0; d < dim; ++d)
            centroid[d] += emb.vector[d];
    float cn = static_cast<float>(n);
    for (float& v : centroid) v /= cn;
    float cnorm = dot_product_simd(centroid.data(), centroid.data(), dim);
    cnorm = std::sqrt(cnorm);
    if (cnorm > 1e-10f) for (float& v : centroid) v /= cnorm;

    // Flagging criterion: nn_outlier only.
    // isolation_score > mean + z_threshold * std → statistically anomalous k-NN distance
    // → likely misassigned taxonomy.

    std::vector<ContaminationCandidate> candidates;
    for (size_t i = 0; i < n; ++i) {
        const bool is_nn_outlier = embeddings_[i].isolation_score > nn_threshold;

        // kmer_div z-score: informational, NOT used for flagging.
        const float kmer_div_z = (embeddings_[i].genome_size > 0 && kmer_div_std > 1e-9)
            ? static_cast<float>(
                (static_cast<double>(embeddings_[i].n_real_bins)
                 / (embeddings_[i].genome_size / 1000.0) - kmer_div_mean) / kmer_div_std)
            : 0.0f;

        if (!is_nn_outlier) continue;

        const float sz_z = (store_.genome_sizes[i] > 0 && std_sz > 1e-9)
            ? static_cast<float>(
                (static_cast<double>(store_.genome_sizes[i]) - size_mean) / std_sz)
            : 0.0f;
        const bool is_size_outlier = std::abs(sz_z) > z_threshold;
        std::string reason;
        if (is_nn_outlier && is_size_outlier) reason = "nn_outlier+size_outlier";
        else if (is_nn_outlier) reason = "nn_outlier";
        else reason = "size_outlier";

        ContaminationCandidate c;
        c.genome_id            = embeddings_[i].genome_id;
        c.centroid_distance    = angular_distance(embeddings_[i].vector, centroid);
        c.isolation_score      = embeddings_[i].isolation_score;
        c.anomaly_score        = embeddings_[i].isolation_score;
        c.genome_size_zscore   = sz_z;
        c.nn_outlier           = is_nn_outlier;
        c.kmer_div_zscore      = kmer_div_z;
        c.margin_to_threshold  = embeddings_[i].isolation_score - nn_threshold;
        c.flag_reason          = std::move(reason);
        c.path                 = embeddings_[i].path;
        candidates.push_back(c);
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.anomaly_score > b.anomaly_score; });

    const size_t n_misassigned = std::count_if(candidates.begin(), candidates.end(),
                                                [](const auto& c) { return c.nn_outlier; });
    if (!candidates.empty())
        spdlog::info("GEODESIC: {} contamination candidates ({} misassigned, "
                     "nn_thr={:.4f}={:.3f}+{}*{:.3f})",
                     candidates.size(), n_misassigned,
                     nn_threshold, iso_mean, z_threshold, iso_std);

    return candidates;
}

std::vector<std::pair<std::filesystem::path, uint64_t>>
GeodesicDerep::get_genome_sizes() const {
    std::vector<std::pair<std::filesystem::path, uint64_t>> result;
    result.reserve(embeddings_.size());
    for (const auto& emb : embeddings_)
        result.emplace_back(emb.path, emb.genome_size);
    return result;
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
        const double d = std::clamp(dist, 0.0, 1.0);
        const double J = std::clamp(std::cos(d * M_PI), 0.0, 1.0);
        const double q = (J > 0.0) ? (2.0 * J / (1.0 + J)) : 0.0;
        const double ani = (q > 0.0) ? std::pow(q, 1.0 / cfg_.kmer_size) : 0.0;
        return std::clamp(ani, 0.0, 1.0);
    };

    // --- Coverage: for each genome, find distance to nearest representative ---
    std::vector<double> coverage_dists(embeddings_.size());
    std::vector<size_t> nearest_rep_idx(embeddings_.size(), SIZE_MAX);
    double coverage_sum = 0.0;
    double coverage_min = std::numeric_limits<double>::max();
    double coverage_max = 0.0;
    size_t n_valid = 0;

#if GEODESIC_USE_OMP
    #pragma omp parallel for reduction(+:coverage_sum,n_valid) reduction(min:coverage_min) reduction(max:coverage_max)
#endif
    for (size_t i = 0; i < embeddings_.size(); ++i) {
        const float* query = store_.row(i);
        // Skip zero-norm embeddings (failed sketch: empty/corrupt FASTA)
        if (dot_product_simd(query, query, runtime_dim_) < 0.01f) {
            coverage_dists[i] = std::numeric_limits<double>::quiet_NaN();
            continue;
        }
        ++n_valid;

        float max_sim = -1.0f;
        size_t best_rep = SIZE_MAX;
        for (size_t rep_idx : rep_indices) {
            const float* rep = store_.row(rep_idx);
            float sim = dot_product_simd(query, rep, runtime_dim_);
            if (sim > max_sim) { max_sim = sim; best_rep = rep_idx; }
        }

        // Convert to angular distance once per genome (output path only)
        double min_dist = std::acos(std::clamp(max_sim, -1.0f, 1.0f)) / M_PI;
        coverage_dists[i] = min_dist;
        nearest_rep_idx[i] = best_rep;
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
            if (std::isfinite(coverage_dists[i]))
                valid_dists.push_back(coverage_dists[i]);
        }
        if (!valid_dists.empty()) {
            std::sort(valid_dists.begin(), valid_dists.end());
            size_t nv = valid_dists.size();
            metrics.coverage_p5_dist  = valid_dists[nv *  5 / 100];
            metrics.coverage_p95_dist = valid_dists[nv * 95 / 100];
        }
    }

    // OPH containment for ANI threshold counts.
    //
    // Validation on Ferruginibacter (1036 genomes, 137 reps):
    //   FastANI ground truth:       below_95 = 473
    //   Skani ground truth:         below_95 = 355-459
    //   OPH containment (this):     below_95 = ~470   ← matches
    //   Nyström embedding:          below_95 = 92     ← 5× too optimistic
    //
    // The embedding-based estimate was wrong: Nyström transitivity over-smooths
    // distances, making distant genomes appear artificially close (false "covered").
    // OPH global containment (n11/nq) correctly reflects genome-wide k-mer similarity
    // and matches alignment-based ANI tools for this purpose.
    //
    // J threshold from Mash formula: ANI = (2J/(1+J))^(1/k) → J = ANI^k / (2 - ANI^k)
    {
        const int m = cfg_.sketch_size;
        const double k = static_cast<double>(cfg_.kmer_size);

        int cnt99 = 0, cnt98 = 0, cnt97 = 0, cnt95 = 0;
#if GEODESIC_USE_OMP
        #pragma omp parallel for reduction(+:cnt99,cnt98,cnt97,cnt95)
#endif
        for (size_t i = 0; i < embeddings_.size(); ++i) {
            if (!std::isfinite(coverage_dists[i])) continue;
            const size_t ridx = nearest_rep_idx[i];
            if (ridx == SIZE_MAX) continue;
            const auto& qe = embeddings_[i];
            const auto& re = embeddings_[ridx];

            double ani;
            if (qe.oph_sig.empty() || re.oph_sig.empty() ||
                qe.real_bins_mask.empty() || re.real_bins_mask.empty()) {
                ani = dist_to_ani(coverage_dists[i]);
            } else {
                int n11 = 0, n_union = 0;
                for (int t = 0; t < m; ++t) {
                    const bool q_real = (qe.real_bins_mask[t >> 6] >> (t & 63)) & 1ULL;
                    const bool r_real = (re.real_bins_mask[t >> 6] >> (t & 63)) & 1ULL;
                    if (!q_real && !r_real) continue;
                    n_union++;
                    if (q_real && r_real && qe.oph_sig[t] == re.oph_sig[t]) n11++;
                }
                if (n_union == 0) continue;
                // Symmetric Jaccard → ANI: ANI = (2J/(1+J))^(1/k)
                const double j = static_cast<double>(n11) / n_union;
                ani = (j > 0.0) ? std::pow(2.0 * j / (1.0 + j), 1.0 / k) : 0.0;
            }
            if (ani < 0.99) cnt99++;
            if (ani < 0.98) cnt98++;
            if (ani < 0.97) cnt97++;
            if (ani < 0.95) cnt95++;
        }
        metrics.coverage_below_99 = cnt99;
        metrics.coverage_below_98 = cnt98;
        metrics.coverage_below_97 = cnt97;
        metrics.coverage_below_95 = cnt95;
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
                    float sim = dot_product_simd(a, b, runtime_dim_);
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
                float sim = dot_product_simd(a, b, runtime_dim_);
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

    // Clear reuse-sensitive state from prior invocations
    embeddings_.clear();
    embeddings_.reserve(genomes.size());
    failed_reads_.clear();
    last_representative_ids_.clear();
    nystrom_applied_ = false;

    // Build path → accession map (stem-based extraction)
    path_to_accession_.clear();
    std::vector<std::string> all_accessions;
    all_accessions.reserve(genomes.size());
    for (const auto& p : genomes) {
        std::string acc = p.stem().string();
        if (acc.size() > 8 && acc.substr(acc.size() - 8) == "_genomic")
            acc = acc.substr(0, acc.size() - 8);
        path_to_accession_[p.string()] = acc;
        all_accessions.push_back(acc);
    }

    // Find which genomes are already embedded
    auto embedded_set = store.get_embedded_set(taxonomy);
    std::vector<std::filesystem::path> missing_genomes;
    missing_genomes.reserve(genomes.size());
    for (const auto& p : genomes) {
        if (embedded_set.find(path_to_accession_[p.string()]) == embedded_set.end())
            missing_genomes.push_back(p);
    }

    // Load cached embeddings (includes OPH signatures for Nyström reconstruction)
    auto existing = store.load_embeddings(taxonomy);

    spdlog::info("GEODESIC: {} genomes total, {} already embedded, {} to embed",
                 genomes.size(), existing.size(), missing_genomes.size());

    // Embed missing genomes from NFS
    size_t newly_embedded = 0;
    if (!missing_genomes.empty()) {
        if (is_verbose()) spdlog::info("GEODESIC: embedding {} new genomes (dim={}, k={}, threads={})",
                     missing_genomes.size(), cfg_.embedding_dim, cfg_.kmer_size, cfg_.threads);

        std::vector<GenomeEmbedding> new_embeddings(missing_genomes.size());
        ProgressCounter embed_progress(missing_genomes.size(), "GEODESIC: embedding");

        const int io_threads = (cfg_.io_threads > 0) ? cfg_.io_threads
                                                       : cfg_.threads;
#if GEODESIC_USE_OMP
        omp_set_num_threads(cfg_.threads);
        {
        std::counting_semaphore<> io_sem(io_threads);
        #pragma omp parallel for schedule(dynamic, 1)
        for (size_t i = 0; i < missing_genomes.size(); ++i) {
            io_sem.acquire();
            new_embeddings[i] = embed_genome(missing_genomes[i], existing.size() + i);
            io_sem.release();
            auto qit = quality_scores.find(missing_genomes[i].string());
            if (qit != quality_scores.end())
                new_embeddings[i].quality_score = static_cast<float>(qit->second);
            embed_progress.increment();
        }
        }
#else
        for (size_t i = 0; i < missing_genomes.size(); ++i) {
            new_embeddings[i] = embed_genome(missing_genomes[i], existing.size() + i);
            auto qit = quality_scores.find(missing_genomes[i].string());
            if (qit != quality_scores.end())
                new_embeddings[i].quality_score = static_cast<float>(qit->second);
            embed_progress.increment();
        }
#endif
        embed_progress.finish();

        // Save new embeddings to store (with OPH signatures for future warm runs)
        std::vector<db::GenomeEmbedding> store_embeddings;
        store_embeddings.reserve(new_embeddings.size());
        for (const auto& emb : new_embeddings) {
            db::GenomeEmbedding se;
            se.accession = path_to_accession_[emb.path.string()];
            se.taxonomy = taxonomy;
            se.file_path = emb.path;
            se.embedding = emb.vector;   // zeros (placeholder)
            se.oph_sig = emb.oph_sig;    // raw OPH hashes
            // oph_sig2 not persisted (lazy; re-materialized for anchors/borderline on each run)
            se.real_bins_mask = emb.real_bins_mask;
            se.n_real_bins = emb.n_real_bins;
            se.isolation_score = emb.isolation_score;
            se.quality_score = emb.quality_score;
            se.genome_size = emb.genome_size;
            store_embeddings.push_back(std::move(se));
        }
        store.insert_embeddings(store_embeddings);
        newly_embedded = new_embeddings.size();

        for (auto& emb : new_embeddings)
            embeddings_.push_back(std::move(emb));
    }

    // Convert cached embeddings to internal format (oph_sig loaded from BLOB)
    for (const auto& se : existing) {
        GenomeEmbedding emb;
        emb.genome_id = embeddings_.size();
        emb.vector.assign(cfg_.embedding_dim, 0.0f);  // placeholder, Nyström will replace
        emb.oph_sig = se.oph_sig;                      // loaded from DB — no NFS read needed
        // oph_sig2 not loaded (lazy; will be re-materialized by apply_nystrom_embeddings)
        emb.real_bins_mask = se.real_bins_mask;        // bitmask for coverage metrics
        emb.n_real_bins = se.n_real_bins;
        emb.isolation_score = se.isolation_score;
        emb.quality_score = se.quality_score;
        emb.genome_size = se.genome_size;
        emb.path = se.file_path;
        embeddings_.push_back(std::move(emb));
    }

    // Reassign genome_ids sequentially so genome_id == index
    for (size_t i = 0; i < embeddings_.size(); ++i)
        embeddings_[i].genome_id = static_cast<uint64_t>(i);

    size_t n = embeddings_.size();
    if (n == 0) return 0;

    // Build canonical ID → row map
    gid_to_row_.clear();
    gid_to_row_.reserve(n);
    for (size_t i = 0; i < n; ++i)
        gid_to_row_[embeddings_[i].genome_id] = i;

    // Copy metadata to SoA store (data already zeroed by resize → value-init)
    store_.resize(n, cfg_.embedding_dim);
    for (size_t i = 0; i < n; ++i) {
        store_.genome_ids[i] = embeddings_[i].genome_id;
        store_.isolation_scores[i] = embeddings_[i].isolation_score;
        store_.quality_scores[i] = embeddings_[i].quality_score;
        store_.genome_sizes[i] = embeddings_[i].genome_size;
        store_.paths[i] = embeddings_[i].path;
    }

    // Run Nyström on all embeddings (new + cached), using OPH signatures.
    // This reconstructs proper spectral embeddings without any NFS genome reads.
    nystrom_applied_ = false;
    if (n > SMALL_N_THRESHOLD) {
        apply_nystrom_embeddings();  // sets nystrom_applied_ = true, updates store_.data
    } else {
        // Small n: use exact OPH Jaccard (do NOT set nystrom_applied_ = true)
        nystrom_applied_ = false;
    }

    // Build HNSW index from Nyström (or placeholder) vectors
    if (n > SMALL_N_THRESHOLD) {
        index_->build(embeddings_, cfg_.hnsw_m, cfg_.hnsw_ef_construction, cfg_.hnsw_ef_search);
        if (is_verbose()) spdlog::info("GEODESIC: HNSW index built ({} embeddings, M={}, ef={})",
                     n, cfg_.hnsw_m, cfg_.hnsw_ef_construction);
    }

    return newly_embedded;
}

void GeodesicDerep::save_embeddings_async(db::EmbeddingStore& store, const std::string& taxonomy,
                                           std::vector<std::vector<float>>&& vec_snap) {
    if (embeddings_.empty()) return;
    const size_t n = embeddings_.size();
    // Stream in batches of 1000: ~80 MB intermediate per batch instead of copying all 29 GB.
    static constexpr size_t BATCH = 1000;
    std::vector<db::GenomeEmbedding> batch;
    batch.reserve(BATCH);

    auto flush = [&]() {
        if (batch.empty()) return;
        if (!store.insert_embeddings(batch))
            spdlog::error("GEODESIC: async save batch failed ({} embeddings)", batch.size());
        batch.clear();
    };

    for (size_t i = 0; i < n; ++i) {
        const auto& emb = embeddings_[i];
        db::GenomeEmbedding se;
        auto it = path_to_accession_.find(emb.path.string());
        if (it != path_to_accession_.end()) {
            se.accession = it->second;
        } else {
            se.accession = emb.path.stem().string();
            if (se.accession.size() > 8 &&
                se.accession.substr(se.accession.size() - 8) == "_genomic")
                se.accession = se.accession.substr(0, se.accession.size() - 8);
        }
        se.taxonomy        = taxonomy;
        se.file_path       = emb.path;
        se.embedding       = std::move(vec_snap[i]);  // move: no copy of float vec
        se.oph_sig         = emb.oph_sig;             // copy: oph_sig not modified by Nyström
        // oph_sig2 is NOT persisted: it is lazily re-materialized for anchors/borderline
        // on each run, so there is no need to store it and no risk of async read/write races.
        se.real_bins_mask  = emb.real_bins_mask;
        se.n_real_bins     = emb.n_real_bins;
        se.isolation_score = emb.isolation_score;
        se.quality_score   = emb.quality_score;
        se.genome_size     = emb.genome_size;
        batch.push_back(std::move(se));
        if (batch.size() == BATCH) flush();
    }
    flush();
    spdlog::info("GEODESIC: Saved {} embeddings to store (async)", n);
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
        se.embedding = emb.vector;          // placeholder zeros (Nyström vectors vary in dim)
        se.oph_sig = emb.oph_sig;           // raw OPH hashes — key for warm-run Nyström
        // oph_sig2 not persisted (lazy; re-materialized for anchors/borderline on each run)
        se.real_bins_mask = emb.real_bins_mask;
        se.n_real_bins = emb.n_real_bins;
        se.isolation_score = emb.isolation_score;
        se.quality_score = emb.quality_score;
        se.genome_size = emb.genome_size;
        store_embeddings.push_back(std::move(se));
    }

    if (!store.insert_embeddings(store_embeddings))
        spdlog::error("GEODESIC: Failed to save {} embeddings to store", store_embeddings.size());
    else
        spdlog::info("GEODESIC: Saved {} embeddings to store", store_embeddings.size());
}

} // namespace derep
