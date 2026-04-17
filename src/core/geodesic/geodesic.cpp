#include "core/geodesic/geodesic.hpp"
#include "core/sketch/minhash.hpp"
#include "core/progress.hpp"
#include "core/logging.hpp"
#include "core/pack_reader.hpp"
#include <genopack/archive.hpp>
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
    uint64_t seed_ = 42;
    std::unique_ptr<hnswlib::InnerProductSpace> space;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index;
    std::vector<GenomeEmbedding>* embeddings;
    int ef_search;

    void build(std::vector<GenomeEmbedding>& embs, int M, int ef_construction, int ef_search_value, int n_threads = 1) {
        embeddings = &embs;
        if (embs.empty()) return;

        ef_search = std::max(1, ef_search_value);
        dim = embs[0].vector.size();
        space = std::make_unique<hnswlib::InnerProductSpace>(dim);
        index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
            space.get(), embs.size(), M, ef_construction, /*seed=*/seed_);
        index->setEf(ef_search);
        // Sequential insertion: hnswlib parallel addPoint races on internal graph
        // structure, producing different topology on each run even with fixed seed.
        // Sequential insertion is fully deterministic and only ~2× slower at M=16
        // (our large-N default), where each insertion touches a small neighbourhood.
        for (size_t i = 0; i < embs.size(); ++i)
            index->addPoint(embs[i].vector.data(), i);
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
    if (is_verbose()) spdlog::info("ANICalibrator: fitted on {} samples, grid_size={}", samples.size(), grid_size);
}

ANICalibrator::Bounds ANICalibrator::predict(double embedding_distance) const {
    if (!fitted_ || distance_grid_.empty()) {
        // Calibration-free formula: ANI = (2J/(1+J))^(1/k), J ≈ cos(π*d)
        // Derived from Mash chain: 2J/(1+J) = ANI^k, k=21
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
    int threads,
    uint64_t seed) {

    // Helper: embed a set of genome indices and return index→embedding map.
    // Uses OPH+CountSketch projection: E[dot(u,v)] = J(A,B).
    auto embed_sample = [&](const std::vector<size_t>& indices,
                             int kmer_size, int sketch_size, int embedding_dim)
        -> std::unordered_map<size_t, std::vector<float>>
    {
        MinHasher hasher({ .kmer_size = kmer_size, .sketch_size = sketch_size,
                           .syncmer_s = 0, .seed = seed });

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

    CalibratedParams params{ 21, 256, 4096 };

    if (genomes.size() < 2) return params;

    // ----------------------------------------------------------------
    // Step 1: sample random pairs + unique genome indices
    // ----------------------------------------------------------------
    std::mt19937_64 rng(seed);
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
    // Step 2: first pass with default params (k=21, sketch=4k, dim=256)
    //         to measure spread and choose final tier
    // ----------------------------------------------------------------
    const int k0 = 21, s0 = 4096, d0 = 256;
    auto cache0 = embed_sample(idx_vec, k0, s0, d0);
    auto dists0  = compute_distances(pairs, cache0);

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
        // Very clonal (all pairs >99% ANI): large sketch/dim for resolution.
        // k=31 is NOT used here: with k=31 the Nyström embedding collapses all
        // within-species genomes into angular distances < 0.01, causing FPS to
        // select exactly 1 representative for 100k+ genome collections.
        // k=21 with sketch=20000/dim=512 preserves sufficient resolution while
        // keeping the embedding spread that FPS needs to function correctly.
        kmer = 21; sketch_size = 20000; embedding_dim = 512;
    } else if (ani_p95_sample >= 95.0) {
        // Typical species (95–99% ANI): standard resolution
        kmer = 21; sketch_size = 10000; embedding_dim = 256;
    } else {
        // Very diverse or cross-genus (<95% ANI in sample)
        kmer = 16; sketch_size = 5000; embedding_dim = 128;
    }

    params.kmer_size     = kmer;
    params.embedding_dim = embedding_dim;
    params.sketch_size   = sketch_size;

    if (is_verbose()) {
        spdlog::info("GEODESIC: Calibration — {} pairs, spread={:.3f}, tier k={}/dim={}/sketch={}",
                     pairs.size(), spread, kmer, embedding_dim, sketch_size);
    }

    return params;
}

// GeodesicDerep implementation

GeodesicDerep::GeodesicDerep(Config cfg)
    : cfg_(std::move(cfg))
    , runtime_dim_(cfg_.embedding_dim)
    , index_(std::make_unique<HNSWIndex>()) {
}

GeodesicDerep::~GeodesicDerep() = default;

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
        .seed = cfg_.seed
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
        .seed = cfg_.seed
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

    const MinHasher::Config cfg2{
        .kmer_size  = cfg_.kmer_size,
        .sketch_size = cfg_.sketch_size,
        .syncmer_s  = cfg_.syncmer_s,
        .seed       = cfg_.seed + 1  // sig2 must differ from sig1 (seed)
    };

    // GPK path: shard-batched fetch + parallel sketch — zero NFS reads.
    if (gpk_reader_ && !gpk_accessions_.empty()) {
        const size_t n_need = need.size();
        if (is_verbose()) spdlog::info("GEODESIC: loading dense k-mer signatures for {} genomes (GPK shard-batch)",
                                       n_need);

        // Collect accessions for the needed indices.
        std::vector<std::string> acc_batch;
        acc_batch.reserve(n_need);
        std::unordered_map<size_t, size_t> out_to_emb;  // batch output idx → embeddings_ idx
        for (size_t k = 0; k < n_need; ++k) {
            const size_t i = need[k];
            if (i < gpk_accessions_.size()) {
                out_to_emb[acc_batch.size()] = i;
                acc_batch.push_back(gpk_accessions_[i]);
            }
        }

        gpk_reader_->visit_shard_batches(acc_batch,
            [&](genopack::ArchiveReader::ShardBatch& batch) {
                const int bsz = static_cast<int>(batch.size());
                #pragma omp parallel for schedule(dynamic, 1) num_threads(cfg_.threads)
                for (int j = 0; j < bsz; ++j) {
                    const size_t out_idx = batch[j].first;
                    const auto& eg = batch[j].second;
                    auto it = out_to_emb.find(out_idx);
                    if (it == out_to_emb.end()) continue;
                    const size_t i = it->second;
                    try {
                        MinHasher hasher2(cfg2);
                        auto oph2 = hasher2.sketch_oph_with_positions_from_buffer(
                            eg.fasta.data(), eg.fasta.size(), cfg_.sketch_size);
                        const size_t m2 = oph2.signature.size();
                        std::vector<uint16_t> sig2(m2);
                        for (size_t t = 0; t < m2; ++t)
                            sig2[t] = static_cast<uint16_t>(oph2.signature[t]);
                        embeddings_[i].oph_sig2 = std::move(sig2);
                    } catch (const std::exception& e) {
                        spdlog::warn("GEODESIC: sig2 GPK materialization failed for {}: {}",
                                     embeddings_[i].path.filename().string(), e.what());
                    }
                }
            });
        return;
    }

    // Fallback: buf_cache or NFS read.
    if (is_verbose()) spdlog::info("GEODESIC: loading dense k-mer signatures for {} genomes ({} cached, {} re-read)",
                                   need.size(), cache_hits, need.size() - cache_hits);

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
                                 const std::unordered_map<std::string, double>& quality_scores) {
    if (is_verbose()) spdlog::info("GEODESIC: embedding {} genomes (dim={}, k={}, threads={})",
                                   genomes.size(), cfg_.embedding_dim, cfg_.kmer_size, cfg_.threads);
    auto t_build_start = std::chrono::steady_clock::now();
    auto t_phase = [&t_build_start](const char* name) {
        if (!is_verbose()) return;
        auto now = std::chrono::steady_clock::now();
        double s = std::chrono::duration<double>(now - t_build_start).count();
        spdlog::info("  [profile] {:30s} {:7.1f}s (cumulative)", name, s);
    };

#if GEODESIC_USE_AVX2
    if (is_verbose()) spdlog::info("GEODESIC: AVX2 SIMD enabled");
#endif
#if GEODESIC_USE_OMP
    if (is_verbose()) spdlog::info("GEODESIC: OpenMP parallel enabled ({} threads)", cfg_.threads);
    // NOTE: omp_set_num_threads() removed — process-global, unsafe with concurrent taxa.
    // All OMP regions use explicit num_threads(cfg_.threads) clauses.
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
    #pragma omp parallel for schedule(dynamic, 1) num_threads(cfg_.threads)
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
        if (is_verbose()) spdlog::info("  [profile] OPH throughput: {:.0f} MB/s ({:.1f} Gbp, {:.1f}s)",
                                       gbp * 1e3 / oph_s, gbp, oph_s);
    }
    t_phase("OPH sketch (embed_genome)");

    finalize_embeddings_();
    t_phase("finalize (SoA+Nyström+HNSW)");
}

void GeodesicDerep::finalize_embeddings_() {
    const size_t n = embeddings_.size();

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

    // Nyström spectral embedding (replaces CountSketch placeholder vectors)
    nystrom_applied_ = false;
    if (n > SMALL_N_THRESHOLD) {
        apply_nystrom_embeddings();  // sets nystrom_applied_ = true, updates store_.data
    }

    // Release buffer cache — anchor sig2 materialization is done.
    buf_cache_ = {};
    buf_cache_bytes_.store(0);

    // Build HNSW index — skip for small n (brute-force pairwise is faster)
    if (n > SMALL_N_THRESHOLD) {
        // Auto-scale HNSW params: M=48/ef=400 is excellent quality but O(N*M*ef) build cost.
        // For large N, M=16/ef=100 gives >99% recall (hnswlib paper default) at a fraction of cost.
        // User-configured values are treated as caps; we only reduce, never exceed.
        int eff_m  = cfg_.hnsw_m;
        int eff_ef = cfg_.hnsw_ef_construction;
        if (n > 50000) {
            eff_m  = std::min(eff_m,  16);
            eff_ef = std::min(eff_ef, 100);
        } else if (n > 10000) {
            eff_m  = std::min(eff_m,  32);
            eff_ef = std::min(eff_ef, 200);
        }
        index_->seed_ = cfg_.seed;
        index_->build(embeddings_, eff_m, eff_ef, cfg_.hnsw_ef_search, cfg_.threads);
        if (is_verbose()) spdlog::info("GEODESIC: HNSW index built ({} embeddings, M={}, ef_construction={})",
                     embeddings_.size(), eff_m, eff_ef);
    }

    // sig1 (oph_sig) is kept in memory for Phase 7c OPH certification.
    // Freeing it here and rematerializing via visit_shard_batches costs a full NFS re-read
    // (~57s for 233k E. coli, ~52s for 367k Salmonella) which dominated total runtime (92%).
    // The memory cost is sketch_size × 2 bytes/genome: 233k × 10000 × 2 ≈ 4.7 GB.
    // Freed by GeodesicDerep destructor after Phase 7c completes.
}

void GeodesicDerep::build_index_from_gpk(
    const std::vector<std::string>& accessions,
    const std::vector<std::filesystem::path>& genomes,
    IPackReader& gpk,
    const std::unordered_map<std::string, double>& quality_scores)
{
    const size_t n = accessions.size();
    if (is_verbose()) spdlog::info("GEODESIC: gpk: loading {} genomes by accession", n);

    // Store GPK reader + accessions for sig2 materialization (avoids NFS fallback).
    gpk_reader_ = &gpk;
    gpk_accessions_ = accessions;

    embeddings_.resize(n);
    store_.resize(n, cfg_.embedding_dim);
    buf_cache_.assign(n, {});
    buf_cache_bytes_.store(0);

    std::vector<std::string> path_strs(n);
    for (size_t i = 0; i < n; ++i) path_strs[i] = genomes[i].string();

    size_t gpk_hits = 0, gpk_misses = 0;

    auto t0 = std::chrono::steady_clock::now();

    // Shard-batch fetch: delivers one shard at a time (~200 genomes),
    // embeds the batch in parallel (OMP), releases shard pages, moves to next.
    // Peak memory: O(genomes_per_shard × fasta_size) ≈ 200MB vs 224GB for all-at-once.
    // Parallelism: OMP over genomes within each shard batch.
    std::vector<bool> gpk_found(n, false);

    gpk.visit_shard_batches(accessions,
        [&](genopack::ArchiveReader::ShardBatch& batch) {
            const int bsz = static_cast<int>(batch.size());
            #pragma omp parallel for schedule(dynamic, 1) num_threads(cfg_.threads)
            for (int j = 0; j < bsz; ++j) {
                size_t i = batch[j].first;
                const auto& eg = batch[j].second;
                gpk_found[i] = true;
                try {
                    embeddings_[i] = embed_genome_from_buffer(eg.fasta.data(), eg.fasta.size(),
                                                              i, genomes[i]);
                } catch (const std::exception& e) {
                    spdlog::error("GEODESIC: gpk embed failed for {}: {}", accessions[i], e.what());
                    embeddings_[i].genome_id = i;
                    embeddings_[i].vector.assign(cfg_.embedding_dim, 0.0f);
                    embeddings_[i].quality_score = 0.0f;
                    embeddings_[i].path = genomes[i];
                }
            }
        });

    for (size_t i = 0; i < n; ++i) {
        if (gpk_found[i]) {
            ++gpk_hits;
            auto it = quality_scores.find(path_strs[i]);
            if (it != quality_scores.end())
                embeddings_[i].quality_score = static_cast<float>(it->second);
            else if (embeddings_[i].quality_score == 0.0f)
                embeddings_[i].quality_score = 50.0f;
        } else {
            ++gpk_misses;
            spdlog::warn("GEODESIC: {} not found in pack — skipping (no NFS fallback)",
                         accessions[i]);
            {
                std::lock_guard<std::mutex> lock(failed_reads_mutex_);
                failed_reads_.emplace_back(genomes[i].string(),
                                           "not found in genopack archive");
            }
            embeddings_[i].genome_id = i;
            embeddings_[i].vector.assign(cfg_.embedding_dim, 0.0f);
            embeddings_[i].quality_score = 0.0f;
            embeddings_[i].path = genomes[i];
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    spdlog::info("GEODESIC: gpk timing: {}ms ({} hits, {} not found in pack)", ms, gpk_hits, gpk_misses);

    finalize_embeddings_();
}

void GeodesicDerep::build_index_from_gpk_sketches(
    const std::vector<std::string>& accessions,
    const std::vector<std::filesystem::path>& genomes,
    IPackReader& gpk,
    const std::unordered_map<std::string, double>& quality_scores)
{
    const size_t n = accessions.size();
    if (is_verbose()) spdlog::info("GEODESIC: gpk-sketch: loading {} genomes from SKCH section", n);

    gpk_reader_ = &gpk;
    gpk_accessions_ = accessions;
    gpk_paths_      = genomes;

    embeddings_.resize(n);
    store_.resize(n, cfg_.embedding_dim);
    buf_cache_.assign(n, {});
    buf_cache_bytes_.store(0);

    auto t0 = std::chrono::steady_clock::now();

    // Pre-probe: calibrate cfg_ from GPK's available SKCH section before the loop.
    // This ensures param-aware lookup picks the right section when multiple k values
    // coexist, and enables hierarchical slicing when cfg_.sketch_size < stored size.
    const uint32_t gpk_k  = gpk.sketch_kmer_size();
    const uint32_t gpk_sz = gpk.sketch_sketch_size();
    const bool use_param_aware = (gpk_k > 0);

    if (use_param_aware) {
        if (static_cast<int>(gpk_k) != cfg_.kmer_size) {
            // For multi-k archives, check if the requested k is actually available
            // before falling back. This allows maybe_reselect_k to switch k properly.
            const auto avail_ks = gpk.available_kmer_sizes();
            const uint32_t req_k = static_cast<uint32_t>(cfg_.kmer_size);
            const bool requested_k_available = std::find(avail_ks.begin(), avail_ks.end(), req_k) != avail_ks.end();
            if (!requested_k_available) {
                spdlog::warn("GEODESIC: GPK has k={} sketches, cfg requested k={} — using GPK k={}",
                             gpk_k, cfg_.kmer_size, gpk_k);
                cfg_.kmer_size = static_cast<int>(gpk_k);
            }
        }
        if (gpk_sz > 0 && static_cast<int>(gpk_sz) < cfg_.sketch_size) {
            spdlog::warn("GEODESIC: GPK sketch_size={} < requested {} — capping to GPK size",
                         gpk_sz, cfg_.sketch_size);
            cfg_.sketch_size = static_cast<int>(gpk_sz);
        }

        // k pre-probe: determine optimal k from a small sample before loading all N
        // sketches, so we never need a full rebuild on k-switch (maybe_reselect_k).
        // Only runs when the GPK has multiple k values and N is large enough to matter.
        if (n > 100 && !kmer_size_locked_) {
            const int probed_k = probe_kmer_size_(accessions, gpk);
            if (probed_k > 0 && probed_k != cfg_.kmer_size
                    && gpk.has_kmer_size(static_cast<uint32_t>(probed_k))) {
                cfg_.kmer_size = probed_k;
            }
        }
        spdlog::info("GEODESIC: GPK sketch params: k={} size={}", cfg_.kmer_size, cfg_.sketch_size);
    }

    // Resolve accessions to genome_ids, then fetch sketches from SKCH.
    // Genomes not found in SKCH are excluded (treated as failed reads) rather than
    // inserted with zero vectors — identical zero points corrupt hnswlib's graph.
    //
    // Uses visit_sketch_batches() to group accessions by archive and decompress each
    // archive's SKCH exactly once before releasing — avoids thrashing the LRU when
    // accessions interleave across multiple archive parts (peak memory unchanged: one
    // archive SKCH decompressed at a time, same as max_hot_archives_=1).
    size_t misses = 0;
    std::vector<std::string> path_strs(n);
    for (size_t i = 0; i < n; ++i) path_strs[i] = genomes[i].string();

    // uint8_t (not vector<bool>) so distinct-i concurrent writes from the
    // parallelised sketch_for_ids callback are well-defined.
    std::vector<uint8_t> valid(n, 0);

    // First pass: record which accessions are missing from the archive entirely.
    // visit_sketch_batches silently skips misses; we need to attribute them correctly.
    std::vector<uint8_t> in_archive(n, 0);
    for (size_t i = 0; i < n; ++i) {
        if (gpk.genome_meta_by_accession(accessions[i]))
            in_archive[i] = 1;
    }

    const uint32_t req_k  = use_param_aware ? static_cast<uint32_t>(cfg_.kmer_size)  : 0;
    const uint32_t req_sz = use_param_aware ? static_cast<uint32_t>(cfg_.sketch_size) : 0;

    // Callback runs CONCURRENTLY under OMP from SkchReader::sketch_for_ids.
    // Each i is unique → embeddings_[i]/valid[i] writes are race-free.
    // hits counter is atomic; params observed on first hit published once via flag.
    std::atomic<size_t> hits_atomic{0};
    std::atomic<bool>   params_seen{false};
    uint32_t observed_k = 0, observed_sz = 0;
    std::mutex params_mu;

    gpk.visit_sketch_batches(accessions, req_k, req_sz,
        [&](size_t i, const genopack::SketchResult& sk) {
            if (!params_seen.load(std::memory_order_acquire)) {
                std::lock_guard<std::mutex> lk(params_mu);
                if (!params_seen.load(std::memory_order_relaxed)) {
                    observed_k  = sk.kmer_size;
                    observed_sz = sk.sketch_size;
                    params_seen.store(true, std::memory_order_release);
                }
            }
            hits_atomic.fetch_add(1, std::memory_order_relaxed);
            valid[i] = 1;
            auto& emb = embeddings_[i];
            emb.genome_id = i;
            emb.vector.assign(cfg_.embedding_dim, 0.0f);
            emb.oph_sig.assign(sk.sig, sk.sig + sk.sketch_size);
            emb.n_real_bins = sk.n_real_bins;
            emb.genome_size = sk.genome_length;
            emb.real_bins_mask.assign(sk.mask, sk.mask + sk.mask_words);
            emb.isolation_score = 0.0f;
            emb.path = genomes[i];
            auto it = quality_scores.find(path_strs[i]);
            emb.quality_score = (it != quality_scores.end())
                ? static_cast<float>(it->second) : 50.0f;
        });

    const size_t hits = hits_atomic.load(std::memory_order_relaxed);
    if (params_seen.load(std::memory_order_acquire)) {
        cfg_.kmer_size   = static_cast<int>(observed_k);
        cfg_.sketch_size = static_cast<int>(observed_sz);
    }

    // Attribute misses: not-in-archive vs found-in-archive-but-no-sketch.
    for (size_t i = 0; i < n; ++i) {
        if (valid[i]) continue;
        ++misses;
        std::lock_guard<std::mutex> lock(failed_reads_mutex_);
        failed_reads_.emplace_back(genomes[i].string(),
            in_archive[i] ? "sketch not found in SKCH section"
                          : "accession not found in genopack archive");
    }

    // Compact: remove miss entries so only valid genomes enter the pipeline.
    if (misses > 0) {
        std::vector<GenomeEmbedding>      compacted;
        std::vector<std::string>           compacted_acc;
        std::vector<std::filesystem::path> compacted_paths;
        compacted.reserve(hits);
        compacted_acc.reserve(hits);
        compacted_paths.reserve(hits);
        for (size_t i = 0; i < n; ++i) {
            if (!valid[i]) continue;
            auto& emb = embeddings_[i];
            emb.genome_id = compacted.size();  // re-index
            compacted.push_back(std::move(emb));
            compacted_acc.push_back(gpk_accessions_[i]);
            compacted_paths.push_back(gpk_paths_[i]);
        }
        embeddings_     = std::move(compacted);
        gpk_accessions_ = std::move(compacted_acc);
        gpk_paths_      = std::move(compacted_paths);

        store_.resize(hits, cfg_.embedding_dim);
        buf_cache_.assign(hits, {});
    }

    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    spdlog::info("GEODESIC: gpk-sketch timing: {}ms ({} hits, {} misses)", ms, hits, misses);

    if (misses > 0)
        spdlog::warn("GEODESIC: {} genomes excluded (not in SKCH) — {} remain", misses, hits);

    finalize_embeddings_();
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
        NNDistStats stats;
        stats.p5  = nn_dists[n * 5  / 100];
        stats.p50 = nn_dists[n * 50 / 100];
        stats.p95 = nn_dists[n * 95 / 100];
        return stats;
    }
    // k_iso: neighbours used for isolation score (mean angular distance to k nearest).
    // Scales with log2(n) in [k_iso_min, 20] for stable density estimates.
    // K_cap: HNSW query budget. k_conn_min and k_stable are determined from data below.
    size_t n_emb = embeddings_.size();
    const int k_iso_min = cfg_.isolation_k;
    const int k_iso = (n_emb >= 4)
        ? std::max(k_iso_min, std::min(20, static_cast<int>(std::log2(static_cast<double>(n_emb)))))
        : k_iso_min;
    // Scale initial K_cap with N: large clonal clusters (Salmonella, E. coli)
    // will never connect at K_cap=64 — start high to avoid 2-3 wasted retry passes.
    const int k_cap_init = (n_emb > 50000) ? 256 : (n_emb > 5000) ? 128 : 64;
    int K_cap = std::min(k_cap_init, static_cast<int>(n_emb) - 1);

    if (is_verbose()) spdlog::info("GEODESIC: computing isolation scores (k_iso={}, K_cap={})",
                                   k_iso, K_cap);

    // Single HNSW pass at K_cap: isolation scores + neighbor table for multi-prefix MST.
    const int saved_ef = index_->ef_search;
    index_->ef_search = std::max(K_cap * 2, std::min(200, static_cast<int>(n_emb / 100)));
    std::vector<float> nn_dists(n_emb, 1.0f);

    // Neighbor table: nb_ids[ei*K_cap + j] = row-index of j-th neighbor of genome ei.
    // Sentinel UINT32_MAX = slot unfilled.
    std::vector<uint32_t> nb_ids(n_emb * K_cap, UINT32_MAX);
    std::vector<float>    nb_dists(n_emb * K_cap, 1.0f);

    // Lambda: full HNSW query at current K_cap, populates nb_ids/nb_dists/nn_dists/isolation scores.
    auto run_hnsw_query = [&](int query_k_cap) {
#if GEODESIC_USE_OMP
        #pragma omp parallel for schedule(dynamic, 100) num_threads(cfg_.threads)
#endif
        for (size_t ei = 0; ei < n_emb; ++ei) {
            auto& emb = embeddings_[ei];
            auto neighbors = index_->search(emb.vector, query_k_cap + 1);

            float total_dist = 0.0f;
            int iso_count = 0;
            int edge_count = 0;
            for (const auto& [id, dist] : neighbors) {
                if (id != emb.genome_id) {
                    if (iso_count == 0) nn_dists[ei] = dist;
                    if (iso_count < k_iso) total_dist += dist;
                    ++iso_count;
                    if (edge_count < query_k_cap) {
                        auto it = gid_to_row_.find(id);
                        if (it != gid_to_row_.end()) {
                            nb_ids  [ei * query_k_cap + edge_count] = static_cast<uint32_t>(it->second);
                            nb_dists[ei * query_k_cap + edge_count] = dist;
                            ++edge_count;
                        }
                    }
                }
            }

            emb.isolation_score = (iso_count > 0) ? total_dist / std::min(iso_count, k_iso) : 1.0f;
        }
    };

    run_hnsw_query(K_cap);
    index_->ef_search = saved_ef;

    // Kruskal's MST: minimum θ where k-NN graph becomes connected.
    // Phase A: DSU connectivity scan finds k_conn_min (minimum k where graph connects).
    // Phase B: probe ladder finds k_stable (smallest k within 3% of B(K_cap)).
    // Outlier genomes (isolation_score > mean+2σ) are excluded from the MST core.
    double mst_max_edge = 0.0;
    double mst_true_max = 0.0;
    double mst_w2 = 0.0;
    uint32_t bridge_min_side = 0;
    std::vector<double> mst_weights;
    size_t mst_n_main = 0;
    bool disconnected = false;
    int k_conn_min = -1;
    int k_stable_val = -1;
    {
        // Mean + 2σ outlier exclusion on isolation scores.
        double iso_mean = 0.0, iso_m2 = 0.0;
        size_t iso_n = 0;
        for (size_t i = 0; i < n_emb; ++i) {
            double d = embeddings_[i].isolation_score - iso_mean;
            iso_mean += d / ++iso_n;
            iso_m2   += d * (embeddings_[i].isolation_score - iso_mean);
        }
        const double iso_std = (iso_n > 1) ? std::sqrt(iso_m2 / iso_n) : 0.0;
        const float  iso_thr = static_cast<float>(iso_mean + 2.0 * iso_std);

        size_t n_main = 0;
        std::vector<uint32_t> main_remap(n_emb, UINT32_MAX);
        for (size_t i = 0; i < n_emb; ++i) {
            if (embeddings_[i].isolation_score < iso_thr)
                main_remap[i] = static_cast<uint32_t>(n_main++);
        }
        mst_n_main = n_main;

        if (n_main >= 2) {
            // === Phase A: DSU connectivity scan ===
            // Add k-th neighbor edges column by column; stop at first connected k.
            // Activation rank r(u,v) = min(rank_u(v), rank_v(u)): adding column k processes
            // all edges whose activation rank is exactly k, so connectivity at each step
            // matches the symmetrized k-NN graph — same object Kruskal sees in Phase B.
            {
                std::vector<uint32_t> par(n_main);
                std::vector<uint8_t>  rnk(n_main, 0);
                std::iota(par.begin(), par.end(), 0);
                auto find_a = [&](uint32_t x) {
                    while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; }
                    return x;
                };
                auto unite_a = [&](uint32_t a, uint32_t b) -> bool {
                    a = find_a(a); b = find_a(b);
                    if (a == b) return false;
                    if (rnk[a] < rnk[b]) std::swap(a, b);
                    par[b] = a;
                    if (rnk[a] == rnk[b]) ++rnk[a];
                    return true;
                };
                int comps = static_cast<int>(n_main);
                for (int k = 0; k < K_cap && comps > 1; ++k) {
                    for (size_t ei = 0; ei < n_emb; ++ei) {
                        if (main_remap[ei] == UINT32_MAX) continue;
                        uint32_t nid = nb_ids[ei * K_cap + k];
                        if (nid == UINT32_MAX || nid >= n_emb) continue;
                        if (main_remap[nid] == UINT32_MAX) continue;
                        if (unite_a(main_remap[ei], main_remap[nid])) --comps;
                    }
                    if (comps == 1) { k_conn_min = k + 1; break; }
                }
            }

            // === Adaptive K_cap retry ===
            // If DSU scan never connected, retry with larger K_cap values.
            // ANN results are NOT prefix-stable: full requery required at each K_cap.
            if (k_conn_min < 0) {
                const int retry_caps[] = {128, 256, cfg_.k_cap_max};
                for (int rc : retry_caps) {
                    int new_K_cap = std::min(rc, static_cast<int>(n_emb) - 1);
                    if (new_K_cap <= K_cap) continue;

                    int new_ef = std::max(new_K_cap * 2, std::min(200, static_cast<int>(n_emb / 100)));
                    spdlog::info("GEODESIC: k-NN disconnected at K_cap={}, retrying with K_cap={} ef_search={}",
                                 K_cap, new_K_cap, new_ef);

                    // Full requery: reallocate and overwrite neighbor tables completely
                    K_cap = new_K_cap;
                    index_->ef_search = new_ef;
                    nb_ids.assign(n_emb * K_cap, UINT32_MAX);
                    nb_dists.assign(n_emb * K_cap, 1.0f);
                    std::fill(nn_dists.begin(), nn_dists.end(), 1.0f);
                    run_hnsw_query(K_cap);
                    index_->ef_search = saved_ef;

                    // Re-run DSU scan from scratch
                    std::vector<uint32_t> par(n_main);
                    std::vector<uint8_t>  rnk(n_main, 0);
                    std::iota(par.begin(), par.end(), 0);
                    auto find_r = [&](uint32_t x) {
                        while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; }
                        return x;
                    };
                    auto unite_r = [&](uint32_t a, uint32_t b) -> bool {
                        a = find_r(a); b = find_r(b);
                        if (a == b) return false;
                        if (rnk[a] < rnk[b]) std::swap(a, b);
                        par[b] = a;
                        if (rnk[a] == rnk[b]) ++rnk[a];
                        return true;
                    };
                    int comps = static_cast<int>(n_main);
                    for (int k = 0; k < K_cap && comps > 1; ++k) {
                        for (size_t ei = 0; ei < n_emb; ++ei) {
                            if (main_remap[ei] == UINT32_MAX) continue;
                            uint32_t nid = nb_ids[ei * K_cap + k];
                            if (nid == UINT32_MAX || nid >= n_emb) continue;
                            if (main_remap[nid] == UINT32_MAX) continue;
                            if (unite_r(main_remap[ei], main_remap[nid])) --comps;
                        }
                        if (comps == 1) { k_conn_min = k + 1; break; }
                    }
                    if (k_conn_min > 0) break;
                }
                if (k_conn_min < 0) {
                    spdlog::warn("GEODESIC: k-NN graph still disconnected after retry at K_cap={}", K_cap);
                    // Bridge diagnostic: sample up to 50 cross-component pairs to distinguish
                    // ANN recall gap (true similarity > J_bridge) vs structural disconnection.
                    if (!embeddings_.empty() && !embeddings_[0].oph_sig.empty()) {
                        // Rebuild component labels from final nb_ids state.
                        std::vector<uint32_t> bd_par(n_main);
                        std::iota(bd_par.begin(), bd_par.end(), 0);
                        auto bd_find = [&](uint32_t x) -> uint32_t {
                            while (bd_par[x] != x) { bd_par[x] = bd_par[bd_par[x]]; x = bd_par[x]; }
                            return x;
                        };
                        for (int k = 0; k < K_cap; ++k) {
                            for (size_t ei = 0; ei < n_emb; ++ei) {
                                if (main_remap[ei] == UINT32_MAX) continue;
                                uint32_t nid = nb_ids[ei * K_cap + k];
                                if (nid == UINT32_MAX || nid >= n_emb) continue;
                                if (main_remap[nid] == UINT32_MAX) continue;
                                uint32_t ra = bd_find(main_remap[ei]);
                                uint32_t rb = bd_find(main_remap[nid]);
                                if (ra != rb) bd_par[rb] = ra;
                            }
                        }
                        // Collect component members (ei index → component root).
                        std::unordered_map<uint32_t, std::vector<uint32_t>> comp_members;
                        for (uint32_t ei = 0; ei < static_cast<uint32_t>(n_emb); ++ei) {
                            if (main_remap[ei] == UINT32_MAX) continue;
                            comp_members[bd_find(main_remap[ei])].push_back(ei);
                        }
                        const size_t m_oph = embeddings_[0].oph_sig.size();
                        // Two-tier thresholds: J_cert (ANI threshold) = definite recall gap;
                        // J_80 (80% ANI) = possible gap / weaker bridge.
                        // Sampling uses va[0]/vb[0]: one genome per component pair — sufficient
                        // for detection but a negative result means "NO_BRIDGE_FOUND", not "STRUCTURAL".
                        const double q_ani  = std::pow(cfg_.ani_threshold,
                                                       static_cast<double>(cfg_.kmer_size));
                        const double J_cert_bd = q_ani / (2.0 - q_ani);
                        const double q80    = std::pow(0.80, static_cast<double>(cfg_.kmer_size));
                        const double J_80   = q80 / (2.0 - q80);
                        double max_cross_jac = 0.0;
                        size_t n_above_cert = 0, n_above_80 = 0, n_pairs_sampled = 0;
                        constexpr size_t max_pairs = 50;
                        std::vector<uint32_t> comp_keys;
                        comp_keys.reserve(comp_members.size());
                        for (auto& [k, _] : comp_members) comp_keys.push_back(k);
                        for (size_t ci = 0; ci < comp_keys.size() && n_pairs_sampled < max_pairs; ++ci) {
                            for (size_t cj = ci + 1; cj < comp_keys.size() && n_pairs_sampled < max_pairs; ++cj) {
                                const auto& va = comp_members[comp_keys[ci]];
                                const auto& vb = comp_members[comp_keys[cj]];
                                if (va.empty() || vb.empty()) continue;
                                const uint32_t a = va[0], b = vb[0];
                                const auto& sig_a = embeddings_[a].oph_sig;
                                const auto& sig_b = embeddings_[b].oph_sig;
                                if (sig_a.size() != m_oph || sig_b.size() != m_oph) continue;
                                const double jac = refine_jaccard_ptr(sig_a.data(), sig_b.data(), m_oph);
                                max_cross_jac = std::max(max_cross_jac, jac);
                                if (jac > J_cert_bd) ++n_above_cert;
                                if (jac > J_80)      ++n_above_80;
                                ++n_pairs_sampled;
                            }
                        }
                        const char* diagnosis = (n_above_cert > 0)
                            ? "ANN_RECALL_GAP (pairs above ANI-threshold Jaccard — HNSW missed edges)"
                            : (n_above_80 > 0)
                                ? "POSSIBLE_GAP (pairs 80-95% ANI — weak bridges missed by HNSW)"
                                : "NO_BRIDGE_FOUND (sampled pairs below 80% ANI — likely structural)";
                        spdlog::warn("GEODESIC: bridge diagnostic: {} pairs sampled, max_jac={:.4f} "
                                     "(J_cert={:.4f}, J_80={:.4f}), {} above cert, {} above 80% → {}",
                                     n_pairs_sampled, max_cross_jac, J_cert_bd, J_80,
                                     n_above_cert, n_above_80, diagnosis);
                    }
                }
            }

            // === Phase B: Bottleneck stability probe ===
            // Kruskal at up to k_lim neighbors per genome. Edge (u,v) included if it
            // appears in either u's or v's first k_lim neighbors — equivalent to
            // activation rank <= k_lim. Duplicate directed edges are harmless (second
            // unite is a no-op; min-dist wins in Kruskal sort).
            struct KNNEdge { uint32_t u, v; float dist; };
            struct MstResult {
                double   max_edge      = 0.0;
                double   trimmed_max   = 0.0;  // max edge excluding outlier bridges (bridge_min <= sqrt(n))
                double   p90_edge      = 0.0;  // retained for logging only
                double   w2            = 0.0;
                uint32_t bridge_min    = 0;
                std::vector<double>   weights;
                std::vector<uint32_t> labels;
                size_t   components = 0;
            };
            auto run_kruskal = [&](int k_lim) -> MstResult {
                std::vector<KNNEdge> edges;
                edges.reserve(n_emb * k_lim);
                for (size_t ei = 0; ei < n_emb; ++ei) {
                    if (main_remap[ei] == UINT32_MAX) continue;
                    for (int j = 0; j < k_lim; ++j) {
                        uint32_t nid = nb_ids  [ei * K_cap + j];
                        float    nd  = nb_dists[ei * K_cap + j];
                        if (nid == UINT32_MAX || nd >= 1.0f) break;
                        if (nid >= n_emb || main_remap[nid] == UINT32_MAX) continue;
                        uint32_t ru = main_remap[ei], rv = main_remap[nid];
                        if      (ru < rv) edges.push_back({ru, rv, nd});
                        else if (ru > rv) edges.push_back({rv, ru, nd});
                    }
                }
                std::sort(edges.begin(), edges.end(),
                          [](const KNNEdge& a, const KNNEdge& b) { return a.dist < b.dist; });

                std::vector<uint32_t> parent(n_main), comp_sz(n_main, 1);
                std::vector<uint8_t>  rk(n_main, 0);
                std::iota(parent.begin(), parent.end(), 0);
                auto find_uf = [&](uint32_t x) -> uint32_t {
                    while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
                    return x;
                };
                auto unite_uf = [&](uint32_t a, uint32_t b, uint32_t& oa, uint32_t& ob) -> bool {
                    a = find_uf(a); b = find_uf(b);
                    if (a == b) { oa = ob = 0; return false; }
                    oa = comp_sz[a]; ob = comp_sz[b];
                    if (rk[a] < rk[b]) std::swap(a, b);
                    parent[b] = a; comp_sz[a] += comp_sz[b];
                    if (rk[a] == rk[b]) ++rk[a];
                    return true;
                };

                MstResult res;
                res.components = n_main;
                res.weights.reserve(n_main > 1 ? n_main - 1 : 0);
                // Outlier-bridge threshold: exclude edges whose smaller component side
                // is ≤ ceil(sqrt(n_main)) — these are singleton/pair outliers connecting
                // to the main mass and should not set the diversity threshold.
                const uint32_t bridge_thresh = static_cast<uint32_t>(
                    std::ceil(std::sqrt(static_cast<double>(n_main))));
                for (const auto& e : edges) {
                    uint32_t sa = 0, sb = 0;
                    if (unite_uf(e.u, e.v, sa, sb)) {
                        res.w2         = res.max_edge;
                        res.max_edge   = e.dist;
                        res.bridge_min = std::min(sa, sb);
                        res.weights.push_back(e.dist);
                        if (res.bridge_min > bridge_thresh)
                            res.trimmed_max = e.dist;  // last qualifying edge = max qualifying
                        if (--res.components == 1) break;
                    }
                }
                // Fall back to true max when all edges have tiny bridge sides (small taxa).
                if (res.trimmed_max <= 0.0) res.trimmed_max = res.max_edge;
                // Retain P90 for logging/comparison only.
                if (!res.weights.empty()) {
                    size_t p90_idx = static_cast<size_t>(res.weights.size() * 0.90);
                    p90_idx = std::min(p90_idx, res.weights.size() - 1);
                    res.p90_edge = res.weights[p90_idx];
                }
                res.labels.resize(n_main);
                std::unordered_map<uint32_t, uint32_t> root_label;
                uint32_t next_lbl = 0;
                for (uint32_t j = 0; j < static_cast<uint32_t>(n_main); ++j) {
                    uint32_t root = find_uf(j);
                    auto [it, ins] = root_label.emplace(root, next_lbl);
                    if (ins) ++next_lbl;
                    res.labels[j] = it->second;
                }
                return res;
            };

            // Fixed probe ladder; always includes k_conn_min (if connected) and K_cap.
            static constexpr int kLadder[] = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64};
            constexpr float kTau = 0.03f;  // 3% relative tolerance
            std::vector<int> probes;
            if (k_conn_min > 0) {
                probes.push_back(k_conn_min);
                for (int x : kLadder)
                    if (x > k_conn_min && x <= K_cap) probes.push_back(x);
            }
            if (probes.empty() || probes.back() != K_cap) probes.push_back(K_cap);
            probes.erase(std::unique(probes.begin(), probes.end()), probes.end());

            // Reference bottleneck B(K_cap); probe from smallest k upward.
            const auto res_cap = run_kruskal(K_cap);
            const float B_ref  = static_cast<float>(res_cap.max_edge);

            k_stable_val = K_cap;
            MstResult chosen_storage;
            bool used_cap = true;
            for (int k : probes) {
                if (k == K_cap) break;
                auto res = run_kruskal(k);
                float Bk  = static_cast<float>(res.max_edge);
                float gap = (B_ref > 1e-9f) ? std::max(0.0f, Bk - B_ref) / B_ref : 0.0f;
                if (gap <= kTau) {
                    k_stable_val   = k;
                    chosen_storage = std::move(res);
                    used_cap       = false;
                    break;
                }
            }
            const MstResult& chosen = used_cap ? res_cap : chosen_storage;

            mst_true_max    = chosen.max_edge;
            mst_max_edge    = chosen.trimmed_max;  // bridge-size-conditioned; excludes outlier bridges
            mst_w2          = chosen.w2;
            bridge_min_side = chosen.bridge_min;
            mst_weights     = chosen.weights;
            disconnected    = (chosen.components > 1);

            component_ids_.assign(n_emb, -1);
            for (size_t i = 0; i < n_emb; ++i) {
                if (main_remap[i] != UINT32_MAX)
                    component_ids_[i] = static_cast<int>(chosen.labels[main_remap[i]]);
            }

            if (is_verbose()) {
                double tail_r = (mst_max_edge > 1e-9) ? mst_true_max / mst_max_edge : 0.0;
                spdlog::info("GEODESIC: MST over {} non-outlier genomes "
                             "(k_conn={} k_stable={} K_cap={} components={} "
                             "trimmed={:.5f} P90={:.5f} max={:.5f} B_ref={:.5f} tail_ratio={:.2f})",
                             n_main, k_conn_min, k_stable_val, K_cap,
                             chosen.components, mst_max_edge, chosen.p90_edge, mst_true_max, B_ref, tail_r);
            } else if (disconnected) {
                spdlog::info("GEODESIC: k-NN graph disconnected at K_cap={} (components={}); "
                             "per-component thresholds will be used", K_cap, chosen.components);
            }
        }
    }

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
        if (!component_ids_.empty()) {
            std::vector<int> sorted_cids(n_emb);
            for (size_t i = 0; i < n_emb; ++i) sorted_cids[i] = component_ids_[perm[i]];
            component_ids_ = std::move(sorted_cids);
        }
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
    NNDistStats stats;
    stats.p5  = nn_dists[n * 5  / 100];
    stats.p50 = nn_dists[n * 50 / 100];
    stats.p95 = nn_dists[n * 95 / 100];
    stats.mst_max_edge    = mst_max_edge;
    stats.mst_true_max    = mst_true_max;
    stats.tail_ratio      = (mst_max_edge > 1e-9) ? mst_true_max / mst_max_edge : 0.0;
    stats.mst_w2          = mst_w2;
    stats.bridge_min_side = bridge_min_side;
    stats.k_conn          = k_conn_min;
    stats.k_stable        = k_stable_val;
    stats.k_cap           = K_cap;
    stats.low_pair_count   = (mst_n_main < 20);
    stats.disconnected_mst = disconnected;
    // Pathological bridge detection: an anomalous single accession bridging two taxa
    // has a tiny smaller-side component AND the terminal MST merge is isolated
    // (no other MST edges exist near that scale). Genuine multi-scale population
    // structure (e.g. S. enterica serovars) has substantial components on both sides
    // and many inter-group MST edges near the same scale → stays silent.
    if (mst_true_max > 0.0 && mst_n_main >= 2) {
        const int tail_count = static_cast<int>(std::count_if(
            mst_weights.begin(), mst_weights.end(),
            [w1 = mst_true_max](double w) { return w >= 0.8 * w1; }));
        const bool is_tiny_side = (bridge_min_side <= 10)
                               && (static_cast<double>(bridge_min_side) / mst_n_main <= 0.005);
        const bool is_isolated  = (tail_count <= 2)
                               || (mst_w2 > 1e-9 && mst_true_max / mst_w2 >= 1.8);
        stats.pathological_bridge = is_tiny_side && is_isolated;
    }
    // Log multiscale separation as a diagnostic (not a warning — high ratio is expected
    // for species with genuine serovar/pathotype structure).
    if (mst_true_max > 0.0 && stats.p95 > 1e-9) {
        if (is_verbose() || mst_true_max / stats.p95 > 5.0)
            spdlog::info("GEODESIC: multiscale diagnostic: mst_max/P95={:.1f} "
                         "(mst_max={:.4f} mst_P90={:.4f} P95={:.4f} w2={:.4f} bridge_min_side={})",
                         mst_true_max / stats.p95, mst_true_max, mst_max_edge, stats.p95,
                         mst_w2, bridge_min_side);
    }
    return stats;
}

void GeodesicDerep::apply_nystrom_embeddings() {
    const size_t n = embeddings_.size();
    if (n == 0) return;

    auto t_nys_start = std::chrono::steady_clock::now();
    auto t_nys = [&t_nys_start](const char* name) {
        if (!is_verbose()) return;
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
        std::mt19937_64 rng(cfg_.seed);
        const bool do_stratify = (n_anchors >= 5 && n >= 5 && cfg_.sketch_size > 0);
        if (do_stratify) {
            // Dense-only anchor pool: only genomes with f_i >= 0.85 and n_real_bins >= 50
            // are eligible as anchors. This keeps the anchor Gram matrix K and its
            // eigensystem W free from OPH deflation artefacts introduced by sparse MAGs.
            // Sparse queries still project correctly via the per-anchor one-sided
            // fill-fraction correction in Step 4. If the dense pool is exhausted,
            // we pad from the next-densest available genomes (sorted by descending f_i)
            // to guarantee n_anchors anchors are always available.
            const float dense_threshold =
                static_cast<float>(cfg_.sketch_size) * 0.85f;

            // Partition into dense (f >= 0.85) and fallback (f < 0.85, sorted descending).
            std::vector<size_t> dense_pool, fallback_pool;
            std::vector<std::pair<float, size_t>> fallback_sorted;
            for (size_t i = 0; i < n; ++i) {
                const uint32_t nr = embeddings_[i].n_real_bins;
                if (nr < 50) continue;
                if (static_cast<float>(nr) >= dense_threshold)
                    dense_pool.push_back(i);
                else
                    fallback_sorted.push_back({static_cast<float>(nr), i});
            }
            std::shuffle(dense_pool.begin(), dense_pool.end(), rng);
            // Sort fallback descending by fill fraction (densest sparse first).
            std::sort(fallback_sorted.begin(), fallback_sorted.end(),
                      [](const auto& a, const auto& b){ return a.first > b.first; });

            // Fill anchor_idx: dense first, then fallback if needed.
            for (size_t t = 0; t < dense_pool.size() && anchor_idx.size() < n_anchors; ++t)
                anchor_idx.push_back(dense_pool[t]);
            if (anchor_idx.size() < n_anchors) {
                spdlog::warn("GEODESIC/Nyström: dense pool ({} genomes) smaller than "
                             "n_anchors ({}); padding with {}/{} sparser genomes — "
                             "consider a larger collection or smaller --geodesic-dim",
                             dense_pool.size(), n_anchors,
                             std::min(n_anchors - anchor_idx.size(), fallback_sorted.size()),
                             fallback_sorted.size());
                for (size_t t = 0; t < fallback_sorted.size() && anchor_idx.size() < n_anchors; ++t)
                    anchor_idx.push_back(fallback_sorted[t].second);
            }

            // Last resort: pad from any remaining genome (e.g., n_real_bins < 50).
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

    // Note: containment blend for sparse MAGs was removed from the Nyström kernel.
    // The blend (1-a)*J + a*max(C(i→j), C(j→i)) is not guaranteed PSD, breaking
    // spectral theory. Containment corrections belong in certification (Phase 7/8),
    // not in the anchor kernel. Pure Jaccard is used here for all genome pairs.

#if GEODESIC_USE_OMP
    #pragma omp parallel for schedule(dynamic, 4) num_threads(cfg_.threads)
#endif
    for (int i = 0; i < static_cast<int>(n_anchors); ++i) {
        K_rm(i, i) = 1.0f;
        for (int j = i + 1; j < static_cast<int>(n_anchors); ++j) {
            size_t ai = anchor_idx[static_cast<size_t>(i)];
            size_t aj = anchor_idx[static_cast<size_t>(j)];
            const auto& sa = embeddings_[ai].oph_sig;
            const auto& sb = embeddings_[aj].oph_sig;
            float jac = static_cast<float>(
                refine_jaccard_ptr(sa.data(), sb.data(), sa.size()));
            // Improvement 5: average two sketches for variance reduction
            if (!embeddings_[ai].oph_sig2.empty() && !embeddings_[aj].oph_sig2.empty()) {
                const auto& s2a = embeddings_[ai].oph_sig2;
                const auto& s2b = embeddings_[aj].oph_sig2;
                float jac2 = static_cast<float>(
                    refine_jaccard_ptr(s2a.data(), s2b.data(), s2a.size()));
                jac = (jac + jac2) * 0.5f;
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

    // Numerical regularization only — does not make full kernel PSD.
    // In-place eigenvalue shift: if anchor Gram has negative eigenvalues beyond
    // numerical noise, shift the entire spectrum up so lambda_min >= 1e-6.
    // Eigenvectors are invariant under K → K + δI, so no re-decomposition needed.
    Eigen::VectorXf eigenvalues_shifted = solver.eigenvalues();
    const float lambda_min = eigenvalues_shifted(0);  // ascending order in Eigen
    if (lambda_min < -1e-8f) {
        const float delta = std::abs(lambda_min) + 1e-6f;
        spdlog::info("GEODESIC/Nystrom: indefinite anchor matrix (lambda_min={:.6e}), ridge delta={:.6e}",
                     lambda_min, delta);
        eigenvalues_shifted.array() += delta;
    }

    // Auto-select d: fewest top eigenvectors that capture >= nystrom_min_variance.
    const float total_eig = eigenvalues_shifted.cwiseMax(0.0f).sum();
    const int max_d = std::min(cfg_.embedding_dim, static_cast<int>(n_anchors));
    int actual_d = 1;
    if (total_eig > 0.0f) {
        float cumsum = 0.0f;
        for (int i = static_cast<int>(n_anchors) - 1;
             i >= 0 && actual_d <= max_d; --i) {
            cumsum += std::max(0.0f, eigenvalues_shifted(i));
            if (cumsum / total_eig >= cfg_.nystrom_min_variance) break;
            if (actual_d < max_d) ++actual_d;
        }
    } else {
        actual_d = max_d;
    }

    // Round up to multiple of 16 for hnswlib SIMD alignment.
    // hnswlib's InnerProductSpace dispatches to SIMD16 when dim%16==0,
    // avoiding the Residuals path whose prefetch reads one past the neighbor list.
    actual_d = ((actual_d + 15) / 16) * 16;
    actual_d = std::min(actual_d, max_d);

    Eigen::MatrixXf U = solver.eigenvectors().rightCols(actual_d); // [n_anchors × d]
    Eigen::VectorXf lam = eigenvalues_shifted.tail(actual_d);      // [d]
    lam = lam.cwiseMax(1e-10f);  // safety floor for inv-sqrt

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
    // causing cache/TLB misses when scanning them per genome. The slab is ~4 MB
    // (512 × 4096 × 2 bytes) — fits in L2/L3 and stays warm across the parallel loop.
    const size_t m_sig = cfg_.sketch_size;
    std::vector<uint16_t> anchor_slab;
    if (m_sig > 0 && !anchor_idx.empty() && !embeddings_[anchor_idx[0]].oph_sig.empty()) {
        anchor_slab.resize(n_anchors * m_sig);
        for (size_t a = 0; a < n_anchors; ++a) {
            const size_t ai = anchor_idx[a];
            const auto& sig = embeddings_[ai].oph_sig;
            const size_t copy_m = std::min(sig.size(), m_sig);
            std::memcpy(anchor_slab.data() + a * m_sig, sig.data(), copy_m * sizeof(uint16_t));
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

    // Batch size for GEMM: k_mat[BATCH×L] × W[L×D] = e_mat[BATCH×D].
    // BLAS-3 reuses W (512×256 = 2 MB) across the batch; individual GEMV re-reads it per genome.
    // At BATCH=64: k_mat(64KB) + e_mat(64KB) stay in L2, W tiles through L3 once per batch.
    constexpr int kNysBatch = 64;
    const int n_batches = static_cast<int>((n + kNysBatch - 1) / kNysBatch);
    const Eigen::RowVectorXf d_anc_row = d_anchor_inv_sqrt.transpose();

    using RowMajMat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

#if GEODESIC_USE_OMP
    // Eigen's internal parallelism must be disabled inside the OMP parallel region:
    // each of cfg_.threads workers would otherwise spawn its own Eigen thread pool,
    // causing cfg_.threads² threads competing for the same cores.
    Eigen::setNbThreads(1);
    #pragma omp parallel num_threads(cfg_.threads)
    {
        // Thread-local batch matrices: allocated once per thread, reused per batch.
        RowMajMat k_mat(kNysBatch, static_cast<int>(n_anchors));
        RowMajMat e_mat(kNysBatch, actual_d);

        #pragma omp for schedule(static)
        for (int b = 0; b < n_batches; ++b) {
            const size_t i0 = static_cast<size_t>(b) * kNysBatch;
            const size_t i1 = std::min(i0 + static_cast<size_t>(kNysBatch), n);
            const int bs = static_cast<int>(i1 - i0);

            for (int bi = 0; bi < bs; ++bi) {
                const size_t i = i0 + static_cast<size_t>(bi);
                const int32_t ap = anchor_pos[i];
                if (ap >= 0) {
                    // Anchor: row already degree-normalized in K (Step 2+3).
                    k_mat.row(bi) = K.row(ap);
                } else {
                    if (embeddings_[i].oph_sig.empty()) {
                        k_mat.row(bi).setZero();
                        continue;
                    }
                    const uint16_t* sig_i_ptr = embeddings_[i].oph_sig.data();
                    for (size_t a = 0; a < n_anchors; ++a) {
                        float jac;
                        if (!anchor_slab.empty()) {
                            jac = static_cast<float>(
                                refine_jaccard_ptr(sig_i_ptr,
                                                  anchor_slab.data() + a * m_sig, m_sig));
                        } else {
                            const auto& sig_a = embeddings_[anchor_idx[a]].oph_sig;
                            jac = static_cast<float>(
                                refine_jaccard_ptr(sig_i_ptr, sig_a.data(), m_sig));
                        }
                        k_mat(bi, static_cast<int>(a)) = jac;
                    }
                    // Fill-fraction correction for sparse genomes (MAGs with low
                    // completeness). OPH Jaccard between a sparse query (f_i = n_real_i/m)
                    // and an anchor of fill f_a is deflated by ~f_i relative to a dense
                    // anchor; after Laplacian degree normalisation the residual bias is
                    // ~sqrt(f_i). Per-anchor scaling by sqrt(n_real_a / n_real_i) corrects
                    // each element independently, avoiding over-correction when sparse MAGs
                    // appear in the anchor set (FPS can select them). Capped at 4× to bound
                    // noise amplification. Skipped when f_i > 0.85 (negligible correction)
                    // or n_real_i < 50 (too noisy; earlier quality filters should catch these).
                    {
                        const uint32_t n_real_i = embeddings_[i].n_real_bins;
                        if (m_sig > 0 && n_real_i >= 50 &&
                            n_real_i < static_cast<uint32_t>(m_sig) * 85 / 100) {
                            for (size_t a = 0; a < n_anchors; ++a) {
                                const uint32_t n_real_a =
                                    embeddings_[anchor_idx[a]].n_real_bins;
                                // One-sided: uplift sparse-query vs dense-anchor pairs;
                                // never suppress when anchor is equally/more sparse than
                                // the query (no theoretical basis for shrinkage there).
                                const float corr = std::min(
                                    std::sqrt(static_cast<float>(std::max(n_real_a, n_real_i)) /
                                              static_cast<float>(n_real_i)),
                                    4.0f);
                                k_mat(bi, static_cast<int>(a)) *= corr;
                            }
                        }
                    }
                    if (cfg_.nystrom_degree_normalize) {
                        float d_i = k_mat.row(bi).sum();
                        if (d_i < 1e-10f) d_i = 1.0f;
                        k_mat.row(bi) =
                            k_mat.row(bi).cwiseProduct(d_anc_row) / std::sqrt(d_i);
                    }
                }
            }

            // Batched GEMM: e_mat[bs×D] = k_mat[bs×L] × W[L×D]
            e_mat.topRows(bs).noalias() = k_mat.topRows(bs) * W;

            for (int bi = 0; bi < bs; ++bi) {
                const size_t i = i0 + static_cast<size_t>(bi);
                float norm = e_mat.row(bi).norm();
                if (norm > 1e-10f) e_mat.row(bi) /= norm;
                std::copy(e_mat.row(bi).data(), e_mat.row(bi).data() + actual_d,
                          embeddings_[i].vector.data());
                std::copy(e_mat.row(bi).data(), e_mat.row(bi).data() + actual_d,
                          store_.row(i));
            }
        }
    }
#else
    {
        RowMajMat k_mat(kNysBatch, static_cast<int>(n_anchors));
        RowMajMat e_mat(kNysBatch, actual_d);
        for (int b = 0; b < n_batches; ++b) {
            const size_t i0 = static_cast<size_t>(b) * kNysBatch;
            const size_t i1 = std::min(i0 + static_cast<size_t>(kNysBatch), n);
            const int bs = static_cast<int>(i1 - i0);
            for (int bi = 0; bi < bs; ++bi) {
                const size_t i = i0 + static_cast<size_t>(bi);
                const int32_t ap = anchor_pos[i];
                if (ap >= 0) {
                    k_mat.row(bi) = K.row(ap);
                } else {
                    if (embeddings_[i].oph_sig.empty()) { k_mat.row(bi).setZero(); continue; }
                    const uint16_t* sig_i_ptr = embeddings_[i].oph_sig.data();
                    for (size_t a = 0; a < n_anchors; ++a) {
                        float jac;
                        if (!anchor_slab.empty()) {
                            jac = static_cast<float>(
                                refine_jaccard_ptr(sig_i_ptr,
                                                  anchor_slab.data() + a * m_sig, m_sig));
                        } else {
                            const auto& sig_a = embeddings_[anchor_idx[a]].oph_sig;
                            jac = static_cast<float>(
                                refine_jaccard_ptr(sig_i_ptr, sig_a.data(), m_sig));
                        }
                        k_mat(bi, static_cast<int>(a)) = jac;
                    }
                    // Fill-fraction correction for sparse genomes (MAGs with low
                    // completeness). OPH Jaccard between a sparse query (f_i = n_real_i/m)
                    // and an anchor of fill f_a is deflated by ~f_i relative to a dense
                    // anchor; after Laplacian degree normalisation the residual bias is
                    // ~sqrt(f_i). Per-anchor scaling by sqrt(n_real_a / n_real_i) corrects
                    // each element independently, avoiding over-correction when sparse MAGs
                    // appear in the anchor set (FPS can select them). Capped at 4× to bound
                    // noise amplification. Skipped when f_i > 0.85 (negligible correction)
                    // or n_real_i < 50 (too noisy; earlier quality filters should catch these).
                    {
                        const uint32_t n_real_i = embeddings_[i].n_real_bins;
                        if (m_sig > 0 && n_real_i >= 50 &&
                            n_real_i < static_cast<uint32_t>(m_sig) * 85 / 100) {
                            for (size_t a = 0; a < n_anchors; ++a) {
                                const uint32_t n_real_a =
                                    embeddings_[anchor_idx[a]].n_real_bins;
                                // One-sided: uplift sparse-query vs dense-anchor pairs;
                                // never suppress when anchor is equally/more sparse than
                                // the query (no theoretical basis for shrinkage there).
                                const float corr = std::min(
                                    std::sqrt(static_cast<float>(std::max(n_real_a, n_real_i)) /
                                              static_cast<float>(n_real_i)),
                                    4.0f);
                                k_mat(bi, static_cast<int>(a)) *= corr;
                            }
                        }
                    }
                    if (cfg_.nystrom_degree_normalize) {
                        float d_i = k_mat.row(bi).sum();
                        if (d_i < 1e-10f) d_i = 1.0f;
                        k_mat.row(bi) =
                            k_mat.row(bi).cwiseProduct(d_anc_row) / std::sqrt(d_i);
                    }
                }
            }
            e_mat.topRows(bs).noalias() = k_mat.topRows(bs) * W;
            for (int bi = 0; bi < bs; ++bi) {
                const size_t i = i0 + static_cast<size_t>(bi);
                float norm = e_mat.row(bi).norm();
                if (norm > 1e-10f) e_mat.row(bi) /= norm;
                std::copy(e_mat.row(bi).data(), e_mat.row(bi).data() + actual_d,
                          embeddings_[i].vector.data());
                std::copy(e_mat.row(bi).data(), e_mat.row(bi).data() + actual_d,
                          store_.row(i));
            }
        }
    }
#endif

    Eigen::setNbThreads(0);  // restore to auto (hardware concurrency)

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
    // Sort by isolation × sqrt(size), with quality as tie-breaker only.
    // Quality should NOT multiply fitness — that anti-correlates with isolation when
    // using ad-hoc scores, creating a parabolic fitness that selects mediocre genomes.
    {
        std::vector<size_t> perm(n);
        std::iota(perm.begin(), perm.end(), 0);
        std::sort(perm.begin(), perm.end(), [&](size_t a, size_t b) {
            float la = std::sqrt(static_cast<float>(embeddings_[a].genome_size) / median_sz);
            float lb = std::sqrt(static_cast<float>(embeddings_[b].genome_size) / median_sz);
            // Primary: isolation × sqrt(size) — pure diversity signal
            float fa = embeddings_[a].isolation_score * la;
            float fb = embeddings_[b].isolation_score * lb;
            if (fa != fb) return fa > fb;
            // Tie-breaker: higher quality wins
            if (embeddings_[a].quality_score != embeddings_[b].quality_score)
                return embeddings_[a].quality_score > embeddings_[b].quality_score;
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

void GeodesicDerep::exclude_from_reps(const std::unordered_set<std::string>& paths) {
    for (size_t i = 0; i < store_.n; ++i) {
        if (paths.count(store_.paths[i].string()))
            store_.quality_scores[i] = 0.0f;
    }
}

void GeodesicDerep::compute_adhoc_quality_scores() {
    // Compute ad-hoc quality scores for genomes without CheckM2 data.
    // Quality = sketch completeness (n_real_bins / sketch_size) × 100
    //
    // Design rationale (per GPT-5.4 analysis):
    // - Quality must NOT depend on isolation/centrality — that anti-correlates with
    //   the FPS objective (isolation × quality), creating a parabolic fitness that
    //   selects mediocre mid-isolation genomes instead of the most diverse.
    // - Sketch completeness is orthogonal to embedding geometry, measuring how much
    //   of the OPH sketch is filled (proxy for assembly completeness/quality).
    // - Quality is used as a tie-breaker in FPS, not the primary selection signal.

    const size_t n = store_.n;
    if (n == 0) return;

    const float sketch_size = static_cast<float>(cfg_.sketch_size);
    size_t updated = 0;

    for (size_t i = 0; i < n; ++i) {
        // Skip excluded genomes and those with real CheckM2 scores (not exactly 50.0)
        if (store_.quality_scores[i] == 0.0f) continue;
        if (std::abs(store_.quality_scores[i] - 50.0f) > 0.01f) continue;  // has CheckM2 data

        // Sketch completeness: fraction of OPH bins filled (0-1) → scale to 0-100
        const auto& emb = embeddings_[i];
        float completeness = static_cast<float>(emb.n_real_bins) / sketch_size;
        float proxy = completeness * 100.0f;

        // Clamp to [1, 99] — 0.0 is reserved as exclusion sentinel
        store_.quality_scores[i] = std::max(1.0f, std::min(99.0f, proxy));
        embeddings_[i].quality_score = store_.quality_scores[i];
        ++updated;
    }

    if (is_verbose() && updated > 0) {
        spdlog::info("GEODESIC: computed ad-hoc quality scores for {} genomes (sketch completeness)",
                     updated);
    }
}

std::vector<SimilarityEdge> GeodesicDerep::select_representatives() {
    if (is_verbose()) spdlog::info("GEODESIC: FPS + Electrostatic Refinement (SIMD+parallel optimized)");

    std::vector<uint64_t> representatives;
    std::vector<SimilarityEdge> edges;
    size_t n = store_.n;
    edges.reserve(n);
    size_t dim = store_.dim;
    // NOTE: do NOT call omp_set_num_threads() here — it is process-global and
    // corrupts libgomp state when multiple taxa run concurrently. All OMP regions
    // below use explicit num_threads(cfg_.threads) clauses instead.

    // Track max similarity to any representative (higher = closer = more redundant)
    // Using similarity instead of distance avoids acos in the hot loop
    std::vector<float> max_sim_to_rep(n, -1.0f);
    std::vector<uint64_t> nearest_rep(n, UINT64_MAX);
    std::vector<bool> is_rep_row(n, false);

    // Similarity function: exact Jaccard for small n (no Nyström), dot product otherwise.
    auto sim_fn = [&](size_t i, size_t j) -> float {
        if (!nystrom_applied_) {
            const auto& sa = embeddings_[i].oph_sig;
            const auto& sb = embeddings_[j].oph_sig;
            return static_cast<float>(refine_jaccard_ptr(sa.data(), sb.data(), sa.size()));
        }
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
            is_rep_row[i] = true;
            const size_t rep_store_i = i;
#if GEODESIC_USE_OMP
            #pragma omp parallel for num_threads(cfg_.threads)
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
    // for argmax of isolation × sqrt(size), with quality as tie-breaker.
    // Quality should NOT multiply the score — it conflicts with isolation for ad-hoc scores.
    if (representatives.empty()) {
        size_t first_idx = 0;
        {
            float best = -1.0f;
            float best_quality = -1.0f;
            std::vector<uint64_t> tmp_sizes(n);
            for (size_t i = 0; i < n; ++i) tmp_sizes[i] = store_.genome_sizes[i];
            auto mid = tmp_sizes.begin() + static_cast<ptrdiff_t>(n / 2);
            std::nth_element(tmp_sizes.begin(), mid, tmp_sizes.end());
            float median_sz = std::max(1.0f, static_cast<float>(*mid));
            for (size_t i = 0; i < n; ++i) {
                if (store_.quality_scores[i] == 0.0f) continue;
                float len = std::sqrt(static_cast<float>(store_.genome_sizes[i]) / median_sz);
                // Primary: isolation × sqrt(size)
                float score = store_.isolation_scores[i] * len;
                if (score > best || (score == best && store_.quality_scores[i] > best_quality)) {
                    best = score;
                    best_quality = store_.quality_scores[i];
                    first_idx = i;
                }
            }
        }
        if (first_idx == n) first_idx = 0;  // fallback: all excluded (shouldn't happen)
        uint64_t first_rep = store_.genome_ids[first_idx];
        representatives.push_back(first_rep);
        is_rep_row[first_idx] = true;
        if (is_verbose()) {
            spdlog::info("GEODESIC: First representative (most isolated): genome_{} (isolation={:.4f})",
                         first_rep, store_.isolation_scores[first_idx]);
        }

        // Update similarities to first rep
#if GEODESIC_USE_OMP
        #pragma omp parallel for num_threads(cfg_.threads)
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

    const size_t max_reps = static_cast<size_t>(n * cfg_.max_rep_fraction);

    while (!active.empty() && representatives.size() < max_reps) {
        // Compute fitness for all active members in parallel
        // FPS fitness = distance × sqrt(size) — quality is a tie-breaker only
        struct FitEntry { float dist_fit; float quality; size_t idx; };
        std::vector<FitEntry> fit_active(active.size());
#if GEODESIC_USE_OMP
        #pragma omp parallel for num_threads(cfg_.threads)
#endif
        for (size_t ai = 0; ai < active.size(); ++ai) {
            size_t i = active[ai];
            float sim = max_sim_to_rep[i];
            // Fast angular distance proxy: sqrt(2(1-sim)) ≈ acos(sim) for sim near 1.
            // Preserves ranking for FPS candidate selection (monotonic in sim).
            float d_proxy = std::sqrt(2.0f * std::max(0.0f, 1.0f - sim));
            // Primary fitness: distance × sqrt(size) — pure diversity signal
            fit_active[ai] = {d_proxy * length_factors[i], store_.quality_scores[i], i};
        }

        // Partial sort: bring top-B to front (O(n) average)
        // Compare by (dist_fit DESC, quality DESC) — quality breaks ties only
        if (fit_active.empty()) break;
        auto cmp = [](const FitEntry& x, const FitEntry& y) {
            if (x.dist_fit != y.dist_fit) return x.dist_fit > y.dist_fit;
            return x.quality > y.quality;  // tie-breaker
        };
        size_t b = std::min(FPS_BATCH, fit_active.size());
        std::nth_element(fit_active.begin(), fit_active.begin() + b, fit_active.end(), cmp);
        std::sort(fit_active.begin(), fit_active.begin() + b, cmp);

        // Stopping criterion: top candidate covered ↔ sim > cos_diversity
        float top_sim = max_sim_to_rep[fit_active[0].idx];
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
            size_t i = fit_active[k].idx;
            // dist < diversity_threshold ↔ sim > cos_diversity
            if (max_sim_to_rep[i] > cos_diversity) continue;
            batch_idx.push_back(i);
            batch_gids.push_back(store_.genome_ids[i]);
            batch_vecs.push_back(store_.row(i));
            representatives.push_back(store_.genome_ids[i]);
            is_rep_row[i] = true;
        }

        // Update only active (uncovered) genomes + compact.
        // After 50% coverage, each iteration scans only the remainder → ~40-50% less total work.
        for (size_t bi : batch_idx) max_sim_to_rep[bi] = 1.0f;

        const size_t nb = batch_idx.size();
        const size_t n_active = active.size();
#if GEODESIC_USE_OMP
        #pragma omp parallel for schedule(static) num_threads(cfg_.threads)
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

        // Build list of rep store indices for top-K embedding scan.
        std::vector<size_t> rep_store_idx;
        rep_store_idx.reserve(representatives.size());
        for (size_t i = 0; i < n; ++i)
            if (is_rep_row[i])
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
                if (is_rep_row[i]) continue;
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
            if (is_rep_row[i]) continue;
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

            // Verify each top-K rep with sig1 pre-filter + dual-sketch OPH Jaccard.
            // sig1 is cheap (already in memory); skip sig2 when sig1 clearly shows
            // the pair is distant — saves ~60-70% of sig2 materializations.
            bool covered = false;
            for (int ki = 0; ki < k_check && !covered; ++ki) {
                size_t rep_emb_i = top_reps[ki].second;
                const auto& sig_a  = embeddings_[i].oph_sig;
                const auto& sig_b  = embeddings_[rep_emb_i].oph_sig;
                if (sig_a.empty() || sig_b.empty()) continue;
                double jac_exact = refine_jaccard_ptr(sig_a.data(), sig_b.data(), sig_a.size());
                // Pre-filter: if sig1 alone is well below threshold, skip sig2
                if (static_cast<float>(jac_exact) < cos_diversity * 0.9f) continue;
                if (!embeddings_[i].oph_sig2.empty() && !embeddings_[rep_emb_i].oph_sig2.empty()) {
                    const auto& s2a = embeddings_[i].oph_sig2;
                    const auto& s2b = embeddings_[rep_emb_i].oph_sig2;
                    double jac2 = refine_jaccard_ptr(s2a.data(), s2b.data(), s2a.size());
                    jac_exact = (jac_exact + jac2) * 0.5;
                }
                // dist_exact < diversity_threshold ↔ jac_exact > cos_diversity
                if (static_cast<float>(jac_exact) > cos_diversity) covered = true;
            }

            if (!covered) {
                // No top-K rep truly covers this genome → add as representative.
                representatives.push_back(store_.genome_ids[i]);
                is_rep_row[i] = true;
                rep_store_idx.push_back(i);
                newly_added.push_back(i);
                max_sim_to_rep[i] = 1.0f;
            }
        }
        if (!newly_added.empty()) {
            if (is_verbose()) spdlog::info("GEODESIC: Borderline refinement added {} representatives "
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
                const auto& sa = embeddings_[ri].oph_sig;
                const auto& sb = embeddings_[rj].oph_sig;
                double jac = refine_jaccard_ptr(sa.data(), sb.data(), sa.size());
                // dist < min_rep_distance ↔ jac > cos_min_rep
                if (static_cast<float>(jac) > cos_min_rep) {
                    unite(i, j);
                    spdlog::debug("GEODESIC: Merged rep {} into rep {} (jaccard={:.4f})", j, i, jac);
                }
            }
        }
    } else {
        // Nyström mode (large n): parallel radius searches then serial union-find.
        // search_radius() is thread-safe (read-only brute-force scan).
        std::vector<std::pair<size_t, size_t>> merge_pairs;

#if GEODESIC_USE_OMP
        std::mutex merge_mutex;
        #pragma omp parallel num_threads(cfg_.threads)
        {
            std::vector<std::pair<size_t, size_t>> local_pairs;
            #pragma omp for nowait
            for (size_t i = 0; i < representatives.size(); ++i) {
                uint64_t rep_i = representatives[i];
                auto row_it = gid_to_row_.find(rep_i);
                if (row_it == gid_to_row_.end()) continue;
                auto neighbors = index_->search_radius(embeddings_[row_it->second].vector, cfg_.min_rep_distance, &rep_set);
                for (const auto& [neighbor_id, dist] : neighbors) {
                    if (neighbor_id == rep_i) continue;
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
            auto neighbors = index_->search_radius(embeddings_[row_it->second].vector, cfg_.min_rep_distance, &rep_set);
            for (const auto& [neighbor_id, dist] : neighbors) {
                if (neighbor_id == rep_i) continue;
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

        std::fill(is_rep_row.begin(), is_rep_row.end(), false);
        for (uint64_t gid : representatives) {
            auto it = gid_to_row_.find(gid);
            if (it != gid_to_row_.end()) is_rep_row[it->second] = true;
        }

        // Selective recompute: only genomes whose nearest_rep was merged away.
        // Genomes assigned to surviving reps keep their current assignment.
#if GEODESIC_USE_OMP
        #pragma omp parallel for num_threads(cfg_.threads)
#endif
        for (size_t i = 0; i < n; ++i) {
            if (is_rep_row[i]) {
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

    // Rematerialize sig1 if it was freed after Nyström (GPK path memory optimisation).
    // Only rematerialize genomes that actually need OPH certification (non-rep, quality>0).
    // Reps need it too for the fast-path check against their assigned non-rep genomes.
    if (nystrom_applied_ && gpk_reader_ && !gpk_accessions_.empty()
        && !embeddings_.empty() && embeddings_[0].oph_sig.empty()) {
        std::vector<std::string> acc_batch;
        std::unordered_map<size_t, size_t> out_to_emb;
        acc_batch.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            if (store_.quality_scores[i] == 0.0f) continue;
            if (i >= gpk_accessions_.size()) continue;
            out_to_emb[acc_batch.size()] = i;
            acc_batch.push_back(gpk_accessions_[i]);
        }
        const MinHasher::Config cfg1{
            .kmer_size   = cfg_.kmer_size,
            .sketch_size = cfg_.sketch_size,
            .syncmer_s   = cfg_.syncmer_s,
            .seed        = cfg_.seed
        };
        gpk_reader_->visit_shard_batches(acc_batch,
            [&](genopack::ArchiveReader::ShardBatch& batch) {
                const int bsz = static_cast<int>(batch.size());
                #pragma omp parallel for schedule(dynamic, 1) num_threads(cfg_.threads)
                for (int j = 0; j < bsz; ++j) {
                    const size_t out_idx = batch[j].first;
                    const auto& eg = batch[j].second;
                    auto it = out_to_emb.find(out_idx);
                    if (it == out_to_emb.end()) continue;
                    const size_t i = it->second;
                    try {
                        MinHasher hasher1(cfg1);
                        auto oph1 = hasher1.sketch_oph_with_positions_from_buffer(
                            eg.fasta.data(), eg.fasta.size(), cfg_.sketch_size);
                        const size_t m1 = oph1.signature.size();
                        std::vector<uint16_t> sig1(m1);
                        for (size_t t = 0; t < m1; ++t)
                            sig1[t] = static_cast<uint16_t>(oph1.signature[t]);
                        embeddings_[i].oph_sig = std::move(sig1);
                    } catch (const std::exception& e) {
                        spdlog::warn("GEODESIC: sig1 rematerialize failed for {}: {}",
                                     embeddings_[i].path.filename().string(), e.what());
                    }
                }
            });
        if (is_verbose()) spdlog::info("GEODESIC: sig1 rematerialized for Phase 7c ({} genomes)",
                                       acc_batch.size());
    }

    // Phase 7c: Universal sketch-space certification.
    // The Nyström embedding is a candidate-generation layer, not a coverage guarantee.
    // Densified OPH bins inflate kernel values for sparse MAGs, causing embedding
    // dot-products to overestimate similarity. We verify every non-rep, non-excluded
    // genome against its assigned representative using exact OPH Jaccard (sig1, SIMD).
    // COVERAGE GUARANTEE: every non-rep genome is within J_cert (Jaccard equivalent of
    // ani_threshold) of some representative *in sketch space*. This is NOT a guarantee
    // in true ANI space — OPH estimation error depends on real-bin occupancy and J, so
    // borderline genomes (near the ANI threshold) may be slightly miscertified.
    // For failures, scan all current representatives exhaustively before promoting.
    if (nystrom_applied_ && !embeddings_.empty() && !embeddings_[0].oph_sig.empty()) {
        // Build rep-row index.
        std::vector<size_t> cert_rep_idx;
        cert_rep_idx.reserve(representatives.size());
        for (size_t i = 0; i < n; ++i)
            if (is_rep_row[i]) cert_rep_idx.push_back(i);

        const size_t m_oph = embeddings_[0].oph_sig.size();

        // OPH Jaccard equivalent of the user ANI threshold (e.g. 95% → J≈0.212 at k=21).
        // This is the coverage criterion: every non-rep genome must have at least one rep
        // within mapping distance. Using cos_diversity (diversity_threshold ≈ 99.9% ANI)
        // would wrongly flag all 95-99% genomes as uncovered, triggering a catastrophic
        // exhaustive scan for the whole collection.
        const double q_cert = std::pow(cfg_.ani_threshold,
                                       static_cast<double>(cfg_.kmer_size));
        const float J_cert  = static_cast<float>(q_cert / (2.0 - q_cert));

        // Two-arm OPH certification:
        // Arm 1: symmetric Jaccard(i, ri) >= J_cert (standard case).
        // Arm 2: directional containment C(small→large) for size-asymmetric pairs
        // (MAG << complete genome, size_ratio < 0.5). Containment = n_match / n_small_real
        // where n_small_real counts ALL real OPH bins in the smaller genome (from
        // real_bins_mask), and n_match counts co-occupied bins with matching hash values.
        // Using n_small_real as denominator (not co-occupied bins n_both_real) avoids
        // inflating the score when the large genome has many empty bins that reduce
        // n_both_real but do not represent missing k-mer content in the small genome.
        // oph_cert_jac: two-arm certification + Jaccard in one pass.
        // Returns {certified, jaccard} — eliminates the double refine_jaccard_ptr call
        // that the previous separate oph_certified + outer refine call required.
        // Arm 1: Jaccard >= J_cert. Arm 2: containment for MAG-vs-complete pairs.
        auto oph_cert_jac = [&](size_t i, size_t ri) -> std::pair<bool, double> {
            const auto& sig_i = embeddings_[i].oph_sig;
            const auto& sig_r = embeddings_[ri].oph_sig;
            if (sig_i.size() != m_oph || sig_r.size() != m_oph) return {false, 0.0};
            // Arm 1: symmetric Jaccard.
            const double jac = refine_jaccard_ptr(sig_i.data(), sig_r.data(), m_oph);
            if (static_cast<float>(jac) >= J_cert) return {true, jac};
            // Arm 2: directional containment for sketch-asymmetric pairs.
            // Trigger on n_real_bins ratio (not genome_size): a fragmented MAG may have
            // similar genome_size to a complete genome but far fewer occupied OPH bins.
            // n_real_bins directly measures sketch coverage, matching containment semantics.
            const uint32_t bins_i = embeddings_[i].n_real_bins;
            const uint32_t bins_r = embeddings_[ri].n_real_bins;
            if (bins_i == 0 || bins_r == 0) return {false, jac};
            const double bins_ratio = (bins_i < bins_r)
                ? static_cast<double>(bins_i) / static_cast<double>(bins_r)
                : static_cast<double>(bins_r) / static_cast<double>(bins_i);
            if (bins_ratio > 0.85) return {false, jac};
            // Identify small and large by n_real_bins (sketch coverage).
            const bool i_is_small = (bins_i <= bins_r);
            const auto& mask_small = i_is_small
                ? embeddings_[i].real_bins_mask : embeddings_[ri].real_bins_mask;
            const auto& mask_large = i_is_small
                ? embeddings_[ri].real_bins_mask : embeddings_[i].real_bins_mask;
            const uint16_t* sig_small = i_is_small ? sig_i.data() : sig_r.data();
            const uint16_t* sig_large = i_is_small ? sig_r.data() : sig_i.data();
            if (mask_small.empty() || mask_large.empty()) return {false, jac};
            int n_match = 0, n_small_real = 0, n_both_real = 0;
            const size_t n_words = std::min(mask_small.size(), mask_large.size());
            for (size_t w = 0; w < n_words; ++w) {
                n_small_real += __builtin_popcountll(mask_small[w]);
                uint64_t both_real = mask_small[w] & mask_large[w];
                n_both_real += __builtin_popcountll(both_real);
                while (both_real) {
                    const int bit = __builtin_ctzll(both_real);
                    const size_t t = w * 64 + static_cast<size_t>(bit);
                    if (t < m_oph && sig_small[t] == sig_large[t]) ++n_match;
                    both_real &= both_real - 1;
                }
            }
            if (n_small_real == 0) return {false, jac};
            // Containment threshold is q_cert = ANI^k (raw probability), not J_cert.
            // J_cert = q/(2-q) is derived for symmetric Jaccard; at 95% ANI / k=21,
            // q_cert ≈ 0.34 vs J_cert ≈ 0.21. Using J_cert here would be too permissive.
            //
            // False-match correction: each co-occupied bin has a 1/65536 chance of a
            // hash collision (uint16 OPH). Subtract the expected false matches before
            // computing containment. The maximum shift is bounded by
            // n_both_real / (65536 × n_small_real) ≤ 1/65536 ≈ 0.0015 pp — effectively
            // zero for any practical threshold. Included for statistical correctness.
            const double expected_collisions = n_both_real / 65536.0;
            const double containment =
                std::max(0.0, static_cast<double>(n_match) - expected_collisions)
                / n_small_real;
            return {containment >= q_cert, jac};
        };

        // cert_scan: find the best certified rep for genome i.
        //
        // Strategy: use HNSW to get top-K nearest reps in O(log n) instead of O(n_reps × d).
        // The HNSW index covers all n genomes; we filter to rep_set to get rep-only neighbors.
        // For correctness, if no certified rep is found in the HNSW top-K, fall back to a
        // linear scan over cert_rep_vecs (the same O(n_reps × d) path as before, but rare).
        static constexpr size_t CERT_TOP_K = 64;
        const size_t n_cert_reps = cert_rep_idx.size();
        const size_t embed_d = (!embeddings_.empty() && !embeddings_[0].vector.empty())
            ? embeddings_[0].vector.size() : 0;

        // Fallback: gather rep embedding pointers for the rare linear-scan case.
        // Used when n_reps <= CERT_TOP_K or when HNSW fails to certify in top-K.
        std::vector<const float*> cert_rep_vecs;
        cert_rep_vecs.resize(n_cert_reps);
        for (size_t k = 0; k < n_cert_reps; ++k)
            cert_rep_vecs[k] = embeddings_[cert_rep_idx[k]].vector.data();

        // Build gid→store-row lookup for cert_rep_idx (needed to map HNSW results to rows).
        // gid_to_row_ maps genome_id → embeddings_ row; reps are a tiny subset of all n.
        const bool use_hnsw_cert = (index_->index && embed_d > 0 && n_cert_reps > CERT_TOP_K);

        auto cert_scan = [&](size_t i,
                             double& best_cert_jac, size_t& best_rep_row) {
            if (use_hnsw_cert) {
                // HNSW search filtered to rep_set: O(log n) candidate generation.
                // ef is temporarily raised to ensure enough rep neighbors are returned
                // even when reps are sparse (n_reps << n). The saved value is restored.
                const int saved_ef = index_->ef_search;
                index_->index->setEf(std::max(saved_ef,
                    static_cast<int>(std::min(n_cert_reps * 4, n))));
                auto neighbors = index_->search(embeddings_[i].vector, CERT_TOP_K, &rep_set);
                index_->index->setEf(saved_ef);

                // Phase 1: HNSW top-K (sorted by ascending angular distance = descending sim)
                for (const auto& [gid, dist] : neighbors) {
                    auto row_it = gid_to_row_.find(gid);
                    if (row_it == gid_to_row_.end()) continue;
                    const size_t ri = row_it->second;
                    auto [cert, jac] = oph_cert_jac(i, ri);
                    if (!cert) continue;
                    if (jac > best_cert_jac) { best_cert_jac = jac; best_rep_row = ri; }
                }
                // Phase 2: linear fallback only if HNSW top-K found nothing certified
                if (best_rep_row == SIZE_MAX) {
                    for (size_t k = 0; k < n_cert_reps; ++k) {
                        size_t ri = cert_rep_idx[k];
                        auto [cert, jac] = oph_cert_jac(i, ri);
                        if (!cert) continue;
                        if (jac > best_cert_jac) { best_cert_jac = jac; best_rep_row = ri; }
                    }
                }
            } else {
                // Plain scan: small n_reps or no HNSW index
                for (size_t ri : cert_rep_idx) {
                    auto [cert, jac] = oph_cert_jac(i, ri);
                    if (!cert) continue;
                    if (jac > best_cert_jac) { best_cert_jac = jac; best_rep_row = ri; }
                }
            }
        };

        std::vector<size_t> repair_queue;
#if GEODESIC_USE_OMP
        {
            std::vector<std::vector<size_t>> tl_repair(static_cast<size_t>(cfg_.threads));
            #pragma omp parallel num_threads(cfg_.threads)
            {
                auto& local_repair = tl_repair[static_cast<size_t>(omp_get_thread_num())];
                #pragma omp for schedule(dynamic, 256)
                for (size_t i = 0; i < n; ++i) {
                    if (is_rep_row[i]) continue;
                    if (store_.quality_scores[i] == 0.0f) continue;
                    const auto& sig_i = embeddings_[i].oph_sig;
                    if (sig_i.size() != m_oph) continue;

                    // Fast path: check assigned rep first (single O(m) check).
                    auto rep_it = gid_to_row_.find(nearest_rep[i]);
                    if (rep_it != gid_to_row_.end()) {
                        auto [cert, jac] = oph_cert_jac(i, rep_it->second);
                        if (cert) {
                            max_sim_to_rep[i] = static_cast<float>(jac);
                            continue;
                        }
                    }

                    // Exhaustive fallback: embedding-ranked scan with early phase-1 termination.
                    double best_cert_jac = -1.0;
                    size_t best_rep_row = SIZE_MAX;
                    cert_scan(i, best_cert_jac, best_rep_row);

                    if (best_rep_row != SIZE_MAX) {
                        nearest_rep[i]    = store_.genome_ids[best_rep_row];
                        max_sim_to_rep[i] = static_cast<float>(best_cert_jac);
                    } else {
                        local_repair.push_back(i);
                    }
                }
            }
            for (auto& local : tl_repair)
                repair_queue.insert(repair_queue.end(), local.begin(), local.end());
            std::sort(repair_queue.begin(), repair_queue.end());
        }
#else
        for (size_t i = 0; i < n; ++i) {
            if (is_rep_row[i]) continue;
            if (store_.quality_scores[i] == 0.0f) continue;
            const auto& sig_i = embeddings_[i].oph_sig;
            if (sig_i.size() != m_oph) continue;

            // Fast path: check assigned rep first.
            auto rep_it = gid_to_row_.find(nearest_rep[i]);
            if (rep_it != gid_to_row_.end()) {
                auto [cert, jac] = oph_cert_jac(i, rep_it->second);
                if (cert) {
                    max_sim_to_rep[i] = static_cast<float>(jac);
                    continue;
                }
            }

            // Exhaustive fallback: embedding-ranked scan.
            double best_cert_jac = -1.0;
            size_t best_rep_row = SIZE_MAX;
            cert_scan(i, best_cert_jac, best_rep_row);

            if (best_rep_row != SIZE_MAX) {
                nearest_rep[i]    = store_.genome_ids[best_rep_row];
                max_sim_to_rep[i] = static_cast<float>(best_cert_jac);
            } else {
                repair_queue.push_back(i);
            }
        }
#endif

        const size_t n_reps_pre_repair = cert_rep_idx.size();
        if (!repair_queue.empty()) {
            spdlog::info("GEODESIC: Phase 7c repair: {} reps added ({} → {}); "
                         "post-FPS sketch-space repair (Nyström/Laplacian embedding coverage gap)",
                         repair_queue.size(), n_reps_pre_repair,
                         n_reps_pre_repair + repair_queue.size());
            std::vector<const float*> repair_vecs;
            std::vector<uint64_t>     repair_gids;
            repair_vecs.reserve(repair_queue.size());
            repair_gids.reserve(repair_queue.size());
            for (size_t i : repair_queue) {
                representatives.push_back(store_.genome_ids[i]);
                is_rep_row[i]    = true;
                max_sim_to_rep[i] = 1.0f;
                nearest_rep[i]   = store_.genome_ids[i];
                repair_vecs.push_back(store_.row(i));
                repair_gids.push_back(store_.genome_ids[i]);
            }
            const size_t n_repair = repair_queue.size();
#if GEODESIC_USE_OMP
            #pragma omp parallel for schedule(static) num_threads(cfg_.threads)
#endif
            for (size_t j = 0; j < n; ++j) {
                if (is_rep_row[j]) continue;
                const float* vj = store_.row(j);
                for (size_t k = 0; k < n_repair; ++k) {
                    float s = dot_product_simd(repair_vecs[k], vj, dim);
                    if (s > max_sim_to_rep[j]) {
                        max_sim_to_rep[j] = s;
                        nearest_rep[j]    = repair_gids[k];
                    }
                }
            }
        } else if (is_verbose()) {
            spdlog::info("GEODESIC: Phase 7c: all non-rep genomes OPH-certified");
        }

        // Pruning pass: remove redundant representatives (sequential greedy removal).
        //
        // The single-pass approach ("remove r if all its genomes have other coverage")
        // fails because of cascading: removing A makes B's genomes uncovered, then B
        // can't be removed even if it was conceptually redundant — but worse, removing
        // all reps simultaneously breaks everything.
        //
        // Correct approach:
        //   1. Precompute cert_reps[i] = list of ALL reps that OPH-certify genome i
        //      (up to kPruneK candidates via HNSW, or exhaustive for small n_reps).
        //   2. Sequentially try to remove each rep (sorted most-redundant first).
        //      "Can remove r" iff:
        //        a. every non-rep genome assigned to r has another surviving certifier, AND
        //        b. genome r itself (now demoted to non-rep) has a surviving certifier.
        //   3. If removable: erase from survivors, reassign affected genomes.
        //
        // This handles cascading correctly because we check against the live survivor set.
        if (representatives.size() >= 2) {
            const size_t n_reps_pre_prune = representatives.size();
            static constexpr size_t kPruneK = 16;
            const bool hnsw_prune = (index_->index && embed_d > 0 &&
                                     cert_rep_idx.size() > kPruneK);
            std::unordered_set<uint64_t> prune_rep_set(representatives.begin(),
                                                        representatives.end());

            // Step 1: for each genome (including reps, to handle rep demotion), find
            // which reps cover it at the FPS diversity_threshold scale.
            // IMPORTANT: use sim_fn / cos_diversity (same as FPS), NOT J_cert (ANI threshold).
            // Pruning with J_cert would remove reps that are still needed for diversity
            // coverage — FPS's diversity_threshold is much tighter than J_cert, and
            // removing a rep that is "covered at 95% ANI" but not "covered at diversity
            // scale" destroys the uniformity of the rep set.
            std::vector<std::vector<uint64_t>> cert_reps(n);
#if GEODESIC_USE_OMP
            #pragma omp parallel for schedule(dynamic, 128) num_threads(cfg_.threads)
#endif
            for (size_t i = 0; i < n; ++i) {
                if (store_.quality_scores[i] == 0.0f) continue;

                std::vector<uint64_t> certifiers;
                if (hnsw_prune) {
                    auto neighbors = index_->search(embeddings_[i].vector, kPruneK,
                                                    &prune_rep_set);
                    for (const auto& [rep_gid, dist] : neighbors) {
                        // dist < diversity_threshold ↔ dist (angular) < cfg_.diversity_threshold
                        // HNSW returns angular distance; compare directly.
                        if (dist < cfg_.diversity_threshold)
                            certifiers.push_back(rep_gid);
                    }
                } else {
                    for (size_t ri : cert_rep_idx) {
                        float s = sim_fn(i, ri);
                        // sim > cos_diversity ↔ dist < diversity_threshold
                        if (s > cos_diversity)
                            certifiers.push_back(store_.genome_ids[ri]);
                    }
                }
                cert_reps[i] = std::move(certifiers);
            }

            // Step 2: build genome-to-rep assignment and sequential greedy removal.
            std::unordered_set<uint64_t> survivors(representatives.begin(),
                                                    representatives.end());
            // assigned_rep[i]: which surviving rep genome i is currently assigned to.
            std::vector<uint64_t> assigned_rep(n, UINT64_MAX);
            for (size_t i = 0; i < n; ++i) {
                if (store_.quality_scores[i] == 0.0f) continue;
                if (cert_reps[i].empty()) continue;
                // Initial assignment: prefer nearest_rep[i] if it certifies, else first certifier.
                for (uint64_t r : cert_reps[i]) {
                    if (r == nearest_rep[i]) { assigned_rep[i] = r; break; }
                }
                if (assigned_rep[i] == UINT64_MAX) assigned_rep[i] = cert_reps[i][0];
            }

            // Helper: for genome i, find its best surviving certifier excluding 'skip'.
            auto best_surviving_cert = [&](size_t i, uint64_t skip) -> uint64_t {
                for (uint64_t cr : cert_reps[i])
                    if (cr != skip && survivors.count(cr)) return cr;
                return UINT64_MAX;
            };

            // Sort reps: try those with fewest exclusively-assigned genomes first
            // (most likely to be removable without cascading).
            std::vector<uint64_t> reps_sorted = representatives;
            {
                std::unordered_map<uint64_t, int> excl_count;
                excl_count.reserve(reps_sorted.size());
                for (uint64_t r : reps_sorted) excl_count[r] = 0;
                for (size_t i = 0; i < n; ++i) {
                    if (assigned_rep[i] == UINT64_MAX) continue;
                    if (store_.quality_scores[i] == 0.0f) continue;
                    if (is_rep_row[i]) continue;  // rep genomes handled separately
                    if (best_surviving_cert(i, assigned_rep[i]) == UINT64_MAX)
                        ++excl_count[assigned_rep[i]];
                }
                std::stable_sort(reps_sorted.begin(), reps_sorted.end(),
                    [&](uint64_t a, uint64_t b) { return excl_count[a] < excl_count[b]; });
            }

            size_t n_pruned_total = 0;
            for (uint64_t r : reps_sorted) {
                if (!survivors.count(r)) continue;

                // Get store index of rep genome r (needed for case b check).
                auto r_row_it = gid_to_row_.find(r);
                if (r_row_it == gid_to_row_.end()) continue;
                const size_t r_row = r_row_it->second;

                // Case (a): all non-rep genomes currently assigned to r have another certifier.
                bool can_remove = true;
                for (size_t i = 0; i < n; ++i) {
                    if (assigned_rep[i] != r) continue;
                    if (is_rep_row[i]) continue;
                    if (store_.quality_scores[i] == 0.0f) continue;
                    if (best_surviving_cert(i, r) == UINT64_MAX) { can_remove = false; break; }
                }
                if (!can_remove) continue;

                // Case (b): genome r itself (if demoted) would be certified by a surviving rep.
                // We need a surviving rep s ≠ r that certifies r.
                if (best_surviving_cert(r_row, r) == UINT64_MAX) continue;

                // Safe to remove.
                survivors.erase(r);
                ++n_pruned_total;

                // Reassign affected non-rep genomes.
                for (size_t i = 0; i < n; ++i) {
                    if (assigned_rep[i] != r) continue;
                    if (is_rep_row[i]) continue;
                    if (store_.quality_scores[i] == 0.0f) continue;
                    uint64_t next = best_surviving_cert(i, r);
                    if (next != UINT64_MAX) assigned_rep[i] = next;
                }
            }

            if (n_pruned_total > 0) {
                spdlog::info("GEODESIC: Pruning: removed {} redundant reps ({} → {})",
                             n_pruned_total, n_reps_pre_prune, survivors.size());

                representatives.clear();
                for (uint64_t gid : survivors) representatives.push_back(gid);

                // Rebuild is_rep_row, cert_rep_idx.
                std::fill(is_rep_row.begin(), is_rep_row.end(), false);
                cert_rep_idx.clear();
                for (uint64_t gid : representatives) {
                    auto it = gid_to_row_.find(gid);
                    if (it != gid_to_row_.end()) {
                        is_rep_row[it->second] = true;
                        cert_rep_idx.push_back(it->second);
                    }
                }
                // Update nearest_rep / max_sim_to_rep from precomputed cert_reps.
                for (size_t i = 0; i < n; ++i) {
                    if (is_rep_row[i]) continue;
                    if (store_.quality_scores[i] == 0.0f) continue;
                    // Check assigned rep is still alive; if not, find first surviving certifier.
                    auto rep_it = gid_to_row_.find(nearest_rep[i]);
                    const bool old_ok = (rep_it != gid_to_row_.end() &&
                                         is_rep_row[rep_it->second]);
                    if (old_ok) continue;
                    uint64_t new_rep = best_surviving_cert(i, UINT64_MAX);
                    if (new_rep != UINT64_MAX) {
                        nearest_rep[i] = new_rep;
                        auto nrit = gid_to_row_.find(new_rep);
                        if (nrit != gid_to_row_.end())
                            max_sim_to_rep[i] = sim_fn(i, nrit->second);
                    } else {
                        // No known certifier in cert_reps — fall back to cert_scan.
                        double best_jac = -1.0;
                        size_t best_row = SIZE_MAX;
                        cert_scan(i, best_jac, best_row);
                        if (best_row != SIZE_MAX) {
                            nearest_rep[i]    = store_.genome_ids[best_row];
                            max_sim_to_rep[i] = static_cast<float>(best_jac);
                        }
                    }
                }
            } else if (is_verbose()) {
                spdlog::info("GEODESIC: Pruning: no redundant reps found");
            }
        }
    }

    // Phase 4: Build edges (genome → nearest representative)
    // weight_raw = ANI fraction derived from Jaccard via Mash formula.
    auto jaccard_to_ani_frac = [&](double J) -> float {
        if (J <= 0.0) return 0.0f;
        double ratio = 2.0 * J / (1.0 + J);
        return static_cast<float>(std::pow(ratio, 1.0 / cfg_.kmer_size));
    };
    // Build O(1) lookup: genome_id → index (same index into embeddings_)
    std::unordered_map<uint64_t, size_t> id_to_idx;
    id_to_idx.reserve(n);
    for (size_t r = 0; r < n; ++r) id_to_idx[store_.genome_ids[r]] = r;

    // Precompute path strings to avoid repeated .string() conversions
    std::vector<std::string> path_strings(n);
    for (size_t i = 0; i < n; ++i) path_strings[i] = store_.paths[i].string();

    // OPH refinement: for pairs whose 256-dim estimate is near the threshold,
    // refine J via exact bin-counting. SNR of direct OPH comparison with M=4096:
    //   at 90% ANI (J=0.065): SNR=26  vs  CountSketch@256: SNR=1
    //   at 95% ANI (J=0.212): SNR=50  vs  CountSketch@256: SNR=3.5
    // Borderline zone ≈ 3-sigma of the CountSketch estimator: ±3/√dim ≈ ±0.19
    const bool has_oph = !embeddings_.empty() && !embeddings_[0].oph_sig.empty();
    const double q_t = std::pow(cfg_.ani_threshold, static_cast<double>(cfg_.kmer_size));
    const double J_threshold = q_t / (2.0 - q_t);
    const double J_margin = 3.0 / std::sqrt(static_cast<double>(runtime_dim_));

    for (size_t i = 0; i < n; ++i) {
        if (is_rep_row[i]) continue;

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
            const auto& sa = embeddings_[i].oph_sig;
            const auto& sb = embeddings_[rep_idx].oph_sig;
            J_est = refine_jaccard_ptr(sa.data(), sb.data(), sa.size());
        }
        J_est = std::clamp(J_est, 0.0, 1.0);

        SimilarityEdge edge;
        edge.source   = path_strings[i];
        edge.target   = path_strings[rep_idx];
        edge.weight_raw = jaccard_to_ani_frac(J_est);
        edge.weight     = jaccard_to_ani_frac(J_est);
        edge.aln_frac   = 1.0;
        edges.push_back(edge);
    }

    if (is_verbose()) spdlog::info("GEODESIC: DONE - {} diverse representatives from {} genomes (SIMD+parallel, NO skani!)",
                 representatives.size(), n);

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

std::vector<GeodesicDerep::OutlierCandidate>
GeodesicDerep::detect_outlier_candidates(float z_threshold) const {
    if (embeddings_.size() <= SMALL_N_THRESHOLD) return {};

    size_t n   = embeddings_.size();
    size_t dim = static_cast<size_t>(runtime_dim_);

    // Per-component isolation score thresholds (MAD-based).
    // Genomes in the same connected subpopulation (component_ids_[i] >= 0) are compared
    // to their component's local median + z * 1.4826 * MAD. MAD is robust to heavy tails
    // and prevents a few extreme outliers from inflating the threshold (as mean+z*std does).
    // Genomes with component_ids_[i] == -1 were already global outliers during MST and
    // are always flagged.
    const int n_comps = [&]() -> int {
        if (component_ids_.empty() || component_ids_.size() < n) return 0;
        int mx = -1;
        for (size_t i = 0; i < n; ++i) if (component_ids_[i] > mx) mx = component_ids_[i];
        return mx + 1;
    }();

    // Collect per-component isolation scores for MAD computation.
    std::vector<std::vector<float>> comp_scores(n_comps);
    for (size_t i = 0; i < n; ++i) {
        const int c = (n_comps > 0) ? component_ids_[i] : -1;
        if (c < 0) continue;
        comp_scores[c].push_back(embeddings_[i].isolation_score);
    }
    std::vector<float> comp_thr(n_comps, std::numeric_limits<float>::max());
    for (int c = 0; c < n_comps; ++c) {
        auto& sc = comp_scores[c];
        if (sc.size() < 2) continue;
        size_t mid = sc.size() / 2;
        std::nth_element(sc.begin(), sc.begin() + mid, sc.end());
        float median_c = sc[mid];
        // MAD: median of absolute deviations from median
        std::vector<float> abs_devs(sc.size());
        for (size_t j = 0; j < sc.size(); ++j)
            abs_devs[j] = std::abs(sc[j] - median_c);
        std::nth_element(abs_devs.begin(), abs_devs.begin() + mid, abs_devs.end());
        float mad_c = abs_devs[mid];
        if (mad_c > 1e-9f) {
            comp_thr[c] = median_c + z_threshold * 1.4826f * mad_c;
        } else {
            // MAD=0 fallback: use IQR
            size_t q1i = sc.size() / 4;
            size_t q3i = 3 * sc.size() / 4;
            std::nth_element(sc.begin(), sc.begin() + q1i, sc.end());
            float q1 = sc[q1i];
            std::nth_element(sc.begin(), sc.begin() + q3i, sc.end());
            float q3 = sc[q3i];
            float iqr = q3 - q1;
            if (iqr > 1e-9f) {
                comp_thr[c] = median_c + z_threshold * iqr;
            }
            // else: comp_thr[c] stays max_float (no flagging)
        }
    }

    // Global fallback (used when component_ids_ unavailable, e.g. brute-force small-n path)
    // MAD-based threshold: robust to heavy-tailed isolation score distributions.
    std::vector<float> iso_scores(n);
    for (size_t i = 0; i < n; ++i)
        iso_scores[i] = embeddings_[i].isolation_score;
    size_t g_mid = n / 2;
    std::nth_element(iso_scores.begin(), iso_scores.begin() + g_mid, iso_scores.end());
    const float iso_median = iso_scores[g_mid];
    std::vector<float> iso_abs_devs(n);
    for (size_t i = 0; i < n; ++i)
        iso_abs_devs[i] = std::abs(iso_scores[i] - iso_median);
    std::nth_element(iso_abs_devs.begin(), iso_abs_devs.begin() + g_mid, iso_abs_devs.end());
    const float iso_mad = iso_abs_devs[g_mid];
    float nn_threshold;
    if (iso_mad > 1e-9f) {
        nn_threshold = iso_median + z_threshold * 1.4826f * iso_mad;
    } else {
        // MAD=0 fallback: use IQR
        size_t g_q1 = n / 4;
        size_t g_q3 = 3 * n / 4;
        std::nth_element(iso_scores.begin(), iso_scores.begin() + g_q1, iso_scores.end());
        float q1 = iso_scores[g_q1];
        std::nth_element(iso_scores.begin(), iso_scores.begin() + g_q3, iso_scores.end());
        float q3 = iso_scores[g_q3];
        float iqr = q3 - q1;
        if (iqr > 1e-9f) {
            nn_threshold = iso_median + z_threshold * iqr;
        } else {
            nn_threshold = std::numeric_limits<float>::max();
        }
    }

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
    // isolation_score > median + z_threshold * 1.4826 * MAD → statistically anomalous k-NN distance
    // → likely misassigned taxonomy.

    std::vector<OutlierCandidate> candidates;
    for (size_t i = 0; i < n; ++i) {
        const int comp_id = (n_comps > 0) ? component_ids_[i] : -1;
        // Per-component threshold: use max(comp_thr, global_thr).
        // The max ensures tight components (near-zero MAD → comp_thr ≈ median)
        // fall back to the global threshold and don't over-flag normal variation.
        // Per-component threshold only tightens flagging when a component is more diverse
        // than the global average — protecting rare-but-genuine heterogeneous sublineages.
        const float thr = (comp_id >= 0)
            ? std::max(comp_thr[comp_id], nn_threshold) : nn_threshold;
        const bool is_nn_outlier = (comp_id < 0) || (embeddings_[i].isolation_score > thr);

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
        // Note: only is_nn_outlier reaches here (is_nn_outlier guard above); size_outlier-only is dead.
        std::string reason = is_size_outlier ? "nn_outlier+size_outlier" : "nn_outlier";
        // Composite hints: secondary signals that reinforce or explain the primary flag.
        // Threshold z>3 rather than z>2: at z>2, ~2% of genomes in a normal distribution would
        // be flagged by chance; z>3 reduces false-positive hints to ~0.1%.
        if (kmer_div_z > 3.0f) reason += ":kmer_diverse";
        if (sz_z < -2.0f) reason += ":small_genome";
        if (embeddings_[i].n_contigs > 100) reason += ":fragmented";

        OutlierCandidate c;
        c.genome_id            = embeddings_[i].genome_id;
        c.centroid_distance    = angular_distance(embeddings_[i].vector, centroid);
        c.isolation_score      = embeddings_[i].isolation_score;
        c.anomaly_score        = embeddings_[i].isolation_score;
        c.genome_size_zscore   = sz_z;
        c.nn_outlier           = is_nn_outlier;
        c.kmer_div_zscore      = kmer_div_z;
        c.margin_to_threshold  = embeddings_[i].isolation_score - thr;
        c.flag_reason          = std::move(reason);
        c.path                 = embeddings_[i].path;
        c.n_contigs            = embeddings_[i].n_contigs;
        c.genome_length_bp     = embeddings_[i].genome_size;
        candidates.push_back(c);
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.anomaly_score > b.anomaly_score; });

    const size_t n_misassigned = std::count_if(candidates.begin(), candidates.end(),
                                                [](const auto& c) { return c.nn_outlier; });
    if (!candidates.empty())
        spdlog::info("GEODESIC: {} outlier candidates ({} misassigned, "
                     "{} components, global_thr={:.4f}, median={:.3f}, MAD={:.4f})",
                     candidates.size(), n_misassigned,
                     n_comps, nn_threshold, iso_median, iso_mad);

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
    #pragma omp parallel for reduction(+:coverage_sum,n_valid) reduction(min:coverage_min) reduction(max:coverage_max) num_threads(cfg_.threads)
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

        // Convert to distance. After Nyström the embedding vectors live in spectral
        // space, so dot products no longer estimate OPH Jaccard. Use OPH Jaccard directly
        // for the coverage_mean/percentile stats when Nyström has been applied.
        double min_dist;
        if (nystrom_applied_ && best_rep != SIZE_MAX &&
            !embeddings_[i].oph_sig.empty() && !embeddings_[best_rep].oph_sig.empty() &&
            !embeddings_[i].real_bins_mask.empty() && !embeddings_[best_rep].real_bins_mask.empty()) {
            const auto& qe = embeddings_[i];
            const auto& re = embeddings_[best_rep];
            int n11 = 0, n_union = 0;
            const int m = cfg_.sketch_size;
            for (int t = 0; t < m; ++t) {
                const bool q_real = (qe.real_bins_mask[t >> 6] >> (t & 63)) & 1ULL;
                const bool r_real = (re.real_bins_mask[t >> 6] >> (t & 63)) & 1ULL;
                if (!q_real && !r_real) continue;
                ++n_union;
                if (q_real && r_real && qe.oph_sig[t] == re.oph_sig[t]) ++n11;
            }
            const double j = (n_union > 0) ? static_cast<double>(n11) / n_union : 0.0;
            min_dist = std::acos(std::clamp(j, 0.0, 1.0)) / M_PI;
        } else {
            min_dist = std::acos(std::clamp(max_sim, -1.0f, 1.0f)) / M_PI;
        }
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
        #pragma omp parallel for reduction(+:cnt99,cnt98,cnt97,cnt95) num_threads(cfg_.threads)
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
        std::mt19937_64 rng(cfg_.seed);

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

// ── Adaptive k-selection ──────────────────────────────────────────────────────

int GeodesicDerep::probe_kmer_size_(const std::vector<std::string>& accessions,
                                    IPackReader& gpk) const
{
    static constexpr size_t kProbeN    = 300;
    static constexpr size_t kProbeBins = 500;

    const size_t n = accessions.size();
    if (n < 20) return 0;

    const auto avail_ks = gpk.available_kmer_sizes();
    if (avail_ks.size() < 2) return 0;

    const size_t probe_n  = std::min(kProbeN, n / 5);
    const size_t stride   = n / probe_n;
    const uint32_t probe_k  = avail_ks.back();    // largest k → most conservative (avoids k-switch)
    const uint32_t probe_sz = static_cast<uint32_t>(
        std::min(kProbeBins, static_cast<size_t>(cfg_.sketch_size)));

    // Collect probe accessions, then batch-fetch via visit_sketch_batches so V3
    // archives can parallelise frame decompression instead of paying one decompress
    // per call (300× single-threaded frame decompress would dominate setup).
    std::vector<std::string> probe_accs;
    probe_accs.reserve(probe_n);
    for (size_t i = 0; i < probe_n; ++i)
        probe_accs.push_back(accessions[i * stride]);

    // Pre-size slots so concurrent callback writes (under OMP in sketch_for_ids)
    // land in distinct indices — no mutex needed.
    std::vector<std::vector<uint32_t>> sigs_slots(probe_n);
    std::vector<uint8_t>               sig_valid(probe_n, 0);
    gpk.visit_sketch_batches(probe_accs, probe_k, probe_sz,
        [&](size_t i, const genopack::SketchResult& sk) {
            sigs_slots[i].assign(sk.sig, sk.sig + sk.sketch_size);
            sig_valid[i] = 1;
        });
    std::vector<std::vector<uint32_t>> sigs;
    sigs.reserve(probe_n);
    for (size_t i = 0; i < probe_n; ++i)
        if (sig_valid[i]) sigs.push_back(std::move(sigs_slots[i]));
    if (sigs.size() < 10) return 0;

    // Brute-force NN Jaccard distance for each probe genome.
    const size_t m = sigs.size();
    std::vector<float> nn(m, 1.0f);
    for (size_t i = 0; i < m; ++i) {
        const size_t bins = sigs[i].size();
        for (size_t j = 0; j < m; ++j) {
            if (i == j) continue;
            size_t matches = 0;
            for (size_t b = 0; b < bins; ++b)
                matches += (sigs[i][b] == sigs[j][b]);
            float d = 1.0f - float(matches) / float(bins);
            if (d < nn[i]) nn[i] = d;
        }
    }
    std::sort(nn.begin(), nn.end());
    const float p95 = nn[static_cast<size_t>(m * 0.95)];
    const int best_k = select_best_k_for_diversity(p95);
    spdlog::info("GEODESIC: k pre-probe (n={} sample, bins={}): p95_nn={:.4f} → k={}",
                 m, probe_sz, p95, best_k);
    return best_k;
}

int GeodesicDerep::select_best_k_for_diversity(float p95_nn_dist) {
    // Thresholds derived from OPH Jaccard ↔ ANI relationship:
    //   p95 < 0.002  → clonal (>99.9% ANI) → long k discriminates paralogs better
    //   p95 < 0.010  → moderate diversity (>99% ANI)
    //   p95 ≥ 0.010  → diverse (cross-species contamination likely) → short k has higher sensitivity
    if (p95_nn_dist < 0.002f) return 31;
    if (p95_nn_dist < 0.010f) return 21;
    return 16;
}

bool GeodesicDerep::maybe_reselect_k(
    const NNDistStats& stats,
    const std::unordered_map<std::string, double>& quality_scores)
{
    if (!gpk_reader_ || gpk_accessions_.empty()) return false;

    const int best_k = select_best_k_for_diversity(static_cast<float>(stats.p95));
    if (best_k == cfg_.kmer_size) return false;

    if (!gpk_reader_->has_kmer_size(static_cast<uint32_t>(best_k))) {
        std::string avail;
        for (uint32_t k : gpk_reader_->available_kmer_sizes())
            { if (!avail.empty()) avail += ','; avail += std::to_string(k); }
        spdlog::info("GEODESIC: adaptive k: best_k={} not available in GPK (stored: [{}]), keeping k={}",
                     best_k, avail, cfg_.kmer_size);
        return false;
    }

    spdlog::info("GEODESIC: adaptive k: p95_nn={:.4f} → switching k={} → k={}, re-embedding",
                 stats.p95, cfg_.kmer_size, best_k);

    const int prev_k = cfg_.kmer_size;
    cfg_.kmer_size = best_k;

    // Reset state that will be rebuilt.
    embeddings_.clear();
    store_ = SoAStore{};
    index_ = std::make_unique<HNSWIndex>();
    component_ids_.clear();
    gid_to_row_.clear();
    failed_reads_.clear();
    buf_cache_.clear();
    buf_cache_bytes_.store(0);
    nystrom_applied_ = false;

    kmer_size_locked_ = true;
    build_index_from_gpk_sketches(gpk_accessions_, gpk_paths_, *gpk_reader_, quality_scores);
    kmer_size_locked_ = false;

    return true;
}

} // namespace derep
