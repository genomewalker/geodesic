#include "core/sketch/minhash.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#ifdef __AVX2__
#include <immintrin.h>
#endif
#include "io/gz_reader.hpp"

namespace derep {

namespace {

// Lookup table for base encoding: A=0, C=1, G=2, T=3, invalid=255
alignas(64) constexpr uint8_t BASE_ENCODE[256] = {
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,  0,255,  1,255,255,255,  2,255,255,255,255,255,255,255,255,
    255,255,255,255,  3,255,255,255,255,255,255,255,255,255,255,255,
    255,  0,255,  1,255,255,255,  2,255,255,255,255,255,255,255,255,
    255,255,255,255,  3,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
};

inline uint64_t encode_base(char c) {
    return BASE_ENCODE[static_cast<uint8_t>(c)];
}

template <uint32_t Bins>
[[gnu::always_inline]] inline uint32_t fast_range_u64(uint64_t h) {
    static_assert(Bins > 0);
    return static_cast<uint32_t>((static_cast<__uint128_t>(h) * Bins) >> 64);
}

[[gnu::always_inline]] inline uint32_t fast_range_u64(uint64_t h, uint32_t bins) {
    return static_cast<uint32_t>((static_cast<__uint128_t>(h) * bins) >> 64);
}

[[gnu::always_inline]] inline uint64_t wymix(uint64_t a, uint64_t b) {
    const __uint128_t product = static_cast<__uint128_t>(a) * b;
    return static_cast<uint64_t>(product) ^ static_cast<uint64_t>(product >> 64);
}

[[gnu::always_inline]] inline uint64_t oph_hash_wymix(uint64_t canonical, uint64_t seed) {
    constexpr uint64_t P0 = 0xa0761d6478bd642fULL;
    constexpr uint64_t P1 = 0xe7037ed1a0b428dbULL;
    return wymix(canonical ^ (seed + P0), canonical ^ P1);
}

[[gnu::always_inline]] inline uint32_t oph_sig32(uint64_t canonical_hash) {
    return static_cast<uint32_t>(canonical_hash >> 32);
}

struct OPHKmerState {
    uint64_t fwd = 0;
    uint64_t rev = 0;
    int valid = 0;

    void reset() {
        fwd = 0;
        rev = 0;
        valid = 0;
    }
};

struct SyncmerEntry {
    uint64_t hash = 0;
    uint32_t pos = 0;
};

template <size_t Capacity>
struct SyncmerMinQueue {
    static_assert((Capacity & (Capacity - 1)) == 0, "Syncmer queue capacity must be a power of two");

    std::array<SyncmerEntry, Capacity> entries{};
    size_t head = 0;
    size_t tail = 0;

    void reset() {
        head = 0;
        tail = 0;
    }

    [[nodiscard]] bool empty() const { return head == tail; }

    SyncmerEntry& front() { return entries[head & (Capacity - 1)]; }
    const SyncmerEntry& front() const { return entries[head & (Capacity - 1)]; }

    SyncmerEntry& back() { return entries[(tail - 1) & (Capacity - 1)]; }
    const SyncmerEntry& back() const { return entries[(tail - 1) & (Capacity - 1)]; }

    void pop_front() { ++head; }
    void pop_back() { --tail; }

    void push_back(SyncmerEntry entry) {
        entries[tail & (Capacity - 1)] = entry;
        ++tail;
    }
};

[[gnu::always_inline]] inline uint64_t canonical_rolling_hash(const OPHKmerState& state) {
    return state.fwd ^ ((state.fwd ^ state.rev) & -(uint64_t)(state.fwd > state.rev));
}

// Compute inter-contig k-mer heterogeneity as mean pairwise OPH Jaccard distance [0,1].
// Uses a lightweight 64-bin OPH; a second pass over the in-memory buffer (already cached).
// Normal within-species variance: ~0.30-0.35. Score > 0.45 indicates chimeric contigs.
// Returns 0.0 if fewer than 2 contigs >= MIN_CONTIG_LEN are found.
float compute_contig_chimera_score(const char* data, size_t size,
                                   const MinHasher::Config& cfg) {
    constexpr int    CBINS            = 64;
    constexpr size_t MIN_CONTIG_LEN   = 10000;
    constexpr size_t MAX_CONTIGS      = 100;   // cap O(N²) pairwise cost
    constexpr uint32_t EMPTY          = std::numeric_limits<uint32_t>::max();

    const int      k         = cfg.kmer_size;
    const uint64_t seed      = cfg.seed;
    const uint64_t k_mask    = (k == 32) ? UINT64_MAX : ((1ULL << (2 * k)) - 1);
    const int      rev_shift = 2 * (k - 1);

    if (k <= 0 || k > 32) return 0.0f;

    std::vector<std::array<uint32_t, CBINS>> contig_sigs;
    std::array<uint32_t, CBINS> cur_sig;
    cur_sig.fill(EMPTY);
    size_t cur_len = 0;
    OPHKmerState state;
    bool in_header = false;

    auto mix32 = [](uint64_t x) -> uint32_t {
        x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27; x *= 0x94d049bb133111ebULL;
        x ^= x >> 31;
        return static_cast<uint32_t>(x >> 32);
    };

    auto finalize_contig = [&]() {
        if (cur_len >= MIN_CONTIG_LEN && contig_sigs.size() < MAX_CONTIGS) {
            // Densify empty bins (forward then backward pass)
            for (int t = 1; t < CBINS; ++t)
                if (cur_sig[t] == EMPTY && cur_sig[t - 1] != EMPTY)
                    cur_sig[t] = mix32(static_cast<uint64_t>(cur_sig[t - 1]) ^ static_cast<uint64_t>(t));
            for (int t = CBINS - 2; t >= 0; --t)
                if (cur_sig[t] == EMPTY && cur_sig[t + 1] != EMPTY)
                    cur_sig[t] = mix32(static_cast<uint64_t>(cur_sig[t + 1]) ^ static_cast<uint64_t>(t));
            if (cur_sig[0] != EMPTY)
                contig_sigs.push_back(cur_sig);
        }
        cur_sig.fill(EMPTY);
        cur_len = 0;
        state.reset();
    };

    size_t i = 0;
    while (i < size) {
        const char c = data[i];
        if (c == '>') {
            finalize_contig();
            in_header = true;
            const char* nl = static_cast<const char*>(std::memchr(data + i, '\n', size - i));
            if (nl) { i = static_cast<size_t>(nl - data) + 1; in_header = false; }
            else break;
            continue;
        }
        if (in_header) { if (c == '\n') in_header = false; ++i; continue; }
        if (c == '\n' || c == '\r') { ++i; continue; }

        const uint8_t base = encode_base(c);
        if (base == 255) { state.reset(); ++i; continue; }

        state.fwd = ((state.fwd << 2) | base) & k_mask;
        state.rev = (state.rev >> 2) | (static_cast<uint64_t>(3 ^ base) << rev_shift);
        if (++state.valid >= k) {
            const uint64_t canonical = canonical_rolling_hash(state);
            const uint64_t h         = oph_hash_wymix(canonical, seed);
            const uint32_t bin       = fast_range_u64(h, static_cast<uint32_t>(CBINS));
            const uint32_t sig       = oph_sig32(h);
            if (sig < cur_sig[bin]) cur_sig[bin] = sig;
        }
        ++cur_len;
        ++i;
    }
    finalize_contig();

    if (contig_sigs.size() < 2) return 0.0f;

    // max_i mean_j d(i,j): for each contig, compute its mean OPH Jaccard distance to all
    // other contigs, then return the maximum. This isolates outlier contigs from a foreign
    // organism without being diluted by the O(n²) clean-clean pairs.
    // A chimeric contig has mean dist ≈ 1.0 to all clean contigs → drives score high.
    // A clean genome has low max (all contigs similar) ≈ 0.30-0.35 within-species.
    const size_t n = contig_sigs.size();
    double max_mean_dist = 0.0;
    for (size_t a = 0; a < n; ++a) {
        double sum = 0.0;
        for (size_t b = 0; b < n; ++b) {
            if (a == b) continue;
            size_t matches = 0;
            for (int t = 0; t < CBINS; ++t)
                if (contig_sigs[a][t] == contig_sigs[b][t]) ++matches;
            sum += 1.0 - static_cast<double>(matches) / CBINS;
        }
        const double mean_dist = sum / static_cast<double>(n - 1);
        if (mean_dist > max_mean_dist) max_mean_dist = mean_dist;
    }
    return static_cast<float>(max_mean_dist);
}

inline void finalize_oph_sketch(OPHSketch& result, int m) {
    const uint32_t EMPTY = std::numeric_limits<uint32_t>::max();

    if (!result.real_bins_bitmask.empty()) {
        for (auto w : result.real_bins_bitmask)
            result.n_real_bins += static_cast<size_t>(__builtin_popcountll(w));
    } else {
        for (int t = 0; t < m; ++t)
            if (result.signature[t] != EMPTY) ++result.n_real_bins;
    }

    auto mix = [](uint64_t x) -> uint32_t {
        x ^= x >> 30;
        x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27;
        x *= 0x94d049bb133111ebULL;
        x ^= x >> 31;
        return static_cast<uint32_t>(x >> 32);
    };
    for (int t = 1; t < m; ++t)
        if (result.signature[t] == EMPTY && result.signature[t - 1] != EMPTY)
            result.signature[t] = mix(
                static_cast<uint64_t>(result.signature[t - 1]) ^ static_cast<uint64_t>(t));
    for (int t = m - 2; t >= 0; --t)
        if (result.signature[t] == EMPTY && result.signature[t + 1] != EMPTY)
            result.signature[t] = mix(
                static_cast<uint64_t>(result.signature[t + 1]) ^ static_cast<uint64_t>(t));
}

template <bool TrackRealBins, uint32_t FixedBins>
[[gnu::always_inline]] inline void update_oph_bin(
        uint32_t* __restrict signature,
        uint64_t* __restrict real_bins,
        uint32_t runtime_bins,
        uint64_t canonical_hash) {
    const uint32_t bin = [&]() -> uint32_t {
        if constexpr (FixedBins != 0) return fast_range_u64<FixedBins>(canonical_hash);
        return fast_range_u64(canonical_hash, runtime_bins);
    }();

    const uint32_t h = oph_sig32(canonical_hash);
    uint32_t& cur = signature[bin];
    if (__builtin_expect(h < cur, 0)) {
        cur = h;
        if constexpr (TrackRealBins) real_bins[bin >> 6] |= (1ULL << (bin & 63));
    }
}

template <bool TrackRealBins, uint32_t FixedBins>
[[gnu::always_inline]] inline size_t process_valid_run(
        const char* data,
        size_t i,
        size_t len,
        int k,
        uint64_t seed,
        uint64_t k_mask,
        int rev_shift,
        uint32_t runtime_bins,
        OPHKmerState& state,
        uint64_t /*run_start_pos*/,
        uint32_t* __restrict signature,
        uint64_t* __restrict real_bins) {
    // Extract to locals so GCC can register-allocate fwd/rev/valid
    // (passing OPHKmerState& by reference prevents register allocation).
    uint64_t fwd   = state.fwd;
    uint64_t rev   = state.rev;
    int      valid = state.valid;
    size_t j = i;

    for (; j < len && valid < k; ++j) {
        const char c = data[j];
        if (c == '>' || c == '\n' || c == '\r') break;

        const uint8_t base = static_cast<uint8_t>(encode_base(c));
        if (base == 255) break;

        fwd = ((fwd << 2) | base) & k_mask;
        rev = (rev >> 2) | ((3ULL - base) << rev_shift);
        ++valid;

        if (valid >= k) {
            const uint64_t canonical = fwd ^ ((fwd ^ rev) & -(uint64_t)(fwd > rev));
            const uint64_t h = oph_hash_wymix(canonical, seed);
            update_oph_bin<TrackRealBins, FixedBins>(signature, real_bins, runtime_bins, h);
        }
    }

    for (; j < len; ++j) {
        const char c = data[j];
        if (c == '>' || c == '\n' || c == '\r') break;

        const uint8_t base = static_cast<uint8_t>(encode_base(c));
        if (base == 255) break;

        fwd = ((fwd << 2) | base) & k_mask;
        rev = (rev >> 2) | ((3ULL - base) << rev_shift);

        const uint64_t canonical = fwd ^ ((fwd ^ rev) & -(uint64_t)(fwd > rev));
        const uint64_t h = oph_hash_wymix(canonical, seed);
        update_oph_bin<TrackRealBins, FixedBins>(signature, real_bins, runtime_bins, h);
    }

    state.fwd   = fwd;
    state.rev   = rev;
    state.valid = valid;
    return j;
}

// Scan forward from data[0] to find the length of a valid ACGT run.
// Returns number of consecutive bytes where BASE_ENCODE[byte] != 255.
// Stops at '>', '\n', '\r', 'N', or any non-ACGT character.
// Uses AVX2 to test 32 bytes at a time when available.
[[gnu::always_inline]] static inline size_t scan_valid_run(const char* data, size_t len) noexcept {
#ifdef __AVX2__
    if (__builtin_expect(len >= 32, 1)) {
        const __m256i v_mask = _mm256_set1_epi8(static_cast<char>(0xDF));  // clear lowercase bit
        const __m256i v_A    = _mm256_set1_epi8('A');
        const __m256i v_C    = _mm256_set1_epi8('C');
        const __m256i v_G    = _mm256_set1_epi8('G');
        const __m256i v_T    = _mm256_set1_epi8('T');
        size_t j = 0;
        for (; j + 32 <= len; j += 32) {
            __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + j));
            __m256i upper = _mm256_and_si256(chunk, v_mask);
            __m256i ok    = _mm256_or_si256(
                _mm256_or_si256(_mm256_cmpeq_epi8(upper, v_A), _mm256_cmpeq_epi8(upper, v_C)),
                _mm256_or_si256(_mm256_cmpeq_epi8(upper, v_G), _mm256_cmpeq_epi8(upper, v_T)));
            uint32_t bad = static_cast<uint32_t>(~_mm256_movemask_epi8(ok));
            if (bad != 0)
                return j + static_cast<size_t>(__builtin_ctz(bad));
        }
        for (; j < len; ++j)
            if (BASE_ENCODE[static_cast<uint8_t>(data[j])] == 255) return j;
        return len;
    }
#endif
    size_t j = 0;
    while (j < len && BASE_ENCODE[static_cast<uint8_t>(data[j])] != 255) ++j;
    return j;
}

// Like process_valid_run but operates on a pre-validated run [i, run_end).
// Caller guarantees data[i..run_end) are all valid ACGT bases (no break conditions needed).
// Eliminates per-character branch checks, allowing the compiler to unroll more aggressively.
template <bool TrackRealBins, uint32_t FixedBins>
[[gnu::always_inline]] inline void process_valid_run_bounded(
        const char* data,
        size_t i,
        size_t run_end,
        int k,
        uint64_t seed,
        uint64_t k_mask,
        int rev_shift,
        uint32_t runtime_bins,
        OPHKmerState& state,
        uint64_t /*run_start_pos*/,
        uint32_t* __restrict signature,
        uint64_t* __restrict real_bins) {
    uint64_t fwd   = state.fwd;
    uint64_t rev   = state.rev;
    int      valid = state.valid;
    size_t   j     = i;

    // Warmup: fill k-mer window (valid < k)
    for (; j < run_end && valid < k; ++j) {
        const uint8_t base = BASE_ENCODE[static_cast<uint8_t>(data[j])];
        fwd = ((fwd << 2) | base) & k_mask;
        rev = (rev >> 2) | ((3ULL - base) << rev_shift);
        ++valid;
        if (valid >= k) {
            const uint64_t canonical = fwd ^ ((fwd ^ rev) & -(uint64_t)(fwd > rev));
            const uint64_t h = oph_hash_wymix(canonical, seed);
            update_oph_bin<TrackRealBins, FixedBins>(signature, real_bins, runtime_bins, h);
        }
    }

    // Main: emit a k-mer for every base (no break conditions)
    for (; j < run_end; ++j) {
        const uint8_t base = BASE_ENCODE[static_cast<uint8_t>(data[j])];
        fwd = ((fwd << 2) | base) & k_mask;
        rev = (rev >> 2) | ((3ULL - base) << rev_shift);
        const uint64_t canonical = fwd ^ ((fwd ^ rev) & -(uint64_t)(fwd > rev));
        const uint64_t h = oph_hash_wymix(canonical, seed);
        update_oph_bin<TrackRealBins, FixedBins>(signature, real_bins, runtime_bins, h);
    }

    state.fwd   = fwd;
    state.rev   = rev;
    state.valid = valid;
}

template <bool TrackRealBins, uint32_t FixedBins, bool ComputeChimera = false>
OPHSketch sketch_oph_impl(const std::filesystem::path& fasta_path,
                          int m,
                          const MinHasher::Config& cfg) {
    const uint32_t EMPTY = std::numeric_limits<uint32_t>::max();

    OPHSketch result;
    result.genome_length = 0;
    result.signature.assign(m, EMPTY);
    if constexpr (TrackRealBins) result.real_bins_bitmask.assign((m + 63) / 64, 0ULL);

    const int k = cfg.kmer_size;
    if (k <= 0 || k > 32) return result;

    const uint64_t seed = cfg.seed;
    const uint64_t k_mask = (k == 32) ? UINT64_MAX : ((1ULL << (2 * k)) - 1);
    const int rev_shift = 2 * (k - 1);
    const uint32_t runtime_bins = static_cast<uint32_t>(m);

    OPHKmerState state;
    bool in_header = false;
    uint64_t genome_pos = 0;  // cumulative position across all contigs

    uint32_t* __restrict signature = result.signature.data();
    uint64_t* __restrict real_bins = TrackRealBins ? result.real_bins_bitmask.data() : nullptr;

    auto process_bases = [&](const char* data, size_t len) {
        size_t i = 0;
        while (i < len) {
            const char c = data[i];

            if (c == '>') {
                ++result.n_contigs;
                state.reset();
                in_header = true;
                const char* nl = static_cast<const char*>(std::memchr(data + i, '\n', len - i));
                if (nl) {
                    i = static_cast<size_t>(nl - data) + 1;
                    in_header = false;
                } else {
                    return; // header spans into next buffer
                }
                continue;
            }
            if (in_header) {
                if (c == '\n') in_header = false;
                ++i;
                continue;
            }
            if (c == '\n' || c == '\r') {
                ++i;
                continue;
            }

            if (encode_base(c) == 255) {
                state.reset();
                ++i;
                continue;
            }

            const size_t run_start = i;
            const uint64_t run_start_pos = genome_pos;
            const size_t run_end = i + scan_valid_run(data + i, len - i);
            process_valid_run_bounded<TrackRealBins, FixedBins>(
                data, i, run_end, k, seed, k_mask, rev_shift, runtime_bins,
                state, run_start_pos, signature, real_bins);
            i = run_end;

            const uint64_t consumed = static_cast<uint64_t>(i - run_start);
            genome_pos += consumed;
            result.genome_length += consumed;

            if (i < len) {
                const char stop = data[i];
                if (stop != '\n' && stop != '\r' && stop != '>') {
                    state.reset();
                    ++i;
                }
            }
        }
    };

    std::string path_str = fasta_path.string();
    const bool is_gzipped = path_str.ends_with(".gz");
    const bool is_zstd    = path_str.ends_with(".zst");

    if (is_gzipped || is_zstd) {
        auto buf = GzReader::decompress_file(path_str);
        if (!buf.empty()) {
            process_bases(buf.data(), buf.size());
            if constexpr (ComputeChimera)
                result.chimera_score = compute_contig_chimera_score(buf.data(), buf.size(), cfg);
        }
    } else {
        if constexpr (ComputeChimera) {
            // Slurp full file so chimera scorer sees all contig boundaries at once
            std::ifstream in(fasta_path, std::ios::binary);
            if (!in) return result;
            in.seekg(0, std::ios::end);
            std::vector<char> buf(static_cast<size_t>(in.tellg()));
            in.seekg(0);
            in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
            if (!buf.empty()) {
                process_bases(buf.data(), buf.size());
                result.chimera_score = compute_contig_chimera_score(buf.data(), buf.size(), cfg);
            }
        } else {
            constexpr size_t BUF_SIZE = 1 << 22;
            std::vector<char> buf(BUF_SIZE);
            std::ifstream in(fasta_path, std::ios::binary);
            if (!in) return result;
            while (in) {
                in.read(buf.data(), BUF_SIZE);
                std::streamsize bytes_read = in.gcount();
                if (bytes_read <= 0) break;
                process_bases(buf.data(), static_cast<size_t>(bytes_read));
            }
        }
    }

    finalize_oph_sketch(result, m);

    return result;
}

// Like sketch_oph_impl but operates on a pre-decompressed FASTA buffer.
// Avoids all file I/O; used by the producer-consumer path in build_index().
template <bool TrackRealBins, uint32_t FixedBins>
OPHSketch sketch_oph_impl_from_buffer(const char* data, size_t size,
                                       int m, const MinHasher::Config& cfg) {
    const uint32_t EMPTY = std::numeric_limits<uint32_t>::max();

    OPHSketch result;
    result.genome_length = 0;
    result.signature.assign(m, EMPTY);
    if constexpr (TrackRealBins) result.real_bins_bitmask.assign((m + 63) / 64, 0ULL);

    const int k = cfg.kmer_size;
    if (k <= 0 || k > 32) return result;

    const uint64_t seed = cfg.seed;
    const uint64_t k_mask = (k == 32) ? UINT64_MAX : ((1ULL << (2 * k)) - 1);
    const int rev_shift = 2 * (k - 1);
    const uint32_t runtime_bins = static_cast<uint32_t>(m);

    OPHKmerState state;
    bool in_header = false;
    uint64_t genome_pos = 0;

    uint32_t* __restrict signature = result.signature.data();
    uint64_t* __restrict real_bins = TrackRealBins ? result.real_bins_bitmask.data() : nullptr;

    auto process_bases = [&](const char* d, size_t len) {
        size_t i = 0;
        while (i < len) {
            const char c = d[i];

            if (c == '>') {
                ++result.n_contigs;
                state.reset();
                in_header = true;
                const char* nl = static_cast<const char*>(std::memchr(d + i, '\n', len - i));
                if (nl) {
                    i = static_cast<size_t>(nl - d) + 1;
                    in_header = false;
                } else {
                    return;
                }
                continue;
            }
            if (in_header) {
                if (c == '\n') in_header = false;
                ++i;
                continue;
            }
            if (c == '\n' || c == '\r') {
                ++i;
                continue;
            }

            if (encode_base(c) == 255) {
                state.reset();
                ++i;
                continue;
            }

            const size_t run_start = i;
            const uint64_t run_start_pos = genome_pos;
            const size_t run_end = i + scan_valid_run(d + i, len - i);
            process_valid_run_bounded<TrackRealBins, FixedBins>(
                d, i, run_end, k, seed, k_mask, rev_shift, runtime_bins,
                state, run_start_pos, signature, real_bins);
            i = run_end;

            const uint64_t consumed = static_cast<uint64_t>(i - run_start);
            genome_pos += consumed;
            result.genome_length += consumed;

            if (i < len) {
                const char stop = d[i];
                if (stop != '\n' && stop != '\r' && stop != '>') {
                    state.reset();
                    ++i;
                }
            }
        }
    };

    process_bases(data, size);
    finalize_oph_sketch(result, m);
    return result;
}

template <bool TrackRealBins, uint32_t FixedBins>
OPHSketch sketch_oph_syncmer_impl(const std::filesystem::path& fasta_path,
                                  int m,
                                  int s,
                                  const MinHasher::Config& cfg) {
    const uint32_t EMPTY = std::numeric_limits<uint32_t>::max();
    constexpr size_t QUEUE_CAPACITY = 64;

    OPHSketch result;
    result.genome_length = 0;
    result.signature.assign(m, EMPTY);
    if constexpr (TrackRealBins) result.real_bins_bitmask.assign((m + 63) / 64, 0ULL);

    const int k = cfg.kmer_size;
    if (k <= 0 || k > 32 || s <= 0 || s >= k) return result;

    const uint64_t seed = cfg.seed;
    const uint64_t k_mask = (k == 32) ? UINT64_MAX : ((1ULL << (2 * k)) - 1);
    const uint64_t s_mask = (s == 32) ? UINT64_MAX : ((1ULL << (2 * s)) - 1);
    const int k_rev_shift = 2 * (k - 1);
    const int s_rev_shift = 2 * (s - 1);
    const uint32_t runtime_bins = static_cast<uint32_t>(m);
    const uint32_t window_span = static_cast<uint32_t>(k - s + 1);

    OPHKmerState k_state;
    OPHKmerState s_state;
    SyncmerMinQueue<QUEUE_CAPACITY> min_queue;
    bool in_header = false;
    uint64_t genome_pos = 0;

    uint32_t* __restrict signature = result.signature.data();
    uint64_t* __restrict real_bins = TrackRealBins ? result.real_bins_bitmask.data() : nullptr;

    auto reset_states = [&]() {
        k_state.reset();
        s_state.reset();
        min_queue.reset();
    };

    auto process_bases = [&](const char* data, size_t len) {
        for (size_t i = 0; i < len; ++i) {
            const char c = data[i];

            if (c == '>') {
                ++result.n_contigs;
                reset_states();
                in_header = true;
                const char* nl = static_cast<const char*>(std::memchr(data + i, '\n', len - i));
                if (nl) {
                    i = static_cast<size_t>(nl - data);
                    in_header = false;
                } else {
                    return;
                }
                continue;
            }
            if (in_header) {
                if (c == '\n') in_header = false;
                continue;
            }
            if (c == '\n' || c == '\r') continue;

            const uint8_t base = static_cast<uint8_t>(encode_base(c));
            if (base == 255) {
                reset_states();
                continue;
            }

            ++result.genome_length;
            ++genome_pos;

            k_state.fwd = ((k_state.fwd << 2) | base) & k_mask;
            k_state.rev = (k_state.rev >> 2) | ((3ULL - base) << k_rev_shift);
            if (k_state.valid < k) ++k_state.valid;

            s_state.fwd = ((s_state.fwd << 2) | base) & s_mask;
            s_state.rev = (s_state.rev >> 2) | ((3ULL - base) << s_rev_shift);
            if (s_state.valid < s) ++s_state.valid;

            if (s_state.valid >= s) {
                const uint32_t s_pos = static_cast<uint32_t>(genome_pos - static_cast<uint64_t>(s));
                const uint64_t s_hash = canonical_rolling_hash(s_state) ^ seed;

                while (!min_queue.empty() && min_queue.back().hash >= s_hash)
                    min_queue.pop_back();
                min_queue.push_back({s_hash, s_pos});

                const uint32_t min_valid_pos =
                    (s_pos >= window_span - 1) ? (s_pos - (window_span - 1)) : 0;
                while (!min_queue.empty() && min_queue.front().pos < min_valid_pos)
                    min_queue.pop_front();

                if (k_state.valid >= k && !min_queue.empty()) {
                    const uint32_t k_pos = static_cast<uint32_t>(genome_pos - static_cast<uint64_t>(k));
                    if (min_queue.front().pos == k_pos) {
                        const uint64_t canonical = canonical_rolling_hash(k_state);
                        const uint64_t h = oph_hash_wymix(canonical, seed);
                        update_oph_bin<TrackRealBins, FixedBins>(
                            signature, real_bins, runtime_bins, h);
                    }
                }
            }
        }
    };

    std::string path_str = fasta_path.string();
    const bool is_gzipped = path_str.ends_with(".gz");
    const bool is_zstd    = path_str.ends_with(".zst");

    if (is_gzipped || is_zstd) {
        auto buf = GzReader::decompress_file(path_str);
        if (!buf.empty())
            process_bases(buf.data(), buf.size());
    } else {
        constexpr size_t BUF_SIZE = 1 << 22;
        std::vector<char> buf(BUF_SIZE);
        std::ifstream in(fasta_path, std::ios::binary);
        if (!in) return result;
        while (in) {
            in.read(buf.data(), BUF_SIZE);
            std::streamsize bytes_read = in.gcount();
            if (bytes_read <= 0) break;
            process_bases(buf.data(), static_cast<size_t>(bytes_read));
        }
    }

    finalize_oph_sketch(result, m);

    return result;
}

} // anonymous namespace

uint64_t canonical_kmer_hash(const char* seq, int k, uint64_t seed) {
    uint64_t fwd = 0;
    uint64_t rev = 0;
    uint64_t rev_shift = 2 * (k - 1);

    for (int i = 0; i < k; ++i) {
        uint64_t base = encode_base(seq[i]);
        if (base == 255) return UINT64_MAX;

        fwd = (fwd << 2) | base;
        rev = rev | ((3ULL - base) << (rev_shift - 2 * i));
    }

    uint64_t canonical = fwd ^ ((fwd ^ rev) & -(uint64_t)(fwd > rev));
    return murmur64(canonical ^ seed);
}

double MinHashSketch::jaccard(const MinHashSketch& other) const {
    if (hashes.empty() || other.hashes.empty()) return 0.0;

    size_t shared = 0;
    size_t i = 0, j = 0;
    size_t limit = std::min(hashes.size(), other.hashes.size());

    while (i < hashes.size() && j < other.hashes.size()) {
        if (hashes[i] == other.hashes[j]) {
            ++shared;
            ++i;
            ++j;
        } else if (hashes[i] < other.hashes[j]) {
            ++i;
        } else {
            ++j;
        }
    }

    return static_cast<double>(shared) / static_cast<double>(limit);
}

double MinHashSketch::mash_distance(const MinHashSketch& other, int k) const {
    double j = jaccard(other);
    if (j <= 0.0) return 1.0;
    if (j >= 1.0) return 0.0;

    double d = -1.0 / k * std::log(2.0 * j / (1.0 + j));
    return std::clamp(d, 0.0, 1.0);
}

double MinHashSketch::estimate_ani(const MinHashSketch& other, int k) const {
    double d = mash_distance(other, k);
    return std::clamp(1.0 - d, 0.0, 1.0);
}

MinHasher::MinHasher() : cfg_{} {}
MinHasher::MinHasher(Config cfg) : cfg_(std::move(cfg)) {}

void MinHasher::add_kmers_from_sequence(const std::string& seq,
                                         std::vector<uint64_t>& hashes) const {
    const int k = cfg_.kmer_size;
    if (static_cast<int>(seq.size()) < k) return;

    const size_t n = seq.size();
    const char* data = seq.data();
    const uint64_t k_mask = (1ULL << (2 * k)) - 1;
    const int sketch_size = cfg_.sketch_size;
    const uint64_t seed = cfg_.seed;

    // Rolling hash state
    uint64_t fwd = 0;
    uint64_t rev = 0;
    int valid_bases = 0;
    const int rev_shift = 2 * (k - 1);

    // Threshold-based filtering (avoid heap entirely)
    uint64_t threshold = UINT64_MAX;
    size_t buffer_limit = static_cast<size_t>(sketch_size) * 2;

    for (size_t i = 0; i < n; ++i) {
        uint64_t base = encode_base(data[i]);

        if (base == 255) {
            valid_bases = 0;
            fwd = 0;
            rev = 0;
            continue;
        }

        fwd = ((fwd << 2) | base) & k_mask;
        rev = ((rev >> 2) | ((3ULL - base) << rev_shift));
        ++valid_bases;

        if (valid_bases >= k) {
            uint64_t canonical = fwd ^ ((fwd ^ rev) & -(uint64_t)(fwd > rev));
            uint64_t h = murmur64(canonical ^ seed);

            // Fast path: skip if above threshold
            if (h >= threshold) continue;

            hashes.push_back(h);

            // Compact when buffer fills
            if (hashes.size() >= buffer_limit) {
                auto mid = hashes.begin() + sketch_size;
                std::nth_element(hashes.begin(), mid, hashes.end());
                threshold = *mid;
                hashes.resize(sketch_size);
            }
        }
    }
}

MinHashSketch MinHasher::sketch_sequence(const std::string& sequence) const {
    MinHashSketch result;
    result.genome_length = sequence.size();

    std::vector<uint64_t> hashes;
    hashes.reserve(static_cast<size_t>(cfg_.sketch_size) * 2);

    add_kmers_from_sequence(sequence, hashes);

    // Final compaction and sort
    if (static_cast<int>(hashes.size()) > cfg_.sketch_size) {
        auto mid = hashes.begin() + cfg_.sketch_size;
        std::nth_element(hashes.begin(), mid, hashes.end());
        hashes.resize(cfg_.sketch_size);
    }
    std::sort(hashes.begin(), hashes.end());
    hashes.erase(std::unique(hashes.begin(), hashes.end()), hashes.end());

    result.hashes = std::move(hashes);
    return result;
}

MinHashSketch MinHasher::sketch(const std::filesystem::path& fasta_path) const {
    struct KmerState {
        uint64_t fwd = 0, rev = 0;
        int valid = 0;
        void reset() { fwd = rev = 0; valid = 0; }
    };

    MinHashSketch result;
    result.genome_length = 0;

    const int k = cfg_.kmer_size;
    const int sketch_size = cfg_.sketch_size;
    const uint64_t seed = cfg_.seed;
    const uint64_t k_mask = (1ULL << (2 * k)) - 1;
    const int rev_shift = 2 * (k - 1);

    std::vector<uint64_t> hashes;
    hashes.reserve(static_cast<size_t>(sketch_size) * 2);

    uint64_t threshold = UINT64_MAX;
    size_t buffer_limit = static_cast<size_t>(sketch_size) * 2;

    KmerState state;
    bool in_header = false;

    auto process_bases = [&](const char* data, size_t len) {
        for (size_t i = 0; i < len; ++i) {
            char c = data[i];

            if (c == '>') {
                state.reset();
                in_header = true;
                const char* nl = static_cast<const char*>(
                    std::memchr(data + i, '\n', len - i));
                if (nl) {
                    i = static_cast<size_t>(nl - data);
                    in_header = false;
                } else {
                    return; // header spans into next buffer
                }
                continue;
            }

            if (in_header) {
                if (c == '\n') {
                    in_header = false;
                }
                continue;
            }

            if (c == '\n' || c == '\r') continue;

            uint64_t base = encode_base(c);
            if (base == 255) {
                state.reset();
                continue;
            }

            ++result.genome_length;
            state.fwd = ((state.fwd << 2) | base) & k_mask;
            state.rev = ((state.rev >> 2) | ((3ULL - base) << rev_shift));
            ++state.valid;

            if (state.valid >= k) {
                uint64_t canonical = state.fwd ^ ((state.fwd ^ state.rev) & -(uint64_t)(state.fwd > state.rev));
                uint64_t h = murmur64(canonical ^ seed);

                if (h >= threshold) continue;

                hashes.push_back(h);

                if (hashes.size() >= buffer_limit) {
                    auto mid = hashes.begin() + sketch_size;
                    std::nth_element(hashes.begin(), mid, hashes.end());
                    threshold = *mid;
                    hashes.resize(sketch_size);
                }
            }
        }
    };

    std::string path_str = fasta_path.string();
    const bool is_gzipped = path_str.ends_with(".gz");
    const bool is_zstd    = path_str.ends_with(".zst");

    if (is_gzipped || is_zstd) {
        auto buf = GzReader::decompress_file(path_str);
        if (!buf.empty())
            process_bases(buf.data(), buf.size());
    } else {
        constexpr size_t BUF_SIZE = 1 << 22;
        std::vector<char> buf(BUF_SIZE);
        std::ifstream in(fasta_path, std::ios::binary);
        if (!in) return result;
        while (in) {
            in.read(buf.data(), BUF_SIZE);
            std::streamsize bytes_read = in.gcount();
            if (bytes_read <= 0) break;
            process_bases(buf.data(), static_cast<size_t>(bytes_read));
        }
    }

    // Final compaction and sort
    if (static_cast<int>(hashes.size()) > sketch_size) {
        auto mid = hashes.begin() + sketch_size;
        std::nth_element(hashes.begin(), mid, hashes.end());
        hashes.resize(sketch_size);
    }
    std::sort(hashes.begin(), hashes.end());
    hashes.erase(std::unique(hashes.begin(), hashes.end()), hashes.end());

    result.hashes = std::move(hashes);
    return result;
}

OPHSketch MinHasher::sketch_oph(const std::filesystem::path& fasta_path, int m) const {
    if (cfg_.syncmer_s > 0)
        return sketch_oph_syncmer(fasta_path, m, cfg_.syncmer_s);
    if (m == OPH_BINS)
        return sketch_oph_impl<false, OPH_BINS>(fasta_path, m, cfg_);
    return sketch_oph_impl<false, 0>(fasta_path, m, cfg_);
}

OPHSketch MinHasher::sketch_oph_syncmer(
        const std::filesystem::path& fasta_path, int m, int s) const {
    if (s <= 0 || s >= cfg_.kmer_size) {
        if (m == OPH_BINS)
            return sketch_oph_impl<false, OPH_BINS>(fasta_path, m, cfg_);
        return sketch_oph_impl<false, 0>(fasta_path, m, cfg_);
    }

    if (m == OPH_BINS)
        return sketch_oph_syncmer_impl<false, OPH_BINS>(fasta_path, m, s, cfg_);
    return sketch_oph_syncmer_impl<false, 0>(fasta_path, m, s, cfg_);
}

OPHSketch MinHasher::sketch_oph_with_positions(
        const std::filesystem::path& fasta_path, int m) const {
    if (cfg_.syncmer_s > 0) {
        if (m == OPH_BINS)
            return sketch_oph_syncmer_impl<true, OPH_BINS>(fasta_path, m, cfg_.syncmer_s, cfg_);
        return sketch_oph_syncmer_impl<true, 0>(fasta_path, m, cfg_.syncmer_s, cfg_);
    }
    if (m == OPH_BINS)
        return sketch_oph_impl<true, OPH_BINS, false>(fasta_path, m, cfg_);
    return sketch_oph_impl<true, 0, false>(fasta_path, m, cfg_);
}

OPHSketch MinHasher::sketch_oph_with_positions_from_buffer(
        const char* data, size_t len, int m) const {
    if (m == OPH_BINS)
        return sketch_oph_impl_from_buffer<true, OPH_BINS>(data, len, m, cfg_);
    return sketch_oph_impl_from_buffer<true, 0>(data, len, m, cfg_);
}

} // namespace derep
