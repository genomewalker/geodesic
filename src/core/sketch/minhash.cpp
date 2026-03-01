#include "core/sketch/minhash.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
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
    bool is_gzipped = path_str.ends_with(".gz");

    constexpr size_t BUF_SIZE = 1 << 18;
    std::vector<char> buf(BUF_SIZE);

    if (is_gzipped) {
        try {
            GzReader gz(path_str);
            while (true) {
                size_t bytes_read = gz.read(buf.data(), BUF_SIZE);
                if (bytes_read == 0) break;
                process_bases(buf.data(), bytes_read);
            }
        } catch (const std::exception&) { return result; }
    } else {
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
    struct KmerState {
        uint64_t fwd = 0, rev = 0;
        int valid = 0;
        void reset() { fwd = rev = 0; valid = 0; }
    };

    OPHSketch result;
    result.genome_length = 0;
    result.signature.assign(m, std::numeric_limits<uint64_t>::max());

    const int k = cfg_.kmer_size;
    const uint64_t seed = cfg_.seed;
    const uint64_t k_mask = (1ULL << (2 * k)) - 1;
    const int rev_shift = 2 * (k - 1);
    const uint64_t m64 = static_cast<uint64_t>(m);

    KmerState state;
    bool in_header = false;

    auto process_bases = [&](const char* data, size_t len) {
        for (size_t i = 0; i < len; ++i) {
            char c = data[i];

            if (c == '>') {
                state.reset();
                in_header = true;
                const char* nl = static_cast<const char*>(std::memchr(data + i, '\n', len - i));
                if (nl) { i = static_cast<size_t>(nl - data); in_header = false; }
                else return;
                continue;
            }
            if (in_header) { if (c == '\n') in_header = false; continue; }
            if (c == '\n' || c == '\r') continue;

            uint64_t base = encode_base(c);
            if (base == 255) { state.reset(); continue; }

            ++result.genome_length;
            state.fwd = ((state.fwd << 2) | base) & k_mask;
            state.rev = ((state.rev >> 2) | ((3ULL - base) << rev_shift));
            ++state.valid;

            if (state.valid >= k) {
                uint64_t canonical = state.fwd ^ ((state.fwd ^ state.rev) & -(uint64_t)(state.fwd > state.rev));
                uint64_t h = murmur64(canonical ^ seed);
                int bin = static_cast<int>(h % m64);
                if (h < result.signature[bin])
                    result.signature[bin] = h;
            }
        }
    };

    std::string path_str = fasta_path.string();
    bool is_gzipped = path_str.ends_with(".gz");
    constexpr size_t BUF_SIZE = 1 << 18;
    std::vector<char> buf(BUF_SIZE);

    if (is_gzipped) {
        try {
            GzReader gz(path_str);
            while (true) {
                size_t bytes_read = gz.read(buf.data(), BUF_SIZE);
                if (bytes_read == 0) break;
                process_bases(buf.data(), bytes_read);
            }
        } catch (const std::exception&) { return result; }
    } else {
        std::ifstream in(fasta_path, std::ios::binary);
        if (!in) return result;
        while (in) {
            in.read(buf.data(), BUF_SIZE);
            std::streamsize bytes_read = in.gcount();
            if (bytes_read <= 0) break;
            process_bases(buf.data(), static_cast<size_t>(bytes_read));
        }
    }

    // Densify empty bins: forward pass then backward pass, re-mixing value with bin index
    // so each bin gets a distinct token even when borrowing from a neighbor.
    const uint64_t EMPTY = std::numeric_limits<uint64_t>::max();
    // splitmix64 finalizer for re-mixing
    auto mix = [](uint64_t x) -> uint64_t {
        x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27; x *= 0x94d049bb133111ebULL;
        x ^= x >> 31; return x;
    };
    for (int t = 1; t < m; ++t)
        if (result.signature[t] == EMPTY && result.signature[t - 1] != EMPTY)
            result.signature[t] = mix(result.signature[t - 1] ^ static_cast<uint64_t>(t));
    for (int t = m - 2; t >= 0; --t)
        if (result.signature[t] == EMPTY && result.signature[t + 1] != EMPTY)
            result.signature[t] = mix(result.signature[t + 1] ^ static_cast<uint64_t>(t));

    return result;
}

OPHSketch MinHasher::sketch_oph_with_positions(
        const std::filesystem::path& fasta_path, int m) const {
    struct KmerState {
        uint64_t fwd = 0, rev = 0;
        int valid = 0;
        void reset() { fwd = rev = 0; valid = 0; }
    };

    const uint64_t EMPTY = std::numeric_limits<uint64_t>::max();

    OPHSketch result;
    result.genome_length = 0;
    result.signature.assign(m, EMPTY);
    result.positions.assign(m, EMPTY);  // UINT64_MAX = no real k-mer (densified)

    const int k = cfg_.kmer_size;
    const uint64_t seed = cfg_.seed;
    const uint64_t k_mask = (1ULL << (2 * k)) - 1;
    const int rev_shift = 2 * (k - 1);
    const uint64_t m64 = static_cast<uint64_t>(m);

    KmerState state;
    bool in_header = false;
    uint64_t genome_pos = 0;  // cumulative position across all contigs

    auto process_bases = [&](const char* data, size_t len) {
        for (size_t i = 0; i < len; ++i) {
            char c = data[i];

            if (c == '>') {
                state.reset();
                in_header = true;
                const char* nl = static_cast<const char*>(std::memchr(data + i, '\n', len - i));
                if (nl) { i = static_cast<size_t>(nl - data); in_header = false; }
                else return;
                continue;
            }
            if (in_header) { if (c == '\n') in_header = false; continue; }
            if (c == '\n' || c == '\r') continue;

            uint64_t base = encode_base(c);
            if (base == 255) { state.reset(); continue; }

            ++result.genome_length;
            ++genome_pos;
            state.fwd = ((state.fwd << 2) | base) & k_mask;
            state.rev = ((state.rev >> 2) | ((3ULL - base) << rev_shift));
            ++state.valid;

            if (state.valid >= k) {
                uint64_t canonical = state.fwd ^ ((state.fwd ^ state.rev) & -(uint64_t)(state.fwd > state.rev));
                uint64_t h = murmur64(canonical ^ seed);
                int bin = static_cast<int>(h % m64);
                if (h < result.signature[bin]) {
                    result.signature[bin] = h;
                    result.positions[bin] = genome_pos - static_cast<uint64_t>(k);
                }
            }
        }
    };

    std::string path_str = fasta_path.string();
    bool is_gzipped = path_str.ends_with(".gz");
    constexpr size_t BUF_SIZE = 1 << 18;
    std::vector<char> buf(BUF_SIZE);

    if (is_gzipped) {
        try {
            GzReader gz(path_str);
            while (true) {
                size_t bytes_read = gz.read(buf.data(), BUF_SIZE);
                if (bytes_read == 0) break;
                process_bases(buf.data(), bytes_read);
            }
        } catch (const std::exception&) { return result; }
    } else {
        std::ifstream in(fasta_path, std::ios::binary);
        if (!in) return result;
        while (in) {
            in.read(buf.data(), BUF_SIZE);
            std::streamsize bytes_read = in.gcount();
            if (bytes_read <= 0) break;
            process_bases(buf.data(), static_cast<size_t>(bytes_read));
        }
    }

    // Densify empty bins — positions for densified bins stay UINT64_MAX
    auto mix = [](uint64_t x) -> uint64_t {
        x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27; x *= 0x94d049bb133111ebULL;
        x ^= x >> 31; return x;
    };
    for (int t = 1; t < m; ++t)
        if (result.signature[t] == EMPTY && result.signature[t - 1] != EMPTY)
            result.signature[t] = mix(result.signature[t - 1] ^ static_cast<uint64_t>(t));
    for (int t = m - 2; t >= 0; --t)
        if (result.signature[t] == EMPTY && result.signature[t + 1] != EMPTY)
            result.signature[t] = mix(result.signature[t + 1] ^ static_cast<uint64_t>(t));

    return result;
}

} // namespace derep
