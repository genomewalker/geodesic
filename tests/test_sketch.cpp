#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "core/geodesic/geodesic.hpp"
#include "core/sketch/minhash.hpp"

#include <array>
#include <fstream>
#include <limits>

using namespace derep;

namespace {

uint64_t test_encode_base(char c) {
    switch (c) {
        case 'A':
        case 'a':
            return 0;
        case 'C':
        case 'c':
            return 1;
        case 'G':
        case 'g':
            return 2;
        case 'T':
        case 't':
            return 3;
        default:
            return 255;
    }
}

uint64_t test_canonical_code(const std::string& seq, size_t start, int len) {
    uint64_t fwd = 0;
    uint64_t rev = 0;
    const int rev_shift = 2 * (len - 1);
    for (int i = 0; i < len; ++i) {
        const uint64_t base = test_encode_base(seq[start + static_cast<size_t>(i)]);
        fwd = (fwd << 2) | base;
        rev = (rev >> 2) | ((3ULL - base) << rev_shift);
    }
    return std::min(fwd, rev);
}

uint64_t test_wymix(uint64_t a, uint64_t b) {
    const __uint128_t product = static_cast<__uint128_t>(a) * b;
    return static_cast<uint64_t>(product) ^ static_cast<uint64_t>(product >> 64);
}

uint64_t test_oph_hash_wymix(uint64_t canonical, uint64_t seed) {
    constexpr uint64_t P0 = 0xa0761d6478bd642fULL;
    constexpr uint64_t P1 = 0xe7037ed1a0b428dbULL;
    return test_wymix(canonical ^ (seed + P0), canonical ^ P1);
}

uint32_t test_fast_range_u64(uint64_t h, uint32_t bins) {
    return static_cast<uint32_t>((static_cast<__uint128_t>(h) * bins) >> 64);
}

std::vector<uint32_t> manual_syncmer_starts(
        const std::string& seq,
        int k,
        int s,
        uint64_t seed,
        bool include_right_edge) {
    std::vector<uint32_t> starts;
    if (seq.size() < static_cast<size_t>(k)) return starts;

    const int window_span = k - s + 1;
    for (size_t k_pos = 0; k_pos + static_cast<size_t>(k) <= seq.size(); ++k_pos) {
        uint64_t min_hash = std::numeric_limits<uint64_t>::max();
        uint32_t min_pos = 0;
        for (int offset = 0; offset < window_span; ++offset) {
            const uint32_t s_pos = static_cast<uint32_t>(k_pos + static_cast<size_t>(offset));
            const uint64_t s_hash = test_canonical_code(seq, s_pos, s) ^ seed;
            if (s_hash <= min_hash) {
                min_hash = s_hash;
                min_pos = s_pos;
            }
        }

        const uint32_t start = static_cast<uint32_t>(k_pos);
        const uint32_t right = start + static_cast<uint32_t>(k - s);
        if (min_pos == start || (include_right_edge && min_pos == right))
            starts.push_back(start);
    }

    return starts;
}

OPHSketch manual_open_syncmer_oph(
        const std::string& seq,
        int k,
        int s,
        int m,
        uint64_t seed) {
    const uint32_t EMPTY = std::numeric_limits<uint32_t>::max();

    OPHSketch result;
    result.genome_length = seq.size();
    result.signature.assign(static_cast<size_t>(m), EMPTY);
    result.real_bins_bitmask.assign(static_cast<size_t>((m + 63) / 64), 0ULL);

    for (uint32_t k_pos : manual_syncmer_starts(seq, k, s, seed, false)) {
        const uint64_t canonical = test_canonical_code(seq, k_pos, k);
        const uint64_t hash = test_oph_hash_wymix(canonical, seed);
        const uint32_t bin = test_fast_range_u64(hash, static_cast<uint32_t>(m));
        const uint32_t sig = static_cast<uint32_t>(hash >> 32);
        if (sig < result.signature[bin]) {
            result.signature[bin] = sig;
            result.real_bins_bitmask[bin >> 6] |= (1ULL << (bin & 63));
        }
    }

    for (uint64_t w : result.real_bins_bitmask)
        result.n_real_bins += static_cast<size_t>(__builtin_popcountll(w));

    auto mix = [](uint64_t x) -> uint32_t {
        x ^= x >> 30;
        x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27;
        x *= 0x94d049bb133111ebULL;
        x ^= x >> 31;
        return static_cast<uint32_t>(x >> 32);
    };
    for (int t = 1; t < m; ++t) {
        if (result.signature[t] == EMPTY && result.signature[t - 1] != EMPTY)
            result.signature[t] = mix(
                static_cast<uint64_t>(result.signature[t - 1]) ^ static_cast<uint64_t>(t));
    }
    for (int t = m - 2; t >= 0; --t) {
        if (result.signature[t] == EMPTY && result.signature[t + 1] != EMPTY)
            result.signature[t] = mix(
                static_cast<uint64_t>(result.signature[t + 1]) ^ static_cast<uint64_t>(t));
    }

    return result;
}

std::string make_syncmer_regression_sequence(int k, int s, uint64_t seed) {
    for (uint32_t attempt = 1; attempt < 2048; ++attempt) {
        uint32_t state = attempt * 2654435761U;
        std::string seq(96, 'A');
        for (char& base : seq) {
            state = state * 1664525U + 1013904223U;
            base = "ACGT"[(state >> 30) & 3];
        }

        const auto open = manual_syncmer_starts(seq, k, s, seed, false);
        const auto closed = manual_syncmer_starts(seq, k, s, seed, true);
        if (open != closed) return seq;
    }

    return {};
}

} // namespace

TEST_CASE("MinHasher: identical sequences have Jaccard 1.0", "[minhash]") {
    MinHasher hasher({.kmer_size = 16, .sketch_size = 1000});

    std::string seq(10000, 'A');
    for (size_t i = 0; i < seq.size(); i += 4) {
        seq[i] = "ACGT"[i % 4];
        seq[i+1] = "CGTA"[(i/4) % 4];
        seq[i+2] = "GTAC"[(i/8) % 4];
        seq[i+3] = "TACG"[(i/16) % 4];
    }

    auto s1 = hasher.sketch_sequence(seq);
    auto s2 = hasher.sketch_sequence(seq);

    REQUIRE(s1.jaccard(s2) == 1.0);
}

TEST_CASE("MinHasher: different sequences have Jaccard < 1.0", "[minhash]") {
    MinHasher hasher({.kmer_size = 8, .sketch_size = 1000});

    std::string seq1, seq2;
    seq1.reserve(50000);
    seq2.reserve(50000);

    uint32_t seed1 = 111111;
    uint32_t seed2 = 999999;

    for (int i = 0; i < 50000; ++i) {
        seed1 = seed1 * 1103515245 + 12345;
        seed2 = seed2 * 1103515245 + 12345;
        seq1 += "ACGT"[(seed1 >> 16) % 4];
        seq2 += "ACGT"[(seed2 >> 16) % 4];
    }

    auto s1 = hasher.sketch_sequence(seq1);
    auto s2 = hasher.sketch_sequence(seq2);

    REQUIRE(s1.jaccard(s2) < 1.0);
}

TEST_CASE("MinHasher: estimate_ani correlates with sequence similarity", "[minhash]") {
    MinHasher hasher({.kmer_size = 16, .sketch_size = 1000});

    std::string seq1(50000, 'A');
    uint32_t seed = 12345;
    for (size_t i = 0; i < seq1.size(); ++i) {
        seed = seed * 1103515245 + 12345;
        seq1[i] = "ACGT"[(seed >> 16) % 4];
    }

    std::string seq2 = seq1;
    for (size_t i = 0; i < seq2.size(); i += 20) {
        seq2[i] = "TGCA"[seq2[i] % 4];
    }

    auto s1 = hasher.sketch_sequence(seq1);
    auto s2 = hasher.sketch_sequence(seq2);

    double ani = s1.estimate_ani(s2, 16);

    REQUIRE(ani > 0.85);
    REQUIRE(ani < 1.0);

    auto s3 = hasher.sketch_sequence(seq1);
    REQUIRE(s1.estimate_ani(s3, 16) == 1.0);
}

TEST_CASE("MinHasher: mash_distance inversely correlates with Jaccard", "[minhash]") {
    MinHasher hasher({.kmer_size = 8, .sketch_size = 1000});

    std::string seq1;
    seq1.reserve(50000);
    uint32_t seed = 12345;
    for (int i = 0; i < 50000; ++i) {
        seed = seed * 1103515245 + 12345;
        seq1 += "ACGT"[(seed >> 16) % 4];
    }

    std::string seq2 = seq1;
    for (size_t i = 0; i < seq2.size(); i += 10) {
        seq2[i] = "TGCA"[seq2[i] % 4];
    }

    auto s1 = hasher.sketch_sequence(seq1);
    auto s2 = hasher.sketch_sequence(seq2);

    double j = s1.jaccard(s2);
    double d = s1.mash_distance(s2, 8);

    REQUIRE(j > 0.0);
    REQUIRE(j < 1.0);
    REQUIRE(d > 0.0);
    REQUIRE(d < 1.0);
}

TEST_CASE("MinHasher: sketch_size limits output", "[minhash]") {
    MinHasher hasher({.kmer_size = 8, .sketch_size = 100});

    std::string seq(10000, 'A');
    for (size_t i = 0; i < seq.size(); i += 4) {
        seq[i] = "ACGT"[i % 4];
    }

    auto s = hasher.sketch_sequence(seq);

    REQUIRE(s.hashes.size() <= 100);
    REQUIRE(s.genome_length == 10000);
}

TEST_CASE("MinHasher: hashes are sorted ascending", "[minhash]") {
    MinHasher hasher({.kmer_size = 16, .sketch_size = 500});

    std::string seq(50000, 'A');
    for (size_t i = 0; i < seq.size(); ++i) {
        seq[i] = "ACGTACGTACGTACGT"[i % 16];
    }

    auto s = hasher.sketch_sequence(seq);

    REQUIRE(std::is_sorted(s.hashes.begin(), s.hashes.end()));
}

TEST_CASE("MinHasher: syncmer OPH produces real bins and positions", "[minhash][syncmer]") {
    const auto fasta_path = std::filesystem::temp_directory_path() / "geodesic_syncmer_test.fa";
    {
        std::ofstream out(fasta_path);
        REQUIRE(out.good());
        out << ">seq\n";
        uint32_t state = 123456789U;
        for (int i = 0; i < 4096; ++i) {
            state = state * 1664525U + 1013904223U;
            out << "ACGT"[(state >> 30) & 3];
        }
        out << '\n';
    }

    MinHasher hasher({.kmer_size = 21, .sketch_size = 256, .syncmer_s = 11});
    auto oph = hasher.sketch_oph_with_positions(fasta_path, 256);

    std::filesystem::remove(fasta_path);

    REQUIRE(oph.genome_length == 4096);
    REQUIRE(oph.n_real_bins > 0);
    REQUIRE(oph.signature.size() == 256);
    REQUIRE(oph.real_bins_bitmask.size() == (256 + 63) / 64);

    bool has_real_bin = false;
    for (uint64_t w : oph.real_bins_bitmask) {
        if (w != 0) {
            has_real_bin = true;
            break;
        }
    }
    REQUIRE(has_real_bin);
}

// ---------------------------------------------------------------------------
// Chimera score tests
// ---------------------------------------------------------------------------

namespace {

// Generate a sequence using only bases from a 4-char alphabet (biased composition).
// Using "AATT" (AT-only) vs "CCGG" (GC-only) creates disjoint k-mer universes at k=21,
// so cross-organism Jaccard ≈ 0 → pairwise distance ≈ 1.0.
std::string gen_biased_seq(uint32_t seed, size_t len, const char* alphabet) {
    std::string s;
    s.reserve(len);
    for (size_t i = 0; i < len; ++i) {
        seed = seed * 1664525U + 1013904223U;
        s += alphabet[(seed >> 30) & 3];
    }
    return s;
}

std::string make_fasta(const std::vector<std::pair<std::string, std::string>>& contigs) {
    std::string out;
    for (const auto& [name, seq] : contigs) {
        out += '>';
        out += name;
        out += '\n';
        out += seq;
        out += '\n';
    }
    return out;
}

} // namespace

TEST_CASE("MinHasher: syncmer OPH uses strict open-syncmer selection", "[minhash][syncmer]") {
    constexpr int k = 9;
    constexpr int s = 5;
    constexpr int m = 512;
    constexpr uint64_t seed = 42;

    const std::string seq = make_syncmer_regression_sequence(k, s, seed);
    REQUIRE_FALSE(seq.empty());
    REQUIRE(manual_syncmer_starts(seq, k, s, seed, false) !=
            manual_syncmer_starts(seq, k, s, seed, true));

    const auto fasta_path = std::filesystem::temp_directory_path() / "geodesic_open_syncmer_test.fa";
    {
        std::ofstream out(fasta_path);
        REQUIRE(out.good());
        out << ">seq\n" << seq << '\n';
    }

    MinHasher hasher({.kmer_size = k, .sketch_size = m, .syncmer_s = s, .seed = seed});
    const auto actual = hasher.sketch_oph_with_positions(fasta_path, m);
    std::filesystem::remove(fasta_path);

    const auto expected = manual_open_syncmer_oph(seq, k, s, m, seed);
    REQUIRE(actual.genome_length == expected.genome_length);
    REQUIRE(actual.n_real_bins == expected.n_real_bins);
    REQUIRE(actual.signature == expected.signature);
    REQUIRE(actual.real_bins_bitmask == expected.real_bins_bitmask);
}

// ---------------------------------------------------------------------------
// find_diversity_threshold tests
// ---------------------------------------------------------------------------

TEST_CASE("find_diversity_threshold bimodal", "[geodesic][threshold]") {
    // Bimodal: 1000 points near 0.01, 1000 points near 0.10
    std::vector<float> dists;
    dists.reserve(2000);
    std::mt19937 rng(42);
    std::normal_distribution<float> close(0.01f, 0.002f);
    std::normal_distribution<float> far(0.10f, 0.02f);
    for (int i = 0; i < 1000; ++i) dists.push_back(std::abs(close(rng)));
    for (int i = 0; i < 1000; ++i) dists.push_back(std::abs(far(rng)));
    std::sort(dists.begin(), dists.end());
    float thresh = derep::GeodesicDerep::find_diversity_threshold(dists, 0.5f);
    REQUIRE(thresh > 0.02f);  // above intra-strain peak
    REQUIRE(thresh < 0.08f);  // below inter-strain peak
}

TEST_CASE("find_diversity_threshold unimodal fallback", "[geodesic][threshold]") {
    // Unimodal: all near 0.05
    std::vector<float> dists;
    dists.reserve(2000);
    std::mt19937 rng(42);
    std::normal_distribution<float> d(0.05f, 0.01f);
    for (int i = 0; i < 2000; ++i) dists.push_back(std::abs(d(rng)));
    std::sort(dists.begin(), dists.end());
    float thresh = derep::GeodesicDerep::find_diversity_threshold(dists, 0.5f);
    REQUIRE(thresh > 0.03f);
    REQUIRE(thresh < 0.15f);
}

TEST_CASE("find_diversity_threshold respects ANI cap", "[geodesic][threshold]") {
    // All points far apart — fallback MAD would exceed cap; result must stay <= cap
    std::vector<float> dists(1000, 0.50f);
    std::sort(dists.begin(), dists.end());
    float thresh = derep::GeodesicDerep::find_diversity_threshold(dists, 0.10f);
    REQUIRE(thresh <= 0.10f);
}
