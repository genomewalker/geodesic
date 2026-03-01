#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "core/sketch/minhash.hpp"

using namespace derep;

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
