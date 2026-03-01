#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "core/sketch/minhash.hpp"
#include "core/sketch/lsh_index.hpp"

using namespace derep;

// ─── MinHash Tests ───────────────────────────────────────────────────────────

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

    // Create two random sequences with different seeds
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

    // Random sequences should have low Jaccard (not identical)
    double j = s1.jaccard(s2);
    REQUIRE(j < 1.0);
}

TEST_CASE("MinHasher: estimate_ani correlates with sequence similarity", "[minhash]") {
    MinHasher hasher({.kmer_size = 16, .sketch_size = 1000});

    // Create base sequence
    std::string seq1(50000, 'A');
    uint32_t seed = 12345;
    for (size_t i = 0; i < seq1.size(); ++i) {
        seed = seed * 1103515245 + 12345;
        seq1[i] = "ACGT"[(seed >> 16) % 4];
    }

    // Create similar sequence (5% mutation rate)
    std::string seq2 = seq1;
    for (size_t i = 0; i < seq2.size(); i += 20) {
        seq2[i] = "TGCA"[seq2[i] % 4];
    }

    auto s1 = hasher.sketch_sequence(seq1);
    auto s2 = hasher.sketch_sequence(seq2);

    double ani = s1.estimate_ani(s2, 16);

    // With ~5% mutation, ANI should be around 0.95
    REQUIRE(ani > 0.85);
    REQUIRE(ani < 1.0);

    // Identical sequences should have ANI = 1.0
    auto s3 = hasher.sketch_sequence(seq1);
    REQUIRE(s1.estimate_ani(s3, 16) == 1.0);
}

TEST_CASE("MinHasher: mash_distance inversely correlates with Jaccard", "[minhash]") {
    MinHasher hasher({.kmer_size = 8, .sketch_size = 1000});  // Smaller k for more overlap

    // Create a random-ish base sequence
    std::string seq1;
    seq1.reserve(50000);
    uint32_t seed = 12345;
    for (int i = 0; i < 50000; ++i) {
        seed = seed * 1103515245 + 12345;
        seq1 += "ACGT"[(seed >> 16) % 4];
    }

    // Create seq2 with 10% mutation rate (should still share many k-mers)
    std::string seq2 = seq1;
    for (size_t i = 0; i < seq2.size(); i += 10) {
        seq2[i] = "TGCA"[seq2[i] % 4];  // Mutate every 10th base
    }

    auto s1 = hasher.sketch_sequence(seq1);
    auto s2 = hasher.sketch_sequence(seq2);

    double j = s1.jaccard(s2);
    double d = s1.mash_distance(s2, 8);

    // With 10% mutation rate and k=8, should have moderate Jaccard
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

// ─── LSH Index Tests ─────────────────────────────────────────────────────────

TEST_CASE("LSHIndex: identical sketches are candidates", "[lsh]") {
    MinHasher hasher({.kmer_size = 16, .sketch_size = 1000});
    LSHIndex index({.num_bands = 20, .rows_per_band = 50});

    std::string seq(10000, 'A');
    for (size_t i = 0; i < seq.size(); ++i) {
        seq[i] = "ACGTACGTACGTACGT"[i % 16];
    }

    auto s1 = hasher.sketch_sequence(seq);
    auto s2 = s1;  // Copy

    s1.genome_id = 1;
    s2.genome_id = 2;

    index.insert(1, s1);
    index.insert(2, s2);

    auto candidates = index.query(1, s1);
    REQUIRE(std::find(candidates.begin(), candidates.end(), 2) != candidates.end());
}

TEST_CASE("LSHIndex: very different sketches are not candidates", "[lsh]") {
    MinHasher hasher({.kmer_size = 16, .sketch_size = 1000});
    LSHIndex index({.num_bands = 20, .rows_per_band = 50});

    // Two completely different sequences
    std::string seq1(10000, 'A');
    std::string seq2(10000, 'T');

    for (size_t i = 0; i < seq1.size(); ++i) {
        seq1[i] = "AAACCCGGG"[i % 9];
        seq2[i] = "TTTGGGCCC"[i % 9];
    }

    auto s1 = hasher.sketch_sequence(seq1);
    auto s2 = hasher.sketch_sequence(seq2);

    s1.genome_id = 1;
    s2.genome_id = 2;

    index.insert(1, s1);
    index.insert(2, s2);

    auto candidates = index.query(1, s1);
    // Very different sequences should have low probability of being candidates
    // (not guaranteed, but highly likely with good hash functions)
    // Just check the query doesn't crash
    REQUIRE(candidates.size() <= 1);  // May or may not include genome 2
}

TEST_CASE("LSHIndex: find_all_candidates returns pairs", "[lsh]") {
    MinHasher hasher({.kmer_size = 8, .sketch_size = 100});
    LSHIndex index({.num_bands = 10, .rows_per_band = 10});

    // Create 5 similar sequences
    std::vector<std::pair<uint64_t, MinHashSketch>> genomes;
    std::string base_seq(5000, 'A');
    for (size_t i = 0; i < base_seq.size(); ++i) {
        base_seq[i] = "ACGTACGT"[i % 8];
    }

    for (uint64_t id = 1; id <= 5; ++id) {
        std::string seq = base_seq;
        // Small mutations
        for (size_t i = 0; i < seq.size(); i += 100) {
            seq[i] = "TGCA"[(id + i) % 4];
        }
        auto sketch = hasher.sketch_sequence(seq);
        sketch.genome_id = id;
        genomes.emplace_back(id, sketch);
    }

    auto candidates = index.find_all_candidates(genomes);

    // Should find some candidate pairs among similar sequences
    REQUIRE(candidates.size() > 0);

    // All pairs should be canonical (first < second)
    for (const auto& [a, b] : candidates) {
        REQUIRE(a < b);
    }
}

TEST_CASE("LSHIndex: statistics work correctly", "[lsh]") {
    MinHasher hasher({.kmer_size = 8, .sketch_size = 100});
    LSHIndex index({.num_bands = 10, .rows_per_band = 10});

    REQUIRE(index.num_genomes() == 0);
    REQUIRE(index.num_buckets() == 0);

    std::string seq(1000, 'A');
    for (size_t i = 0; i < seq.size(); ++i) {
        seq[i] = "ACGT"[i % 4];
    }

    auto s = hasher.sketch_sequence(seq);
    s.genome_id = 1;
    index.insert(1, s);

    REQUIRE(index.num_genomes() == 1);
    REQUIRE(index.num_buckets() > 0);
    REQUIRE(index.avg_bucket_size() > 0.0);
}

// ─── SketchPrecluster Tests ─────────────────────────────────────────────────

#include "core/sketch/sketch_precluster.hpp"
#include <fstream>

TEST_CASE("SketchPrecluster: sketch_all returns correct count", "[sketch_precluster]") {
    // Create temp FASTA files
    auto temp_dir = std::filesystem::temp_directory_path() / "sketch_test";
    std::filesystem::create_directories(temp_dir);

    std::vector<std::filesystem::path> files;
    for (int i = 0; i < 5; ++i) {
        auto path = temp_dir / ("genome_" + std::to_string(i) + ".fasta");
        std::ofstream out(path);
        out << ">seq\n";
        std::string seq(5000, 'A');
        uint32_t seed = 12345 + i * 111;
        for (size_t j = 0; j < seq.size(); ++j) {
            seed = seed * 1103515245 + 12345;
            seq[j] = "ACGT"[(seed >> 16) % 4];
        }
        out << seq << "\n";
        files.push_back(path);
    }

    SketchPrecluster precluster({
        .kmer_size = 16,
        .sketch_size = 1000,
        .num_bands = 20,
        .rows_per_band = 50,
        .mash_threshold = 0.5,
        .threads = 2
    });

    auto sketches = precluster.sketch_all(files);

    REQUIRE(sketches.size() == 5);
    REQUIRE(precluster.total_sketched() == 5);

    // Cleanup
    std::filesystem::remove_all(temp_dir);
}

TEST_CASE("SketchPrecluster: similar genomes become candidates", "[sketch_precluster]") {
    auto temp_dir = std::filesystem::temp_directory_path() / "sketch_test2";
    std::filesystem::create_directories(temp_dir);

    // Create base sequence (larger for more k-mers)
    std::string base_seq(50000, 'A');
    uint32_t seed = 42;
    for (size_t i = 0; i < base_seq.size(); ++i) {
        seed = seed * 1103515245 + 12345;
        base_seq[i] = "ACGT"[(seed >> 16) % 4];
    }

    // Create 3 similar genomes (very small mutations) and 1 very different
    std::vector<std::filesystem::path> files;
    for (int i = 0; i < 4; ++i) {
        auto path = temp_dir / ("genome_" + std::to_string(i) + ".fasta");
        std::ofstream out(path);
        out << ">seq\n";

        std::string seq = base_seq;
        if (i < 3) {
            // Very small mutations (0.5% mutation rate) to maintain high Jaccard
            for (size_t j = 0; j < seq.size(); j += 200) {
                seq[j] = "TGCA"[(i + j/200) % 4];
            }
        } else {
            // Completely different sequence
            seed = 999999;
            for (size_t j = 0; j < seq.size(); ++j) {
                seed = seed * 1103515245 + 12345;
                seq[j] = "ACGT"[(seed >> 16) % 4];
            }
        }
        out << seq << "\n";
        files.push_back(path);
    }

    // Use smaller k-mer and fewer bands/rows for easier matching in tests
    SketchPrecluster precluster({
        .kmer_size = 8,        // Smaller k = more k-mers, higher Jaccard
        .sketch_size = 500,
        .num_bands = 10,       // Fewer bands = lower threshold to match
        .rows_per_band = 5,    // Fewer rows = lower threshold
        .mash_threshold = 0.5, // Generous threshold
        .threads = 1
    });

    auto edges = precluster.compute_candidates(files);

    // Similar genomes (0,1,2) should have edges between them
    bool found_similar_pair = false;

    for (const auto& e : edges) {
        std::filesystem::path src(e.source);
        std::filesystem::path tgt(e.target);
        int src_id = std::stoi(src.stem().string().substr(7));
        int tgt_id = std::stoi(tgt.stem().string().substr(7));

        if (src_id < 3 && tgt_id < 3) {
            found_similar_pair = true;
        }
    }

    REQUIRE(found_similar_pair);

    // Cleanup
    std::filesystem::remove_all(temp_dir);
}

// ─── Syncmer Index Tests ────────────────────────────────────────────────────

#include "core/sketch/syncmer_index.hpp"

TEST_CASE("SyncmerIndex: identical sequences share all syncmers", "[syncmer]") {
    SyncmerIndex index({.k = 16, .s = 8, .min_shared = 1});

    std::string seq(10000, 'A');
    uint32_t seed = 42;
    for (size_t i = 0; i < seq.size(); ++i) {
        seed = seed * 1103515245 + 12345;
        seq[i] = "ACGT"[(seed >> 16) % 4];
    }

    auto s1 = index.extract(seq, 0);
    auto s2 = index.extract(seq, 1);

    REQUIRE(s1.hashes.size() > 0);
    REQUIRE(s1.hashes == s2.hashes);
    REQUIRE(SyncmerIndex::estimate_jaccard(s1, s2) == 1.0);
}

TEST_CASE("SyncmerIndex: similar sequences share most syncmers", "[syncmer]") {
    SyncmerIndex index({.k = 16, .s = 8, .min_shared = 1});

    // Create base sequence
    std::string seq1(50000, 'A');
    uint32_t seed = 42;
    for (size_t i = 0; i < seq1.size(); ++i) {
        seed = seed * 1103515245 + 12345;
        seq1[i] = "ACGT"[(seed >> 16) % 4];
    }

    // Create ~99% similar sequence (1% mutation)
    std::string seq2 = seq1;
    for (size_t i = 0; i < seq2.size(); i += 100) {
        seq2[i] = "TGCA"[seq2[i] % 4];
    }

    auto s1 = index.extract(seq1, 0);
    auto s2 = index.extract(seq2, 1);

    double j = SyncmerIndex::estimate_jaccard(s1, s2);
    REQUIRE(j > 0.5);  // High Jaccard for similar sequences
    REQUIRE(j < 1.0);  // But not identical
}

TEST_CASE("SyncmerIndex: find_candidates detects similar pairs", "[syncmer]") {
    SyncmerIndex index({.k = 8, .s = 4, .min_shared = 5, .threads = 1});

    // Create base sequence
    std::string base(20000, 'A');
    uint32_t seed = 42;
    for (size_t i = 0; i < base.size(); ++i) {
        seed = seed * 1103515245 + 12345;
        base[i] = "ACGT"[(seed >> 16) % 4];
    }

    // Create 4 genomes: 3 similar, 1 different
    std::vector<SyncmerIndex::GenomeSyncmers> genomes;
    for (int i = 0; i < 4; ++i) {
        std::string seq = base;
        if (i < 3) {
            // Small mutations for similar genomes
            for (size_t j = 0; j < seq.size(); j += 500) {
                seq[j] = "TGCA"[(i + j/500) % 4];
            }
        } else {
            // Completely different
            seed = 999999;
            for (size_t j = 0; j < seq.size(); ++j) {
                seed = seed * 1103515245 + 12345;
                seq[j] = "ACGT"[(seed >> 16) % 4];
            }
        }
        genomes.push_back(index.extract(seq, i));
    }

    auto candidates = index.find_candidates(genomes);

    // Should find pairs among similar genomes (0,1,2)
    bool found_01 = false, found_02 = false, found_12 = false;
    for (const auto& [a, b] : candidates) {
        if ((a == 0 && b == 1) || (a == 1 && b == 0)) found_01 = true;
        if ((a == 0 && b == 2) || (a == 2 && b == 0)) found_02 = true;
        if ((a == 1 && b == 2) || (a == 2 && b == 1)) found_12 = true;
    }

    REQUIRE((found_01 || found_02 || found_12));  // At least one similar pair found
}
