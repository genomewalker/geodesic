#pragma once
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace derep {

// MurmurHash3 64-bit finalizer for k-mer hashing
inline uint64_t murmur64(uint64_t k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return k;
}

// Canonical k-mer: min of forward and reverse complement hash
uint64_t canonical_kmer_hash(const char* seq, int k, uint64_t seed = 42);

// OPH (One-Permutation Hashing) signature: m bins, each holds the minimum
// hash value among all k-mers whose hash falls in that bin.
// Empty bins are densified deterministically.
// Property: P[sig_A[t] == sig_B[t]] = J(A,B) for each bin t.
struct OPHSketch {
    std::vector<uint32_t> signature;        // m bins, each: min high-32 hash in that bin
    std::vector<uint64_t> real_bins_bitmask; // ceil(m/64) words; bit t=1 iff bin t has a real k-mer
                                             // empty unless computed via sketch_oph_with_positions
    size_t genome_length = 0;
    size_t n_real_bins = 0;                 // non-empty bins before densification (for containment)
    size_t n_contigs = 0;                   // Number of sequences (FASTA headers)
    float chimera_score = 0.0f;             // max_i mean_j OPH Jaccard distance among large contigs [0,1];
                                             // stored for future analysis; not used for flagging
                                             // (real contigs score ~1.0 regardless of chimeric status)
};

struct MinHashSketch {
    uint64_t genome_id = 0;
    uint64_t genome_length = 0;
    std::vector<uint64_t> hashes;  // Bottom-s MinHash, sorted ascending

    // Jaccard similarity estimate
    double jaccard(const MinHashSketch& other) const;

    // Mash distance: -1/k * ln(2J / (1+J))
    double mash_distance(const MinHashSketch& other, int k) const;

    // Estimated ANI from Mash distance: ANI = 1 - D
    // Based on Ondov et al. 2016 (Mash paper)
    // Returns value in [0, 1] range (multiply by 100 for percentage)
    double estimate_ani(const MinHashSketch& other, int k) const;
};

class MinHasher {
public:
    static constexpr int DEFAULT_SKETCH_SIZE = 10000;
    static constexpr int OPH_BINS = 10000;

    struct Config {
        int kmer_size = 16;      // k=16 gives J~0.29 at 95% ANI
        int sketch_size = DEFAULT_SKETCH_SIZE; // Bottom-s
        int syncmer_s = 0;       // 0 = disabled, >0 = open syncmer prefilter for OPH
        uint64_t seed = 42;
    };

    MinHasher();
    explicit MinHasher(Config cfg);

    // Compute bottom-s MinHash sketch from FASTA file (plain or gzipped)
    MinHashSketch sketch(const std::filesystem::path& fasta_path) const;

    // Compute sketch from sequence string
    MinHashSketch sketch_sequence(const std::string& sequence) const;

    // Compute OPH signature from FASTA file.
    // m = number of bins (typically sketch_size); uses cfg_.kmer_size and cfg_.seed.
    OPHSketch sketch_oph(const std::filesystem::path& fasta_path, int m) const;

    // Compute OPH signature using an open-syncmer prefilter.
    // A k-mer is retained if the minimum s-mer in the k-mer occurs at offset 0.
    OPHSketch sketch_oph_syncmer(const std::filesystem::path& fasta_path, int m, int s) const;

    // Like sketch_oph but also records result.real_bins_bitmask (bit t=1 iff bin t has a real
    // k-mer). Used to build the per-genome real-bin mask for 2D OPH coverage estimation.
    OPHSketch sketch_oph_with_positions(const std::filesystem::path& fasta_path, int m) const;

    // Same as sketch_oph_with_positions but operates on a pre-decompressed FASTA buffer.
    // Used by the producer-consumer path in build_index() to overlap I/O and compute.
    OPHSketch sketch_oph_with_positions_from_buffer(const char* data, size_t len, int m) const;

    [[nodiscard]] int kmer_size() const { return cfg_.kmer_size; }
    [[nodiscard]] int sketch_size() const { return cfg_.sketch_size; }

private:
    Config cfg_;

    void add_kmers_from_sequence(const std::string& seq,
                                  std::vector<uint64_t>& heap) const;
};

static_assert(MinHasher::DEFAULT_SKETCH_SIZE == MinHasher::OPH_BINS,
              "Default sketch_size must match OPH_BINS for the fixed-bin OPH hot path.");

} // namespace derep
