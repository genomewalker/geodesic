#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace derep {

struct SkaniSketch {
    std::string accession;
    std::vector<uint64_t> hashes;  // sorted canonical k-mer hashes, density ~1/c
    uint64_t genome_length = 0;
};

struct SkaniResult {
    double ani  = 0.0;
    double af   = 0.0;
    double c_ab = 0.0;
    double c_ba = 0.0;
};

// Build a sketch using skani's compression model: select k-mer if canonical_hash % c == 0.
// Sketch size scales with genome_length / c, identical to skani -c parameter.
SkaniSketch build_sketch(std::string_view accession, std::string_view fasta,
                          int k = 21, int c = 125);

SkaniResult compute_ani(const SkaniSketch& a, const SkaniSketch& b, int k = 21);

} // namespace derep
