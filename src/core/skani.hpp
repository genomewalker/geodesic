#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace derep {

struct SkaniSketch {
    std::string accession;
    std::vector<uint64_t> hashes;  // sorted, sketch_size smallest syncmer hashes
    uint64_t genome_length = 0;
};

struct SkaniResult {
    double ani  = 0.0;
    double af   = 0.0;
    double c_ab = 0.0;
    double c_ba = 0.0;
};

SkaniSketch build_sketch(std::string_view accession, std::string_view fasta,
                          int k = 21, int s = 7, int sketch_size = 1000);

SkaniResult compute_ani(const SkaniSketch& a, const SkaniSketch& b, int k = 21);

} // namespace derep
