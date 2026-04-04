#pragma once
#include <filesystem>
#include <string>

namespace derep::taxonomy {

struct PartitionConfig {
    std::filesystem::path input_tsv;
    std::filesystem::path output_dir;
    int n_parts = 1;
    std::string rank = "g";   // "g"=genus, "f"=family
};

// Partition <input_tsv> into <n_parts> part_N.tsv files under <output_dir>
// using LPT (Longest Processing Time) greedy bin-packing by taxonomy rank.
// All genomes of the same genus land in the same part.
// Parts are sorted by taxonomy for better shard compression.
// Returns total genome count.
size_t partition_tsv(const PartitionConfig& cfg);

} // namespace derep::taxonomy
