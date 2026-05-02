#pragma once
#include <filesystem>
#include <string>
#include <unordered_map>

namespace derep::taxonomy {

struct PartitionConfig {
    std::filesystem::path input_accessions;  // single-column accession list
    std::filesystem::path output_dir;
    int n_parts = 1;
    std::string rank = "g";   // "g"=genus, "f"=family, "s"=species
    // Taxonomy string per accession (resolved from pack TAXN section by caller).
    const std::unordered_map<std::string, std::string>* acc_taxonomy = nullptr;
};

// Partition the accession list into <n_parts> part_N.txt files under <output_dir>
// using LPT (Longest Processing Time) greedy bin-packing by taxonomy rank.
// All genomes of the same rank group land in the same part.
// Returns total genome count.
size_t partition_accessions(const PartitionConfig& cfg);

} // namespace derep::taxonomy
