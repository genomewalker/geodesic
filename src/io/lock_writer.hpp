#pragma once
#include <filesystem>
#include <string>
#include <cstdint>

namespace derep {

struct LockData {
    std::filesystem::path gpk_path;
    uint64_t gpk_snapshot_id = 0;
    std::string geodesic_version = "1.0.0";
    int kmer_size = 0;
    int sketch_size = 0;
    int syncmer_s = 0;
    double ani_threshold = 0.0;
    uint64_t seed1 = 42;
    uint64_t seed2 = 43;
    uint32_t params_hash = 0;
    uint64_t geodf_hash = 0;
    std::string timestamp;
    size_t n_genomes = 0;
    size_t n_taxa = 0;
    size_t n_reps = 0;
    std::filesystem::path geodf_path;
};

void write_lock_file(const std::filesystem::path& path, const LockData& data);
LockData read_lock_file(const std::filesystem::path& path);

// FNV-1a-64 of a file's last 32 bytes — fast version-sensitive hash
uint64_t file_tail_hash(const std::filesystem::path& path);

} // namespace derep
