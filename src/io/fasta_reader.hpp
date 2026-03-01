#pragma once
#include <cstdint>
#include <filesystem>
#include <stdexcept>

namespace derep {

class FastaParseError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class CompressionMode { kAuto, kPlain, kGzip };

struct FastaReadOptions {
    CompressionMode compression = CompressionMode::kAuto;
    std::size_t buffer_bytes = 1U << 20;  // 1 MiB
    bool strict_ascii_sequence = false;
};

[[nodiscard]] bool is_gzip_file(const std::filesystem::path& path);
[[nodiscard]] std::uint64_t calculate_genome_length(
    const std::filesystem::path& path,
    const FastaReadOptions& opts = {});

} // namespace derep
