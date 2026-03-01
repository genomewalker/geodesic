#pragma once
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace derep {

class TsvParseError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct TsvReadOptions {
    char delimiter = '\t';
    bool has_header = true;
    bool strict = true;
    bool trim_fields = true;
    bool allow_comments = true;
    char comment_prefix = '#';
};

struct GenomeRow {
    std::string accession;
    std::string taxonomy;
    std::filesystem::path file_path;
};

struct CheckM2Quality {
    double completeness = 0.0;
    double contamination = 0.0;
    [[nodiscard]] double quality_score() const noexcept {
        return completeness - 5.0 * contamination;
    }
};

[[nodiscard]] std::string canonical_accession(std::string_view acc);

[[nodiscard]] std::vector<GenomeRow> read_genomes_tsv(
    const std::filesystem::path& path,
    const TsvReadOptions& opts = {});

[[nodiscard]] std::unordered_map<std::string, CheckM2Quality> read_checkm2_tsv(
    const std::filesystem::path& path,
    const TsvReadOptions& opts = {});

[[nodiscard]] std::unordered_map<std::string, std::string> read_fixed_taxa_tsv(
    const std::filesystem::path& path,
    const TsvReadOptions& opts = {});

[[nodiscard]] std::size_t count_checkm2_matches(
    const std::vector<GenomeRow>& genomes,
    const std::unordered_map<std::string, CheckM2Quality>& quality);

} // namespace derep
