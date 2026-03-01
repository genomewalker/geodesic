#pragma once
#include "core/types.hpp"
#include "core/subprocess.hpp"
#include <filesystem>
#include <string>
#include <vector>

namespace derep {

class SimilarityMeasure {
public:
    SimilarityMeasure(int threads, std::filesystem::path temp_dir)
        : threads_(threads), temp_dir_(std::move(temp_dir)) {}
    virtual ~SimilarityMeasure() = default;

    [[nodiscard]] virtual std::vector<SimilarityEdge> compute_pairwise(
        const std::vector<std::filesystem::path>& assemblies) = 0;

    [[nodiscard]] virtual std::vector<SimilarityEdge> filter(
        std::vector<SimilarityEdge> edges) const = 0;

    [[nodiscard]] std::vector<std::filesystem::path> missing_assemblies(
        const std::vector<SimilarityEdge>& edges,
        const std::vector<std::filesystem::path>& all_assemblies) const;

protected:
    int threads_;
    std::filesystem::path temp_dir_;

    [[nodiscard]] std::filesystem::path write_assembly_list(
        const std::vector<std::filesystem::path>& assemblies,
        const std::string& prefix) const;

    [[nodiscard]] std::filesystem::path make_temp_subdir(const std::string& prefix) const;
};

} // namespace derep
