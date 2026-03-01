#pragma once
#include "core/similarity/similarity_base.hpp"
#include "core/genome_cache.hpp"
#include <unordered_map>

namespace derep {

struct SkaniConfig {
    int kmer_size = 15;
    int sketch_size = 10000;
    double min_conf = 0.2;
    double target_ani = 0.95;
    int min_kmers = 100;
    double ani_threshold = 95.0;
    int large_cluster_threshold = 1000;
};

class Skani : public SimilarityMeasure {
public:
    Skani(int threads, std::filesystem::path temp_dir,
          GenomeCache& cache, SkaniConfig config = {});

    [[nodiscard]] std::vector<SimilarityEdge> compute_pairwise(
        const std::vector<std::filesystem::path>& assemblies) override;

    // Compute ANI only for specific candidate pairs (O(n) complexity)
    // candidate_pairs: vector of (source_idx, target_idx) pairs into assemblies
    [[nodiscard]] std::vector<SimilarityEdge> compute_candidates(
        const std::vector<std::filesystem::path>& assemblies,
        const std::vector<std::pair<size_t, size_t>>& candidate_pairs);

    [[nodiscard]] std::vector<SimilarityEdge> filter(
        std::vector<SimilarityEdge> edges) const override;

private:
    GenomeCache& cache_;
    SkaniConfig config_;

    [[nodiscard]] std::vector<SimilarityEdge> run_triangle(
        const std::vector<std::filesystem::path>& assemblies,
        const std::filesystem::path& temp) const;

    [[nodiscard]] std::vector<SimilarityEdge> run_sketch_dist(
        const std::vector<std::filesystem::path>& assemblies,
        const std::filesystem::path& temp) const;

    // Run skani dist for query genome against specific references
    [[nodiscard]] std::vector<SimilarityEdge> run_dist_pairs(
        const std::filesystem::path& query,
        const std::vector<std::filesystem::path>& references,
        const std::filesystem::path& temp) const;

    [[nodiscard]] std::vector<SimilarityEdge> parse_skani_output(
        const std::filesystem::path& output_file) const;
};

} // namespace derep
