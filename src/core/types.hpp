#pragma once
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace derep {

struct Genome {
    std::string accession;
    std::string taxonomy;
    std::filesystem::path file_path;
    uint64_t genome_length = 0;
    std::optional<double> completeness;
    std::optional<double> contamination;

    [[nodiscard]] double quality_score() const noexcept {
        if (completeness && contamination)
            return *completeness - 5.0 * (*contamination);
        return 50.0;
    }
};

struct Taxon {
    std::string taxonomy;
    std::vector<Genome> genomes;
    std::optional<std::string> forced_representative;
    [[nodiscard]] std::size_t size() const noexcept { return genomes.size(); }
    [[nodiscard]] bool is_singleton() const noexcept { return genomes.size() == 1; }
};

struct SimilarityEdge {
    std::string source;
    std::string target;
    double weight_raw = 0.0;
    double aln_frac = 1.0;
    double weight = 0.0;
    int64_t source_len = 0;
    int64_t target_len = 0;

    void canonicalize() {
        if (source > target) {
            std::swap(source, target);
            std::swap(source_len, target_len);
        }
    }
};

struct ClusterResult {
    std::unordered_map<std::string, int> partition;
    std::vector<std::string> representatives;
    std::vector<std::string> filtered_representatives;
    std::optional<double> filter_weight;
};

enum class TaxonStatus { SUCCESS, SINGLETON, SKIPPED, FIXED, FAILED };

struct TaxonResult {
    std::string taxonomy;
    TaxonStatus status = TaxonStatus::FAILED;
    std::size_t n_genomes = 0;
    std::size_t n_representatives = 0;
    std::vector<std::string> representative_files;
    std::string error_message;
    std::string method;
    std::size_t n_communities = 0;
};

struct RepresentativeStats {
    std::string taxonomy;
    std::string representative;
    int n_nodes = 0;
    int n_nodes_selected = 0;
    int n_nodes_discarded = 0;
    std::optional<double> graph_avg_weight;
    std::optional<double> graph_sd_weight;
    std::optional<double> graph_avg_weight_raw;
    std::optional<double> graph_sd_weight_raw;
    std::optional<double> subgraph_selected_avg_weight;
    std::optional<double> subgraph_selected_sd_weight;
    std::optional<double> subgraph_selected_avg_weight_raw;
    std::optional<double> subgraph_selected_sd_weight_raw;
    std::optional<double> subgraph_discarded_avg_weight;
    std::optional<double> subgraph_discarded_sd_weight;
    std::optional<double> subgraph_discarded_avg_weight_raw;
    std::optional<double> subgraph_discarded_sd_weight_raw;
};

struct TaxonDiversityStats {
    std::string taxonomy;
    std::string method;
    int n_genomes = 0;
    int n_representatives = 0;
    double reduction_ratio = 0.0;
    double runtime_seconds = 0.0;

    // Coverage: ANI from each genome to nearest representative
    double coverage_mean_ani = 0.0;
    double coverage_min_ani = 0.0;
    double coverage_max_ani = 0.0;
    int coverage_below_99 = 0;
    int coverage_below_98 = 0;
    int coverage_below_97 = 0;
    int coverage_below_95 = 0;

    // Diversity: pairwise ANI among representatives
    double diversity_mean_ani = 0.0;
    double diversity_min_ani = 0.0;
    double diversity_max_ani = 0.0;
    double diversity_ani_range = 0.0;
    int diversity_n_pairs = 0;

    // Contamination: genomes flagged as potential contamination
    int n_contaminated = 0;
};

enum class PipelineStage : int {
    NOT_STARTED = 0,
    PRECLUSTER_DONE = 1,
    SIMILARITY_DONE = 2,
    GRAPH_DONE = 3,
    PARTITION_DONE = 4,
    REPRESENTATIVES_DONE = 5,
    COMPLETE = 6
};

} // namespace derep
