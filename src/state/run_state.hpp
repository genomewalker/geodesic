#pragma once
#include "core/types.hpp"
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace derep {

struct FailedGenomeRecord {
    std::string accession;
    std::string taxonomy;
    std::string file;
    std::string reason;
};

struct OutlierRecord {
    std::string accession;
    double centroid_distance;
    double isolation_score;
    double anomaly_score;
    double genome_size_zscore = 0.0;
    bool nn_outlier = false;
    double kmer_div_zscore = 0.0;
    double margin_to_threshold = 0.0;
    std::string flag_reason;
    // category: "misassigned" | "low_quality" | "contaminated"
    // misassigned  = embedding outlier (wrong species, divergent strain)
    // low_quality  = fragmented/incomplete assembly (pre-filter or quality floor)
    // contaminated = chimeric assembly (GUNC failure)
    std::string category;
    uint32_t n_contigs = 0;
    uint64_t genome_length_bp = 0;
    bool excluded = true;
};

struct TaxonOutput {
    TaxonResult result;
    TaxonDiversityStats diversity_stats;

    std::vector<std::string> all_accessions;
    std::vector<std::string> representatives;
    std::unordered_map<std::string, double> ani_map;

    std::vector<std::vector<float>>                     rep_embeddings;
    std::unordered_map<std::string, std::string>        member_to_rep;
    std::unordered_map<std::string, double>             member_nn_dist;       // OPH distance to assigned rep
    std::unordered_map<std::string, float>              member_fill_ratio;    // n_real_bins / sketch_size
    std::unordered_map<std::string, uint32_t>           rep_cluster_size;
    int                                                 sketch_kmer_used = 0;

    std::vector<OutlierRecord> outliers;

    std::vector<Genome> completed_genomes;
    std::vector<FailedGenomeRecord> failed_genomes;

    // Pipeline health counters
    int n_input = 0;
    int n_preflight_excluded = 0;
    int n_quality_floor_excluded = 0;
    int n_outliers_excluded = 0;
    int n_outliers_retained = 0;
    // MST/threshold diagnostics
    double mst_p90_edge = 0.0;
    double mst_true_max = 0.0;
    double ani_threshold_used = 0.0;
};

class RunState {
public:
    void push(TaxonOutput output);

    size_t total_genomes() const;
    size_t total_reps() const;
    size_t total_failed() const;
    const std::vector<TaxonOutput>& taxa() const;

    // Sort taxa_ by taxonomy for deterministic emission order. push() is called from
    // worker threads in completion order — call this after scheduler.join() and before
    // any reader (writers, .gpd archive, etc) to make outputs bit-identical across runs.
    void finalize_sort();

private:
    mutable std::mutex mutex_;
    std::vector<TaxonOutput> taxa_;
};

} // namespace derep
