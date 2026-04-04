#include "io/results_writer.hpp"
#include "state/run_state.hpp"

#include <fstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include <spdlog/spdlog.h>

namespace derep {

ResultsWriter::ResultsWriter(std::filesystem::path output_dir, std::string prefix)
    : output_dir_(std::move(output_dir)), prefix_(std::move(prefix)) {}

void ResultsWriter::write_derep_genomes(const RunState& state,
                                        const std::vector<GenomeRow>& all_genomes) const {
    // Build accession → file_path lookup from input rows
    std::unordered_map<std::string, std::string> acc_to_file;
    acc_to_file.reserve(all_genomes.size());
    for (const auto& row : all_genomes)
        acc_to_file[row.accession] = row.file_path.string();

    auto path = output_dir_ / (prefix_ + "_derep_genomes.tsv");
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot open output file: " + path.string());

    out << "accession\ttaxonomy\tfile\trepresentative\n";

    for (const auto& taxon : state.taxa()) {
        const std::string& taxonomy = taxon.result.taxonomy;
        std::unordered_set<std::string> rep_set(taxon.representatives.begin(),
                                                taxon.representatives.end());
        for (const auto& acc : taxon.all_accessions) {
            auto file_it = acc_to_file.find(acc);
            const std::string file = (file_it != acc_to_file.end()) ? file_it->second : "";
            bool is_rep = rep_set.count(acc) > 0;
            out << acc << '\t' << taxonomy << '\t' << file << '\t' << is_rep << '\n';
        }
    }

    spdlog::info("Wrote derep genomes to {}", path.string());
}

void ResultsWriter::write_stats(const RunState& state) const {
    auto path = output_dir_ / (prefix_ + "_stats.tsv");
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot open output file: " + path.string());

    out << "taxonomy\tmethod\tn_input\tn_preflight_excluded\tn_quality_floor_excluded\t"
           "n_outliers_excluded\tn_outliers_retained\tn_failed\tn_embedded\t"
           "n_representatives\trep_fraction\t"
           "mst_p90_edge\tmst_true_max\tani_threshold_used\t"
           "n_outliers_fragmented\tn_outliers_size\tn_outliers_nn_only\n";

    for (const auto& taxon : state.taxa()) {
        if (taxon.result.status == TaxonStatus::FAILED) continue;

        const int n_reps   = static_cast<int>(taxon.representatives.size());
        const int n_failed = static_cast<int>(taxon.failed_genomes.size());
        const int n_embedded = taxon.n_input - taxon.n_preflight_excluded - n_failed;
        const double rep_frac = taxon.n_input > 0
            ? static_cast<double>(n_reps) / taxon.n_input
            : 0.0;

        int n_frag = 0, n_size = 0, n_nn = 0;
        for (const auto& o : taxon.outliers) {
            const bool has_frag = o.flag_reason.find("fragmented") != std::string::npos;
            const bool has_size = o.flag_reason.find("size_outlier") != std::string::npos;
            if (has_frag)       ++n_frag;
            else if (has_size)  ++n_size;
            else                ++n_nn;
        }

        out << taxon.result.taxonomy << '\t'
            << taxon.result.method << '\t'
            << taxon.n_input << '\t'
            << taxon.n_preflight_excluded << '\t'
            << taxon.n_quality_floor_excluded << '\t'
            << taxon.n_outliers_excluded << '\t'
            << taxon.n_outliers_retained << '\t'
            << n_failed << '\t'
            << n_embedded << '\t'
            << n_reps << '\t'
            << rep_frac << '\t'
            << taxon.mst_p90_edge << '\t'
            << taxon.mst_true_max << '\t'
            << taxon.ani_threshold_used << '\t'
            << n_frag << '\t'
            << n_size << '\t'
            << n_nn << '\n';
    }

    spdlog::info("Wrote pipeline stats to {}", path.string());
}

void ResultsWriter::write_diversity_stats(const RunState& state) const {
    auto path = output_dir_ / (prefix_ + "_diversity_stats.tsv");
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot open output file: " + path.string());

    out << "taxonomy\tmethod\tn_genomes\tn_representatives\treduction_ratio\truntime_seconds\t"
           "coverage_mean_ani\tcoverage_min_ani\tcoverage_max_ani\t"
           "coverage_below_99\tcoverage_below_98\tcoverage_below_97\tcoverage_below_95\t"
           "diversity_mean_ani\tdiversity_min_ani\tdiversity_max_ani\t"
           "diversity_ani_range\tdiversity_n_pairs\tn_outliers_excluded\tn_outliers_retained\n";

    for (const auto& taxon : state.taxa()) {
        if (taxon.result.status == TaxonStatus::FAILED) continue;
        const auto& d = taxon.diversity_stats;
        if (d.taxonomy.empty()) continue;
        out << d.taxonomy << '\t'
            << d.method << '\t'
            << d.n_genomes << '\t'
            << d.n_representatives << '\t'
            << d.reduction_ratio << '\t'
            << d.runtime_seconds << '\t'
            << d.coverage_mean_ani << '\t'
            << d.coverage_min_ani << '\t'
            << d.coverage_max_ani << '\t'
            << d.coverage_below_99 << '\t'
            << d.coverage_below_98 << '\t'
            << d.coverage_below_97 << '\t'
            << d.coverage_below_95 << '\t'
            << d.diversity_mean_ani << '\t'
            << d.diversity_min_ani << '\t'
            << d.diversity_max_ani << '\t'
            << d.diversity_ani_range << '\t'
            << d.diversity_n_pairs << '\t'
            << d.n_outliers_excluded << '\t'
            << d.n_outliers_retained << '\n';
    }

    spdlog::info("Wrote diversity stats to {}", path.string());
}

void ResultsWriter::write_results(const RunState& state) const {
    auto path = output_dir_ / (prefix_ + "_results.tsv");
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot open output file: " + path.string());

    out << "taxonomy\tmethod\tn_genomes\tn_genomes_derep\tcommunities\tweight\n";

    for (const auto& taxon : state.taxa()) {
        const auto& r = taxon.result;
        out << r.taxonomy << '\t'
            << r.method << '\t'
            << r.n_genomes << '\t'
            << r.n_representatives << '\t'
            << r.n_communities << '\t'
            << "NA" << '\n';
    }

    spdlog::info("Wrote results to {}", path.string());
}

void ResultsWriter::write_failed(const RunState& state) const {
    auto path = output_dir_ / (prefix_ + "_failed.tsv");
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot open output file: " + path.string());

    out << "accession\ttaxonomy\tfile\treason\n";

    for (const auto& taxon : state.taxa()) {
        for (const auto& f : taxon.failed_genomes) {
            out << f.accession << '\t'
                << f.taxonomy << '\t'
                << f.file << '\t';
            if (f.reason.empty())
                out << "NA";
            else
                out << f.reason;
            out << '\n';
        }
    }

    spdlog::info("Wrote failed jobs to {}", path.string());
}

void ResultsWriter::write_outliers(const RunState& state) const {
    auto path = output_dir_ / (prefix_ + "_outliers.tsv");
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot open output file: " + path.string());

    out << "taxonomy\taccession\tnn_outlier\tisolation_score\tkmer_div_zscore\t"
           "genome_size_zscore\tcentroid_distance\tanomaly_score\t"
           "genome_length_bp\tn_contigs\tmargin_to_threshold\tflag_reason\texcluded\n";

    for (const auto& taxon : state.taxa()) {
        for (const auto& c : taxon.outliers) {
            out << taxon.result.taxonomy << '\t'
                << c.accession << '\t'
                << c.nn_outlier << '\t'
                << c.isolation_score << '\t'
                << c.kmer_div_zscore << '\t'
                << c.genome_size_zscore << '\t'
                << c.centroid_distance << '\t'
                << c.anomaly_score << '\t'
                << c.genome_length_bp << '\t'
                << c.n_contigs << '\t'
                << c.margin_to_threshold << '\t'
                << c.flag_reason << '\t'
                << c.excluded << '\n';
        }
    }

    spdlog::info("Wrote outliers to {}", path.string());
}

void ResultsWriter::write_all(const RunState& state,
                              const std::vector<GenomeRow>& all_genomes) const {
    std::filesystem::create_directories(output_dir_);
    write_derep_genomes(state, all_genomes);
    write_stats(state);
    write_diversity_stats(state);
    write_results(state);
    write_failed(state);
    write_outliers(state);
}

} // namespace derep
