#include "io/results_writer.hpp"
#include "db/db_manager.hpp"

#include <fstream>
#include <stdexcept>

#include <spdlog/spdlog.h>

namespace derep {

ResultsWriter::ResultsWriter(std::filesystem::path output_dir, std::string prefix)
    : output_dir_(std::move(output_dir)), prefix_(std::move(prefix)) {}

void ResultsWriter::write_derep_genomes(db::DBManager& db) const {
    auto path = output_dir_ / (prefix_ + "_derep_genomes.tsv");
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot open output file: " + path.string());

    out << "accession\ttaxonomy\tfile\trepresentative\n";

    auto result = db.query(
        "SELECT g.accession, g.taxonomy, g.file, gd.representative "
        "FROM genomes g JOIN genomes_derep gd ON g.accession = gd.accession "
        "ORDER BY g.taxonomy, g.accession");

    for (auto& row : *result) {
        out << row.GetValue<std::string>(0) << '\t'
            << row.GetValue<std::string>(1) << '\t'
            << row.GetValue<std::string>(2) << '\t'
            << row.GetValue<bool>(3) << '\n';
    }

    spdlog::info("Wrote derep genomes to {}", path.string());
}

void ResultsWriter::write_stats(db::DBManager& db) const {
    auto path = output_dir_ / (prefix_ + "_stats.tsv");
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot open output file: " + path.string());

    out << "taxonomy\trepresentative\tn_nodes\tn_nodes_selected\tn_nodes_discarded\t"
           "graph_avg_weight\tgraph_sd_weight\tgraph_avg_weight_raw\tgraph_sd_weight_raw\t"
           "subgraph_selected_avg_weight\tsubgraph_selected_sd_weight\t"
           "subgraph_selected_avg_weight_raw\tsubgraph_selected_sd_weight_raw\t"
           "subgraph_discarded_avg_weight\tsubgraph_discarded_sd_weight\t"
           "subgraph_discarded_avg_weight_raw\tsubgraph_discarded_sd_weight_raw\n";

    auto result = db.query(
        "SELECT taxonomy, representative, n_nodes, n_nodes_selected, n_nodes_discarded, "
        "graph_avg_weight, graph_sd_weight, graph_avg_weight_raw, graph_sd_weight_raw, "
        "subgraph_selected_avg_weight, subgraph_selected_sd_weight, "
        "subgraph_selected_avg_weight_raw, subgraph_selected_sd_weight_raw, "
        "subgraph_discarded_avg_weight, subgraph_discarded_sd_weight, "
        "subgraph_discarded_avg_weight_raw, subgraph_discarded_sd_weight_raw "
        "FROM stats ORDER BY taxonomy, representative");

    for (auto& row : *result) {
        out << row.GetValue<std::string>(0);
        for (int i = 1; i < 17; ++i) {
            out << '\t';
            if (row.IsNull(i)) {
                out << "NA";
            } else if (i <= 1) {
                out << row.GetValue<std::string>(i);
            } else if (i <= 4) {
                out << row.GetValue<int32_t>(i);
            } else {
                out << row.GetValue<double>(i);
            }
        }
        out << '\n';
    }

    spdlog::info("Wrote stats to {}", path.string());
}

void ResultsWriter::write_diversity_stats(db::DBManager& db) const {
    auto path = output_dir_ / (prefix_ + "_diversity_stats.tsv");
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot open output file: " + path.string());

    out << "taxonomy\tmethod\tn_genomes\tn_representatives\treduction_ratio\truntime_seconds\t"
           "coverage_mean_ani\tcoverage_min_ani\tcoverage_max_ani\t"
           "coverage_below_99\tcoverage_below_98\tcoverage_below_97\tcoverage_below_95\t"
           "diversity_mean_ani\tdiversity_min_ani\tdiversity_max_ani\t"
           "diversity_ani_range\tdiversity_n_pairs\tn_contaminated\n";

    auto result = db.query(
        "SELECT taxonomy, method, n_genomes, n_representatives, reduction_ratio, runtime_seconds, "
        "coverage_mean_ani, coverage_min_ani, coverage_max_ani, "
        "coverage_below_99, coverage_below_98, coverage_below_97, coverage_below_95, "
        "diversity_mean_ani, diversity_min_ani, diversity_max_ani, "
        "diversity_ani_range, diversity_n_pairs, n_contaminated "
        "FROM diversity_stats ORDER BY taxonomy");

    for (auto& row : *result) {
        out << row.GetValue<std::string>(0) << '\t'   // taxonomy
            << row.GetValue<std::string>(1) << '\t';  // method
        for (int i = 2; i < 19; ++i) {
            if (i > 2) out << '\t';
            if (row.IsNull(i)) {
                out << "NA";
            } else if (i <= 3 || (i >= 9 && i <= 12) || i >= 17) {
                // Integer columns: n_genomes, n_representatives, coverage_below_*, diversity_n_pairs, n_contaminated
                out << row.GetValue<int32_t>(i);
            } else {
                // Double columns
                out << row.GetValue<double>(i);
            }
        }
        out << '\n';
    }

    spdlog::info("Wrote diversity stats to {}", path.string());
}

void ResultsWriter::write_results(db::DBManager& db) const {
    auto path = output_dir_ / (prefix_ + "_results.tsv");
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot open output file: " + path.string());

    out << "taxonomy\tmethod\tn_genomes\tn_genomes_derep\tcommunities\tweight\n";

    auto result = db.query(
        "SELECT taxonomy, method, n_genomes, n_genomes_derep, communities, weight "
        "FROM results ORDER BY taxonomy");

    for (auto& row : *result) {
        out << row.GetValue<std::string>(0) << '\t'
            << row.GetValue<std::string>(1) << '\t';
        for (int i = 2; i < 6; ++i) {
            if (i > 2) out << '\t';
            if (row.IsNull(i)) {
                out << "NA";
            } else if (i <= 3) {
                out << row.GetValue<int32_t>(i);
            } else {
                out << row.GetValue<double>(i);
            }
        }
        out << '\n';
    }

    spdlog::info("Wrote results to {}", path.string());
}

void ResultsWriter::write_failed(db::DBManager& db) const {
    auto path = output_dir_ / (prefix_ + "_failed.tsv");
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot open output file: " + path.string());

    out << "accession\ttaxonomy\tfile\treason\n";

    auto result = db.query(
        "SELECT accession, taxonomy, file, reason "
        "FROM jobs_failed ORDER BY taxonomy, accession");

    for (auto& row : *result) {
        out << row.GetValue<std::string>(0) << '\t'
            << row.GetValue<std::string>(1) << '\t'
            << row.GetValue<std::string>(2) << '\t';
        if (row.IsNull(3)) {
            out << "NA";
        } else {
            out << row.GetValue<std::string>(3);
        }
        out << '\n';
    }

    spdlog::info("Wrote failed jobs to {}", path.string());
}

void ResultsWriter::write_contamination(db::DBManager& db) const {
    auto path = output_dir_ / (prefix_ + "_contamination.tsv");
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot open output file: " + path.string());

    out << "taxonomy\taccession\tnn_outlier\tisolation_score\tkmer_div_zscore\t"
           "genome_size_zscore\tcentroid_distance\tanomaly_score\t"
           "genome_length_bp\tn_contigs\tmargin_to_threshold\tflag_reason\n";

    auto result = db.query(
        "SELECT c.taxonomy, c.accession, c.nn_outlier, c.isolation_score, "
        "c.kmer_div_zscore, c.genome_size_zscore, c.centroid_distance, c.anomaly_score, "
        "COALESCE(g.genome_length, 0), COALESCE(g.n_contigs, 0), "
        "COALESCE(c.margin_to_threshold, 0.0), COALESCE(c.flag_reason, '') "
        "FROM contamination_candidates c "
        "LEFT JOIN genomes g ON c.accession = g.accession "
        "ORDER BY c.taxonomy, c.anomaly_score DESC");

    for (auto& row : *result) {
        out << row.GetValue<std::string>(0) << '\t'   // taxonomy
            << row.GetValue<std::string>(1) << '\t'   // accession
            << row.GetValue<bool>(2) << '\t'          // nn_outlier
            << row.GetValue<double>(3) << '\t'        // isolation_score
            << row.GetValue<double>(4) << '\t'        // kmer_div_zscore
            << row.GetValue<double>(5) << '\t'        // genome_size_zscore
            << row.GetValue<double>(6) << '\t'        // centroid_distance
            << row.GetValue<double>(7) << '\t'        // anomaly_score
            << row.GetValue<int64_t>(8) << '\t'       // genome_length_bp
            << row.GetValue<int32_t>(9) << '\t'       // n_contigs
            << row.GetValue<float>(10) << '\t'        // margin_to_threshold
            << row.GetValue<std::string>(11) << '\n'; // flag_reason
    }

    spdlog::info("Wrote contamination candidates to {}", path.string());
}

void ResultsWriter::write_all(db::DBManager& db) const {
    std::filesystem::create_directories(output_dir_);
    write_derep_genomes(db);
    write_stats(db);
    write_diversity_stats(db);
    write_results(db);
    write_failed(db);
    write_contamination(db);
}

} // namespace derep
