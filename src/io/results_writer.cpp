#include "io/results_writer.hpp"
#include "state/run_state.hpp"

#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <spdlog/spdlog.h>

namespace derep {

ResultsWriter::ResultsWriter(std::filesystem::path output_dir, std::string prefix)
    : output_dir_(std::move(output_dir)), prefix_(std::move(prefix)) {}

void ResultsWriter::write_derep_genomes(const RunState& state) const {
    auto path = output_dir_ / (prefix_ + "_derep_genomes.tsv");
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot open output file: " + path.string());

    out << "accession\ttaxonomy\trepresentative\tcluster_rep\tnn_dist\tsketch_fill\n";

    for (const auto& taxon : state.taxa()) {
        const std::string& taxonomy = taxon.result.taxonomy;
        std::unordered_set<std::string> rep_set(taxon.representatives.begin(),
                                                taxon.representatives.end());
        for (const auto& acc : taxon.all_accessions) {
            bool is_rep = rep_set.count(acc) > 0;
            std::string cluster_rep = is_rep ? acc : "";
            double nn_dist_val = 0.0;
            if (!is_rep) {
                auto mit = taxon.member_to_rep.find(acc);
                if (mit != taxon.member_to_rep.end()) cluster_rep = mit->second;
                auto dit = taxon.member_nn_dist.find(acc);
                if (dit != taxon.member_nn_dist.end()) nn_dist_val = dit->second;
            }
            float fill = 1.0f;
            auto fit = taxon.member_fill_ratio.find(acc);
            if (fit != taxon.member_fill_ratio.end()) fill = fit->second;
            out << acc << '\t' << taxonomy << '\t' << is_rep << '\t'
                << cluster_rep << '\t';
            if (is_rep)
                out << "0\t" << fill << '\n';
            else
                out << nn_dist_val << '\t' << fill << '\n';
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
           "n_outliers_fragmented\tn_outliers_size\tn_outliers_distance\n";

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

    out << "taxonomy\taccession\tcategory\tnn_outlier\tisolation_score\tkmer_div_zscore\t"
           "genome_size_zscore\tcentroid_distance\tanomaly_score\t"
           "genome_length_bp\tn_contigs\tmargin_to_threshold\tflag_reason\texcluded\n";

    for (const auto& taxon : state.taxa()) {
        for (const auto& c : taxon.outliers) {
            out << taxon.result.taxonomy << '\t'
                << c.accession << '\t'
                << c.category << '\t'
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

void ResultsWriter::write_all(const RunState& state) const {
    std::filesystem::create_directories(output_dir_);
    write_derep_genomes(state);
    write_stats(state);
    write_diversity_stats(state);
    write_results(state);
    write_failed(state);
    write_outliers(state);
}

// ── Resume support ────────────────────────────────────────────────────────────

std::filesystem::path ResultsWriter::ckpt_path(const std::string& ck_key,
                                                const std::string& suffix) const {
    return resume_dir() / (ck_key + "_" + suffix + ".tsv");
}

void ResultsWriter::cat_checkpoint_rows(std::ofstream& dest,
                                        const std::filesystem::path& src) {
    std::ifstream in(src);
    if (!in) return;
    std::string line;
    std::getline(in, line); // skip header
    while (std::getline(in, line))
        dest << line << '\n';
}

void ResultsWriter::write_checkpoint(const RunState& state,
                                     const std::string& ck_key) const {
    namespace fs = std::filesystem;
    fs::create_directories(resume_dir());

    auto open = [&](const std::string& suffix) -> std::ofstream {
        auto p = ckpt_path(ck_key, suffix);
        std::ofstream f(p);
        if (!f) throw std::runtime_error("Cannot write checkpoint: " + p.string());
        return f;
    };

    {
        auto f = open("derep_genomes");
        f << "accession\ttaxonomy\trepresentative\tcluster_rep\tnn_dist\tsketch_fill\n";
        for (const auto& taxon : state.taxa()) {
            const std::string& taxonomy = taxon.result.taxonomy;
            std::unordered_set<std::string> rep_set(taxon.representatives.begin(),
                                                    taxon.representatives.end());
            for (const auto& acc : taxon.all_accessions) {
                bool is_rep = rep_set.count(acc) > 0;
                std::string cluster_rep = is_rep ? acc : "";
                double nn_dist_val = 0.0;
                if (!is_rep) {
                    auto mit = taxon.member_to_rep.find(acc);
                    if (mit != taxon.member_to_rep.end()) cluster_rep = mit->second;
                    auto dit = taxon.member_nn_dist.find(acc);
                    if (dit != taxon.member_nn_dist.end()) nn_dist_val = dit->second;
                }
                float fill = 1.0f;
                auto fit = taxon.member_fill_ratio.find(acc);
                if (fit != taxon.member_fill_ratio.end()) fill = fit->second;
                f << acc << '\t' << taxonomy << '\t' << is_rep << '\t' << cluster_rep << '\t'
                  << (is_rep ? 0.0 : nn_dist_val) << '\t' << fill << '\n';
            }
        }
    }
    {
        auto f = open("results");
        f << "taxonomy\tmethod\tn_genomes\tn_genomes_derep\tcommunities\tweight\n";
        for (const auto& taxon : state.taxa()) {
            const auto& r = taxon.result;
            f << r.taxonomy << '\t' << r.method << '\t' << r.n_genomes << '\t'
              << r.n_representatives << '\t' << r.n_communities << '\t' << "NA\n";
        }
    }
    {
        auto f = open("stats");
        f << "taxonomy\tmethod\tn_input\tn_preflight_excluded\tn_quality_floor_excluded\t"
             "n_outliers_excluded\tn_outliers_retained\tn_failed\tn_embedded\t"
             "n_representatives\trep_fraction\t"
             "mst_p90_edge\tmst_true_max\tani_threshold_used\t"
             "n_outliers_fragmented\tn_outliers_size\tn_outliers_distance\n";
        for (const auto& taxon : state.taxa()) {
            if (taxon.result.status == TaxonStatus::FAILED) continue;
            const int n_reps   = static_cast<int>(taxon.representatives.size());
            const int n_failed = static_cast<int>(taxon.failed_genomes.size());
            const int n_embedded = taxon.n_input - taxon.n_preflight_excluded - n_failed;
            const double rep_frac = taxon.n_input > 0
                ? static_cast<double>(n_reps) / taxon.n_input : 0.0;
            int n_frag = 0, n_size = 0, n_nn = 0;
            for (const auto& o : taxon.outliers) {
                if (o.flag_reason.find("fragmented") != std::string::npos) ++n_frag;
                else if (o.flag_reason.find("size_outlier") != std::string::npos) ++n_size;
                else ++n_nn;
            }
            f << taxon.result.taxonomy << '\t' << taxon.result.method << '\t'
              << taxon.n_input << '\t' << taxon.n_preflight_excluded << '\t'
              << taxon.n_quality_floor_excluded << '\t' << taxon.n_outliers_excluded << '\t'
              << taxon.n_outliers_retained << '\t' << n_failed << '\t' << n_embedded << '\t'
              << n_reps << '\t' << rep_frac << '\t'
              << taxon.mst_p90_edge << '\t' << taxon.mst_true_max << '\t'
              << taxon.ani_threshold_used << '\t'
              << n_frag << '\t' << n_size << '\t' << n_nn << '\n';
        }
    }
    {
        auto f = open("diversity_stats");
        f << "taxonomy\tmethod\tn_genomes\tn_representatives\treduction_ratio\truntime_seconds\t"
             "coverage_mean_ani\tcoverage_min_ani\tcoverage_max_ani\t"
             "coverage_below_99\tcoverage_below_98\tcoverage_below_97\tcoverage_below_95\t"
             "diversity_mean_ani\tdiversity_min_ani\tdiversity_max_ani\t"
             "diversity_ani_range\tdiversity_n_pairs\tn_outliers_excluded\tn_outliers_retained\n";
        for (const auto& taxon : state.taxa()) {
            if (taxon.result.status == TaxonStatus::FAILED) continue;
            const auto& d = taxon.diversity_stats;
            if (d.taxonomy.empty()) continue;
            f << d.taxonomy << '\t' << d.method << '\t' << d.n_genomes << '\t'
              << d.n_representatives << '\t' << d.reduction_ratio << '\t' << d.runtime_seconds << '\t'
              << d.coverage_mean_ani << '\t' << d.coverage_min_ani << '\t' << d.coverage_max_ani << '\t'
              << d.coverage_below_99 << '\t' << d.coverage_below_98 << '\t'
              << d.coverage_below_97 << '\t' << d.coverage_below_95 << '\t'
              << d.diversity_mean_ani << '\t' << d.diversity_min_ani << '\t' << d.diversity_max_ani << '\t'
              << d.diversity_ani_range << '\t' << d.diversity_n_pairs << '\t'
              << d.n_outliers_excluded << '\t' << d.n_outliers_retained << '\n';
        }
    }
    {
        auto f = open("failed");
        f << "accession\ttaxonomy\tfile\treason\n";
        for (const auto& taxon : state.taxa()) {
            for (const auto& fail : taxon.failed_genomes) {
                f << fail.accession << '\t' << fail.taxonomy << '\t' << fail.file << '\t'
                  << (fail.reason.empty() ? "NA" : fail.reason) << '\n';
            }
        }
    }
    {
        auto f = open("outliers");
        f << "taxonomy\taccession\tcategory\tnn_outlier\tisolation_score\tkmer_div_zscore\t"
             "genome_size_zscore\tcentroid_distance\tanomaly_score\t"
             "genome_length_bp\tn_contigs\tmargin_to_threshold\tflag_reason\texcluded\n";
        for (const auto& taxon : state.taxa()) {
            for (const auto& c : taxon.outliers) {
                f << taxon.result.taxonomy << '\t' << c.accession << '\t' << c.category << '\t'
                  << c.nn_outlier << '\t' << c.isolation_score << '\t' << c.kmer_div_zscore << '\t'
                  << c.genome_size_zscore << '\t' << c.centroid_distance << '\t' << c.anomaly_score << '\t'
                  << c.genome_length_bp << '\t' << c.n_contigs << '\t'
                  << c.margin_to_threshold << '\t' << c.flag_reason << '\t' << c.excluded << '\n';
            }
        }
    }

    // Sentinel written last — a partial checkpoint is never treated as complete.
    {
        auto done = resume_dir() / (ck_key + ".done");
        std::ofstream f(done);
        if (!f) throw std::runtime_error("Cannot write checkpoint sentinel: " + done.string());
        f << ck_key << '\n';
    }
    spdlog::info("RESUME checkpoint: {} written ({} taxa)", ck_key, state.taxa().size());
}

std::unordered_set<std::string> ResultsWriter::scan_done_keys() const {
    namespace fs = std::filesystem;
    std::unordered_set<std::string> done;
    if (!fs::exists(resume_dir())) return done;
    for (const auto& entry : fs::directory_iterator(resume_dir())) {
        const auto name = entry.path().filename().string();
        if (name.size() > 5 && name.substr(name.size() - 5) == ".done")
            done.insert(name.substr(0, name.size() - 5));
    }
    return done;
}

std::unordered_set<std::string> ResultsWriter::load_resume() const {
    namespace fs = std::filesystem;
    const auto done_keys = scan_done_keys();
    if (done_keys.empty()) return {};

    fs::create_directories(output_dir_);

    auto merge_file = [&](const std::string& suffix, const std::string& header) {
        auto final_path = output_dir_ / (prefix_ + "_" + suffix + ".tsv");
        std::ofstream out(final_path);
        if (!out) throw std::runtime_error("Cannot open " + final_path.string());
        out << header << '\n';
        for (const auto& key : done_keys)
            cat_checkpoint_rows(out, ckpt_path(key, suffix));
    };

    merge_file("derep_genomes",
               "accession\ttaxonomy\trepresentative\tcluster_rep\tnn_dist\tsketch_fill");
    merge_file("results",
               "taxonomy\tmethod\tn_genomes\tn_genomes_derep\tcommunities\tweight");
    merge_file("stats",
               "taxonomy\tmethod\tn_input\tn_preflight_excluded\tn_quality_floor_excluded\t"
               "n_outliers_excluded\tn_outliers_retained\tn_failed\tn_embedded\t"
               "n_representatives\trep_fraction\t"
               "mst_p90_edge\tmst_true_max\tani_threshold_used\t"
               "n_outliers_fragmented\tn_outliers_size\tn_outliers_distance");
    merge_file("diversity_stats",
               "taxonomy\tmethod\tn_genomes\tn_representatives\treduction_ratio\truntime_seconds\t"
               "coverage_mean_ani\tcoverage_min_ani\tcoverage_max_ani\t"
               "coverage_below_99\tcoverage_below_98\tcoverage_below_97\tcoverage_below_95\t"
               "diversity_mean_ani\tdiversity_min_ani\tdiversity_max_ani\t"
               "diversity_ani_range\tdiversity_n_pairs\tn_outliers_excluded\tn_outliers_retained");
    merge_file("failed",   "accession\ttaxonomy\tfile\treason");
    merge_file("outliers",
               "taxonomy\taccession\tcategory\tnn_outlier\tisolation_score\tkmer_div_zscore\t"
               "genome_size_zscore\tcentroid_distance\tanomaly_score\t"
               "genome_length_bp\tn_contigs\tmargin_to_threshold\tflag_reason\texcluded");

    std::unordered_set<std::string> completed;
    for (const auto& key : done_keys) {
        std::ifstream in(ckpt_path(key, "results"));
        if (!in) continue;
        std::string line;
        std::getline(in, line); // skip header
        while (std::getline(in, line)) {
            auto tab = line.find('\t');
            if (tab != std::string::npos)
                completed.insert(line.substr(0, tab));
        }
    }
    spdlog::info("RESUME: {} taxa already done ({} checkpoints)", completed.size(), done_keys.size());
    return completed;
}

void ResultsWriter::write_all_append(const RunState& state) const {
    auto open_append = [&](const std::string& suffix) -> std::ofstream {
        auto p = output_dir_ / (prefix_ + "_" + suffix + ".tsv");
        std::ofstream f(p, std::ios::app);
        if (!f) throw std::runtime_error("Cannot open for append: " + p.string());
        return f;
    };

    {
        auto f = open_append("derep_genomes");
        for (const auto& taxon : state.taxa()) {
            const std::string& taxonomy = taxon.result.taxonomy;
            std::unordered_set<std::string> rep_set(taxon.representatives.begin(),
                                                    taxon.representatives.end());
            for (const auto& acc : taxon.all_accessions) {
                bool is_rep = rep_set.count(acc) > 0;
                std::string cluster_rep = is_rep ? acc : "";
                double nn_dist_val = 0.0;
                if (!is_rep) {
                    auto mit = taxon.member_to_rep.find(acc);
                    if (mit != taxon.member_to_rep.end()) cluster_rep = mit->second;
                    auto dit = taxon.member_nn_dist.find(acc);
                    if (dit != taxon.member_nn_dist.end()) nn_dist_val = dit->second;
                }
                float fill = 1.0f;
                auto fit = taxon.member_fill_ratio.find(acc);
                if (fit != taxon.member_fill_ratio.end()) fill = fit->second;
                f << acc << '\t' << taxonomy << '\t' << is_rep << '\t' << cluster_rep << '\t'
                  << (is_rep ? 0.0 : nn_dist_val) << '\t' << fill << '\n';
            }
        }
    }
    {
        auto f = open_append("results");
        for (const auto& taxon : state.taxa()) {
            const auto& r = taxon.result;
            f << r.taxonomy << '\t' << r.method << '\t' << r.n_genomes << '\t'
              << r.n_representatives << '\t' << r.n_communities << '\t' << "NA\n";
        }
    }
    {
        auto f = open_append("stats");
        for (const auto& taxon : state.taxa()) {
            if (taxon.result.status == TaxonStatus::FAILED) continue;
            const int n_reps   = static_cast<int>(taxon.representatives.size());
            const int n_failed = static_cast<int>(taxon.failed_genomes.size());
            const int n_embedded = taxon.n_input - taxon.n_preflight_excluded - n_failed;
            const double rep_frac = taxon.n_input > 0
                ? static_cast<double>(n_reps) / taxon.n_input : 0.0;
            int n_frag = 0, n_size = 0, n_nn = 0;
            for (const auto& o : taxon.outliers) {
                if (o.flag_reason.find("fragmented") != std::string::npos) ++n_frag;
                else if (o.flag_reason.find("size_outlier") != std::string::npos) ++n_size;
                else ++n_nn;
            }
            f << taxon.result.taxonomy << '\t' << taxon.result.method << '\t'
              << taxon.n_input << '\t' << taxon.n_preflight_excluded << '\t'
              << taxon.n_quality_floor_excluded << '\t' << taxon.n_outliers_excluded << '\t'
              << taxon.n_outliers_retained << '\t' << n_failed << '\t' << n_embedded << '\t'
              << n_reps << '\t' << rep_frac << '\t'
              << taxon.mst_p90_edge << '\t' << taxon.mst_true_max << '\t'
              << taxon.ani_threshold_used << '\t'
              << n_frag << '\t' << n_size << '\t' << n_nn << '\n';
        }
    }
    {
        auto f = open_append("diversity_stats");
        for (const auto& taxon : state.taxa()) {
            if (taxon.result.status == TaxonStatus::FAILED) continue;
            const auto& d = taxon.diversity_stats;
            if (d.taxonomy.empty()) continue;
            f << d.taxonomy << '\t' << d.method << '\t' << d.n_genomes << '\t'
              << d.n_representatives << '\t' << d.reduction_ratio << '\t' << d.runtime_seconds << '\t'
              << d.coverage_mean_ani << '\t' << d.coverage_min_ani << '\t' << d.coverage_max_ani << '\t'
              << d.coverage_below_99 << '\t' << d.coverage_below_98 << '\t'
              << d.coverage_below_97 << '\t' << d.coverage_below_95 << '\t'
              << d.diversity_mean_ani << '\t' << d.diversity_min_ani << '\t' << d.diversity_max_ani << '\t'
              << d.diversity_ani_range << '\t' << d.diversity_n_pairs << '\t'
              << d.n_outliers_excluded << '\t' << d.n_outliers_retained << '\n';
        }
    }
    {
        auto f = open_append("failed");
        for (const auto& taxon : state.taxa()) {
            for (const auto& fail : taxon.failed_genomes) {
                f << fail.accession << '\t' << fail.taxonomy << '\t' << fail.file << '\t'
                  << (fail.reason.empty() ? "NA" : fail.reason) << '\n';
            }
        }
    }
    {
        auto f = open_append("outliers");
        for (const auto& taxon : state.taxa()) {
            for (const auto& c : taxon.outliers) {
                f << taxon.result.taxonomy << '\t' << c.accession << '\t' << c.category << '\t'
                  << c.nn_outlier << '\t' << c.isolation_score << '\t' << c.kmer_div_zscore << '\t'
                  << c.genome_size_zscore << '\t' << c.centroid_distance << '\t' << c.anomaly_score << '\t'
                  << c.genome_length_bp << '\t' << c.n_contigs << '\t'
                  << c.margin_to_threshold << '\t' << c.flag_reason << '\t' << c.excluded << '\n';
            }
        }
    }
}

} // namespace derep
