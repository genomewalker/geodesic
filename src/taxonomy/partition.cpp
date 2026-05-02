#include "partition.hpp"
#include <algorithm>
#include <fstream>
#include <map>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace derep::taxonomy {

namespace {

std::string extract_rank_key(std::string_view taxonomy, std::string_view rank_prefix) {
    std::string needle = ";";
    needle += rank_prefix;
    auto pos = taxonomy.find(needle);
    if (pos != std::string_view::npos) {
        auto start = pos + 1;
        auto end = taxonomy.find(';', start);
        return std::string(taxonomy.substr(start,
                           end == std::string_view::npos ? std::string_view::npos
                                                         : end - start));
    }
    if (taxonomy.starts_with(rank_prefix)) {
        auto end = taxonomy.find(';');
        return std::string(taxonomy.substr(0,
                           end == std::string_view::npos ? std::string_view::npos : end));
    }
    return "__unknown__";
}

} // namespace

size_t partition_accessions(const PartitionConfig& cfg) {
    if (cfg.n_parts <= 0)
        throw std::runtime_error("n_parts must be >= 1");
    if (!cfg.acc_taxonomy)
        throw std::runtime_error("partition_accessions: acc_taxonomy map required");

    const std::string rank_prefix = cfg.rank + "__";
    std::filesystem::create_directories(cfg.output_dir);

    std::ifstream fin(cfg.input_accessions);
    if (!fin) throw std::runtime_error("Cannot open: " + cfg.input_accessions.string());

    std::map<std::string, std::vector<std::string>> rank_accs;
    std::string line;
    size_t total = 0;
    while (std::getline(fin, line)) {
        auto s = line.find_first_not_of(" \t\r\n");
        if (s == std::string::npos) continue;
        auto e = line.find_last_not_of(" \t\r\n");
        auto acc = line.substr(s, e - s + 1);
        if (acc.empty() || acc[0] == '#') continue;

        std::string rank_key = "__unknown__";
        auto it = cfg.acc_taxonomy->find(acc);
        if (it != cfg.acc_taxonomy->end())
            rank_key = extract_rank_key(it->second, rank_prefix);

        rank_accs[rank_key].push_back(std::move(acc));
        ++total;
    }

    spdlog::info("taxonomy partition: {} accessions, {} unique {} groups",
                 total, rank_accs.size(), rank_prefix);

    std::vector<std::pair<std::string, std::vector<std::string>>> groups(
        rank_accs.begin(), rank_accs.end());
    std::sort(groups.begin(), groups.end(),
              [](const auto& a, const auto& b) {
                  return a.second.size() > b.second.size();
              });

    std::vector<std::vector<decltype(groups)::iterator>> bins(cfg.n_parts);
    std::vector<size_t> bin_counts(cfg.n_parts, 0);

    for (auto it = groups.begin(); it != groups.end(); ++it) {
        int target = static_cast<int>(
            std::min_element(bin_counts.begin(), bin_counts.end()) - bin_counts.begin());
        bins[target].push_back(it);
        bin_counts[target] += it->second.size();
    }

    for (int i = 0; i < cfg.n_parts; ++i) {
        auto out_path = cfg.output_dir / ("part_" + std::to_string(i) + ".txt");
        std::ofstream fout(out_path);
        if (!fout) throw std::runtime_error("Cannot write: " + out_path.string());
        std::sort(bins[i].begin(), bins[i].end(),
                  [](auto a, auto b) { return a->first < b->first; });
        for (auto it : bins[i])
            for (const auto& acc : it->second)
                fout << acc << '\n';
        spdlog::info("  part_{}: {} genomes, {} {} groups → {}",
                     i, bin_counts[i], bins[i].size(), rank_prefix,
                     out_path.string());
    }

    std::vector<size_t> sorted_counts = bin_counts;
    std::sort(sorted_counts.begin(), sorted_counts.end());
    spdlog::info("taxonomy partition: load balance min={} max={} ({}% imbalance)",
                 sorted_counts.front(), sorted_counts.back(),
                 sorted_counts.front() == 0 ? 0 :
                 100 * (sorted_counts.back() - sorted_counts.front()) / sorted_counts.front());

    return total;
}

} // namespace derep::taxonomy
