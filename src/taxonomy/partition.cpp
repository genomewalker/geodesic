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

// Extract the rank key (e.g., "g__Escherichia") from a taxonomy string.
// rank_prefix is "g__" for genus, "f__" for family, etc.
std::string extract_rank_key(std::string_view taxonomy, std::string_view rank_prefix) {
    // Search for ";g__" or start-of-string "g__"
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
    // Check if it starts with the rank_prefix directly
    if (taxonomy.starts_with(rank_prefix)) {
        auto end = taxonomy.find(';');
        return std::string(taxonomy.substr(0,
                           end == std::string_view::npos ? std::string_view::npos : end));
    }
    return "__unknown__";
}

} // namespace

size_t partition_tsv(const PartitionConfig& cfg) {
    if (cfg.n_parts <= 0)
        throw std::runtime_error("n_parts must be >= 1");

    const std::string rank_prefix = cfg.rank + "__";
    std::filesystem::create_directories(cfg.output_dir);

    // Read header + all rows
    std::ifstream fin(cfg.input_tsv);
    if (!fin) throw std::runtime_error("Cannot open: " + cfg.input_tsv.string());

    std::string header;
    std::getline(fin, header);

    // Group rows by rank key
    // Using map (sorted) so output is deterministic
    std::map<std::string, std::vector<std::string>> rank_rows;
    std::string line;
    size_t total = 0;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        // taxonomy is column index 1
        std::string_view sv(line);
        auto t1 = sv.find('\t');
        if (t1 == std::string_view::npos) { rank_rows["__unknown__"].push_back(line); ++total; continue; }
        auto t2 = sv.find('\t', t1 + 1);
        std::string_view tax = (t2 == std::string_view::npos)
                               ? sv.substr(t1 + 1)
                               : sv.substr(t1 + 1, t2 - t1 - 1);
        rank_rows[extract_rank_key(tax, rank_prefix)].push_back(std::move(line));
        ++total;
    }
    fin.close();

    spdlog::info("taxonomy partition: {} genomes, {} unique {} groups",
                 total, rank_rows.size(), rank_prefix);

    // LPT: sort rank groups descending by size, assign to least-loaded bin
    std::vector<std::pair<std::string, std::vector<std::string>>> groups(
        rank_rows.begin(), rank_rows.end());
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

    // Write part TSVs (sort by taxonomy key within each part for better compression)
    for (int i = 0; i < cfg.n_parts; ++i) {
        auto out_path = cfg.output_dir / ("part_" + std::to_string(i) + ".tsv");
        std::ofstream fout(out_path);
        if (!fout) throw std::runtime_error("Cannot write: " + out_path.string());
        fout << header << '\n';
        // Sort bins[i] by rank key (already sorted since groups came from a map)
        std::sort(bins[i].begin(), bins[i].end(),
                  [](auto a, auto b) { return a->first < b->first; });
        for (auto it : bins[i])
            for (const auto& row : it->second)
                fout << row << '\n';
        spdlog::info("  part_{}: {} genomes, {} {} groups → {}",
                     i, bin_counts[i], bins[i].size(), rank_prefix,
                     out_path.string());
    }

    // Log load balance
    std::vector<size_t> sorted_counts = bin_counts;
    std::sort(sorted_counts.begin(), sorted_counts.end());
    spdlog::info("taxonomy partition: load balance min={} max={} ({}% imbalance)",
                 sorted_counts.front(), sorted_counts.back(),
                 sorted_counts.front() == 0 ? 0 :
                 100 * (sorted_counts.back() - sorted_counts.front()) / sorted_counts.front());

    return total;
}

} // namespace derep::taxonomy
