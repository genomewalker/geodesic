#include "normalize.hpp"
#include "db/taxdb/ncbi_taxdb.hpp"
#include <unordered_map>

namespace derep::taxonomy {

std::string species_stem(std::string_view acc) {
    if (acc.starts_with("RS_") || acc.starts_with("GB_"))
        acc = acc.substr(3);
    if (acc.starts_with("GCF_") || acc.starts_with("GCA_")) {
        auto dot = acc.rfind('.');
        if (dot != std::string_view::npos)
            return std::string(acc.substr(0, dot));
    }
    return std::string(acc);
}

std::string normalize_taxonomy(const std::string& taxonomy,
                               const std::string& accession,
                               const NcbiTaxdb* ncbi) {
    const std::string stem = species_stem(accession);

    // Non-GTDB: synthesise a unique per-accession taxonomy
    if (taxonomy.size() < 3 || taxonomy.substr(0, 3) != "d__") {
        return "d__Unclassified;l__Unclassified;k__Unclassified;"
               "p__Unclassified;c__Unclassified;o__Unclassified;"
               "f__Unclassified;g__Unclassified;s__" + stem +
               ";S__" + stem;
    }

    // Non-prokaryote domains: delegate to NcbiTaxdb when available
    {
        const auto d = taxonomy.substr(0, taxonomy.find(';'));
        const bool is_prokaryote = (d == "d__Bacteria" || d == "d__Archaea");
        if (!is_prokaryote && ncbi && !ncbi->empty()) {
            int tid = ncbi->taxid_for_accession(stem);
            if (tid > 0) {
                auto result = ncbi->taxonomy_for_taxid(tid, stem);
                if (!result.empty()) return result;
            }
        }
    }

    // Parse existing rank tokens
    std::unordered_map<std::string, std::string> rank_map;
    std::string_view sv(taxonomy);
    while (!sv.empty()) {
        auto sep = sv.find(';');
        std::string_view token = sv.substr(0, sep);
        if (token.size() >= 3 && token[1] == '_' && token[2] == '_')
            rank_map.emplace(std::string(token.substr(0, 3)), std::string(token));
        sv = (sep == std::string_view::npos) ? "" : sv.substr(sep + 1);
    }

    // Propagate parent value into missing/empty child ranks
    auto propagate = [&](std::string_view child_pfx, std::string_view parent_pfx) {
        std::string cpfx(child_pfx), ppfx(parent_pfx);
        if (!rank_map.count(cpfx) || rank_map[cpfx] == cpfx) {
            if (rank_map.count(ppfx))
                rank_map[cpfx] = cpfx + rank_map[ppfx].substr(3);
        }
    };
    propagate("l__", "d__");
    propagate("k__", "l__");
    propagate("p__", "k__");
    propagate("c__", "p__");
    propagate("o__", "c__");
    propagate("f__", "o__");
    propagate("g__", "f__");

    // s__: empty stub → accession stem
    {
        auto it = rank_map.find("s__");
        if (it == rank_map.end() || it->second == "s__")
            rank_map["s__"] = "s__" + stem;
    }

    // S__ always from s__
    rank_map["S__"] = "S__" + rank_map["s__"].substr(3);

    std::string result;
    result.reserve(taxonomy.size() + 30);
    for (auto rank : kRanks) {
        if (!result.empty()) result += ';';
        result += rank_map[std::string(rank)];
    }
    return result;
}

bool has_accession_species(const std::string& taxonomy, const std::string& accession) {
    const std::string needle = ";s__" + species_stem(accession) + ";";
    return taxonomy.find(needle) != std::string::npos;
}

} // namespace derep::taxonomy
