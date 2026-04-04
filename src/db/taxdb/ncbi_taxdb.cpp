#include "ncbi_taxdb.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_set>

#include <spdlog/spdlog.h>

namespace derep {

namespace {

// Canonical 10-rank slots in order.
static constexpr std::array<std::string_view, 10> kRanks = {
    "d__", "l__", "k__", "p__", "c__", "o__", "f__", "g__", "s__", "S__"
};

// NCBI rank name → canonical prefix (only the directly mappable ranks)
static const std::unordered_map<std::string, std::string> kNcbiToPrefix = {
    {"superkingdom", "d__"},
    {"kingdom",      "k__"},
    {"phylum",       "p__"},
    {"class",        "c__"},
    {"order",        "o__"},
    {"family",       "f__"},
    {"genus",        "g__"},
    {"species",      "s__"},
    {"subspecies",   "S__"},
};

// Clade names that serve as the l__ (lineage) slot for eukaryotes.
// These appear as NCBI "clade" or "no rank" nodes just above kingdom.
static const std::array<std::string_view, 8> kLineageClades = {
    "Opisthokonta",   // fungi, invertebrates, vertebrates
    "Viridiplantae",  // plants
    "SAR",            // stramenopiles, alveolates, rhizaria
    "Discoba",        // excavates
    "Amoebozoa",
    "Haptista",
    "Cryptista",
    "Rhodophyta",     // red algae (no kingdom rank in NCBI)
};

bool is_lineage_clade(const std::string& name) {
    for (auto c : kLineageClades)
        if (name == c) return true;
    return false;
}

std::string trim(const std::string& s) {
    auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return {};
    auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

// Parse a single field from a pipe-delimited nodes.dmp / names.dmp line.
// NCBI dmp format: field1\t|\tfield2\t|\t...
std::vector<std::string> split_dmp(const std::string& line) {
    std::vector<std::string> fields;
    std::size_t pos = 0;
    while (true) {
        auto sep = line.find("\t|\t", pos);
        if (sep == std::string::npos) {
            fields.push_back(trim(line.substr(pos)));
            break;
        }
        fields.push_back(trim(line.substr(pos, sep - pos)));
        pos = sep + 3;
    }
    return fields;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Static utilities
// ---------------------------------------------------------------------------

std::optional<std::chrono::system_clock::time_point>
NcbiTaxdb::dump_timestamp(const fs::path& dir) {
    auto ts_file = dir / ".timestamp";
    std::ifstream in(ts_file);
    if (!in) return std::nullopt;
    std::time_t t = 0;
    in >> t;
    if (!in) return std::nullopt;
    return std::chrono::system_clock::from_time_t(t);
}

void NcbiTaxdb::ensure_fresh(const fs::path& dir, int max_age_days) {
    auto nodes = dir / "nodes.dmp";
    auto names = dir / "names.dmp";
    auto ts    = dir / ".timestamp";

    bool needs_download = !fs::exists(nodes) || !fs::exists(names);
    if (!needs_download) {
        auto opt_ts = dump_timestamp(dir);
        if (!opt_ts) {
            needs_download = true;
        } else {
            auto age = std::chrono::duration_cast<std::chrono::hours>(
                std::chrono::system_clock::now() - *opt_ts).count();
            needs_download = (age > max_age_days * 24);
        }
    }

    if (!needs_download) {
        spdlog::info("NCBI taxdump is up to date in {}", dir.string());
        return;
    }

    spdlog::info("Downloading NCBI taxdump → {}", dir.string());
    fs::create_directories(dir);

    auto archive = dir / "new_taxdump.tar.gz";
    std::string cmd = "curl -fsSL --retry 5 --retry-wait 10 -o " +
                      archive.string() + " " + kDownloadUrl;
    if (std::system(cmd.c_str()) != 0)
        throw std::runtime_error("Failed to download NCBI taxdump from " +
                                  std::string(kDownloadUrl));

    cmd = "tar -xzf " + archive.string() + " -C " + dir.string();
    if (std::system(cmd.c_str()) != 0)
        throw std::runtime_error("Failed to extract " + archive.string());

    fs::remove(archive);

    // Write timestamp
    std::ofstream out(ts);
    out << std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    spdlog::info("NCBI taxdump downloaded and extracted ({} nodes)",
                 fs::exists(nodes) ? "ok" : "MISSING");
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------

NcbiTaxdb NcbiTaxdb::load(const fs::path& dir) {
    NcbiTaxdb db;

    // Parse nodes.dmp: taxid | parent_taxid | rank | ...
    {
        std::ifstream in(dir / "nodes.dmp");
        if (!in) throw std::runtime_error("Cannot open nodes.dmp in " + dir.string());
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            auto f = split_dmp(line);
            if (f.size() < 3) continue;
            int taxid  = std::stoi(f[0]);
            int parent = std::stoi(f[1]);
            db.nodes_[taxid] = {parent, f[2]};
        }
    }

    // Parse names.dmp: taxid | name | unique_name | name_class | ...
    {
        std::ifstream in(dir / "names.dmp");
        if (!in) throw std::runtime_error("Cannot open names.dmp in " + dir.string());
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            auto f = split_dmp(line);
            if (f.size() < 4) continue;
            if (f[3] != "scientific name") continue;
            int taxid = std::stoi(f[0]);
            db.name_[taxid] = f[1];
        }
    }

    spdlog::info("NcbiTaxdb loaded: {} nodes", db.nodes_.size());
    return db;
}

// ---------------------------------------------------------------------------
// Accession map
// ---------------------------------------------------------------------------

void NcbiTaxdb::load_accession_map(const fs::path& assembly_summary) {
    std::ifstream in(assembly_summary);
    if (!in) throw std::runtime_error("Cannot open assembly summary: " +
                                       assembly_summary.string());
    std::string line;
    // Skip comment headers (lines starting with #)
    while (std::getline(in, line)) {
        if (!line.empty() && line[0] != '#') break;
    }
    // First non-comment line is the header; parse column indices
    std::vector<std::string> headers;
    {
        std::istringstream ss(line);
        std::string col;
        while (std::getline(ss, col, '\t')) headers.push_back(col);
    }
    auto find_col = [&](const std::string& name) -> int {
        for (int i = 0; i < (int)headers.size(); ++i)
            if (headers[i] == name) return i;
        return -1;
    };
    int col_acc   = find_col("# assembly_accession");
    int col_taxid = find_col("taxid");
    if (col_acc < 0)   col_acc   = find_col("assembly_accession");
    if (col_taxid < 0) col_taxid = 5; // default column in NCBI assembly summary

    while (std::getline(in, line)) {
        std::istringstream ss(line);
        std::string field;
        std::vector<std::string> fields;
        while (std::getline(ss, field, '\t')) fields.push_back(field);
        if ((int)fields.size() <= std::max(col_acc, col_taxid)) continue;
        const std::string& acc = fields[col_acc];
        // Strip version suffix for the key
        std::string stem = acc;
        auto dot = stem.rfind('.');
        if (dot != std::string::npos) stem = stem.substr(0, dot);
        try {
            int tid = std::stoi(fields[col_taxid]);
            acc2tid_[stem] = tid;
        } catch (...) {}
    }
    spdlog::info("NcbiTaxdb: loaded {} accession→taxid mappings", acc2tid_.size());
}

int NcbiTaxdb::taxid_for_accession(const std::string& accession) const {
    // Try with and without version suffix
    auto it = acc2tid_.find(accession);
    if (it != acc2tid_.end()) return it->second;
    auto dot = accession.rfind('.');
    if (dot != std::string::npos) {
        it = acc2tid_.find(accession.substr(0, dot));
        if (it != acc2tid_.end()) return it->second;
    }
    return -1;
}

// ---------------------------------------------------------------------------
// Lineage builder
// ---------------------------------------------------------------------------

std::vector<std::pair<std::string, std::string>>
NcbiTaxdb::lineage(int taxid) const {
    std::vector<std::pair<std::string, std::string>> result;
    std::unordered_set<int> visited;
    int cur = taxid;
    while (cur != 1 && cur != 0 && !visited.count(cur)) {
        visited.insert(cur);
        auto nit = nodes_.find(cur);
        if (nit == nodes_.end()) break;
        auto nme = name_.find(cur);
        result.push_back({nit->second.rank,
                          nme != name_.end() ? nme->second : ""});
        cur = nit->second.parent_taxid;
    }
    std::reverse(result.begin(), result.end()); // root → leaf
    return result;
}

std::string NcbiTaxdb::lineage_to_10rank(
    const std::vector<std::pair<std::string, std::string>>& lin,
    const std::string& accession) const {

    // Collect mapped ranks
    std::unordered_map<std::string, std::string> rank_map;

    for (const auto& [ncbi_rank, name] : lin) {
        auto it = kNcbiToPrefix.find(ncbi_rank);
        if (it != kNcbiToPrefix.end()) {
            rank_map[it->second] = it->second + name;
        }
        // Detect l__ from known lineage clade names
        if (!rank_map.count("l__") && is_lineage_clade(name)) {
            rank_map["l__"] = "l__" + name;
        }
    }

    // Viruses: no kingdom/lineage in NCBI — derive l__ and k__ from d__
    // Bacteria/Archaea should not reach here, but handle gracefully
    auto propagate = [&](const std::string& child, const std::string& parent) {
        if (!rank_map.count(child) || rank_map[child] == child) {
            if (rank_map.count(parent))
                rank_map[child] = child + rank_map[parent].substr(3);
        }
    };
    propagate("l__", "d__");
    propagate("k__", "l__");
    propagate("p__", "k__");
    propagate("c__", "p__");
    propagate("o__", "c__");
    propagate("f__", "o__");
    propagate("g__", "f__");

    // s__: if unresolved, use "genus unclassified" convention from taxdb-integration
    if (!rank_map.count("s__") || rank_map["s__"] == "s__") {
        if (rank_map.count("g__"))
            rank_map["s__"] = "s__" + rank_map["g__"].substr(3) + " unclassified";
        else
            rank_map["s__"] = "s__" + accession;
    }

    // S__: always from s__
    rank_map["S__"] = "S__" + rank_map["s__"].substr(3);

    // Reconstruct
    std::string result;
    result.reserve(200);
    for (auto pfx : kRanks) {
        if (!result.empty()) result += ';';
        auto it = rank_map.find(std::string(pfx));
        result += (it != rank_map.end()) ? it->second : std::string(pfx);
    }
    return result;
}

std::string NcbiTaxdb::taxonomy_for_taxid(int taxid,
                                           const std::string& accession) const {
    if (taxid <= 0) return {};
    auto lin = lineage(taxid);
    if (lin.empty()) return {};
    return lineage_to_10rank(lin, accession);
}

} // namespace derep
