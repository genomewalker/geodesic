#include "tsv_reader.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <unordered_set>

#include <spdlog/spdlog.h>

namespace derep {

namespace {

std::string trim(std::string_view s) {
    auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string_view::npos) return {};
    auto end = s.find_last_not_of(" \t\r\n");
    return std::string(s.substr(start, end - start + 1));
}

std::vector<std::string> split_line(std::string_view line, char delim) {
    std::vector<std::string> fields;
    std::size_t pos = 0;
    while (pos <= line.size()) {
        auto next = line.find(delim, pos);
        if (next == std::string_view::npos) {
            fields.emplace_back(line.substr(pos));
            break;
        }
        fields.emplace_back(line.substr(pos, next - pos));
        pos = next + 1;
    }
    return fields;
}

int find_col(const std::vector<std::string>& headers,
             std::initializer_list<std::string_view> names) {
    for (int i = 0; i < static_cast<int>(headers.size()); ++i) {
        std::string lower = headers[i];
        std::transform(lower.begin(), lower.end(), lower.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        for (auto name : names) {
            std::string name_lower(name);
            std::transform(name_lower.begin(), name_lower.end(),
                           name_lower.begin(),
                           [](unsigned char c) { return std::tolower(c); });
            if (lower == name_lower) return i;
        }
    }
    return -1;
}

bool is_comment_or_empty(std::string_view line, const TsvReadOptions& opts) {
    if (line.empty()) return true;
    auto first = line.find_first_not_of(" \t\r\n");
    if (first == std::string_view::npos) return true;
    if (opts.allow_comments && line[first] == opts.comment_prefix) return true;
    return false;
}

} // anonymous namespace

std::string canonical_accession(std::string_view acc) {
    if (acc.starts_with("RS_") || acc.starts_with("GB_")) {
        return std::string(acc.substr(3));
    }
    return std::string(acc);
}

std::vector<GenomeRow> read_genomes_tsv(const std::filesystem::path& path,
                                        const TsvReadOptions& opts) {
    std::ifstream in(path);
    if (!in) {
        throw TsvParseError("Cannot open genome TSV: " + path.string());
    }

    std::string line;
    std::size_t line_num = 0;

    int col_acc = -1, col_tax = -1, col_file = -1;

    if (opts.has_header) {
        if (!std::getline(in, line)) {
            throw TsvParseError("Empty genome TSV: " + path.string());
        }
        ++line_num;
        auto headers = split_line(line, opts.delimiter);
        if (opts.trim_fields) {
            for (auto& h : headers) h = trim(h);
        }

        col_acc = find_col(headers, {"accession", "acc"});
        col_tax = find_col(headers, {"taxonomy", "lineage"});
        col_file = find_col(headers, {"file", "filepath", "path"});

        if (col_acc < 0) col_acc = 0;
        if (col_tax < 0) col_tax = 1;
        if (col_file < 0) col_file = 2;
    } else {
        col_acc = 0;
        col_tax = 1;
        col_file = 2;
    }

    std::vector<GenomeRow> rows;
    std::unordered_set<std::string> seen;

    while (std::getline(in, line)) {
        ++line_num;
        if (is_comment_or_empty(line, opts)) continue;

        auto fields = split_line(line, opts.delimiter);
        if (opts.trim_fields) {
            for (auto& f : fields) f = trim(f);
        }

        int max_col = std::max({col_acc, col_tax, col_file});
        if (static_cast<int>(fields.size()) <= max_col) {
            spdlog::warn("{}:{}: expected at least {} fields, got {}",
                         path.string(), line_num, max_col + 1, fields.size());
            continue;
        }

        GenomeRow row;
        row.accession = std::move(fields[col_acc]);
        row.taxonomy = std::move(fields[col_tax]);
        row.file_path = std::move(fields[col_file]);

        if (opts.strict) {
            if (!seen.insert(row.accession).second) {
                throw TsvParseError("Duplicate accession '" + row.accession +
                                    "' at line " + std::to_string(line_num) +
                                    " in " + path.string());
            }
        }

        // Skip file existence check - too slow on NFS (5.2M stat() calls)
        // Files will be validated when actually accessed
        rows.push_back(std::move(row));
    }

    spdlog::info("Read {} genomes from {}", rows.size(), path.string());
    return rows;
}

std::unordered_map<std::string, CheckM2Quality> read_checkm2_tsv(
    const std::filesystem::path& path, const TsvReadOptions& opts) {
    std::ifstream in(path);
    if (!in) {
        throw TsvParseError("Cannot open CheckM2 TSV: " + path.string());
    }

    std::string line;
    std::size_t line_num = 0;

    int col_name = -1, col_comp = -1, col_cont = -1;

    if (opts.has_header) {
        if (!std::getline(in, line)) {
            throw TsvParseError("Empty CheckM2 TSV: " + path.string());
        }
        ++line_num;
        auto headers = split_line(line, opts.delimiter);
        if (opts.trim_fields) {
            for (auto& h : headers) h = trim(h);
        }

        col_name = find_col(headers, {"name", "Name"});
        col_comp = find_col(headers, {"completeness", "Completeness"});
        col_cont = find_col(headers, {"contamination", "Contamination"});

        if (col_name < 0 || col_comp < 0 || col_cont < 0) {
            throw TsvParseError(
                "CheckM2 TSV missing required columns (Name, Completeness, "
                "Contamination): " +
                path.string());
        }
    } else {
        col_name = 0;
        col_comp = 1;
        col_cont = 2;
    }

    std::unordered_map<std::string, CheckM2Quality> result;

    while (std::getline(in, line)) {
        ++line_num;
        if (is_comment_or_empty(line, opts)) continue;

        auto fields = split_line(line, opts.delimiter);
        if (opts.trim_fields) {
            for (auto& f : fields) f = trim(f);
        }

        int max_col = std::max({col_name, col_comp, col_cont});
        if (static_cast<int>(fields.size()) <= max_col) {
            spdlog::warn("{}:{}: expected at least {} fields, got {}",
                         path.string(), line_num, max_col + 1, fields.size());
            continue;
        }

        auto acc = canonical_accession(fields[col_name]);
        CheckM2Quality q;
        try {
            q.completeness = std::stod(fields[col_comp]);
            q.contamination = std::stod(fields[col_cont]);
        } catch (const std::exception& e) {
            spdlog::warn("{}:{}: failed to parse numeric fields: {}",
                         path.string(), line_num, e.what());
            continue;
        }

        if (q.completeness > 100.0) {
            spdlog::warn("{}:{}: completeness {} > 100 for '{}'",
                         path.string(), line_num, q.completeness, acc);
        }
        if (q.contamination < 0.0) {
            spdlog::warn("{}:{}: contamination {} < 0 for '{}'",
                         path.string(), line_num, q.contamination, acc);
        }

        if (opts.strict) {
            if (result.find(acc) != result.end()) {
                throw TsvParseError("Duplicate accession '" + acc +
                                    "' at line " + std::to_string(line_num) +
                                    " in " + path.string());
            }
        }

        result.insert_or_assign(std::move(acc), q);
    }

    spdlog::info("Read {} CheckM2 entries from {}", result.size(),
                 path.string());
    return result;
}

std::unordered_map<std::string, std::string> read_fixed_taxa_tsv(
    const std::filesystem::path& path, const TsvReadOptions& opts) {
    std::ifstream in(path);
    if (!in) {
        throw TsvParseError("Cannot open fixed taxa TSV: " + path.string());
    }

    std::string line;
    std::size_t line_num = 0;

    if (opts.has_header) {
        if (!std::getline(in, line)) {
            throw TsvParseError("Empty fixed taxa TSV: " + path.string());
        }
        ++line_num;
    }

    std::unordered_map<std::string, std::string> result;

    while (std::getline(in, line)) {
        ++line_num;
        if (is_comment_or_empty(line, opts)) continue;

        auto fields = split_line(line, opts.delimiter);
        if (opts.trim_fields) {
            for (auto& f : fields) f = trim(f);
        }

        if (fields.size() < 2) {
            spdlog::warn("{}:{}: expected at least 2 fields, got {}",
                         path.string(), line_num, fields.size());
            continue;
        }

        auto& taxonomy = fields[0];
        auto& accession = fields[1];

        if (opts.strict) {
            if (result.find(taxonomy) != result.end()) {
                throw TsvParseError("Duplicate taxonomy '" + taxonomy +
                                    "' at line " + std::to_string(line_num) +
                                    " in " + path.string());
            }
        }

        result.insert_or_assign(std::move(taxonomy), std::move(accession));
    }

    spdlog::info("Read {} fixed taxa from {}", result.size(), path.string());
    return result;
}

std::size_t count_checkm2_matches(
    const std::vector<GenomeRow>& genomes,
    const std::unordered_map<std::string, CheckM2Quality>& quality) {
    std::size_t count = 0;
    for (const auto& g : genomes) {
        auto acc = canonical_accession(g.accession);
        if (quality.find(acc) != quality.end()) {
            ++count;
        }
    }
    return count;
}

std::unordered_map<std::string, GuncQuality> read_gunc_tsv(
    const std::filesystem::path& path, const TsvReadOptions& opts) {
    std::ifstream in(path);
    if (!in) {
        throw TsvParseError("Cannot open GUNC TSV: " + path.string());
    }

    std::string line;
    std::size_t line_num = 0;

    int col_name = -1, col_pass = -1, col_css = -1, col_cont = -1;

    if (opts.has_header) {
        if (!std::getline(in, line)) {
            throw TsvParseError("Empty GUNC TSV: " + path.string());
        }
        ++line_num;
        auto headers = split_line(line, opts.delimiter);
        if (opts.trim_fields) {
            for (auto& h : headers) h = trim(h);
        }

        col_name = find_col(headers, {"genome_id", "genome"});
        col_pass = find_col(headers, {"pass.gunc", "pass_gunc"});
        col_css  = find_col(headers, {"clade_separation_score", "css"});
        col_cont = find_col(headers, {"contamination_portion"});

        if (col_name < 0 || col_pass < 0 || col_css < 0 || col_cont < 0) {
            throw TsvParseError(
                "GUNC TSV missing required columns (genome_id, pass.GUNC, "
                "clade_separation_score, contamination_portion): " + path.string());
        }
    } else {
        col_name = 0;
        col_pass = 8;
        col_css  = 6;
        col_cont = 7;
    }

    std::unordered_map<std::string, GuncQuality> result;

    while (std::getline(in, line)) {
        ++line_num;
        if (is_comment_or_empty(line, opts)) continue;

        auto fields = split_line(line, opts.delimiter);
        if (opts.trim_fields) {
            for (auto& f : fields) f = trim(f);
        }

        int max_col = std::max({col_name, col_pass, col_css, col_cont});
        if (static_cast<int>(fields.size()) <= max_col) {
            spdlog::warn("{}:{}: expected at least {} fields, got {}",
                         path.string(), line_num, max_col + 1, fields.size());
            continue;
        }

        GuncQuality q;
        // Strip file extensions from genome_id (e.g. .fna, .fa, .fasta, .gz)
        std::string gid = fields[col_name];
        for (const auto& ext : {".fna.gz", ".fa.gz", ".fasta.gz", ".fna", ".fa", ".fasta"}) {
            if (gid.size() > std::strlen(ext) &&
                gid.compare(gid.size() - std::strlen(ext), std::strlen(ext), ext) == 0) {
                gid.erase(gid.size() - std::strlen(ext));
                break;
            }
        }
        q.genome_id = canonical_accession(gid);

        // pass.GUNC: "True" or "False"
        const std::string& pass_str = fields[col_pass];
        q.pass_gunc = (pass_str == "True" || pass_str == "true" || pass_str == "1");

        try {
            q.clade_separation_score = std::stof(fields[col_css]);
            q.contamination_portion  = std::stof(fields[col_cont]);
        } catch (const std::exception& e) {
            spdlog::warn("{}:{}: failed to parse numeric fields: {}",
                         path.string(), line_num, e.what());
            continue;
        }

        if (opts.strict) {
            if (result.find(q.genome_id) != result.end()) {
                throw TsvParseError("Duplicate genome_id '" + q.genome_id +
                                    "' at line " + std::to_string(line_num) +
                                    " in " + path.string());
            }
        }

        result.insert_or_assign(q.genome_id, q);
    }

    spdlog::info("Read {} GUNC entries from {}", result.size(), path.string());
    return result;
}

} // namespace derep
