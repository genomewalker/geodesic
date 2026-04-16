#include "lock_writer.hpp"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace derep {

uint64_t file_tail_hash(const std::filesystem::path& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f)
        throw std::runtime_error("lock_writer: cannot open file for tail hash: " + path.string());
    uint8_t buf[32];
    size_t n;
    if (std::fseek(f, -32, SEEK_END) == 0) {
        n = std::fread(buf, 1, 32, f);
    } else {
        std::rewind(f);
        n = std::fread(buf, 1, 32, f);
    }
    std::fclose(f);
    uint64_t h = 14695981039346656037ULL;
    for (size_t i = 0; i < n; ++i) { h ^= buf[i]; h *= 1099511628211ULL; }
    return h;
}

void write_lock_file(const std::filesystem::path& path, const LockData& data) {
    std::ofstream out(path);
    if (!out)
        throw std::runtime_error("lock_writer: cannot open for writing: " + path.string());

    out << "{\n";
    out << "  \"geodesic_version\": \"" << data.geodesic_version << "\",\n";
    out << "  \"timestamp\": \"" << data.timestamp << "\",\n";
    out << "  \"gpk\": {\n";
    out << "    \"path\": \"" << data.gpk_path.string() << "\",\n";
    out << "    \"snapshot_id\": " << data.gpk_snapshot_id << "\n";
    out << "  },\n";
    out << "  \"run_params\": {\n";
    out << "    \"kmer_size\": " << data.kmer_size << ",\n";
    out << "    \"sketch_size\": " << data.sketch_size << ",\n";
    out << "    \"syncmer_s\": " << data.syncmer_s << ",\n";
    out << "    \"ani_threshold\": " << data.ani_threshold << ",\n";
    out << "    \"params_hash\": " << data.params_hash << "\n";
    out << "  },\n";
    out << "  \"outputs\": {\n";
    out << "    \"geodf_path\": \"" << data.geodf_path.string() << "\",\n";
    out << "    \"geodf_hash\": " << data.geodf_hash << "\n";
    out << "  },\n";
    out << "  \"stats\": {\n";
    out << "    \"n_genomes\": " << data.n_genomes << ",\n";
    out << "    \"n_taxa\": " << data.n_taxa << ",\n";
    out << "    \"n_reps\": " << data.n_reps << "\n";
    out << "  }\n";
    out << "}\n";

    if (!out)
        throw std::runtime_error("lock_writer: write error: " + path.string());
}

LockData read_lock_file(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("lock_writer: cannot open for reading: " + path.string());

    LockData data;
    std::string line;
    while (std::getline(in, line)) {
        auto extract_str = [&](const std::string& key) -> std::string {
            auto pos = line.find("\"" + key + "\": \"");
            if (pos == std::string::npos) return "";
            pos += key.size() + 5;
            auto end = line.find("\"", pos);
            if (end == std::string::npos) return "";
            return line.substr(pos, end - pos);
        };
        auto extract_uint64 = [&](const std::string& key) -> uint64_t {
            auto pos = line.find("\"" + key + "\": ");
            if (pos == std::string::npos) return 0;
            pos += key.size() + 4;
            try { return std::stoull(line.substr(pos)); } catch (...) { return 0; }
        };
        auto extract_int = [&](const std::string& key) -> int {
            auto pos = line.find("\"" + key + "\": ");
            if (pos == std::string::npos) return 0;
            pos += key.size() + 4;
            try { return std::stoi(line.substr(pos)); } catch (...) { return 0; }
        };
        auto extract_double = [&](const std::string& key) -> double {
            auto pos = line.find("\"" + key + "\": ");
            if (pos == std::string::npos) return 0.0;
            pos += key.size() + 4;
            try { return std::stod(line.substr(pos)); } catch (...) { return 0.0; }
        };

        if (auto v = extract_str("geodesic_version"); !v.empty()) data.geodesic_version = v;
        if (auto v = extract_str("timestamp");        !v.empty()) data.timestamp = v;
        if (auto v = extract_str("path");             !v.empty() && data.gpk_path.empty()) data.gpk_path = v;
        if (auto v = extract_uint64("snapshot_id");   v) data.gpk_snapshot_id = v;
        if (auto v = extract_int("kmer_size");        v) data.kmer_size = v;
        if (auto v = extract_int("sketch_size");      v) data.sketch_size = v;
        if (line.find("\"syncmer_s\":") != std::string::npos) data.syncmer_s = extract_int("syncmer_s");
        if (auto v = extract_double("ani_threshold");  v) data.ani_threshold = v;
        if (auto v = extract_uint64("params_hash");   v) data.params_hash = static_cast<uint32_t>(v);
        if (auto v = extract_str("geodf_path");       !v.empty()) data.geodf_path = v;
        if (auto v = extract_uint64("geodf_hash");    v) data.geodf_hash = v;
        if (auto v = extract_uint64("n_genomes");     v) data.n_genomes = static_cast<size_t>(v);
        if (auto v = extract_uint64("n_taxa");        v) data.n_taxa = static_cast<size_t>(v);
        if (auto v = extract_uint64("n_reps");        v) data.n_reps = static_cast<size_t>(v);
    }
    return data;
}

} // namespace derep
