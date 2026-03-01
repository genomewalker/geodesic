#include "core/similarity/similarity_base.hpp"
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <unordered_set>

namespace derep {

std::vector<std::filesystem::path> SimilarityMeasure::missing_assemblies(
    const std::vector<SimilarityEdge>& edges,
    const std::vector<std::filesystem::path>& all_assemblies) const {

    std::unordered_set<std::string> seen;
    for (const auto& e : edges) {
        seen.insert(e.source);
        seen.insert(e.target);
    }

    std::vector<std::filesystem::path> missing;
    for (const auto& a : all_assemblies) {
        if (!seen.contains(a.string()))
            missing.push_back(a);
    }
    return missing;
}

std::filesystem::path SimilarityMeasure::write_assembly_list(
    const std::vector<std::filesystem::path>& assemblies,
    const std::string& prefix) const {

    auto list_path = temp_dir_ / (prefix + "_assemblies.txt");
    std::ofstream ofs(list_path);
    if (!ofs)
        throw std::runtime_error("Failed to write assembly list: " + list_path.string());
    for (const auto& a : assemblies)
        ofs << a.string() << '\n';
    return list_path;
}

std::filesystem::path SimilarityMeasure::make_temp_subdir(const std::string& prefix) const {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dist(0, 0xFFFFFFFF);

    std::ostringstream oss;
    oss << prefix << '_' << std::hex << std::setfill('0') << std::setw(8) << dist(gen);
    auto subdir = temp_dir_ / oss.str();
    std::filesystem::create_directories(subdir);
    return subdir;
}

} // namespace derep
