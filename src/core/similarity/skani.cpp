#include "core/similarity/skani.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace derep {

Skani::Skani(int threads, std::filesystem::path temp_dir,
             GenomeCache& cache, SkaniConfig config)
    : SimilarityMeasure(threads, std::move(temp_dir))
    , cache_(cache)
    , config_(config) {}

std::vector<SimilarityEdge> Skani::compute_pairwise(
    const std::vector<std::filesystem::path>& assemblies) {

    if (assemblies.size() <= 1) return {};

    auto temp = make_temp_subdir("skani");

    std::vector<SimilarityEdge> edges;
    if (static_cast<int>(assemblies.size()) <= config_.large_cluster_threshold) {
        edges = run_triangle(assemblies, temp);
    } else {
        edges = run_sketch_dist(assemblies, temp);
    }

    std::filesystem::remove_all(temp);
    return edges;
}

std::vector<SimilarityEdge> Skani::filter(
    std::vector<SimilarityEdge> edges) const {

    std::erase_if(edges, [this](const SimilarityEdge& e) {
        return e.weight_raw * 100.0 < config_.ani_threshold;
    });
    return edges;
}

std::vector<SimilarityEdge> Skani::run_triangle(
    const std::vector<std::filesystem::path>& assemblies,
    const std::filesystem::path& temp) const {

    auto list = write_assembly_list(assemblies, "skani_tri");
    auto output = temp / "triangle.tsv";
    auto t_str = std::to_string(threads_);

    auto result = run_subprocess({
        "skani", "triangle",
        "-l", list.string(),
        "-t", t_str,
        "-o", output.string(),
        "--sparse", "--fast"
    });

    if (!result.ok()) {
        throw std::runtime_error(
            "skani triangle failed (exit " + std::to_string(result.exit_code) +
            "): " + result.stderr_output);
    }

    return parse_skani_output(output);
}

std::vector<SimilarityEdge> Skani::run_sketch_dist(
    const std::vector<std::filesystem::path>& assemblies,
    const std::filesystem::path& temp) const {

    auto sketch_dir = temp / "sketches";
    // NOTE: Do NOT create sketch_dir - skani sketch requires it to not exist

    auto list = write_assembly_list(assemblies, "skani_sketch");
    auto t_str = std::to_string(threads_);

    auto sketch_result = run_subprocess({
        "skani", "sketch",
        "-l", list.string(),
        "-o", sketch_dir.string(),
        "-t", t_str
    });

    if (!sketch_result.ok()) {
        throw std::runtime_error(
            "skani sketch failed (exit " + std::to_string(sketch_result.exit_code) +
            "): " + sketch_result.stderr_output);
    }

    auto output = temp / "dist.tsv";

    auto dist_result = run_subprocess({
        "skani", "dist",
        "--qi", sketch_dir.string(),
        "--ri", sketch_dir.string(),
        "-t", t_str,
        "-o", output.string()
    });

    if (!dist_result.ok()) {
        throw std::runtime_error(
            "skani dist failed (exit " + std::to_string(dist_result.exit_code) +
            "): " + dist_result.stderr_output);
    }

    return parse_skani_output(output);
}

std::vector<SimilarityEdge> Skani::compute_candidates(
    const std::vector<std::filesystem::path>& assemblies,
    const std::vector<std::pair<size_t, size_t>>& candidate_pairs) {

    if (assemblies.size() <= 1 || candidate_pairs.empty()) return {};

    auto temp = make_temp_subdir("skani_cand");

    // Group candidate pairs by source genome for batched skani dist
    std::unordered_map<size_t, std::vector<size_t>> source_to_targets;
    for (const auto& [src, tgt] : candidate_pairs) {
        source_to_targets[src].push_back(tgt);
        source_to_targets[tgt].push_back(src);  // Symmetric
    }

    // Pre-sketch all involved genomes once
    std::unordered_set<size_t> involved;
    for (const auto& [src, tgt] : candidate_pairs) {
        involved.insert(src);
        involved.insert(tgt);
    }

    auto sketch_dir = temp / "sketches";
    // NOTE: Do NOT create sketch_dir - skani sketch requires it to not exist

    std::vector<std::filesystem::path> involved_paths;
    involved_paths.reserve(involved.size());
    for (size_t idx : involved) {
        involved_paths.push_back(assemblies[idx]);
    }

    auto list = write_assembly_list(involved_paths, "skani_sketch");
    auto t_str = std::to_string(threads_);

    auto sketch_result = run_subprocess({
        "skani", "sketch",
        "-l", list.string(),
        "-o", sketch_dir.string(),
        "-t", t_str
    });

    if (!sketch_result.ok()) {
        throw std::runtime_error(
            "skani sketch failed (exit " + std::to_string(sketch_result.exit_code) +
            "): " + sketch_result.stderr_output);
    }

    // Build path to sketch file mapping
    auto get_sketch_path = [&](const std::filesystem::path& genome) -> std::filesystem::path {
        return sketch_dir / (genome.stem().string() + ".sketch");
    };

    // Process in batches - group by unique source, run dist for each
    std::vector<SimilarityEdge> all_edges;

    // Build set of valid pairs for filtering (since skani dist may return extra pairs)
    std::unordered_set<uint64_t> valid_pairs;
    for (const auto& [src, tgt] : candidate_pairs) {
        size_t a = std::min(src, tgt);
        size_t b = std::max(src, tgt);
        valid_pairs.insert((static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b));
    }

    // Create path to index mapping
    std::unordered_map<std::string, size_t> path_to_idx;
    for (size_t i = 0; i < assemblies.size(); ++i) {
        path_to_idx[assemblies[i].string()] = i;
    }

    // Batch sources to reduce skani invocations
    // Each batch: write query list + ref list, run skani dist once
    constexpr size_t BATCH_SIZE = 100;
    std::vector<size_t> sources(source_to_targets.size());
    size_t src_idx = 0;
    for (const auto& [src, _] : source_to_targets) {
        sources[src_idx++] = src;
    }

    for (size_t batch_start = 0; batch_start < sources.size(); batch_start += BATCH_SIZE) {
        size_t batch_end = std::min(batch_start + BATCH_SIZE, sources.size());

        // Collect all unique references for this batch
        std::unordered_set<size_t> batch_refs;
        std::vector<std::filesystem::path> query_paths;

        for (size_t i = batch_start; i < batch_end; ++i) {
            size_t src = sources[i];
            query_paths.push_back(assemblies[src]);
            for (size_t tgt : source_to_targets.at(src)) {
                batch_refs.insert(tgt);
            }
        }

        std::vector<std::filesystem::path> ref_paths;
        ref_paths.reserve(batch_refs.size());
        for (size_t ref : batch_refs) {
            ref_paths.push_back(assemblies[ref]);
        }

        // Write lists
        auto query_list = temp / ("query_batch_" + std::to_string(batch_start) + ".txt");
        auto ref_list = temp / ("ref_batch_" + std::to_string(batch_start) + ".txt");

        {
            std::ofstream qf(query_list);
            for (const auto& p : query_paths) qf << p.string() << "\n";
        }
        {
            std::ofstream rf(ref_list);
            for (const auto& p : ref_paths) rf << p.string() << "\n";
        }

        auto output = temp / ("dist_batch_" + std::to_string(batch_start) + ".tsv");

        auto dist_result = run_subprocess({
            "skani", "dist",
            "--ql", query_list.string(),
            "--rl", ref_list.string(),
            "-t", t_str,
            "-o", output.string()
        });

        if (!dist_result.ok()) {
            throw std::runtime_error(
                "skani dist failed (exit " + std::to_string(dist_result.exit_code) +
                "): " + dist_result.stderr_output);
        }

        // Parse and filter to valid candidate pairs
        auto batch_edges = parse_skani_output(output);
        for (auto& e : batch_edges) {
            auto src_it = path_to_idx.find(e.source);
            auto tgt_it = path_to_idx.find(e.target);
            if (src_it == path_to_idx.end() || tgt_it == path_to_idx.end()) continue;

            size_t a = std::min(src_it->second, tgt_it->second);
            size_t b = std::max(src_it->second, tgt_it->second);
            uint64_t pair_key = (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);

            if (valid_pairs.count(pair_key)) {
                all_edges.push_back(std::move(e));
            }
        }
    }

    std::filesystem::remove_all(temp);
    return all_edges;
}

std::vector<SimilarityEdge> Skani::run_dist_pairs(
    const std::filesystem::path& query,
    const std::vector<std::filesystem::path>& references,
    const std::filesystem::path& temp) const {

    if (references.empty()) return {};

    auto ref_list = write_assembly_list(references, "skani_refs");
    auto output = temp / "dist_pairs.tsv";
    auto t_str = std::to_string(threads_);

    auto result = run_subprocess({
        "skani", "dist",
        "-q", query.string(),
        "-r", ref_list.string(),
        "-t", t_str,
        "-o", output.string()
    });

    if (!result.ok()) {
        throw std::runtime_error(
            "skani dist failed (exit " + std::to_string(result.exit_code) +
            "): " + result.stderr_output);
    }

    return parse_skani_output(output);
}

std::vector<SimilarityEdge> Skani::parse_skani_output(
    const std::filesystem::path& output_file) const {

    std::ifstream ifs(output_file);
    if (!ifs)
        throw std::runtime_error("Cannot open skani output: " + output_file.string());

    // Read header to find column indices
    std::string header_line;
    if (!std::getline(ifs, header_line))
        return {};

    std::istringstream hss(header_line);
    std::vector<std::string> headers;
    std::string col;
    while (std::getline(hss, col, '\t'))
        headers.push_back(col);

    // Case-insensitive column lookup
    auto find_col = [&](const std::vector<std::string>& names) -> int {
        for (std::size_t i = 0; i < headers.size(); ++i) {
            std::string lower = headers[i];
            std::transform(lower.begin(), lower.end(), lower.begin(),
                           [](unsigned char c) { return std::tolower(c); });
            for (const auto& name : names) {
                std::string lower_name = name;
                std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(),
                               [](unsigned char c) { return std::tolower(c); });
                if (lower == lower_name) return static_cast<int>(i);
            }
        }
        return -1;
    };

    int ref_col = find_col({"ref_file", "reference", "ref_name"});
    int qry_col = find_col({"query_file", "query", "query_name"});
    int ani_col = find_col({"ani"});
    int af1_col = find_col({"align_fraction_ref", "af1", "ref_align_fraction"});
    int af2_col = find_col({"align_fraction_query", "af2", "query_align_fraction"});

    if (ref_col < 0 || qry_col < 0 || ani_col < 0)
        throw std::runtime_error("Missing required columns in skani output: " + output_file.string());

    std::vector<SimilarityEdge> edges;
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;

        std::istringstream lss(line);
        std::vector<std::string> fields;
        std::string field;
        while (std::getline(lss, field, '\t'))
            fields.push_back(field);

        int max_col = std::max({ref_col, qry_col, ani_col, af1_col, af2_col});
        if (static_cast<int>(fields.size()) <= max_col) continue;

        const auto& source_path = fields[static_cast<std::size_t>(ref_col)];
        const auto& target_path = fields[static_cast<std::size_t>(qry_col)];

        if (source_path == target_path) continue;

        double ani = std::stod(fields[static_cast<std::size_t>(ani_col)]);
        double af1 = (af1_col >= 0) ? std::stod(fields[static_cast<std::size_t>(af1_col)]) : 1.0;
        double af2 = (af2_col >= 0) ? std::stod(fields[static_cast<std::size_t>(af2_col)]) : 1.0;

        double weight_raw = ani / 100.0;
        double aln_frac = (af1 + af2) / 2.0;
        double weight = weight_raw * aln_frac;

        SimilarityEdge edge;
        edge.source = source_path;
        edge.target = target_path;
        edge.weight_raw = weight_raw;
        edge.aln_frac = aln_frac;
        edge.weight = weight;
        edge.source_len = static_cast<int64_t>(cache_.get(std::filesystem::path(source_path)));
        edge.target_len = static_cast<int64_t>(cache_.get(std::filesystem::path(target_path)));
        edge.canonicalize();

        edges.push_back(std::move(edge));
    }

    return edges;
}

} // namespace derep
