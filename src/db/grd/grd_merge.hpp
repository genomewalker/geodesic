// grd_merge.hpp — Merge multiple GRD shard files into a single GRD archive.
//
// Each shard is an independent GRD file written by a distributed worker.
// Merge reads all per-taxon sections (TMTA, GMET, EMBD, PRJ3, EDGE) from
// each shard, renumbers taxon ordinals, and rebuilds the global indexes
// (TIDX, ACCX, STRT, TREE) in the output file.
//
#pragma once
#include <filesystem>
#include <string>
#include <vector>

namespace grd {

struct MergeStats {
    size_t n_shards = 0;
    size_t n_taxa = 0;
    size_t n_genomes = 0;
    size_t output_bytes = 0;
};

// Merge multiple GRD shard files into a single output GRD file.
// Shard paths can be in any order — taxa are sorted by taxonomy hash in output.
MergeStats merge_grd(const std::vector<std::filesystem::path>& shards,
                     const std::filesystem::path& output);

} // namespace grd
