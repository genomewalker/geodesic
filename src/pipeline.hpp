#pragma once
#include "config.hpp"
#include "core/types.hpp"
#include "io/tsv_reader.hpp"
#include <unordered_map>
#include <vector>

namespace derep { class IPackReader; }
namespace derep { class RunState; }
namespace grd { class GrdWriter; }

namespace derep {

int run_pipeline(Config& cfg);
int run_update(Config& cfg);

// Process a set of taxa in parallel, accumulating results into run_state.
// Used by both run_pipeline() and run_update().
void process_taxa_parallel(
    const std::vector<Taxon>& taxa,
    const Config& cfg,
    RunState& run_state,
    IPackReader* gpk_reader = nullptr,
    const std::unordered_map<std::string, GuncQuality>* gunc_scores = nullptr,
    grd::GrdWriter* grd_writer = nullptr);

// Emit derep archive (.gpd) — best-effort, does not throw.
// cfg.gpd_output must be non-empty; cfg.pack_dir must be set.
void emit_gpd_archive(const Config& cfg, const RunState& run_state,
                      IPackReader* gpk_reader);

} // namespace derep
