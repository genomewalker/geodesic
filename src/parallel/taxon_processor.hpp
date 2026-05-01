#pragma once
#include "config.hpp"
#include "core/types.hpp"
#include "io/tsv_reader.hpp"
#include <functional>
#include <unordered_map>

namespace derep { class IPackReader; }
namespace derep { class RunState; }
namespace grd { class GrdWriter; }

namespace derep {

// Process a single taxon.
// thread_budget: number of threads allocated for this taxon (0 = use cfg.threads)
// gunc_scores: optional GUNC quality map (accession → GuncQuality); null = no GUNC filtering
// gpk_reader: if non-null, raw FASTA is loaded from pack instead of NFS reads
// run_state: if non-null, completed TaxonOutput is pushed for in-memory accumulation
// grd_writer: if non-null, writes per-genome embeddings + metadata to GRD archive
TaxonResult process_taxon(
    const Taxon& taxon,
    const Config& cfg,
    int thread_budget,
    const std::unordered_map<std::string, GuncQuality>* gunc_scores = nullptr,
    IPackReader* gpk_reader = nullptr,
    RunState* run_state = nullptr,
    grd::GrdWriter* grd_writer = nullptr,
    std::function<void()> on_serial_phase = {});

// Process a batch of tiny taxa (n <= TINY_BATCH_N).
// acquired_threads: OMP parallelism cap (matches budget actually acquired by caller).
// on_result: invoked once per taxon as soon as it finishes (thread-safe callback);
//            if null, results accumulate and are returned at the end.
std::vector<TaxonResult> process_tiny_batch(
    const std::vector<const Taxon*>& taxa,
    const Config& cfg,
    int acquired_threads,
    const std::unordered_map<std::string, GuncQuality>* gunc_scores = nullptr,
    IPackReader* gpk_reader = nullptr,
    RunState* run_state = nullptr,
    grd::GrdWriter* grd_writer = nullptr,
    std::function<void(TaxonResult&&)> on_result = {});

} // namespace derep
