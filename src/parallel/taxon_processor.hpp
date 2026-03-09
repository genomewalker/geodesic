#pragma once
#include "config.hpp"
#include "core/types.hpp"
#include "db/db_manager.hpp"
#include "io/tsv_reader.hpp"
#include <unordered_map>

namespace derep::db { class EmbeddingStore; }  // Forward declaration

namespace derep {

// Process a single taxon (with optional shared embedding store for incremental updates)
// thread_budget: number of threads allocated for this taxon (0 = use cfg.threads)
// gunc_scores: optional GUNC quality map (accession → GuncQuality); null = no GUNC filtering
// in_batch_txn: caller holds an open transaction; skip inner BEGIN/COMMIT
TaxonResult process_taxon(
    const Taxon& taxon,
    const Config& cfg,
    int thread_budget,
    db::DBManager& db,
    db::EmbeddingStore* emb_store = nullptr,
    const std::unordered_map<std::string, GuncQuality>* gunc_scores = nullptr,
    bool in_batch_txn = false);

// Process a batch of tiny taxa (n <= TINY_BATCH_N) in a single thread slot.
// Returns results in the same order as input.
std::vector<TaxonResult> process_tiny_batch(
    const std::vector<const Taxon*>& taxa,
    const Config& cfg,
    db::DBManager& db);

} // namespace derep
