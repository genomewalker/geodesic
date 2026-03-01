#pragma once
#include "config.hpp"
#include "core/genome_cache.hpp"
#include "core/types.hpp"
#include "db/db_manager.hpp"

namespace derep::db { class EmbeddingStore; }  // Forward declaration

namespace derep {

// Process a single taxon (with optional shared embedding store for incremental updates)
// thread_budget: number of threads allocated for this taxon (0 = use cfg.threads)
TaxonResult process_taxon(
    const Taxon& taxon,
    const Config& cfg,
    int thread_budget,
    db::DBManager& db,
    GenomeCache& cache,
    db::EmbeddingStore* emb_store = nullptr);

} // namespace derep
