#pragma once
#include "config.hpp"
#include "core/types.hpp"
#include "db/db_manager.hpp"
#include "db/async_writer.hpp"
#include "io/tsv_reader.hpp"
#include <unordered_map>

namespace derep::db { class EmbeddingStore; }  // Forward declaration
namespace derep::db { class SketchStore; }      // Forward declaration
namespace derep::db { class GenomePack; }       // Forward declaration

namespace derep {

// Process a single taxon (with optional shared embedding store for incremental updates)
// thread_budget: number of threads allocated for this taxon (0 = use cfg.threads)
// gunc_scores: optional GUNC quality map (accession → GuncQuality); null = no GUNC filtering
// in_batch_txn: caller holds an open transaction; skip inner BEGIN/COMMIT
// async_writer: if non-null, DB writes are pushed to the async writer instead of direct ops
// sketch_store: if non-null, OPH signatures are loaded from cache instead of NFS reads
// genome_pack: if non-null, raw FASTA is loaded from pack instead of NFS reads
TaxonResult process_taxon(
    const Taxon& taxon,
    const Config& cfg,
    int thread_budget,
    db::DBManager& db,
    db::EmbeddingStore* emb_store = nullptr,
    const std::unordered_map<std::string, GuncQuality>* gunc_scores = nullptr,
    bool in_batch_txn = false,
    db::AsyncDBWriter* async_writer = nullptr,
    db::SketchStore* sketch_store = nullptr,
    db::GenomePack* genome_pack = nullptr);

// Process a batch of tiny taxa (n <= TINY_BATCH_N) in a single thread slot.
// Returns results in the same order as input.
std::vector<TaxonResult> process_tiny_batch(
    const std::vector<const Taxon*>& taxa,
    const Config& cfg,
    db::DBManager& db,
    db::AsyncDBWriter* async_writer = nullptr,
    db::SketchStore* sketch_store = nullptr,
    db::GenomePack* genome_pack = nullptr);

} // namespace derep
