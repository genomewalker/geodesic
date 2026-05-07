#pragma once
#include "pack_reader.hpp"
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace derep {

// Decorator over an IPackReader that holds one or more RAM-resident SoA stores
// keyed by kmer_size. A request for visit_sketch_batches at (k, sz) hits the
// per-k store if present and matches sz; otherwise falls through to the inner
// reader.
class PreloadedPackReader : public IPackReader {
public:
    explicit PreloadedPackReader(std::unique_ptr<IPackReader> inner)
        : inner_(std::move(inner)) {}

    // Populate a store for (k, sz) — single-k convenience (clears prior stores).
    std::pair<size_t, size_t> preload(
        const std::vector<std::string>& accessions,
        uint32_t k, uint32_t sz, int n_threads);

    // Populate stores for every k in `ks` at the same sz. Accessions and
    // acc_to_idx_ are shared across k's (one partition, three sketch copies).
    // Clears any prior stores.
    std::pair<size_t, size_t> preload_multi(
        const std::vector<std::string>& accessions,
        const std::vector<uint32_t>& ks,
        uint32_t sz, int n_threads);

    void reload_for_k(uint32_t new_k, int n_threads);

    bool has_preload(uint32_t k, uint32_t sz) const noexcept {
        auto it = k_stores_.find(k);
        return it != k_stores_.end() && it->second.sz >= sz && !acc_to_idx_.empty();
    }
    size_t   preloaded_count() const noexcept { return acc_to_idx_.size(); }
    size_t   bytes() const noexcept;

    // ── IPackReader passthrough ──────────────────────────────────────────
    void scan_genome_accessions(
        const std::function<void(std::string_view, genopack::GenomeId)>& cb) const override {
        inner_->scan_genome_accessions(cb);
    }
    std::optional<genopack::GenomeMeta> genome_meta_by_accession(
        std::string_view acc) const override {
        return inner_->genome_meta_by_accession(acc);
    }
    bool has_sketches() const override { return inner_->has_sketches(); }
    std::optional<genopack::SketchResult> sketch_for(genopack::GenomeId id) const override {
        return inner_->sketch_for(id);
    }
    std::optional<genopack::SketchResult> sketch_for(
        genopack::GenomeId id, uint32_t k, uint32_t sz) const override {
        return inner_->sketch_for(id, k, sz);
    }
    uint32_t sketch_kmer_size()   const override { return inner_->sketch_kmer_size(); }
    uint32_t sketch_sketch_size() const override { return inner_->sketch_sketch_size(); }
    bool has_kmer_size(uint32_t k) const override { return inner_->has_kmer_size(k); }
    std::vector<uint32_t> available_kmer_sizes() const override {
        return inner_->available_kmer_sizes();
    }
    void visit_shard_batches(
        const std::vector<std::string>& accessions,
        const std::function<void(genopack::ArchiveReader::ShardBatch&)>& cb) const override {
        inner_->visit_shard_batches(accessions, cb);
    }

    // Hot path: serves preloaded hits from RAM, delegates misses.
    void visit_sketch_batches(
        const std::vector<std::string>& accessions,
        uint32_t k, uint32_t sz,
        const std::function<void(size_t idx,
                                 const genopack::SketchResult& sk)>& cb) const override;

    void release_sketches() const override {
        inner_->release_sketches();
        const_cast<PreloadedPackReader*>(this)->clear_store_();
    }
    size_t sketch_memory_bytes() const override {
        return inner_->sketch_memory_bytes() + bytes();
    }

    const float* kmer_profile(genopack::GenomeId id) const override {
        return inner_->kmer_profile(id);
    }
    const float* kmer_profile_by_accession(std::string_view acc) const override {
        return inner_->kmer_profile_by_accession(acc);
    }

    uint16_t archive_idx_for_accession(std::string_view acc) const override {
        return inner_->archive_idx_for_accession(acc);
    }
    size_t n_archives() const override { return inner_->n_archives(); }

    std::string taxonomy_for_accession(std::string_view acc) const override {
        return inner_->taxonomy_for_accession(acc);
    }
    void scan_taxonomy(
        const std::function<void(std::string_view, std::string_view)>& cb) const override {
        inner_->scan_taxonomy(cb);
    }

    IPackReader* inner() const { return inner_.get(); }

private:
    struct KStore {
        uint32_t k = 0;
        uint32_t sz = 0;
        uint32_t mask_words = 0;
        std::vector<uint16_t> sigs;    // [N * sz]
        std::vector<uint16_t> sig2s;   // [N * sz]
        std::vector<uint64_t> masks;   // [N * mask_words]
        std::vector<uint32_t> n_real_bins;
        std::vector<uint64_t> genome_lengths;
    };

    std::unique_ptr<IPackReader> inner_;
    std::vector<std::string>     preload_set_;
    std::unordered_map<uint32_t, KStore> k_stores_;
    std::unordered_map<std::string, uint32_t> acc_to_idx_;

    void clear_store_();
    void populate_store_(KStore& st, const std::vector<std::string>& accessions, int n_threads);
};

} // namespace derep
