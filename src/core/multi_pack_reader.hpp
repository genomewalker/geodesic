#pragma once
#include "pack_reader.hpp"
#include <algorithm>
#include <filesystem>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace derep {

// Aggregates N .gpk archives from a directory into a single IPackReader.
//
// Virtual genome IDs encode the source archive: virt_id = (archive_idx << 48) | local_id.
// This ensures genome IDs returned from genome_meta_by_accession() can be passed back
// to sketch_for() without an extra lookup table.
//
// visit_shard_batches() routes each accession to its archive and remaps the local
// ShardBatch indices back to the caller's original global accession indices.
class MultiPackReader : public IPackReader {
public:
    // Open all .gpk subdirectories inside parts_dir (sorted lexicographically).
    // Throws std::runtime_error if no .gpk directories are found.
    static std::unique_ptr<MultiPackReader> open_dir(const std::filesystem::path& parts_dir);

    void scan_genome_accessions(
        const std::function<void(std::string_view, genopack::GenomeId)>& cb) const override;

    std::optional<genopack::GenomeMeta> genome_meta_by_accession(
        std::string_view acc) const override;

    bool has_sketches() const override;

    std::optional<genopack::SketchResult> sketch_for(
        genopack::GenomeId virt_id) const override;

    std::optional<genopack::SketchResult> sketch_for(
        genopack::GenomeId virt_id, uint32_t k, uint32_t sz) const override;

    uint32_t sketch_kmer_size()   const override;
    uint32_t sketch_sketch_size() const override;

    bool has_kmer_size(uint32_t k) const override {
        for (const auto& a : archives_)
            if (a.reader->has_sketches()) {
                auto ks = a.reader->available_sketch_kmer_sizes();
                if (std::find(ks.begin(), ks.end(), k) != ks.end()) return true;
            }
        return false;
    }

    std::vector<uint32_t> available_kmer_sizes() const override {
        std::vector<uint32_t> all;
        for (const auto& a : archives_) {
            for (uint32_t k : a.reader->available_sketch_kmer_sizes())
                if (std::find(all.begin(), all.end(), k) == all.end())
                    all.push_back(k);
        }
        std::sort(all.begin(), all.end());
        return all;
    }

    void visit_shard_batches(
        const std::vector<std::string>& accessions,
        const std::function<void(genopack::ArchiveReader::ShardBatch&)>& cb) const override;

    // Sketch batch visitor: groups accessions by archive, decompresses each archive's
    // SKCH section exactly once, delivers all sketches from it, then releases.
    // Eliminates SKCH thrashing when n_archives > max_hot_archives_.
    // cb(global_idx, sketch_result) — called for each found genome in archive order.
    void visit_sketch_batches(
        const std::vector<std::string>& accessions,
        uint32_t k, uint32_t sz,
        const std::function<void(size_t idx,
                                 const genopack::SketchResult& sk)>& cb) const;

    const float* kmer_profile(genopack::GenomeId virt_id) const override;
    const float* kmer_profile_by_accession(std::string_view acc) const override;

    void release_sketches() const override {
        std::lock_guard<std::mutex> lock(lru_mu_);
        for (auto& a : archives_) a.reader->release_sketches();
        lru_order_.clear();
        lru_pos_.clear();
#if defined(__GLIBC__)
        ::malloc_trim(0);
#endif
    }
    size_t sketch_memory_bytes() const override {
        size_t t = 0;
        for (const auto& a : archives_) t += a.reader->sketch_memory_bytes();
        return t;
    }

    size_t n_archives() const override { return archives_.size(); }
    size_t n_genomes()  const { return acc_to_arch_.size(); }

    uint16_t archive_idx_for_accession(std::string_view acc) const override;

private:
    struct ArchiveEntry {
        std::unique_ptr<genopack::ArchiveReader> reader;
        std::filesystem::path                    path;
        bool has_sketches_flag = false;
    };

    std::vector<ArchiveEntry> archives_;
    // accession → archive_idx (for O(1) routing)
    std::unordered_map<std::string, uint16_t> acc_to_arch_;

    // Per-archive mutex: serialises concurrent visit_sketch_batches calls on the
    // same archive and guards release_sketches() against concurrent sketch_for_ids.
    mutable std::vector<std::mutex>                            arch_sketch_mu_;

    // LRU eviction: keep at most max_hot_archives_ sketch sections decompressed.
    // front = MRU, back = LRU.
    mutable std::mutex                                         lru_mu_;
    mutable std::list<size_t>                                  lru_order_;
    mutable std::unordered_map<size_t,
                std::list<size_t>::iterator>                   lru_pos_;
    size_t                                                     max_hot_archives_ = 2;

    // Mark aidx as recently used; evict LRU archives if over budget.
    // Must be called WITHOUT lru_mu_ held (acquires internally).
    void touch_archive_(size_t aidx) const;

    // Drop kernel page-cache for all files in the archive at path p.
    void fadvise_dontneed_(const std::filesystem::path& p) const noexcept;

    static genopack::GenomeId encode_virt(uint16_t aidx, genopack::GenomeId local) noexcept {
        return (static_cast<genopack::GenomeId>(aidx) << 48)
               | (local & 0x0000'FFFF'FFFF'FFFFull);
    }
    static std::pair<uint16_t, genopack::GenomeId> decode_virt(
        genopack::GenomeId virt) noexcept {
        return {static_cast<uint16_t>(virt >> 48),
                virt & 0x0000'FFFF'FFFF'FFFFull};
    }
};

} // namespace derep
