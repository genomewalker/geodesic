#pragma once
#include "pack_reader.hpp"
#include <algorithm>
#include <cassert>
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

    void visit_sketch_batches_multi_k(
        const std::vector<std::string>& accessions,
        const std::vector<uint32_t>& ks, uint32_t sz,
        const std::function<void(size_t idx, uint32_t k,
                                 const genopack::SketchResult& sk)>& cb) const override;

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

    std::string taxonomy_for_accession(std::string_view acc) const override;
    void scan_taxonomy(
        const std::function<void(std::string_view, std::string_view)>& cb) const override;
    void scan_taxonomy_with_id(
        const std::function<void(std::string_view, std::string_view,
                                 genopack::GenomeId)>& cb) const override;

    bool has_qual() const override {
        for (const auto& a : archives_)
            if (a.reader->has_qual()) return true;
        return false;
    }
    void scan_qual(const std::function<void(const genopack::QualRecord&)>& cb) const override {
        for (const auto& a : archives_)
            a.reader->scan_qual(cb);
    }
    // Genome IDs are local per-archive, so the base-class build_qual_cache_() (which maps
    // genome_id → tier across all archives) aliases IDs across parts. Build a separate
    // per-archive (genome_id → tier) map and route each lookup to the owning archive.
    std::optional<uint8_t> quality_tier_for_accession(std::string_view acc) const override {
        build_arch_qual_cache_();
        uint16_t idx = archive_idx_for_accession(acc);
        if (idx == UINT16_MAX || idx >= arch_tier_cache_.size()) return std::nullopt;
        auto meta = archives_[idx].reader->genome_meta_by_accession(acc);
        if (!meta) return std::nullopt;
        auto it = arch_tier_cache_[idx].find(meta->genome_id);
        return it != arch_tier_cache_[idx].end() ? std::optional{it->second} : std::nullopt;
    }

    // Same per-archive routing as quality_tier_for_accession — the base build_qual_cache_()
    // aliases local genome_ids across parts, so the ranking score must be resolved per-archive.
    std::optional<double> qual_score_for_accession(std::string_view acc) const override {
        build_arch_qual_cache_();
        uint16_t idx = archive_idx_for_accession(acc);
        if (idx == UINT16_MAX || idx >= arch_score_cache_.size()) return std::nullopt;
        auto meta = archives_[idx].reader->genome_meta_by_accession(acc);
        if (!meta) return std::nullopt;
        auto it = arch_score_cache_[idx].find(meta->genome_id);
        return it != arch_score_cache_[idx].end() ? std::optional{it->second} : std::nullopt;
    }

    bool has_gstx() const override {
        for (const auto& a : archives_)
            if (a.reader->has_gstx()) return true;
        return false;
    }
    const genopack::GstxEntry* gstx_for_genus(std::string_view genus) const override {
        for (const auto& a : archives_) {
            auto* e = a.reader->gstx_for_genus(genus);
            if (e) return e;
        }
        return nullptr;
    }

private:
    struct ArchiveEntry {
        std::unique_ptr<genopack::ArchiveReader> reader;
        std::filesystem::path                    path;
        bool has_sketches_flag = false;
    };

    std::vector<ArchiveEntry> archives_;
    // accession → archive_idx (for O(1) routing)
    std::unordered_map<std::string, uint16_t> acc_to_arch_;

    // Per-archive quality caches: built once, keyed by local genome_id per archive
    // (avoids the cross-part ID aliasing of the base build_qual_cache_()).
    mutable std::once_flag                                        arch_tier_once_;
    mutable std::vector<std::unordered_map<genopack::GenomeId, uint8_t>> arch_tier_cache_;
    mutable std::vector<std::unordered_map<genopack::GenomeId, double>>  arch_score_cache_;

    void build_arch_qual_cache_() const {
        std::call_once(arch_tier_once_, [this] {
            arch_tier_cache_.resize(archives_.size());
            arch_score_cache_.resize(archives_.size());
            for (size_t i = 0; i < archives_.size(); ++i) {
                if (!archives_[i].reader->has_qual()) continue;
                archives_[i].reader->scan_qual([&](const genopack::QualRecord& r) {
                    arch_tier_cache_[i][r.genome_id]  = r.quality_tier_u8;
                    arch_score_cache_[i][r.genome_id] = genome_quality_score(r);
                });
            }
        });
    }

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
        // Pack the 16-bit archive index into the high bits and the 48-bit local
        // genome id into the low bits. Assert the local id fits in 48 bits so it
        // can never silently truncate and collide into another archive's range (P24).
        assert((local & 0xFFFF'0000'0000'0000ull) == 0 &&
               "local genome id exceeds 48 bits — virtual-id packing would truncate");
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
