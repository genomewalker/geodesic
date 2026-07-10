#pragma once
#include <genopack/archive.hpp>
#include <algorithm>
#include <cmath>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace derep {

// Minimal abstract interface over pack readers used by geodesic.
// Implemented by SinglePackReader (one .gpk) and MultiPackReader (N .gpk dirs).
struct IPackReader {
    virtual ~IPackReader() = default;

    virtual void scan_genome_accessions(
        const std::function<void(std::string_view, genopack::GenomeId)>& cb) const = 0;

    virtual std::optional<genopack::GenomeMeta> genome_meta_by_accession(
        std::string_view acc) const = 0;

    virtual bool has_sketches() const = 0;

    virtual std::optional<genopack::SketchResult> sketch_for(
        genopack::GenomeId id) const = 0;

    // Param-aware: picks section with matching k, slices to sz bins if sz < stored size.
    // Returns nullopt if no section with kmer_size==k exists or genome not found.
    virtual std::optional<genopack::SketchResult> sketch_for(
        genopack::GenomeId id, uint32_t k, uint32_t sz) const = 0;

    // k-mer size and sketch_size of the first available SKCH section (0 if none).
    virtual uint32_t sketch_kmer_size()   const { return 0; }
    virtual uint32_t sketch_sketch_size() const { return 0; }

    // Returns true if any SKCH section stores sketches for k.
    virtual bool has_kmer_size(uint32_t /*k*/) const { return false; }

    // All k-mer sizes available across all SKCH sections (sorted ascending).
    virtual std::vector<uint32_t> available_kmer_sizes() const { return {}; }

    virtual void visit_shard_batches(
        const std::vector<std::string>& accessions,
        const std::function<void(genopack::ArchiveReader::ShardBatch&)>& cb) const = 0;

    // Sketch batch visitor: groups accessions by archive, decompresses each SKCH
    // section once, delivers all sketches from it, then releases. Eliminates thrashing
    // when accessions interleave across multiple archive parts.
    virtual void visit_sketch_batches(
        const std::vector<std::string>& accessions,
        uint32_t k, uint32_t sz,
        const std::function<void(size_t idx,
                                 const genopack::SketchResult& sk)>& cb) const = 0;

    // Multi-k variant: processes all k-values for each archive part before evicting
    // from page cache. Avoids re-reading the same archive frames N times (once per k).
    // Default: falls back to calling visit_sketch_batches once per k.
    virtual void visit_sketch_batches_multi_k(
        const std::vector<std::string>& accessions,
        const std::vector<uint32_t>& ks, uint32_t sz,
        const std::function<void(size_t idx, uint32_t k,
                                 const genopack::SketchResult& sk)>& cb) const {
        for (uint32_t k : ks)
            visit_sketch_batches(accessions, k, sz,
                [&](size_t idx, const genopack::SketchResult& sk) { cb(idx, k, sk); });
    }

    virtual void release_sketches() const {}
    virtual size_t sketch_memory_bytes() const { return 0; }

    // TNF profile access (KMRX section — mmap'd, no decompression cost).
    // Returns pointer to float[136] L2-normalised k=4 tetranucleotide vector,
    // or nullptr if the archive has no KMRX section.
    virtual const float* kmer_profile(genopack::GenomeId id) const { return nullptr; }
    virtual const float* kmer_profile_by_accession(std::string_view acc) const { return nullptr; }

    virtual uint16_t archive_idx_for_accession(std::string_view acc) const = 0;
    virtual size_t n_archives() const = 0;

    // GSTX section: precomputed per-genus consensus + p90 + TNF centroid.
    virtual bool has_gstx() const { return false; }
    virtual const genopack::GstxEntry* gstx_for_genus(std::string_view /*genus*/) const { return nullptr; }

    // QUAL section: precomputed per-genome quality scores (completeness, contamination, consistency).
    virtual bool has_qual() const { return false; }
    virtual void scan_qual(const std::function<void(const genopack::QualRecord&)>& /*cb*/) const {}

    // Per-record ranking score, 0–100. Prefers genopack's continuous quality_score
    // (threshold-free, [0,1] → ×100); falls back to the legacy completeness−5·contamination
    // proxy only for packs predating the stored score.
    static double genome_quality_score(const genopack::QualRecord& r) {
        const float gp = r.quality_score();
        if (!std::isnan(gp)) return static_cast<double>(gp) * 100.0;
        const float c = !std::isnan(r.completeness_post_decontam)
                        ? r.completeness_post_decontam : r.completeness_cluster_relative;
        return static_cast<double>(c) * 100.0
             - 5.0 * static_cast<double>(r.contamination_leakage) * 100.0;
    }

    // Per-accession quality score (see genome_quality_score).
    // Built once from scan_qual + scan_genome_accessions; thread-safe.
    // NOTE: genome IDs are local per-archive. Multi-pack readers MUST override
    // quality_tier_for_accession() and qual_score_for_accession() to avoid ID aliasing.
    virtual std::optional<double> qual_score_for_accession(std::string_view acc) const {
        build_qual_cache_();
        auto it = qual_score_cache_.find(std::string(acc));
        return it != qual_score_cache_.end() ? std::optional{it->second} : std::nullopt;
    }

    // Returns the stored quality tier (LQ/MQ/HQ) or empty if not available.
    // Requires a pack written by genopack check >= the version that stores quality_tier_u8.
    // Old packs return empty (QTIER_NOT_SET == 0).
    virtual std::optional<uint8_t> quality_tier_for_accession(std::string_view acc) const {
        build_qual_cache_();
        auto it = qual_tier_cache_.find(std::string(acc));
        return it != qual_tier_cache_.end() ? std::optional{it->second} : std::nullopt;
    }

    bool is_lq(std::string_view acc) const {
        auto t = quality_tier_for_accession(acc);
        return t && *t == genopack::QualRecord::QTIER_LQ;
    }

    // Returns completeness_cluster_relative (fraction 0–1), or empty if unavailable.
    // NOTE: this is PANGENOME FRACTION (genus-pangenome breadth), NOT intrinsic
    // completeness. A finished isolate in a diverse genus reports a low value here.
    // Kept for reporting; do NOT use it as a completeness quality gate — use
    // completeness_effective_for_accession() instead.
    std::optional<float> completeness_cr_for_accession(std::string_view acc) const {
        build_qual_cache_();
        auto it = qual_cr_cache_.find(std::string(acc));
        return it != qual_cr_cache_.end() ? std::optional{it->second} : std::nullopt;
    }

    // Intrinsic completeness in [0,1], or empty if no QUAL signal. Mirrors genopack
    // run_check.cpp completeness_effective() exactly: intrinsic priority
    // marker_completeness → aamer_core → post_decontam, with
    // cluster_relative as a soft corroborator (geomean) only when intrinsic reads
    // genuinely partial (< 0.50) and the two disagree by > 0.30 — never on its own.
    virtual std::optional<float> completeness_effective_for_accession(std::string_view acc) const {
        build_qual_cache_();
        const std::string key(acc);
        auto im  = qual_marker_cache_.find(key);
        auto ia  = qual_aamer_core_cache_.find(key);
        auto ip  = qual_post_decontam_cache_.find(key);
        const float mc = im  != qual_marker_cache_.end()        ? im->second  : NAN;
        const float ac = ia  != qual_aamer_core_cache_.end()    ? ia->second  : NAN;
        const float pd = ip  != qual_post_decontam_cache_.end() ? ip->second  : NAN;
        const float intrinsic = !std::isnan(mc) ? mc
                              : (!std::isnan(ac) ? ac : pd);
        auto ic = qual_cr_cache_.find(key);
        const float cr = ic != qual_cr_cache_.end() ? ic->second : NAN;
        if (std::isnan(intrinsic)) {
            // No intrinsic signal at all → cluster_relative is the only proxy.
            return std::isnan(cr) ? std::nullopt : std::optional{cr};
        }
        if (intrinsic < 0.50f && !std::isnan(cr) && cr < intrinsic
            && (intrinsic - cr) > 0.30f)
            return std::sqrt(intrinsic * cr);
        return intrinsic;
    }

private:
    void build_qual_cache_() const {
        std::call_once(qual_cache_once_, [this] {
            if (!has_qual()) return;
            std::unordered_map<genopack::GenomeId, double>  id_scores;
            std::unordered_map<genopack::GenomeId, uint8_t> id_tiers;
            std::unordered_map<genopack::GenomeId, float>   id_cr;
            std::unordered_map<genopack::GenomeId, float>   id_aamer_core;
            std::unordered_map<genopack::GenomeId, float>   id_marker;
            std::unordered_map<genopack::GenomeId, float>   id_post_decontam;
            scan_qual([&](const genopack::QualRecord& r) {
                id_scores[r.genome_id] = genome_quality_score(r);
                id_tiers[r.genome_id]  = r.quality_tier_u8;
                if (!std::isnan(r.completeness_cluster_relative))
                    id_cr[r.genome_id] = r.completeness_cluster_relative;
                if (!std::isnan(r.completeness_aamer_core))
                    id_aamer_core[r.genome_id] = r.completeness_aamer_core;
                if (r.marker_completeness_u8 > 0)
                    id_marker[r.genome_id] = (r.marker_completeness_u8 - 1) / 254.0f;
                if (!std::isnan(r.completeness_post_decontam))
                    id_post_decontam[r.genome_id] = r.completeness_post_decontam;
            });
            scan_genome_accessions([&](std::string_view a, genopack::GenomeId gid) {
                auto is = id_scores.find(gid);
                if (is != id_scores.end()) qual_score_cache_[std::string(a)] = is->second;
                auto it = id_tiers.find(gid);
                if (it != id_tiers.end()) qual_tier_cache_[std::string(a)]  = it->second;
                auto ic = id_cr.find(gid);
                if (ic != id_cr.end()) qual_cr_cache_[std::string(a)]       = ic->second;
                auto iac = id_aamer_core.find(gid);
                if (iac != id_aamer_core.end())
                    qual_aamer_core_cache_[std::string(a)] = iac->second;
                auto imk = id_marker.find(gid);
                if (imk != id_marker.end())
                    qual_marker_cache_[std::string(a)] = imk->second;
                auto ipd = id_post_decontam.find(gid);
                if (ipd != id_post_decontam.end())
                    qual_post_decontam_cache_[std::string(a)] = ipd->second;
            });
        });
    }

    mutable std::once_flag                          qual_cache_once_;
    mutable std::unordered_map<std::string, double> qual_score_cache_;
    mutable std::unordered_map<std::string, uint8_t> qual_tier_cache_;
    mutable std::unordered_map<std::string, float>  qual_cr_cache_;
    mutable std::unordered_map<std::string, float>  qual_aamer_core_cache_;
    mutable std::unordered_map<std::string, float>  qual_marker_cache_;
    mutable std::unordered_map<std::string, float>  qual_post_decontam_cache_;

public:
    // Taxonomy access (TAXN section).
    virtual std::string taxonomy_for_accession(std::string_view acc) const = 0;
    virtual void scan_taxonomy(
        const std::function<void(std::string_view acc,
                                 std::string_view taxonomy)>& cb) const = 0;

    // Combined taxonomy + genome-id scan in one pass.
    // MultiPackReader overrides this with a .meta.tsv fast path (bypasses TAXN
    // decompression).  Default: builds acc→gid from scan_genome_accessions then
    // calls scan_taxonomy, so callers do not need to do two separate passes.
    virtual void scan_taxonomy_with_id(
        const std::function<void(std::string_view acc,
                                 std::string_view taxonomy,
                                 genopack::GenomeId id)>& cb) const {
        std::unordered_map<std::string, genopack::GenomeId> acc_to_gid;
        scan_genome_accessions([&](std::string_view acc, genopack::GenomeId gid) {
            acc_to_gid.emplace(acc, gid);
        });
        scan_taxonomy([&](std::string_view acc, std::string_view tax) {
            auto it = acc_to_gid.find(std::string(acc));
            cb(acc, tax, it != acc_to_gid.end() ? it->second : 0);
        });
    }
};

// Thin wrapper over a single ArchiveReader.
class SinglePackReader : public IPackReader {
public:
    explicit SinglePackReader(std::unique_ptr<genopack::ArchiveReader> r)
        : reader_(std::move(r)) {}

    void scan_genome_accessions(
        const std::function<void(std::string_view, genopack::GenomeId)>& cb) const override {
        reader_->scan_genome_accessions(cb);
    }

    std::optional<genopack::GenomeMeta> genome_meta_by_accession(
        std::string_view acc) const override {
        return reader_->genome_meta_by_accession(acc);
    }

    bool has_sketches() const override { return reader_->has_sketches(); }

    std::optional<genopack::SketchResult> sketch_for(
        genopack::GenomeId id) const override {
        return reader_->sketch_for(id);
    }

    std::optional<genopack::SketchResult> sketch_for(
        genopack::GenomeId id, uint32_t k, uint32_t sz) const override {
        return reader_->sketch_for(id, k, sz);
    }

    uint32_t sketch_kmer_size()   const override { return reader_->sketch_kmer_size(); }
    uint32_t sketch_sketch_size() const override { return reader_->sketch_sketch_size(); }

    bool has_kmer_size(uint32_t k) const override {
        auto ks = reader_->available_sketch_kmer_sizes();
        return std::find(ks.begin(), ks.end(), k) != ks.end();
    }

    std::vector<uint32_t> available_kmer_sizes() const override {
        return reader_->available_sketch_kmer_sizes();
    }

    void visit_shard_batches(
        const std::vector<std::string>& accessions,
        const std::function<void(genopack::ArchiveReader::ShardBatch&)>& cb) const override {
        reader_->visit_shard_batches(accessions, cb);
    }

    void visit_sketch_batches(
        const std::vector<std::string>& accessions,
        uint32_t k, uint32_t sz,
        const std::function<void(size_t idx,
                                 const genopack::SketchResult& sk)>& cb) const override {
        // Resolve accessions → (genome_id, original_idx), sort by genome_id so
        // sketch_for_ids decompresses each SKCH frame at most once.
        std::vector<std::pair<genopack::GenomeId, size_t>> id_idx;
        id_idx.reserve(accessions.size());
        for (size_t i = 0; i < accessions.size(); ++i) {
            auto meta = reader_->genome_meta_by_accession(accessions[i]);
            if (meta) id_idx.emplace_back(meta->genome_id, i);
        }
        std::sort(id_idx.begin(), id_idx.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        std::vector<genopack::GenomeId> sorted_ids;
        sorted_ids.reserve(id_idx.size());
        for (const auto& [gid, _] : id_idx) sorted_ids.push_back(gid);

        size_t pos = 0;
        reader_->sketch_for_ids(sorted_ids, k, sz,
            [&](size_t batch_idx, const genopack::SketchResult& sk) {
                cb(id_idx[batch_idx].second, sk);
            });
        (void)pos;
    }

    void release_sketches() const override { reader_->release_sketches(); }
    size_t sketch_memory_bytes() const override { return reader_->sketch_memory_bytes(); }

    const float* kmer_profile(genopack::GenomeId id) const override {
        return reader_->kmer_profile(id);
    }
    const float* kmer_profile_by_accession(std::string_view acc) const override {
        return reader_->kmer_profile_by_accession(acc);
    }

    bool has_gstx() const override { return reader_->has_gstx(); }
    const genopack::GstxEntry* gstx_for_genus(std::string_view genus) const override {
        return reader_->gstx_for_genus(genus);
    }

    bool has_qual() const override { return reader_->has_qual(); }
    void scan_qual(const std::function<void(const genopack::QualRecord&)>& cb) const override {
        reader_->scan_qual(cb);
    }

    uint16_t archive_idx_for_accession(std::string_view /*acc*/) const override { return 0; }
    size_t n_archives() const override { return 1; }

    std::string taxonomy_for_accession(std::string_view acc) const override {
        auto t = reader_->taxonomy_for_accession(acc);
        return t ? std::string(*t) : std::string{};
    }
    void scan_taxonomy(
        const std::function<void(std::string_view, std::string_view)>& cb) const override {
        reader_->scan_taxonomy(cb);
    }

private:
    std::unique_ptr<genopack::ArchiveReader> reader_;
};

} // namespace derep
