#pragma once
#include <genopack/archive.hpp>
#include <algorithm>
#include <functional>
#include <optional>
#include <string>
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

    virtual void release_sketches() const {}
    virtual size_t sketch_memory_bytes() const { return 0; }

    // TNF profile access (KMRX section — mmap'd, no decompression cost).
    // Returns pointer to float[136] L2-normalised k=4 tetranucleotide vector,
    // or nullptr if the archive has no KMRX section.
    virtual const float* kmer_profile(genopack::GenomeId id) const { return nullptr; }
    virtual const float* kmer_profile_by_accession(std::string_view acc) const { return nullptr; }

    virtual uint16_t archive_idx_for_accession(std::string_view acc) const = 0;
    virtual size_t n_archives() const = 0;

    // Taxonomy access (TAXN section).
    virtual std::string taxonomy_for_accession(std::string_view acc) const = 0;
    virtual void scan_taxonomy(
        const std::function<void(std::string_view acc,
                                 std::string_view taxonomy)>& cb) const = 0;
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
        // Single archive: no inter-archive thrashing. Load SKCH once, scan all.
        for (size_t i = 0; i < accessions.size(); ++i) {
            auto meta = reader_->genome_meta_by_accession(accessions[i]);
            if (!meta) continue;
            std::optional<genopack::SketchResult> sk;
            if (k > 0 && sz > 0) {
                sk = reader_->sketch_for(meta->genome_id, k, sz);
                if (!sk) sk = reader_->sketch_for(meta->genome_id);
            } else {
                sk = reader_->sketch_for(meta->genome_id);
            }
            if (sk) cb(i, *sk);
        }
    }

    void release_sketches() const override { reader_->release_sketches(); }
    size_t sketch_memory_bytes() const override { return reader_->sketch_memory_bytes(); }

    const float* kmer_profile(genopack::GenomeId id) const override {
        return reader_->kmer_profile(id);
    }
    const float* kmer_profile_by_accession(std::string_view acc) const override {
        return reader_->kmer_profile_by_accession(acc);
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
