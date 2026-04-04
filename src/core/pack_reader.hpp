#pragma once
#include <genopack/archive.hpp>
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

    virtual void visit_shard_batches(
        const std::vector<std::string>& accessions,
        const std::function<void(genopack::ArchiveReader::ShardBatch&)>& cb) const = 0;
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

    void visit_shard_batches(
        const std::vector<std::string>& accessions,
        const std::function<void(genopack::ArchiveReader::ShardBatch&)>& cb) const override {
        reader_->visit_shard_batches(accessions, cb);
    }

private:
    std::unique_ptr<genopack::ArchiveReader> reader_;
};

} // namespace derep
