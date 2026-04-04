#pragma once
#include "pack_reader.hpp"
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

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

    void visit_shard_batches(
        const std::vector<std::string>& accessions,
        const std::function<void(genopack::ArchiveReader::ShardBatch&)>& cb) const override;

    size_t n_archives() const { return archives_.size(); }
    size_t n_genomes()  const { return acc_to_arch_.size(); }

private:
    struct ArchiveEntry {
        std::unique_ptr<genopack::ArchiveReader> reader;
        bool has_sketches_flag = false;
    };

    std::vector<ArchiveEntry> archives_;
    // accession → archive_idx (for O(1) routing)
    std::unordered_map<std::string, uint16_t> acc_to_arch_;

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
