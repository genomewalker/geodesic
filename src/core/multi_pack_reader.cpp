#include "multi_pack_reader.hpp"
#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <spdlog/spdlog.h>

namespace derep {

namespace fs = std::filesystem;

std::unique_ptr<MultiPackReader>
MultiPackReader::open_dir(const fs::path& parts_dir) {
    std::vector<fs::path> gpk_paths;
    for (const auto& entry : fs::directory_iterator(parts_dir)) {
        if (entry.is_directory() && entry.path().extension() == ".gpk")
            gpk_paths.push_back(entry.path());
    }
    if (gpk_paths.empty())
        throw std::runtime_error("No .gpk directories found in: " + parts_dir.string());
    std::sort(gpk_paths.begin(), gpk_paths.end());

    auto mp = std::unique_ptr<MultiPackReader>(new MultiPackReader());
    mp->archives_.reserve(gpk_paths.size());

    for (size_t aidx = 0; aidx < gpk_paths.size(); ++aidx) {
        ArchiveEntry entry;
        entry.reader = std::make_unique<genopack::ArchiveReader>();
        entry.reader->open(gpk_paths[aidx]);
        entry.has_sketches_flag = entry.reader->has_sketches();

        const auto aidx16 = static_cast<uint16_t>(aidx);
        entry.reader->scan_genome_accessions(
            [&](std::string_view acc, genopack::GenomeId /*local_id*/) {
                mp->acc_to_arch_.emplace(std::string(acc), aidx16);
            });

        spdlog::debug("multi_pack: opened {} ({} genomes)",
                      gpk_paths[aidx].string(),
                      mp->acc_to_arch_.size());
        mp->archives_.push_back(std::move(entry));
    }

    spdlog::info("multi_pack: {} archives, {} genomes total",
                 mp->archives_.size(), mp->acc_to_arch_.size());
    return mp;
}

void MultiPackReader::scan_genome_accessions(
    const std::function<void(std::string_view, genopack::GenomeId)>& cb) const
{
    for (size_t aidx = 0; aidx < archives_.size(); ++aidx) {
        const auto aidx16 = static_cast<uint16_t>(aidx);
        archives_[aidx].reader->scan_genome_accessions(
            [&](std::string_view acc, genopack::GenomeId local_id) {
                cb(acc, encode_virt(aidx16, local_id));
            });
    }
}

std::optional<genopack::GenomeMeta>
MultiPackReader::genome_meta_by_accession(std::string_view acc) const {
    auto it = acc_to_arch_.find(std::string(acc));
    if (it == acc_to_arch_.end()) return std::nullopt;

    const uint16_t aidx = it->second;
    auto meta = archives_[aidx].reader->genome_meta_by_accession(acc);
    if (!meta) return std::nullopt;
    meta->genome_id = encode_virt(aidx, meta->genome_id);
    return meta;
}

bool MultiPackReader::has_sketches() const {
    for (const auto& e : archives_)
        if (e.has_sketches_flag) return true;
    return false;
}

std::optional<genopack::SketchResult>
MultiPackReader::sketch_for(genopack::GenomeId virt_id) const {
    auto [aidx, local_id] = decode_virt(virt_id);
    if (aidx >= archives_.size()) return std::nullopt;
    return archives_[aidx].reader->sketch_for(local_id);
}

void MultiPackReader::visit_shard_batches(
    const std::vector<std::string>& accessions,
    const std::function<void(genopack::ArchiveReader::ShardBatch&)>& cb) const
{
    struct ArchSlice {
        std::vector<std::string> accs;
        std::vector<size_t>      global_idx;
    };
    std::vector<ArchSlice> slices(archives_.size());

    for (size_t i = 0; i < accessions.size(); ++i) {
        auto it = acc_to_arch_.find(accessions[i]);
        if (it != acc_to_arch_.end()) {
            auto& sl = slices[it->second];
            sl.accs.push_back(accessions[i]);
            sl.global_idx.push_back(i);
        }
    }

    for (size_t aidx = 0; aidx < archives_.size(); ++aidx) {
        auto& sl = slices[aidx];
        if (sl.accs.empty()) continue;
        archives_[aidx].reader->visit_shard_batches(sl.accs,
            [&](genopack::ArchiveReader::ShardBatch& batch) {
                genopack::ArchiveReader::ShardBatch remapped;
                remapped.reserve(batch.size());
                for (auto& [local_idx, genome] : batch)
                    remapped.emplace_back(sl.global_idx[local_idx], std::move(genome));
                cb(remapped);
            });
    }
}

} // namespace derep
