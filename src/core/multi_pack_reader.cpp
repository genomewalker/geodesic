#include "multi_pack_reader.hpp"
#include <algorithm>
#include <fcntl.h>
#include <filesystem>
#include <numeric>
#include <stdexcept>
#include <unistd.h>
#include <utility>
#include <vector>
#include <spdlog/spdlog.h>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace derep {

namespace fs = std::filesystem;

std::unique_ptr<MultiPackReader>
MultiPackReader::open_dir(const fs::path& parts_dir) {
    std::vector<fs::path> gpk_paths;
    for (const auto& entry : fs::directory_iterator(parts_dir)) {
        if (entry.path().extension() == ".gpk" &&
            (entry.is_directory() || entry.is_regular_file() || entry.is_symlink()))
            gpk_paths.push_back(entry.path());
    }
    if (gpk_paths.empty())
        throw std::runtime_error("No .gpk directories found in: " + parts_dir.string());
    std::sort(gpk_paths.begin(), gpk_paths.end());

    auto mp = std::unique_ptr<MultiPackReader>(new MultiPackReader());
    const size_t n_arch = gpk_paths.size();
    mp->archives_.resize(n_arch);
    mp->arch_sketch_mu_ = std::vector<std::mutex>(n_arch);

    // Per-archive accession lists; merged serially after the parallel open so
    // the shared acc_to_arch_ map stays lock-free. Each archive file is
    // independent on disk, so open + scan_genome_accessions run concurrently.
    std::vector<std::vector<std::pair<std::string, uint16_t>>> per_arch_accs(n_arch);

    #pragma omp parallel for schedule(dynamic, 1) num_threads(static_cast<int>(n_arch))
    for (size_t aidx = 0; aidx < n_arch; ++aidx) {
        ArchiveEntry entry;
        entry.reader = std::make_unique<genopack::ArchiveReader>();
        entry.reader->open(gpk_paths[aidx]);
        entry.path              = gpk_paths[aidx];
        entry.has_sketches_flag = entry.reader->has_sketches();

        const auto aidx16 = static_cast<uint16_t>(aidx);
        auto& local = per_arch_accs[aidx];
        entry.reader->scan_genome_accessions(
            [&](std::string_view acc, genopack::GenomeId /*local_id*/) {
                local.emplace_back(std::string(acc), aidx16);
            });

        spdlog::debug("multi_pack: opened {} ({} genomes)",
                      gpk_paths[aidx].string(), local.size());
        mp->archives_[aidx] = std::move(entry);
    }

    size_t total = 0;
    for (const auto& v : per_arch_accs) total += v.size();
    mp->acc_to_arch_.reserve(total);
    for (auto& v : per_arch_accs)
        for (auto& kv : v)
            mp->acc_to_arch_.emplace(std::move(kv.first), kv.second);

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

uint16_t MultiPackReader::archive_idx_for_accession(std::string_view acc) const {
    auto it = acc_to_arch_.find(std::string(acc));
    if (it == acc_to_arch_.end()) return UINT16_MAX;
    return it->second;
}

bool MultiPackReader::has_sketches() const {
    for (const auto& e : archives_)
        if (e.has_sketches_flag) return true;
    return false;
}

void MultiPackReader::touch_archive_(size_t aidx) const {
    std::lock_guard<std::mutex> lock(lru_mu_);
    auto it = lru_pos_.find(aidx);
    if (it != lru_pos_.end()) {
        // Already hot — move to front (MRU).
        lru_order_.splice(lru_order_.begin(), lru_order_, it->second);
        it->second = lru_order_.begin();
        return;
    }
    // New archive — add to front.
    lru_order_.push_front(aidx);
    lru_pos_[aidx] = lru_order_.begin();

    // Evict from back until within budget.
    while (lru_order_.size() > max_hot_archives_) {
        size_t victim = lru_order_.back();
        lru_order_.pop_back();
        lru_pos_.erase(victim);
        spdlog::debug("multi_pack LRU: releasing sketches for archive {}", victim);
        archives_[victim].reader->release_sketches();
    }
}

std::optional<genopack::SketchResult>
MultiPackReader::sketch_for(genopack::GenomeId virt_id) const {
    auto [aidx, local_id] = decode_virt(virt_id);
    if (aidx >= archives_.size()) return std::nullopt;
    touch_archive_(aidx);
    return archives_[aidx].reader->sketch_for(local_id);
}

std::optional<genopack::SketchResult>
MultiPackReader::sketch_for(genopack::GenomeId virt_id, uint32_t k, uint32_t sz) const {
    auto [aidx, local_id] = decode_virt(virt_id);
    if (aidx >= archives_.size()) return std::nullopt;
    touch_archive_(aidx);
    return archives_[aidx].reader->sketch_for(local_id, k, sz);
}

uint32_t MultiPackReader::sketch_kmer_size() const {
    for (const auto& e : archives_) {
        uint32_t k = e.reader->sketch_kmer_size();
        if (k > 0) return k;
    }
    return 0;
}

uint32_t MultiPackReader::sketch_sketch_size() const {
    for (const auto& e : archives_) {
        uint32_t sz = e.reader->sketch_sketch_size();
        if (sz > 0) return sz;
    }
    return 0;
}

const float* MultiPackReader::kmer_profile(genopack::GenomeId virt_id) const {
    auto [aidx, local_id] = decode_virt(virt_id);
    if (aidx >= archives_.size()) return nullptr;
    return archives_[aidx].reader->kmer_profile(local_id);
}

const float* MultiPackReader::kmer_profile_by_accession(std::string_view acc) const {
    auto it = acc_to_arch_.find(std::string(acc));
    if (it == acc_to_arch_.end()) return nullptr;
    return archives_[it->second].reader->kmer_profile_by_accession(acc);
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

void MultiPackReader::visit_sketch_batches(
    const std::vector<std::string>& accessions,
    uint32_t k, uint32_t sz,
    const std::function<void(size_t idx,
                             const genopack::SketchResult& sk)>& cb) const
{
    // Group accessions by archive — one struct per archive.
    struct ArchSlice {
        std::vector<size_t>               global_idx;
        std::vector<genopack::GenomeId>   local_ids;
    };
    std::vector<ArchSlice> slices(archives_.size());

    for (size_t i = 0; i < accessions.size(); ++i) {
        auto it = acc_to_arch_.find(accessions[i]);
        if (it == acc_to_arch_.end()) continue;
        const uint16_t aidx = it->second;
        auto meta = archives_[aidx].reader->genome_meta_by_accession(accessions[i]);
        if (!meta) continue;
        slices[aidx].global_idx.push_back(i);
        slices[aidx].local_ids.push_back(meta->genome_id);
    }

    // For each archive: use sketch_for_ids() which decompresses only the frames
    // that contain the requested genomes (V3 archives) or falls back to per-genome
    // lookup (V1/V2 archives). local_ids must be sorted ascending for sketch_for_ids.
    const size_t n_arch = archives_.size();
    for (size_t aidx = 0; aidx < n_arch; ++aidx) {
        auto& sl = slices[aidx];
        if (sl.global_idx.empty()) continue;

        // Sort local_ids ascending; keep global_idx in sync.
        std::vector<size_t> ord(sl.local_ids.size());
        std::iota(ord.begin(), ord.end(), 0);
        std::sort(ord.begin(), ord.end(),
                  [&](size_t a, size_t b) { return sl.local_ids[a] < sl.local_ids[b]; });

        std::vector<genopack::GenomeId> sorted_ids(sl.local_ids.size());
        std::vector<size_t>             sorted_gidx(sl.local_ids.size());
        for (size_t i = 0; i < ord.size(); ++i) {
            sorted_ids[i]  = sl.local_ids[ord[i]];
            sorted_gidx[i] = sl.global_idx[ord[i]];
        }

        {
            std::lock_guard<std::mutex> lk(arch_sketch_mu_[aidx]);
            archives_[aidx].reader->sketch_for_ids(sorted_ids, k, sz,
                [&](size_t local_idx, const genopack::SketchResult& sk) {
                    cb(sorted_gidx[local_idx], sk);
                });
            archives_[aidx].reader->release_sketches();
            fadvise_dontneed_(archives_[aidx].path);
        }
    }
}

void MultiPackReader::fadvise_dontneed_(const fs::path& p) const noexcept {
    auto advise_file = [](const fs::path& fp) noexcept {
        int fd = ::open(fp.c_str(), O_RDONLY);
        if (fd < 0) return;
        ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        ::close(fd);
    };
    std::error_code ec;
    if (fs::is_directory(p, ec)) {
        for (const auto& e : fs::directory_iterator(p, ec))
            if (e.is_regular_file()) advise_file(e.path());
    } else {
        advise_file(p);
    }
}

} // namespace derep
