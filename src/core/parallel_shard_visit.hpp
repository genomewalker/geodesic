#pragma once
#include "pack_reader.hpp"
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <deque>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <semaphore>
#include <stdexcept>
#include <thread>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace derep {

// Auto is a flat 16 readers whatever -t is: they only pread, and the small ranged
// reads of partial shards are latency-bound, so 8 readers leave -t 32 short of data.
inline int ani_reader_count(int, int requested) {
    return requested ? std::clamp(requested, 1, 64) : 16;
}

// Byte ranges [begin, end) of a shard that ShardReader touches to fetch the
// local genome `ids`: header, directory, dictionary, codec reference blobs, and
// each wanted blob with its checkpoints. Sorted, merged across gaps <= `gap`.
// `head` holds the shard's first bytes, through the end of the directory.
inline std::vector<std::pair<uint64_t, uint64_t>>
shard_read_ranges(const uint8_t* head, uint64_t size, std::vector<uint64_t> ids, uint64_t gap = 1 << 20) {
    using genopack::GenomeDirEntry, genopack::CheckpointEntry;
    const auto& h = *reinterpret_cast<const genopack::ShardHeader*>(head);
    if (h.magic != genopack::GPKS_MAGIC || h.codec > 4) return {{0, size}}; // ShardReader reports it.
    const auto* dir = reinterpret_cast<const GenomeDirEntry*>(head + h.genome_dir_offset);
    std::vector<std::pair<uint64_t, uint64_t>> ranges{{0, h.genome_dir_offset + uint64_t(h.n_genomes) * sizeof(GenomeDirEntry)}};
    auto add = [&](const GenomeDirEntry& e) {
        const uint64_t blob = h.blob_area_offset + e.blob_offset;
        ranges.emplace_back(blob, blob + e.blob_len_cmp);
        if (!h.checkpoint_area_offset || !e.n_checkpoints) return;
        const uint64_t cps = h.checkpoint_area_offset + uint64_t(e.checkpoint_idx) * sizeof(CheckpointEntry);
        ranges.emplace_back(cps, cps + uint64_t(e.n_checkpoints) * sizeof(CheckpointEntry));
    };
    if (h.dict_size && (h.codec == 1 || h.codec == 2)) ranges.emplace_back(h.dict_offset, h.dict_offset + h.dict_size);
    if (h.codec == 3 && h.n_genomes) {
        add(dir[0]);
        if (h.version >= 4 && h.flags < h.n_genomes) add(dir[h.flags]);
    }
    if (h.codec == 4) for (uint32_t i = 0; i < std::min(h.dict_size ? h.dict_size : 1u, h.n_genomes); ++i) add(dir[i]);
    std::sort(ids.begin(), ids.end());
    for (uint32_t i = 0; i < h.n_genomes; ++i)
        if (std::binary_search(ids.begin(), ids.end(), dir[i].genome_id)) add(dir[i]);
    for (auto& [begin, end] : ranges) end = std::min(end, size), begin = std::min(begin, end);
    std::sort(ranges.begin(), ranges.end());
    std::vector<std::pair<uint64_t, uint64_t>> merged;
    for (auto [begin, end] : ranges) {
        if (!merged.empty() && begin <= merged.back().second + gap) merged.back().second = std::max(merged.back().second, end);
        else merged.emplace_back(begin, end);
    }
    return merged;
}

// Readers only copy compressed bytes. A separate fixed pool performs all
// shard initialisation, decoding and sketching, without nested OpenMP teams.
// Slots cover reads, queued shards AND shards currently used by workers.
// Unneeded byte runs longer than `read_gap` are skipped, not read.
template<class Visitor>
void visit_shard_genomes(IPackReader& pack, const std::vector<std::string>& accessions,
                        int threads, int requested_readers, const Visitor& visit,
                        uint64_t read_gap = 1 << 20) {
    using Key = std::pair<uint16_t, uint32_t>;
    struct Request { size_t index; genopack::GenomeMeta meta; };
    std::map<Key, std::vector<Request>> requests;
    for (size_t i = 0; i < accessions.size(); ++i) {
        auto meta = pack.genome_meta_by_accession(accessions[i]);
        if (meta) requests[{pack.archive_idx_for_accession(accessions[i]), meta->shard_id}].push_back({i, *meta});
    }
    struct Source { IPackReader::ShardLocation location; const std::vector<Request>* requests; };
    std::vector<Source> sources;
    for (auto location : pack.shard_locations()) {
        auto it = requests.find({location.archive, location.shard});
        if (it != requests.end()) sources.push_back({location, &it->second});
    }
    if (sources.empty()) return;
    const int readers = std::min(size_t(ani_reader_count(threads, requested_readers)), sources.size());
    std::counting_semaphore<> slots(2 * readers);
    struct Job {
        explicit Job(std::counting_semaphore<>& slots) : slots(slots) {}
        ~Job() { slots.release(); }
        std::counting_semaphore<>& slots;
        const std::vector<Request>* requests;
        std::unique_ptr<uint8_t[]> bytes; // Only the ranges ShardReader touches are read.
        size_t size = 0;
        genopack::ShardReader shard;
        std::once_flag opened;
        size_t next = 0;
    };
    std::deque<std::shared_ptr<Job>> queue;
    std::mutex mutex;
    std::condition_variable ready;
    std::exception_ptr error;
    std::atomic<bool> failed{false};
    int active_readers = readers;
    auto record_error = [&] {
        std::lock_guard lock(mutex);
        if (!error) error = std::current_exception();
        failed = true;
    };
    auto read = [&](int band) {
        try {
            for (size_t i = band * sources.size() / readers;
                 i < (band + 1) * sources.size() / readers && !failed; ++i) {
                slots.acquire();
                // Return the slot even if allocation fails.
                std::shared_ptr<Job> job;
                try { job = std::make_shared<Job>(slots); }
                catch (...) { slots.release(); throw; }
                job->requests = sources[i].requests;
                const auto& loc = sources[i].location;
                job->bytes = std::make_unique_for_overwrite<uint8_t[]>(loc.size);
                job->size = loc.size;
                auto pread_all = [&](uint64_t begin, uint64_t end) {
                    while (begin < end) {
                        ssize_t n = ::pread(loc.fd, job->bytes.get() + begin, end - begin, loc.offset + begin);
                        if (n < 0 && errno == EINTR) continue;
                        if (n <= 0) throw std::runtime_error("ANI shard pread failed or unexpected EOF");
                        begin += static_cast<uint64_t>(n);
                    }
                };
                // Header, then the directory that follows it, then the planned ranges.
                uint64_t have = std::min<uint64_t>(loc.size, sizeof(genopack::ShardHeader));
                pread_all(0, have);
                if (have >= sizeof(genopack::ShardHeader)) {
                    const auto& h = *reinterpret_cast<const genopack::ShardHeader*>(job->bytes.get());
                    const uint64_t dir_end = std::min<uint64_t>(loc.size,
                        h.genome_dir_offset + uint64_t(h.n_genomes) * sizeof(genopack::GenomeDirEntry));
                    if (dir_end > have) pread_all(have, dir_end), have = dir_end;
                    std::vector<uint64_t> ids;
                    for (const auto& req : *job->requests) ids.push_back(req.meta.genome_id & ((uint64_t(1) << 48) - 1));
                    for (auto [begin, end] : shard_read_ranges(job->bytes.get(), loc.size, std::move(ids), read_gap))
                        pread_all(std::max(begin, have), end);
                }
                {
                    std::lock_guard lock(mutex);
                    queue.push_back(std::move(job));
                }
                ready.notify_all();
            }
        } catch (...) { record_error(); }
        {
            std::lock_guard lock(mutex);
            --active_readers;
        }
        ready.notify_all();
    };
    auto compute = [&] {
#ifdef _OPENMP
        // These threads own all decode and sketch work; no hidden nested teams.
        omp_set_max_active_levels(0);
#endif
        for (;;) {
            std::shared_ptr<Job> job;
            size_t index;
            {
                std::unique_lock lock(mutex);
                ready.wait(lock, [&] { return !queue.empty() || active_readers == 0; });
                if (queue.empty()) return;
                job = queue.front();
                index = job->next++;
                if (job->next == job->requests->size()) queue.pop_front();
            }
            if (failed) continue; // Drain slots so producers can always finish.
            try {
                std::call_once(job->opened, [&] { job->shard.open(job->bytes.get(), 0, job->size); });
                const auto& req = job->requests->at(index);
                genopack::ExtractedGenome genome;
                genome.meta = req.meta;
                // MultiPackReader metadata contains virtual IDs; shard IDs are local.
                const auto local_id = req.meta.genome_id & ((uint64_t(1) << 48) - 1);
                try { genome.fasta = job->shard.fetch_genome(local_id); }
                catch (const std::exception&) { genome.fasta.clear(); } // Match genopack's visitor.
                genome.accession = accessions[req.index];
                visit(req.index, genome);
            } catch (...) { record_error(); }
        }
    };
    {
        // Start consumers first. If a thread cannot be created, wake and join
        // the ones already started rather than leaving them blocked forever.
        std::vector<std::jthread> workers, io;
        try {
            for (int i = 0; i < std::max(1, threads); ++i) workers.emplace_back(compute);
            for (int i = 0; i < readers; ++i) io.emplace_back(read, i);
        } catch (...) {
            record_error();
            {
                std::lock_guard lock(mutex);
                active_readers -= readers - static_cast<int>(io.size());
            }
            ready.notify_all();
        }
        // io joins before workers, which continue draining the queue.
    }
    if (error) std::rethrow_exception(error);
}
} // namespace derep
