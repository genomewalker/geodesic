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

inline int ani_reader_count(int threads, int requested) {
    return requested ? std::clamp(requested, 1, 64) : std::max(1, std::min(8, threads / 4));
}

// Readers only copy compressed bytes. A separate fixed pool performs all
// shard initialisation, decoding and sketching, without nested OpenMP teams.
// Slots cover reads, queued shards AND shards currently used by workers.
template<class Visitor>
void visit_shard_genomes(IPackReader& pack, const std::vector<std::string>& accessions,
                        int threads, int requested_readers, const Visitor& visit) {
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
        std::vector<uint8_t> bytes;
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
                job->bytes.resize(loc.size);
                size_t done = 0;
                while (done < job->bytes.size()) {
                    ssize_t n = ::pread(loc.fd, job->bytes.data() + done,
                                        job->bytes.size() - done, loc.offset + done);
                    if (n < 0 && errno == EINTR) continue;
                    if (n <= 0) throw std::runtime_error("ANI shard pread failed or unexpected EOF");
                    done += static_cast<size_t>(n);
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
                std::call_once(job->opened, [&] { job->shard.open(job->bytes.data(), 0, job->bytes.size()); });
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
