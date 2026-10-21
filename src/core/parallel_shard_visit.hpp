#pragma once
#include "pack_reader.hpp"
#include <exception>
#include <map>
#include <mutex>
#include <thread>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace derep {

inline int ani_reader_count(int threads, int requested) {
    const int automatic = std::max(1, std::min(4, threads / 8));
    return std::clamp(requested ? requested : automatic, 1, std::max(1, threads / 8));
}

// Each reader owns a fixed CPU share, reused for decode and sketch phases.
// genopack explicitly requests up to eight decode threads, so admit a reader
// only when eight CPUs are available. Below eight CPUs its decode is serial.
// Reader lifetimes also bound compressed double buffers and decoded batches.
template<class Visitor>
void visit_shard_genomes(IPackReader& pack, const std::vector<std::string>& accessions,
                        int threads, int requested_readers, const Visitor& visit) {
    using Key = std::pair<uint16_t, uint32_t>;
    std::map<Key, std::vector<size_t>> shards;
    for (size_t i = 0; i < accessions.size(); ++i) {
        auto meta = pack.genome_meta_by_accession(accessions[i]);
        if (meta) shards[{pack.archive_idx_for_accession(accessions[i]), meta->shard_id}].push_back(i);
    }
    if (shards.empty()) return;
    int readers = std::min(size_t(ani_reader_count(threads, requested_readers)), shards.size());
#ifndef _OPENMP
    readers = 1;
#endif
    struct Band { std::vector<std::string> accs; std::vector<size_t> indices; };
    std::vector<Band> bands(readers);
    size_t ordinal = 0;
    for (const auto& [key, indices] : shards) {
        auto& band = bands[ordinal++ * readers / shards.size()];
        for (size_t i : indices) { band.accs.push_back(accessions[i]); band.indices.push_back(i); }
    }
    std::exception_ptr error;
    std::mutex error_mutex;
    auto record_error = [&] {
        std::lock_guard<std::mutex> lock(error_mutex);
        if (!error) error = std::current_exception();
    };
    auto work = [&](int b) {
#ifdef _OPENMP
        const int previous_levels = omp_get_max_active_levels();
        const int previous_dynamic = omp_get_dynamic();
        omp_set_dynamic(0);
        omp_set_max_active_levels(threads >= 8 ? 1 : 0);
#endif
        try {
            const auto& band = bands[b];
            pack.visit_shard_batches_parallel(band.accs, 1,
                [&](genopack::ArchiveReader::ShardBatch& batch) {
#ifdef _OPENMP
                    omp_set_max_active_levels(1);
                    #pragma omp parallel for num_threads(threads / readers) schedule(dynamic, 1)
#endif
                    for (size_t i = 0; i < batch.size(); ++i) {
                        auto& entry = batch[i];
                        try { visit(band.indices[entry.first], entry.second); }
                        catch (...) { record_error(); }
                        // Release consumed FASTAs even when a visitor doesn't move them.
                        std::string().swap(entry.second.fasta);
                    }
#ifdef _OPENMP
                    omp_set_max_active_levels(threads >= 8 ? 1 : 0);
#endif
                });
        } catch (...) { record_error(); }
#ifdef _OPENMP
        omp_set_max_active_levels(previous_levels);
        omp_set_dynamic(previous_dynamic);
#endif
    };
    if (readers == 1) work(0);
    else {
        std::vector<std::jthread> workers;
        for (int b = 0; b < readers; ++b) workers.emplace_back(work, b);
    }
    if (error) std::rethrow_exception(error);
}
} // namespace derep
