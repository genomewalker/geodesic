#pragma once
#include "pack_reader.hpp"
#include <exception>
#include <map>
#include <mutex>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace derep {

inline int ani_reader_count(int threads, int requested) {
    const int automatic = std::max(1, std::min(8, threads / 4));
    return std::clamp(requested ? requested : automatic, 1, std::max(1, threads - 1));
}

// One shared OpenMP team: reader tasks own disjoint contiguous shard bands;
// sketch tasks share the rest of the team. A taskgroup retains each callback's
// batch until every genome is consumed, so at most R decoded batches are live.
// Each band calls the parallel API with one reader: genopack's internal decode
// regions then nest in our team and are serialized, rather than spawning 8R
// additional decoding threads. Its compressed prefetch buffers remain bounded.
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
#ifdef _OPENMP
    const int previous_levels = omp_get_max_active_levels();
    const int previous_dynamic = omp_get_dynamic();
    omp_set_dynamic(0);
    // A one-thread outer region is inactive, so disable all nested regions there.
    omp_set_max_active_levels(threads > 1 ? 1 : 0);
    #pragma omp parallel num_threads(threads)
    #pragma omp single
#endif
    {
        for (int b = 0; b < readers; ++b) {
#ifdef _OPENMP
            #pragma omp task firstprivate(b)
#endif
            {
                try {
                    const auto& band = bands[b];
                    pack.visit_shard_batches_parallel(band.accs, 1,
                        [&](genopack::ArchiveReader::ShardBatch& batch) {
#ifdef _OPENMP
                            #pragma omp taskgroup
#endif
                            {
                                for (auto& entry : batch) {
                                    auto* genome = &entry.second;
                                    const size_t idx = band.indices[entry.first];
#ifdef _OPENMP
                                    #pragma omp task firstprivate(genome, idx)
#endif
                                    {
                                        try { visit(idx, *genome); }
                                        catch (...) { record_error(); }
                                    }
                                }
                            }
                        });
                } catch (...) { record_error(); }
            }
        }
    }
#ifdef _OPENMP
    omp_set_max_active_levels(previous_levels);
    omp_set_dynamic(previous_dynamic);
#endif
    if (error) std::rethrow_exception(error);
}
} // namespace derep
