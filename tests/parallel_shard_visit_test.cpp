#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include "core/parallel_shard_visit.hpp"
#include "core/multi_pack_reader.hpp"
#include <cstdlib>
#include <fstream>
#include <atomic>
#include <stdexcept>

namespace {
class FakePack : public derep::SinglePackReader {
public:
    FakePack() : SinglePackReader(nullptr) {}
    bool fail_reader = false;
    mutable std::atomic<int> live{0}, peak{0};
    std::optional<genopack::GenomeMeta> genome_meta_by_accession(std::string_view a) const override {
        if (a == "missing") return std::nullopt;
        genopack::GenomeMeta m;
        m.shard_id = std::stoi(std::string(a)) / 2;
        return m;
    }
    uint16_t archive_idx_for_accession(std::string_view a) const override {
        return std::stoi(std::string(a)) % 2;
    }
    void visit_shard_batches_parallel(const std::vector<std::string>& accs, int readers,
        const std::function<void(genopack::ArchiveReader::ShardBatch&)>& cb) const override {
        if (readers != 1) throw std::runtime_error("unbudgeted genopack readers");
        if (fail_reader) throw std::runtime_error("reader error");
        for (size_t i = 0; i < accs.size(); ++i) {
            // Reproduce genopack's explicit nested decode team.
            int decode_team = 1;
#ifdef _OPENMP
            #pragma omp parallel num_threads(8)
            #pragma omp single
            decode_team = omp_get_num_threads();
#endif
            if (decode_team != 1) throw std::runtime_error("nested decode oversubscribed");
            genopack::ArchiveReader::ShardBatch batch(1);
            batch[0].first = i;
            batch[0].second.accession = accs[i];
            batch[0].second.fasta = accs[i];
            int current = ++live;
            int old = peak.load();
            while (old < current && !peak.compare_exchange_weak(old, current)) {}
            cb(batch);
            --live;
        }
    }
};
}
TEST_CASE("ANI reader budgets reserve sketch capacity", "[parallel-pack]") {
    REQUIRE(derep::ani_reader_count(32, 0) == 8);
    REQUIRE(derep::ani_reader_count(8, 0) == 2);
    REQUIRE(derep::ani_reader_count(1, 0) == 1);
    REQUIRE(derep::ani_reader_count(2, 99) == 1);
    REQUIRE(derep::ani_reader_count(8, 99) == 7);
}
TEST_CASE("Shard tasks preserve caller indices and serialize nested decoding", "[parallel-pack]") {
    for (int threads : {1, 2, 8}) {
        FakePack pack;
        std::vector<std::string> accs{"7", "0", "missing", "2", "7", "1", "6", "3", "4", "5"};
        std::vector<std::string> results(accs.size());
        derep::visit_shard_genomes(pack, accs, threads, 2,
            [&](size_t i, genopack::ExtractedGenome& g) { results[i] = g.fasta; });
        for (size_t i = 0; i < accs.size(); ++i)
            REQUIRE(results[i] == (accs[i] == "missing" ? "" : accs[i]));
        REQUIRE(pack.live == 0);
        REQUIRE(pack.peak <= derep::ani_reader_count(threads, 2));
    }
}
TEST_CASE("Shard task errors reach the caller after all tasks finish", "[parallel-pack]") {
    FakePack pack;
    std::vector<std::string> accs{"0", "1", "2", "3"};
    REQUIRE_THROWS_WITH(derep::visit_shard_genomes(pack, accs, 8, 2,
        [](size_t, genopack::ExtractedGenome&) { throw std::runtime_error("sketch error"); }), "sketch error");
    REQUIRE(pack.live == 0);
    pack.fail_reader = true;
    REQUIRE_THROWS_WITH(derep::visit_shard_genomes(pack, accs, 8, 2,
        [](size_t, genopack::ExtractedGenome&) {}), "reader error");
}

TEST_CASE("Parallel multi-pack batches match serial caller indices", "[parallel-pack][integration]") {
    const char* path = std::getenv("GEODESIC_TEST_PACK");
    const char* list = std::getenv("GEODESIC_TEST_ACCESSIONS");
    if (!path || !list) SKIP("Set GEODESIC_TEST_PACK and GEODESIC_TEST_ACCESSIONS for real-pack coverage");
    auto pack = derep::MultiPackReader::open_dir(path);
    std::ifstream input(list);
    REQUIRE(input.good());
    std::map<uint16_t, std::vector<std::string>> selected;
    for (std::string acc; std::getline(input, acc);) {
        if (!pack->genome_meta_by_accession(acc)) continue;
        auto archive = pack->archive_idx_for_accession(acc);
        if (selected.count(archive) || selected.size() < 3) {
            if (selected[archive].size() < 2) selected[archive].push_back(acc);
        }
    }
    REQUIRE(selected.size() >= 2);
    std::vector<std::string> accs{"__missing_parallel_reader__"};
    for (const auto& [archive, names] : selected)
        accs.insert(accs.end(), names.begin(), names.end());
    accs.push_back(accs[1]);
    std::reverse(accs.begin(), accs.end());
    std::vector<std::string> expected(accs.size());
    pack->visit_shard_batches(accs, [&](auto& batch) {
        for (auto& [i, genome] : batch) expected.at(i) = genome.fasta;
    });
    for (int readers : {1, 2, 8}) {
        std::vector<std::string> actual(accs.size());
        pack->visit_shard_batches_parallel(accs, readers, [&](auto& batch) {
            for (auto& [i, genome] : batch) actual.at(i) = genome.fasta;
        });
        REQUIRE(actual == expected);
        REQUIRE_THROWS_WITH(pack->visit_shard_batches_parallel(accs, readers,
            [](auto&) { throw std::runtime_error("callback error"); }), "callback error");
    }
}
