#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include "core/parallel_shard_visit.hpp"
#include "core/multi_pack_reader.hpp"
#include <cstdlib>
#include <fstream>
#include <atomic>
#include <cstdio>
#include <set>
#include <stdexcept>

namespace {
class FakePack : public derep::SinglePackReader {
public:
    FakePack() : SinglePackReader(nullptr), file(std::tmpfile()) {
        if (!file) throw std::runtime_error("tmpfile failed");
        for (uint32_t shard = 0; shard < 4; ++shard) {
            genopack::ShardWriterConfig cfg;
            cfg.train_dict = false;
            cfg.compress_threads = 1;
            genopack::ShardWriter writer(shard, 0, cfg);
            for (int id = shard * 2; id < int(shard * 2 + 2); ++id) {
                auto fasta = std::to_string(id);
                writer.add_genome(id, 0, fasta.data(), fasta.size());
            }
            // Tiny fixtures should not depend on an external spill directory.
            struct InMemory {
                std::string previous;
                bool had_value;
                InMemory() : had_value(std::getenv("GENOPACK_SPILL_DIR") != nullptr) {
                    if (had_value) previous = std::getenv("GENOPACK_SPILL_DIR");
                    ::unsetenv("GENOPACK_SPILL_DIR");
                }
                ~InMemory() {
                    if (had_value) ::setenv("GENOPACK_SPILL_DIR", previous.c_str(), 1);
                }
            } in_memory;
            auto frozen = writer.freeze();
            if (frozen.bytes.empty() && !frozen.tmp_path.empty()) {
                std::ifstream in(frozen.tmp_path, std::ios::binary);
                frozen.bytes.assign(std::istreambuf_iterator<char>(in), {});
                ::unlink(frozen.tmp_path.c_str()); // exact file created by this fixture
            }
            auto offset = std::ftell(file);
            if (std::fwrite(frozen.bytes.data(), 1, frozen.bytes.size(), file) != frozen.bytes.size())
                throw std::runtime_error("fixture write failed");
            for (uint16_t archive : {0, 1})
                locations.push_back({archive, shard, fileno(file), uint64_t(offset), frozen.bytes.size()});
        }
        std::fflush(file);
    }
    ~FakePack() { std::fclose(file); }
    bool fail_reader = false;
    std::FILE* file;
    std::vector<ShardLocation> locations;
    std::optional<genopack::GenomeMeta> genome_meta_by_accession(std::string_view a) const override {
        if (a == "missing") return std::nullopt;
        genopack::GenomeMeta m;
        m.shard_id = std::stoi(std::string(a)) / 2;
        m.genome_id = (uint64_t(archive_idx_for_accession(a)) << 48) | std::stoi(std::string(a));
        return m;
    }
    uint16_t archive_idx_for_accession(std::string_view a) const override {
        return std::stoi(std::string(a)) % 2;
    }
    std::vector<ShardLocation> shard_locations() const override {
        auto result = locations;
        if (fail_reader) for (auto& loc : result) loc.fd = -1;
        return result;
    }
};
}
TEST_CASE("ANI I/O concurrency is independent of the compute budget", "[parallel-pack]") {
    for (int threads : {1, 4, 8, 16, 32, 64}) REQUIRE(derep::ani_reader_count(threads, 0) == 8);
    REQUIRE(derep::ani_reader_count(2, 16) == 16);
    REQUIRE(derep::ani_reader_count(8, 2) == 2);
    REQUIRE(derep::ani_reader_count(32, 16) == 16);
    REQUIRE(derep::ani_reader_count(32, 99) == 64);
}
TEST_CASE("Shard readers preserve caller indices and bound compute", "[parallel-pack]") {
    FakePack pack;
    for (int threads : {1, 2, 8, 16, 32}) for (int readers : {1, 2, 16}) {
        std::vector<std::string> accs{"7", "0", "missing", "2", "7", "1", "6", "3", "4", "5"};
        std::vector<std::string> results(accs.size());
        std::atomic<int> live{0}, peak{0};
        std::mutex ids_mutex;
        std::set<std::thread::id> ids;
        derep::visit_shard_genomes(pack, accs, threads, readers,
            [&](size_t i, genopack::ExtractedGenome& g) {
                int current = ++live;
                int old = peak.load();
                while (old < current && !peak.compare_exchange_weak(old, current)) {}
                {
                    std::lock_guard lock(ids_mutex);
                    ids.insert(std::this_thread::get_id());
                }
#ifdef _OPENMP
                int team = 0;
                #pragma omp parallel num_threads(8)
                #pragma omp single
                team = omp_get_num_threads();
                if (team != 1) throw std::runtime_error("nested compute team");
#endif
                results[i] = g.fasta;
                --live;
            });
        for (size_t i = 0; i < accs.size(); ++i)
            // ShardReader::fetch_genome returns FASTA; the fixture stores header-less "<id>".
            REQUIRE(results[i] == (accs[i] == "missing" ? "" : ">\n" + accs[i] + "\n"));
        REQUIRE(live == 0);
        REQUIRE(peak <= threads);
        REQUIRE(ids.size() <= size_t(threads));
    }
}
TEST_CASE("Shard task errors reach the caller after all tasks finish", "[parallel-pack]") {
    FakePack pack;
    std::vector<std::string> accs{"0", "1", "2", "3", "4", "5", "6", "7"};
    REQUIRE_THROWS_WITH(derep::visit_shard_genomes(pack, accs, 2, 1,
        [](size_t, genopack::ExtractedGenome&) { throw std::runtime_error("sketch error"); }), "sketch error");
    pack.fail_reader = true;
    REQUIRE_THROWS_WITH(derep::visit_shard_genomes(pack, accs, 2, 16,
        [](size_t, genopack::ExtractedGenome&) {}), "ANI shard pread failed or unexpected EOF");
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
    for (int readers : {1, 2, 16}) {
        std::vector<std::string> actual(accs.size());
        derep::visit_shard_genomes(*pack, accs, 8, readers,
            [&](size_t i, genopack::ExtractedGenome& genome) { actual.at(i) = genome.fasta; });
        REQUIRE(actual == expected);
    }
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
