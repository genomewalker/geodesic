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
// Appends a frozen shard to `file`; returns its offset and size.
std::pair<uint64_t, uint64_t> append_shard(std::FILE* file, genopack::ShardWriter& writer) {
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
    std::fflush(file);
    return {uint64_t(offset), frozen.bytes.size()};
}

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
            auto [offset, size] = append_shard(file, writer);
            for (uint16_t archive : {0, 1})
                locations.push_back({archive, shard, fileno(file), offset, size});
        }
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
    for (int threads : {1, 4, 8, 16, 32, 64}) REQUIRE(derep::ani_reader_count(threads, 0) == 16);
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

// One shard per codec (0 zstd, 1 dict, 2 ref-dict, 3 delta, 4 mem-delta) of 24
// related genomes, so dictionary, reference blobs and checkpoints all matter.
class CodecPack : public derep::SinglePackReader {
public:
    CodecPack() : SinglePackReader(nullptr), file(std::tmpfile()) {
        if (!file) throw std::runtime_error("tmpfile failed");
        uint64_t state = 42;
        auto next = [&] { state = state * 6364136223846793005ULL + 1442695040888963407ULL; return state >> 33; };
        std::string base(30000, 'A');
        for (auto& c : base) c = "ACGT"[next() % 4];
        std::vector<std::string> genomes;
        for (int id = 0; id < 24; ++id) {
            auto seq = base;
            for (int m = 0; m < 300; ++m) seq[next() % seq.size()] = "ACGT"[next() % 4];
            genomes.push_back(">g" + std::to_string(id) + " contig\n" + seq.substr(0, 17000) +
                              "\n>g" + std::to_string(id) + " second\n" + seq.substr(17000) + "\n");
        }
        for (uint32_t codec = 0; codec <= 4; ++codec) {
            genopack::ShardWriterConfig cfg;
            cfg.train_dict = codec == 1;
            cfg.dict_samples = 24;
            cfg.dict_size = 4096;
            cfg.use_reference_dict = codec == 2;
            cfg.use_delta = codec == 3;
            cfg.use_mem_delta = codec == 4;
            cfg.checkpoint_bases = 4096;
            cfg.compress_threads = 1;
            genopack::ShardWriter writer(codec, 0, cfg);
            for (int id = 0; id < 24; ++id) writer.add_genome(id, 0, genomes[id].data(), genomes[id].size());
            auto [offset, size] = append_shard(file, writer);
            locations.push_back({0, codec, fileno(file), offset, size});
        }
    }
    ~CodecPack() { std::fclose(file); }
    std::FILE* file;
    std::vector<ShardLocation> locations;
    std::optional<genopack::GenomeMeta> genome_meta_by_accession(std::string_view a) const override {
        genopack::GenomeMeta m;
        m.shard_id = a[0] - '0';
        m.genome_id = std::stoi(std::string(a.substr(2)));
        return m;
    }
    uint16_t archive_idx_for_accession(std::string_view) const override { return 0; }
    std::vector<ShardLocation> shard_locations() const override { return locations; }
    std::vector<uint8_t> bytes(const ShardLocation& loc) const {
        std::vector<uint8_t> out(loc.size);
        if (::pread(loc.fd, out.data(), out.size(), loc.offset) != ssize_t(out.size())) throw std::runtime_error("fixture read failed");
        return out;
    }
};

uint64_t bytes_read_by_process() {
    std::ifstream io("/proc/self/io");
    std::string key;
    uint64_t value;
    while (io >> key >> value) if (key == "rchar:") return value;
    throw std::runtime_error("no rchar in /proc/self/io");
}

TEST_CASE("Partial shard reads decode like whole-shard reads for every codec", "[parallel-pack]") {
    CodecPack pack;
    uint64_t total = 0;
    for (const auto& loc : pack.locations) total += loc.size;
    std::vector<std::string> accs;
    for (uint32_t codec = 0; codec <= 4; ++codec)
        for (int id : {13, 2, 23}) accs.push_back(std::to_string(codec) + ":" + std::to_string(id));
    std::vector<std::string> actual(accs.size());
    const uint64_t before = bytes_read_by_process();
    derep::visit_shard_genomes(pack, accs, 4, 2,
        [&](size_t i, genopack::ExtractedGenome& g) { actual.at(i) = g.fasta; }, 0);
    const uint64_t read = bytes_read_by_process() - before;
    INFO("visitor read " << read << " of " << total << " shard bytes");
    REQUIRE(read < total * 3 / 4);
    for (const auto& loc : pack.locations) {
        auto whole = pack.bytes(loc);
        const auto& header = *reinterpret_cast<const genopack::ShardHeader*>(whole.data());
        REQUIRE(header.codec == loc.shard); // the fixture really covers this codec
        genopack::ShardReader reader;
        reader.open(whole.data(), 0, whole.size());
        uint64_t planned = 0;
        for (auto [begin, end] : derep::shard_read_ranges(whole.data(), whole.size(), {13, 2, 23}, 0))
            planned += end - begin;
        INFO("codec " << loc.shard << ": " << planned << " of " << whole.size() << " bytes");
        REQUIRE(planned < whole.size());
        for (size_t i = 0; i < accs.size(); ++i) {
            if (uint32_t(accs[i][0] - '0') != loc.shard) continue;
            const auto expected = reader.fetch_genome(std::stoi(accs[i].substr(2)));
            REQUIRE(expected.size() > 30000);
            REQUIRE(actual[i] == expected);
        }
    }
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
