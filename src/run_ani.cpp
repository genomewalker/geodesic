#include "run_ani.hpp"
#include "core/multi_pack_reader.hpp"
#include "core/pack_reader.hpp"
#include "core/parallel_shard_visit.hpp"
#include <chrono>
#include "core/skani.hpp"
#include <genopack/archive.hpp>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace derep {

static std::vector<std::string> read_accessions(const std::filesystem::path& p) {
    std::vector<std::string> v;
    std::ifstream f(p);
    if (!f) throw std::runtime_error("cannot open: " + p.string());
    std::string line;
    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        if (!line.empty() && line[0] != '#') v.push_back(line);
    }
    return v;
}

int run_ani(const Config& cfg) {
    namespace fs = std::filesystem;

    if (!cfg.pack_dir.has_value())
        throw std::runtime_error("--pack is required");
    if (cfg.ani_query_file.empty())
        throw std::runtime_error("--ql is required");

    std::unique_ptr<IPackReader> pack;
    const auto& pack_path = *cfg.pack_dir;
    if (pack_path.extension() == ".gpk") {
        auto ar = std::make_unique<genopack::ArchiveReader>();
        ar->open(pack_path);
        pack = std::make_unique<SinglePackReader>(std::move(ar));
    } else {
        pack = MultiPackReader::open_dir(pack_path);
    }

    const auto ql = read_accessions(cfg.ani_query_file);
    const bool self_pairs = cfg.ani_ref_file.empty();
    const auto rl = self_pairs ? ql : read_accessions(cfg.ani_ref_file);

    spdlog::info("ani: {} query, {} ref accessions{}",
                 ql.size(), rl.size(), self_pairs ? " (self all-pairs)" : "");

    // Union of unique accessions to load
    std::unordered_set<std::string> needed_set(ql.begin(), ql.end());
    needed_set.insert(rl.begin(), rl.end());
    std::vector<std::string> needed(needed_set.begin(), needed_set.end());

    using Clock = std::chrono::steady_clock;
    auto phase_start = Clock::now();
    auto phase = [&](const char* name) {
        const auto now = Clock::now();
        spdlog::info("ani: {} wall: {:.3f} seconds", name,
                     std::chrono::duration<double>(now - phase_start).count());
        phase_start = now;
    };
    spdlog::info("ani: {} total threads, up to {} shard readers", cfg.threads,
                 ani_reader_count(cfg.threads, cfg.io_threads));
    std::vector<SkaniSketch> sketches(needed.size());
    visit_shard_genomes(*pack, needed, cfg.threads, cfg.io_threads,
        [&](size_t idx, genopack::ExtractedGenome& genome) {
            auto fasta = std::move(genome.fasta);
            sketches[idx] = build_sketch(needed[idx], fasta, cfg.ani_k, cfg.ani_c);
        });
    phase("read+sketch");

    std::unordered_map<std::string, const SkaniSketch*> sketch_idx;
    sketch_idx.reserve(sketches.size());
    for (auto& sk : sketches)
        if (!sk.accession.empty()) sketch_idx[sk.accession] = &sk;

    std::vector<const SkaniSketch*> qsks, rsks;
    for (const auto& a : ql) { auto it = sketch_idx.find(a); if (it != sketch_idx.end()) qsks.push_back(it->second); }
    for (const auto& a : rl) { auto it = sketch_idx.find(a); if (it != sketch_idx.end()) rsks.push_back(it->second); }

    spdlog::info("ani: sketched {}/{} FASTAs", sketch_idx.size(), needed.size());
    if (self_pairs) {
        // Main oriented pairs by their address in sketches. Preserve that orientation
        // (including c_ab/c_ba), but traverse only qi < ri in this order.
        std::sort(qsks.begin(), qsks.end());
        rsks = qsks;
    }

    spdlog::info("ani: computing {} × {} pairs …", qsks.size(), rsks.size());

    const fs::path out_path = cfg.ani_output.empty() ? fs::path("ani_results.tsv") : cfg.ani_output;
    std::ofstream out(out_path);
    if (!out) throw std::runtime_error("cannot open output: " + out_path.string());
    out << "query\tref\tani\taf\tc_ab\tc_ba\n";

    const int nq = static_cast<int>(qsks.size());
    const int nr = static_cast<int>(rsks.size());

    struct Row { int qi, ri; double ani, af, c_ab, c_ba; };
    std::vector<Row> rows;

    // A flat inverted index avoids a hash table allocation for every distinct
    // marker. Sorting (hash, index) also sorts each marker's reference list.
    struct Marker { uint64_t hash; int ri; };
    std::vector<Marker> index;
    std::vector<size_t> nm_ref(nr);
    if (cfg.ani_screen > 0) {
        size_t count = 0;
        for (const auto* sk : rsks)
            for (auto h : sk->hashes) count += (h % 8 == 0);
        index.reserve(count);
        for (int ri = 0; ri < nr; ++ri)
            for (auto h : rsks[ri]->hashes)
                if (h % 8 == 0) { index.push_back({h, ri}); ++nm_ref[ri]; }
        std::sort(index.begin(), index.end(), [](const Marker& a, const Marker& b) {
            return a.hash < b.hash || (a.hash == b.hash && a.ri < b.ri);
        });
        if (cfg.ani_max_marker_df) {
            size_t dst = 0;
            for (size_t begin = 0; begin < index.size();) {
                size_t end = begin + 1;
                std::unordered_set<const SkaniSketch*> genomes{rsks[index[begin].ri]};
                while (end < index.size() && index[end].hash == index[begin].hash) {
                    // Duplicate list entries still represent just one genome.
                    genomes.insert(rsks[index[end].ri]);
                    ++end;
                }
                if (genomes.size() <= cfg.ani_max_marker_df)
                    for (size_t i = begin; i < end; ++i) index[dst++] = index[i];
                begin = end;
            }
            index.resize(dst);
        }
        spdlog::info("ani: marker screen {}%, {} indexed marker occurrences",
                     cfg.ani_screen, index.size());
    }
    phase("index");
    const double threshold = std::pow(cfg.ani_screen / 100.0, cfg.ani_k);
    const int nthreads = cfg.threads;
    std::vector<std::vector<Row>> trows(nthreads);
#ifdef _OPENMP
    #pragma omp parallel num_threads(nthreads)
#endif
    {
#ifdef _OPENMP
        auto& local = trows[omp_get_thread_num()];
#else
        auto& local = trows[0];
#endif
        std::vector<uint32_t> counts(cfg.ani_screen > 0 ? nr : 0);
        std::vector<int> touched;
#ifdef _OPENMP
        #pragma omp for schedule(dynamic, 16)
#endif
        for (int qi = 0; qi < nq; ++qi) {
            auto emit = [&](int ri) {
                if (self_pairs && qsks[qi] == rsks[ri]) return;
                auto res = compute_ani(*qsks[qi], *rsks[ri], cfg.ani_k);
                if (res.af >= cfg.ani_min_af)
                    local.push_back({qi, ri, res.ani, res.af, res.c_ab, res.c_ba});
            };
            if (cfg.ani_screen == 0) {
                for (int ri = self_pairs ? qi + 1 : 0; ri < nr; ++ri) emit(ri);
                continue;
            }
            size_t nm_q = 0;
            for (auto h : qsks[qi]->hashes) {
                if (h % 8 != 0) continue;
                ++nm_q;
                auto begin = std::lower_bound(index.begin(), index.end(), h,
                    [](const Marker& m, uint64_t hash) { return m.hash < hash; });
                auto end = std::upper_bound(begin, index.end(), h,
                    [](uint64_t hash, const Marker& m) { return hash < m.hash; });
                if (self_pairs)
                    begin = std::upper_bound(begin, end, qi,
                        [](int q, const Marker& m) { return q < m.ri; });
                for (auto it = begin; it != end; ++it) {
                    if (counts[it->ri]++ == 0) touched.push_back(it->ri);
                }
            }
            for (int ri : touched) {
                const size_t denom = std::min(nm_q, nm_ref[ri]);
                if (denom && double(counts[ri]) / double(denom) >= threshold) emit(ri);
                counts[ri] = 0;
            }
            touched.clear();
        }
    }

    // Merge thread-local results.
    size_t total = 0;
    for (auto& v : trows) total += v.size();
    rows.reserve(total);
    for (auto& v : trows)
        rows.insert(rows.end(), v.begin(), v.end());

    phase("pairs");
    out << std::fixed << std::setprecision(4);
    for (auto& r : rows) {
        out << qsks[r.qi]->accession << '\t' << rsks[r.ri]->accession << '\t'
            << r.ani << '\t' << r.af << '\t' << r.c_ab << '\t' << r.c_ba << '\n';
    }

    out.close();
    if (!out) throw std::runtime_error("cannot write output: " + out_path.string());
    phase("write");
    spdlog::info("ani: wrote {} pairs → {}", rows.size(), out_path.string());
    return 0;
}

} // namespace derep
