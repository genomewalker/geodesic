#include "run_ani.hpp"
#include "core/multi_pack_reader.hpp"
#include "core/pack_reader.hpp"
#include "core/skani.hpp"
#include <genopack/archive.hpp>
#include <spdlog/spdlog.h>
#include <algorithm>
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

    // Load raw FASTA sequences from pack — fully in memory, no disk I/O
    std::unordered_map<std::string, std::string> fastas;
    fastas.reserve(needed.size());
    pack->visit_shard_batches(needed, [&](genopack::ArchiveReader::ShardBatch& batch) {
        for (auto& [idx, genome] : batch)
            fastas[genome.accession] = std::move(genome.fasta);
    });
    spdlog::info("ani: loaded {}/{} FASTAs", fastas.size(), needed.size());

    // Build sketches for all needed accessions in parallel
    std::vector<SkaniSketch> sketches(needed.size());
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic) num_threads(cfg.threads)
#endif
    for (int i = 0; i < (int)needed.size(); ++i) {
        auto it = fastas.find(needed[i]);
        if (it == fastas.end()) continue;
        sketches[i] = build_sketch(needed[i], it->second, cfg.ani_k, cfg.ani_c);
    }
    fastas.clear();

    std::unordered_map<std::string, const SkaniSketch*> sketch_idx;
    sketch_idx.reserve(sketches.size());
    for (auto& sk : sketches)
        if (!sk.accession.empty()) sketch_idx[sk.accession] = &sk;

    std::vector<const SkaniSketch*> qsks, rsks;
    for (const auto& a : ql) { auto it = sketch_idx.find(a); if (it != sketch_idx.end()) qsks.push_back(it->second); }
    for (const auto& a : rl) { auto it = sketch_idx.find(a); if (it != sketch_idx.end()) rsks.push_back(it->second); }

    spdlog::info("ani: computing {} × {} pairs …", qsks.size(), rsks.size());

    const fs::path out_path = cfg.ani_output.empty() ? fs::path("ani_results.tsv") : cfg.ani_output;
    std::ofstream out(out_path);
    if (!out) throw std::runtime_error("cannot open output: " + out_path.string());
    out << "query\tref\tani\taf\tc_ab\tc_ba\n";

    const int nq = static_cast<int>(qsks.size());
    const int nr = static_cast<int>(rsks.size());

    struct Row { int qi, ri; double ani, af, c_ab, c_ba; };
    std::vector<Row> rows;

    // Per-thread row storage avoids a shared critical section on every push_back.
    const int nthreads = cfg.threads;
    std::vector<std::vector<Row>> trows(nthreads);
    for (auto& v : trows) v.reserve((size_t)nq * nr / nthreads + 64);

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 16) num_threads(nthreads)
#endif
    for (int qi = 0; qi < nq; ++qi) {
#ifdef _OPENMP
        auto& local = trows[omp_get_thread_num()];
#else
        auto& local = trows[0];
#endif
        for (int ri = 0; ri < nr; ++ri) {
            if (self_pairs && qsks[qi] == rsks[ri]) continue;
            if (self_pairs && qsks[qi] > rsks[ri]) continue;
            auto res = compute_ani(*qsks[qi], *rsks[ri], cfg.ani_k);
            if (res.af < cfg.ani_min_af) continue;
            local.push_back({qi, ri, res.ani, res.af, res.c_ab, res.c_ba});
        }
    }

    // Merge thread-local results.
    size_t total = 0;
    for (auto& v : trows) total += v.size();
    rows.reserve(total);
    for (auto& v : trows)
        rows.insert(rows.end(), v.begin(), v.end());

    out << std::fixed << std::setprecision(4);
    for (auto& r : rows) {
        out << qsks[r.qi]->accession << '\t' << rsks[r.ri]->accession << '\t'
            << r.ani << '\t' << r.af << '\t' << r.c_ab << '\t' << r.c_ba << '\n';
    }

    spdlog::info("ani: wrote {} pairs → {}", rows.size(), out_path.string());
    return 0;
}

} // namespace derep
