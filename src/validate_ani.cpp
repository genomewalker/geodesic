#include "validate_ani.hpp"
#include "core/multi_pack_reader.hpp"
#include "core/pack_reader.hpp"
#include "core/preloaded_pack_reader.hpp"
#include "core/skani.hpp"
#include <genopack/archive.hpp>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace derep {
namespace {

double oph_jaccard(const uint16_t* a, const uint16_t* b, size_t m) {
    if (m == 0) return 0.0;
    size_t matches = 0, union_cnt = 0;
    for (size_t t = 0; t < m; ++t) {
        if (a[t] == 0xFFFFu && b[t] == 0xFFFFu) continue;
        ++union_cnt;
        if (a[t] == b[t]) ++matches;
    }
    if (union_cnt == 0) return 0.0;
    const double j_raw = double(matches) / double(union_cnt);
    constexpr double inv_2b = 1.0 / 65536.0;
    return std::max(0.0, (j_raw - inv_2b) / (1.0 - inv_2b));
}

double jaccard_to_ani(double J, int k) {
    if (J <= 0.0) return 70.0;
    if (J >= 1.0) return 100.0;
    return std::max(70.0, std::min(100.0, std::pow(2.0 * J / (1.0 + J), 1.0 / k) * 100.0));
}

} // anonymous namespace

int run_validate_ani(const Config& cfg) {
    namespace fs = std::filesystem;

    if (!cfg.pack_dir.has_value())
        throw std::runtime_error("--pack is required");

    // Open pack
    std::unique_ptr<IPackReader> pack;
    const auto& pack_path = *cfg.pack_dir;
    if (pack_path.extension() == ".gpk") {
        auto ar = std::make_unique<genopack::ArchiveReader>();
        ar->open(pack_path);
        pack = std::make_unique<SinglePackReader>(std::move(ar));
    } else {
        pack = MultiPackReader::open_dir(pack_path);
    }

    const auto avail_ks = pack->available_kmer_sizes();
    if (avail_ks.empty())
        throw std::runtime_error("pack has no SKCH section — rebuild with --sketch-kmers");

    const uint32_t sketch_sz = [&] {
        uint32_t stored = pack->sketch_sketch_size();
        uint32_t req    = static_cast<uint32_t>(cfg.sketch_size);
        return (stored > 0 && req > stored) ? stored : req;
    }();

    spdlog::info("validate-ani: pack k=[{}] sketch_size={}",
        [&]{ std::string s; for (auto k : avail_ks) { if (!s.empty()) s+=','; s+=std::to_string(k);} return s;}(),
        sketch_sz);

    // Read accessions
    std::vector<std::string> accs;
    {
        std::ifstream f(cfg.genomes_file);
        if (!f) throw std::runtime_error("cannot open: " + cfg.genomes_file.string());
        std::string line;
        while (std::getline(f, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
            if (!line.empty() && line[0] != '#') accs.push_back(line);
        }
    }
    if (accs.size() < 2)
        throw std::runtime_error("need at least 2 accessions, got " + std::to_string(accs.size()));
    spdlog::info("validate-ani: {} accessions", accs.size());

    // Sample pairs
    std::mt19937_64 rng(cfg.seed);
    std::vector<std::pair<size_t, size_t>> pairs;
    {
        const size_t n = accs.size();
        std::uniform_int_distribution<size_t> rdi(0, n - 1);
        int attempts = 0, limit = cfg.validate_pairs * 20;
        while ((int)pairs.size() < cfg.validate_pairs && attempts++ < limit) {
            size_t i = rdi(rng), j = rdi(rng);
            if (i != j) pairs.emplace_back(std::min(i, j), std::max(i, j));
        }
        std::sort(pairs.begin(), pairs.end());
        pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
    }
    spdlog::info("validate-ani: {} unique pairs", pairs.size());

    // Collect unique accessions needed
    std::unordered_set<std::string> needed_set;
    for (auto [i, j] : pairs) { needed_set.insert(accs[i]); needed_set.insert(accs[j]); }
    std::vector<std::string> needed(needed_set.begin(), needed_set.end());

    // Load sketches for every needed accession at all k-values in a single archive pass.
    // visit_sketch_batches_multi_k holds each archive's pages warm across all k-values
    // before evicting, avoiding N_k re-reads of the same NFS frames.
    using SigMap = std::unordered_map<std::string, std::unordered_map<uint32_t, std::vector<uint16_t>>>;
    SigMap sigs;
    pack->visit_sketch_batches_multi_k(needed, avail_ks, sketch_sz,
        [&](size_t idx, uint32_t k, const genopack::SketchResult& sk) {
            auto& v = sigs[needed[idx]][k];
            v.assign(sk.sig, sk.sig + sk.sketch_size);
        });
    spdlog::info("validate-ani: loaded sketches for {}/{} accessions", sigs.size(), needed.size());

    // Load raw FASTA sequences from pack and build FracMinHash sketches in-process
    std::unordered_map<std::string, std::string> fastas;
    fastas.reserve(needed.size());
    pack->visit_shard_batches(needed, [&](genopack::ArchiveReader::ShardBatch& batch) {
        for (auto& [idx, genome] : batch)
            fastas[genome.accession] = std::move(genome.fasta);
    });
    spdlog::info("validate-ani: loaded {}/{} FASTAs", fastas.size(), needed.size());

    std::vector<SkaniSketch> ani_sketches(needed.size());
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic) num_threads(cfg.threads)
#endif
    for (int i = 0; i < (int)needed.size(); ++i) {
        auto it = fastas.find(needed[i]);
        if (it == fastas.end()) continue;
        ani_sketches[i] = build_sketch(needed[i], it->second, cfg.ani_k, cfg.ani_c);
    }
    fastas.clear();

    std::unordered_map<std::string, const SkaniSketch*> sketch_idx;
    sketch_idx.reserve(ani_sketches.size());
    for (auto& sk : ani_sketches)
        if (!sk.accession.empty()) sketch_idx[sk.accession] = &sk;

    std::unordered_map<std::string, double> ani_map;
    ani_map.reserve(pairs.size());
    for (auto [i, j] : pairs) {
        const auto& a = accs[i];
        const auto& b = accs[j];
        auto ia = sketch_idx.find(a), ib = sketch_idx.find(b);
        if (ia == sketch_idx.end() || ib == sketch_idx.end()) continue;
        auto res = compute_ani(*ia->second, *ib->second, cfg.ani_k);
        ani_map[a + "\t" + b] = res.ani;
    }
    spdlog::info("validate-ani: computed {} FracMinHash ANI pairs", ani_map.size());

    // Write output TSV
    const fs::path out_path = cfg.validate_output.empty()
        ? fs::path("ani_validation.tsv") : cfg.validate_output;
    std::ofstream out(out_path);
    if (!out) throw std::runtime_error("cannot open output: " + out_path.string());

    out << "query\tref\tani_geo";
    for (uint32_t k : avail_ks)
        out << "\tj_k" << k << "\tani_est_k" << k << "\terr_k" << k;
    out << "\tfill_query\tfill_ref\n";

    size_t n_written = 0, n_missing = 0;
    for (auto [i, j] : pairs) {
        const auto& a = accs[i];
        const auto& b = accs[j];

        auto it = ani_map.find(a + "\t" + b);
        if (it == ani_map.end()) { ++n_missing; continue; }
        const double ani_geo = it->second;

        out << a << "\t" << b << "\t" << std::fixed << std::setprecision(4) << ani_geo;

        const auto& sigs_a = sigs[a];
        const auto& sigs_b = sigs[b];

        double fill_a = -1, fill_b = -1;
        for (uint32_t k : avail_ks) {
            auto ka = sigs_a.find(k), kb = sigs_b.find(k);
            if (ka == sigs_a.end() || kb == sigs_b.end()) { out << "\t\t\t"; continue; }
            const auto& sa = ka->second;
            const auto& sb = kb->second;
            const size_t m = std::min(sa.size(), sb.size());
            const double J   = oph_jaccard(sa.data(), sb.data(), m);
            const double est = jaccard_to_ani(J, static_cast<int>(k));
            out << "\t" << J << "\t" << est << "\t" << (est - ani_geo);

            // fill fraction from the first k only
            if (fill_a < 0) {
                size_t ra = 0, rb = 0;
                for (size_t t = 0; t < m; ++t) {
                    if (sa[t] != 0xFFFFu) ++ra;
                    if (sb[t] != 0xFFFFu) ++rb;
                }
                fill_a = double(ra) / double(m);
                fill_b = double(rb) / double(m);
            }
        }
        out << "\t" << fill_a << "\t" << fill_b << "\n";
        ++n_written;
    }

    spdlog::info("validate-ani: wrote {} rows → {} ({} pairs missing)",
                 n_written, out_path.string(), n_missing);
    return 0;
}

} // namespace derep
