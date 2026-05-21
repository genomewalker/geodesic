#include "validate_ani.hpp"
#include "core/multi_pack_reader.hpp"
#include "core/pack_reader.hpp"
#include "core/preloaded_pack_reader.hpp"
#include "core/subprocess.hpp"
#include <genopack/archive.hpp>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>
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

// skani dist stdout: Query_file\tRef_file\tANI\t...
// Returns map of "acc_a\tacc_b" -> ANI (both orderings stored)
std::unordered_map<std::string, double> parse_skani_output(const std::string& text) {
    std::unordered_map<std::string, double> result;
    std::istringstream ss(text);
    std::string line;
    bool header = true;
    while (std::getline(ss, line)) {
        if (header) { header = false; continue; }
        if (line.empty()) continue;
        std::vector<std::string> cols;
        std::istringstream ls(line);
        std::string tok;
        while (std::getline(ls, tok, '\t')) cols.push_back(tok);
        if (cols.size() < 3) continue;
        const std::string& qa = std::filesystem::path(cols[0]).stem().string();
        const std::string& rb = std::filesystem::path(cols[1]).stem().string();
        double ani = 0.0;
        try { ani = std::stod(cols[2]); } catch (...) { continue; }
        result[qa + "\t" + rb] = ani;
        result[rb + "\t" + qa] = ani;
    }
    return result;
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

    // Extract FASTAs via genopack
    fs::path fasta_dir = cfg.tmp_dir / "validate_ani_fasta";
    fs::create_directories(fasta_dir);

    {
        std::ofstream f(fasta_dir / "accs.txt");
        for (const auto& a : needed) f << a << "\n";
    }

    const std::string gpk_bin = cfg.genopack_bin.empty() ? "genopack" : cfg.genopack_bin;
    spdlog::info("validate-ani: extracting {} FASTAs …", needed.size());

    // For multipart directories, genopack extract only reads one part — run it per part.
    std::vector<fs::path> gpk_targets;
    if (fs::is_directory(pack_path)) {
        for (const auto& e : fs::directory_iterator(pack_path))
            if (e.path().extension() == ".gpk") gpk_targets.push_back(e.path());
        std::sort(gpk_targets.begin(), gpk_targets.end());
    } else {
        gpk_targets.push_back(pack_path);
    }

    for (const auto& gpk : gpk_targets) {
        auto extract = run_subprocess(
            {gpk_bin, "extract", gpk.string(),
             "--accessions-file", (fasta_dir / "accs.txt").string(),
             "--output-dir", fasta_dir.string()},
            {.capture_stderr = true});
        if (!extract.ok())
            throw std::runtime_error("genopack extract failed on " + gpk.string()
                + " (exit " + std::to_string(extract.exit_code)
                + "): " + extract.stderr_output);
    }

    // Write separate query/ref lists for skani — one side of each pair per list.
    // This avoids O(n_unique²) all-pairs; skani only computes the cross-product
    // of unique left-side vs unique right-side accessions (~2×n_pairs instead of n_unique²).
    {
        std::unordered_set<std::string> ql_set, rl_set;
        for (auto [i, j] : pairs) { ql_set.insert(accs[i]); rl_set.insert(accs[j]); }
        {
            std::ofstream f(fasta_dir / "ql.txt");
            for (const auto& a : ql_set) f << (fasta_dir / (a + ".fa")).string() << "\n";
        }
        {
            std::ofstream f(fasta_dir / "rl.txt");
            for (const auto& a : rl_set) f << (fasta_dir / (a + ".fa")).string() << "\n";
        }
    }

    spdlog::info("validate-ani: running skani dist …");
    SubprocessOptions skani_opts;
    skani_opts.capture_stdout = true;
    skani_opts.capture_stderr = false;
    auto skani = run_subprocess(
        {"skani", "dist",
         "--ql", (fasta_dir / "ql.txt").string(),
         "--rl", (fasta_dir / "rl.txt").string(),
         "-t",  std::to_string(cfg.threads),
         "--min-af", "0.0"},
        skani_opts);
    if (!skani.ok())
        spdlog::warn("validate-ani: skani exited {}", skani.exit_code);

    auto skani_map = parse_skani_output(skani.stdout_output);
    spdlog::info("validate-ani: {} skani ANI entries", skani_map.size());

    // Write output TSV
    const fs::path out_path = cfg.validate_output.empty()
        ? fs::path("ani_validation.tsv") : cfg.validate_output;
    std::ofstream out(out_path);
    if (!out) throw std::runtime_error("cannot open output: " + out_path.string());

    out << "query\tref\tani_skani";
    for (uint32_t k : avail_ks)
        out << "\tj_k" << k << "\tani_est_k" << k << "\terr_k" << k;
    out << "\tfill_query\tfill_ref\n";

    size_t n_written = 0, n_no_skani = 0;
    for (auto [i, j] : pairs) {
        const auto& a = accs[i];
        const auto& b = accs[j];

        auto it = skani_map.find(a + "\t" + b);
        if (it == skani_map.end()) { ++n_no_skani; continue; }
        const double ani_skani = it->second;

        out << a << "\t" << b << "\t" << std::fixed << std::setprecision(4) << ani_skani;

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
            out << "\t" << J << "\t" << est << "\t" << (est - ani_skani);

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

    spdlog::info("validate-ani: wrote {} rows → {} ({} pairs not in skani output)",
                 n_written, out_path.string(), n_no_skani);
    return 0;
}

} // namespace derep
