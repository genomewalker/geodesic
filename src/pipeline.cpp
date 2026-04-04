#include "pipeline.hpp"
#include <iostream>
#include <unordered_set>
#include "parallel/taxon_processor.hpp"
#include "state/run_state.hpp"
#include "core/logging.hpp"
#include "core/types.hpp"
#include "core/sketch/minhash.hpp"
#include "core/pack_reader.hpp"
#include "core/multi_pack_reader.hpp"
#include "taxonomy/normalize.hpp"
#include "db/geodf/geodf_writer.hpp"
#include "db/geodf/geodf_reader.hpp"
#include "db/taxdb/ncbi_taxdb.hpp"
#include <genopack/archive.hpp>
#include "io/gz_reader.hpp"
#include "io/lock_writer.hpp"
#include "io/report_writer.hpp"
#include "io/results_writer.hpp"
#include "io/tsv_reader.hpp"

#include <BS_thread_pool.hpp>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <unistd.h>
#include <fstream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace derep {

namespace {

namespace fs = std::filesystem;

std::unordered_set<std::string> read_selected_taxa(const fs::path& path) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("Cannot open selected taxa file: " + path.string());

    std::unordered_set<std::string> taxa;
    std::string line;
    while (std::getline(in, line)) {
        auto start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        auto end = line.find_last_not_of(" \t\r\n");
        auto trimmed = line.substr(start, end - start + 1);
        if (trimmed.empty() || trimmed[0] == '#') continue;
        taxa.insert(std::move(trimmed));
    }
    spdlog::info("Read {} selected taxa from {}", taxa.size(), path.string());
    return taxa;
}

// Derive a stable species-name stem from an accession, following GTDB convention:
//   - Strip GTDB prefixes RS_/GB_ first
//   - For NCBI-style accessions (GCF_/GCA_): strip the version suffix (.N)
//   - For MAG/other IDs (TARA_*, spire_*, GOMC.*): use as-is
// e.g. "RS_GCF_003697165.2" → "GCF_003697165", "spire_mag_00498234" → "spire_mag_00498234"
using derep::taxonomy::normalize_taxonomy;
using derep::taxonomy::has_accession_species;

std::vector<Taxon> group_by_taxonomy(
    std::vector<Genome>& genomes,
    const std::unordered_map<std::string, std::string>& fixed_taxa) {
    std::unordered_map<std::string, std::vector<Genome>> groups;
    std::vector<Taxon> taxa;

    size_t n_accession_species = 0;
    for (auto& g : genomes) {
        if (has_accession_species(g.taxonomy, g.accession))
            ++n_accession_species;
        groups[g.taxonomy].push_back(std::move(g));
    }

    if (n_accession_species > 0)
        spdlog::info("{} genome(s) with unresolved species assigned accession-derived name (natural singletons)",
                     n_accession_species);

    taxa.reserve(taxa.size() + groups.size());
    for (auto& [taxonomy, genome_vec] : groups) {
        Taxon t;
        t.taxonomy = taxonomy;
        t.genomes = std::move(genome_vec);
        if (auto it = fixed_taxa.find(taxonomy); it != fixed_taxa.end())
            t.forced_representative = it->second;
        taxa.push_back(std::move(t));
    }

    std::sort(taxa.begin(), taxa.end(),
              [](const Taxon& a, const Taxon& b) { return a.size() > b.size(); });

    return taxa;
}

std::vector<Genome> rows_to_genomes(
    const std::vector<GenomeRow>& rows,
    const std::unordered_map<std::string, CheckM2Quality>& checkm2,
    const NcbiTaxdb* ncbi = nullptr) {
    std::vector<Genome> genomes;
    genomes.reserve(rows.size());
    for (const auto& row : rows) {
        Genome g;
        g.accession = row.accession;
        g.taxonomy = normalize_taxonomy(row.taxonomy, row.accession, ncbi);
        g.file_path = row.file_path;

        auto acc = canonical_accession(row.accession);
        if (auto it = checkm2.find(acc); it != checkm2.end()) {
            g.completeness = it->second.completeness;
            g.contamination = it->second.contamination;
        }
        genomes.push_back(std::move(g));
    }
    return genomes;
}

void setup_logging(const fs::path& log_file, int verbosity) {
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_file.string(), true);

    // File sink always captures INFO+: useful even in quiet/nohup runs
    file_sink->set_level(spdlog::level::info);

    // Console sink respects verbosity
    if (verbosity == 0)
        console_sink->set_level(spdlog::level::warn);
    else if (verbosity == 1)
        console_sink->set_level(spdlog::level::info);
    else if (verbosity == 2)
        console_sink->set_level(spdlog::level::debug);
    else
        console_sink->set_level(spdlog::level::trace);

    auto logger = std::make_shared<spdlog::logger>(
        "geodesic", spdlog::sinks_init_list{console_sink, file_sink});
    logger->set_level(spdlog::level::info);  // logger passes info+; sinks filter further
    logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    spdlog::set_default_logger(logger);

    set_verbosity(verbosity);  // keep g_verbosity in sync for is_quiet()/is_verbose() guards
    spdlog::flush_on(spdlog::level::info);
}

} // anonymous namespace

void process_taxa_parallel(
    const std::vector<Taxon>& taxa,
    const Config& cfg,
    RunState& run_state,
    IPackReader* gpk_reader,
    const std::unordered_map<std::string, GuncQuality>* gunc_scores) {

    RunState* run_state_ptr = &run_state;
    IPackReader* gpk_reader_ptr = gpk_reader;
    const std::unordered_map<std::string, GuncQuality>* gunc_scores_ptr = gunc_scores;

    const int total_budget = cfg.workers * cfg.threads;
    BS::thread_pool pool(static_cast<BS::concurrency_t>(total_budget));

    std::vector<int> taxon_threads(taxa.size(), 1);
    {
        const size_t n = taxa.size();
        if (n >= 50) {
            std::vector<size_t> sizes(n);
            for (size_t i = 0; i < n; ++i) sizes[i] = taxa[i].size();
            std::vector<size_t> sorted_sizes = sizes;
            std::sort(sorted_sizes.begin(), sorted_sizes.end());
            const size_t p25 = sorted_sizes[n * 25 / 100];
            const size_t p75 = sorted_sizes[n * 75 / 100];
            const size_t p95 = sorted_sizes[n * 95 / 100];
            const int t_giant  = std::max(12, total_budget / 2);
            const int t_large  = std::max(1, total_budget / 6);
            const int t_medium = std::max(1, total_budget / 12);
            size_t cnt_giant = 0, cnt_large = 0, cnt_medium = 0, cnt_small = 0;
            for (size_t i = 0; i < n; ++i) {
                if      (sizes[i] >= p95) { taxon_threads[i] = t_giant;  ++cnt_giant; }
                else if (sizes[i] >= p75) { taxon_threads[i] = t_large;  ++cnt_large; }
                else if (sizes[i] >= p25) { taxon_threads[i] = t_medium; ++cnt_medium; }
                else                      { taxon_threads[i] = 1;        ++cnt_small; }
            }
            spdlog::info(
                "Execution plan: budget={} | giant={} ({}t, ~{} concurrent) | large={} ({}t) | medium={} ({}t) | small={} (1t)",
                total_budget, cnt_giant, t_giant, total_budget / t_giant,
                cnt_large, t_large, cnt_medium, t_medium, cnt_small);
        } else {
            size_t large_genomes = 0;
            for (size_t i = 0; i < n; ++i)
                if (taxa[i].size() > 10)
                    large_genomes += taxa[i].size();
            if (large_genomes > 0) {
                for (size_t i = 0; i < n; ++i) {
                    if (taxa[i].size() <= 10) continue;
                    taxon_threads[i] = std::max(1, static_cast<int>(std::round(
                        static_cast<double>(total_budget) *
                        static_cast<double>(taxa[i].size()) /
                        static_cast<double>(large_genomes))));
                }
            } else {
                for (size_t i = 0; i < n; ++i) taxon_threads[i] = total_budget;
            }
            spdlog::info("Execution plan (proportional, budget={}, {} taxa):",
                         total_budget, n);
            for (size_t i = 0; i < n; ++i)
                spdlog::info("  [{}] {} genomes → {} threads",
                             i, taxa[i].size(), taxon_threads[i]);
        }
    }

    int budget_avail = total_budget;
    std::mutex budget_mtx;
    std::condition_variable budget_cv;

    auto budget_acquire = [&](int desired) -> int {
        std::unique_lock lock(budget_mtx);
        budget_cv.wait(lock, [&] { return budget_avail > 0; });
        int taken = std::min(budget_avail, desired);
        budget_avail -= taken;
        return taken;
    };
    auto budget_release = [&](int n) {
        { std::lock_guard lock(budget_mtx); budget_avail += n; }
        budget_cv.notify_one();
    };

    std::queue<TaxonResult> done_queue;
    std::mutex done_mutex;
    std::condition_variable done_cv;

    static constexpr size_t TINY_SCHED_THRESHOLD = 100;
    static constexpr size_t TINY_BATCH_SIZE = 200;

    std::vector<size_t> large_indices;
    std::vector<std::vector<size_t>> tiny_batches;
    {
        std::vector<size_t> tiny_indices;
        for (size_t i = 0; i < taxa.size(); ++i) {
            if (taxa[i].size() > TINY_SCHED_THRESHOLD)
                large_indices.push_back(i);
            else
                tiny_indices.push_back(i);
        }
        for (size_t off = 0; off < tiny_indices.size(); off += TINY_BATCH_SIZE) {
            size_t end = std::min(off + TINY_BATCH_SIZE, tiny_indices.size());
            tiny_batches.emplace_back(tiny_indices.begin() + off,
                                      tiny_indices.begin() + end);
        }
        spdlog::info("Scheduler: {} large taxa (individual), {} tiny taxa in {} batches",
                     large_indices.size(), tiny_indices.size(), tiny_batches.size());
    }

    std::thread scheduler([&] {
        for (size_t i : large_indices) {
            int desired  = taxon_threads[i];
            int acquired = budget_acquire(desired);
            pool.detach_task(
                [&taxa, i, &cfg, gunc_scores_ptr,
                 gpk_reader_ptr, run_state_ptr,
                 &done_queue, &done_mutex, &done_cv,
                 &budget_release, acquired] {
                    auto result = process_taxon(taxa[i], cfg, acquired,
                                               gunc_scores_ptr,
                                               gpk_reader_ptr, run_state_ptr);
                    {
                        std::lock_guard lock(done_mutex);
                        done_queue.push(std::move(result));
                    }
                    done_cv.notify_one();
                    budget_release(acquired);
                });
        }
        for (const auto& batch_indices : tiny_batches) {
            budget_acquire(1);
            std::vector<const Taxon*> batch_taxa;
            batch_taxa.reserve(batch_indices.size());
            for (size_t i : batch_indices)
                batch_taxa.push_back(&taxa[i]);
            pool.detach_task(
                [batch_taxa, &cfg,
                 gpk_reader_ptr, gunc_scores_ptr, run_state_ptr, &done_queue, &done_mutex, &done_cv,
                 &budget_release] {
                    auto results = process_tiny_batch(batch_taxa, cfg,
                                                      gunc_scores_ptr, gpk_reader_ptr,
                                                      run_state_ptr);
                    {
                        std::lock_guard lock(done_mutex);
                        for (auto& r : results)
                            done_queue.push(std::move(r));
                    }
                    done_cv.notify_all();
                    budget_release(1);
                });
        }
    });

    std::size_t success = 0, failed = 0, skipped = 0, singleton = 0, fixed = 0;
    std::size_t genomes_done = 0, reps_done = 0;
    const std::size_t total = taxa.size();
    const bool tty = isatty(STDERR_FILENO);
    auto t_start = std::chrono::steady_clock::now();
    auto t_last_tty = t_start;

    auto fmt_duration = [](double s) -> std::string {
        int h = static_cast<int>(s) / 3600;
        int m = (static_cast<int>(s) % 3600) / 60;
        int sec = static_cast<int>(s) % 60;
        char buf[32];
        if (h > 0) std::snprintf(buf, sizeof(buf), "%d:%02d:%02d", h, m, sec);
        else       std::snprintf(buf, sizeof(buf), "%d:%02d", m, sec);
        return buf;
    };
    auto fmt_count = [](std::size_t n) -> std::string {
        if (n >= 1000000) { char b[32]; std::snprintf(b, sizeof(b), "%.1fM", n / 1e6); return b; }
        if (n >= 1000)    { char b[32]; std::snprintf(b, sizeof(b), "%.1fk", n / 1e3); return b; }
        return std::to_string(n);
    };

    for (std::size_t collected = 0; collected < total; ++collected) {
        TaxonResult result;
        {
            std::unique_lock lock(done_mutex);
            done_cv.wait(lock, [&] { return !done_queue.empty(); });
            result = std::move(done_queue.front());
            done_queue.pop();
        }
        switch (result.status) {
            case TaxonStatus::SUCCESS:   ++success;   break;
            case TaxonStatus::FAILED:    ++failed;    break;
            case TaxonStatus::SKIPPED:   ++skipped;   break;
            case TaxonStatus::SINGLETON: ++singleton; break;
            case TaxonStatus::FIXED:     ++fixed;     break;
        }
        genomes_done += result.n_genomes;
        reps_done    += result.n_representatives;

        if (result.status == TaxonStatus::FAILED) {
            if (tty) std::fprintf(stderr, "\r\033[K");
            spdlog::warn("Taxon '{}' failed: {}", result.taxonomy, result.error_message);
        }

        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - t_start).count();
        const bool last = (collected + 1 == total);

        if (tty) {
            double since_last = std::chrono::duration<double>(now - t_last_tty).count();
            if (since_last >= 0.25 || last) {
                t_last_tty = now;
                double pct = 100.0 * (collected + 1) / total;
                std::string eta_str = "?";
                if (collected > 0) {
                    double eta = elapsed / (collected + 1) * (total - collected - 1);
                    eta_str = fmt_duration(eta);
                }
                std::fprintf(stderr,
                    "\r  %zu/%zu taxa (%.1f%%)  |  %s genomes  |  %s reps  |  %s elapsed  |  ETA %s    ",
                    collected + 1, total, pct,
                    fmt_count(genomes_done).c_str(),
                    fmt_count(reps_done).c_str(),
                    fmt_duration(elapsed).c_str(),
                    eta_str.c_str());
                std::fflush(stderr);
                if (last) std::fprintf(stderr, "\n");
            }
        }

        if ((collected + 1) % 100 == 0 || last) {
            spdlog::info("Progress: {}/{} ({:.1f}%) | {} genomes | {} reps | elapsed {}",
                         collected + 1, total,
                         100.0 * (collected + 1) / total,
                         fmt_count(genomes_done),
                         fmt_count(reps_done),
                         fmt_duration(elapsed));
        }
    }

    scheduler.join();

    spdlog::info("Done: {} success, {} failed, {} singleton, {} fixed, {} skipped",
                 success, failed, singleton, fixed, skipped);
}

int run_pipeline(Config& cfg) {
    // 1. Setup directories
    fs::path results_dir = cfg.out_dir
        ? *cfg.out_dir / cfg.prefix
        : fs::current_path() / cfg.prefix;
    fs::create_directories(results_dir);

    fs::path temp_dir = cfg.tmp_dir / ("geodesic-" + cfg.timestamp);
    fs::create_directories(temp_dir);

    cfg.results_dir = results_dir;
    cfg.temp_dir = temp_dir;
    cfg.log_file = results_dir / "geodesic.log";

    // 2. Setup logging
    int verbosity = cfg.debug ? 3 : cfg.verbosity;
    setup_logging(cfg.log_file, verbosity);
    spdlog::info("geodesic starting (timestamp {})", cfg.timestamp);
    spdlog::info("Results dir: {}", results_dir.string());
    spdlog::info("Temp dir: {}", temp_dir.string());

    // 3. Load input
    auto genome_rows = read_genomes_tsv(cfg.tax_file);

    std::unordered_map<std::string, CheckM2Quality> checkm2;
    if (cfg.checkm2_file) {
        checkm2 = read_checkm2_tsv(*cfg.checkm2_file);
        auto matched = count_checkm2_matches(genome_rows, checkm2);
        spdlog::info("CheckM2: {} of {} genomes have quality data",
                     matched, genome_rows.size());
    }

    std::unordered_map<std::string, GuncQuality> gunc_scores;
    if (cfg.gunc_file) {
        gunc_scores = read_gunc_tsv(*cfg.gunc_file);
        spdlog::info("GUNC: {} entries loaded from {}", gunc_scores.size(),
                     cfg.gunc_file->string());
    }
    const std::unordered_map<std::string, GuncQuality>* gunc_scores_ptr =
        cfg.gunc_file ? &gunc_scores : nullptr;

    std::unordered_map<std::string, std::string> fixed_taxa;
    if (cfg.fixed_taxa_file) {
        fixed_taxa = read_fixed_taxa_tsv(*cfg.fixed_taxa_file);
    }

    // Load NCBI taxdump for Eukaryote/Virus taxonomy resolution (optional).
    std::unique_ptr<NcbiTaxdb> ncbi_taxdb;
    if (cfg.ncbi_taxdump_dir) {
        NcbiTaxdb::ensure_fresh(*cfg.ncbi_taxdump_dir);
        auto db = NcbiTaxdb::load(*cfg.ncbi_taxdump_dir);
        spdlog::info("NCBI taxdump loaded: {} nodes, timestamp: {}",
                     db.size(),
                     [&]() -> std::string {
                         auto ts = NcbiTaxdb::dump_timestamp(*cfg.ncbi_taxdump_dir);
                         if (!ts) return "unknown";
                         auto t = std::chrono::system_clock::to_time_t(*ts);
                         char buf[32];
                         std::strftime(buf, sizeof(buf), "%Y-%m-%d", std::localtime(&t));
                         return buf;
                     }());
        ncbi_taxdb = std::make_unique<NcbiTaxdb>(std::move(db));
    }

    auto genomes = rows_to_genomes(genome_rows, checkm2, ncbi_taxdb.get());

    // 5. Group by taxonomy and build Taxon objects
    auto taxa = group_by_taxonomy(genomes, fixed_taxa);

    // 6. Filter by selected taxa
    if (cfg.selected_taxa_file) {
        auto selected = read_selected_taxa(*cfg.selected_taxa_file);
        std::erase_if(taxa, [&](const Taxon& t) {
            return selected.find(t.taxonomy) == selected.end();
        });
        spdlog::info("Filtered to {} taxa by selected taxa file", taxa.size());
    }

    // 8. Log stats
    std::size_t total_genomes = 0;
    for (const auto& t : taxa) total_genomes += t.size();
    spdlog::info("{} taxa, {} genomes total", taxa.size(), total_genomes);

    // In-memory accumulator: receives a TaxonOutput for every completed taxon.
    RunState run_state;

    // Open genome pack if --pack was provided.
    // Single .gpk directory → SinglePackReader.
    // Directory containing multiple .gpk subdirectories → MultiPackReader (no merge needed).
    std::unique_ptr<IPackReader> gpk_reader;
    if (cfg.pack_dir.has_value()) {
        try {
            const auto& pack_path = *cfg.pack_dir;
            if (pack_path.extension() == ".gpk") {
                auto ar = std::make_unique<genopack::ArchiveReader>();
                ar->open(pack_path);
                gpk_reader = std::make_unique<SinglePackReader>(std::move(ar));
                spdlog::info("genopack single archive opened: {}", pack_path.string());
            } else {
                gpk_reader = MultiPackReader::open_dir(pack_path);
                spdlog::info("genopack multi-pack opened: {} archives, {} genomes",
                             static_cast<MultiPackReader*>(gpk_reader.get())->n_archives(),
                             static_cast<MultiPackReader*>(gpk_reader.get())->n_genomes());
            }

            // Filter taxa to only genomes indexed in the pack.
            std::unordered_set<std::string> archive_accessions;
            archive_accessions.reserve(genome_rows.size());
            gpk_reader->scan_genome_accessions(
                [&](std::string_view acc, genopack::GenomeId) {
                    archive_accessions.emplace(acc);
                });
            spdlog::info("genopack: {} genomes in archive", archive_accessions.size());

            size_t n_before = 0, n_after = 0;
            for (auto& taxon : taxa) {
                n_before += taxon.size();
                taxon.genomes.erase(
                    std::remove_if(taxon.genomes.begin(), taxon.genomes.end(),
                        [&](const Genome& g) {
                            return archive_accessions.count(g.accession) == 0;
                        }),
                    taxon.genomes.end());
                n_after += taxon.size();
            }
            taxa.erase(std::remove_if(taxa.begin(), taxa.end(),
                                      [](const Taxon& t) { return t.size() == 0; }),
                       taxa.end());
            spdlog::info("genopack filter: {}/{} genomes in taxa after archive filter",
                         n_after, n_before);
        } catch (const std::exception& e) {
            spdlog::warn("Failed to open genopack archive at {}: {} — proceeding without",
                         cfg.pack_dir->string(), e.what());
            gpk_reader.reset();
        }
    }

    // 9. Parallel processing
    process_taxa_parallel(taxa, cfg, run_state,
                          gpk_reader.get(), gunc_scores_ptr);

    // 11. Summary and output
    ResultsWriter writer(results_dir, cfg.prefix);
    writer.write_all(run_state, genome_rows);

    ReportWriter report_writer(results_dir, cfg.prefix, cfg.timestamp);
    report_writer.write(run_state);

    // Write GEODF results file if requested
    if (!cfg.geodf_output.empty()) {
        try {
            geodf::GeodfWriter geodf_writer(cfg.geodf_output);

            if (cfg.pack_dir.has_value() && cfg.pack_dir->extension() == ".gpk") {
                uint64_t snap = geodf::gpk_snapshot_hash(*cfg.pack_dir);
                uint32_t ph   = geodf::hash_run_params(cfg.kmer_size, cfg.sketch_size,
                                                       cfg.syncmer_s, cfg.ani_threshold);
                geodf_writer.set_provenance(snap, ph);
            }

            for (const auto& taxon : run_state.taxa()) {
                geodf::TaxonResult tr;
                tr.taxonomy      = taxon.result.taxonomy;
                tr.ani_threshold = static_cast<float>(cfg.ani_threshold);

                if (taxon.result.status == TaxonStatus::FAILED) {
                    tr.stage         = geodf::PipelineStage::FAILED;
                    tr.error_message = taxon.result.error_message;
                } else {
                    tr.stage = geodf::PipelineStage::COMPLETE;

                    // Build genome_ids, is_rep, contamination, all_accessions from TaxonOutput
                    const std::unordered_set<std::string> rep_set(
                        taxon.representatives.begin(), taxon.representatives.end());
                    for (uint32_t i = 0; i < static_cast<uint32_t>(taxon.all_accessions.size()); ++i) {
                        const auto& acc = taxon.all_accessions[i];
                        tr.genome_ids.push_back(i);
                        tr.is_rep.push_back(rep_set.count(acc) > 0);
                        tr.contamination.push_back(0.0f);
                        tr.all_accessions.push_back(acc);
                    }

                    // Build reps list in order
                    for (uint32_t i = 0; i < static_cast<uint32_t>(taxon.representatives.size()); ++i) {
                        geodf::RepGenome rep;
                        rep.genome_id = i;
                        rep.accession = taxon.representatives[i];
                        tr.reps.push_back(std::move(rep));
                    }
                }

                geodf_writer.write_taxon(tr);
            }

            geodf_writer.close();
            spdlog::info("GEODF results written to {}", cfg.geodf_output.string());
        } catch (const std::exception& e) {
            spdlog::warn("GEODF write failed: {}", e.what());
        }
    }

    // Write lock file if requested
    if (!cfg.lock_output.empty()) {
        try {
            derep::LockData lock;
            lock.kmer_size     = cfg.kmer_size;
            lock.sketch_size   = cfg.sketch_size;
            lock.syncmer_s     = cfg.syncmer_s;
            lock.ani_threshold = cfg.ani_threshold;
            lock.params_hash   = geodf::hash_run_params(cfg.kmer_size, cfg.sketch_size,
                                                        cfg.syncmer_s, cfg.ani_threshold);
            if (cfg.pack_dir.has_value() && cfg.pack_dir->extension() == ".gpk") {
                lock.gpk_path        = *cfg.pack_dir;
                lock.gpk_snapshot_id = geodf::gpk_snapshot_hash(*cfg.pack_dir);
            }
            lock.geodf_path = cfg.geodf_output;
            if (!cfg.geodf_output.empty())
                lock.geodf_hash = derep::file_tail_hash(cfg.geodf_output);
            lock.n_taxa = run_state.taxa().size();
            lock.n_genomes = run_state.total_genomes();
            lock.n_reps    = run_state.total_reps();
            {
                auto now = std::chrono::system_clock::now();
                auto tt  = std::chrono::system_clock::to_time_t(now);
                char buf[32];
                std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&tt));
                lock.timestamp = buf;
            }
            write_lock_file(cfg.lock_output, lock);
            spdlog::info("Lock file written to {}", cfg.lock_output.string());
        } catch (const std::exception& e) {
            spdlog::warn("Lock file write failed: {}", e.what());
        }
    }

    // 12. Cleanup
    std::error_code ec;
    fs::remove_all(temp_dir, ec);
    if (ec) spdlog::warn("Failed to remove temp dir {}: {}", temp_dir.string(), ec.message());

    return run_state.total_failed() > 0 ? 1 : 0;
}

} // namespace derep
