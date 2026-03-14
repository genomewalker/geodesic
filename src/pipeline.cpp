#include "pipeline.hpp"
#include "parallel/taxon_processor.hpp"
#include "core/logging.hpp"
#include "core/types.hpp"
#include "core/sketch/minhash.hpp"
#include "db/async_writer.hpp"
#include "db/db_manager.hpp"
#include "db/embedding_store.hpp"
#include "db/genome_pack.hpp"
#include "db/operations.hpp"
#include "db/schema.hpp"
#include "db/sketch_store.hpp"
#include "io/gz_reader.hpp"
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

std::vector<Genome> rows_to_genomes(
    const std::vector<GenomeRow>& rows,
    const std::unordered_map<std::string, CheckM2Quality>& checkm2) {
    std::vector<Genome> genomes;
    genomes.reserve(rows.size());
    for (const auto& row : rows) {
        Genome g;
        g.accession = row.accession;
        g.taxonomy = row.taxonomy;
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

// Returns true when the species field is unresolved (ends with "s__" with no name after it).
// Such genomes cannot be safely grouped — they may represent distinct species that
// GTDB-tk failed to assign. Each is treated as a singleton to preserve diversity.
static bool has_empty_species(const std::string& taxonomy) {
    auto pos = taxonomy.rfind(";s__");
    if (pos == std::string::npos) return false;
    const auto after = taxonomy.find_first_not_of(" \t", pos + 4);
    return after == std::string::npos;
}

std::vector<Taxon> group_by_taxonomy(
    std::vector<Genome>& genomes,
    const std::unordered_map<std::string, std::string>& fixed_taxa) {
    std::unordered_map<std::string, std::vector<Genome>> groups;
    std::vector<Taxon> taxa;

    size_t n_empty_species = 0;
    for (auto& g : genomes) {
        if (has_empty_species(g.taxonomy)) {
            ++n_empty_species;
            Taxon t;
            t.taxonomy = g.taxonomy;
            if (auto it = fixed_taxa.find(g.taxonomy); it != fixed_taxa.end())
                t.forced_representative = it->second;
            t.genomes.push_back(std::move(g));
            taxa.push_back(std::move(t));
        } else {
            groups[g.taxonomy].push_back(std::move(g));
        }
    }

    if (n_empty_species > 0)
        spdlog::info("{} genome(s) with unresolved species (s__) treated as singletons",
                     n_empty_species);

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

    // 3. Open DB and create schema
    // Only auto-place DB if user used the bare default name (not an explicit path)
    if (cfg.db_path == fs::path("geodesic.db")) {
        cfg.db_path = results_dir / (cfg.prefix + ".db");
    }
    db::DBManager db({.db_path = cfg.db_path});
    db::schema::create_all(db.thread_connection());
    db::schema::migrate(db.thread_connection());
    db::ops::migrate_pipeline_stages_v7(db);

    // 4. Load input
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

    auto genomes = rows_to_genomes(genome_rows, checkm2);

    // 5. Group by taxonomy and build Taxon objects
    auto taxa = group_by_taxonomy(genomes, fixed_taxa);

    // 6. Insert taxa and genomes into DB (skip if already loaded for resume)
    {
        auto count_result = db.query("SELECT COUNT(*) FROM genomes");
        int64_t existing_count = 0;
        if (!count_result->HasError()) {
            auto chunk = count_result->Fetch();
            if (chunk && chunk->size() > 0) {
                existing_count = chunk->GetValue(0, 0).GetValue<int64_t>();
            }
        }
        if (existing_count == 0) {
            spdlog::info("Inserting {} taxa and {} genomes into database (bulk)...",
                         taxa.size(), genome_rows.size());
            db::ops::insert_taxa_and_genomes_bulk(db, taxa);
        } else {
            spdlog::info("Resuming: {} genomes already in database, skipping insert", existing_count);
        }
    }

    // 7. Filter by selected taxa
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

    // 9. Parallel processing
    // Thread budget: total = workers * threads. Large taxa claim more slots
    // (reducing concurrency); small taxa claim 1 (maximizing parallelism).
    // A scheduler thread submits tasks as budget permits, while the main thread
    // collects results independently — no head-of-line blocking either way.
    const int total_budget = cfg.workers * cfg.threads;
    // Pool size = total_budget so there are always enough worker slots.
    BS::thread_pool pool(static_cast<BS::concurrency_t>(total_budget));

    // Build execution plan from the full taxa size distribution.
    // Taxa are sorted large-to-small. We compute quantile thresholds and assign
    // each taxon a static thread count before the run starts, forming tiers:
    //   Giant  (>= p95): all threads — process serially but fast
    //   Large  (>= p75): total/4 threads — a few concurrent
    //   Medium (>= p25): 2 threads — moderate concurrency
    //   Small  (<  p25): 1 thread — full concurrency (up to total_budget)
    // This is computed once and stored per-taxon so the scheduler just reads it.
    std::vector<int> taxon_threads(taxa.size(), 1);
    {
        const size_t n = taxa.size();
        if (n >= 50) {
            // Large fleet: quantile-based tiers give reliable percentile thresholds.
            std::vector<size_t> sizes(n);
            for (size_t i = 0; i < n; ++i) sizes[i] = taxa[i].size();
            std::vector<size_t> sorted_sizes = sizes;
            std::sort(sorted_sizes.begin(), sorted_sizes.end());
            const size_t p25 = sorted_sizes[n * 25 / 100];
            const size_t p75 = sorted_sizes[n * 75 / 100];
            const size_t p95 = sorted_sizes[n * 95 / 100];
            // All tiers derived from total_budget to saturate cores:
            //   Giant  (p95+): total_budget  threads, 1 concurrent
            //   Large  (p75+): total_budget/4 threads, 4 concurrent
            //   Medium (p25+): total_budget/8 threads, 8 concurrent  (min 1)
            //   Small  (<p25): 1 thread,      total_budget concurrent
            const int t_giant  = std::max(12, total_budget / 2);  // 2 concurrent giants
            const int t_large  = std::max(1, total_budget / 6);   // 6 concurrent large
            const int t_medium = std::max(1, total_budget / 12);  // 12 concurrent medium
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
            // Small fleet: quantile thresholds collapse with few taxa, so allocate
            // threads proportional to genome count (work scales as O(n) for OPH+embed).
            // Tiny taxa (≤10 genomes, handled as batches) always get 1 thread.
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

    // Async DB writer: background thread batches and flushes taxon results.
    // Workers push TaxonWritePayload and return immediately; writer handles all DB I/O.
    db::AsyncDBWriter async_writer(db);
    db::AsyncDBWriter* async_writer_ptr = &async_writer;

    // Create shared embedding store for incremental updates
    std::unique_ptr<db::EmbeddingStore> emb_store;
    if (cfg.embedding_db.has_value()) {
        emb_store = std::make_unique<db::EmbeddingStore>(cfg.embedding_dim);
        if (!emb_store->open(*cfg.embedding_db)) {
            spdlog::warn("Failed to open embedding store at {}, proceeding without",
                         cfg.embedding_db->string());
            emb_store.reset();
        }
    }
    db::EmbeddingStore* emb_store_ptr = emb_store.get();

    // Open sketch cache if --sketch-db was provided
    std::unique_ptr<db::SketchStore> sketch_store;
    if (cfg.sketch_db.has_value()) {
        sketch_store = std::make_unique<db::SketchStore>();
        try {
            db::SketchStore::Meta meta{
                .kmer_size   = cfg.kmer_size,
                .sketch_size = cfg.sketch_size,
                .syncmer_s   = cfg.syncmer_s,
            };
            sketch_store->open(*cfg.sketch_db, meta);
            spdlog::info("Sketch cache opened: {}", cfg.sketch_db->string());
        } catch (const std::exception& e) {
            spdlog::warn("Failed to open sketch cache at {}: {} — proceeding without",
                         cfg.sketch_db->string(), e.what());
            sketch_store.reset();
        }
    }
    db::SketchStore* sketch_store_ptr = sketch_store.get();

    // Open genome pack if --pack was provided
    std::unique_ptr<db::GenomePack> genome_pack;
    if (cfg.pack_dir.has_value()) {
        genome_pack = std::make_unique<db::GenomePack>();
        try {
            genome_pack->open_read(*cfg.pack_dir);
            spdlog::info("Genome pack opened: {}", cfg.pack_dir->string());
        } catch (const std::exception& e) {
            spdlog::warn("Failed to open genome pack at {}: {} — proceeding without",
                         cfg.pack_dir->string(), e.what());
            genome_pack.reset();
        }
    }
    db::GenomePack* genome_pack_ptr = genome_pack.get();

    // Adaptive thread budget: acquire as many slots as available (up to desired),
    // minimum 1 to guarantee forward progress.
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

    // Completion queue: workers push results on finish; main thread pulls.
    std::queue<TaxonResult> done_queue;
    std::mutex done_mutex;
    std::condition_variable done_cv;

    // Partition taxa into large (size > 10) and tiny batches (size <= 10, grouped by 100).
    static constexpr size_t TINY_SCHED_THRESHOLD = 100;  // batch taxa ≤100 genomes
    static constexpr size_t TINY_BATCH_SIZE = 200;       // larger batches

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

    // Scheduler thread: submits tasks as budget permits, runs concurrently
    // with the main collection loop below.
    std::thread scheduler([&] {
        // Submit large taxa individually (existing logic)
        for (size_t i : large_indices) {
            int desired  = taxon_threads[i];
            int acquired = budget_acquire(desired);
            pool.detach_task(
                [&taxa, i, &cfg, &db, emb_store_ptr, gunc_scores_ptr, async_writer_ptr,
                 sketch_store_ptr, genome_pack_ptr,
                 &done_queue, &done_mutex, &done_cv,
                 &budget_release, acquired] {
                    auto result = process_taxon(taxa[i], cfg, acquired, db, emb_store_ptr,
                                               gunc_scores_ptr, false, async_writer_ptr,
                                               sketch_store_ptr, genome_pack_ptr);
                    {
                        std::lock_guard lock(done_mutex);
                        done_queue.push(std::move(result));
                    }
                    done_cv.notify_one();
                    budget_release(acquired);
                });
        }
        // Submit tiny batches: 1 budget slot per batch of up to 100 taxa
        for (const auto& batch_indices : tiny_batches) {
            budget_acquire(1);
            std::vector<const Taxon*> batch_taxa;
            batch_taxa.reserve(batch_indices.size());
            for (size_t i : batch_indices)
                batch_taxa.push_back(&taxa[i]);
            pool.detach_task(
                [batch_taxa, &cfg, &db, async_writer_ptr, sketch_store_ptr, genome_pack_ptr,
                 &done_queue, &done_mutex, &done_cv,
                 &budget_release] {
                    auto results = process_tiny_batch(batch_taxa, cfg, db, async_writer_ptr,
                                                      sketch_store_ptr, genome_pack_ptr);
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

    // 10. Collect results in completion order
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
            if (tty) std::fprintf(stderr, "\r\033[K");  // clear progress line before warning
            spdlog::warn("Taxon '{}' failed: {}", result.taxonomy, result.error_message);
        }

        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - t_start).count();
        const bool last = (collected + 1 == total);

        // TTY progress bar: update at most ~4x/s to avoid I/O overhead
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

        // Log file progress every 100 taxa
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

    // Flush all pending DB writes before writing results files
    spdlog::info("Flushing async DB writer ({} pending)...", async_writer.pending());
    async_writer.shutdown();
    spdlog::info("Async writer flushed ({} taxa, {} rows written)",
                 async_writer.total_taxa_written(), async_writer.total_rows_written());

    // 11. Summary and output
    spdlog::info("Done: {} success, {} failed, {} singleton, {} fixed, {} skipped",
                 success, failed, singleton, fixed, skipped);

    ResultsWriter writer(results_dir, cfg.prefix);
    writer.write_all(db);

    ReportWriter report_writer(results_dir, cfg.prefix, cfg.timestamp);
    report_writer.write(db);

    if (!cfg.keep_intermediates) {
        db::schema::prune_intermediate_tables(db.thread_connection());
    }
    db.checkpoint();

    // 12. Cleanup
    std::error_code ec;
    fs::remove_all(temp_dir, ec);
    if (ec) spdlog::warn("Failed to remove temp dir {}: {}", temp_dir.string(), ec.message());

    return failed > 0 ? 1 : 0;
}

// ============================================================================
// run_sketch: pre-compute OPH sketches for all genomes and persist to DuckDB.
// Processes genomes in chunks: parallel I/O+OPH per chunk, then batch write.
// Chunk size bounds peak memory to CHUNK × ~20 KB = ~1 GB.
// ============================================================================
int run_sketch(Config& cfg) {
    // 1. Setup logging (console-only for sketch subcommand)
    {
        int verbosity = cfg.verbosity;
        auto sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        sink->set_level(verbosity == 0 ? spdlog::level::warn :
                        verbosity >= 2 ? spdlog::level::debug :
                                        spdlog::level::info);
        auto logger = std::make_shared<spdlog::logger>("geodesic", sink);
        logger->set_level(spdlog::level::info);
        logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
        spdlog::set_default_logger(logger);
        spdlog::flush_on(spdlog::level::info);
    }

    spdlog::info("geodesic sketch: kmer={} sketch={} syncmer={} threads={}",
                 cfg.kmer_size, cfg.sketch_size, cfg.syncmer_s, cfg.threads);

    // 2. Parse TSV
    spdlog::info("Parsing taxonomy file: {}", cfg.tax_file.string());
    auto rows = read_genomes_tsv(cfg.tax_file);
    spdlog::info("{} genomes in taxonomy file", rows.size());

    // 3. Open SketchStore
    db::SketchStore store;
    db::SketchStore::Meta meta{
        .kmer_size   = cfg.kmer_size,
        .sketch_size = cfg.sketch_size,
        .syncmer_s   = cfg.syncmer_s,
    };
    store.open(*cfg.sketch_db, meta);

    // 4. Load already-completed and failed accessions
    auto completed = store.load_completed_accessions();
    auto failed    = store.load_failed_accessions();

    // Build work list: skip completed; skip failed (can add --retry flag later)
    struct WorkItem { std::string accession; std::string taxonomy; std::string path; };
    std::vector<WorkItem> work;
    work.reserve(rows.size());
    for (const auto& row : rows) {
        if (completed.count(row.accession) || failed.count(row.accession)) continue;
        work.push_back({row.accession, row.taxonomy, row.file_path.string()});
    }
    spdlog::info("{} genomes to sketch ({} already done, {} previously failed)",
                 work.size(), completed.size(), failed.size());

    if (work.empty()) {
        spdlog::info("All genomes already sketched. Done.");
        store.checkpoint();
        return 0;
    }

    // 5. Chunk-based parallel I/O+OPH → batch write
    // Each chunk: submit all tasks, wait, then write results.
    // Bounds peak memory to CHUNK × ~20 KB = ~1 GB.
    const size_t CHUNK = 50000;
    const size_t total = work.size();

    MinHasher::Config hasher_cfg{
        .kmer_size   = cfg.kmer_size,
        .sketch_size = cfg.sketch_size,
        .syncmer_s   = cfg.syncmer_s,
    };

    BS::thread_pool pool(cfg.threads);
    size_t n_done = 0;
    size_t n_failed = 0;

    for (size_t chunk_start = 0; chunk_start < total; chunk_start += CHUNK) {
        const size_t chunk_end = std::min(chunk_start + CHUNK, total);
        const size_t chunk_sz  = chunk_end - chunk_start;

        // Per-chunk result storage (pre-allocated, indexed by position in chunk)
        std::vector<db::SketchStore::SketchRecord> results(chunk_sz);
        std::vector<std::string> errors(chunk_sz);

        // Submit one task per genome in the chunk
        for (size_t ci = 0; ci < chunk_sz; ++ci) {
            pool.detach_task([&, ci] {
                const auto& item = work[chunk_start + ci];
                auto& rec = results[ci];
                rec.accession = item.accession;
                rec.taxonomy  = item.taxonomy;
                try {
                    auto buf = GzReader::decompress_file(item.path);
                    // Thread-local hasher: constructed once per thread
                    static thread_local MinHasher tl_hasher(hasher_cfg);
                    auto oph = tl_hasher.sketch_oph_with_positions_from_buffer(
                        buf.data(), buf.size(), cfg.sketch_size);
                    const int m = cfg.sketch_size;
                    rec.oph_sig.resize(m);
                    for (int b = 0; b < m; ++b)
                        rec.oph_sig[b] = static_cast<uint16_t>(oph.signature[b]);
                    rec.n_real_bins   = static_cast<uint32_t>(oph.n_real_bins);
                    rec.genome_length = static_cast<uint64_t>(oph.genome_length);
                } catch (const std::exception& e) {
                    errors[ci] = e.what();
                }
            });
        }
        pool.wait();

        // Partition results and write
        std::vector<db::SketchStore::SketchRecord> write_batch;
        std::vector<db::SketchStore::SketchFailure> fail_batch;
        write_batch.reserve(chunk_sz);

        for (size_t ci = 0; ci < chunk_sz; ++ci) {
            if (!errors[ci].empty()) {
                db::SketchStore::SketchFailure f;
                f.accession     = results[ci].accession;
                f.taxonomy      = results[ci].taxonomy;
                f.file_path     = work[chunk_start + ci].path;
                f.error_message = errors[ci];
                fail_batch.push_back(std::move(f));
                ++n_failed;
            } else {
                write_batch.push_back(std::move(results[ci]));
            }
        }

        if (!write_batch.empty()) store.insert_batch(write_batch);
        if (!fail_batch.empty()) store.record_failures(fail_batch);
        n_done += write_batch.size();

        spdlog::info("Sketch progress: {}/{} ({:.1f}%) | {} failed",
                     n_done + n_failed, total,
                     100.0 * (n_done + n_failed) / total,
                     n_failed);
    }

    spdlog::info("Sketch complete: {} succeeded, {} failed", n_done, n_failed);
    store.checkpoint();
    return n_failed > 0 ? 1 : 0;
}

// ============================================================================
// run_pack: pre-convert scattered gzipped FASTA files into taxonomy-grouped,
// zstd-compressed pack files on fast local storage.
// ============================================================================
int run_pack(Config& cfg) {
    // 1. Setup logging
    {
        int verbosity = cfg.verbosity;
        auto sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        sink->set_level(verbosity == 0 ? spdlog::level::warn :
                        verbosity >= 2 ? spdlog::level::debug :
                                        spdlog::level::info);
        auto logger = std::make_shared<spdlog::logger>("geodesic", sink);
        logger->set_level(spdlog::level::info);
        logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
        spdlog::set_default_logger(logger);
        spdlog::flush_on(spdlog::level::info);
    }

    spdlog::info("geodesic pack: threads={} io_threads={} zstd_level={}",
                 cfg.threads, cfg.io_threads, cfg.pack_zstd_level);

    // 2. Parse TSV
    spdlog::info("Parsing taxonomy file: {}", cfg.tax_file.string());
    auto rows = read_genomes_tsv(cfg.tax_file);
    spdlog::info("{} genomes in taxonomy file", rows.size());

    // 3. Group by taxonomy
    std::unordered_map<std::string, std::vector<std::pair<std::string, std::string>>> taxa_map;
    // taxa_map: taxonomy -> [(accession, file_path)]
    taxa_map.reserve(rows.size() / 10);
    for (const auto& row : rows)
        taxa_map[row.taxonomy].emplace_back(row.accession, row.file_path.string());

    spdlog::info("{} unique taxa", taxa_map.size());

    // 4. Open GenomePack for writing
    db::GenomePack::Config pack_cfg;
    pack_cfg.zstd_level = cfg.pack_zstd_level;
    db::GenomePack pack(pack_cfg);
    pack.open_write(*cfg.pack_dir);

    // 5. Resume: load completed taxa and erase from map
    auto completed = pack.load_completed_taxa();
    if (!completed.empty()) {
        spdlog::info("Resuming: {} taxa already packed", completed.size());
        for (const auto& tax : completed)
            taxa_map.erase(tax);
    }
    spdlog::info("{} taxa to pack", taxa_map.size());

    if (taxa_map.empty()) {
        spdlog::info("All taxa already packed. Done.");
        pack.checkpoint();
        return 0;
    }

    // 6. Process taxa with thread pool
    const int io_threads = (cfg.io_threads > 0) ? cfg.io_threads : cfg.threads;
    BS::thread_pool pool(static_cast<BS::concurrency_t>(cfg.threads));

    // Semaphore to bound concurrent NFS readers
    std::counting_semaphore<> io_sem(io_threads);

    std::atomic<size_t> n_taxa_done{0};
    std::atomic<size_t> n_taxa_failed{0};
    const size_t total_taxa = taxa_map.size();

    // Convert to vector for indexed iteration
    std::vector<std::pair<std::string, std::vector<std::pair<std::string, std::string>>>> taxa_vec(
        taxa_map.begin(), taxa_map.end());

    std::mutex result_mutex;

    for (const auto& [taxonomy, genomes] : taxa_vec) {
        pool.detach_task([&, taxonomy = taxonomy, genomes = genomes] {
            try {
                std::vector<std::pair<std::string, std::vector<char>>> genome_bufs;
                genome_bufs.reserve(genomes.size());

                for (const auto& [accession, path] : genomes) {
                    io_sem.acquire();
                    std::vector<char> buf;
                    std::string err;
                    try {
                        buf = GzReader::decompress_file(path);
                    } catch (const std::exception& e) {
                        err = e.what();
                    }
                    io_sem.release();

                    if (!err.empty()) {
                        spdlog::warn("pack: failed to read {} ({}): {}", accession, path, err);
                        continue;
                    }
                    genome_bufs.emplace_back(accession, std::move(buf));
                }

                if (!genome_bufs.empty())
                    pack.write_taxon(taxonomy, genome_bufs);

                size_t done = ++n_taxa_done;
                if (done % 1000 == 0 || done == total_taxa) {
                    spdlog::info("Pack progress: {}/{} taxa ({:.1f}%) | {} failed",
                                 done, total_taxa, 100.0 * done / total_taxa,
                                 n_taxa_failed.load());
                }
            } catch (const std::exception& e) {
                ++n_taxa_failed;
                spdlog::error("pack: failed taxon {}: {}", taxonomy, e.what());
            }
        });
    }

    pool.wait();

    spdlog::info("Pack complete: {}/{} taxa done, {} failed",
                 n_taxa_done.load(), total_taxa, n_taxa_failed.load());
    pack.checkpoint();
    return n_taxa_failed.load() > 0 ? 1 : 0;
}

} // namespace derep
