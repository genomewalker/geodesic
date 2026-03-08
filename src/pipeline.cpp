#include "pipeline.hpp"
#include "parallel/taxon_processor.hpp"
#include "core/genome_cache.hpp"
#include "core/logging.hpp"
#include "core/types.hpp"
#include "db/db_manager.hpp"
#include "db/embedding_store.hpp"
#include "db/operations.hpp"
#include "db/schema.hpp"
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
    GenomeCache cache;

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
            const int t_giant  = total_budget;
            const int t_large  = std::max(1, total_budget / 4);
            const int t_medium = std::max(1, total_budget / 8);
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
    static constexpr size_t TINY_SCHED_THRESHOLD = 10;
    static constexpr size_t TINY_BATCH_SIZE = 100;

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
                [&taxa, i, &cfg, &db, &cache, emb_store_ptr, gunc_scores_ptr,
                 &done_queue, &done_mutex, &done_cv,
                 &budget_release, acquired] {
                    auto result = process_taxon(taxa[i], cfg, acquired, db, cache, emb_store_ptr, gunc_scores_ptr);
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
                [batch_taxa, &cfg, &db, &cache,
                 &done_queue, &done_mutex, &done_cv,
                 &budget_release] {
                    auto results = process_tiny_batch(batch_taxa, cfg, db, cache);
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
    const std::size_t total = taxa.size();
    auto t_start = std::chrono::steady_clock::now();
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
        if (result.status == TaxonStatus::FAILED) {
            spdlog::warn("Taxon '{}' failed: {}", result.taxonomy, result.error_message);
        }
        const bool do_progress = is_quiet() && isatty(STDERR_FILENO);
        const bool do_log = (collected + 1) % 100 == 0 || collected + 1 == total;
        if (do_log || do_progress) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration<double>(now - t_start).count();
            if (do_progress) {
                std::fprintf(stderr, "\r  %zu/%zu (%.1f%%) | elapsed %.0fs    ",
                             collected + 1, total, 100.0 * (collected + 1) / total, elapsed);
                std::fflush(stderr);
                if (collected + 1 == total) std::fprintf(stderr, "\n");
            }
            if (do_log) {
                spdlog::info("Progress: {}/{} ({:.1f}%) elapsed {:.0f}s",
                             collected + 1, total, 100.0 * (collected + 1) / total, elapsed);
            }
        }
    }

    scheduler.join();

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

} // namespace derep
