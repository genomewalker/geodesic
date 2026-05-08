#include "pipeline.hpp"
#include <iostream>
#include <unordered_set>
#include "parallel/taxon_processor.hpp"
#include "state/run_state.hpp"
#include "core/logging.hpp"
#include "core/types.hpp"
#include "core/pack_reader.hpp"
#include "core/multi_pack_reader.hpp"
#include "core/preloaded_pack_reader.hpp"
#include "core/kmer_probe.hpp"
#include "taxonomy/normalize.hpp"
#include "db/geodf/geodf_writer.hpp"
#include "db/geodf/geodf_reader.hpp"
#include "db/grd/grd_writer.hpp"

#include <genopack/archive.hpp>
#include "io/lock_writer.hpp"
#include "io/report_writer.hpp"
#include "io/results_writer.hpp"
#include "derep/derep_archive.hpp"
#include <genopack/archive_set_reader.hpp>
#include "io/tsv_reader.hpp"

#include <BS_thread_pool.hpp>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <unistd.h>
#include <fstream>
#include <map>
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

std::vector<Genome> accessions_to_genomes(
    const std::vector<std::string>& accessions,
    IPackReader& pack,
    const std::unordered_map<std::string, CheckM2Quality>& checkm2) {
    std::vector<Genome> genomes;
    genomes.reserve(accessions.size());
    size_t n_missing_tax = 0;
    for (const auto& acc : accessions) {
        Genome g;
        g.accession = acc;
        g.taxonomy = pack.taxonomy_for_accession(acc);
        if (g.taxonomy.empty()) {
            ++n_missing_tax;
            continue;
        }
        g.taxonomy = taxonomy::normalize_taxonomy(g.taxonomy, acc);
        auto canon = canonical_accession(acc);
        if (auto it = checkm2.find(canon); it != checkm2.end()) {
            g.completeness = it->second.completeness;
            g.contamination = it->second.contamination;
        }
        genomes.push_back(std::move(g));
    }
    if (n_missing_tax > 0)
        spdlog::warn("{} accessions have no taxonomy in pack (skipped)", n_missing_tax);
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
    const std::unordered_map<std::string, GuncQuality>* gunc_scores,
    grd::GrdWriter* grd_writer) {

    RunState* run_state_ptr = &run_state;
    IPackReader* gpk_reader_ptr = gpk_reader;
    const std::unordered_map<std::string, GuncQuality>* gunc_scores_ptr = gunc_scores;
    grd::GrdWriter* grd_writer_ptr = grd_writer;

    const int total_budget = cfg.workers * cfg.threads;
    BS::thread_pool pool(static_cast<BS::concurrency_t>(total_budget));

    // Single-queue scheduler: absolute launch-width bands, pre-sorted desc by size,
    // first-fit backfill. Avoids head-of-line blocking and serial-phase starvation.
    auto launch_band = [total_budget](size_t sz) -> int {
        int t;
        if      (sz <=   500) t = 1;
        else if (sz <=  5000) t = 4;
        else if (sz <= 20000) t = 8;
        else                  t = 12;
        return std::min(t, total_budget);
    };

    struct ReadyTask { size_t idx; int launch; size_t size; };
    std::vector<ReadyTask> tasks;
    tasks.reserve(taxa.size());
    for (size_t i = 0; i < taxa.size(); ++i)
        tasks.push_back({i, launch_band(taxa[i].size()), taxa[i].size()});
    std::sort(tasks.begin(), tasks.end(),
              [](const ReadyTask& a, const ReadyTask& b) { return a.size > b.size; });
    {
        size_t c1 = 0, c4 = 0, c8 = 0, c12 = 0;
        for (const auto& t : tasks) {
            if      (t.launch <= 1) ++c1;
            else if (t.launch <= 4) ++c4;
            else if (t.launch <= 8) ++c8;
            else                    ++c12;
        }
        spdlog::info(
            "Scheduler (single-queue, budget={}): 1t={} 4t={} 8t={} 12t+={}",
            total_budget, c1, c4, c8, c12);
    }

    int budget_avail = total_budget;
    std::mutex budget_mtx;
    std::condition_variable budget_cv;
    auto budget_release = [&](int n) {
        { std::lock_guard lock(budget_mtx); budget_avail += n; }
        budget_cv.notify_all();
    };

    std::queue<TaxonResult> done_queue;
    std::mutex done_mutex;
    std::condition_variable done_cv;

    std::vector<char> dispatched(tasks.size(), 0);
    static constexpr size_t BACKFILL_WINDOW = 64;

    std::thread scheduler([&] {
        size_t head = 0;
        size_t remaining = tasks.size();
        while (remaining > 0) {
            while (head < tasks.size() && dispatched[head]) ++head;
            if (head >= tasks.size()) break;

            std::unique_lock lock(budget_mtx);
            budget_cv.wait(lock, [&] { return budget_avail > 0; });

            ssize_t pick = -1;
            if (tasks[head].launch <= budget_avail) {
                pick = static_cast<ssize_t>(head);
            } else {
                size_t limit = std::min(head + BACKFILL_WINDOW, tasks.size());
                for (size_t j = head + 1; j < limit; ++j) {
                    if (!dispatched[j] && tasks[j].launch <= budget_avail) {
                        pick = static_cast<ssize_t>(j);
                        break;
                    }
                }
                if (pick < 0) {
                    for (size_t j = limit; j < tasks.size(); ++j) {
                        if (!dispatched[j] && tasks[j].launch <= budget_avail) {
                            pick = static_cast<ssize_t>(j);
                            break;
                        }
                    }
                }
                if (pick < 0) {
                    int prev_avail = budget_avail;
                    budget_cv.wait(lock, [&] { return budget_avail > prev_avail; });
                    continue;
                }
            }

            int acquired = tasks[pick].launch;
            budget_avail -= acquired;
            dispatched[pick] = 1;
            --remaining;
            size_t ti = tasks[pick].idx;
            lock.unlock();

            pool.detach_task(
                [&taxa, ti, &cfg, gunc_scores_ptr,
                 gpk_reader_ptr, run_state_ptr, grd_writer_ptr,
                 &done_queue, &done_mutex, &done_cv,
                 &budget_release, acquired] {
                    // Mid-taxon release: shared int so we know how much is still
                    // held when the task ends (callback may or may not fire).
                    auto held = std::make_shared<int>(acquired);
                    std::function<void()> on_serial = nullptr;
                    if (acquired > 1) {
                        on_serial = [held, &budget_release, acquired]() {
                            int extra = acquired - 1;
                            if (*held >= acquired) {
                                *held = 1;
                                budget_release(extra);
                            }
                        };
                    }
                    auto result = process_taxon(taxa[ti], cfg, acquired,
                                               gunc_scores_ptr,
                                               gpk_reader_ptr, run_state_ptr,
                                               grd_writer_ptr,
                                               std::move(on_serial));
                    {
                        std::lock_guard dlock(done_mutex);
                        done_queue.push(std::move(result));
                    }
                    done_cv.notify_one();
                    budget_release(*held);
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

    // Sort taxa by taxonomy for deterministic emission across runs (workers push
    // in completion order). Must be done after join() and before any reader.
    run_state.finalize_sort();

    // Release SKCH buffers now that all taxa are done with sketch loading.
    if (gpk_reader_ptr) {
        size_t skch_mb = gpk_reader_ptr->sketch_memory_bytes() / (1024 * 1024);
        gpk_reader_ptr->release_sketches();
        if (skch_mb > 0) spdlog::info("Released {}MB SKCH buffers", skch_mb);
    }

    spdlog::info("Done: {} success, {} failed, {} singleton, {} fixed, {} skipped",
                 success, failed, singleton, fixed, skipped);
}

void emit_gpd_archive(const Config& cfg, const RunState& run_state,
                      IPackReader* gpk_reader) {
    if (cfg.gpd_output.empty()) return;
    if (!cfg.pack_dir.has_value()) {
        spdlog::warn("--emit-gpd requires --pack; skipping .gpd emission");
        return;
    }
    try {
        fs::path gpd_path = cfg.gpd_output;

        genopack::ArchiveSetReader src;
        src.open(*cfg.pack_dir);

        geodesic::DerepArchiveBuilderConfig gcfg;
        gcfg.output_path      = gpd_path;
        gcfg.embedding_dim    = static_cast<uint16_t>(cfg.embedding_dim);
        gcfg.embedding_dtype  = 1; // f16
        gcfg.emit_armp        = true;
        gcfg.zstd_level       = 6;
        gcfg.geodesic_version = "geodesic 1.0.0";

        geodesic::DerepArchiveBuilder ab(gcfg);
        ab.set_source_pack(src);
        ab.set_params({static_cast<uint8_t>(cfg.kmer_size)},
                      static_cast<uint32_t>(cfg.sketch_size),
                      cfg.seed, cfg.seed + 1,
                      static_cast<float>(cfg.ani_threshold / 100.0));

        auto f32_to_f16 = [](float fv) -> uint16_t {
            uint32_t f;
            std::memcpy(&f, &fv, 4);
            uint32_t sign = (f >> 16) & 0x8000u;
            int32_t  exp  = static_cast<int32_t>((f >> 23) & 0xFF) - 127 + 15;
            uint32_t mant = f & 0x7FFFFFu;
            if (exp <= 0) {
                if (exp < -10) return static_cast<uint16_t>(sign);
                mant |= 0x800000u;
                uint32_t shift = static_cast<uint32_t>(14 - exp);
                uint32_t halfm = mant >> shift;
                if ((mant >> (shift - 1)) & 1) ++halfm;
                return static_cast<uint16_t>(sign | halfm);
            }
            if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
            uint32_t out = sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
            if (mant & 0x1000u) ++out;
            return static_cast<uint16_t>(out);
        };

        auto get_locator = [&](std::string_view acc) -> uint64_t {
            if (!gpk_reader) return 0ULL;
            auto m = gpk_reader->genome_meta_by_accession(acc);
            return m ? static_cast<uint64_t>(m->genome_id) : 0ULL;
        };

        std::unordered_set<std::string> excluded_accs;
        for (const auto& taxon : run_state.taxa()) {
            for (const auto& o : taxon.outliers)
                if (o.excluded) excluded_accs.insert(o.accession);
            for (const auto& f : taxon.failed_genomes)
                excluded_accs.insert(f.accession);
        }

        std::vector<uint16_t> emb_f16(cfg.embedding_dim, 0);
        size_t n_reps_total = 0, n_genomes_total = 0;

        for (const auto& taxon : run_state.taxa()) {
            if (taxon.result.status == TaxonStatus::FAILED) continue;

            const std::unordered_set<std::string> rep_set(
                taxon.representatives.begin(), taxon.representatives.end());
            std::unordered_map<std::string, size_t> rep_idx;
            rep_idx.reserve(taxon.representatives.size());
            for (size_t i = 0; i < taxon.representatives.size(); ++i)
                rep_idx[taxon.representatives[i]] = i;

            const uint16_t kmer_used =
                taxon.sketch_kmer_used > 0
                    ? static_cast<uint16_t>(taxon.sketch_kmer_used)
                    : static_cast<uint16_t>(cfg.kmer_size);

            for (const auto& acc : taxon.all_accessions) {
                const uint64_t loc = get_locator(acc);
                if (excluded_accs.count(acc)) {
                    ab.add(acc, geodesic::DerepArchiveBuilder::Kind::Unclustered,
                           {}, loc, kmer_used, 0, nullptr);
                    ++n_genomes_total;
                    continue;
                }
                auto rit = rep_idx.find(acc);
                if (rit != rep_idx.end()) {
                    const auto& vec = (rit->second < taxon.rep_embeddings.size())
                        ? taxon.rep_embeddings[rit->second]
                        : std::vector<float>();
                    for (int d = 0; d < cfg.embedding_dim; ++d) {
                        float fv = (d < static_cast<int>(vec.size())) ? vec[d] : 0.0f;
                        emb_f16[d] = f32_to_f16(fv);
                    }
                    uint32_t cs = 1;
                    auto csit = taxon.rep_cluster_size.find(acc);
                    if (csit != taxon.rep_cluster_size.end()) cs = csit->second;
                    ab.add(acc, geodesic::DerepArchiveBuilder::Kind::Representative,
                           acc, loc, kmer_used, cs, emb_f16.data());
                    ++n_reps_total;
                    ++n_genomes_total;
                } else {
                    std::string rep_acc;
                    auto mit = taxon.member_to_rep.find(acc);
                    if (mit != taxon.member_to_rep.end()) {
                        rep_acc = mit->second;
                    } else if (!taxon.representatives.empty()) {
                        rep_acc = taxon.representatives.front();
                    }
                    if (rep_acc.empty()) {
                        ab.add(acc, geodesic::DerepArchiveBuilder::Kind::Unclustered,
                               {}, loc, kmer_used, 0, nullptr);
                    } else {
                        ab.add(acc, geodesic::DerepArchiveBuilder::Kind::Member,
                               rep_acc, loc, kmer_used, 0, nullptr);
                    }
                    ++n_genomes_total;
                }
            }
        }

        ab.finalize();
        spdlog::info("wrote derep archive: {} (n_reps={}, n_genomes={}, dim={})",
                     gpd_path.string(), n_reps_total, n_genomes_total,
                     static_cast<int>(cfg.embedding_dim));
    } catch (const std::exception& e) {
        spdlog::warn("GPD emission failed: {} — TSV outputs unaffected", e.what());
    }
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
    auto accessions = read_accession_list(cfg.genomes_file);

    std::unordered_map<std::string, CheckM2Quality> checkm2;
    if (cfg.checkm2_file) {
        checkm2 = read_checkm2_tsv(*cfg.checkm2_file);
        spdlog::info("CheckM2: {} entries loaded", checkm2.size());
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

    // In-memory accumulator: receives a TaxonOutput for every completed taxon.
    RunState run_state;

    // Open genome pack (required — taxonomy, sequences, and sketches are read from pack).
    if (!cfg.pack_dir.has_value())
        throw std::runtime_error("--pack is required");
    std::unique_ptr<IPackReader> gpk_reader;
    {
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
    }

    // Build Genome objects — taxonomy resolved from pack TAXN section.
    auto genomes = accessions_to_genomes(accessions, *gpk_reader, checkm2);

    // Group by taxonomy and build Taxon objects.
    auto taxa = group_by_taxonomy(genomes, fixed_taxa);

    // Log stats
    std::size_t total_genomes = 0;
    for (const auto& t : taxa) total_genomes += t.size();
    spdlog::info("{} taxa, {} genomes total", taxa.size(), total_genomes);

    // 9. Open GRD writer if requested (stream-writes per taxon during parallel processing)
    std::unique_ptr<grd::GrdWriter> grd_writer;
    if (!cfg.grd_output.empty()) {
        grd_writer = std::make_unique<grd::GrdWriter>(cfg.grd_output);
        spdlog::info("GRD output: {}", cfg.grd_output.string());
    }

    // 10. Parallel processing — bucketed by archive, preloading dominant k
    //     k's per archive so adaptive k (maybe_reselect_k) stays RAM-resident.
    //     Set GEODESIC_NO_PRELOAD=1 to bypass.
    const char* no_pre = std::getenv("GEODESIC_NO_PRELOAD");
    const bool bypass_preload = (no_pre && no_pre[0] == '1');

    if (bypass_preload || !gpk_reader) {
        process_taxa_parallel(taxa, cfg, run_state,
                              gpk_reader.get(), gunc_scores_ptr,
                              grd_writer.get());
    } else {
        auto wrapped = std::make_unique<PreloadedPackReader>(std::move(gpk_reader));
        IPackReader* raw_inner = wrapped->inner();

        uint32_t stored_sz = raw_inner->sketch_sketch_size();
        uint32_t pre_sz = static_cast<uint32_t>(cfg.sketch_size);
        if (stored_sz > 0 && pre_sz > stored_sz) pre_sz = stored_sz;

        const std::vector<uint32_t> avail_ks = raw_inner->available_kmer_sizes();

        struct ArchBucket {
            std::vector<size_t> taxa_indices;
            std::vector<std::string> accs;
            size_t total_genomes = 0;
        };
        std::map<uint16_t, ArchBucket> per_arch;
        ArchBucket cross;

        for (size_t ti = 0; ti < taxa.size(); ++ti) {
            const auto& tx = taxa[ti];
            std::unordered_set<uint16_t> arcs;
            for (const auto& g : tx.genomes) {
                uint16_t a = raw_inner->archive_idx_for_accession(g.accession);
                if (a != UINT16_MAX) arcs.insert(a);
            }
            if (arcs.size() == 1) {
                auto& b = per_arch[*arcs.begin()];
                b.taxa_indices.push_back(ti);
                for (const auto& g : tx.genomes) b.accs.push_back(g.accession);
                b.total_genomes += tx.genomes.size();
            } else {
                for (const auto& g : tx.genomes) cross.accs.push_back(g.accession);
                cross.taxa_indices.push_back(ti);
                cross.total_genomes += tx.genomes.size();
            }
        }

        std::vector<std::pair<uint16_t, ArchBucket>> ordered(
            std::make_move_iterator(per_arch.begin()),
            std::make_move_iterator(per_arch.end()));
        std::sort(ordered.begin(), ordered.end(),
                  [](const auto& a, const auto& b){ return a.second.total_genomes > b.second.total_genomes; });

        spdlog::info("BUCKET: {} per-archive buckets ({} cross-archive taxa, {} genomes); ks=[{}] sz={}",
                     ordered.size(), cross.taxa_indices.size(), cross.total_genomes,
                     [&]{ std::string s; for (auto k : avail_ks) { if(!s.empty()) s+=','; s+=std::to_string(k);} return s; }(),
                     pre_sz);

        // Wave-budgeted bucket processing: split each per-archive bucket into
        // memory-budgeted waves so peak RSS stays bounded regardless of how
        // many genomes the bucket contains. Taxa are kept atomic — never
        // split across waves — so process_taxa_parallel sees complete taxa.
        //
        // Budget: GEODESIC_BUCKET_RAM_GB caps the resident sketch buffers per
        //         wave (default 64 GB; tune down on smaller nodes).
        const char* env_budget = std::getenv("GEODESIC_BUCKET_RAM_GB");
        const uint64_t budget_gb = (env_budget && *env_budget)
            ? std::strtoull(env_budget, nullptr, 10)
            : 64ull;
        // GEODESIC_BUCKET_RAM_MB overrides for fine-grained testing.
        const char* env_budget_mb = std::getenv("GEODESIC_BUCKET_RAM_MB");
        uint64_t budget_bytes = (env_budget_mb && *env_budget_mb)
            ? (std::strtoull(env_budget_mb, nullptr, 10) << 20)
            : (budget_gb << 30);
        if (budget_bytes < (1ull << 20)) budget_bytes = 1ull << 30;  // 1 GB floor on missing/zero
        const uint64_t mask_words   = (pre_sz + 63u) / 64u;
        // Budget for a single k (probe selects dominant k per wave; others fall through to disk).
        const uint64_t per_genome_bytes =
            static_cast<uint64_t>(pre_sz) * 2ull * sizeof(uint16_t)   // sigs+sig2s
            + mask_words * sizeof(uint64_t);                           // mask
        const size_t wave_max_genomes = std::max<size_t>(
            1, static_cast<size_t>(budget_bytes / std::max<uint64_t>(1, per_genome_bytes)));

        spdlog::info("BUCKET wave budget: {} MB → {} genomes/wave ({} bytes/genome, single-k preload)",
                     budget_bytes >> 20, wave_max_genomes, per_genome_bytes);

        for (auto& [arch, b] : ordered) {
            // Pack taxa into waves preserving original order (disk locality).
            struct Wave {
                std::vector<size_t> taxa_indices;
                std::vector<std::string> accs;
                size_t total_genomes = 0;
            };
            std::vector<Wave> waves;
            Wave cur;
            for (size_t idx : b.taxa_indices) {
                const auto& tx = taxa[idx];
                const size_t tx_sz = tx.genomes.size();
                if (!cur.taxa_indices.empty()
                    && cur.total_genomes + tx_sz > wave_max_genomes) {
                    waves.push_back(std::move(cur));
                    cur = Wave{};
                }
                cur.taxa_indices.push_back(idx);
                for (const auto& g : tx.genomes) cur.accs.push_back(g.accession);
                cur.total_genomes += tx_sz;
            }
            if (!cur.taxa_indices.empty()) waves.push_back(std::move(cur));

            spdlog::info("BUCKET arch={}: {} taxa, {} genomes → {} wave(s)",
                         arch, b.taxa_indices.size(), b.total_genomes, waves.size());

            for (size_t wi = 0; wi < waves.size(); ++wi) {
                auto& w = waves[wi];
                // Probe a small sample to pick the dominant k for this wave.
                // Preloading a single k keeps peak RSS ~3× lower than all ks.
                const uint32_t dominant_k = [&]() -> uint32_t {
                    if (avail_ks.size() < 2) return avail_ks.empty()
                        ? static_cast<uint32_t>(cfg.kmer_size) : avail_ks[0];
                    const int probed = derep::probe_taxon_kmer(
                        w.accs, *raw_inner, cfg.kmer_size, pre_sz);
                    return probed > 0 ? static_cast<uint32_t>(probed)
                                     : static_cast<uint32_t>(cfg.kmer_size);
                }();
                spdlog::info("BUCKET arch={} wave {}/{}: {} taxa, {} genomes — preloading k={}",
                             arch, wi + 1, waves.size(),
                             w.taxa_indices.size(), w.total_genomes, dominant_k);
                try {
                    wrapped->preload_multi(w.accs, {dominant_k}, pre_sz, cfg.threads);
                } catch (const std::exception& e) {
                    spdlog::warn("BUCKET preload failed: {} — processing without preload", e.what());
                }
                std::vector<Taxon> sub_taxa;
                sub_taxa.reserve(w.taxa_indices.size());
                for (size_t idx : w.taxa_indices) sub_taxa.push_back(taxa[idx]);
                process_taxa_parallel(sub_taxa, cfg, run_state, wrapped.get(),
                                      gunc_scores_ptr, grd_writer.get());
                wrapped->release_sketches();
            }
        }

        if (!cross.taxa_indices.empty()) {
            spdlog::info("BUCKET cross-archive: {} taxa, {} genomes — processing via inner reader (no preload)",
                         cross.taxa_indices.size(), cross.total_genomes);
            std::vector<Taxon> cross_taxa;
            cross_taxa.reserve(cross.taxa_indices.size());
            for (size_t idx : cross.taxa_indices) cross_taxa.push_back(taxa[idx]);
            process_taxa_parallel(cross_taxa, cfg, run_state, raw_inner,
                                  gunc_scores_ptr, grd_writer.get());
        }

        gpk_reader = std::move(wrapped);
    }

    // 11. Summary and output
    ResultsWriter writer(results_dir, cfg.prefix);
    writer.write_all(run_state);

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

    // Finalize GRD archive (write global sections + TOC + TailLocator)
    if (grd_writer) {
        try {
            grd_writer->close();
        } catch (const std::exception& e) {
            spdlog::warn("GRD finalize failed: {}", e.what());
        }
    }

    // Emit derep archive (.gpd) — best-effort; TSVs already written.
    emit_gpd_archive(cfg, run_state, gpk_reader.get());

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
