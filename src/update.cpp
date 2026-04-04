#include "pipeline.hpp"
#include "config.hpp"
#include "io/lock_writer.hpp"
#include "io/tsv_reader.hpp"
#include "io/results_writer.hpp"
#include "io/report_writer.hpp"
#include "db/geodf/geodf_reader.hpp"
#include "db/geodf/geodf_writer.hpp"
#include "state/run_state.hpp"
#include "core/types.hpp"
#include "core/pack_reader.hpp"
#include "core/multi_pack_reader.hpp"
#include <genopack/archive.hpp>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <chrono>

namespace derep {

namespace fs = std::filesystem;

int run_update(Config& cfg) {
    // 1. Setup logging (console-only, same pattern as run_sketch)
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

    // 2. Read prior lock file
    if (cfg.lock_input.empty()) {
        spdlog::error("--lock is required for 'geodesic update'");
        return 1;
    }
    LockData lock = read_lock_file(cfg.lock_input);
    spdlog::info("update: prior run had {} genomes, {} taxa (timestamp: {})",
                 lock.n_genomes, lock.n_taxa, lock.timestamp);

    // 3. Read new taxonomy TSV
    auto genome_rows = read_genomes_tsv(cfg.tax_file);
    spdlog::info("update: {} genomes in new taxonomy file", genome_rows.size());

    // 4. Identify prior accessions from GEODF
    std::unordered_set<std::string> prior_accessions;
    if (!lock.geodf_path.empty() && fs::exists(lock.geodf_path)) {
        geodf::GeodfReader prior_reader(lock.geodf_path);
        prior_reader.for_each_complete([&](const geodf::TaxonData& td) {
            for (const auto& acc : td.all_accessions)
                prior_accessions.insert(acc);
        });
    }
    spdlog::info("update: {} prior accessions from GEODF", prior_accessions.size());

    // 5. Build per-accession and per-taxon index from new taxonomy
    std::unordered_map<std::string, std::vector<size_t>> taxon_to_rows;
    for (size_t i = 0; i < genome_rows.size(); ++i)
        taxon_to_rows[genome_rows[i].taxonomy].push_back(i);

    // 6. Identify affected taxa (taxa containing at least one new genome)
    std::unordered_set<std::string> all_accessions;
    for (const auto& r : genome_rows)
        all_accessions.insert(r.accession);

    std::unordered_set<std::string> new_acc_set;
    for (const auto& acc : all_accessions)
        if (!prior_accessions.count(acc))
            new_acc_set.insert(acc);
    spdlog::info("update: {} new genomes (not in prior run)", new_acc_set.size());

    std::unordered_set<std::string> affected_taxa;
    for (const auto& r : genome_rows)
        if (new_acc_set.count(r.accession))
            affected_taxa.insert(r.taxonomy);
    spdlog::info("update: {} affected taxa to re-dereplicate", affected_taxa.size());

    // 7. Open new .gpk archive if provided (single or multi-pack)
    std::unique_ptr<IPackReader> gpk_reader;
    if (cfg.pack_dir.has_value()) {
        try {
            const auto& pack_path = *cfg.pack_dir;
            if (pack_path.extension() == ".gpk") {
                auto ar = std::make_unique<genopack::ArchiveReader>();
                ar->open(pack_path);
                gpk_reader = std::make_unique<SinglePackReader>(std::move(ar));
                spdlog::info("update: genopack single archive opened: {}", pack_path.string());
            } else {
                gpk_reader = MultiPackReader::open_dir(pack_path);
                spdlog::info("update: genopack multi-pack opened: {} archives",
                             static_cast<MultiPackReader*>(gpk_reader.get())->n_archives());
            }
        } catch (const std::exception& e) {
            spdlog::warn("update: failed to open genopack archive at {}: {} — proceeding without",
                         cfg.pack_dir->string(), e.what());
            gpk_reader.reset();
        }
    }

    // 8. Load unchanged taxa from prior GEODF into run_state
    RunState run_state;
    if (!lock.geodf_path.empty() && fs::exists(lock.geodf_path)) {
        geodf::GeodfReader prior_reader(lock.geodf_path);
        prior_reader.for_each_complete([&](const geodf::TaxonData& td) {
            if (affected_taxa.count(td.taxonomy))
                return;  // will be re-processed

            TaxonOutput to;
            to.result.taxonomy         = td.taxonomy;
            to.result.status           = TaxonStatus::SUCCESS;
            to.result.n_genomes        = static_cast<int>(td.genome_ids.size());
            to.result.n_representatives = static_cast<int>(td.rep_accessions.size());
            to.all_accessions          = td.all_accessions;
            to.representatives         = td.rep_accessions;
            run_state.push(std::move(to));
        });
        spdlog::info("update: {} unchanged taxa copied from prior GEODF",
                     run_state.taxa().size());
    }

    // 9. Build Taxon objects for affected taxa and re-dereplicate them
    std::vector<Taxon> affected_taxa_vec;
    for (const auto& [tax, row_indices] : taxon_to_rows) {
        if (!affected_taxa.count(tax))
            continue;
        Taxon t;
        t.taxonomy = tax;
        for (size_t idx : row_indices) {
            const auto& r = genome_rows[idx];
            Genome g;
            g.accession = r.accession;
            g.taxonomy  = r.taxonomy;
            g.file_path = r.file_path;
            t.genomes.push_back(std::move(g));
        }
        affected_taxa_vec.push_back(std::move(t));
    }
    spdlog::info("update: processing {} affected taxa", affected_taxa_vec.size());

    process_taxa_parallel(affected_taxa_vec, cfg, run_state,
                          gpk_reader.get(), nullptr /*gunc_scores*/);

    // 10. Write GEODF if requested
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

                    const std::unordered_set<std::string> rep_set(
                        taxon.representatives.begin(), taxon.representatives.end());
                    for (uint32_t i = 0; i < static_cast<uint32_t>(taxon.all_accessions.size()); ++i) {
                        const auto& acc = taxon.all_accessions[i];
                        tr.genome_ids.push_back(i);
                        tr.is_rep.push_back(rep_set.count(acc) > 0);
                        tr.contamination.push_back(0.0f);
                        tr.all_accessions.push_back(acc);
                    }

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
            spdlog::info("update: GEODF written to {}", cfg.geodf_output.string());
        } catch (const std::exception& e) {
            spdlog::warn("update: GEODF write failed: {}", e.what());
        }
    }

    // 11. Write TSV outputs
    if (!cfg.prefix.empty() && cfg.out_dir.has_value()) {
        fs::create_directories(*cfg.out_dir);
        ResultsWriter results_writer(*cfg.out_dir, cfg.prefix);
        results_writer.write_all(run_state, genome_rows);
        ReportWriter report_writer(*cfg.out_dir, cfg.prefix, cfg.timestamp);
        report_writer.write(run_state);
    }

    // 12. Write new lock file
    if (!cfg.lock_output.empty()) {
        try {
            LockData new_lock;
            new_lock.kmer_size     = cfg.kmer_size;
            new_lock.sketch_size   = cfg.sketch_size;
            new_lock.syncmer_s     = cfg.syncmer_s;
            new_lock.ani_threshold = cfg.ani_threshold;
            new_lock.params_hash   = geodf::hash_run_params(cfg.kmer_size, cfg.sketch_size,
                                                            cfg.syncmer_s, cfg.ani_threshold);
            if (cfg.pack_dir.has_value() && cfg.pack_dir->extension() == ".gpk") {
                new_lock.gpk_path        = *cfg.pack_dir;
                new_lock.gpk_snapshot_id = geodf::gpk_snapshot_hash(*cfg.pack_dir);
            }
            new_lock.geodf_path = cfg.geodf_output;
            if (!cfg.geodf_output.empty() && fs::exists(cfg.geodf_output))
                new_lock.geodf_hash = file_tail_hash(cfg.geodf_output);
            new_lock.n_taxa    = run_state.taxa().size();
            new_lock.n_genomes = run_state.total_genomes();
            new_lock.n_reps    = run_state.total_reps();
            {
                auto now = std::chrono::system_clock::now();
                auto tt  = std::chrono::system_clock::to_time_t(now);
                char buf[32];
                std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&tt));
                new_lock.timestamp = buf;
            }
            write_lock_file(cfg.lock_output, new_lock);
            spdlog::info("update: lock file written to {}", cfg.lock_output.string());
        } catch (const std::exception& e) {
            spdlog::warn("update: lock file write failed: {}", e.what());
        }
    }

    return run_state.total_failed() > 0 ? 1 : 0;
}

} // namespace derep
