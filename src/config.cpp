#include "config.hpp"
#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>
#include <ctime>
#include <stdexcept>
#include <thread>

namespace derep {

Config parse_args(int argc, char** argv) {
    Config cfg;

    CLI::App app{"geodesic: spherical genome embeddings for diverse representative selection"};

    app.add_option("-t,--tax-file", cfg.tax_file, "Taxonomy file (TSV: accession, taxonomy, file_path)")
        ->required()
        ->check(CLI::ExistingFile);

    app.add_option("-d,--db-path", cfg.db_path, "DuckDB database path")
        ->default_val("geodesic.db");

    app.add_option("--selected-taxa", cfg.selected_taxa_file, "File with selected taxa to process")
        ->check(CLI::ExistingFile);

    app.add_option("--checkm2", cfg.checkm2_file, "CheckM2 quality report file")
        ->check(CLI::ExistingFile);

    app.add_option("--fixed-taxa", cfg.fixed_taxa_file, "File with fixed representative assignments")
        ->check(CLI::ExistingFile);

    app.add_option("-o,--out-dir", cfg.out_dir, "Output directory for representative copies");

    app.add_option("-p,--prefix", cfg.prefix, "Prefix for output files");

    app.add_option("--tmp-dir", cfg.tmp_dir, "Temporary directory")
        ->default_val("/tmp");

    app.add_option("--threads", cfg.threads, "Total CPU threads to use")
        ->default_val(1);

    // -w/--workers: advanced override, hidden from normal help.
    // When set, total_budget = workers * threads (legacy behaviour).
    // When omitted, total_budget = threads (recommended).
    bool user_set_workers = false;
    app.add_option("-w,--workers", cfg.workers, "Workers (advanced: overrides total_budget = workers * threads)")
        ->default_val(1)
        ->group("")  // hide from --help
        ->each([&user_set_workers](const std::string&) { user_set_workers = true; });

    app.add_option("-z,--z-threshold", cfg.z_threshold, "Z-score threshold for filtering")
        ->default_val(2.0);

    app.add_option("--ani-threshold", cfg.ani_threshold, "ANI threshold for clustering")
        ->default_val(95.0);

    // GEODESIC params
    app.add_flag("--geodesic-auto-calibrate,!--geodesic-no-auto-calibrate",
        cfg.auto_calibrate,
        "Auto-calibrate params from ANI sample (default: on)")->default_val(true);
    app.add_option("--geodesic-calibration-pairs", cfg.calibration_pairs,
        "Number of pairs to sample for auto-calibration")->default_val(50);
    app.add_option("--geodesic-dim", cfg.embedding_dim,
        "GEODESIC embedding dimension (higher = more accuracy)")->default_val(256);
    app.add_option("--geodesic-kmer-size", cfg.kmer_size,
        "GEODESIC k-mer size (larger = more discriminative at high ANI)")->default_val(21);
    app.add_option("--geodesic-sketch-size", cfg.sketch_size,
        "GEODESIC sketch size (larger = more accurate Jaccard)")->default_val(10000);
    app.add_option("--geodesic-diversity-threshold", cfg.diversity_threshold,
        "Min embedding distance gain to add representative (lower = more reps)")->default_val(0.02f);
    app.add_option("--geodesic-max-rep-fraction", cfg.max_rep_fraction,
        "Max fraction of genomes as representatives")->default_val(0.2f);

    // Incremental embedding store (DuckDB-VSS)
    app.add_option("--embedding-db", cfg.embedding_db,
        "Path to persistent embedding store (DuckDB). Enables incremental updates.");
    app.add_flag("--incremental", cfg.incremental,
        "Enable incremental mode: reuse existing embeddings, only embed new genomes");

    app.add_flag("--copy-reps", cfg.copy_reps, "Copy representative genomes to output directory");
    app.add_flag("-v,--verbose", [&cfg](int64_t count) { cfg.verbosity = 2; },
        "Verbose output (show per-genome progress)");
    app.add_flag("-q,--quiet", [&cfg](int64_t count) { cfg.verbosity = 0; },
        "Quiet output (only errors and summary)");
    app.add_flag("--debug", cfg.debug, "Enable debug logging (sets verbosity=3)");
    app.add_flag("--keep-intermediates", cfg.keep_intermediates, "Keep intermediate files");

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        std::exit(app.exit(e));
    }

    if (cfg.copy_reps && !cfg.out_dir)
        throw std::runtime_error("--copy-reps requires --out-dir");
    if (cfg.out_dir && !cfg.copy_reps)
        throw std::runtime_error("--out-dir requires --copy-reps");

    auto now = std::time(nullptr);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", std::localtime(&now));
    cfg.timestamp = buf;

    // When -w not set: workers=1 so total_budget = 1 * threads = threads (total cores).
    // When -w set: total_budget = workers * threads (legacy/advanced).
    if (!user_set_workers)
        cfg.workers = 1;

    return cfg;
}

} // namespace derep
