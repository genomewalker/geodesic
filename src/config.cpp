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
    app.require_subcommand(1);

    // ── derep subcommand ────────────────────────────────────────────────────
    auto* derep = app.add_subcommand("derep", "Run dereplication pipeline");

    derep->add_option("-t,--tax-file", cfg.tax_file, "Taxonomy file (TSV: accession, taxonomy, file_path)")
        ->required()
        ->check(CLI::ExistingFile);

    derep->add_option("-d,--db-path", cfg.db_path, "DuckDB database path")
        ->default_val("geodesic.db");

    derep->add_option("--selected-taxa", cfg.selected_taxa_file, "File with selected taxa to process")
        ->check(CLI::ExistingFile);

    derep->add_option("--checkm2", cfg.checkm2_file, "CheckM2 quality report file")
        ->check(CLI::ExistingFile);

    derep->add_option("--gunc-scores", cfg.gunc_file,
        "GUNC output TSV (genome_id, pass.GUNC, clade_separation_score, ...)")
        ->check(CLI::ExistingFile);

    derep->add_option("--fixed-taxa", cfg.fixed_taxa_file, "File with fixed representative assignments")
        ->check(CLI::ExistingFile);

    derep->add_option("--fixed-reps", cfg.fixed_reps_file,
        "File with accessions (one per line) to always include as representatives")
        ->check(CLI::ExistingFile);

    derep->add_option("-o,--out-dir", cfg.out_dir, "Output directory for representative copies");

    derep->add_option("-p,--prefix", cfg.prefix, "Prefix for output files");

    derep->add_option("--tmp-dir", cfg.tmp_dir, "Temporary directory")
        ->default_val("/tmp");

    derep->add_option("--threads", cfg.threads, "Total CPU threads to use")
        ->default_val(1);

    derep->add_option("--io-threads", cfg.io_threads,
        "Max concurrent NFS file readers during genome embedding (0=auto: threads)")
        ->default_val(0)
        ->group("");

    bool user_set_workers = false;
    derep->add_option("-w,--workers", cfg.workers, "Workers (advanced: overrides total_budget = workers * threads)")
        ->default_val(1)
        ->group("")
        ->each([&user_set_workers](const std::string&) { user_set_workers = true; });

    derep->add_option("-z,--z-threshold", cfg.z_threshold, "Z-score threshold for filtering")
        ->default_val(2.0);

    derep->add_option("--ani-threshold", cfg.ani_threshold, "ANI threshold for clustering")
        ->default_val(95.0);

    derep->add_flag("--geodesic-auto-calibrate,!--geodesic-no-auto-calibrate",
        cfg.auto_calibrate,
        "Auto-calibrate params from ANI sample (default: on)")->default_val(true);
    derep->add_option("--geodesic-calibration-pairs", cfg.calibration_pairs,
        "Number of pairs to sample for auto-calibration")->default_val(50);
    derep->add_option("--geodesic-dim", cfg.embedding_dim,
        "GEODESIC embedding dimension (higher = more accuracy)")->default_val(256);
    derep->add_option("--geodesic-kmer-size", cfg.kmer_size,
        "GEODESIC k-mer size (larger = more discriminative at high ANI)")->default_val(21);
    derep->add_option("--geodesic-sketch-size", cfg.sketch_size,
        "GEODESIC sketch size (larger = more accurate Jaccard)")->default_val(10000);
    derep->add_option("--geodesic-syncmer-s", cfg.syncmer_s,
        "GEODESIC open-syncmer submer length (0=disabled, smaller=faster/sparser OPH)")
        ->default_val(0);
    derep->add_option("--geodesic-diversity-threshold", cfg.diversity_threshold,
        "Min embedding distance gain to add representative (lower = more reps)")->default_val(0.02f);
    derep->add_option("--geodesic-max-rep-fraction", cfg.max_rep_fraction,
        "Max fraction of genomes as representatives")->default_val(0.2f);
    derep->add_option("--k-cap-max", cfg.k_cap_max,
        "Max K_cap for adaptive retry on disconnected k-NN graphs")->default_val(256);
    derep->add_option("--nystrom-diagonal-loading", cfg.nystrom_diagonal_loading,
        "Tikhonov regularization fraction for Nyström Gram matrix diagonal (default: 0.01)")
        ->default_val(0.01);
    derep->add_flag("--no-nystrom-degree-normalize{false},--nystrom-degree-normalize{true}",
        cfg.nystrom_degree_normalize,
        "Symmetric Laplacian normalization of Nyström Gram matrix (default: on)");
    derep->add_option("--embedding-db", cfg.embedding_db,
        "Path to persistent embedding store (DuckDB). Enables incremental updates.");
    derep->add_flag("--incremental", cfg.incremental,
        "Enable incremental mode: reuse existing embeddings, only embed new genomes");

    derep->add_option("--sketch-db", cfg.sketch_db,
        "Path to sketch cache DuckDB (built by 'geodesic sketch'). Skips NFS re-reads.");
    derep->add_flag("--require-sketches", cfg.require_sketches,
        "Fail if any genome is missing from sketch cache (no NFS fallback)");
    derep->add_option("--pack", cfg.pack_dir,
        "Path to genome pack directory (built by 'geodesic pack'). Reads sequences from local pack instead of NFS.");

    derep->add_flag("--copy-reps", cfg.copy_reps, "Copy representative genomes to output directory");
    derep->add_flag("-v,--verbose", [&cfg](int64_t) { cfg.verbosity = 2; },
        "Verbose output (show per-genome progress)");
    derep->add_flag("-q,--quiet", [&cfg](int64_t) { cfg.verbosity = 0; },
        "Quiet output (only errors and summary)");
    derep->add_flag("--debug", cfg.debug, "Enable debug logging (sets verbosity=3)");
    derep->add_flag("--keep-intermediates", cfg.keep_intermediates, "Keep intermediate files");

    // ── sketch subcommand ───────────────────────────────────────────────────
    auto* sketch_cmd = app.add_subcommand("sketch",
        "Pre-compute OPH sketches for all genomes and cache to DuckDB on /scratch");

    sketch_cmd->add_option("-t,--tax-file", cfg.tax_file, "Taxonomy file (TSV: accession, taxonomy, file_path)")
        ->required()
        ->check(CLI::ExistingFile);

    sketch_cmd->add_option("-s,--sketch-db", cfg.sketch_db,
        "Output sketch cache DuckDB path (on fast local storage, e.g. /scratch/...)")
        ->required();

    sketch_cmd->add_option("--threads", cfg.threads, "Total CPU threads to use")
        ->default_val(1);

    sketch_cmd->add_option("--io-threads", cfg.io_threads,
        "Max concurrent NFS file readers (0=auto)")
        ->default_val(0);

    sketch_cmd->add_option("--geodesic-kmer-size", cfg.kmer_size,
        "k-mer size (must match derep run)")->default_val(21);
    sketch_cmd->add_option("--geodesic-sketch-size", cfg.sketch_size,
        "Sketch size / OPH bins (must match derep run)")->default_val(10000);
    sketch_cmd->add_option("--geodesic-syncmer-s", cfg.syncmer_s,
        "Open-syncmer submer length (0=disabled, must match derep run)")->default_val(0);
    sketch_cmd->add_option("--pack", cfg.pack_dir,
        "Path to genome pack directory. Reads sequences from local pack instead of NFS.");

    sketch_cmd->add_flag("-v,--verbose", [&cfg](int64_t) { cfg.verbosity = 2; },
        "Verbose output");
    sketch_cmd->add_flag("-q,--quiet", [&cfg](int64_t) { cfg.verbosity = 0; },
        "Quiet output");

    // ── pack subcommand ─────────────────────────────────────────────────────
    auto* pack_cmd = app.add_subcommand("pack",
        "Pre-pack all genome FASTA files into taxonomy-indexed zstd archives for fast local access");

    pack_cmd->add_option("-t,--tax-file", cfg.tax_file, "Taxonomy file (TSV: accession, taxonomy, file_path)")
        ->required()
        ->check(CLI::ExistingFile);

    pack_cmd->add_option("-o,--out", cfg.pack_dir,
        "Output directory for genome pack (e.g. /projects/caeg/scratch/kbd606/gtdb-pack/)")
        ->required();

    pack_cmd->add_option("--threads", cfg.threads, "Total CPU threads")->default_val(4);
    pack_cmd->add_option("--io-threads", cfg.io_threads,
        "Max concurrent NFS file readers (0=auto: threads)")->default_val(0);
    pack_cmd->add_option("--zstd-level", cfg.pack_zstd_level,
        "zstd compression level (1-22, higher=smaller/slower)")->default_val(15);
    pack_cmd->add_flag("-v,--verbose", [&cfg](int64_t) { cfg.verbosity = 2; }, "Verbose output");
    pack_cmd->add_flag("-q,--quiet",   [&cfg](int64_t) { cfg.verbosity = 0; }, "Quiet output");

    // ── report subcommand ───────────────────────────────────────────────────
    auto* report_cmd = app.add_subcommand("report", "Generate HTML report from existing database");

    report_cmd->add_option("-d,--db-path", cfg.db_path, "DuckDB database path")
        ->required()
        ->check(CLI::ExistingFile);

    report_cmd->add_option("-p,--prefix", cfg.prefix, "Prefix used when running derep")
        ->required();

    report_cmd->add_option("-o,--out-dir", cfg.out_dir, "Output directory for report files");

    // ── parse ───────────────────────────────────────────────────────────────
    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        std::exit(app.exit(e));
    }

    if (report_cmd->parsed()) {
        cfg.command = Command::Report;
        cfg.report_only = true;
    } else if (sketch_cmd->parsed()) {
        cfg.command = Command::Sketch;
    } else if (pack_cmd->parsed()) {
        cfg.command = Command::Pack;
    } else {
        cfg.command = Command::Derep;
    }

    auto now = std::time(nullptr);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", std::localtime(&now));
    cfg.timestamp = buf;

    if (cfg.command == Command::Derep) {
        if (cfg.copy_reps && !cfg.out_dir)
            throw std::runtime_error("--copy-reps requires --out-dir");
        if (cfg.out_dir && !cfg.copy_reps)
            throw std::runtime_error("--out-dir requires --copy-reps");
        if (!user_set_workers)
            cfg.workers = 1;
    }

    return cfg;
}

} // namespace derep
