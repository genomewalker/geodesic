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

    derep->add_option("--tmp-dir", cfg.tmp_dir, "Temporary directory (default: current directory)")
        ->default_val(".");

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
    derep->add_option("--seed", cfg.seed,
        "Master RNG seed (HNSW, sketching, Nyström anchors, diversity sampling)")->default_val(42);
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
    derep->add_option("--ncbi-taxdump", cfg.ncbi_taxdump_dir,
        "Directory for NCBI taxdump (nodes.dmp + names.dmp). "
        "Downloaded automatically if absent or older than 30 days. "
        "Used for Eukaryote/Virus 10-rank taxonomy resolution.");

    derep->add_option("--pack", cfg.pack_dir,
        "Path to genopack archive (.gpk). Reads sequences from local pack instead of NFS.");

    derep->add_option("--geodf-output", cfg.geodf_output,
        "Path to write GEODF results file (binary format; empty = disabled)");

    derep->add_option("--grd-output", cfg.grd_output,
        "Path to write GRD results archive with per-genome embeddings for visualization");

    derep->add_option("--lock-output", cfg.lock_output,
        "Write a geodesic.lock provenance file (JSON) alongside the run outputs");

    derep->add_flag("--copy-reps", cfg.copy_reps, "Copy representative genomes to output directory");
    derep->add_flag("-v,--verbose", [&cfg](int64_t) { cfg.verbosity = 2; },
        "Verbose output (show per-genome progress)");
    derep->add_flag("-q,--quiet", [&cfg](int64_t) { cfg.verbosity = 0; },
        "Quiet output (only errors and summary)");
    derep->add_flag("--debug", cfg.debug, "Enable debug logging (sets verbosity=3)");
    derep->add_flag("--keep-intermediates", cfg.keep_intermediates, "Keep intermediate files");

    // ── update subcommand ───────────────────────────────────────────────────
    auto* update_cmd = app.add_subcommand("update",
        "Incrementally re-dereplicate: sketch new genomes, re-run only affected taxa");

    update_cmd->add_option("-t,--tax-file", cfg.tax_file,
        "New taxonomy file (TSV: accession, taxonomy, file_path)")
        ->required()
        ->check(CLI::ExistingFile);

    update_cmd->add_option("--lock", cfg.lock_input,
        "Path to prior run's lock file (geodesic.lock)")
        ->required()
        ->check(CLI::ExistingFile);

    update_cmd->add_option("--pack", cfg.pack_dir,
        "Path to genome pack (.gpk archive). Reads sequences from local pack instead of NFS.");

    update_cmd->add_option("--geodf-output", cfg.geodf_output,
        "Path to write updated GEODF results file");

    update_cmd->add_option("--lock-output", cfg.lock_output,
        "Write updated geodesic.lock provenance file");

    update_cmd->add_option("--threads", cfg.threads, "Total CPU threads to use")
        ->default_val(1);

    bool update_user_set_workers = false;
    update_cmd->add_option("-w,--workers", cfg.workers,
        "Workers (advanced: overrides total_budget = workers * threads)")
        ->default_val(1)
        ->group("")
        ->each([&update_user_set_workers](const std::string&) { update_user_set_workers = true; });

    update_cmd->add_option("--ani-threshold", cfg.ani_threshold, "ANI threshold for clustering")
        ->default_val(95.0);

    update_cmd->add_option("--geodesic-kmer-size", cfg.kmer_size,
        "k-mer size (must match prior run)")->default_val(21);
    update_cmd->add_option("--geodesic-sketch-size", cfg.sketch_size,
        "Sketch size (must match prior run)")->default_val(10000);
    update_cmd->add_option("--geodesic-syncmer-s", cfg.syncmer_s,
        "Open-syncmer submer length (0=disabled, must match prior run)")->default_val(0);

    update_cmd->add_flag("-v,--verbose", [&cfg](int64_t) { cfg.verbosity = 2; },
        "Verbose output");
    update_cmd->add_flag("-q,--quiet", [&cfg](int64_t) { cfg.verbosity = 0; },
        "Quiet output");

    // ── scatter subcommand ─────────────────────────────────────────────────
    auto* scatter_cmd = app.add_subcommand("scatter",
        "Partition input TSV for distributed execution across nodes");

    scatter_cmd->add_option("-t,--tax-file", cfg.tax_file,
        "Input taxonomy file (TSV: accession, taxonomy, file_path)")
        ->required()
        ->check(CLI::ExistingFile);

    scatter_cmd->add_option("-n,--partitions", cfg.n_partitions,
        "Number of partitions (typically = number of worker nodes)")
        ->required();

    scatter_cmd->add_option("-o,--output-dir", cfg.scatter_dir,
        "Output directory for partition files and worker script")
        ->required();

    scatter_cmd->add_option("--rank", cfg.partition_rank,
        "Taxonomy rank for grouping (g=genus, f=family, s=species)")
        ->default_val("g");

    scatter_cmd->add_option("--pack", cfg.pack_dir,
        "Path to genopack archive (passed through to worker commands)");

    scatter_cmd->add_option("--tmp-dir", cfg.tmp_dir,
        "Temporary directory for workers (default: scatter output dir)")
        ->default_val("");

    scatter_cmd->add_option("--threads", cfg.threads, "Threads per worker")
        ->default_val(4);

    // ── gather subcommand ──────────────────────────────────────────────────
    auto* gather_cmd = app.add_subcommand("gather",
        "Merge distributed shard results (GRD + TSV) into unified output");

    gather_cmd->add_option("-d,--shard-dir", cfg.gather_dir,
        "Directory containing shard results (from scatter workers)")
        ->required()
        ->check(CLI::ExistingDirectory);

    gather_cmd->add_option("-o,--output", cfg.gather_output,
        "Output path for merged GRD file")
        ->required();

    gather_cmd->add_option("-p,--prefix", cfg.prefix,
        "Prefix for merged TSV output files")
        ->default_val("merged");

    // ── parse ───────────────────────────────────────────────────────────────
    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        std::exit(app.exit(e));
    }

    if (update_cmd->parsed()) {
        cfg.command = Command::Update;
        if (!update_user_set_workers)
            cfg.workers = 1;
    } else if (scatter_cmd->parsed()) {
        cfg.command = Command::Scatter;
        if (cfg.tmp_dir.empty())
            cfg.tmp_dir = cfg.scatter_dir / "tmp";
    } else if (gather_cmd->parsed()) {
        cfg.command = Command::Gather;
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
