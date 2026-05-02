#pragma once
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace derep {

enum class Command { Derep, Update, Scatter, Gather };

struct Config {
    Command command = Command::Derep;

    // I/O
    std::filesystem::path genomes_file;     // --genomes: single-column accession list
    std::optional<std::filesystem::path> references_file;  // --references: pin as reps
    std::optional<std::filesystem::path> checkm2_file;
    std::optional<std::filesystem::path> gunc_file;
    std::optional<std::filesystem::path> fixed_taxa_file;
    std::optional<std::filesystem::path> out_dir;
    std::string prefix;
    std::filesystem::path tmp_dir{"."};

    // Runtime paths (set during setup)
    std::filesystem::path results_dir;
    std::filesystem::path temp_dir;
    std::filesystem::path log_file;
    std::string timestamp;

    // Parallelism
    int workers = 2;
    int threads = 4;
    // Max concurrent NFS file readers (0 = auto: threads)
    int io_threads = 0;

    // Thresholds
    double z_threshold = 2.0;
    double ani_threshold = 95.0;

    // GEODESIC params (geodesic_ prefix dropped)
    bool auto_calibrate = true;
    int calibration_pairs = 50;
    uint64_t seed = 42;  // Master RNG seed; all sub-seeds derived from this
    int embedding_dim = 256;
    int kmer_size = 21;
    int sketch_size = 10000;
    int syncmer_s = 0;
    float diversity_threshold = 0.02f;
    float max_rep_fraction = 0.2f;
    int k_cap_max = 256;  // Max K_cap for adaptive retry on disconnected k-NN
    float nystrom_diagonal_loading = 0.01f;
    bool nystrom_degree_normalize = true;

    // NCBI taxdump directory for Eukaryote/Virus taxonomy resolution.
    // If set and the dump is absent/stale, it will be downloaded automatically.
    // If not set, non-prokaryote genomes are normalized from their input taxonomy string.
    std::optional<std::filesystem::path> ncbi_taxdump_dir;

    // Genome pack (.gpk archive built by genopack)
    std::optional<std::filesystem::path> pack_dir;

    // GEODF output (optional; empty = disabled)
    std::filesystem::path geodf_output;

    // GRD output — geodesic results data archive with per-genome embeddings (optional)
    std::filesystem::path grd_output;

    // GPD output — derep archive (.gpd) with per-genome cluster/embedding data (optional)
    std::filesystem::path gpd_output;

    // Lock file input (for 'geodesic update' — path to prior run's lock file)
    std::filesystem::path lock_input;

    // Lock file output (optional; empty = disabled)
    std::filesystem::path lock_output;

    // Flags
    bool copy_reps = false;
    bool debug = false;
    bool keep_intermediates = false;

    // Logging verbosity: 0=quiet, 1=normal (default), 2=verbose, 3=debug
    int verbosity = 1;

    // Scatter/Gather (distributed mode)
    int n_partitions = 0;                    // scatter: number of partitions
    std::string partition_rank = "g";        // scatter: taxonomy rank for grouping
    std::filesystem::path scatter_dir;       // scatter: output directory for partitions
    std::filesystem::path gather_dir;        // gather: directory containing shard results
    std::filesystem::path gather_output;     // gather: merged output path
};

Config parse_args(int argc, char** argv);

} // namespace derep
