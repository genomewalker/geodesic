#pragma once
#include <filesystem>
#include <optional>
#include <string>

namespace derep {

struct Config {
    // I/O
    std::filesystem::path tax_file;
    std::filesystem::path db_path{"geodesic.db"};
    std::optional<std::filesystem::path> selected_taxa_file;
    std::optional<std::filesystem::path> checkm2_file;
    std::optional<std::filesystem::path> gunc_file;
    std::optional<std::filesystem::path> fixed_taxa_file;
    std::optional<std::filesystem::path> fixed_reps_file;
    std::optional<std::filesystem::path> out_dir;
    std::string prefix;
    std::filesystem::path tmp_dir{"/tmp"};

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
    int embedding_dim = 512;
    int kmer_size = 21;
    int sketch_size = 10000;
    int syncmer_s = 0;
    float diversity_threshold = 0.02f;
    float max_rep_fraction = 0.2f;
    int k_cap_max = 256;  // Max K_cap for adaptive retry on disconnected k-NN
    float nystrom_diagonal_loading = 0.01f;
    bool nystrom_degree_normalize = true;

    // Incremental embedding store (DuckDB-VSS)
    std::optional<std::filesystem::path> embedding_db;
    bool incremental = false;

    // Flags
    bool copy_reps = false;
    bool debug = false;
    bool keep_intermediates = false;
    bool report_only = false;

    // Logging verbosity: 0=quiet, 1=normal (default), 2=verbose, 3=debug
    int verbosity = 1;
};

Config parse_args(int argc, char** argv);

} // namespace derep
