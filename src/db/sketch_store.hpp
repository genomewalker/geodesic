#pragma once
#include <duckdb.hpp>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace derep::db {

// SketchStore: global, persistent OPH sketch cache.
// Stores oph_sig (seed1) and lazy oph_sig2 (seed2) for every genome.
// Lives in a separate DuckDB file on fast local storage (/scratch).
// Thread-safe: one connection per thread (same pattern as DBManager).
// Writer thread should use a single connection and commit in batches.
class SketchStore {
public:
    // Parameters that identify the sketch format.
    // Opening an existing store with mismatched params throws.
    struct Meta {
        int format_version = 1;
        int kmer_size = 21;
        int sketch_size = 10000;
        int syncmer_s = 0;
        uint64_t seed1 = 42;
        uint64_t seed2 = 1337;
    };

    struct Config {
        size_t insert_batch_rows = 4096;       // genomes per transaction during sketch pass
        size_t checkpoint_every_batches = 16;  // WAL checkpoint cadence
        size_t fetch_temp_table_threshold = 0; // unused: always use temp-table join
    };

    struct SketchRecord {
        std::string accession;
        std::string taxonomy;
        std::vector<uint16_t> oph_sig;   // empty = not present
        std::vector<uint16_t> oph_sig2;  // empty = not materialized
        uint32_t n_real_bins = 0;
        uint64_t genome_length = 0;
    };

    struct SketchFailure {
        std::string accession;
        std::string taxonomy;
        std::string file_path;
        std::string error_message;
    };

    SketchStore() : SketchStore(Config{}) {}
    explicit SketchStore(Config cfg);
    ~SketchStore();
    SketchStore(const SketchStore&) = delete;
    SketchStore& operator=(const SketchStore&) = delete;

    // Open (or create) sketch DB at path; validate/init meta. Throws on mismatch.
    void open(const std::filesystem::path& db_path, const Meta& expected);
    void close();

    // Load all completed/failed accessions into memory (called once at startup).
    std::unordered_set<std::string> load_completed_accessions();
    std::unordered_set<std::string> load_failed_accessions();

    // Insert a batch of sketches. Must be called from a single writer thread.
    // Commits every cfg_.insert_batch_rows rows; caller controls when to flush.
    void insert_batch(const std::vector<SketchRecord>& batch);

    // Upsert sig2 for genomes that had it lazily computed.
    void upsert_sig2(const std::vector<std::pair<std::string, std::vector<uint16_t>>>& pairs);

    // Record failed genomes (separate table, not mixed with successes).
    void record_failures(const std::vector<SketchFailure>& failures);

    // Fetch sketches in caller-specified order. Missing accessions get an empty record.
    // Uses a temp-table join with an ord column to preserve order and detect misses.
    // Thread-safe: uses per-thread connection.
    std::vector<SketchRecord> fetch_ordered(const std::vector<std::string>& accessions);

    // Fetch sig2 for a set of accessions. Missing entries omitted from result.
    std::unordered_map<std::string, std::vector<uint16_t>>
    fetch_sig2(const std::vector<std::string>& accessions);

    // Flush WAL to main file. Call periodically and at shutdown.
    void checkpoint();

    bool is_open() const { return database_ != nullptr; }

private:
    void create_schema();
    void validate_or_init_meta(const Meta& expected);
    duckdb::Connection& thread_connection();

    Config cfg_;
    std::filesystem::path db_path_;
    std::unique_ptr<duckdb::DuckDB> database_;

    // Per-thread connection pool
    struct ThreadIdHash {
        std::size_t operator()(const std::thread::id& id) const noexcept {
            return std::hash<std::thread::id>{}(id);
        }
    };
    mutable std::mutex pool_mutex_;
    std::unordered_map<std::thread::id,
                       std::unique_ptr<duckdb::Connection>,
                       ThreadIdHash> pool_;

    // Batch counter for checkpoint cadence
    size_t batches_since_checkpoint_ = 0;
};

} // namespace derep::db
