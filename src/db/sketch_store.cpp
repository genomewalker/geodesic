#include "db/sketch_store.hpp"
#include <duckdb.hpp>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <sstream>
#include <cstring>

namespace derep::db {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static void check(duckdb::unique_ptr<duckdb::MaterializedQueryResult>& res, const char* ctx) {
    if (res->HasError())
        throw std::runtime_error(std::string(ctx) + ": " + res->GetError());
}

static std::vector<uint16_t> blob_to_sig(const duckdb::Value& v) {
    if (v.IsNull()) return {};
    auto blob = v.GetValueUnsafe<duckdb::string_t>();
    const auto* ptr = reinterpret_cast<const uint16_t*>(blob.GetDataUnsafe());
    size_t n = blob.GetSize() / sizeof(uint16_t);
    return {ptr, ptr + n};
}

static duckdb::Value sig_to_blob(const std::vector<uint16_t>& sig) {
    if (sig.empty()) return duckdb::Value(duckdb::LogicalType::BLOB);
    std::string bytes(reinterpret_cast<const char*>(sig.data()),
                      sig.size() * sizeof(uint16_t));
    return duckdb::Value::BLOB_RAW(bytes);
}

// ---------------------------------------------------------------------------
// SketchStore
// ---------------------------------------------------------------------------
SketchStore::SketchStore(Config cfg) : cfg_(cfg) {}

SketchStore::~SketchStore() {
    close();
}

void SketchStore::open(const std::filesystem::path& db_path, const Meta& expected) {
    db_path_ = db_path;
    duckdb::DBConfig dcfg;
    dcfg.options.maximum_threads = 2;
    database_ = std::make_unique<duckdb::DuckDB>(db_path.string(), &dcfg);
    create_schema();
    validate_or_init_meta(expected);
    spdlog::info("SketchStore opened: {}", db_path.string());
}

void SketchStore::close() {
    if (!database_) return;
    {
        std::lock_guard lock(pool_mutex_);
        pool_.clear();
    }
    database_.reset();
}

duckdb::Connection& SketchStore::thread_connection() {
    auto tid = std::this_thread::get_id();
    std::lock_guard lock(pool_mutex_);
    auto [it, inserted] = pool_.emplace(tid, nullptr);
    if (inserted)
        it->second = std::make_unique<duckdb::Connection>(*database_);
    return *it->second;
}

void SketchStore::create_schema() {
    auto conn = std::make_unique<duckdb::Connection>(*database_);

    auto r0 = conn->Query(R"(
        CREATE TABLE IF NOT EXISTS sketch_store_meta (
            key   VARCHAR PRIMARY KEY,
            value VARCHAR NOT NULL
        )
    )");
    check(r0, "create sketch_store_meta");

    auto r1 = conn->Query(R"(
        CREATE TABLE IF NOT EXISTS sketches (
            accession     VARCHAR PRIMARY KEY,
            taxonomy      VARCHAR,
            oph_sig       BLOB    NOT NULL,
            oph_sig2      BLOB,
            n_real_bins   UINTEGER NOT NULL,
            genome_length UBIGINT  NOT NULL
        )
    )");
    check(r1, "create sketches");

    auto r2 = conn->Query(
        "CREATE INDEX IF NOT EXISTS idx_sketches_taxonomy ON sketches(taxonomy)");
    check(r2, "create taxonomy index");

    auto r3 = conn->Query(R"(
        CREATE TABLE IF NOT EXISTS sketch_failures (
            accession     VARCHAR PRIMARY KEY,
            taxonomy      VARCHAR,
            file_path     VARCHAR NOT NULL,
            error_message VARCHAR NOT NULL,
            attempts      UINTEGER NOT NULL
        )
    )");
    check(r3, "create sketch_failures");
}

void SketchStore::validate_or_init_meta(const Meta& expected) {
    auto conn = std::make_unique<duckdb::Connection>(*database_);

    // Check if any rows exist
    auto res = conn->Query("SELECT COUNT(*) FROM sketch_store_meta");
    check(res, "meta count");
    auto chunk = res->Fetch();
    int64_t count = chunk ? chunk->GetValue(0, 0).GetValue<int64_t>() : 0;

    auto get_meta = [&](const std::string& key) -> std::string {
        auto r = conn->Query("SELECT value FROM sketch_store_meta WHERE key = '" + key + "'");
        check(r, "meta fetch");
        auto c = r->Fetch();
        if (!c || c->size() == 0) return "";
        return c->GetValue(0, 0).GetValue<std::string>();
    };

    auto set_meta = [&](const std::string& key, const std::string& value) {
        auto r = conn->Query(
            "INSERT OR REPLACE INTO sketch_store_meta (key, value) VALUES ('" +
            key + "', '" + value + "')");
        check(r, "meta insert");
    };

    if (count == 0) {
        // Fresh store — write metadata
        set_meta("format_version", std::to_string(expected.format_version));
        set_meta("kmer_size",      std::to_string(expected.kmer_size));
        set_meta("sketch_size",    std::to_string(expected.sketch_size));
        set_meta("syncmer_s",      std::to_string(expected.syncmer_s));
        set_meta("seed1",          std::to_string(expected.seed1));
        set_meta("seed2",          std::to_string(expected.seed2));
        return;
    }

    // Validate existing metadata
    auto check_int = [&](const std::string& key, int expected_val) {
        std::string val = get_meta(key);
        if (val.empty() || std::stoi(val) != expected_val) {
            throw std::runtime_error(
                "SketchStore meta mismatch for '" + key +
                "': stored=" + (val.empty() ? "<missing>" : val) +
                " expected=" + std::to_string(expected_val) +
                ". Use a different --sketch-db path or rebuild the cache.");
        }
    };

    check_int("format_version", expected.format_version);
    check_int("kmer_size",      expected.kmer_size);
    check_int("sketch_size",    expected.sketch_size);
    check_int("syncmer_s",      expected.syncmer_s);
}

std::unordered_set<std::string> SketchStore::load_completed_accessions() {
    auto conn = std::make_unique<duckdb::Connection>(*database_);
    auto res = conn->Query("SELECT accession FROM sketches");
    check(res, "load_completed");
    std::unordered_set<std::string> out;
    for (;;) {
        auto chunk = res->Fetch();
        if (!chunk || chunk->size() == 0) break;
        for (size_t row = 0; row < chunk->size(); ++row)
            out.insert(chunk->GetValue(0, row).GetValue<std::string>());
    }
    spdlog::info("SketchStore: {} genomes already sketched", out.size());
    return out;
}

std::unordered_set<std::string> SketchStore::load_failed_accessions() {
    auto conn = std::make_unique<duckdb::Connection>(*database_);
    auto res = conn->Query("SELECT accession FROM sketch_failures");
    check(res, "load_failed");
    std::unordered_set<std::string> out;
    for (;;) {
        auto chunk = res->Fetch();
        if (!chunk || chunk->size() == 0) break;
        for (size_t row = 0; row < chunk->size(); ++row)
            out.insert(chunk->GetValue(0, row).GetValue<std::string>());
    }
    if (!out.empty())
        spdlog::info("SketchStore: {} previously failed genomes", out.size());
    return out;
}

void SketchStore::insert_batch(const std::vector<SketchRecord>& batch) {
    if (batch.empty()) return;

    auto& conn = thread_connection();
    auto rb = conn.Query("BEGIN TRANSACTION");
    check(rb, "insert_batch BEGIN");

    try {
        duckdb::Appender app(conn, "sketches");
        for (const auto& rec : batch) {
            app.BeginRow();
            app.Append(duckdb::Value(rec.accession));
            app.Append(rec.taxonomy.empty() ? duckdb::Value() : duckdb::Value(rec.taxonomy));
            app.Append(sig_to_blob(rec.oph_sig));
            app.Append(sig_to_blob(rec.oph_sig2));  // NULL if empty
            app.Append(static_cast<uint32_t>(rec.n_real_bins));
            app.Append(static_cast<uint64_t>(rec.genome_length));
            app.EndRow();
        }
        app.Close();
        auto rc = conn.Query("COMMIT");
        check(rc, "insert_batch COMMIT");
    } catch (...) {
        conn.Query("ROLLBACK");
        throw;
    }

    if (++batches_since_checkpoint_ >= cfg_.checkpoint_every_batches) {
        checkpoint();
        batches_since_checkpoint_ = 0;
    }
}

void SketchStore::upsert_sig2(
    const std::vector<std::pair<std::string, std::vector<uint16_t>>>& pairs)
{
    if (pairs.empty()) return;
    auto& conn = thread_connection();
    conn.Query("BEGIN TRANSACTION");
    try {
        for (const auto& [acc, sig2] : pairs) {
            // Use prepared-style via concatenation — DuckDB C++ API has no ? binding
            // for BLOB; we use a hex literal approach via Value
            auto stmt = conn.Prepare(
                "UPDATE sketches SET oph_sig2 = $1 WHERE accession = $2");
            stmt->Execute(sig_to_blob(sig2), duckdb::Value(acc));
        }
        conn.Query("COMMIT");
    } catch (...) {
        conn.Query("ROLLBACK");
        throw;
    }
}

void SketchStore::record_failures(const std::vector<SketchFailure>& failures) {
    if (failures.empty()) return;
    auto& conn = thread_connection();
    conn.Query("BEGIN TRANSACTION");
    try {
        duckdb::Appender app(conn, "sketch_failures");
        for (const auto& f : failures) {
            app.BeginRow();
            app.Append(duckdb::Value(f.accession));
            app.Append(f.taxonomy.empty() ? duckdb::Value() : duckdb::Value(f.taxonomy));
            app.Append(duckdb::Value(f.file_path));
            app.Append(duckdb::Value(f.error_message));
            app.Append(static_cast<uint32_t>(1));
            app.EndRow();
        }
        app.Close();
        conn.Query("COMMIT");
    } catch (...) {
        conn.Query("ROLLBACK");
        throw;
    }
}

std::vector<SketchStore::SketchRecord>
SketchStore::fetch_ordered(const std::vector<std::string>& accessions) {
    if (accessions.empty()) return {};

    auto& conn = thread_connection();

    // Use a temp table with ord so we preserve caller order and detect misses.
    // Temp tables are connection-local in DuckDB — safe for per-thread connections.
    auto rc0 = conn.Query(
        "CREATE TEMP TABLE IF NOT EXISTS _fetch_acc (ord UINTEGER, accession VARCHAR)");
    check(rc0, "create _fetch_acc");
    conn.Query("DELETE FROM _fetch_acc");

    {
        duckdb::Appender app(conn, "_fetch_acc");
        for (size_t i = 0; i < accessions.size(); ++i) {
            app.BeginRow();
            app.Append(static_cast<uint32_t>(i));
            app.Append(duckdb::Value(accessions[i]));
            app.EndRow();
        }
        app.Close();
    }

    auto res = conn.Query(R"(
        SELECT a.ord, s.accession, s.taxonomy,
               s.oph_sig, s.oph_sig2, s.n_real_bins, s.genome_length
        FROM _fetch_acc a
        LEFT JOIN sketches s USING (accession)
        ORDER BY a.ord
    )");
    check(res, "fetch_ordered query");

    std::vector<SketchRecord> out(accessions.size());
    // Pre-fill accession so callers can detect misses by checking oph_sig.empty()
    for (size_t i = 0; i < accessions.size(); ++i)
        out[i].accession = accessions[i];

    for (;;) {
        auto chunk = res->Fetch();
        if (!chunk || chunk->size() == 0) break;
        for (size_t row = 0; row < chunk->size(); ++row) {
            uint32_t ord = chunk->GetValue(0, row).GetValue<uint32_t>();
            auto& rec = out[ord];
            // accession already set; rest may be NULL if not found
            if (!chunk->GetValue(1, row).IsNull()) {
                if (!chunk->GetValue(2, row).IsNull())
                    rec.taxonomy = chunk->GetValue(2, row).GetValue<std::string>();
                rec.oph_sig     = blob_to_sig(chunk->GetValue(3, row));
                rec.oph_sig2    = blob_to_sig(chunk->GetValue(4, row));
                rec.n_real_bins = chunk->GetValue(5, row).GetValue<uint32_t>();
                rec.genome_length = chunk->GetValue(6, row).GetValue<uint64_t>();
            }
        }
    }

    return out;
}

std::unordered_map<std::string, std::vector<uint16_t>>
SketchStore::fetch_sig2(const std::vector<std::string>& accessions) {
    if (accessions.empty()) return {};

    auto& conn = thread_connection();

    conn.Query("CREATE TEMP TABLE IF NOT EXISTS _sig2_acc (accession VARCHAR)");
    conn.Query("DELETE FROM _sig2_acc");

    {
        duckdb::Appender app(conn, "_sig2_acc");
        for (const auto& acc : accessions) {
            app.BeginRow();
            app.Append(duckdb::Value(acc));
            app.EndRow();
        }
        app.Close();
    }

    auto res = conn.Query(R"(
        SELECT s.accession, s.oph_sig2
        FROM _sig2_acc a
        JOIN sketches s USING (accession)
        WHERE s.oph_sig2 IS NOT NULL
    )");
    check(res, "fetch_sig2");

    std::unordered_map<std::string, std::vector<uint16_t>> out;
    for (;;) {
        auto chunk = res->Fetch();
        if (!chunk || chunk->size() == 0) break;
        for (size_t row = 0; row < chunk->size(); ++row) {
            auto acc = chunk->GetValue(0, row).GetValue<std::string>();
            out[acc] = blob_to_sig(chunk->GetValue(1, row));
        }
    }
    return out;
}

void SketchStore::checkpoint() {
    auto conn = std::make_unique<duckdb::Connection>(*database_);
    auto r = conn->Query("CHECKPOINT");
    if (r->HasError())
        spdlog::warn("SketchStore checkpoint failed: {}", r->GetError());
}

} // namespace derep::db
