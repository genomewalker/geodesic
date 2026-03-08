#include "db/embedding_store.hpp"
#include "core/logging.hpp"
#include <spdlog/spdlog.h>
#include <cstring>
#include <sstream>
#include <iomanip>

namespace derep::db {

EmbeddingStore::EmbeddingStore(int embedding_dim)
    : embedding_dim_(embedding_dim) {}

EmbeddingStore::~EmbeddingStore() {
    close();
}

bool EmbeddingStore::open(const std::filesystem::path& db_path) {
    if (db_) {
        spdlog::warn("[EmbeddingStore] Already open");
        return false;
    }

    auto try_open = [&]() {
        db_ = std::make_unique<duckdb::DuckDB>(db_path.string());
        conn_ = std::make_unique<duckdb::Connection>(*db_);
        conn_->Query("PRAGMA threads=1");
        conn_->Query("PRAGMA enable_external_access=true");
        if (!load_vss_extension())
            spdlog::warn("[EmbeddingStore] VSS not available, using brute-force search");
        if (!create_schema()) {
            close();
            throw std::runtime_error("Failed to create schema");
        }
        if (vss_loaded_) {
            // Only rebuild if the index isn't already persisted — avoids expensive
            // rebuild on every restart when the DB has accumulated embeddings.
            auto r = conn_->Query(
                "SELECT COUNT(*) FROM duckdb_indexes() WHERE index_name = 'idx_emb_hnsw'");
            bool idx_exists = false;
            if (!r->HasError()) {
                auto c = r->Fetch();
                if (c && c->size() > 0)
                    idx_exists = c->GetValue(0, 0).GetValue<int64_t>() > 0;
            }
            if (idx_exists)
                index_exists_ = true;
            else
                rebuild_index();
        }
        spdlog::info("[EmbeddingStore] Opened at {} (dim={}, vss={})",
                     db_path.string(), embedding_dim_, vss_loaded_);
    };

    try {
        try_open();
        return true;
    } catch (const std::exception& e) {
        std::string msg = e.what();
        // WAL replay fails when it contains HNSW ops but VSS isn't loaded yet at replay time.
        // Self-heal: discard the WAL (embedding cache is rebuildable) and reopen fresh.
        if (msg.find("replaying WAL") != std::string::npos) {
            spdlog::warn("[EmbeddingStore] Stale HNSW ops in WAL, discarding and reopening");
            db_.reset();
            conn_.reset();
            std::error_code ec;
            std::filesystem::remove(db_path.string() + ".wal", ec);
            try {
                try_open();
                return true;
            } catch (const std::exception& e2) {
                spdlog::error("[EmbeddingStore] Failed to open after WAL recovery: {}", e2.what());
            }
        } else {
            spdlog::error("[EmbeddingStore] Failed to open: {}", msg);
        }
        db_.reset();
        conn_.reset();
        return false;
    }
}

void EmbeddingStore::close() {
    std::lock_guard lock(mutex_);
    conn_.reset();
    db_.reset();
    vss_loaded_ = false;
    index_exists_ = false;
}

bool EmbeddingStore::load_vss_extension() {
    try {
        conn_->Query("INSTALL vss; LOAD vss;");
        vss_loaded_ = true;
        spdlog::info("[EmbeddingStore] VSS extension loaded");
        return true;
    } catch (...) {
        vss_loaded_ = false;
        return false;
    }
}

bool EmbeddingStore::create_schema() {
    try {
        // Check if table already exists and detect its dimension
        auto check = conn_->Query(
            "SELECT cardinality(embedding) FROM genome_embeddings LIMIT 1");
        if (!check->HasError()) {
            auto chunk = check->Fetch();
            if (chunk && chunk->size() > 0) {
                int existing_dim = chunk->GetValue(0, 0).GetValue<int>();
                if (existing_dim > 0 && existing_dim != embedding_dim_) {
                    spdlog::info("[EmbeddingStore] Adapting to existing dimension: {} -> {}",
                                 embedding_dim_, existing_dim);
                    embedding_dim_ = existing_dim;
                }
                // Table exists with data, skip creation
                goto create_indexes;
            }
        }

        {
            // FLOAT[N] fixed-size array required by VSS/HNSW
            std::string create_sql =
                "CREATE TABLE IF NOT EXISTS genome_embeddings ("
                "    accession VARCHAR PRIMARY KEY,"
                "    taxonomy VARCHAR NOT NULL,"
                "    file_path VARCHAR NOT NULL,"
                "    embedding FLOAT[" + std::to_string(embedding_dim_) + "] NOT NULL,"
                "    isolation_score FLOAT DEFAULT 0.0,"
                "    quality_score FLOAT DEFAULT 50.0,"
                "    genome_size BIGINT DEFAULT 0,"
                "    created_at TIMESTAMP DEFAULT current_timestamp,"
                "    oph_sig BLOB,"
                "    oph_sig2 BLOB,"
                "    real_bins_mask BLOB,"
                "    chimera_score FLOAT DEFAULT 0.0"
                ")";
            auto result = conn_->Query(create_sql);
            if (result->HasError()) {
                spdlog::error("[EmbeddingStore] Schema error: {}", result->GetError());
                return false;
            }
        }

        // Backward compat: add oph_sig / real_bins_mask to existing DBs that predate these columns.
        {
            auto col_check = conn_->Query("SELECT oph_sig FROM genome_embeddings LIMIT 0");
            if (col_check->HasError()) {
                auto r = conn_->Query("ALTER TABLE genome_embeddings ADD COLUMN oph_sig BLOB");
                if (r->HasError())
                    spdlog::warn("[EmbeddingStore] Could not add oph_sig column: {}", r->GetError());
            }
        }
        {
            auto col_check = conn_->Query("SELECT real_bins_mask FROM genome_embeddings LIMIT 0");
            if (col_check->HasError()) {
                auto r = conn_->Query("ALTER TABLE genome_embeddings ADD COLUMN real_bins_mask BLOB");
                if (r->HasError())
                    spdlog::warn("[EmbeddingStore] Could not add real_bins_mask column: {}", r->GetError());
            }
        }
        {
            auto col_check = conn_->Query("SELECT oph_sig2 FROM genome_embeddings LIMIT 0");
            if (col_check->HasError()) {
                auto r = conn_->Query("ALTER TABLE genome_embeddings ADD COLUMN oph_sig2 BLOB");
                if (r->HasError())
                    spdlog::warn("[EmbeddingStore] Could not add oph_sig2 column: {}", r->GetError());
            }
        }
        {
            auto col_check = conn_->Query("SELECT chimera_score FROM genome_embeddings LIMIT 0");
            if (col_check->HasError()) {
                auto r = conn_->Query("ALTER TABLE genome_embeddings ADD COLUMN chimera_score FLOAT DEFAULT 0.0");
                if (r->HasError())
                    spdlog::warn("[EmbeddingStore] Could not add chimera_score column: {}", r->GetError());
            }
        }

        create_indexes:

        // Create index on taxonomy for filtered queries
        conn_->Query("CREATE INDEX IF NOT EXISTS idx_emb_taxonomy ON genome_embeddings(taxonomy)");

        // Representatives table - tracks which genomes are representatives per taxonomy
        conn_->Query(R"(
            CREATE TABLE IF NOT EXISTS representatives (
                accession VARCHAR PRIMARY KEY,
                taxonomy VARCHAR NOT NULL,
                selected_at TIMESTAMP DEFAULT current_timestamp
            )
        )");
        conn_->Query("CREATE INDEX IF NOT EXISTS idx_rep_taxonomy ON representatives(taxonomy)");

        return true;
    } catch (const std::exception& e) {
        spdlog::error("[EmbeddingStore] Schema error: {}", e.what());
        return false;
    }
}

bool EmbeddingStore::rebuild_index() {
    if (!vss_loaded_) return false;

    std::lock_guard lock(mutex_);

    try {
        // Drop existing index
        conn_->Query("DROP INDEX IF EXISTS idx_emb_hnsw");

        // Create HNSW index with cosine metric (matches normalized embeddings)
        std::ostringstream sql;
        sql << "CREATE INDEX idx_emb_hnsw ON genome_embeddings USING HNSW (embedding) "
            << "WITH (metric = 'cosine')";

        auto result = conn_->Query(sql.str());
        if (result->HasError()) {
            spdlog::warn("[EmbeddingStore] HNSW index creation failed: {}", result->GetError());
            index_exists_ = false;
            return false;
        }

        // Compact to remove any deleted entries
        conn_->Query("PRAGMA hnsw_compact_index('idx_emb_hnsw')");

        index_exists_ = true;
        spdlog::info("[EmbeddingStore] HNSW index rebuilt");
        return true;

    } catch (const std::exception& e) {
        spdlog::warn("[EmbeddingStore] HNSW index error: {}", e.what());
        index_exists_ = false;
        return false;
    }
}

bool EmbeddingStore::compact_index() {
    if (!vss_loaded_ || !index_exists_) return false;

    std::lock_guard lock(mutex_);
    try {
        conn_->Query("PRAGMA hnsw_compact_index('idx_emb_hnsw')");
        return true;
    } catch (...) {
        return false;
    }
}

std::string EmbeddingStore::embedding_to_sql(const std::vector<float>& emb) const {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6) << "[";
    for (size_t i = 0; i < emb.size(); ++i) {
        if (i > 0) ss << ",";
        ss << emb[i];
    }
    ss << "]::FLOAT[" << embedding_dim_ << "]";
    return ss.str();
}

bool EmbeddingStore::insert_embedding(const GenomeEmbedding& emb) {
    if (emb.embedding.size() != static_cast<size_t>(embedding_dim_)) {
        spdlog::error("[EmbeddingStore] Dimension mismatch: {} vs {}",
                      emb.embedding.size(), embedding_dim_);
        return false;
    }

    std::lock_guard lock(mutex_);

    try {
        std::ostringstream sql;
        sql << "INSERT OR REPLACE INTO genome_embeddings "
            << "(accession, taxonomy, file_path, embedding, isolation_score, quality_score, genome_size) "
            << "VALUES ('" << emb.accession << "', "
            << "'" << emb.taxonomy << "', "
            << "'" << emb.file_path.string() << "', "
            << embedding_to_sql(emb.embedding) << ", "
            << emb.isolation_score << ", "
            << emb.quality_score << ", "
            << emb.genome_size << ")";

        auto result = conn_->Query(sql.str());
        if (result->HasError()) {
            spdlog::error("[EmbeddingStore] Insert error: {}", result->GetError());
            return false;
        }
        return true;

    } catch (const std::exception& e) {
        spdlog::error("[EmbeddingStore] Insert error: {}", e.what());
        return false;
    }
}

bool EmbeddingStore::insert_embeddings(const std::vector<GenomeEmbedding>& embeddings) {
    if (embeddings.empty()) return true;

    std::lock_guard lock(mutex_);

    try {
        // Check actual dim of this batch against the schema dim.
        // The geodesic module picks dim adaptively (128/256/512 based on ANI spread),
        // so batches can arrive with a different dim than the schema was created with.
        // Skip mismatched batches rather than spamming cast errors.
        int actual_dim = static_cast<int>(embeddings[0].embedding.size());
        if (actual_dim != embedding_dim_) {
            if (is_verbose())
                spdlog::debug("[EmbeddingStore] Skipping batch: dim {} != schema {}",
                              actual_dim, embedding_dim_);
            return false;
        }

        // Stage into a FLOAT[] temp table (Appender can't write FLOAT[N] directly),
        // then cast-insert into the main FLOAT[N] table.
        conn_->Query(
            "CREATE TEMP TABLE IF NOT EXISTS _emb_stage ("
            "  accession VARCHAR, taxonomy VARCHAR, file_path VARCHAR,"
            "  embedding FLOAT[],"
            "  isolation_score FLOAT, quality_score FLOAT, genome_size BIGINT,"
            "  oph_sig BLOB, oph_sig2 BLOB, real_bins_mask BLOB, chimera_score FLOAT)");
        conn_->Query("DELETE FROM _emb_stage");

        {
            duckdb::Appender appender(*conn_, "_emb_stage");
            for (const auto& emb : embeddings) {
                if (emb.embedding.empty()) {
                    spdlog::warn("[EmbeddingStore] Skipping {}: empty embedding", emb.accession);
                    continue;
                }
                duckdb::vector<duckdb::Value> arr;
                arr.reserve(emb.embedding.size());
                for (float v : emb.embedding) arr.push_back(duckdb::Value::FLOAT(v));

                // Store OPH signature as raw BLOB (BLOB_RAW bypasses UTF-8 validation)
                duckdb::Value oph_val;
                if (!emb.oph_sig.empty()) {
                    std::string sig_bytes(emb.oph_sig.size() * sizeof(uint16_t), '\0');
                    std::memcpy(sig_bytes.data(), emb.oph_sig.data(), sig_bytes.size());
                    oph_val = duckdb::Value::BLOB_RAW(sig_bytes);
                } else {
                    oph_val = duckdb::Value(duckdb::LogicalType::BLOB);
                }

                duckdb::Value oph_val2;
                if (!emb.oph_sig2.empty()) {
                    std::string sig_bytes2(emb.oph_sig2.size() * sizeof(uint16_t), '\0');
                    std::memcpy(sig_bytes2.data(), emb.oph_sig2.data(), sig_bytes2.size());
                    oph_val2 = duckdb::Value::BLOB_RAW(sig_bytes2);
                } else {
                    oph_val2 = duckdb::Value(duckdb::LogicalType::BLOB);
                }

                duckdb::Value mask_val;
                if (!emb.real_bins_mask.empty()) {
                    std::string mask_bytes(emb.real_bins_mask.size() * sizeof(uint64_t), '\0');
                    std::memcpy(mask_bytes.data(), emb.real_bins_mask.data(), mask_bytes.size());
                    mask_val = duckdb::Value::BLOB_RAW(mask_bytes);
                } else {
                    mask_val = duckdb::Value(duckdb::LogicalType::BLOB);
                }

                appender.BeginRow();
                appender.Append<duckdb::string_t>(duckdb::string_t(emb.accession));
                appender.Append<duckdb::string_t>(duckdb::string_t(emb.taxonomy));
                appender.Append<duckdb::string_t>(duckdb::string_t(emb.file_path.string()));
                appender.Append(duckdb::Value::LIST(duckdb::LogicalType::FLOAT, std::move(arr)));
                appender.Append<float>(emb.isolation_score);
                appender.Append<float>(emb.quality_score);
                appender.Append<int64_t>(static_cast<int64_t>(emb.genome_size));
                appender.Append(std::move(oph_val));
                appender.Append(std::move(oph_val2));
                appender.Append(std::move(mask_val));
                appender.Append<float>(emb.chimera_score);
                appender.EndRow();
            }
            appender.Close();
        }

        // Upsert into main table with cast to FLOAT[N]
        std::string upsert_sql =
            "INSERT OR REPLACE INTO genome_embeddings "
            "SELECT accession, taxonomy, file_path,"
            "  embedding::FLOAT[" + std::to_string(embedding_dim_) + "],"
            "  isolation_score, quality_score, genome_size, current_timestamp, oph_sig, oph_sig2, real_bins_mask, chimera_score"
            " FROM _emb_stage";
        auto r = conn_->Query(upsert_sql);
        if (r->HasError()) {
            spdlog::error("[EmbeddingStore] Batch insert error: {}", r->GetError());
            return false;
        }
        return true;

    } catch (const std::exception& e) {
        spdlog::error("[EmbeddingStore] Batch insert error: {}", e.what());
        return false;
    }
}

bool EmbeddingStore::delete_embedding(const std::string& accession) {
    std::lock_guard lock(mutex_);

    try {
        auto result = conn_->Query(
            "DELETE FROM genome_embeddings WHERE accession = '" + accession + "'");
        return !result->HasError();
    } catch (...) {
        return false;
    }
}

bool EmbeddingStore::has_embedding(const std::string& accession) {
    try {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
        auto result = conn_->Query(
            "SELECT 1 FROM genome_embeddings WHERE accession = '" + accession + "' LIMIT 1");
        if (result->HasError()) return false;
        auto chunk = result->Fetch();
        return chunk && chunk->size() > 0;
    } catch (...) {
        return false;
    }
}

std::optional<GenomeEmbedding> EmbeddingStore::get_embedding(const std::string& accession) {
    try {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
        auto result = conn_->Query(
            "SELECT accession, taxonomy, file_path, embedding, isolation_score, "
            "quality_score, genome_size FROM genome_embeddings "
            "WHERE accession = '" + accession + "'");

        if (result->HasError()) return std::nullopt;
        auto chunk = result->Fetch();
        if (!chunk || chunk->size() == 0) return std::nullopt;

        GenomeEmbedding emb;
        emb.accession = chunk->GetValue(0, 0).GetValue<std::string>();
        emb.taxonomy = chunk->GetValue(1, 0).GetValue<std::string>();
        emb.file_path = chunk->GetValue(2, 0).GetValue<std::string>();

        // Extract embedding array
        auto arr = chunk->GetValue(3, 0);
        auto& children = duckdb::ArrayValue::GetChildren(arr);
        emb.embedding.reserve(children.size());
        for (const auto& v : children) {
            emb.embedding.push_back(v.GetValue<float>());
        }

        emb.isolation_score = chunk->GetValue(4, 0).GetValue<float>();
        emb.quality_score = chunk->GetValue(5, 0).GetValue<float>();
        emb.genome_size = chunk->GetValue(6, 0).GetValue<int64_t>();

        return emb;

    } catch (const std::exception& e) {
        spdlog::error("[EmbeddingStore] Get error: {}", e.what());
        return std::nullopt;
    }
}

std::vector<std::string> EmbeddingStore::get_embedded_accessions(const std::string& taxonomy) {
    std::vector<std::string> accessions;

    try {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
        std::string sql = "SELECT accession FROM genome_embeddings";
        if (!taxonomy.empty()) {
            sql += " WHERE taxonomy = '" + taxonomy + "'";
        }

        auto result = conn_->Query(sql);
        if (result->HasError()) return accessions;

        while (auto chunk = result->Fetch()) {
            for (size_t i = 0; i < chunk->size(); ++i) {
                accessions.push_back(chunk->GetValue(0, i).GetValue<std::string>());
            }
        }
    } catch (...) {}

    return accessions;
}

size_t EmbeddingStore::count_embeddings(const std::string& taxonomy) {
    try {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
        std::string sql = "SELECT COUNT(*) FROM genome_embeddings";
        if (!taxonomy.empty()) {
            sql += " WHERE taxonomy = '" + taxonomy + "'";
        }

        auto result = conn_->Query(sql);
        if (result->HasError()) return 0;
        auto chunk = result->Fetch();
        if (!chunk || chunk->size() == 0) return 0;
        return chunk->GetValue(0, 0).GetValue<int64_t>();
    } catch (...) {
        return 0;
    }
}

size_t EmbeddingStore::total_embeddings() const {
    // Cast away const for query (read-only)
    auto* self = const_cast<EmbeddingStore*>(this);
    return self->count_embeddings();
}

std::vector<NearestNeighbor> EmbeddingStore::find_nearest(
    const std::vector<float>& query_embedding,
    int k,
    const std::string& taxonomy_filter) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    std::vector<NearestNeighbor> results;

    if (query_embedding.size() != static_cast<size_t>(embedding_dim_)) {
        spdlog::error("[EmbeddingStore] Query dimension mismatch");
        return results;
    }

    try {
        std::ostringstream sql;
        sql << "SELECT accession, "
            << "array_cosine_similarity(embedding, " << embedding_to_sql(query_embedding) << ") AS similarity "
            << "FROM genome_embeddings ";

        if (!taxonomy_filter.empty()) {
            sql << "WHERE taxonomy = '" << taxonomy_filter << "' ";
        }

        // ORDER BY array_distance uses HNSW index if available
        sql << "ORDER BY array_distance(embedding, " << embedding_to_sql(query_embedding) << ") "
            << "LIMIT " << k;

        auto result = conn_->Query(sql.str());
        if (result->HasError()) {
            spdlog::error("[EmbeddingStore] Search error: {}", result->GetError());
            return results;
        }

        while (auto chunk = result->Fetch()) {
            for (size_t i = 0; i < chunk->size(); ++i) {
                NearestNeighbor nn;
                nn.accession = chunk->GetValue(0, i).GetValue<std::string>();
                nn.similarity = chunk->GetValue(1, i).GetValue<float>();
                nn.distance = 1.0f - nn.similarity;  // cosine distance
                results.push_back(nn);
            }
        }

    } catch (const std::exception& e) {
        spdlog::error("[EmbeddingStore] Search error: {}", e.what());
    }

    return results;
}

std::vector<NearestNeighbor> EmbeddingStore::find_nearest_by_accession(
    const std::string& accession,
    int k,
    const std::string& taxonomy_filter) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    // Get the embedding for this accession
    auto emb_opt = get_embedding(accession);
    if (!emb_opt) {
        spdlog::warn("[EmbeddingStore] Accession not found: {}", accession);
        return {};
    }

    // Find nearest, excluding self
    auto results = find_nearest(emb_opt->embedding, k + 1, taxonomy_filter);

    // Remove self from results
    results.erase(
        std::remove_if(results.begin(), results.end(),
                       [&](const NearestNeighbor& nn) { return nn.accession == accession; }),
        results.end());

    // Trim to k
    if (results.size() > static_cast<size_t>(k)) {
        results.resize(k);
    }

    return results;
}

// ============================================================================
// Incremental update support
// ============================================================================

std::vector<std::string> EmbeddingStore::get_missing_accessions(
    const std::vector<std::string>& all_accessions) {

    auto embedded = get_embedded_set();
    std::vector<std::string> missing;
    missing.reserve(all_accessions.size());

    for (const auto& acc : all_accessions) {
        if (embedded.find(acc) == embedded.end()) {
            missing.push_back(acc);
        }
    }

    return missing;
}

std::unordered_set<std::string> EmbeddingStore::get_embedded_set(const std::string& taxonomy) {
    std::unordered_set<std::string> result;

    try {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
        std::string sql = "SELECT accession FROM genome_embeddings";
        if (!taxonomy.empty()) {
            sql += " WHERE taxonomy = '" + taxonomy + "'";
        }

        auto qr = conn_->Query(sql);
        if (qr->HasError()) return result;

        while (auto chunk = qr->Fetch()) {
            for (size_t i = 0; i < chunk->size(); ++i) {
                result.insert(chunk->GetValue(0, i).GetValue<std::string>());
            }
        }
    } catch (...) {}

    return result;
}

std::vector<GenomeEmbedding> EmbeddingStore::load_embeddings(const std::string& taxonomy) {
    std::vector<GenomeEmbedding> embeddings;

    try {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
        std::string sql =
            "SELECT accession, taxonomy, file_path, embedding, "
            "isolation_score, quality_score, genome_size, oph_sig, oph_sig2, real_bins_mask, chimera_score "
            "FROM genome_embeddings";
        if (!taxonomy.empty()) {
            sql += " WHERE taxonomy = '" + taxonomy + "'";
        }

        auto result = conn_->Query(sql);
        if (result->HasError()) {
            spdlog::error("[EmbeddingStore] Load error: {}", result->GetError());
            return embeddings;
        }

        while (auto chunk = result->Fetch()) {
            for (size_t i = 0; i < chunk->size(); ++i) {
                GenomeEmbedding emb;
                emb.accession = chunk->GetValue(0, i).GetValue<std::string>();
                emb.taxonomy = chunk->GetValue(1, i).GetValue<std::string>();
                emb.file_path = chunk->GetValue(2, i).GetValue<std::string>();

                // Extract embedding array (placeholder zeros — actual vectors rebuilt by Nyström)
                auto arr = chunk->GetValue(3, i);
                auto& children = duckdb::ArrayValue::GetChildren(arr);
                emb.embedding.reserve(children.size());
                for (const auto& v : children) {
                    emb.embedding.push_back(v.GetValue<float>());
                }

                emb.isolation_score = chunk->GetValue(4, i).GetValue<float>();
                emb.quality_score = chunk->GetValue(5, i).GetValue<float>();
                emb.genome_size = chunk->GetValue(6, i).GetValue<int64_t>();

                // Deserialize OPH signature from BLOB
                auto oph_val = chunk->GetValue(7, i);
                if (!oph_val.IsNull()) {
                    const std::string& bytes = duckdb::StringValue::Get(oph_val);
                    size_t n_hashes = bytes.size() / sizeof(uint16_t);
                    emb.oph_sig.resize(n_hashes);
                    std::memcpy(emb.oph_sig.data(), bytes.data(), bytes.size());
                }

                // Deserialize second OPH signature from BLOB
                auto oph_val2 = chunk->GetValue(8, i);
                if (!oph_val2.IsNull()) {
                    const std::string& bytes = duckdb::StringValue::Get(oph_val2);
                    size_t n_hashes = bytes.size() / sizeof(uint16_t);
                    emb.oph_sig2.resize(n_hashes);
                    std::memcpy(emb.oph_sig2.data(), bytes.data(), bytes.size());
                }

                // Deserialize real-bin bitmask from BLOB
                auto mask_val = chunk->GetValue(9, i);
                if (!mask_val.IsNull()) {
                    const std::string& bytes = duckdb::StringValue::Get(mask_val);
                    size_t n_words = bytes.size() / sizeof(uint64_t);
                    emb.real_bins_mask.resize(n_words);
                    std::memcpy(emb.real_bins_mask.data(), bytes.data(), bytes.size());
                    emb.n_real_bins = 0;
                    for (uint64_t w : emb.real_bins_mask)
                        emb.n_real_bins += static_cast<uint32_t>(__builtin_popcountll(w));
                }

                auto chimera_val = chunk->GetValue(10, i);
                if (!chimera_val.IsNull())
                    emb.chimera_score = chimera_val.GetValue<float>();

                embeddings.push_back(std::move(emb));
            }
        }

        spdlog::info("[EmbeddingStore] Loaded {} embeddings", embeddings.size());

    } catch (const std::exception& e) {
        spdlog::error("[EmbeddingStore] Load error: {}", e.what());
    }

    return embeddings;
}

bool EmbeddingStore::update_isolation_score(const std::string& accession, float score) {
    std::lock_guard lock(mutex_);

    try {
        std::ostringstream sql;
        sql << "UPDATE genome_embeddings SET isolation_score = " << score
            << " WHERE accession = '" << accession << "'";

        auto result = conn_->Query(sql.str());
        return !result->HasError();
    } catch (...) {
        return false;
    }
}

bool EmbeddingStore::update_isolation_scores(
    const std::vector<std::pair<std::string, float>>& scores) {

    if (scores.empty()) return true;

    std::lock_guard lock(mutex_);

    try {
        // Use a transaction for batch updates
        conn_->Query("BEGIN TRANSACTION");

        for (const auto& [accession, score] : scores) {
            std::ostringstream sql;
            sql << "UPDATE genome_embeddings SET isolation_score = " << score
                << " WHERE accession = '" << accession << "'";
            conn_->Query(sql.str());
        }

        conn_->Query("COMMIT");
        return true;

    } catch (const std::exception& e) {
        conn_->Query("ROLLBACK");
        spdlog::error("[EmbeddingStore] Update scores error: {}", e.what());
        return false;
    }
}

// ============================================================================
// Representative tracking
// ============================================================================

bool EmbeddingStore::set_representatives(const std::string& taxonomy,
                                         const std::vector<std::string>& rep_accessions) {
    std::lock_guard lock(mutex_);

    try {
        conn_->Query("BEGIN TRANSACTION");

        // Clear existing representatives for this taxonomy
        conn_->Query("DELETE FROM representatives WHERE taxonomy = '" + taxonomy + "'");

        // Insert new representatives (replace to handle cross-taxonomy reps)
        for (const auto& acc : rep_accessions) {
            std::ostringstream sql;
            sql << "INSERT OR REPLACE INTO representatives (accession, taxonomy, selected_at) "
                << "VALUES ('" << acc << "', '" << taxonomy << "', current_timestamp)";
            auto result = conn_->Query(sql.str());
            if (result->HasError()) {
                spdlog::warn("[EmbeddingStore] Failed to set rep {}: {}", acc, result->GetError());
            }
        }

        conn_->Query("COMMIT");

        if (is_verbose()) spdlog::info("[EmbeddingStore] Set {} representatives for {}",
                     rep_accessions.size(), taxonomy.substr(taxonomy.rfind(';') + 1));
        return true;

    } catch (const std::exception& e) {
        conn_->Query("ROLLBACK");
        spdlog::error("[EmbeddingStore] Set reps error: {}", e.what());
        return false;
    }
}

bool EmbeddingStore::clear_representatives(const std::string& taxonomy) {
    std::lock_guard lock(mutex_);

    try {
        std::string sql = "DELETE FROM representatives";
        if (!taxonomy.empty()) {
            sql += " WHERE taxonomy = '" + taxonomy + "'";
        }
        auto result = conn_->Query(sql);
        return !result->HasError();
    } catch (...) {
        return false;
    }
}

std::vector<std::string> EmbeddingStore::get_representatives(const std::string& taxonomy) {
    std::vector<std::string> reps;

    try {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
        std::string sql = "SELECT accession FROM representatives";
        if (!taxonomy.empty()) {
            sql += " WHERE taxonomy = '" + taxonomy + "'";
        }

        auto result = conn_->Query(sql);
        if (result->HasError()) return reps;

        while (auto chunk = result->Fetch()) {
            for (size_t i = 0; i < chunk->size(); ++i) {
                reps.push_back(chunk->GetValue(0, i).GetValue<std::string>());
            }
        }
    } catch (...) {}

    return reps;
}

std::vector<RepresentativeInfo> EmbeddingStore::get_representative_info(const std::string& taxonomy) {
    std::vector<RepresentativeInfo> info;

    try {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
        std::string sql =
            "SELECT e.accession, e.taxonomy, e.file_path, "
            "e.isolation_score, e.quality_score, e.genome_size "
            "FROM genome_embeddings e "
            "JOIN representatives r ON e.accession = r.accession";
        if (!taxonomy.empty()) {
            sql += " WHERE r.taxonomy = '" + taxonomy + "'";
        }

        auto result = conn_->Query(sql);
        if (result->HasError()) return info;

        while (auto chunk = result->Fetch()) {
            for (size_t i = 0; i < chunk->size(); ++i) {
                RepresentativeInfo ri;
                ri.accession = chunk->GetValue(0, i).GetValue<std::string>();
                ri.taxonomy = chunk->GetValue(1, i).GetValue<std::string>();
                ri.file_path = chunk->GetValue(2, i).GetValue<std::string>();
                ri.isolation_score = chunk->GetValue(3, i).GetValue<float>();
                ri.quality_score = chunk->GetValue(4, i).GetValue<float>();
                ri.genome_size = chunk->GetValue(5, i).GetValue<int64_t>();
                info.push_back(ri);
            }
        }
    } catch (...) {}

    return info;
}

bool EmbeddingStore::is_representative(const std::string& accession) {
    try {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
        auto result = conn_->Query(
            "SELECT 1 FROM representatives WHERE accession = '" + accession + "' LIMIT 1");
        if (result->HasError()) return false;
        auto chunk = result->Fetch();
        return chunk && chunk->size() > 0;
    } catch (...) {
        return false;
    }
}

// ============================================================================
// Genome removal
// ============================================================================

bool EmbeddingStore::remove_genomes(const std::vector<std::string>& accessions) {
    if (accessions.empty()) return true;

    std::lock_guard lock(mutex_);

    try {
        conn_->Query("BEGIN TRANSACTION");

        for (const auto& acc : accessions) {
            conn_->Query("DELETE FROM genome_embeddings WHERE accession = '" + acc + "'");
            conn_->Query("DELETE FROM representatives WHERE accession = '" + acc + "'");
        }

        conn_->Query("COMMIT");
        return true;

    } catch (const std::exception& e) {
        conn_->Query("ROLLBACK");
        spdlog::error("[EmbeddingStore] Remove genomes error: {}", e.what());
        return false;
    }
}

bool EmbeddingStore::remove_taxonomy(const std::string& taxonomy) {
    std::lock_guard lock(mutex_);

    try {
        conn_->Query("BEGIN TRANSACTION");
        conn_->Query("DELETE FROM genome_embeddings WHERE taxonomy = '" + taxonomy + "'");
        conn_->Query("DELETE FROM representatives WHERE taxonomy = '" + taxonomy + "'");
        conn_->Query("COMMIT");
        return true;

    } catch (const std::exception& e) {
        conn_->Query("ROLLBACK");
        spdlog::error("[EmbeddingStore] Remove taxonomy error: {}", e.what());
        return false;
    }
}

} // namespace derep::db
