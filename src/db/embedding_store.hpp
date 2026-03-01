#pragma once
// EmbeddingStore: Persistent genome embedding storage with DuckDB-VSS
//
// Architecture (following cc-soul patterns):
// - Separate embeddings database to isolate HNSW from main DB
// - threads=1 to avoid HNSW parallelism races
// - No persistent HNSW (rebuild at startup for stability)
// - Single write connection with mutex for thread-safe writes
// - Cosine metric for normalized embeddings
//
// Incremental update workflow:
// 1. get_missing_accessions() - find genomes not yet embedded
// 2. Embed only the missing genomes
// 3. insert_embeddings() - store new embeddings
// 4. rebuild_index() - rebuild HNSW with all embeddings
// 5. set_representatives() - mark selected representatives

#include <duckdb.hpp>
#include <filesystem>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace derep::db {

struct GenomeEmbedding {
    std::string accession;
    std::string taxonomy;
    std::filesystem::path file_path;
    std::vector<float> embedding;
    float isolation_score = 0.0f;
    float quality_score = 50.0f;
    uint64_t genome_size = 0;
};

struct NearestNeighbor {
    std::string accession;
    float similarity;
    float distance;
};

struct RepresentativeInfo {
    std::string accession;
    std::string taxonomy;
    std::filesystem::path file_path;
    float isolation_score;
    float quality_score;
    uint64_t genome_size;
};

class EmbeddingStore {
public:
    explicit EmbeddingStore(int embedding_dim = 512);
    ~EmbeddingStore();

    // Open/close
    bool open(const std::filesystem::path& db_path);
    void close();
    bool is_open() const { return db_ != nullptr; }

    // CRUD operations
    bool insert_embedding(const GenomeEmbedding& emb);
    bool insert_embeddings(const std::vector<GenomeEmbedding>& embeddings);
    bool delete_embedding(const std::string& accession);
    bool has_embedding(const std::string& accession);
    std::optional<GenomeEmbedding> get_embedding(const std::string& accession);

    // Batch operations
    std::vector<std::string> get_embedded_accessions(const std::string& taxonomy = "");
    size_t count_embeddings(const std::string& taxonomy = "");

    // Incremental update support
    std::vector<std::string> get_missing_accessions(
        const std::vector<std::string>& all_accessions);
    std::unordered_set<std::string> get_embedded_set(const std::string& taxonomy = "");

    // Load all embeddings for a taxonomy (for building in-memory index)
    std::vector<GenomeEmbedding> load_embeddings(const std::string& taxonomy = "");

    // Update isolation scores after computing (incremental)
    bool update_isolation_score(const std::string& accession, float score);
    bool update_isolation_scores(const std::vector<std::pair<std::string, float>>& scores);

    // Representative tracking
    bool set_representatives(const std::string& taxonomy,
                            const std::vector<std::string>& rep_accessions);
    bool clear_representatives(const std::string& taxonomy);
    std::vector<std::string> get_representatives(const std::string& taxonomy);
    std::vector<RepresentativeInfo> get_representative_info(const std::string& taxonomy);
    bool is_representative(const std::string& accession);

    // Genome removal (for pruning outdated genomes)
    bool remove_genomes(const std::vector<std::string>& accessions);
    bool remove_taxonomy(const std::string& taxonomy);

    // Vector search (uses HNSW index)
    std::vector<NearestNeighbor> find_nearest(
        const std::vector<float>& query_embedding,
        int k = 10,
        const std::string& taxonomy_filter = "");

    // Find nearest from existing genome
    std::vector<NearestNeighbor> find_nearest_by_accession(
        const std::string& accession,
        int k = 10,
        const std::string& taxonomy_filter = "");

    // Index management
    bool rebuild_index();
    bool compact_index();

    // Stats
    size_t total_embeddings() const;

private:
    int embedding_dim_;
    std::unique_ptr<duckdb::DuckDB> db_;
    std::unique_ptr<duckdb::Connection> conn_;
    std::recursive_mutex mutex_;  // guards ALL conn_ access; recursive for nested calls
    bool vss_loaded_ = false;
    bool index_exists_ = false;

    bool create_schema();
    bool load_vss_extension();
    std::string embedding_to_sql(const std::vector<float>& emb) const;
};

} // namespace derep::db
