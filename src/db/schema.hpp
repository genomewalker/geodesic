#pragma once
#include <array>
#include <string>
#include <string_view>
#include <unordered_set>
#include <duckdb.hpp>

namespace derep::db::schema {

inline constexpr std::string_view kCreateTaxa = R"sql(
CREATE TABLE IF NOT EXISTS taxa (
    taxonomy VARCHAR PRIMARY KEY,
    count INTEGER NOT NULL
);
)sql";

inline constexpr std::string_view kCreateGenomes = R"sql(
CREATE TABLE IF NOT EXISTS genomes (
    accession VARCHAR PRIMARY KEY,
    taxonomy VARCHAR NOT NULL,
    file VARCHAR NOT NULL,
    genome_length BIGINT DEFAULT 0,
    completeness DOUBLE,
    contamination DOUBLE,
    quality_score DOUBLE
);
CREATE INDEX IF NOT EXISTS idx_genomes_taxonomy ON genomes(taxonomy);
)sql";

inline constexpr std::string_view kCreateGenomesDerep = R"sql(
CREATE TABLE IF NOT EXISTS genomes_derep (
    accession VARCHAR PRIMARY KEY,
    taxonomy VARCHAR NOT NULL,
    representative BOOLEAN NOT NULL,
    derep BOOLEAN NOT NULL DEFAULT TRUE,
    ani_to_rep DOUBLE
);
CREATE INDEX IF NOT EXISTS idx_genomes_derep_rep ON genomes_derep(representative);
CREATE INDEX IF NOT EXISTS idx_genomes_derep_taxonomy ON genomes_derep(taxonomy);
)sql";

inline constexpr std::string_view kCreateResults = R"sql(
CREATE TABLE IF NOT EXISTS results (
    taxonomy VARCHAR NOT NULL,
    method VARCHAR NOT NULL,
    weight DOUBLE,
    communities INTEGER,
    n_genomes INTEGER,
    n_genomes_derep INTEGER,
    PRIMARY KEY (taxonomy, method)
);
)sql";

inline constexpr std::string_view kCreateStats = R"sql(
CREATE TABLE IF NOT EXISTS stats (
    taxonomy VARCHAR NOT NULL,
    representative VARCHAR PRIMARY KEY,
    n_nodes INTEGER,
    n_nodes_selected INTEGER,
    n_nodes_discarded INTEGER,
    graph_avg_weight DOUBLE,
    graph_sd_weight DOUBLE,
    graph_avg_weight_raw DOUBLE,
    graph_sd_weight_raw DOUBLE,
    subgraph_selected_avg_weight DOUBLE,
    subgraph_selected_sd_weight DOUBLE,
    subgraph_selected_avg_weight_raw DOUBLE,
    subgraph_selected_sd_weight_raw DOUBLE,
    subgraph_discarded_avg_weight DOUBLE,
    subgraph_discarded_sd_weight DOUBLE,
    subgraph_discarded_avg_weight_raw DOUBLE,
    subgraph_discarded_sd_weight_raw DOUBLE
);
CREATE INDEX IF NOT EXISTS idx_stats_taxonomy ON stats(taxonomy);
)sql";

inline constexpr std::string_view kCreateJobsDone = R"sql(
CREATE TABLE IF NOT EXISTS jobs_done (
    accession VARCHAR PRIMARY KEY,
    taxonomy VARCHAR NOT NULL,
    file VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_done_taxonomy ON jobs_done(taxonomy);
)sql";

inline constexpr std::string_view kCreateJobsFailed = R"sql(
CREATE TABLE IF NOT EXISTS jobs_failed (
    accession VARCHAR PRIMARY KEY,
    taxonomy VARCHAR NOT NULL,
    file VARCHAR NOT NULL,
    reason VARCHAR
);
CREATE INDEX IF NOT EXISTS idx_jobs_failed_taxonomy ON jobs_failed(taxonomy);
)sql";

inline constexpr std::string_view kCreatePipelineStages = R"sql(
CREATE TABLE IF NOT EXISTS pipeline_stages (
    taxonomy VARCHAR PRIMARY KEY,
    stage INTEGER NOT NULL DEFAULT 0,
    last_updated TIMESTAMP DEFAULT current_timestamp,
    precluster_method VARCHAR,
    similarity_method VARCHAR,
    error_message VARCHAR
);
)sql";

inline constexpr std::string_view kCreatePreclusterEdges = R"sql(
CREATE TABLE IF NOT EXISTS precluster_edges (
    taxonomy VARCHAR NOT NULL,
    source VARCHAR NOT NULL,
    target VARCHAR NOT NULL,
    distance DOUBLE NOT NULL,
    weight DOUBLE NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_precluster_taxonomy ON precluster_edges(taxonomy);
)sql";

inline constexpr std::string_view kCreateSimilarityEdges = R"sql(
CREATE TABLE IF NOT EXISTS similarity_edges (
    taxonomy VARCHAR NOT NULL,
    cluster_id INTEGER NOT NULL,
    source VARCHAR NOT NULL,
    target VARCHAR NOT NULL,
    weight_raw DOUBLE NOT NULL,
    aln_frac DOUBLE NOT NULL,
    weight DOUBLE NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_similarity_taxonomy ON similarity_edges(taxonomy);
)sql";

inline constexpr std::string_view kCreatePartitions = R"sql(
CREATE TABLE IF NOT EXISTS partitions (
    taxonomy VARCHAR NOT NULL,
    cluster_id INTEGER NOT NULL,
    node VARCHAR NOT NULL,
    community_id INTEGER NOT NULL,
    PRIMARY KEY (taxonomy, cluster_id, node)
);
)sql";

inline constexpr std::string_view kCreateCheckm2Quality = R"sql(
CREATE TABLE IF NOT EXISTS checkm2_quality (
    accession VARCHAR PRIMARY KEY,
    completeness DOUBLE NOT NULL,
    contamination DOUBLE NOT NULL,
    quality_score DOUBLE NOT NULL
);
)sql";

inline constexpr std::string_view kCreateFixedTaxa = R"sql(
CREATE TABLE IF NOT EXISTS fixed_taxa (
    taxonomy VARCHAR PRIMARY KEY,
    accession VARCHAR NOT NULL
);
)sql";

inline constexpr std::string_view kCreateSimilarityClustersDone = R"sql(
CREATE TABLE IF NOT EXISTS similarity_clusters_done (
    taxonomy VARCHAR NOT NULL,
    cluster_id INTEGER NOT NULL,
    PRIMARY KEY (taxonomy, cluster_id)
);
)sql";

inline constexpr std::string_view kCreateDiversityStats = R"sql(
CREATE TABLE IF NOT EXISTS diversity_stats (
    taxonomy VARCHAR PRIMARY KEY,
    method VARCHAR NOT NULL,
    n_genomes INTEGER NOT NULL,
    n_representatives INTEGER NOT NULL,
    reduction_ratio DOUBLE NOT NULL,
    runtime_seconds DOUBLE NOT NULL,
    coverage_mean_ani DOUBLE,
    coverage_min_ani DOUBLE,
    coverage_max_ani DOUBLE,
    coverage_below_99 INTEGER,
    coverage_below_98 INTEGER,
    coverage_below_97 INTEGER,
    coverage_below_95 INTEGER,
    diversity_mean_ani DOUBLE,
    diversity_min_ani DOUBLE,
    diversity_max_ani DOUBLE,
    diversity_ani_range DOUBLE,
    diversity_n_pairs INTEGER,
    n_contaminated INTEGER DEFAULT 0
);
)sql";

inline constexpr std::string_view kCreateContaminationCandidates = R"sql(
CREATE TABLE IF NOT EXISTS contamination_candidates (
    taxonomy VARCHAR NOT NULL,
    accession VARCHAR NOT NULL,
    centroid_distance DOUBLE NOT NULL,
    isolation_score DOUBLE NOT NULL,
    anomaly_score DOUBLE NOT NULL,
    PRIMARY KEY (taxonomy, accession)
);
CREATE INDEX IF NOT EXISTS idx_contamination_taxonomy ON contamination_candidates(taxonomy);
)sql";

inline constexpr std::array<std::string_view, 16> kSchemaStatements = {
    kCreateTaxa,
    kCreateGenomes,
    kCreateGenomesDerep,
    kCreateResults,
    kCreateStats,
    kCreateJobsDone,
    kCreateJobsFailed,
    kCreatePipelineStages,
    kCreatePreclusterEdges,
    kCreateSimilarityEdges,
    kCreatePartitions,
    kCreateCheckm2Quality,
    kCreateFixedTaxa,
    kCreateSimilarityClustersDone,
    kCreateDiversityStats,
    kCreateContaminationCandidates
};

inline void create_all(duckdb::Connection& conn) {
    for (const auto& sql : kSchemaStatements) {
        auto result = conn.Query(std::string(sql));
        if (result->HasError())
            throw std::runtime_error("Schema creation failed: " + result->GetError());
    }
}

inline void validate_required_objects(duckdb::Connection& conn) {
    static constexpr std::array<std::string_view, 16> required_tables = {
        "taxa", "genomes", "genomes_derep", "results", "stats",
        "jobs_done", "jobs_failed", "pipeline_stages", "precluster_edges",
        "similarity_edges", "partitions", "checkm2_quality", "fixed_taxa",
        "similarity_clusters_done", "diversity_stats", "contamination_candidates"
    };
    auto result = conn.Query("SHOW TABLES;");
    if (result->HasError()) throw std::runtime_error(result->GetError());
    std::unordered_set<std::string> existing;
    for (auto& row : *result) existing.insert(row.GetValue<std::string>(0));
    for (const auto& tbl : required_tables) {
        if (!existing.count(std::string(tbl)))
            throw std::runtime_error("Missing table: " + std::string(tbl));
    }
}

// Apply schema migrations to existing databases (idempotent)
inline void migrate(duckdb::Connection& conn) {
    static constexpr std::array<std::string_view, 3> migrations = {
        "ALTER TABLE diversity_stats ADD COLUMN IF NOT EXISTS n_contaminated INTEGER DEFAULT 0",
        "ALTER TABLE genomes_derep ADD COLUMN IF NOT EXISTS taxonomy VARCHAR DEFAULT ''",
        "ALTER TABLE genomes_derep ADD COLUMN IF NOT EXISTS ani_to_rep DOUBLE",
    };
    for (const auto& sql : migrations) {
        auto r = conn.Query(std::string(sql));
        if (r->HasError())
            throw std::runtime_error("Migration failed: " + r->GetError());
    }
    // Create new tables introduced after initial schema (IF NOT EXISTS handles idempotency)
    auto r = conn.Query(std::string(kCreateContaminationCandidates));
    if (r->HasError())
        throw std::runtime_error("Migration failed: " + r->GetError());
}

inline void prune_intermediate_tables(duckdb::Connection& conn) {
    for (const auto& tbl : {"precluster_edges", "similarity_edges",
                             "partitions", "similarity_clusters_done"}) {
        conn.Query("DELETE FROM " + std::string(tbl) + ";");
    }
}

} // namespace derep::db::schema
