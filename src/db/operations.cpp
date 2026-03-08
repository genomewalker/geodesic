#include "db/operations.hpp"
#include <duckdb.hpp>
#include <spdlog/spdlog.h>
#include <unordered_set>

namespace derep::db::ops {

void insert_genomes(DBManager& db, const std::vector<Genome>& genomes) {
    auto& conn = db.thread_connection();
    duckdb::Appender appender(conn, "genomes");
    for (const auto& g : genomes) {
        appender.BeginRow();
        appender.Append<duckdb::string_t>(duckdb::string_t(g.accession));
        appender.Append<duckdb::string_t>(duckdb::string_t(g.taxonomy));
        appender.Append<duckdb::string_t>(duckdb::string_t(g.file_path.string()));
        appender.Append<int64_t>(static_cast<int64_t>(g.genome_length));
        if (g.completeness)
            appender.Append<double>(*g.completeness);
        else
            appender.AppendDefault();
        if (g.contamination)
            appender.Append<double>(*g.contamination);
        else
            appender.AppendDefault();
        appender.Append<double>(g.quality_score());
        appender.Append<int32_t>(0);  // n_contigs: filled after sketching
        appender.EndRow();
    }
    appender.Close();
}

void insert_taxa_and_genomes_bulk(DBManager& db, const std::vector<Taxon>& taxa) {
    auto& conn = db.thread_connection();
    conn.Query("BEGIN TRANSACTION");

    // Single Appender for all taxa rows
    {
        duckdb::Appender taxa_app(conn, "taxa");
        for (const auto& t : taxa) {
            taxa_app.BeginRow();
            taxa_app.Append<duckdb::string_t>(duckdb::string_t(t.taxonomy));
            taxa_app.Append<int32_t>(static_cast<int32_t>(t.size()));
            taxa_app.EndRow();
        }
        taxa_app.Close();
    }

    // Single Appender for all genome rows across all taxa
    {
        duckdb::Appender genome_app(conn, "genomes");
        for (const auto& t : taxa) {
            for (const auto& g : t.genomes) {
                genome_app.BeginRow();
                genome_app.Append<duckdb::string_t>(duckdb::string_t(g.accession));
                genome_app.Append<duckdb::string_t>(duckdb::string_t(g.taxonomy));
                genome_app.Append<duckdb::string_t>(duckdb::string_t(g.file_path.string()));
                genome_app.Append<int64_t>(static_cast<int64_t>(g.genome_length));
                if (g.completeness)
                    genome_app.Append<double>(*g.completeness);
                else
                    genome_app.AppendDefault();
                if (g.contamination)
                    genome_app.Append<double>(*g.contamination);
                else
                    genome_app.AppendDefault();
                genome_app.Append<double>(g.quality_score());
                genome_app.Append<int32_t>(0);  // n_contigs: filled after sketching
                genome_app.EndRow();
            }
        }
        genome_app.Close();
    }

    conn.Query("COMMIT");
}

void insert_taxon(DBManager& db, const std::string& taxonomy, int count) {
    auto& conn = db.thread_connection();
    auto result = conn.Query(
        "INSERT OR REPLACE INTO taxa (taxonomy, count) VALUES ($1, $2)",
        taxonomy, count);
    if (result->HasError())
        throw DbError("insert_taxon: " + result->GetError());
}

void set_pipeline_stage(DBManager& db, const std::string& taxonomy, PipelineStage stage,
                        const std::string& precluster_method,
                        const std::string& similarity_method,
                        const std::string& error_message) {
    auto& conn = db.thread_connection();
    auto result = conn.Query(
        "INSERT OR REPLACE INTO pipeline_stages "
        "(taxonomy, stage, last_updated, precluster_method, similarity_method, error_message) "
        "VALUES ($1, $2, current_timestamp, $3, $4, $5)",
        taxonomy, static_cast<int>(stage),
        precluster_method.empty() ? duckdb::Value() : duckdb::Value(precluster_method),
        similarity_method.empty() ? duckdb::Value() : duckdb::Value(similarity_method),
        error_message.empty() ? duckdb::Value() : duckdb::Value(error_message));
    if (result->HasError())
        throw DbError("set_pipeline_stage: " + result->GetError());
}

PipelineStage get_pipeline_stage(DBManager& db, const std::string& taxonomy) {
    auto& conn = db.thread_connection();
    auto result = conn.Query(
        "SELECT stage FROM pipeline_stages WHERE taxonomy = $1", taxonomy);
    if (result->HasError())
        throw DbError("get_pipeline_stage: " + result->GetError());
    auto chunk = result->Fetch();
    if (!chunk || chunk->size() == 0)
        return PipelineStage::NOT_STARTED;
    return static_cast<PipelineStage>(chunk->GetValue(0, 0).GetValue<int32_t>());
}

void insert_precluster_edges(DBManager& db, const std::string& taxonomy,
                             const std::vector<SimilarityEdge>& edges) {
    auto& conn = db.thread_connection();
    conn.Query("DELETE FROM precluster_edges WHERE taxonomy = $1", taxonomy);
    duckdb::Appender appender(conn, "precluster_edges");
    for (const auto& e : edges) {
        appender.BeginRow();
        appender.Append<duckdb::string_t>(duckdb::string_t(taxonomy));
        appender.Append<duckdb::string_t>(duckdb::string_t(e.source));
        appender.Append<duckdb::string_t>(duckdb::string_t(e.target));
        appender.Append<double>(e.weight_raw);
        appender.Append<double>(e.weight);
        appender.EndRow();
    }
    appender.Close();
}

void insert_similarity_edges(DBManager& db, const std::string& taxonomy,
                              int cluster_id, const std::vector<SimilarityEdge>& edges) {
    auto& conn = db.thread_connection();
    conn.Query(
        "DELETE FROM similarity_edges WHERE taxonomy = $1 AND cluster_id = $2",
        taxonomy, cluster_id);
    duckdb::Appender appender(conn, "similarity_edges");
    for (const auto& e : edges) {
        appender.BeginRow();
        appender.Append<duckdb::string_t>(duckdb::string_t(taxonomy));
        appender.Append<int32_t>(cluster_id);
        appender.Append<duckdb::string_t>(duckdb::string_t(e.source));
        appender.Append<duckdb::string_t>(duckdb::string_t(e.target));
        appender.Append<double>(e.weight_raw);
        appender.Append<double>(e.aln_frac);
        appender.Append<double>(e.weight);
        appender.EndRow();
    }
    appender.Close();
}

void mark_similarity_cluster_done(DBManager& db, const std::string& taxonomy, int cluster_id) {
    auto& conn = db.thread_connection();
    auto result = conn.Query(
        "INSERT OR IGNORE INTO similarity_clusters_done (taxonomy, cluster_id) VALUES ($1, $2)",
        taxonomy, cluster_id);
    if (result->HasError())
        throw DbError("mark_similarity_cluster_done: " + result->GetError());
}

bool is_similarity_cluster_done(DBManager& db, const std::string& taxonomy, int cluster_id) {
    auto& conn = db.thread_connection();
    auto result = conn.Query(
        "SELECT 1 FROM similarity_clusters_done WHERE taxonomy = $1 AND cluster_id = $2",
        taxonomy, cluster_id);
    if (result->HasError())
        throw DbError("is_similarity_cluster_done: " + result->GetError());
    auto chunk = result->Fetch();
    return chunk && chunk->size() > 0;
}

std::vector<SimilarityEdge> load_similarity_edges(DBManager& db,
                                                   const std::string& taxonomy, int cluster_id) {
    auto& conn = db.thread_connection();
    auto result = conn.Query(
        "SELECT source, target, weight_raw, aln_frac, weight "
        "FROM similarity_edges WHERE taxonomy = $1 AND cluster_id = $2",
        taxonomy, cluster_id);
    if (result->HasError())
        throw DbError("load_similarity_edges: " + result->GetError());

    std::vector<SimilarityEdge> edges;
    while (auto chunk = result->Fetch()) {
        for (duckdb::idx_t r = 0; r < chunk->size(); ++r) {
            SimilarityEdge e;
            e.source    = chunk->GetValue(0, r).GetValue<std::string>();
            e.target    = chunk->GetValue(1, r).GetValue<std::string>();
            e.weight_raw = chunk->GetValue(2, r).GetValue<double>();
            e.aln_frac  = chunk->GetValue(3, r).GetValue<double>();
            e.weight    = chunk->GetValue(4, r).GetValue<double>();
            edges.push_back(std::move(e));
        }
    }
    return edges;
}

void insert_partitions(DBManager& db, const std::string& taxonomy, int cluster_id,
                       const std::unordered_map<std::string, int>& partition) {
    auto& conn = db.thread_connection();
    conn.Query(
        "DELETE FROM partitions WHERE taxonomy = $1 AND cluster_id = $2",
        taxonomy, cluster_id);
    duckdb::Appender appender(conn, "partitions");
    for (const auto& [node, community_id] : partition) {
        appender.BeginRow();
        appender.Append<duckdb::string_t>(duckdb::string_t(taxonomy));
        appender.Append<int32_t>(cluster_id);
        appender.Append<duckdb::string_t>(duckdb::string_t(node));
        appender.Append<int32_t>(community_id);
        appender.EndRow();
    }
    appender.Close();
}

void insert_result(DBManager& db, const TaxonResult& result) {
    auto& conn = db.thread_connection();
    auto weight = duckdb::Value();  // NULL
    auto r = conn.Query(
        "INSERT OR REPLACE INTO results "
        "(taxonomy, method, weight, communities, n_genomes, n_genomes_derep) "
        "VALUES ($1, $2, $3, $4, $5, $6)",
        result.taxonomy, result.method, weight,
        static_cast<int>(result.n_communities),
        static_cast<int>(result.n_genomes),
        static_cast<int>(result.n_representatives));
    if (r->HasError())
        throw DbError("insert_result: " + r->GetError());
}

static void append_optional_double(duckdb::Appender& appender, const std::optional<double>& val) {
    if (val)
        appender.Append<double>(*val);
    else
        appender.AppendDefault();
}

void insert_stats(DBManager& db, const std::vector<RepresentativeStats>& stats) {
    if (stats.empty()) return;
    auto& conn = db.thread_connection();
    conn.Query("DELETE FROM stats WHERE taxonomy = $1", stats[0].taxonomy);
    duckdb::Appender appender(conn, "stats");
    for (const auto& s : stats) {
        appender.BeginRow();
        appender.Append<duckdb::string_t>(duckdb::string_t(s.taxonomy));
        appender.Append<duckdb::string_t>(duckdb::string_t(s.representative));
        appender.Append<int32_t>(s.n_nodes);
        appender.Append<int32_t>(s.n_nodes_selected);
        appender.Append<int32_t>(s.n_nodes_discarded);
        append_optional_double(appender, s.graph_avg_weight);
        append_optional_double(appender, s.graph_sd_weight);
        append_optional_double(appender, s.graph_avg_weight_raw);
        append_optional_double(appender, s.graph_sd_weight_raw);
        append_optional_double(appender, s.subgraph_selected_avg_weight);
        append_optional_double(appender, s.subgraph_selected_sd_weight);
        append_optional_double(appender, s.subgraph_selected_avg_weight_raw);
        append_optional_double(appender, s.subgraph_selected_sd_weight_raw);
        append_optional_double(appender, s.subgraph_discarded_avg_weight);
        append_optional_double(appender, s.subgraph_discarded_sd_weight);
        append_optional_double(appender, s.subgraph_discarded_avg_weight_raw);
        append_optional_double(appender, s.subgraph_discarded_sd_weight_raw);
        appender.EndRow();
    }
    appender.Close();
}

void insert_diversity_stats(DBManager& db, const TaxonDiversityStats& s) {
    auto& conn = db.thread_connection();
    auto result = conn.Query(
        "INSERT OR REPLACE INTO diversity_stats "
        "(taxonomy, method, n_genomes, n_representatives, reduction_ratio, runtime_seconds, "
        "coverage_mean_ani, coverage_min_ani, coverage_max_ani, "
        "coverage_below_99, coverage_below_98, coverage_below_97, coverage_below_95, "
        "diversity_mean_ani, diversity_min_ani, diversity_max_ani, diversity_ani_range, diversity_n_pairs, "
        "n_contaminated) "
        "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)",
        s.taxonomy, s.method,
        s.n_genomes, s.n_representatives, s.reduction_ratio, s.runtime_seconds,
        s.coverage_mean_ani, s.coverage_min_ani, s.coverage_max_ani,
        s.coverage_below_99, s.coverage_below_98, s.coverage_below_97, s.coverage_below_95,
        s.diversity_mean_ani, s.diversity_min_ani, s.diversity_max_ani,
        s.diversity_ani_range, s.diversity_n_pairs, s.n_contaminated);
    if (result->HasError())
        throw DbError("insert_diversity_stats: " + result->GetError());
}

void insert_contamination_candidates(DBManager& db, const std::string& taxonomy,
                                      const std::vector<ContaminationRecord>& candidates) {
    if (candidates.empty()) return;
    auto& conn = db.thread_connection();
    conn.Query("DELETE FROM contamination_candidates WHERE taxonomy = $1", taxonomy);
    duckdb::Appender appender(conn, "contamination_candidates");
    for (const auto& c : candidates) {
        appender.BeginRow();
        appender.Append<duckdb::string_t>(duckdb::string_t(taxonomy));
        appender.Append<duckdb::string_t>(duckdb::string_t(c.accession));
        appender.Append<double>(c.centroid_distance);
        appender.Append<double>(c.isolation_score);
        appender.Append<double>(c.anomaly_score);
        appender.Append<double>(c.genome_size_zscore);
        appender.Append<bool>(c.nn_outlier);
        appender.Append<double>(c.kmer_div_zscore);
        appender.EndRow();
    }
    appender.Close();
}

void insert_genomes_derep(DBManager& db, const std::string& taxonomy,
                          const std::vector<std::string>& all_accessions,
                          const std::vector<std::string>& representatives,
                          const std::unordered_map<std::string, double>& ani_map) {
    std::unordered_set<std::string> rep_set(representatives.begin(), representatives.end());
    auto& conn = db.thread_connection();
    conn.Query("DELETE FROM genomes_derep WHERE taxonomy = $1", taxonomy);
    duckdb::Appender appender(conn, "genomes_derep");
    for (const auto& acc : all_accessions) {
        bool is_rep = rep_set.count(acc) > 0;
        appender.BeginRow();
        appender.Append<duckdb::string_t>(duckdb::string_t(acc));
        appender.Append<duckdb::string_t>(duckdb::string_t(taxonomy));
        appender.Append<bool>(is_rep);
        appender.Append<bool>(true);
        // ani_to_rep: 100.0 for representatives, from map for non-reps, NULL if absent
        if (is_rep) {
            appender.Append<double>(100.0);
        } else {
            auto it = ani_map.find(acc);
            if (it != ani_map.end())
                appender.Append<double>(it->second);
            else
                appender.AppendDefault();
        }
        appender.EndRow();
    }
    appender.Close();
}

void mark_jobs_done(DBManager& db, const std::vector<Genome>& genomes) {
    if (genomes.empty()) return;
    auto& conn = db.thread_connection();
    conn.Query("DELETE FROM jobs_done WHERE taxonomy = $1", genomes[0].taxonomy);
    duckdb::Appender appender(conn, "jobs_done");
    for (const auto& g : genomes) {
        appender.BeginRow();
        appender.Append<duckdb::string_t>(duckdb::string_t(g.accession));
        appender.Append<duckdb::string_t>(duckdb::string_t(g.taxonomy));
        appender.Append<duckdb::string_t>(duckdb::string_t(g.file_path.string()));
        appender.EndRow();
    }
    appender.Close();
}

bool is_taxon_complete(DBManager& db, const std::string& taxonomy) {
    return get_pipeline_stage(db, taxonomy) == PipelineStage::COMPLETE;
}

void update_genome_sizes(DBManager& db,
                          const std::unordered_map<std::string,
                              std::pair<uint64_t, uint32_t>>& accession_sizes) {
    if (accession_sizes.empty()) return;
    auto& conn = db.thread_connection();
    // Bulk UPDATE via temp table + Appender — O(1) SQL vs O(n) prepared executions.
    conn.Query("CREATE TEMP TABLE IF NOT EXISTS _tmp_genome_sizes "
               "(accession VARCHAR, genome_length BIGINT, n_contigs INTEGER)");
    conn.Query("DELETE FROM _tmp_genome_sizes");
    {
        duckdb::Appender app(conn, "_tmp_genome_sizes");
        for (const auto& [acc, sz] : accession_sizes) {
            app.BeginRow();
            app.Append<duckdb::string_t>(duckdb::string_t(acc));
            app.Append<int64_t>(static_cast<int64_t>(sz.first));
            app.Append<int32_t>(static_cast<int32_t>(sz.second));
            app.EndRow();
        }
        app.Close();
    }
    conn.Query("UPDATE genomes SET genome_length = s.genome_length, n_contigs = s.n_contigs "
               "FROM _tmp_genome_sizes s WHERE genomes.accession = s.accession");
    conn.Query("DROP TABLE _tmp_genome_sizes");
}

} // namespace derep::db::ops
