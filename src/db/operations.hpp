#pragma once
#include "db/db_manager.hpp"
#include "core/types.hpp"
#include <string>
#include <unordered_map>
#include <vector>

namespace derep::db::ops {

void insert_genomes(DBManager& db, const std::vector<Genome>& genomes);

void insert_taxon(DBManager& db, const std::string& taxonomy, int count);

// Bulk-insert all taxa and genomes in a single transaction with one Appender each.
// ~100x faster than calling insert_taxon/insert_genomes per taxon for large datasets.
void insert_taxa_and_genomes_bulk(DBManager& db, const std::vector<Taxon>& taxa);

void set_pipeline_stage(DBManager& db, const std::string& taxonomy, PipelineStage stage,
                        const std::string& precluster_method = "",
                        const std::string& similarity_method = "",
                        const std::string& error_message = "");

PipelineStage get_pipeline_stage(DBManager& db, const std::string& taxonomy);

void insert_precluster_edges(DBManager& db, const std::string& taxonomy,
                             const std::vector<SimilarityEdge>& edges);

void insert_similarity_edges(DBManager& db, const std::string& taxonomy,
                              int cluster_id, const std::vector<SimilarityEdge>& edges);

void mark_similarity_cluster_done(DBManager& db, const std::string& taxonomy, int cluster_id);

bool is_similarity_cluster_done(DBManager& db, const std::string& taxonomy, int cluster_id);

std::vector<SimilarityEdge> load_similarity_edges(DBManager& db,
                                                   const std::string& taxonomy, int cluster_id);

void insert_partitions(DBManager& db, const std::string& taxonomy, int cluster_id,
                       const std::unordered_map<std::string, int>& partition);

void insert_result(DBManager& db, const TaxonResult& result);

void insert_stats(DBManager& db, const std::vector<RepresentativeStats>& stats);

void insert_diversity_stats(DBManager& db, const TaxonDiversityStats& stats);

struct ContaminationRecord {
    std::string accession;
    double centroid_distance;
    double isolation_score;
    double anomaly_score;
};

void insert_contamination_candidates(DBManager& db, const std::string& taxonomy,
                                      const std::vector<ContaminationRecord>& candidates);

// ani_map: accession → ANI (0-100). Representatives get 100.0. Absent entries → NULL.
void insert_genomes_derep(DBManager& db, const std::string& taxonomy,
                          const std::vector<std::string>& all_accessions,
                          const std::vector<std::string>& representatives,
                          const std::unordered_map<std::string, double>& ani_map = {});

void mark_jobs_done(DBManager& db, const std::vector<Genome>& genomes);

bool is_taxon_complete(DBManager& db, const std::string& taxonomy);

} // namespace derep::db::ops
