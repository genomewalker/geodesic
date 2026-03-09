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
    double genome_size_zscore = 0.0;
    bool nn_outlier = false;
    double kmer_div_zscore = 0.0;
    double margin_to_threshold = 0.0;
    std::string flag_reason;
};

void insert_contamination_candidates(DBManager& db, const std::string& taxonomy,
                                      const std::vector<ContaminationRecord>& candidates);

// Batch-update genome_length and n_contigs after embedding (values not known at insert time).
// accession_sizes: accession → {genome_length_bp, n_contigs}
void update_genome_sizes(DBManager& db,
                          const std::unordered_map<std::string,
                              std::pair<uint64_t, uint32_t>>& accession_sizes);

// ani_map: accession → ANI (0-100). Representatives get 100.0. Absent entries → NULL.
void insert_genomes_derep(DBManager& db, const std::string& taxonomy,
                          const std::vector<std::string>& all_accessions,
                          const std::vector<std::string>& representatives,
                          const std::unordered_map<std::string, double>& ani_map = {});

void mark_jobs_done(DBManager& db, const std::vector<Genome>& genomes);

bool is_taxon_complete(DBManager& db, const std::string& taxonomy);

// One-time startup migration: stage=6 was COMPLETE in older builds; it now means
// EMBEDDING_DONE. Promote stage=6 rows that have results to COMPLETE=7, and warn
// about any that will re-run from the embedding checkpoint.
void migrate_pipeline_stages_v7(DBManager& db);

} // namespace derep::db::ops
