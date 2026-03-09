#include "parallel/taxon_processor.hpp"
#include "core/logging.hpp"
#include "core/geodesic/geodesic.hpp"
#include "core/similarity/skani.hpp"
#include "core/sketch/minhash.hpp"
#include "db/embedding_store.hpp"
#include "db/operations.hpp"

#include <algorithm>
#include <chrono>
#include <deque>
#include <fstream>
#include <memory>
#include <numeric>
#include <spdlog/spdlog.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace derep {
namespace {

// Build quality_score map from the taxon's genomes.
std::unordered_map<std::string, double> build_quality_map(const Taxon& taxon) {
    std::unordered_map<std::string, double> qs;
    qs.reserve(taxon.genomes.size());
    for (const auto& g : taxon.genomes)
        qs[g.file_path.string()] = g.quality_score();
    return qs;
}

// Collect file paths from all genomes in the taxon.
std::vector<std::filesystem::path> collect_paths(const Taxon& taxon) {
    std::vector<std::filesystem::path> paths;
    paths.reserve(taxon.genomes.size());
    for (const auto& g : taxon.genomes)
        paths.push_back(g.file_path);
    return paths;
}

// Collect all accession strings from the taxon.
std::vector<std::string> collect_accessions(const Taxon& taxon) {
    std::vector<std::string> acc;
    acc.reserve(taxon.genomes.size());
    for (const auto& g : taxon.genomes)
        acc.push_back(g.accession);
    return acc;
}

// Build a map from file path string to accession.
std::unordered_map<std::string, std::string> build_path_to_accession(const Taxon& taxon) {
    std::unordered_map<std::string, std::string> m;
    m.reserve(taxon.genomes.size());
    for (const auto& g : taxon.genomes)
        m[g.file_path.string()] = g.accession;
    return m;
}

// Find connected components from precluster edges (paths as node names).
std::vector<std::vector<std::string>> find_precluster_components(
    const std::vector<std::filesystem::path>& all_paths,
    const std::vector<SimilarityEdge>& edges) {

    std::unordered_map<std::string, std::vector<std::string>> adj;
    for (const auto& e : edges) {
        adj[e.source].push_back(e.target);
        adj[e.target].push_back(e.source);
    }

    std::unordered_map<std::string, int> node_idx;
    int n = 0;
    for (const auto& p : all_paths) node_idx[p.string()] = n++;
    for (const auto& e : edges) {
        if (!node_idx.count(e.source)) node_idx[e.source] = n++;
        if (!node_idx.count(e.target)) node_idx[e.target] = n++;
    }

    std::vector<int> component(n, -1);
    int comp_id = 0;
    for (const auto& [node, idx] : node_idx) {
        if (component[idx] >= 0) continue;
        std::deque<std::string> queue = {node};
        component[idx] = comp_id;
        while (!queue.empty()) {
            auto cur = queue.front(); queue.pop_front();
            auto it = adj.find(cur);
            if (it == adj.end()) continue;
            for (const auto& nb : it->second) {
                auto nb_it = node_idx.find(nb);
                if (nb_it == node_idx.end()) continue;
                if (component[nb_it->second] >= 0) continue;
                component[nb_it->second] = comp_id;
                queue.push_back(nb);
            }
        }
        ++comp_id;
    }

    std::vector<std::string> idx_to_str(n);
    for (const auto& [s, i] : node_idx) idx_to_str[i] = s;

    std::vector<std::vector<std::string>> raw(comp_id);
    for (int i = 0; i < n; ++i)
        raw[component[i]].push_back(idx_to_str[i]);

    std::unordered_set<std::string> path_set;
    path_set.reserve(all_paths.size());
    for (const auto& p : all_paths) path_set.insert(p.string());

    std::vector<std::vector<std::string>> result;
    result.reserve(raw.size());
    for (auto& comp : raw) {
        bool has_path = false;
        for (const auto& s : comp) {
            if (path_set.find(s) != path_set.end()) { has_path = true; break; }
        }
        if (has_path) result.push_back(std::move(comp));
    }
    return result;
}

} // anonymous namespace

TaxonResult process_taxon(
    const Taxon& taxon,
    const Config& cfg,
    int thread_budget,
    db::DBManager& db,
    GenomeCache& cache,
    db::EmbeddingStore* emb_store,
    const std::unordered_map<std::string, GuncQuality>* gunc_scores,
    bool in_batch_txn) {
    try {
        const int threads = (thread_budget > 0) ? thread_budget : cfg.threads;
        // -----------------------------------------------------------
        // 1. RESUME CHECK
        // -----------------------------------------------------------
        if (db::ops::is_taxon_complete(db, taxon.taxonomy)) {
            if (is_verbose()) spdlog::info("[{}] already complete, skipping", taxon.taxonomy);
            TaxonResult r;
            r.taxonomy = taxon.taxonomy;
            r.status = TaxonStatus::SUCCESS;
            r.n_genomes = taxon.size();
            return r;
        }

        auto all_accessions = collect_accessions(taxon);

        // -----------------------------------------------------------
        // 2. FIXED TAXA
        // -----------------------------------------------------------
        if (taxon.forced_representative.has_value()) {
            spdlog::info("[{}] fixed representative: {}", taxon.taxonomy,
                         *taxon.forced_representative);

            TaxonResult r;
            r.taxonomy = taxon.taxonomy;
            r.status = TaxonStatus::FIXED;
            r.n_genomes = taxon.size();
            r.n_representatives = 1;
            r.method = "fixed";

            db::ops::insert_result(db, r);
            db::ops::insert_genomes_derep(db, taxon.taxonomy, all_accessions,
                                          {*taxon.forced_representative});
            db::ops::set_pipeline_stage(db, taxon.taxonomy, PipelineStage::COMPLETE);
            return r;
        }

        // -----------------------------------------------------------
        // 3. SINGLETON
        // -----------------------------------------------------------
        if (taxon.is_singleton()) {
            spdlog::info("[{}] singleton", taxon.taxonomy);

            TaxonResult r;
            r.taxonomy = taxon.taxonomy;
            r.status = TaxonStatus::SINGLETON;
            r.n_genomes = 1;
            r.n_representatives = 1;
            r.method = "singleton";

            db::ops::insert_result(db, r);
            db::ops::insert_genomes_derep(db, taxon.taxonomy,
                                          {taxon.genomes[0].accession},
                                          {taxon.genomes[0].accession});
            db::ops::set_pipeline_stage(db, taxon.taxonomy, PipelineStage::COMPLETE);
            return r;
        }

        auto file_paths = collect_paths(taxon);
        auto quality_scores = build_quality_map(taxon);
        auto path_to_accession = build_path_to_accession(taxon);

        // Build path_to_idx once, reuse everywhere
        std::unordered_map<std::string, size_t> path_to_idx;
        path_to_idx.reserve(file_paths.size());
        for (size_t i = 0; i < file_paths.size(); ++i)
            path_to_idx[file_paths[i].string()] = i;

        // -----------------------------------------------------------
        // 4a. Fast path: n <= TINY_N_THRESHOLD — direct OPH Jaccard, skip full pipeline
        // -----------------------------------------------------------
        static constexpr size_t TINY_N_THRESHOLD = 20;
        if (taxon.size() <= TINY_N_THRESHOLD) {
            auto t0 = std::chrono::steady_clock::now();
            const size_t n = taxon.size();

            MinHasher hasher({
                .kmer_size   = cfg.kmer_size,
                .sketch_size = cfg.sketch_size,
                .syncmer_s   = cfg.syncmer_s,
                .seed        = 42
            });

            // Sketch all n genomes with OPH
            std::vector<std::vector<uint32_t>> sigs(n);
            for (size_t i = 0; i < n; ++i)
                sigs[i] = hasher.sketch_oph(file_paths[i], cfg.sketch_size).signature;

            // Compute all pairwise Jaccard
            std::vector<std::vector<double>> jac(n, std::vector<double>(n, 1.0));
            for (size_t i = 0; i < n; ++i)
                for (size_t j = i + 1; j < n; ++j)
                    jac[i][j] = jac[j][i] = GeodesicDerep::refine_jaccard(sigs[i], sigs[j]);

            // Convert ANI threshold to Jaccard threshold
            double ani_threshold_frac = cfg.ani_threshold / 100.0;
            double q = std::pow(ani_threshold_frac, cfg.kmer_size);
            double jaccard_threshold = q / (2.0 - q);

            // Sort genome indices by quality descending
            std::vector<size_t> order(n);
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                return taxon.genomes[a].quality_score() > taxon.genomes[b].quality_score();
            });

            // Greedy quality-sorted cover
            std::vector<size_t> rep_indices;
            std::vector<bool> is_rep(n, false);
            for (size_t idx : order) {
                bool covered = false;
                for (size_t ri : rep_indices) {
                    if (jac[idx][ri] >= jaccard_threshold) { covered = true; break; }
                }
                if (!covered) {
                    rep_indices.push_back(idx);
                    is_rep[idx] = true;
                }
            }

            // Build representatives list
            std::vector<std::string> representatives;
            representatives.reserve(rep_indices.size());
            for (size_t ri : rep_indices)
                representatives.push_back(all_accessions[ri]);

            // Build ani_to_rep_map for non-reps: best Jaccard to any rep
            std::unordered_map<std::string, double> ani_to_rep_map;
            for (size_t i = 0; i < n; ++i) {
                if (is_rep[i]) continue;
                double best_j = 0.0;
                for (size_t ri : rep_indices)
                    best_j = std::max(best_j, jac[i][ri]);
                ani_to_rep_map[all_accessions[i]] =
                    GeodesicDerep::jaccard_to_ani(best_j, cfg.kmer_size);
            }

            // Coverage stats: per-genome best ANI to nearest rep
            std::vector<double> genome_to_rep_ani(n);
            for (size_t i = 0; i < n; ++i) {
                if (is_rep[i]) {
                    genome_to_rep_ani[i] = 100.0;
                } else {
                    double best_j = 0.0;
                    for (size_t ri : rep_indices)
                        best_j = std::max(best_j, jac[i][ri]);
                    genome_to_rep_ani[i] = GeodesicDerep::jaccard_to_ani(best_j, cfg.kmer_size);
                }
            }
            double cov_sum = 0.0, cov_min = 100.0, cov_max = 0.0;
            for (double v : genome_to_rep_ani) {
                cov_sum += v;
                cov_min = std::min(cov_min, v);
                cov_max = std::max(cov_max, v);
            }
            double cov_mean = cov_sum / static_cast<double>(n);

            // Diversity stats: pairwise ANI among reps only
            double div_sum = 0.0, div_min = 100.0, div_max = 0.0;
            int div_pairs = 0;
            for (size_t a = 0; a < rep_indices.size(); ++a) {
                for (size_t b = a + 1; b < rep_indices.size(); ++b) {
                    double ani = GeodesicDerep::jaccard_to_ani(
                        jac[rep_indices[a]][rep_indices[b]], cfg.kmer_size);
                    div_sum += ani;
                    div_min = std::min(div_min, ani);
                    div_max = std::max(div_max, ani);
                    ++div_pairs;
                }
            }
            double div_mean = (div_pairs > 0) ? div_sum / div_pairs : 100.0;
            if (div_pairs == 0) { div_min = 100.0; div_max = 100.0; }

            double runtime = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();

            if (!is_quiet())
                spdlog::info("[{}] {} → {} reps ({:.2f}s) [tiny]", taxon.taxonomy,
                             n, representatives.size(), runtime);

            TaxonResult r;
            r.taxonomy          = taxon.taxonomy;
            r.status            = TaxonStatus::SUCCESS;
            r.n_genomes         = static_cast<int>(n);
            r.n_representatives = static_cast<int>(representatives.size());
            r.method            = "geodesic-tiny";

            TaxonDiversityStats div_stats;
            div_stats.taxonomy          = taxon.taxonomy;
            div_stats.method            = "geodesic-tiny";
            div_stats.n_genomes         = static_cast<int>(n);
            div_stats.n_representatives = static_cast<int>(representatives.size());
            div_stats.reduction_ratio   = 1.0 - static_cast<double>(representatives.size()) /
                                                static_cast<double>(n);
            div_stats.runtime_seconds   = runtime;
            div_stats.coverage_mean_ani = cov_mean;
            div_stats.coverage_min_ani  = cov_min;
            div_stats.coverage_max_ani  = cov_max;
            div_stats.diversity_mean_ani = div_mean;
            div_stats.diversity_min_ani  = div_min;
            div_stats.diversity_max_ani  = div_max;
            div_stats.diversity_n_pairs  = div_pairs;

            auto& conn = db.thread_connection();
            if (!in_batch_txn) conn.Query("BEGIN TRANSACTION");
            db::ops::insert_result(db, r);
            db::ops::insert_genomes_derep(db, taxon.taxonomy, all_accessions,
                                          representatives, ani_to_rep_map);
            db::ops::insert_diversity_stats(db, div_stats);
            db::ops::set_pipeline_stage(db, taxon.taxonomy, PipelineStage::COMPLETE);
            if (!in_batch_txn) conn.Query("COMMIT");
            return r;
        }

        // -----------------------------------------------------------
        // 4. GEODESIC: Physics-inspired paradigm shift (O(n log n))
        // -----------------------------------------------------------
        auto geodesic_start = std::chrono::steady_clock::now();
        if (is_verbose()) spdlog::info("[{}] GEODESIC mode: {} genomes", taxon.taxonomy, taxon.size());

        // Auto-calibrate parameters based on ANI span if enabled
        int kmer_size = cfg.kmer_size;
        int embedding_dim = cfg.embedding_dim;
        int sketch_size = cfg.sketch_size;
        float diversity_threshold = cfg.diversity_threshold;

        double ani_threshold_frac = cfg.ani_threshold / 100.0;

        // Tier selection only: auto-calibrate chooses k/dim/sketch based on spread,
        // but we derive all thresholds from actual data after embedding.
        // Skip on warm (incremental) runs — OPH sigs already encode the original params,
        // and auto_calibrate reads NFS genome files which defeats the caching purpose.
        const bool is_warm_run = emb_store && cfg.incremental &&
                                 emb_store->count_embeddings(taxon.taxonomy) > 0;
        if (cfg.auto_calibrate && file_paths.size() >= 50 && !is_warm_run) {
            auto params = GeodesicDerep::auto_calibrate(
                file_paths, cfg.calibration_pairs, threads);
            kmer_size     = params.kmer_size;
            embedding_dim = params.embedding_dim;
            sketch_size   = params.sketch_size;
            // Thresholds are computed from real data below — not from calibration heuristics.
        }

        // ANI threshold as angular distance — used as the upper cap on diversity_threshold.
        // The actual value is set from the real NN distribution after build_index.
        {
            const double user_ani = cfg.ani_threshold / 100.0;
            double ak = std::pow(user_ani, static_cast<double>(kmer_size));
            double j = std::clamp(ak / (2.0 - ak), 0.0, 1.0);
            diversity_threshold = static_cast<float>(std::acos(j) / M_PI);
        }

        // Both thresholds are placeholders — replaced below using real NN distances.
        float min_rep_distance = diversity_threshold * 0.5f;

        GeodesicDerep::Config gcfg{
            .embedding_dim = embedding_dim,
            .sketch_size = sketch_size,
            .kmer_size = kmer_size,
            .syncmer_s = cfg.syncmer_s,
            .ani_threshold = ani_threshold_frac,
            .hnsw_m = 16,
            .hnsw_ef_construction = 64,
            .hnsw_ef_search = 50,
            .threads = threads,
            .io_threads = cfg.io_threads,
            .calibration_samples = 0,
            .isolation_k = 10,
            .diversity_threshold = diversity_threshold,
            .min_rep_distance = min_rep_distance,
            .max_rep_fraction = cfg.max_rep_fraction,
            .nystrom_diagonal_loading = cfg.nystrom_diagonal_loading,
            .nystrom_degree_normalize = cfg.nystrom_degree_normalize
        };

        GeodesicDerep geodesic(gcfg);

        // Build index with incremental support if embedding store is available
        size_t newly_embedded = 0;
        if (emb_store && cfg.incremental) {
            newly_embedded = geodesic.build_index_incremental(
                file_paths, *emb_store, taxon.taxonomy, quality_scores);
            spdlog::info("[{}] Incremental: {} new embeddings (reused {})",
                         taxon.taxonomy, newly_embedded,
                         file_paths.size() - newly_embedded);
        } else {
            geodesic.build_index(file_paths, quality_scores, emb_store, taxon.taxonomy);
            newly_embedded = file_paths.size();
        }
        // Record permanently failed genome reads in jobs_failed (NFS errors, corrupt files).
        for (const auto& [file_path, reason] : geodesic.failed_reads()) {
            auto it = path_to_accession.find(file_path);
            const std::string accession = (it != path_to_accession.end()) ? it->second : file_path;
            // Escape single quotes in reason to prevent SQL injection.
            std::string safe_reason = reason;
            for (size_t pos = 0; (pos = safe_reason.find('\'', pos)) != std::string::npos; pos += 2)
                safe_reason.replace(pos, 1, "''");
            db.execute("INSERT OR IGNORE INTO jobs_failed (accession, taxonomy, file, reason) "
                       "VALUES ('" + accession + "', '" + taxon.taxonomy + "', '" +
                       file_path + "', '" + safe_reason + "')");
        }

        // Persist genome_length + n_contigs from OPH sketch (input TSV doesn't have them).
        {
            std::unordered_map<std::string, std::pair<uint64_t, uint32_t>> sizes;
            for (const auto& emb : geodesic.embeddings()) {
                if (emb.genome_size == 0) continue;
                auto it = path_to_accession.find(emb.path.string());
                if (it != path_to_accession.end())
                    sizes[it->second] = std::make_pair(emb.genome_size, emb.n_contigs);
            }
            db::ops::update_genome_sizes(db, sizes);
        }

        // Phase 3: single HNSW pass — isolation scores + 1-NN distribution fused.
        auto nn = geodesic.compute_isolation_scores();

        // diversity_threshold = min(θ_ANI, MST_max_edge):
        //   MST_max_edge: maximum edge in the minimum spanning tree of the k-NN graph.
        //   This is the binary_search_filter analog from the original graph-based pipeline:
        //   the minimum θ at which the population k-NN graph becomes connected — the natural
        //   scale at which distinct sub-populations are linked.
        //   θ_ANI is the hard upper cap from --ani-threshold (never merge across species).
        //   Falls back to NN_P95 if MST is unavailable (brute-force small-n path).
        // min_rep_distance = P5 of NN distances: electrostatic merge collapses near-duplicates.
        {
            const float knn_thr = (nn.mst_max_edge > 0.0)
                ? static_cast<float>(nn.mst_max_edge)
                : static_cast<float>(nn.p95);
            diversity_threshold = std::max(1e-6f,
                std::min(diversity_threshold, knn_thr));
            min_rep_distance = std::min(static_cast<float>(nn.p5),
                                        diversity_threshold * 0.5f);

            geodesic.set_diversity_threshold(diversity_threshold);
            geodesic.set_min_rep_distance(min_rep_distance);

            double thr_j   = std::cos(static_cast<double>(diversity_threshold) * M_PI);
            double thr_ani = GeodesicDerep::jaccard_to_ani(std::max(0.0, thr_j), kmer_size);
            spdlog::info("[{}] geodesic: k={} dim={} sketch={} | "
                         "div_thr={:.4f} ({:.1f}% ANI, from MST={:.4f}) | "
                         "min_rep={:.4f} (NN P5={:.4f} P50={:.4f} P95={:.4f})",
                         taxon.taxonomy, kmer_size, embedding_dim, sketch_size,
                         diversity_threshold, thr_ani, nn.mst_max_edge,
                         min_rep_distance, nn.p5, nn.p50, nn.p95);
        }

        // Detect potential contamination before selection
        auto contamination = geodesic.detect_contamination_candidates(cfg.z_threshold);
        std::unordered_set<std::string> contaminated_paths;
        for (const auto& c : contamination) {
            if (is_verbose()) spdlog::warn("[{}] Potential contamination: {} (centroid_dist={:.3f}, "
                         "isolation={:.3f}, kmer_div_z={:.2f}, nn_outlier={})",
                         taxon.taxonomy, c.path.filename().string(),
                         c.centroid_distance, c.isolation_score, c.kmer_div_zscore, c.nn_outlier);
            contaminated_paths.insert(c.path.string());
        }

        // Apply GUNC flags: genomes failing GUNC are excluded from reps.
        // GUNC-only failures (not already in embedding candidates) are collected
        // into gunc_contam_records and merged into contam_records before the single
        // batch insert — never inserted individually to avoid DELETE-per-genome.
        std::vector<db::ops::ContaminationRecord> gunc_contam_records;
        if (gunc_scores && !gunc_scores->empty()) {
            std::unordered_set<std::string> already_flagged;
            for (const auto& c : contamination) {
                auto it = path_to_accession.find(c.path.string());
                if (it != path_to_accession.end()) already_flagged.insert(it->second);
            }
            for (const auto& g : taxon.genomes) {
                auto acc = canonical_accession(g.accession);
                auto git = gunc_scores->find(acc);
                if (git == gunc_scores->end()) continue;
                if (git->second.pass_gunc) continue;
                contaminated_paths.insert(g.file_path.string());
                if (is_verbose())
                    spdlog::warn("[{}] GUNC fail: {} (CSS={:.3f})",
                                 taxon.taxonomy, acc,
                                 git->second.clade_separation_score);
                if (already_flagged.count(acc)) continue;
                // kmer_div_zscore stores CSS; nn_outlier=false (embedding not the source)
                db::ops::ContaminationRecord rec;
                rec.accession          = acc;
                rec.centroid_distance  = 0.0;
                rec.isolation_score    = 0.0;
                rec.anomaly_score      = 0.0;
                rec.nn_outlier         = false;
                rec.kmer_div_zscore    = static_cast<double>(git->second.clade_separation_score);
                rec.genome_size_zscore = 0.0;
                gunc_contam_records.push_back(std::move(rec));
            }
        }

        // Exclude contaminated genomes from rep selection before running FPS
        geodesic.exclude_from_reps(contaminated_paths);

        // Pre-seed fixed representatives (--fixed-reps) before FPS
        if (cfg.fixed_reps_file) {
            static const std::unordered_set<std::string> fixed_accessions = [&]() {
                std::unordered_set<std::string> s;
                std::ifstream f(*cfg.fixed_reps_file);
                std::string line;
                while (std::getline(f, line)) {
                    if (!line.empty()) s.insert(line);
                }
                return s;
            }();

            if (!fixed_accessions.empty()) {
                std::unordered_map<std::string, std::string> accession_to_path;
                for (const auto& [path, acc] : path_to_accession)
                    accession_to_path[acc] = path;

                std::unordered_set<std::string> pinned_paths;
                for (const auto& acc : fixed_accessions) {
                    auto it = accession_to_path.find(acc);
                    if (it != accession_to_path.end())
                        pinned_paths.insert(it->second);
                }
                if (!pinned_paths.empty()) {
                    geodesic.set_pinned_representatives(pinned_paths);
                    spdlog::info("[{}] pinned {} fixed representatives", taxon.taxonomy, pinned_paths.size());
                }
            }
        }

        // Select representatives
        auto edges = geodesic.select_representatives();

        // Collect unique representatives using index bitmaps
        std::vector<bool> covered_bm(file_paths.size(), false);
        std::vector<bool> rep_bm(file_paths.size(), false);
        for (const auto& e : edges) {
            auto src_it = path_to_idx.find(e.source);
            if (src_it != path_to_idx.end())
                covered_bm[src_it->second] = true;
            // Only add as rep if not contaminated (safety net)
            if (contaminated_paths.find(e.target) == contaminated_paths.end()) {
                auto tgt_it = path_to_idx.find(e.target);
                if (tgt_it != path_to_idx.end())
                    rep_bm[tgt_it->second] = true;
            }
        }
        // Genomes not covered elect themselves (if not contaminated)
        for (size_t i = 0; i < file_paths.size(); ++i) {
            if (!covered_bm[i] &&
                contaminated_paths.find(file_paths[i].string()) == contaminated_paths.end()) {
                rep_bm[i] = true;
            }
        }

        // Build rep_set for downstream code that still needs string lookups
        std::unordered_set<std::string> rep_set;
        for (size_t i = 0; i < file_paths.size(); ++i) {
            if (rep_bm[i])
                rep_set.insert(file_paths[i].string());
        }

        std::vector<db::ops::ContaminationRecord> contam_records;
        contam_records.reserve(contamination.size() + gunc_contam_records.size());
        for (const auto& c : contamination) {
            auto it = path_to_accession.find(c.path.string());
            if (it != path_to_accession.end()) {
                db::ops::ContaminationRecord rec;
                rec.accession          = it->second;
                rec.centroid_distance  = static_cast<double>(c.centroid_distance);
                rec.isolation_score    = static_cast<double>(c.isolation_score);
                rec.anomaly_score      = static_cast<double>(c.anomaly_score);
                rec.genome_size_zscore = static_cast<double>(c.genome_size_zscore);
                rec.nn_outlier         = c.nn_outlier;
                rec.kmer_div_zscore    = static_cast<double>(c.kmer_div_zscore);
                rec.margin_to_threshold = static_cast<double>(c.margin_to_threshold);
                rec.flag_reason        = c.flag_reason;
                contam_records.push_back(std::move(rec));
            }
        }
        // Append GUNC-only failures (merged here so a single batch insert handles all)
        for (auto& rec : gunc_contam_records)
            contam_records.push_back(std::move(rec));

        // Convert paths to accessions
        std::vector<std::string> all_representatives;
        for (const auto& path : rep_set) {
            auto it = path_to_accession.find(path);
            if (it != path_to_accession.end()) {
                all_representatives.push_back(it->second);
            }
        }

        // Save representatives to embedding store if available
        if (emb_store) {
            emb_store->set_representatives(taxon.taxonomy, all_representatives);
            if (is_verbose()) spdlog::info("[{}] Marked {} representatives in embedding store",
                         taxon.taxonomy, all_representatives.size());
        }

        auto geodesic_end = std::chrono::steady_clock::now();
        double runtime_secs = std::chrono::duration<double>(geodesic_end - geodesic_start).count();

        if (!is_quiet()) {
            if (contamination.empty()) {
                spdlog::info("[{}] {} → {} reps ({:.1f}s)",
                             taxon.taxonomy, taxon.size(),
                             all_representatives.size(), runtime_secs);
            } else {
                spdlog::info("[{}] {} → {} reps, {} contaminated ({:.1f}s)",
                             taxon.taxonomy, taxon.size(),
                             all_representatives.size(), contamination.size(), runtime_secs);
            }
        }

        // Compute diversity metrics from embeddings
        std::vector<uint64_t> rep_ids;
        const auto& embeds = geodesic.embeddings();
        for (const auto& emb : embeds) {
            if (rep_set.count(emb.path.string())) {
                rep_ids.push_back(emb.genome_id);
            }
        }
        auto div_metrics = geodesic.compute_diversity_metrics(rep_ids);

        // Calibration-free ANI: ANI = (2J/(1+J))^(1/k), J ≈ cos(π*d) from OPH+CountSketch
        auto dist_to_ani = [kmer_size](double dist) -> double {
            if (dist <= 0.0) return 100.0;
            if (dist >= 0.5) return 70.0;
            double cos_sim = std::cos(dist * M_PI);
            if (cos_sim <= 0.0) return 70.0;
            double ratio = 2.0 * cos_sim / (1.0 + cos_sim);
            double ani = std::pow(ratio, 1.0 / kmer_size);
            return std::max(70.0, std::min(100.0, ani * 100.0));
        };

        TaxonDiversityStats div_stats;
        div_stats.taxonomy = taxon.taxonomy;
        div_stats.method = "geodesic";
        div_stats.n_genomes = static_cast<int>(taxon.size());
        div_stats.n_representatives = static_cast<int>(all_representatives.size());
        div_stats.reduction_ratio = 1.0 - static_cast<double>(all_representatives.size()) /
                                          static_cast<double>(taxon.size());
        div_stats.runtime_seconds = runtime_secs;

        // Coverage (convert dist to ANI %; p95 dist = robust worst-case ANI)
        div_stats.coverage_mean_ani = dist_to_ani(div_metrics.coverage_mean_dist);
        div_stats.coverage_min_ani  = dist_to_ani(div_metrics.coverage_p95_dist);  // p95 dist = p5 ANI (robust worst case)
        div_stats.coverage_max_ani  = dist_to_ani(div_metrics.coverage_p5_dist);   // p5 dist = p95 ANI (best covered)
        div_stats.coverage_below_99 = div_metrics.coverage_below_99;
        div_stats.coverage_below_98 = div_metrics.coverage_below_98;
        div_stats.coverage_below_97 = div_metrics.coverage_below_97;
        div_stats.coverage_below_95 = div_metrics.coverage_below_95;

        // Diversity (convert dist to ANI %)
        div_stats.diversity_mean_ani  = dist_to_ani(div_metrics.diversity_mean_dist);
        div_stats.diversity_min_ani   = dist_to_ani(div_metrics.diversity_p95_dist);  // p95 dist = most divergent reps
        div_stats.diversity_max_ani   = dist_to_ani(div_metrics.diversity_p5_dist);   // p5 dist = most similar reps
        div_stats.diversity_ani_range = div_stats.diversity_max_ani - div_stats.diversity_min_ani;
        div_stats.diversity_n_pairs = div_metrics.diversity_n_pairs;
        div_stats.n_contaminated = static_cast<int>(contamination.size());

        TaxonResult r;
        r.taxonomy = taxon.taxonomy;
        r.status = TaxonStatus::SUCCESS;
        r.n_genomes = taxon.size();
        r.n_representatives = static_cast<int>(all_representatives.size());
        r.method = "geodesic";

        // Build ani_to_rep map: weight_raw = ANI fraction (Mash formula applied at emission)
        std::unordered_map<std::string, double> ani_to_rep_map;
        for (const auto& e : edges) {
            auto it = path_to_accession.find(e.source);
            if (it == path_to_accession.end()) continue;
            double ani = std::clamp(static_cast<double>(e.weight_raw) * 100.0, 70.0, 100.0);
            auto& best = ani_to_rep_map[it->second];
            best = std::max(best, ani);
        }

        {
            auto& conn = db.thread_connection();
            conn.Query("BEGIN TRANSACTION");
            if (!contam_records.empty())
                db::ops::insert_contamination_candidates(db, taxon.taxonomy, contam_records);
            db::ops::insert_diversity_stats(db, div_stats);
            db::ops::insert_result(db, r);
            db::ops::insert_genomes_derep(db, taxon.taxonomy, all_accessions,
                                          all_representatives, ani_to_rep_map);
            db::ops::set_pipeline_stage(db, taxon.taxonomy, PipelineStage::COMPLETE);
            conn.Query("COMMIT");
        }

        return r;

    } catch (const std::exception& e) {
        spdlog::error("[{}] failed: {}", taxon.taxonomy, e.what());
        db::ops::set_pipeline_stage(db, taxon.taxonomy,
                                    PipelineStage::NOT_STARTED, "", "",
                                    e.what());
        TaxonResult r;
        r.taxonomy = taxon.taxonomy;
        r.status = TaxonStatus::FAILED;
        r.n_genomes = taxon.size();
        r.error_message = e.what();
        return r;
    }
}

std::vector<TaxonResult> process_tiny_batch(
    const std::vector<const Taxon*>& taxa,
    const Config& cfg,
    db::DBManager& db,
    GenomeCache& cache) {
    std::vector<TaxonResult> results;
    results.reserve(taxa.size());
    auto& conn = db.thread_connection();
    conn.Query("BEGIN TRANSACTION");
    try {
        for (const Taxon* t : taxa)
            results.push_back(process_taxon(*t, cfg, 1, db, cache, nullptr, nullptr, /*in_batch_txn=*/true));
        conn.Query("COMMIT");
    } catch (...) {
        conn.Query("ROLLBACK");
        throw;
    }
    return results;
}

} // namespace derep
