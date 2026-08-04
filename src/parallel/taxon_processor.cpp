#include "parallel/taxon_processor.hpp"
#include "state/run_state.hpp"
#include "core/logging.hpp"
#include "core/geodesic/geodesic.hpp"
#include "core/pack_reader.hpp"
#include "db/grd/grd_writer.hpp"
#include <genopack/archive.hpp>

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

// Build quality_score map from the taxon's genomes, keyed by accession.
// If gpk_reader has QUAL section data, it overrides the TSV-derived score.
std::unordered_map<std::string, double> build_quality_map(
    const Taxon& taxon, IPackReader* gpk_reader = nullptr)
{
    std::unordered_map<std::string, double> qs;
    qs.reserve(taxon.genomes.size());
    for (const auto& g : taxon.genomes) {
        double score = g.quality_score();
        if (gpk_reader) {
            auto q = gpk_reader->qual_score_for_accession(g.accession);
            if (q) score = *q;
        }
        qs[g.accession] = score;
    }
    return qs;
}

// Collect all accession strings from the taxon.
std::vector<std::string> collect_accessions(const Taxon& taxon) {
    std::vector<std::string> acc;
    acc.reserve(taxon.genomes.size());
    for (const auto& g : taxon.genomes)
        acc.push_back(g.accession);
    return acc;
}

} // anonymous namespace

TaxonResult process_taxon(
    const Taxon& taxon,
    const Config& cfg,
    int thread_budget,
    const std::unordered_map<std::string, GuncQuality>* gunc_scores,
    IPackReader* gpk_reader,
    RunState* run_state,
    grd::GrdWriter* grd_writer,
    std::function<void()> on_serial_phase,
    BS::thread_pool* pool) {
    try {
        const int threads = (thread_budget > 0) ? thread_budget : cfg.threads;

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

            if (run_state) {
                TaxonOutput out;
                out.result = r;
                out.all_accessions = all_accessions;
                out.representatives = {*taxon.forced_representative};
                out.rep_embeddings.assign(1, std::vector<float>(cfg.embedding_dim, 0.0f));
                out.rep_cluster_size[*taxon.forced_representative] =
                    static_cast<uint32_t>(all_accessions.size());
                for (const auto& a : all_accessions)
                    if (a != *taxon.forced_representative)
                        out.member_to_rep[a] = *taxon.forced_representative;
                out.sketch_kmer_used = cfg.kmer_size;
                run_state->push(std::move(out));
            }
            return r;
        }

        // -----------------------------------------------------------
        // 3. SINGLETON
        // -----------------------------------------------------------
        if (taxon.is_singleton()) {
            if (is_verbose()) spdlog::info("[{}] singleton", taxon.taxonomy);

            TaxonResult r;
            r.taxonomy = taxon.taxonomy;
            r.status = TaxonStatus::SINGLETON;
            r.n_genomes = 1;
            r.n_representatives = 1;
            r.method = "singleton";

            if (run_state) {
                TaxonOutput out;
                out.result = r;
                out.all_accessions = {taxon.genomes[0].accession};
                out.representatives = {taxon.genomes[0].accession};
                out.rep_embeddings.assign(1, std::vector<float>(cfg.embedding_dim, 0.0f));
                out.rep_cluster_size[taxon.genomes[0].accession] = 1u;
                out.sketch_kmer_used = cfg.kmer_size;
                run_state->push(std::move(out));
            }
            return r;
        }

        auto quality_scores = build_quality_map(taxon, gpk_reader);

        // Completeness lookup: accession → completeness (%) for quality floor decisions.
        std::unordered_map<std::string, double> acc_completeness;
        acc_completeness.reserve(taxon.genomes.size());
        for (const auto& g : taxon.genomes)
            if (g.completeness) acc_completeness[g.accession] = *g.completeness;

        // Build acc_to_idx once, reuse everywhere (index into all_accessions).
        std::unordered_map<std::string, size_t> acc_to_idx;
        acc_to_idx.reserve(all_accessions.size());
        for (size_t i = 0; i < all_accessions.size(); ++i)
            acc_to_idx[all_accessions[i]] = i;

        // -----------------------------------------------------------
        // 4a. Fast path: n <= TINY_N_THRESHOLD — direct OPH Jaccard, skip full pipeline
        // -----------------------------------------------------------
        static constexpr size_t TINY_N_THRESHOLD = 20;
        if (taxon.size() <= TINY_N_THRESHOLD) {
            auto t0 = std::chrono::steady_clock::now();
            const size_t n = taxon.size();

            if (!gpk_reader || !gpk_reader->has_sketches()) {
                throw std::runtime_error(
                    "geodesic requires a V4 genopack archive with SKCH section "
                    "(sketches = dual-seed OPH); FASTA/V3 ingestion removed");
            }

            // Fetch sketches from the genopack archive — V4 SKCH stores dual-seed OPH
            // (sig + sig2 + real_bins mask), the same data the old path recomputed from FASTA.
            // visit_sketch_batches groups accessions by archive part and decompresses each
            // SKCH section exactly once; callback delivers SketchResult per hit.
            std::vector<std::vector<uint16_t>> sigs(n);
            std::vector<std::vector<uint64_t>> real_masks(n);
            std::vector<uint32_t> n_real_vec(n, 0);
            std::unordered_set<size_t> failed_indices;
            std::vector<uint8_t> hit(n, 0);

            const uint32_t req_k  = static_cast<uint32_t>(cfg.kmer_size);
            const uint32_t req_sz = static_cast<uint32_t>(cfg.sketch_size);

            gpk_reader->visit_sketch_batches(all_accessions, req_k, req_sz,
                [&](size_t i, const genopack::SketchResult& sk) {
                    hit[i] = 1;
                    sigs[i].assign(sk.sig, sk.sig + sk.sketch_size);
                    real_masks[i].assign(sk.mask, sk.mask + sk.mask_words);
                    n_real_vec[i] = sk.n_real_bins;
                });

            for (size_t i = 0; i < n; ++i) {
                if (!hit[i]) failed_indices.insert(i);
            }

            // Record permanently failed genome reads (missing from SKCH / not in archive).
            std::vector<FailedGenomeRecord> tiny_failed_records;
            for (size_t i : failed_indices) {
                tiny_failed_records.push_back({all_accessions[i], taxon.taxonomy, "",
                                               "sketch not found in genopack SKCH"});
            }

            // GUNC exclusion: genomes failing GUNC are excluded from rep selection.
            std::unordered_set<size_t> excluded_indices = failed_indices;
            std::vector<OutlierRecord> gunc_contam_records;
            if (gunc_scores && !gunc_scores->empty()) {
                for (size_t i = 0; i < n; ++i) {
                    auto acc = canonical_accession(all_accessions[i]);
                    auto git = gunc_scores->find(acc);
                    if (git == gunc_scores->end() || git->second.pass_gunc) continue;
                    excluded_indices.insert(i);
                    if (is_verbose())
                        spdlog::warn("[{}] GUNC fail: {} (CSS={:.3f})", taxon.taxonomy, acc,
                                     git->second.clade_separation_score);
                    OutlierRecord rec;
                    rec.accession       = acc;
                    rec.kmer_div_zscore = static_cast<double>(git->second.clade_separation_score);
                    rec.nn_outlier      = false;
                    rec.flag_reason     = "gunc_fail";
                    rec.category        = "contaminated";
                    gunc_contam_records.push_back(std::move(rec));
                }
            }

            auto coverage_score = [&](size_t query, size_t target) -> double {
                if (sigs[query].empty() || sigs[target].empty()) return 0.0;
                return GeodesicDerep::score_pair(
                    sigs[query].data(), sigs[target].data(),
                    static_cast<uint32_t>(cfg.sketch_size),
                    n_real_vec[query], n_real_vec[target],
                    cfg.kmer_size,
                    GeodesicDerep::refine_jaccard(sigs[query], sigs[target]));
            };

            // Compute all pairwise Jaccard (between non-failed genomes)
            std::vector<std::vector<double>> jac(n, std::vector<double>(n, 1.0));
            for (size_t i = 0; i < n; ++i) {
                if (failed_indices.count(i)) continue;
                for (size_t j = i + 1; j < n; ++j) {
                    if (failed_indices.count(j)) { jac[i][j] = jac[j][i] = 0.0; continue; }
                    jac[i][j] = jac[j][i] = GeodesicDerep::refine_jaccard(sigs[i], sigs[j]);
                }
            }

            // Convert ANI threshold to Jaccard threshold
            double ani_threshold_frac = cfg.ani_threshold / 100.0;
            double q = std::pow(ani_threshold_frac, cfg.kmer_size);
            double jaccard_threshold = q / (2.0 - q);

            // Sort genome indices by quality descending (excluded genomes sorted last)
            std::vector<size_t> order(n);
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                const double qa = quality_scores.count(all_accessions[a])
                                  ? quality_scores.at(all_accessions[a])
                                  : taxon.genomes[a].quality_score();
                const double qb = quality_scores.count(all_accessions[b])
                                  ? quality_scores.at(all_accessions[b])
                                  : taxon.genomes[b].quality_score();
                if (qa != qb) return qa > qb;
                return all_accessions[a] < all_accessions[b];  // tie-break on accession
            });

            // Greedy quality-sorted cover: excluded genomes cannot be reps
            std::vector<size_t> rep_indices;
            std::vector<bool> is_rep(n, false);
            for (size_t idx : order) {
                if (excluded_indices.count(idx)) continue;  // never a rep
                bool covered = false;
                for (size_t ri : rep_indices) {
                    if (coverage_score(idx, ri) >= static_cast<double>(ani_threshold_frac)) { covered = true; break; }
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

            // Build ani_to_rep_map and coverage stats; use Jaccard for ANI estimation
            std::unordered_map<std::string, double> ani_to_rep_map;
            std::vector<double> genome_to_rep_ani(n);
            std::unordered_map<std::string, std::string> tiny_member_to_rep;
            std::unordered_map<std::string, double>      tiny_member_nn_dist;
            std::unordered_map<std::string, float>       tiny_member_fill_ratio;
            std::unordered_map<std::string, uint32_t>    tiny_cluster_size;
            for (size_t ri : rep_indices) tiny_cluster_size[all_accessions[ri]] = 1u;
            for (size_t i = 0; i < n; ++i) {
                if (is_rep[i]) { genome_to_rep_ani[i] = 100.0; continue; }
                if (failed_indices.count(i)) { genome_to_rep_ani[i] = 0.0; continue; }
                double best_j = 0.0;
                size_t best_ri = rep_indices.empty() ? SIZE_MAX : rep_indices[0];
                for (size_t ri : rep_indices) {
                    if (jac[i][ri] >= best_j) { best_j = jac[i][ri]; best_ri = ri; }
                }
                float best_score = (best_ri != SIZE_MAX && !sigs[i].empty())
                    ? coverage_score(i, best_ri)
                    : static_cast<float>(best_j);
                double ani = std::max(70.0, std::min(100.0,
                    static_cast<double>(best_score) * 100.0));
                ani_to_rep_map[all_accessions[i]] = ani;
                genome_to_rep_ani[i] = ani;
                if (best_ri != SIZE_MAX && !excluded_indices.count(i)) {
                    tiny_member_to_rep[all_accessions[i]] = all_accessions[best_ri];
                    tiny_member_nn_dist[all_accessions[i]] = 1.0 - best_score;
                    tiny_member_fill_ratio[all_accessions[i]] = cfg.sketch_size > 0
                        ? static_cast<float>(n_real_vec[i]) / cfg.sketch_size : 1.0f;
                    ++tiny_cluster_size[all_accessions[best_ri]];
                }
            }
            // Coverage stats over non-failed genomes
            size_t n_valid = n - failed_indices.size();
            double cov_sum = 0.0, cov_min = 100.0, cov_max = 0.0;
            for (size_t i = 0; i < n; ++i) {
                if (failed_indices.count(i)) continue;
                cov_sum += genome_to_rep_ani[i];
                cov_min = std::min(cov_min, genome_to_rep_ani[i]);
                cov_max = std::max(cov_max, genome_to_rep_ani[i]);
            }
            double cov_mean = (n_valid > 0) ? cov_sum / static_cast<double>(n_valid) : 0.0;

            // Diversity stats: pairwise ANI among reps only
            double div_sum = 0.0, div_min = 100.0, div_max = 0.0;
            int div_pairs = 0;
            for (size_t a = 0; a < rep_indices.size(); ++a) {
                for (size_t b = a + 1; b < rep_indices.size(); ++b) {
                    size_t ra = rep_indices[a], rb = rep_indices[b];
                    double ani = std::max(70.0, std::min(100.0,
                        static_cast<double>(GeodesicDerep::score_pair(
                            sigs[ra].data(), sigs[rb].data(),
                            static_cast<uint32_t>(cfg.sketch_size),
                            n_real_vec[ra], n_real_vec[rb], cfg.kmer_size,
                            jac[ra][rb])) * 100.0));
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

            const size_t n_gunc = gunc_contam_records.size();
            if (is_verbose() || (!is_quiet() && n >= 10)) {
                if (n_gunc > 0)
                    spdlog::info("[{}] {} → {} reps, {} GUNC ({:.2f}s) [tiny]",
                                 taxon.taxonomy, n, representatives.size(), n_gunc, runtime);
                else
                    spdlog::info("[{}] {} → {} reps ({:.2f}s) [tiny]",
                                 taxon.taxonomy, n, representatives.size(), runtime);
            }

            TaxonResult r;
            r.taxonomy          = taxon.taxonomy;
            r.status            = TaxonStatus::SUCCESS;
            r.n_genomes         = static_cast<int>(n);
            r.n_representatives = static_cast<int>(representatives.size());
            r.method            = "geodesic-tiny";

            TaxonDiversityStats div_stats;
            div_stats.taxonomy           = taxon.taxonomy;
            div_stats.method             = "geodesic-tiny";
            div_stats.n_genomes          = static_cast<int>(n);
            div_stats.n_representatives  = static_cast<int>(representatives.size());
            div_stats.reduction_ratio    = 1.0 - static_cast<double>(representatives.size()) /
                                                 static_cast<double>(n);
            div_stats.runtime_seconds    = runtime;
            div_stats.coverage_mean_ani  = cov_mean;
            div_stats.coverage_min_ani   = cov_min;
            div_stats.coverage_max_ani   = cov_max;
            div_stats.diversity_mean_ani = div_mean;
            div_stats.diversity_min_ani  = div_min;
            div_stats.diversity_max_ani  = div_max;
            div_stats.diversity_n_pairs   = div_pairs;
            div_stats.n_outliers_excluded = static_cast<int>(n_gunc);
            div_stats.n_outliers_retained = 0;

            if (run_state) {
                TaxonOutput out;
                out.result          = r;
                out.diversity_stats = div_stats;
                out.all_accessions  = all_accessions;
                out.representatives = representatives;
                out.ani_map         = ani_to_rep_map;
                out.outliers        = gunc_contam_records;
                out.failed_genomes  = std::move(tiny_failed_records);
                out.member_to_rep    = std::move(tiny_member_to_rep);
                out.member_nn_dist   = std::move(tiny_member_nn_dist);
                out.member_fill_ratio = std::move(tiny_member_fill_ratio);
                out.rep_cluster_size = std::move(tiny_cluster_size);
                out.rep_embeddings.assign(representatives.size(),
                                          std::vector<float>(cfg.embedding_dim, 0.0f));
                out.sketch_kmer_used = cfg.kmer_size;
                // Pipeline health
                out.n_input                  = static_cast<int>(n);
                out.n_preflight_excluded     = 0;
                out.n_quality_floor_excluded = 0;
                out.n_outliers_excluded      = static_cast<int>(n_gunc);
                out.n_outliers_retained      = 0;
                out.mst_p90_edge             = 0.0;
                out.mst_true_max             = 0.0;
                out.ani_threshold_used       = cfg.ani_threshold;
                run_state->push(std::move(out));
            }
            return r;
        }

        // -----------------------------------------------------------
        // 4. GEODESIC: Physics-inspired paradigm shift (O(n log n))
        // -----------------------------------------------------------
        auto geodesic_start = std::chrono::steady_clock::now();
        if (is_verbose()) spdlog::info("[{}] GEODESIC mode: {} genomes", taxon.taxonomy, taxon.size());

        // Parameters (no FASTA auto-calibration — GPK encodes k/sketch_size).
        int kmer_size = cfg.kmer_size;
        int embedding_dim = cfg.embedding_dim;
        int sketch_size = cfg.sketch_size;
        float diversity_threshold = cfg.diversity_threshold;

        double ani_threshold_frac = cfg.ani_threshold / 100.0;

        // ANI threshold as angular distance — used as the upper cap on diversity_threshold.
        // The actual value is set from the real NN distribution after build_index.
        {
            const double user_ani = cfg.ani_threshold / 100.0;
            double ak = std::pow(user_ani, static_cast<double>(kmer_size));
            double j = std::clamp(ak / (2.0 - ak), 0.0, 1.0);
            diversity_threshold = static_cast<float>(std::acos(j) / M_PI);
        }

        // Both thresholds are placeholders — replaced below using real NN distances.
        // theta/4 ensures coverage-preserving merge: by triangle inequality,
        // theta/2 can leave genomes 1.5*theta away from kept rep.
        float min_rep_distance = diversity_threshold * 0.25f;

        GeodesicDerep::Config gcfg{
            .embedding_dim = embedding_dim,
            .sketch_size = sketch_size,
            .kmer_size = kmer_size,
            .syncmer_s = cfg.syncmer_s,
            .ani_threshold = ani_threshold_frac,
            .hnsw_m = 48,
            .hnsw_ef_construction = 400,
            .hnsw_ef_search = 50,
            .threads = threads,
            .io_threads = cfg.io_threads,
            .isolation_k = 10,
            .k_cap_max = cfg.k_cap_max,
            .diversity_threshold = diversity_threshold,
            .min_rep_distance = min_rep_distance,
            .max_rep_fraction = cfg.max_rep_fraction,
            .nystrom_diagonal_loading = cfg.nystrom_diagonal_loading,
            .nystrom_degree_normalize = cfg.nystrom_degree_normalize,
            .seed = cfg.seed
        };
        gcfg.pool = pool;

        GeodesicDerep geodesic(gcfg);
        if (on_serial_phase) geodesic.set_on_serial_phase(std::move(on_serial_phase));

        // Stage 1: Pre-filter severely fragmented assemblies using archive metadata.
        // Uses average contig size (genome_length / n_contigs) instead of a hard contig
        // count cutoff.  A genome with many contigs but decent average size (e.g. 600
        // contigs × 5 kb = 3 Mbp) is fine; one with avg < 1 kb is genuinely junk.
        // Only applied when a .gpk archive is available (provides metadata without FASTA read).
        // r232_v5 audit: at 1000bp, 56% of flags sat within 5% of the cutoff (900+
        // genuinely-borderline exclusions) with a clean gap in the distribution between
        // 657bp and 803bp; lowered to keep the genuinely fragmented tail out only.
        static constexpr double MIN_AVG_CONTIG_BP = 700.0;  // avg contig size floor
        std::vector<GeodesicDerep::OutlierCandidate> preflagged_fragmented;
        if (gpk_reader && !taxon.genomes.empty()) {
            std::vector<Genome> clean_genomes;
            clean_genomes.reserve(taxon.genomes.size());
            for (const auto& g : taxon.genomes) {
                auto meta = gpk_reader->genome_meta_by_accession(g.accession);
                const bool is_fragmented = meta && meta->n_contigs > 0 &&
                    (static_cast<double>(meta->genome_length) /
                     static_cast<double>(meta->n_contigs)) < MIN_AVG_CONTIG_BP;
                if (is_fragmented) {
                    GeodesicDerep::OutlierCandidate c;
                    c.genome_id            = 0;
                    c.nn_outlier           = true;
                    c.isolation_score      = 1.0f;
                    c.kmer_div_zscore      = 0.0f;
                    c.genome_size_zscore   = 0.0f;
                    c.centroid_distance    = 1.0f;
                    c.anomaly_score        = 1.0f;
                    c.margin_to_threshold  = 0.5f;
                    c.flag_reason          = "fragmented:pre_filter";
                    c.accession            = g.accession;
                    c.n_contigs            = static_cast<uint32_t>(meta->n_contigs);
                    c.genome_length_bp     = static_cast<uint64_t>(meta->genome_length);
                    preflagged_fragmented.push_back(std::move(c));
                } else {
                    clean_genomes.push_back(g);
                }
            }
            if (!preflagged_fragmented.empty()) {
                spdlog::debug("[{}] Stage-1 pre-filter: {} fragmented (avg contig < {} bp, {} remain)",
                              taxon.taxonomy, preflagged_fragmented.size(),
                              static_cast<int>(MIN_AVG_CONTIG_BP), clean_genomes.size());
                // Rebuild all_accessions for the clean subset only.
                all_accessions.clear();
                for (const auto& g : clean_genomes)
                    all_accessions.push_back(g.accession);
                const_cast<Taxon&>(taxon).genomes = std::move(clean_genomes);
                // Rebuild acc_to_idx — old indices are stale after shrink.
                acc_to_idx.clear();
                acc_to_idx.reserve(all_accessions.size());
                for (size_t i = 0; i < all_accessions.size(); ++i)
                    acc_to_idx[all_accessions[i]] = i;
            }
        }

        // Build index from V4 GPK SKCH sketches — FASTA no longer supported.
        if (!gpk_reader || !gpk_reader->has_sketches()) {
            throw std::runtime_error(
                "geodesic requires a V4 genopack archive with SKCH section "
                "(sketches = dual-seed OPH); FASTA/V3 ingestion removed");
        }
        geodesic.build_index_from_gpk_sketches(all_accessions, *gpk_reader, quality_scores);
        size_t newly_embedded = all_accessions.size();

        // Record permanently failed genome reads (absent from SKCH, corrupt entry).
        // failed_reads() now returns (accession, reason) directly.
        std::vector<FailedGenomeRecord> geodesic_failed_records;
        for (const auto& [accession, reason] : geodesic.failed_reads()) {
            geodesic_failed_records.push_back({accession, taxon.taxonomy, "", reason});
        }

        // All genomes excluded (e.g. entire taxon missing from SKCH) — record failures and skip.
        if (geodesic.embeddings().empty()) {
            spdlog::warn("[{}] all {} genomes excluded (no valid embeddings) — skipping taxon",
                         taxon.taxonomy, taxon.size());

            TaxonResult r;
            r.taxonomy          = taxon.taxonomy;
            r.status            = TaxonStatus::FAILED;
            r.n_genomes         = static_cast<int>(taxon.size());
            r.n_representatives = 0;
            r.method            = "geodesic-skipped";

            if (run_state) {
                TaxonOutput out;
                out.result          = r;
                out.all_accessions  = all_accessions;
                out.failed_genomes  = std::move(geodesic_failed_records);
                run_state->push(std::move(out));
            }
            return r;
        }

        // Phase 3: single HNSW pass — isolation scores + 1-NN distribution fused.
        auto nn = geodesic.compute_isolation_scores();

        // Adaptive k-selection: if GPK has multi-k sketches, re-embed with the k
        // that best matches this taxon's diversity (clonal→31, moderate→21, diverse→16).
        // maybe_reselect_k() returns true only when k actually changed; in that case
        // re-run isolation scores on the fresh HNSW.
        if (gpk_reader && geodesic.maybe_reselect_k(nn, quality_scores)) {
            nn = geodesic.compute_isolation_scores();
        }

        // Compute ad-hoc quality scores for genomes without CheckM2 data.
        // Uses centrality (inverse isolation) and kmer density as quality proxy.
        geodesic.compute_adhoc_quality_scores();

        // diversity_threshold = min(θ_ANI, MST_max_edge):
        //   MST_max_edge: maximum edge in the minimum spanning tree of the k-NN graph.
        //   θ_ANI is the hard upper cap from --ani-threshold (never merge across species).
        //   Falls back to NN_P95 if MST is unavailable (brute-force small-n path).
        // min_rep_distance = P5 of NN distances: electrostatic merge collapses near-duplicates.
        {
            // Use MST connectivity scale as the diversity threshold.
            // Falls back to NN_P95 when MST unavailable (brute-force path).
            const float ani_floor = [&]() -> float {
                double ak = std::pow(ani_threshold_frac, static_cast<double>(kmer_size));
                double j  = std::clamp(ak / (2.0 - ak), 0.0, 1.0);
                return static_cast<float>(std::acos(j) / M_PI) * 0.25f;
            }();
            float mst_threshold = (nn.mst_max_edge > 0.0)
                ? static_cast<float>(nn.mst_max_edge)
                : static_cast<float>(nn.p95);
            // When the k-NN graph captures only near-duplicates (e.g., a large clonal
            // database where all K nearest neighbours have Jaccard > 0.998), mst_max_edge
            // falls far below the ANI species scale.  Clamp to ani_floor = θ_ANI/4 so the
            // diversity threshold stays biologically meaningful.
            if (mst_threshold < ani_floor)
                mst_threshold = ani_floor;

            diversity_threshold = std::max(1e-6f,
                std::min(diversity_threshold, mst_threshold));
            // theta/4 ensures coverage-preserving merge: by triangle inequality,
            // theta/2 can leave genomes 1.5*theta away from kept rep.
            min_rep_distance = std::min(static_cast<float>(nn.p5),
                                        diversity_threshold * 0.25f);

            geodesic.set_diversity_threshold(diversity_threshold);
            geodesic.set_min_rep_distance(min_rep_distance);

            // When the k-NN graph captured only near-duplicates, the Nyström embedding
            // maps all genomes to nearly the same unit vector (dot-products ≈ 1.0),
            // so dot-product FPS immediately covers everyone → too few representatives.
            // Detect this by comparing P95 NN distance to the diversity scale:
            // if P95 << div_thr the embedding is degenerate at the diversity scale,
            // and we fall back to actual Jaccard for FPS (slower but correct).
            if (static_cast<float>(nn.p95) < diversity_threshold * 0.3f) {
                if (nn.nystrom_taxon_applied) {
                    if (geodesic.nystrom_oph_sphere_applied()) {
                        spdlog::info("[{}] Near-duplicate k-NN (p95_nn={:.4f} << div_thr={:.4f}): "
                                     "OPH token sphere active (ANN recall gap, σ≈0.005), retaining hypersphere FPS",
                                     taxon.taxonomy, nn.p95, diversity_threshold);
                    } else {
                        const float j_floor = geodesic.nystrom_scaled_j_floor();
                        if (j_floor > 0.0f) {
                            spdlog::info("[{}] Near-duplicate k-NN (p95_nn={:.4f} << div_thr={:.4f}): "
                                         "kernel-scaled Nyström (ANN recall gap, j_floor={:.4f}), retaining hypersphere FPS",
                                         taxon.taxonomy, nn.p95, diversity_threshold, j_floor);
                        } else {
                            spdlog::info("[{}] Near-duplicate k-NN (p95_nn={:.4f} << div_thr={:.4f}): "
                                         "per-taxon Nyström active with bridge anchors, retaining hypersphere FPS",
                                         taxon.taxonomy, nn.p95, diversity_threshold);
                        }
                    }
                } else {
                    spdlog::info("[{}] Near-duplicate k-NN (p95_nn={:.4f} << div_thr={:.4f}): "
                                 "disabling Nyström dot-product FPS, using actual Jaccard",
                                 taxon.taxonomy, nn.p95, diversity_threshold);
                    geodesic.set_nystrom_applied(false);
                }
            }

            const bool heavy_tail = (nn.tail_ratio > 2.0);
            if (nn.low_pair_count || nn.pathological_bridge || nn.disconnected_mst || heavy_tail) {
                spdlog::warn("[{}] Threshold instability detected: low_pairs={} bridge={} disconnected={} "
                             "heavy_tail={} (tail_ratio={:.2f}). "
                             "diversity_threshold={:.4f} ani_floor={:.4f} "
                             "(inferred from ANI cap + MST; adjust --ani-threshold if unsuitable)",
                             taxon.taxonomy,
                             nn.low_pair_count, nn.pathological_bridge, nn.disconnected_mst,
                             heavy_tail, nn.tail_ratio,
                             diversity_threshold, ani_floor);
            }

            double thr_j   = std::cos(static_cast<double>(diversity_threshold) * M_PI);
            double thr_ani = GeodesicDerep::jaccard_to_ani(std::max(0.0, thr_j), kmer_size);
            spdlog::info("[{}] geodesic: k={} dim={} sketch={} | "
                         "div_thr={:.4f} ({:.1f}% ANI, MST={:.4f} floor={:.4f} tail_ratio={:.2f}) | "
                         "min_rep={:.4f} (NN P5={:.4f} P50={:.4f} P95={:.4f}) | "
                         "k_conn={} k_stable={} K_cap={}",
                         taxon.taxonomy, kmer_size, embedding_dim, sketch_size,
                         diversity_threshold, thr_ani, nn.mst_max_edge, ani_floor, nn.tail_ratio,
                         min_rep_distance, nn.p5, nn.p50, nn.p95,
                         nn.k_conn, nn.k_stable, nn.k_cap);
        }

        // Stage 3: Adaptive z-threshold — scale by log2(n) so large clonal species
        // get a stricter threshold (fewer false positives) while rare diverse species
        // get a looser one (fewer false negatives on genuine rare lineages).
        // base_z=2 at n=10, grows to ~4 at n=50k. User's --z-threshold is the base.
        const float adaptive_z = [&]() -> float {
            const size_t n = taxon.size();
            if (n < 10) return static_cast<float>(cfg.z_threshold);
            return static_cast<float>(cfg.z_threshold) *
                   (1.0f + std::log2(static_cast<float>(n) / 10.0f) * 0.25f);
        }();

        // Detect potential contamination before selection
        auto contamination = geodesic.detect_outlier_candidates(adaptive_z);

        // Merge Stage-1 pre-flagged fragmented genomes into contamination candidate list
        for (auto& c : preflagged_fragmented)
            contamination.push_back(std::move(c));

        // Completeness-aware exclusion: outlier candidates with reasonable avg contig
        // size (>= 1 kb) and completeness >= 50% are flagged but retained for rep selection.
        std::unordered_set<std::string> contaminated_accessions;
        for (auto& c : contamination) {
            const bool has_decent_assembly =
                (c.n_contigs > 0 && c.genome_length_bp > 0 &&
                 (static_cast<double>(c.genome_length_bp) /
                  static_cast<double>(c.n_contigs)) >= MIN_AVG_CONTIG_BP) &&
                [&]() -> bool {
                    auto cit = acc_completeness.find(c.accession);
                    return (cit != acc_completeness.end() && cit->second >= 50.0);
                }();
            c.excluded = !has_decent_assembly;

            if (is_verbose()) spdlog::warn("[{}] Potential contamination: {} (centroid_dist={:.3f}, "
                         "isolation={:.3f}, kmer_div_z={:.2f}, nn_outlier={}, excluded={})",
                         taxon.taxonomy, c.accession,
                         c.centroid_distance, c.isolation_score, c.kmer_div_zscore, c.nn_outlier,
                         c.excluded);
            if (c.excluded)
                contaminated_accessions.insert(c.accession);
        }

        // Pre-FPS quality floor: exclude genomes with avg contig size < 1 kb OR
        // completeness < 50% that were not already caught by outlier detection.
        // These are excluded from rep selection but do NOT generate an OutlierRecord.
        int n_quality_floor = 0;
        {
            std::unordered_set<std::string> already_in_contamination;
            for (const auto& c : contamination)
                already_in_contamination.insert(c.accession);

            for (const auto& emb : geodesic.embeddings()) {
                const std::string& acc = emb.accession;
                if (already_in_contamination.count(acc)) continue;

                const bool fragmented = emb.n_contigs > 0 &&
                    (static_cast<double>(emb.genome_size) /
                     static_cast<double>(emb.n_contigs)) < MIN_AVG_CONTIG_BP;
                const auto cit = acc_completeness.find(acc);
                const bool low_completeness = (cit != acc_completeness.end())
                    ? cit->second < 50.0
                    : false;

                if (fragmented || low_completeness) {
                    contaminated_accessions.insert(acc);
                    ++n_quality_floor;
                }
            }
        }

        // Apply GUNC flags: genomes failing GUNC are excluded from reps.
        // GUNC-only failures (not already in embedding candidates) are collected
        // into gunc_contam_records and merged into contam_records before the single
        // batch insert — never inserted individually to avoid DELETE-per-genome.
        std::vector<OutlierRecord> gunc_contam_records;
        if (gunc_scores && !gunc_scores->empty()) {
            std::unordered_set<std::string> already_flagged;
            for (const auto& c : contamination)
                already_flagged.insert(c.accession);
            for (const auto& g : taxon.genomes) {
                auto acc = canonical_accession(g.accession);
                auto git = gunc_scores->find(acc);
                if (git == gunc_scores->end()) continue;
                if (git->second.pass_gunc) continue;
                contaminated_accessions.insert(g.accession);
                if (is_verbose())
                    spdlog::warn("[{}] GUNC fail: {} (CSS={:.3f})",
                                 taxon.taxonomy, acc,
                                 git->second.clade_separation_score);
                if (already_flagged.count(acc)) continue;
                // kmer_div_zscore stores CSS; nn_outlier=false (embedding not the source)
                OutlierRecord rec;
                rec.accession          = acc;
                rec.centroid_distance  = 0.0;
                rec.isolation_score    = 0.0;
                rec.anomaly_score      = 0.0;
                rec.nn_outlier         = false;
                rec.kmer_div_zscore    = static_cast<double>(git->second.clade_separation_score);
                rec.genome_size_zscore = 0.0;
                rec.flag_reason        = "gunc_fail";
                rec.category           = "contaminated";
                gunc_contam_records.push_back(std::move(rec));
            }
        }

        // Exclude contaminated genomes from rep selection before running FPS
        geodesic.exclude_from_reps(contaminated_accessions);

        // Pre-seed fixed representatives (--fixed-reps) before FPS
        if (cfg.references_file) {
            const auto fixed_accessions = [&]() {
                std::unordered_set<std::string> s;
                std::ifstream f(*cfg.references_file);
                std::string line;
                while (std::getline(f, line)) {
                    if (!line.empty()) s.insert(line);
                }
                return s;
            }();

            if (!fixed_accessions.empty()) {
                std::unordered_set<std::string> pinned_accessions;
                for (const auto& acc : fixed_accessions) {
                    if (acc_to_idx.count(acc))
                        pinned_accessions.insert(acc);
                }
                if (!pinned_accessions.empty()) {
                    geodesic.set_pinned_representatives(pinned_accessions);
                    spdlog::info("[{}] pinned {} fixed representatives",
                                 taxon.taxonomy, pinned_accessions.size());
                }
            }
        }

        // Select representatives (edges' source/target are accessions)
        auto edges = geodesic.select_representatives();

        // Collect unique representatives using index bitmaps (indexed by all_accessions).
        std::vector<bool> covered_bm(all_accessions.size(), false);
        std::vector<bool> rep_bm(all_accessions.size(), false);
        for (const auto& e : edges) {
            auto src_it = acc_to_idx.find(e.source);
            if (src_it != acc_to_idx.end())
                covered_bm[src_it->second] = true;
            // Only add as rep if not contaminated (safety net)
            if (contaminated_accessions.find(e.target) == contaminated_accessions.end()) {
                auto tgt_it = acc_to_idx.find(e.target);
                if (tgt_it != acc_to_idx.end())
                    rep_bm[tgt_it->second] = true;
            }
        }
        // Genomes not covered elect themselves (if not contaminated)
        for (size_t i = 0; i < all_accessions.size(); ++i) {
            if (!covered_bm[i] &&
                contaminated_accessions.find(all_accessions[i]) == contaminated_accessions.end()) {
                rep_bm[i] = true;
            }
        }

        // Build rep_set (accessions) for downstream code that still needs string lookups
        std::unordered_set<std::string> rep_set;
        for (size_t i = 0; i < all_accessions.size(); ++i) {
            if (rep_bm[i])
                rep_set.insert(all_accessions[i]);
        }

        std::vector<OutlierRecord> contam_records;
        contam_records.reserve(contamination.size() + gunc_contam_records.size());
        for (const auto& c : contamination) {
            if (c.accession.empty()) continue;
            OutlierRecord rec;
            rec.accession          = c.accession;
            rec.centroid_distance  = static_cast<double>(c.centroid_distance);
            rec.isolation_score    = static_cast<double>(c.isolation_score);
            rec.anomaly_score      = static_cast<double>(c.anomaly_score);
            rec.genome_size_zscore = static_cast<double>(c.genome_size_zscore);
            rec.nn_outlier         = c.nn_outlier;
            rec.kmer_div_zscore    = static_cast<double>(c.kmer_div_zscore);
            rec.margin_to_threshold = static_cast<double>(c.margin_to_threshold);
            rec.flag_reason        = c.flag_reason;
            rec.category           = (c.flag_reason == "fragmented:pre_filter")
                                     ? "low_quality" : "misassigned";
            rec.n_contigs          = c.n_contigs;
            rec.genome_length_bp   = c.genome_length_bp;
            rec.excluded           = (contaminated_accessions.count(c.accession) > 0);
            contam_records.push_back(std::move(rec));
        }
        // Append GUNC-only failures (merged here so a single batch insert handles all)
        for (auto& rec : gunc_contam_records)
            contam_records.push_back(std::move(rec));

        // rep_set already contains accessions.
        // Sort: unordered_set iteration order is impl-defined → cascades into rep
        // emission order in .gpd/.tsv outputs and any downstream rep iteration.
        std::vector<std::string> all_representatives(rep_set.begin(), rep_set.end());
        std::sort(all_representatives.begin(), all_representatives.end());

        auto geodesic_end = std::chrono::steady_clock::now();
        double runtime_secs = std::chrono::duration<double>(geodesic_end - geodesic_start).count();

        if (is_verbose() || (!is_quiet() && taxon.size() >= 10)) {
            int n_misassigned = 0, n_low_quality = 0, n_gunc = 0;
            for (const auto& rec : contam_records) {
                if      (rec.category == "misassigned")  ++n_misassigned;
                else if (rec.category == "low_quality")  ++n_low_quality;
                else if (rec.category == "contaminated") ++n_gunc;
            }
            if (n_misassigned == 0 && n_low_quality == 0 && n_gunc == 0) {
                spdlog::info("[{}] {} → {} reps ({:.1f}s)",
                             taxon.taxonomy, taxon.size(),
                             all_representatives.size(), runtime_secs);
            } else {
                std::string suffix;
                if (n_misassigned > 0) suffix += std::to_string(n_misassigned) + " misassigned";
                if (n_low_quality > 0) {
                    if (!suffix.empty()) suffix += ", ";
                    suffix += std::to_string(n_low_quality) + " low_quality";
                }
                if (n_gunc > 0) {
                    if (!suffix.empty()) suffix += ", ";
                    suffix += std::to_string(n_gunc) + " contaminated (GUNC)";
                }
                spdlog::info("[{}] {} → {} reps, {} ({:.1f}s)",
                             taxon.taxonomy, taxon.size(),
                             all_representatives.size(), suffix, runtime_secs);
            }
        }

        // Compute diversity metrics from embeddings
        std::vector<uint64_t> rep_ids;
        const auto& embeds = geodesic.embeddings();
        for (const auto& emb : embeds) {
            if (rep_set.count(emb.accession)) {
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

        int n_olr_excl = 0, n_olr_ret = 0;
        for (const auto& r : contam_records) {
            if (r.excluded) ++n_olr_excl; else ++n_olr_ret;
        }
        div_stats.n_outliers_excluded = n_olr_excl;
        div_stats.n_outliers_retained = n_olr_ret;

        TaxonResult r;
        r.taxonomy = taxon.taxonomy;
        r.status = TaxonStatus::SUCCESS;
        r.n_genomes = taxon.size();
        r.n_representatives = static_cast<int>(all_representatives.size());
        r.method = "geodesic";

        // Build ani_to_rep map: weight_raw = ANI fraction (Mash formula applied at emission)
        std::unordered_map<std::string, double> ani_to_rep_map;
        for (const auto& e : edges) {
            if (!acc_to_idx.count(e.source)) continue;
            double ani = std::clamp(static_cast<double>(e.weight_raw) * 100.0, 70.0, 100.0);
            auto& best = ani_to_rep_map[e.source];
            best = std::max(best, ani);
        }

        std::unordered_map<std::string, std::string> geo_member_to_rep;
        std::unordered_map<std::string, double>      geo_member_nn_dist;
        std::unordered_map<std::string, float>       geo_member_fill_ratio;
        std::unordered_map<std::string, uint32_t>    geo_cluster_size;
        if (cfg.sketch_size > 0) {
            for (const auto& em : geodesic.embeddings())
                geo_member_fill_ratio[em.accession] =
                    static_cast<float>(em.n_real_bins) / cfg.sketch_size;
        }
        for (const auto& acc : all_representatives) geo_cluster_size[acc] = 1u;
        {
            std::unordered_map<std::string, double> best_w;
            for (const auto& e : edges) {
                if (rep_set.count(e.source)) continue;
                if (!rep_set.count(e.target)) continue;
                if (contaminated_accessions.count(e.source)) continue;
                auto it = best_w.find(e.source);
                if (it == best_w.end() || e.weight_raw > it->second) {
                    best_w[e.source] = e.weight_raw;
                    geo_member_to_rep[e.source] = e.target;
                }
            }
            for (const auto& [m, r2] : geo_member_to_rep) {
                (void)m;
                ++geo_cluster_size[r2];
            }
            for (const auto& [m, w] : best_w)
                geo_member_nn_dist[m] = 1.0 - w;
        }

        std::vector<std::vector<float>> geo_rep_embeddings(all_representatives.size());
        {
            std::unordered_map<std::string, size_t> rep_idx;
            rep_idx.reserve(all_representatives.size());
            for (size_t i = 0; i < all_representatives.size(); ++i)
                rep_idx[all_representatives[i]] = i;
            const auto& embs = geodesic.embeddings();
            for (const auto& em : embs) {
                auto it = rep_idx.find(em.accession);
                if (it == rep_idx.end()) continue;
                geo_rep_embeddings[it->second] = em.vector;
            }
            for (auto& v : geo_rep_embeddings)
                if (v.empty()) v.assign(cfg.embedding_dim, 0.0f);
        }

        if (run_state) {
            TaxonOutput out;
            out.result          = r;
            out.diversity_stats = div_stats;
            out.all_accessions  = all_accessions;
            out.representatives = all_representatives;
            out.ani_map         = ani_to_rep_map;
            out.outliers        = contam_records;
            out.failed_genomes  = std::move(geodesic_failed_records);
            out.member_to_rep     = std::move(geo_member_to_rep);
            out.member_nn_dist    = std::move(geo_member_nn_dist);
            out.member_fill_ratio = std::move(geo_member_fill_ratio);
            out.rep_cluster_size  = std::move(geo_cluster_size);
            out.rep_embeddings  = std::move(geo_rep_embeddings);
            out.sketch_kmer_used = cfg.kmer_size;
            // Pipeline health
            out.n_input                 = static_cast<int>(all_accessions.size());
            out.n_preflight_excluded    = static_cast<int>(preflagged_fragmented.size());
            out.n_quality_floor_excluded = n_quality_floor;
            out.n_outliers_excluded     = n_olr_excl;
            out.n_outliers_retained     = n_olr_ret;
            out.mst_p90_edge            = nn.mst_max_edge;
            out.mst_true_max            = nn.mst_true_max;
            out.ani_threshold_used      = dist_to_ani(static_cast<double>(diversity_threshold));

            run_state->push(std::move(out));
        }

        // GRD: write per-genome embeddings + metadata directly to archive
        if (grd_writer) {
            const auto& embeds = geodesic.embeddings();
            const auto& comp_ids = geodesic.component_ids();
            const size_t n_emb = embeds.size();
            const uint32_t edim = embeds.empty() ? 0u
                : static_cast<uint32_t>(embeds[0].vector.size());

            grd::TaxonData td;
            td.taxonomy = taxon.taxonomy;
            td.embed_dim = edim;
            td.sketch_size = static_cast<uint32_t>(sketch_size);
            td.kmer_size = static_cast<uint32_t>(kmer_size);
            td.k_conn = (nn.k_conn >= 0) ? static_cast<uint32_t>(nn.k_conn) : 0;
            td.diversity_threshold = diversity_threshold;
            td.ani_threshold = static_cast<float>(cfg.ani_threshold);
            td.mst_p90_edge = static_cast<float>(nn.mst_max_edge);
            td.mst_true_max = static_cast<float>(nn.mst_true_max);

            // Build accession index map
            std::unordered_map<std::string, uint32_t> acc_idx_map;
            acc_idx_map.reserve(all_accessions.size());
            for (uint32_t i = 0; i < static_cast<uint32_t>(all_accessions.size()); ++i)
                acc_idx_map[all_accessions[i]] = i;

            const uint32_t n_acc = static_cast<uint32_t>(all_accessions.size());
            td.accessions = all_accessions;

            // Status
            td.status.resize(n_acc, grd::GenomeStatus::MEMBER);
            for (const auto& acc : all_representatives) {
                auto it = acc_idx_map.find(acc);
                if (it != acc_idx_map.end())
                    td.status[it->second] = grd::GenomeStatus::REPRESENTATIVE;
            }
            for (const auto& c : contam_records) {
                if (!c.excluded) continue;
                auto it = acc_idx_map.find(c.accession);
                if (it != acc_idx_map.end())
                    td.status[it->second] = grd::GenomeStatus::CONTAMINATED;
            }

            // Per-genome arrays
            td.embeddings.resize(static_cast<size_t>(n_acc) * edim, 0.0f);
            td.component_id.resize(n_acc, 0);
            td.nearest_rep_idx.resize(n_acc, UINT32_MAX);
            td.nearest_rep_dist.resize(n_acc, 0.0f);
            td.outlier_zscore.resize(n_acc, 0.0f);
            td.genome_length.resize(n_acc, 0);

            for (size_t ei = 0; ei < n_emb; ++ei) {
                auto pit = acc_idx_map.find(embeds[ei].accession);
                if (pit == acc_idx_map.end()) continue;
                uint32_t ai = pit->second;
                if (edim > 0 && embeds[ei].vector.size() == edim)
                    std::memcpy(td.embeddings.data() + static_cast<size_t>(ai) * edim,
                                embeds[ei].vector.data(), edim * sizeof(float));
                if (ei < comp_ids.size())
                    td.component_id[ai] = static_cast<uint32_t>(comp_ids[ei]);
                td.genome_length[ai] = embeds[ei].genome_size;
            }

            for (const auto& c : contam_records) {
                auto ait = acc_idx_map.find(c.accession);
                if (ait != acc_idx_map.end()) {
                    td.outlier_zscore[ait->second] =
                        static_cast<float>(c.genome_size_zscore);
                    if (td.genome_length[ait->second] == 0)
                        td.genome_length[ait->second] = c.genome_length_bp;
                }
            }

            // Edges (source/target are accessions directly)
            for (const auto& e : edges) {
                auto mi = acc_idx_map.find(e.source);
                auto ri = acc_idx_map.find(e.target);
                if (mi == acc_idx_map.end() || ri == acc_idx_map.end()) continue;

                float dist = static_cast<float>(
                    std::acos(std::clamp(e.weight_raw, 0.0, 1.0)) / M_PI);
                td.edges.push_back({mi->second, ri->second, dist});

                if (td.nearest_rep_idx[mi->second] == UINT32_MAX ||
                    dist < td.nearest_rep_dist[mi->second]) {
                    td.nearest_rep_idx[mi->second] = ri->second;
                    td.nearest_rep_dist[mi->second] = dist;
                }
            }

            grd_writer->write_taxon(td);
        }

        return r;

    } catch (const std::exception& e) {
        spdlog::error("[{}] failed: {}", taxon.taxonomy, e.what());
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
    int acquired_threads,
    const std::unordered_map<std::string, GuncQuality>* gunc_scores,
    IPackReader* gpk_reader,
    RunState* run_state,
    grd::GrdWriter* grd_writer,
    std::function<void(TaxonResult&&)> on_result) {
    const int n = static_cast<int>(taxa.size());
    const int nthreads = std::max(1, std::min<int>(acquired_threads, n));
    std::vector<TaxonResult> results;
    if (!on_result) results.resize(n);
    // Parallelise over taxa: each uses 1 thread internally; genopack visit_sketch_batches
    // is thread-safe (concurrent callback), run_state and grd_writer are mutex-protected.
    // nthreads matches the budget actually acquired by the caller — launching cfg.threads
    // unconditionally oversubscribes the CPU when tiny batches overlap with large-taxon tasks.
#if GEODESIC_USE_OMP
    #pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
#endif
    for (int i = 0; i < n; ++i) {
        TaxonResult r = process_taxon(*taxa[i], cfg, 1, gunc_scores,
                                      gpk_reader, run_state, grd_writer);
        if (on_result) on_result(std::move(r));
        else           results[i] = std::move(r);
    }
    return results;
}

} // namespace derep
