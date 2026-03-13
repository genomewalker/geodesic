#pragma once
#include "db/db_manager.hpp"
#include "db/operations.hpp"
#include "core/types.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>
#include <spdlog/spdlog.h>

namespace derep::db {

// All data needed to persist one taxon's results
struct TaxonWritePayload {
    // Core result
    TaxonResult result;
    TaxonDiversityStats diversity_stats;

    // Genome assignments (for genomes_derep table)
    std::vector<std::string> all_accessions;
    std::vector<std::string> representatives;
    std::unordered_map<std::string, double> ani_map;

    // Contamination candidates
    std::vector<ops::ContaminationRecord> contamination;

    // Jobs done (genome accessions that completed)
    std::vector<Genome> completed_genomes;
};

// Async DB writer with micro-batching
// Workers push TaxonWritePayload; writer thread batches and flushes
class AsyncDBWriter {
public:
    struct Config {
        size_t max_taxa_per_batch = 500;        // Flush after this many taxa
        size_t max_rows_per_batch = 100000;     // Flush after this many genomes_derep rows
        size_t flush_interval_ms = 500;         // Flush after this much idle time
        size_t queue_capacity = 2000;           // Max pending payloads before blocking
    };

    explicit AsyncDBWriter(DBManager& db, Config cfg = {})
        : db_(db), cfg_(cfg), running_(true), total_taxa_(0), total_rows_(0) {
        writer_thread_ = std::thread(&AsyncDBWriter::writer_loop, this);
    }

    ~AsyncDBWriter() {
        shutdown();
    }

    // Push a payload (may block if queue is full)
    void push(TaxonWritePayload payload) {
        std::unique_lock lock(mutex_);
        cv_not_full_.wait(lock, [this] {
            return queue_.size() < cfg_.queue_capacity || !running_;
        });
        if (!running_) return;
        queue_.push_back(std::move(payload));
        cv_not_empty_.notify_one();
    }

    // Flush all pending and stop
    void shutdown() {
        {
            std::lock_guard lock(mutex_);
            running_ = false;
        }
        cv_not_empty_.notify_one();
        cv_not_full_.notify_all();
        if (writer_thread_.joinable()) {
            writer_thread_.join();
        }
    }

    // Wait for queue to drain (for progress reporting)
    void wait_until_drained() {
        std::unique_lock lock(mutex_);
        cv_drained_.wait(lock, [this] { return queue_.empty(); });
    }

    size_t pending() const {
        std::lock_guard lock(mutex_);
        return queue_.size();
    }

    size_t total_taxa_written() const { return total_taxa_.load(); }
    size_t total_rows_written() const { return total_rows_.load(); }

private:
    void writer_loop() {
        std::vector<TaxonWritePayload> batch;
        batch.reserve(cfg_.max_taxa_per_batch);

        while (true) {
            batch.clear();
            size_t row_count = 0;

            {
                std::unique_lock lock(mutex_);

                // Wait for data or timeout
                auto deadline = std::chrono::steady_clock::now() +
                                std::chrono::milliseconds(cfg_.flush_interval_ms);
                cv_not_empty_.wait_until(lock, deadline, [this] {
                    return !queue_.empty() || !running_;
                });

                if (!running_ && queue_.empty()) break;

                // Collect batch
                while (!queue_.empty() &&
                       batch.size() < cfg_.max_taxa_per_batch &&
                       row_count < cfg_.max_rows_per_batch) {
                    row_count += queue_.front().all_accessions.size();
                    batch.push_back(std::move(queue_.front()));
                    queue_.pop_front();
                }

                if (!queue_.empty()) {
                    cv_not_empty_.notify_one();  // More work available
                }
            }

            cv_not_full_.notify_all();  // Space available

            if (!batch.empty()) {
                flush_batch(batch);
                total_taxa_ += batch.size();
                total_rows_ += row_count;
            }

            // Signal if fully drained
            {
                std::lock_guard lock(mutex_);
                if (queue_.empty()) {
                    cv_drained_.notify_all();
                }
            }
        }

        // Final flush on shutdown
        std::vector<TaxonWritePayload> final_batch;
        {
            std::lock_guard lock(mutex_);
            while (!queue_.empty()) {
                final_batch.push_back(std::move(queue_.front()));
                queue_.pop_front();
            }
        }
        if (!final_batch.empty()) {
            flush_batch(final_batch);
            total_taxa_ += final_batch.size();
        }
        cv_drained_.notify_all();
    }

    void flush_batch(const std::vector<TaxonWritePayload>& batch) {
        auto& conn = db_.connection();

        // Single transaction for entire batch
        conn.Query("BEGIN TRANSACTION");

        try {
            // 1. Batch DELETE for results/diversity_stats (if overwriting)
            //    Using IN clause is much faster than per-taxon deletes
            if (!batch.empty()) {
                std::string taxa_list;
                for (size_t i = 0; i < batch.size(); ++i) {
                    if (i > 0) taxa_list += ",";
                    taxa_list += "'" + batch[i].result.taxonomy + "'";
                }
                conn.Query("DELETE FROM results WHERE taxonomy IN (" + taxa_list + ")");
                conn.Query("DELETE FROM diversity_stats WHERE taxonomy IN (" + taxa_list + ")");
                conn.Query("DELETE FROM genomes_derep WHERE taxonomy IN (" + taxa_list + ")");
                conn.Query("DELETE FROM contamination_candidates WHERE taxonomy IN (" + taxa_list + ")");
            }

            // 2. Bulk INSERT results
            {
                duckdb::Appender app(conn, "results");
                for (const auto& p : batch) {
                    const auto& r = p.result;
                    app.BeginRow();
                    app.Append(r.taxonomy);
                    app.Append(static_cast<int64_t>(r.n_genomes));
                    app.Append(static_cast<int64_t>(r.n_representatives));
                    app.Append(static_cast<int>(r.status));
                    app.Append(r.method);
                    app.Append(static_cast<int64_t>(r.n_communities));
                    app.Append(r.error_message.empty() ? duckdb::Value() : duckdb::Value(r.error_message));
                    app.EndRow();
                }
                app.Close();
            }

            // 3. Bulk INSERT diversity_stats
            {
                duckdb::Appender app(conn, "diversity_stats");
                for (const auto& p : batch) {
                    const auto& s = p.diversity_stats;
                    app.BeginRow();
                    app.Append(s.taxonomy);
                    app.Append(s.method);
                    app.Append(s.n_genomes);
                    app.Append(s.n_representatives);
                    app.Append(s.reduction_ratio);
                    app.Append(s.runtime_seconds);
                    app.Append(s.coverage_mean_ani);
                    app.Append(s.coverage_min_ani);
                    app.Append(s.coverage_max_ani);
                    app.Append(s.coverage_below_99);
                    app.Append(s.coverage_below_98);
                    app.Append(s.coverage_below_97);
                    app.Append(s.coverage_below_95);
                    app.Append(s.diversity_mean_ani);
                    app.Append(s.diversity_min_ani);
                    app.Append(s.diversity_max_ani);
                    app.Append(s.diversity_ani_range);
                    app.Append(s.diversity_n_pairs);
                    app.Append(s.n_contaminated);
                    app.EndRow();
                }
                app.Close();
            }

            // 4. Bulk INSERT genomes_derep (the big one)
            {
                duckdb::Appender app(conn, "genomes_derep");
                for (const auto& p : batch) {
                    std::unordered_set<std::string> rep_set(
                        p.representatives.begin(), p.representatives.end());
                    for (const auto& acc : p.all_accessions) {
                        app.BeginRow();
                        app.Append(acc);
                        app.Append(p.result.taxonomy);
                        app.Append(rep_set.count(acc) > 0);
                        auto it = p.ani_map.find(acc);
                        if (it != p.ani_map.end()) {
                            app.Append(it->second);
                        } else {
                            app.Append(duckdb::Value());
                        }
                        app.EndRow();
                    }
                }
                app.Close();
            }

            // 5. Bulk INSERT contamination_candidates
            {
                duckdb::Appender app(conn, "contamination_candidates");
                for (const auto& p : batch) {
                    for (const auto& c : p.contamination) {
                        app.BeginRow();
                        app.Append(c.accession);
                        app.Append(p.result.taxonomy);
                        app.Append(c.centroid_distance);
                        app.Append(c.isolation_score);
                        app.Append(c.anomaly_score);
                        app.Append(c.genome_size_zscore);
                        app.Append(c.nn_outlier);
                        app.Append(c.kmer_div_zscore);
                        app.Append(c.margin_to_threshold);
                        app.Append(c.flag_reason);
                        app.EndRow();
                    }
                }
                app.Close();
            }

            // 6. Bulk INSERT jobs_done
            {
                duckdb::Appender app(conn, "jobs_done");
                for (const auto& p : batch) {
                    for (const auto& g : p.completed_genomes) {
                        app.BeginRow();
                        app.Append(g.accession);
                        app.Append(g.taxonomy);
                        app.EndRow();
                    }
                }
                app.Close();
            }

            conn.Query("COMMIT");
        } catch (const std::exception& e) {
            conn.Query("ROLLBACK");
            spdlog::error("AsyncDBWriter batch flush failed: {}", e.what());
            throw;
        }
    }

    DBManager& db_;
    Config cfg_;
    std::thread writer_thread_;

    mutable std::mutex mutex_;
    std::condition_variable cv_not_empty_;
    std::condition_variable cv_not_full_;
    std::condition_variable cv_drained_;
    std::deque<TaxonWritePayload> queue_;

    std::atomic<bool> running_;
    std::atomic<size_t> total_taxa_;
    std::atomic<size_t> total_rows_;
};

} // namespace derep::db
