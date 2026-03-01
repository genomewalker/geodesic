#pragma once
#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <spdlog/spdlog.h>
#include <string>
#include <unistd.h>

namespace derep {

// Global verbosity level: 0=quiet, 1=normal, 2=verbose, 3=debug
inline std::atomic<int> g_verbosity{1};

// Set global verbosity
// 0=quiet (errors only), 1=normal (info+), 2=verbose (debug+), 3=debug (trace+)
// NOTE: do NOT call spdlog::set_level() here — sink-level filtering is
// configured per-sink in setup_logging(). Calling spdlog::set_level() would
// override the logger-level set there, causing the file sink to miss INFO messages.
inline void set_verbosity(int level) {
    g_verbosity.store(level);
}

// Check verbosity
inline bool is_verbose() { return g_verbosity.load() >= 2; }
inline bool is_quiet() { return g_verbosity.load() == 0; }
inline bool is_debug() { return g_verbosity.load() >= 3; }

// Thread-safe progress aggregator for multi-worker scenarios
class MultiWorkerProgress {
public:
    MultiWorkerProgress(int num_workers, size_t total_items, std::string prefix = "Progress")
        : num_workers_(num_workers), total_(total_items), prefix_(std::move(prefix)),
          is_tty_(isatty(STDERR_FILENO)), start_time_(std::chrono::steady_clock::now()) {}

    void update(int worker_id, size_t worker_done, size_t worker_total) {
        std::lock_guard<std::mutex> lock(mutex_);
        worker_progress_[worker_id] = {worker_done, worker_total};

        // Only update display every 100ms to reduce flicker
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - last_update_).count() < 0.1) {
            return;
        }
        last_update_ = now;

        // Aggregate progress across all workers
        size_t total_done = 0;
        size_t total_items = 0;
        int active_workers = 0;
        for (const auto& [wid, prog] : worker_progress_) {
            total_done += prog.first;
            total_items += prog.second;
            if (prog.first < prog.second) active_workers++;
        }

        double elapsed = std::chrono::duration<double>(now - start_time_).count();
        double rate = total_done / std::max(elapsed, 0.001);

        if (is_tty_ && g_verbosity.load() >= 1) {
            fprintf(stderr, "\r%s: %zu items, %d active workers (%.1f/s)    ",
                    prefix_.c_str(), total_done, active_workers, rate);
            fflush(stderr);
        }
    }

    void finish() {
        std::lock_guard<std::mutex> lock(mutex_);
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time_).count();

        size_t total_done = 0;
        for (const auto& [wid, prog] : worker_progress_) {
            total_done += prog.first;
        }

        if (is_tty_ && g_verbosity.load() >= 1) {
            fprintf(stderr, "\r%s: %zu items completed in %.1fs              \n",
                    prefix_.c_str(), total_done, elapsed);
            fflush(stderr);
        }
    }

private:
    int num_workers_;
    size_t total_;
    std::string prefix_;
    bool is_tty_;
    std::chrono::steady_clock::time_point start_time_;
    std::chrono::steady_clock::time_point last_update_;
    std::mutex mutex_;
    std::unordered_map<int, std::pair<size_t, size_t>> worker_progress_;
};

// Per-taxon logger with worker prefix
class TaxonLogger {
public:
    TaxonLogger(int worker_id, const std::string& taxonomy)
        : worker_id_(worker_id), taxonomy_(short_taxonomy(taxonomy)) {}

    template<typename... Args>
    void info(const char* fmt, Args&&... args) {
        if (g_verbosity.load() >= 2) {
            spdlog::info("[W{}:{}] {}", worker_id_, taxonomy_,
                        fmt::format(fmt, std::forward<Args>(args)...));
        }
    }

    template<typename... Args>
    void progress(const char* fmt, Args&&... args) {
        // Progress messages only in verbose mode
        if (g_verbosity.load() >= 2) {
            spdlog::info("[W{}:{}] {}", worker_id_, taxonomy_,
                        fmt::format(fmt, std::forward<Args>(args)...));
        }
    }

    template<typename... Args>
    void result(const char* fmt, Args&&... args) {
        // Results always shown (unless quiet)
        if (g_verbosity.load() >= 1) {
            spdlog::info("[{}] {}", taxonomy_,
                        fmt::format(fmt, std::forward<Args>(args)...));
        }
    }

private:
    int worker_id_;
    std::string taxonomy_;

    static std::string short_taxonomy(const std::string& full) {
        // Extract species name from full taxonomy
        auto pos = full.rfind("s__");
        if (pos != std::string::npos) {
            auto species = full.substr(pos + 3);
            if (species.length() > 25) {
                species = species.substr(0, 22) + "...";
            }
            return species;
        }
        if (full.length() > 25) {
            return full.substr(0, 22) + "...";
        }
        return full;
    }
};

} // namespace derep
