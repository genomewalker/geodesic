#pragma once
#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <string>
#include <unistd.h>

namespace derep {

class ProgressBar {
public:
    ProgressBar(size_t total, std::string prefix = "", size_t update_interval = 500)
        : total_(total), prefix_(std::move(prefix)), update_interval_(update_interval),
          is_tty_(isatty(STDERR_FILENO)), start_time_(std::chrono::steady_clock::now()) {}

    void update(size_t current) {
        if (current % update_interval_ != 0 && current != total_) return;

        std::lock_guard<std::mutex> lock(mutex_);
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time_).count();
        double rate = current / std::max(elapsed, 0.001);
        double eta = (total_ - current) / std::max(rate, 0.001);

        if (is_tty_) {
            // Carriage return for in-place update
            fprintf(stderr, "\r%s: %zu/%zu (%.1f/s, ETA %.0fs)   ",
                    prefix_.c_str(), current, total_, rate, eta);
            fflush(stderr);
        }
        last_current_ = current;
    }

    void finish() {
        std::lock_guard<std::mutex> lock(mutex_);
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time_).count();

        if (is_tty_) {
            fprintf(stderr, "\r%s: %zu/%zu done in %.1fs                    \n",
                    prefix_.c_str(), total_, total_, elapsed);
            fflush(stderr);
        }
    }

private:
    size_t total_;
    std::string prefix_;
    size_t update_interval_;
    bool is_tty_;
    std::chrono::steady_clock::time_point start_time_;
    std::mutex mutex_;
    size_t last_current_ = 0;
};

// Thread-safe progress counter with carriage-return display
class ProgressCounter {
public:
    ProgressCounter(size_t total, std::string prefix = "Progress")
        : bar_(total, std::move(prefix)), count_(0) {}

    void increment() {
        size_t current = ++count_;
        bar_.update(current);
    }

    void finish() {
        bar_.finish();
    }

private:
    ProgressBar bar_;
    std::atomic<size_t> count_;
};

} // namespace derep
