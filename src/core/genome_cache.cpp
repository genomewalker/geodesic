#include "genome_cache.hpp"
#include "../io/fasta_reader.hpp"

#include <algorithm>
#include <mutex>
#include <unordered_set>

#if __has_include(<execution>)
#include <execution>
#define DEREP_HAS_EXECUTION 1
#else
#define DEREP_HAS_EXECUTION 0
#endif

namespace derep {

std::string GenomeCache::normalize_key(const std::filesystem::path& p) {
    return std::filesystem::weakly_canonical(p).string();
}

GenomeCache::GenomeCache()
    : loader_{[](const std::filesystem::path& p) {
          return derep::calculate_genome_length(p);
      }} {}

GenomeCache::GenomeCache(LengthLoader loader) : loader_{std::move(loader)} {}

std::uint64_t GenomeCache::get(const std::filesystem::path& path) {
    auto key = normalize_key(path);

    {
        std::shared_lock lock(mutex_);
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            hits_.fetch_add(1, std::memory_order_relaxed);
            return it->second;
        }
    }

    misses_.fetch_add(1, std::memory_order_relaxed);
    std::uint64_t length = loader_(path);

    {
        std::unique_lock lock(mutex_);
        cache_.emplace(key, length);
    }

    return length;
}

void GenomeCache::prefetch(const std::vector<std::filesystem::path>& paths) {
    // Normalize and deduplicate
    std::vector<std::pair<std::string, std::filesystem::path>> unique_paths;
    {
        std::unordered_set<std::string> seen;
        for (const auto& p : paths) {
            auto key = normalize_key(p);
            if (seen.insert(key).second) {
                unique_paths.emplace_back(std::move(key), p);
            }
        }
    }

    // Collect missing keys
    std::vector<std::pair<std::string, std::filesystem::path>> missing;
    {
        std::shared_lock lock(mutex_);
        for (auto& [key, path] : unique_paths) {
            if (cache_.find(key) == cache_.end()) {
                missing.emplace_back(std::move(key), std::move(path));
            }
        }
    }

    if (missing.empty()) return;

    // Compute lengths outside lock
    std::vector<std::pair<std::string, std::uint64_t>> results(missing.size());

#if DEREP_HAS_EXECUTION
    std::transform(
        std::execution::par_unseq, missing.begin(), missing.end(),
        results.begin(),
        [this](const auto& entry) -> std::pair<std::string, std::uint64_t> {
            return {entry.first, loader_(entry.second)};
        });
#else
    std::transform(
        missing.begin(), missing.end(), results.begin(),
        [this](const auto& entry) -> std::pair<std::string, std::uint64_t> {
            return {entry.first, loader_(entry.second)};
        });
#endif

    // Insert all results under write lock
    {
        std::unique_lock lock(mutex_);
        for (auto& [key, length] : results) {
            cache_.insert_or_assign(std::move(key), length);
        }
    }
}

bool GenomeCache::contains(const std::filesystem::path& path) const {
    auto key = normalize_key(path);
    std::shared_lock lock(mutex_);
    return cache_.find(key) != cache_.end();
}

void GenomeCache::clear() {
    std::unique_lock lock(mutex_);
    cache_.clear();
    hits_.store(0, std::memory_order_relaxed);
    misses_.store(0, std::memory_order_relaxed);
}

GenomeCache::Stats GenomeCache::stats() const noexcept {
    Stats s;
    {
        std::shared_lock lock(mutex_);
        s.entries = cache_.size();
    }
    s.hits = hits_.load(std::memory_order_relaxed);
    s.misses = misses_.load(std::memory_order_relaxed);
    return s;
}

} // namespace derep
