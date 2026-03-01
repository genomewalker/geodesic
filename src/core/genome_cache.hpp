#pragma once
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace derep {

class GenomeCache {
public:
    using LengthLoader = std::function<std::uint64_t(const std::filesystem::path&)>;

    struct Stats {
        std::uint64_t hits = 0;
        std::uint64_t misses = 0;
        std::size_t entries = 0;
    };

    GenomeCache();
    explicit GenomeCache(LengthLoader loader);

    [[nodiscard]] std::uint64_t get(const std::filesystem::path& path);
    void prefetch(const std::vector<std::filesystem::path>& paths);
    [[nodiscard]] bool contains(const std::filesystem::path& path) const;
    void clear();
    [[nodiscard]] Stats stats() const noexcept;

private:
    [[nodiscard]] static std::string normalize_key(const std::filesystem::path& p);

    LengthLoader loader_;
    mutable std::shared_mutex mutex_;
    std::unordered_map<std::string, std::uint64_t> cache_;
    std::atomic<std::uint64_t> hits_{0};
    std::atomic<std::uint64_t> misses_{0};
};

} // namespace derep
