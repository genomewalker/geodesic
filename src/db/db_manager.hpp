#pragma once
#include <duckdb.hpp>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <unordered_map>

namespace derep::db {

class DbError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class DBManager final {
public:
    struct Options {
        std::filesystem::path db_path;
        bool read_only = false;
    };

    explicit DBManager(Options options);
    ~DBManager();
    DBManager(const DBManager&) = delete;
    DBManager& operator=(const DBManager&) = delete;
    DBManager(DBManager&&) = delete;
    DBManager& operator=(DBManager&&) = delete;

    [[nodiscard]] duckdb::Connection& thread_connection();
    [[nodiscard]] std::unique_ptr<duckdb::Connection> create_connection() const;
    void execute(std::string_view sql);
    [[nodiscard]] std::unique_ptr<duckdb::MaterializedQueryResult> query(std::string_view sql);
    void checkpoint();
    void close_thread_connection();
    [[nodiscard]] std::size_t pooled_connection_count() const;

private:
    struct ThreadIdHash {
        std::size_t operator()(const std::thread::id& id) const noexcept {
            return std::hash<std::thread::id>{}(id);
        }
    };

    Options options_;
    std::unique_ptr<duckdb::DuckDB> database_;
    mutable std::mutex pool_mutex_;
    std::unordered_map<std::thread::id,
                       std::unique_ptr<duckdb::Connection>,
                       ThreadIdHash> pool_;
};

} // namespace derep::db
