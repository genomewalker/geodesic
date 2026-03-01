#include "db/db_manager.hpp"
#include <stdexcept>
#include <spdlog/spdlog.h>

namespace derep::db {

DBManager::DBManager(Options options) : options_(std::move(options)) {
    spdlog::debug("Opening database: {}", options_.db_path.string());
    duckdb::DBConfig cfg;
    database_ = std::make_unique<duckdb::DuckDB>(options_.db_path.string(), &cfg);
}

DBManager::~DBManager() {
    pool_.clear();
}

duckdb::Connection& DBManager::thread_connection() {
    auto tid = std::this_thread::get_id();
    std::lock_guard lock(pool_mutex_);
    auto [it, inserted] = pool_.emplace(tid, nullptr);
    if (inserted) {
        it->second = std::make_unique<duckdb::Connection>(*database_);
    }
    return *it->second;
}

std::unique_ptr<duckdb::Connection> DBManager::create_connection() const {
    return std::make_unique<duckdb::Connection>(*database_);
}

void DBManager::execute(std::string_view sql) {
    auto conn = create_connection();
    auto result = conn->Query(std::string(sql));
    if (result->HasError()) {
        throw DbError("SQL error: " + result->GetError());
    }
}

std::unique_ptr<duckdb::MaterializedQueryResult> DBManager::query(std::string_view sql) {
    auto conn = create_connection();
    auto result = conn->Query(std::string(sql));
    if (result->HasError()) {
        throw DbError("SQL error: " + result->GetError());
    }
    return result;
}

void DBManager::checkpoint() {
    execute("CHECKPOINT;");
}

void DBManager::close_thread_connection() {
    std::lock_guard lock(pool_mutex_);
    pool_.erase(std::this_thread::get_id());
}

std::size_t DBManager::pooled_connection_count() const {
    std::lock_guard lock(pool_mutex_);
    return pool_.size();
}

} // namespace derep::db
