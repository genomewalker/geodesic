#pragma once
#include "db/db_manager.hpp"
#include <filesystem>
#include <string>

namespace derep {

class ReportWriter {
public:
    ReportWriter(std::filesystem::path output_dir, std::string prefix, std::string timestamp);
    void write(db::DBManager& db) const;

private:
    std::filesystem::path dir_;
    std::string prefix_;
    std::string ts_;

    std::string build_json(db::DBManager& db) const;
};

} // namespace derep
