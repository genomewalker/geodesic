#pragma once
#include <filesystem>
#include <string>

namespace derep::db {
class DBManager;
}

namespace derep {

class ResultsWriter {
public:
    explicit ResultsWriter(std::filesystem::path output_dir, std::string prefix);

    void write_derep_genomes(db::DBManager& db) const;
    void write_stats(db::DBManager& db) const;
    void write_diversity_stats(db::DBManager& db) const;
    void write_results(db::DBManager& db) const;
    void write_failed(db::DBManager& db) const;
    void write_contamination(db::DBManager& db) const;
    void write_all(db::DBManager& db) const;

private:
    std::filesystem::path output_dir_;
    std::string prefix_;
};

} // namespace derep
