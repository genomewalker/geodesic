#pragma once
#include "io/tsv_reader.hpp"
#include "state/run_state.hpp"
#include <filesystem>
#include <string>

namespace derep {

class ResultsWriter {
public:
    explicit ResultsWriter(std::filesystem::path output_dir, std::string prefix);

    void write_derep_genomes(const RunState& state, const std::vector<GenomeRow>& all_genomes) const;
    void write_stats(const RunState& state) const;
    void write_diversity_stats(const RunState& state) const;
    void write_results(const RunState& state) const;
    void write_failed(const RunState& state) const;
    void write_outliers(const RunState& state) const;
    void write_all(const RunState& state, const std::vector<GenomeRow>& all_genomes) const;

private:
    std::filesystem::path output_dir_;
    std::string prefix_;
};

} // namespace derep
