#pragma once

#include "state/run_state.hpp"
#include <filesystem>
#include <string>
#include <unordered_set>

namespace derep {

class ResultsWriter {
public:
    explicit ResultsWriter(std::filesystem::path output_dir, std::string prefix);

    void write_derep_genomes(const RunState& state) const;
    void write_stats(const RunState& state) const;
    void write_diversity_stats(const RunState& state) const;
    void write_results(const RunState& state) const;
    void write_failed(const RunState& state) const;
    void write_outliers(const RunState& state) const;
    void write_all(const RunState& state) const;

    // Append data rows (no headers) to existing output files. Used on resume
    // after load_resume() has written headers + checkpoint rows.
    void write_all_append(const RunState& state) const;

    // Write a per-wave checkpoint.  ck_key is an opaque string tag such as
    // "3_w0" (arch=3, wave=0) or "cross_w0".  A <ck_key>.done sentinel is
    // written last so a partial checkpoint is never treated as complete.
    void write_checkpoint(const RunState& state, const std::string& ck_key) const;

    // Scan .geodesic_resume/ for *.done sentinels and return their key set.
    std::unordered_set<std::string> scan_done_keys() const;

    // Write output-file headers, cat all checkpoint data rows (in arbitrary
    // order), and return the set of taxonomy strings already completed.
    std::unordered_set<std::string> load_resume() const;

private:
    std::filesystem::path output_dir_;
    std::string prefix_;

    std::filesystem::path resume_dir() const { return output_dir_ / ".geodesic_resume"; }
    std::filesystem::path ckpt_path(const std::string& ck_key,
                                    const std::string& suffix) const;

    static void cat_checkpoint_rows(std::ofstream& dest,
                                    const std::filesystem::path& src);
};

} // namespace derep
