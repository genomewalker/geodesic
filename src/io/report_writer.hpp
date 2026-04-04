#pragma once
#include "state/run_state.hpp"
#include <filesystem>
#include <string>

namespace derep {

class ReportWriter {
public:
    ReportWriter(std::filesystem::path output_dir, std::string prefix, std::string timestamp);
    void write(const RunState& state) const;

private:
    std::filesystem::path dir_;
    std::string prefix_;
    std::string ts_;

    std::string build_json(const RunState& state) const;
};

} // namespace derep
