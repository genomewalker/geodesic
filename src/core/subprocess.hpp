#pragma once
#include <chrono>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace derep {

class SubprocessError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct SubprocessOptions {
    std::optional<std::filesystem::path> working_dir;
    std::optional<std::filesystem::path> stdout_file;
    bool capture_stdout = false;
    bool capture_stderr = true;
    std::chrono::seconds timeout{3600};
    std::unordered_map<std::string, std::string> env_overrides;
    bool clear_environment = false;
};

struct SubprocessResult {
    int exit_code = -1;
    bool timed_out = false;
    std::string stdout_output;
    std::string stderr_output;
    std::chrono::milliseconds wall_time{0};
    [[nodiscard]] bool ok() const noexcept { return !timed_out && exit_code == 0; }
};

SubprocessResult run_subprocess(
    const std::vector<std::string>& argv,
    const SubprocessOptions& options = {});

} // namespace derep
