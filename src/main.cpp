#include "config.hpp"
#include "pipeline.hpp"
#include <spdlog/spdlog.h>

int main(int argc, char** argv) {
    try {
        auto cfg = derep::parse_args(argc, argv);
        return derep::run_pipeline(cfg);
    } catch (const std::exception& e) {
        spdlog::critical("Fatal: {}", e.what());
        return 1;
    }
}
