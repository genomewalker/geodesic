#include "config.hpp"
#include "distributed.hpp"
#include "pipeline.hpp"
#include <spdlog/spdlog.h>

int main(int argc, char** argv) {
    try {
        auto cfg = derep::parse_args(argc, argv);

        switch (cfg.command) {
        case derep::Command::Derep:
            return derep::run_pipeline(cfg);
        case derep::Command::Update:
            return derep::run_update(cfg);
        case derep::Command::Scatter:
            return derep::run_scatter(cfg);
        case derep::Command::Gather:
            return derep::run_gather(cfg);
        }
    } catch (const std::exception& e) {
        spdlog::critical("Fatal: {}", e.what());
        return 1;
    }
}
