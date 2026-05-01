#include "config.hpp"
#include "distributed.hpp"
#include "pipeline.hpp"
#include <Eigen/Core>
#include <spdlog/spdlog.h>

int main(int argc, char** argv) {
    // Bit-determinism: Eigen's internal threading reorders FMA/SIMD reductions
    // across runs. Pin to 1 thread globally; the pipeline parallelizes at the
    // taxon level (and inside Nyström project_pass2 via OMP), so Eigen's nested
    // pool would only oversubscribe anyway.
    Eigen::setNbThreads(1);
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
