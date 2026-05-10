#include "config.hpp"
#include "distributed.hpp"
#include "pipeline.hpp"
#include <Eigen/Core>
#include <omp.h>
#include <spdlog/spdlog.h>

int main(int argc, char** argv) {
    // Bit-determinism: Eigen's internal threading reorders FMA/SIMD reductions
    // across runs. Pin to 1 thread globally; the pipeline parallelizes at the
    // taxon level (and inside Nyström project_pass2 via OMP), so Eigen's nested
    // pool would only oversubscribe anyway.
    Eigen::setNbThreads(1);
    try {
        auto cfg = derep::parse_args(argc, argv);

        // Cap the global OMP pool to --threads. Without this, libraries that
        // use #pragma omp parallel without an explicit num_threads() clause
        // (e.g. genopack's sketch_for_ids) default to hardware_concurrency and
        // create far more OS threads than requested.
        omp_set_num_threads(cfg.threads);

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
