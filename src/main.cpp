#include "config.hpp"
#include "pipeline.hpp"
#include "db/db_manager.hpp"
#include "io/report_writer.hpp"
#include <spdlog/spdlog.h>
#include <filesystem>

int main(int argc, char** argv) {
    try {
        auto cfg = derep::parse_args(argc, argv);

        switch (cfg.command) {
        case derep::Command::Report: {
            namespace fs = std::filesystem;
            fs::path out = cfg.out_dir ? *cfg.out_dir
                                       : cfg.db_path.parent_path().empty()
                                           ? fs::path(".")
                                           : cfg.db_path.parent_path();
            fs::create_directories(out);
            derep::db::DBManager db({.db_path = cfg.db_path, .read_only = true});
            derep::ReportWriter rw(out, cfg.prefix, cfg.timestamp);
            rw.write(db);
            return 0;
        }
        case derep::Command::Sketch:
            return derep::run_sketch(cfg);
        case derep::Command::Derep:
            return derep::run_pipeline(cfg);
        }
    } catch (const std::exception& e) {
        spdlog::critical("Fatal: {}", e.what());
        return 1;
    }
}
