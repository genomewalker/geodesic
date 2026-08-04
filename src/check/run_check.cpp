#include "run_check.hpp"
#include "../core/multi_pack_reader.hpp"
#include "../core/pack_reader.hpp"
#include <genopack/archive.hpp>
#include <genopack/qual.hpp>
#include <spdlog/spdlog.h>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

namespace derep {

namespace fs = std::filesystem;

static const char* mimag_tier(float completeness, float contamination) {
    if (std::isnan(completeness) || std::isnan(contamination)) return "NA";
    if (completeness >= 0.90f && contamination < 0.05f) return "HQ";
    if (completeness >= 0.50f && contamination < 0.10f) return "MQ";
    return "LQ";
}

int run_check(const Config& cfg) {
    if (!cfg.pack_dir.has_value())
        throw std::runtime_error("--pack is required");

    const auto& pack_path = *cfg.pack_dir;
    std::unique_ptr<IPackReader> pack;
    if (pack_path.extension() == ".gpk") {
        auto ar = std::make_unique<genopack::ArchiveReader>();
        ar->open(pack_path);
        pack = std::make_unique<SinglePackReader>(std::move(ar));
    } else {
        pack = MultiPackReader::open_dir(pack_path);
    }

    if (!pack->has_qual()) {
        spdlog::warn("check: archive has no QUAL section — run 'genopack check' first");
        return 1;
    }

    // Build genome_id → accession map via combined scan
    std::unordered_map<genopack::GenomeId, std::string> id_to_acc;
    pack->scan_taxonomy_with_id([&](std::string_view acc, std::string_view /*tax*/,
                                    genopack::GenomeId gid) {
        id_to_acc.emplace(gid, std::string(acc));
    });

    // Collect QUAL records
    std::vector<genopack::QualRecord> records;
    pack->scan_qual([&](const genopack::QualRecord& r) {
        records.push_back(r);
    });

    spdlog::info("check: {} genomes, {} QUAL records", id_to_acc.size(), records.size());

    // Output
    const bool to_stdout = cfg.validate_output.empty();
    std::ofstream fout;
    if (!to_stdout) {
        fout.open(cfg.validate_output);
        if (!fout) throw std::runtime_error("cannot open output: " + cfg.validate_output.string());
    }
    std::ostream& out = to_stdout ? std::cout : fout;

    out << "accession"
        << "\tcompleteness_cluster_relative"
        << "\tcompleteness_post_decontam"
        << "\tfmh_contamination"
        << "\tcontamination_leakage"
        << "\tcontamination_contig_outlier"
        << "\tcontamination_cross_genus"
        << "\tquality_tier"      // genopack's authoritative tier (completeness-decoupled)
        << "\tmimag_tier"        // derived classic MIMAG view (completeness AND contamination)
        << "\tqual_flags"
        << "\n";

    int flagged = 0;
    for (const auto& r : records) {
        auto it = id_to_acc.find(r.genome_id);
        if (it == id_to_acc.end()) continue;

        const float comp = !std::isnan(r.completeness_post_decontam)
                           ? r.completeness_post_decontam
                           : r.completeness_cluster_relative;
        const float fmh  = r.fmh_minority_u8 > 0 ? r.fmh_minority_u8 / 255.0f : NAN;
        const float cont = r.contamination_leakage;
        const float cco  = r.contig_outlier_u8 / 255.0f;
        const float cg   = r.cross_genus_u8 / 255.0f;
        // Use FMH as primary contamination signal for tier; fall back to leakage if FMH not scored
        const float cont_primary = !std::isnan(fmh) ? fmh : cont;
        const char* tier = mimag_tier(comp, cont_primary);
        // genopack's authoritative tier from the QUAL section (do not recompute).
        const char* gp_tier =
            r.quality_tier_u8 == genopack::QualRecord::QTIER_HQ ? "HQ" :
            r.quality_tier_u8 == genopack::QualRecord::QTIER_MQ ? "MQ" :
            r.quality_tier_u8 == genopack::QualRecord::QTIER_LQ ? "LQ" : "NA";

        auto fmt_f = [](float v) -> std::string {
            if (std::isnan(v)) return "NA";
            char buf[32];
            snprintf(buf, sizeof(buf), "%.6f", static_cast<double>(v));
            return buf;
        };

        out << it->second
            << '\t' << fmt_f(r.completeness_cluster_relative)
            << '\t' << fmt_f(r.completeness_post_decontam)
            << '\t' << fmt_f(fmh)
            << '\t' << fmt_f(cont)
            << '\t' << fmt_f(cco)
            << '\t' << fmt_f(cg)
            << '\t' << gp_tier
            << '\t' << tier
            << '\t' << static_cast<int>(r.qual_flags)
            << '\n';

        if (!std::isnan(cont_primary) && cont_primary > cfg.check_leakage_threshold)
            ++flagged;
    }

    if (!to_stdout)
        spdlog::info("check: wrote {} records to {} ({} flagged leakage > {:.0f}%)",
                     records.size(), cfg.validate_output.string(),
                     flagged, cfg.check_leakage_threshold * 100.0f);
    return 0;
}

} // namespace derep
