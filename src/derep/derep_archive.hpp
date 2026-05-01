#pragma once
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace genopack { class ArchiveSetReader; }

namespace geodesic {

struct DerepArchiveBuilderConfig {
    std::filesystem::path output_path;
    uint16_t              embedding_dim;
    uint8_t               embedding_dtype; // 0=f32, 1=f16
    bool                  emit_armp  = true;
    bool                  emit_cstat = false;
    int                   zstd_level = 19;
    std::string           geodesic_version;
};

class DerepArchiveBuilder {
public:
    explicit DerepArchiveBuilder(DerepArchiveBuilderConfig cfg);
    ~DerepArchiveBuilder();

    void set_source_pack(const genopack::ArchiveSetReader& pack);

    // For tests or callers that already have fingerprint data.
    struct PartFingerprint {
        uint8_t  archive_uuid[16];
        uint64_t generation;
        uint64_t n_genomes_total;
        uint64_t n_genomes_live;
        uint64_t accession_set_hash;
    };
    void set_source_pack_manual(std::vector<PartFingerprint> parts);

    void set_params(const std::vector<uint8_t>& kmer_sizes,
                    uint32_t sketch_size,
                    uint64_t sig1_seed, uint64_t sig2_seed,
                    float    jaccard_thresh);

    enum class Kind { Representative, Member, Unclustered };

    void add(std::string_view accession,
             Kind             kind,
             std::string_view rep_accession,
             uint64_t         source_locator,
             uint16_t         sketch_kmer,
             uint32_t         cluster_size,
             const void*      embedding_or_null);

    void finalize();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace geodesic
