// geodf_reader.hpp — GEODF read API
//
// Crash recovery: on open(), scan TaxonHeaders to find all completed taxa.
// Provides O(log n) lookup by taxonomy string.
//
#pragma once
#include "geodf.hpp"
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace geodf {

struct TaxonData {
    std::string              taxonomy;
    std::vector<uint32_t>    genome_ids;
    std::vector<bool>        is_rep;
    std::vector<float>       contamination;
    std::vector<std::string> all_accessions;  // parallel to genome_ids (all genomes)
    std::vector<std::string> rep_accessions;  // rep accessions (subset, for compatibility)
    std::vector<std::vector<float>> rep_embeddings;
    PipelineStage            stage;
    float                    diversity_threshold;
    float                    ani_threshold;
    uint32_t                 taxon_id;
    std::string              error_message;   // non-empty when stage == FAILED
};

class GeodfReader {
public:
    explicit GeodfReader(const std::filesystem::path& path);
    ~GeodfReader();

    GeodfReader(const GeodfReader&) = delete;
    GeodfReader& operator=(const GeodfReader&) = delete;

    // Returns set of all taxonomy strings that have stage == COMPLETE.
    std::unordered_set<std::string> completed_taxa() const;

    // Returns all TaxonHeaders without decompressing payloads (fast scan).
    std::vector<std::pair<TaxonHeader, std::string>> scan_headers() const;

    // Lookup by taxonomy string. Returns nullopt if not found.
    std::optional<TaxonData> find(const std::string& taxonomy) const;

    // Iterate all complete taxa (sequential scan, decompresses payloads).
    void for_each_complete(const std::function<void(const TaxonData&)>& cb) const;

    size_t n_taxa() const { return index_.size(); }

    uint64_t gpk_snapshot_id() const { return file_header_.gpk_snapshot_id; }
    uint32_t params_hash()     const { return file_header_.params_hash; }

private:
    TaxonData   decompress_taxon(const TaxonHeader& hdr) const;
    std::string lookup_string(uint32_t id) const;
    std::string lookup_recovered(size_t idx) const;
    std::string resolve_taxonomy(const TaxonHeader& hdr) const;
    void        load_strings() const;
    void        load_index();
    void        scan_and_recover(uint64_t file_size);

    int                                fd_      = -1;
    std::vector<TaxonIndexEntry>       index_;
    mutable std::vector<std::string>   strings_;
    mutable bool                       strings_loaded_ = false;
    FileTrailer                        trailer_{};
    FileHeader                         file_header_{};
    // Crash-recovery mode: maps header_offset → taxonomy string (stable after sort)
    std::unordered_map<uint64_t, std::string> recovered_strings_;
};

} // namespace geodf
