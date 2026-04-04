// geodf_writer.hpp — GEODF write API
//
// Usage:
//   GeodfWriter w("results.geodf");
//   w.write_taxon(result);   // call per completed taxon
//   w.close();               // writes StringTable + TaxonIndex + FileTrailer
//
// Thread safety: single writer (all calls from one thread).
// Crash safety: write_taxon() writes the TaxonHeader LAST as a completion marker.
//
#pragma once
#include "geodf.hpp"
#include <filesystem>
#include <functional>
#include <string>
#include <vector>
#include <unordered_map>

namespace geodf {

struct RepGenome {
    uint32_t             genome_id;
    std::string          accession;
    std::vector<float>   embedding;   // empty if not available
};

struct TaxonResult {
    std::string                    taxonomy;
    std::vector<uint32_t>          genome_ids;
    std::vector<bool>              is_rep;           // parallel to genome_ids
    std::vector<float>             contamination;    // parallel to genome_ids
    std::vector<std::string>       all_accessions;   // parallel to genome_ids (all genomes)
    std::vector<RepGenome>         reps;
    PipelineStage                  stage             = PipelineStage::COMPLETE;
    float                          diversity_threshold = 0.0f;
    float                          ani_threshold       = 0.0f;
    std::string                    error_message;   // set when stage == FAILED
};

class GeodfWriter {
public:
    explicit GeodfWriter(const std::filesystem::path& path);
    ~GeodfWriter();

    GeodfWriter(const GeodfWriter&) = delete;
    GeodfWriter& operator=(const GeodfWriter&) = delete;

    // Write a complete taxon result atomically.
    // Header is written last — safe to call concurrently if serialized externally.
    void write_taxon(const TaxonResult& result);

    // Set provenance fields written into the FileHeader.
    // Must be called before the first write_taxon().
    void set_provenance(uint64_t gpk_snapshot_id, uint32_t params_hash);

    // Finalize: write StringTable + TaxonIndex + FileTrailer.
    // Must be called exactly once before destruction.
    void close();

    bool is_closed() const { return closed_; }

private:
    void write_file_header();
    uint64_t write_payload(const TaxonResult& r, uint32_t strtable_off);
    void write_taxon_header(const TaxonHeader& hdr);
    uint32_t intern_string(const std::string& s);
    void write_bytes(const void* data, size_t len, uint64_t offset);
    void append_bytes(const void* data, size_t len);

    int      fd_              = -1;
    uint64_t offset_          = 0;
    uint32_t next_id_         = 0;
    bool     closed_          = false;
    uint64_t gpk_snapshot_id_ = 0;
    uint32_t params_hash_     = 0;

    // StringTable accumulator
    std::vector<std::string>               strings_;
    std::unordered_map<std::string, uint32_t> string_index_;

    // Index accumulator
    std::vector<TaxonIndexEntry>           index_;
};

} // namespace geodf
