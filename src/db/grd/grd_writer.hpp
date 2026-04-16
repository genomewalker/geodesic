// grd_writer.hpp — GRD write API
//
// Usage:
//   GrdWriter w("results.grd");
//   w.write_taxon(taxon_data);  // per completed taxon (thread-safe via internal mutex)
//   w.close();                  // writes global sections + TOC + TailLocator
//
#pragma once
#include "grd.hpp"
#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace grd {

// Input data for one taxon's write_taxon() call.
struct TaxonData {
    std::string taxonomy;

    // Per-genome arrays (all parallel, size = n_genomes)
    std::vector<std::string> accessions;
    std::vector<GenomeStatus> status;
    std::vector<uint32_t> nearest_rep_idx;   // index into this taxon's genome array
    std::vector<float> nearest_rep_dist;     // geodesic distance to nearest rep
    std::vector<uint32_t> component_id;
    std::vector<float> outlier_zscore;
    std::vector<uint64_t> genome_length;

    // Embeddings: n_genomes × embed_dim, row-major
    uint32_t embed_dim = 256;
    std::vector<float> embeddings;           // size = n_genomes * embed_dim

    // Edges: rep→member assignments
    std::vector<EdgeEntry> edges;

    // Taxon-level parameters
    uint32_t sketch_size = 10000;
    uint32_t kmer_size = 21;
    uint32_t k_conn = 0;
    float diversity_threshold = 0.0f;
    float ani_threshold = 0.0f;
    float mst_p90_edge = 0.0f;
    float mst_true_max = 0.0f;
};

class GrdWriter {
public:
    explicit GrdWriter(const std::filesystem::path& path);
    ~GrdWriter();

    GrdWriter(const GrdWriter&) = delete;
    GrdWriter& operator=(const GrdWriter&) = delete;

    // Write a complete taxon. Thread-safe (serialized internally).
    void write_taxon(const TaxonData& data);

    // Finalize: write TIDX, TREE, ACCX, STRT, TOC, TailLocator.
    void close();

    bool is_closed() const { return closed_; }

private:
    // Write helpers
    void write_file_header();
    std::vector<uint8_t> compress(const void* data, size_t size) const;
    uint64_t write_section(uint32_t type, uint64_t section_id,
                           const void* data, size_t size, uint64_t item_count,
                           uint64_t aux0 = 0, uint64_t aux1 = 0);
    void append_bytes(const void* data, size_t len);
    void pwrite_bytes(const void* data, size_t len, uint64_t offset);

    // 3D PCA projection from embeddings
    static std::vector<float> compute_pca3(const float* embeddings,
                                           uint32_t n, uint32_t dim);

    // String table
    uint32_t intern_string(const std::string& s);

    int      fd_       = -1;
    uint64_t offset_   = 0;
    uint32_t next_id_  = 0;
    bool     closed_   = false;
    std::mutex mu_;

    // Accumulate section descriptors for TOC
    std::vector<SectionDesc> sections_;

    // String table
    std::vector<std::string> strings_;
    std::unordered_map<std::string, uint32_t> string_index_;

    // Taxon directory entries (for TIDX)
    std::vector<TaxonIndexEntry> taxon_index_;

    // Cross-taxon accession index (for ACCX)
    std::vector<AccessionIndexEntry> accession_index_;

    // Taxonomy strings for tree building
    std::vector<std::string> taxonomy_strings_;

    // Global counters
    uint64_t total_genomes_ = 0;
};

} // namespace grd
