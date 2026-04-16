// grd.hpp — Geodesic Results Data format
//
// Binary archive for geodesic dereplication results with per-genome embeddings.
// Designed for HTTP range-request serving to a WebGPU visualization client.
//
// File layout (mirrors genopack's section-based architecture):
//   [FileHeader]          128 bytes
//   [Section 0 data]      zstd-compressed
//   [Section 1 data]      ...
//   ...
//   [TocHeader + SectionDesc[N]]   zstd-compressed TOC block
//   [TailLocator]         64 bytes (always at EOF-64)
//
// The TailLocator at EOF enables bootstrapping: read last 64 bytes to find
// the TOC, then read the TOC to discover all sections. Works with HTTP Range
// requests against a static file on any CDN/S3 bucket.
//
#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace grd {

// ── Magic numbers ────────────────────────────────────────────────────────────

static constexpr uint32_t GRD1_MAGIC = 0x31445247u; // "GRD1"
static constexpr uint32_t GRDT_MAGIC = 0x54445247u; // "GRDT" (tail)
static constexpr uint32_t TOCB_MAGIC = 0x42434F54u; // "TOCB"

// Section type codes (4CC as uint32_t)
static constexpr uint32_t SEC_TMTA = 0x41544D54u; // "TMTA" — taxon metadata
static constexpr uint32_t SEC_GMET = 0x54454D47u; // "GMET" — per-genome metadata
static constexpr uint32_t SEC_EMBD = 0x44424D45u; // "EMBD" — 256-dim embeddings
static constexpr uint32_t SEC_PRJ3 = 0x33524A50u; // "PRJ3" — 3D projection
static constexpr uint32_t SEC_EDGE = 0x45474445u; // "EDGE" — rep→member edges
static constexpr uint32_t SEC_TIDX = 0x58444954u; // "TIDX" — taxon directory
static constexpr uint32_t SEC_TREE = 0x45455254u; // "TREE" — taxonomy hierarchy
static constexpr uint32_t SEC_ACCX = 0x58434341u; // "ACCX" — accession index
static constexpr uint32_t SEC_STRT = 0x54525453u; // "STRT" — string table

// ── File structures ──────────────────────────────────────────────────────────

struct FileHeader {
    uint32_t magic;             // GRD1_MAGIC
    uint16_t version_major;     // 1
    uint16_t version_minor;     // 0
    uint64_t file_uuid_lo;
    uint64_t file_uuid_hi;
    uint64_t created_at_unix;
    uint64_t flags;
    uint8_t  reserved[88];
};
static_assert(sizeof(FileHeader) == 128);

struct SectionDesc {
    uint32_t type;              // SEC_* constant
    uint16_t version;
    uint16_t flags;
    uint64_t section_id;        // taxon_ordinal * 16 + offset for per-taxon sections
    uint64_t file_offset;
    uint64_t compressed_size;
    uint64_t uncompressed_size;
    uint64_t item_count;        // n_genomes for per-taxon, n_taxa for global
    uint64_t aux0;
    uint64_t aux1;
    uint8_t  checksum[16];
};
static_assert(sizeof(SectionDesc) == 80);

struct TocHeader {
    uint32_t magic;             // TOCB_MAGIC
    uint16_t version;
    uint16_t flags;
    uint64_t section_count;
    uint64_t n_taxa;
    uint64_t n_genomes_total;
    uint8_t  reserved[96];
};
static_assert(sizeof(TocHeader) == 128);

struct TailLocator {
    uint32_t magic;             // GRDT_MAGIC
    uint16_t version;
    uint16_t flags;
    uint64_t toc_offset;
    uint64_t toc_size;
    uint64_t n_taxa;
    uint8_t  reserved[32];
};
static_assert(sizeof(TailLocator) == 64);

// ── Per-genome status ────────────────────────────────────────────────────────

enum class GenomeStatus : uint8_t {
    REPRESENTATIVE = 0,
    MEMBER         = 1,
    CONTAMINATED   = 2,
    OUTLIER        = 3,  // flagged but retained
    FAILED         = 4,
};

// ── TMTA section header (inside compressed payload) ──────────────────────────

struct TaxonMetaHeader {
    uint32_t magic;              // SEC_TMTA
    uint32_t n_genomes;
    uint32_t n_reps;
    uint32_t n_contaminated;
    uint32_t embed_dim;          // 256
    uint32_t sketch_size;
    uint32_t kmer_size;
    uint32_t k_conn;
    float    diversity_threshold;
    float    ani_threshold;
    float    mst_p90_edge;
    float    mst_true_max;
    uint8_t  reserved[16];
};
static_assert(sizeof(TaxonMetaHeader) == 64);

// ── TIDX entry (global taxon directory) ──────────────────────────────────────

struct TaxonIndexEntry {
    uint64_t taxonomy_hash;      // FNV-1a-64 of taxonomy string
    uint64_t section_id_base;    // base section_id for this taxon's sections
    uint32_t strtable_offset;    // offset into STRT string table
    uint32_t n_genomes;
    uint32_t n_reps;
    uint32_t n_contaminated;
    float    centroid_3d[3];     // mean 3D position for meta-sphere view
    uint32_t reserved;
};
static_assert(sizeof(TaxonIndexEntry) == 48);

// ── EDGE entry ───────────────────────────────────────────────────────────────

struct EdgeEntry {
    uint32_t member_idx;
    uint32_t rep_idx;
    float    distance;           // geodesic distance
};
static_assert(sizeof(EdgeEntry) == 12);

// ── ACCX entry (cross-taxon accession lookup) ────────────────────────────────

struct AccessionIndexEntry {
    uint64_t accession_hash;     // FNV-1a-64 of accession string
    uint32_t taxon_ordinal;
    uint32_t genome_idx;         // index within that taxon's genome array
};
static_assert(sizeof(AccessionIndexEntry) == 16);

// ── TREE node (hierarchical taxonomy navigator) ──────────────────────────────

struct TreeNode {
    uint32_t strtable_offset;    // name of this rank level (e.g. "Proteobacteria")
    uint32_t parent_idx;         // index into tree node array (UINT32_MAX for root)
    uint32_t n_children;
    uint32_t first_child_idx;    // index of first child in sorted tree node array
    uint32_t n_genomes_subtree;  // total genomes under this node
    uint32_t n_species_subtree;  // total leaf species under this node
    uint8_t  rank;               // 0=domain, 1=phylum, ..., 9=subspecies
    uint8_t  reserved[3];
};
static_assert(sizeof(TreeNode) == 28);

// ── Hash helper ──────────────────────────────────────────────────────────────

inline uint64_t fnv1a_64(const char* data, size_t len) noexcept {
    uint64_t h = 14695981039346656037ULL;
    for (size_t i = 0; i < len; ++i) {
        h ^= static_cast<uint8_t>(data[i]);
        h *= 1099511628211ULL;
    }
    return h;
}

inline uint64_t fnv1a_64(const std::string& s) noexcept {
    return fnv1a_64(s.data(), s.size());
}

} // namespace grd
