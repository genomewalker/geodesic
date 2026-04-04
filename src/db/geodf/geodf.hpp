// geodf.hpp — Geodesic Data Format
//
// Single-file binary store for the geodesic dereplication pipeline.
// Designed for NFS reliability (pwrite-based, no mmap), crash safety (write-header
// after payload), and fast pipeline-state scanning (uncompressed headers).
//
// Inspired by MCAF (per-reference uncompressed headers) and genopack (TailLocator).
//
// File layout:
//   FileHeader         (32B)
//   [TaxonBlock]*      — TaxonHeader (64B, UNCOMPRESSED) + zstd-compressed payload
//   SketchBlock        — optional per-genome OPH sketch cache
//   StringTable        — deduped taxonomy strings, one zstd block
//   TaxonIndex         — sorted entries for O(log n) lookup
//   FileTrailer        (32B, magic at end for validation)
//
// Crash safety: TaxonHeader is written AFTER the compressed payload.
// On open, scan for complete (magic-valid) headers; truncate at first incomplete block.
//
// Threading: one writer, many concurrent readers (headers are read-only after write).
//
#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <filesystem>
#include <stdexcept>

namespace geodf {

// ── Constants ────────────────────────────────────────────────────────────────

static constexpr char FILE_MAGIC[8]    = {'G','E','O','D','F','M','T','1'};
static constexpr char TRAILER_MAGIC[8] = {'G','E','O','D','T','R','L','1'};
static constexpr char TAXON_MAGIC[4]   = {'G','T','A','X'};
static constexpr char SKETCH_MAGIC[4]  = {'G','S','K','T'};

static constexpr uint16_t FORMAT_MAJOR = 1;
static constexpr uint16_t FORMAT_MINOR = 1;

// ── Pipeline stages ───────────────────────────────────────────────────────────

enum class PipelineStage : uint8_t {
    NOT_STARTED    = 0,
    EMBEDDING_DONE = 1,
    COMPLETE       = 2,
    FAILED         = 3,
};

// ── FileHeader (32 bytes) ─────────────────────────────────────────────────────

struct FileHeader {
    char     magic[8];           //  0: FILE_MAGIC
    uint16_t version_major;      //  8: FORMAT_MAJOR
    uint16_t version_minor;      // 10: FORMAT_MINOR
    uint32_t n_taxa_expected;    // 12: total taxa this file will contain (0 = unknown)
    uint64_t gpk_snapshot_id;    // 16: FNV-1a-64 of last 32 bytes of source .gpk (8-aligned)
    uint32_t params_hash;        // 24: FNV-1a-32 of run parameters
    uint32_t flags;              // 28: bit 0: has_sketch_block
};
static_assert(sizeof(FileHeader) == 32);

// ── TaxonHeader (64 bytes, UNCOMPRESSED) ──────────────────────────────────────
//
// Written AFTER the compressed payload as a completion marker.
// Scanning for valid TAXON_MAGIC headers recovers all completed taxa after crash.
// Zone-map fields enable predicate pushdown without decompressing payload.

struct TaxonHeader {
    char           magic[4];              //  0: TAXON_MAGIC
    PipelineStage  stage;                 //  4: pipeline stage
    uint8_t        reserved0[3];          //  5: zero
    uint32_t       n_genomes;             //  8: genomes in this taxon
    uint32_t       n_reps;                // 12: representatives selected
    uint32_t       n_contaminated;        // 16: contamination candidates
    float          diversity_threshold;   // 20: geodesic distance threshold used
    float          ani_threshold;         // 24: effective ANI (100 - threshold*100)
    uint32_t       strtable_string_id;    // 28: ordinal index into StringTable (0-based)
    uint64_t       taxonomy_hash;         // 32: FNV-1a hash of taxonomy string (for index)
    uint64_t       payload_offset;        // 40: byte offset of compressed payload in file
    uint32_t       payload_size;          // 48: compressed payload size in bytes
    float          contamination_rate;    // 52: n_contaminated / n_genomes (zone map); contaminated = score > 0
    uint32_t       taxon_id;             // 56: monotonically increasing per-file ID
    uint8_t        reserved1[4];         // 60: zero
};
static_assert(sizeof(TaxonHeader) == 64);

// ── TaxonPayload (zstd-compressed columnar data) ──────────────────────────────
//
// FORMAT_MINOR=0 (legacy) on-disk layout:
//   taxonomy_len uint32 + taxonomy bytes
//   genome_ids[n_genomes]             : uint32[]
//   is_rep[ceil(n_genomes/8)]         : uint8[] bitpacked (bit=1 → representative)
//   contamination[n_genomes]          : float32[]
//   rep_accession_offsets[n_reps]     : uint32[] (offsets into accession string block)
//   accession_data                    : utf8 bytes (REP accessions only, null-terminated)
//   embed_dim                         : uint32 (stored before embeddings, 0 if absent)
//   embeddings[n_reps × embed_dim]    : float32[] (representative embeddings only)
//   error_message_len uint32 + bytes
//
// FORMAT_MINOR=1 on-disk layout:
//   taxonomy_len uint32 + taxonomy bytes
//   genome_ids[n_genomes]             : uint32[]
//   is_rep[ceil(n_genomes/8)]         : uint8[] bitpacked (bit=1 → representative)
//   contamination[n_genomes]          : float32[]
//   all_accession_offsets[n_genomes]  : uint32[] (offsets into accession string block for ALL genomes)
//   accession_data                    : utf8 bytes (ALL accessions, null-terminated)
//   rep_indices[n_reps]               : uint32[] (indices into genome_ids for reps)
//   embed_dim                         : uint32 (stored before embeddings, 0 if absent)
//   embeddings[n_reps × embed_dim]    : float32[] (representative embeddings only)
//   error_message_len uint32 + bytes
//
// This avoids storing all 4.7M genome embeddings (~4.7GB) while keeping
// representative embeddings (~2GB total) for incremental similarity queries.

// ── SketchBlockHeader (32 bytes) ──────────────────────────────────────────────
//
// Optional block storing per-genome OPH sketches for repeated experiments.
// Omit for single runs (recompute from genopack in ~39 min).

struct SketchBlockHeader {
    char     magic[4];        //  0: SKETCH_MAGIC
    uint8_t  reserved[4];     //  4: zero
    uint32_t n_genomes;       //  8: genomes in this block
    uint32_t sketch_size;     // 12: OPH sketch length (e.g. 10000)
    uint32_t kmer_size;       // 16: k (e.g. 21)
    uint32_t syncmer_s;       // 20: syncmer s parameter (0 = disabled)
    uint64_t data_offset;     // 24: byte offset of compressed sketch data
};
static_assert(sizeof(SketchBlockHeader) == 32);

// ── TaxonIndexEntry (24 bytes) ────────────────────────────────────────────────

struct TaxonIndexEntry {
    uint64_t       taxonomy_hash;   // FNV-1a hash for binary search
    uint64_t       header_offset;   // byte offset of TaxonHeader in file
    uint32_t       taxon_id;
    PipelineStage  stage;
    uint8_t        reserved[3];
};
static_assert(sizeof(TaxonIndexEntry) == 24);

// ── StringTableHeader (16 bytes) ──────────────────────────────────────────────

struct StringTableHeader {
    uint32_t n_strings;          // number of taxonomy strings
    uint32_t compressed_size;    // zstd compressed size
    uint32_t uncompressed_size;  // uncompressed size
    uint32_t reserved;
};
static_assert(sizeof(StringTableHeader) == 16);

// ── FileTrailer (32 bytes) ────────────────────────────────────────────────────

struct FileTrailer {
    uint64_t index_offset;        //  0: byte offset of TaxonIndex
    uint64_t strtable_offset;     //  8: byte offset of StringTable
    uint64_t sketch_block_offset; // 16: byte offset of SketchBlock (0 = absent)
    char     magic[8];            // 24: TRAILER_MAGIC
};
static_assert(sizeof(FileTrailer) == 32);

// ── Shared hash helpers ───────────────────────────────────────────────────────

inline uint64_t taxonomy_hash(const std::string& s) noexcept {
    uint64_t h = 14695981039346656037ULL;  // FNV-1a-64 offset basis
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ULL; }
    return h;
}

// FNV-1a-64 of the last 32 bytes of the .gpk file.
// Identifies the source GenomePack snapshot without hashing the whole file.
inline uint64_t gpk_snapshot_hash(const std::filesystem::path& gpk_path) {
    FILE* f = std::fopen(gpk_path.c_str(), "rb");
    if (!f)
        throw std::runtime_error("geodf: cannot open gpk for snapshot hash: " + gpk_path.string());
    if (std::fseek(f, -32, SEEK_END) != 0) {
        std::fclose(f);
        throw std::runtime_error("geodf: gpk file too small to hash: " + gpk_path.string());
    }
    uint8_t buf[32];
    size_t n = std::fread(buf, 1, 32, f);
    std::fclose(f);
    uint64_t h = 14695981039346656037ULL;
    for (size_t i = 0; i < n; ++i) { h ^= buf[i]; h *= 1099511628211ULL; }
    return h;
}

// FNV-1a-32 of the four key run parameters.
inline uint32_t hash_run_params(int kmer_size, int sketch_size, int syncmer_s, double ani_threshold) noexcept {
    uint32_t h = 2166136261u;  // FNV-1a-32 offset basis
    auto mix = [&](uint32_t v) {
        for (int i = 0; i < 4; ++i) {
            h ^= static_cast<uint8_t>(v & 0xFF);
            h *= 16777619u;
            v >>= 8;
        }
    };
    mix(static_cast<uint32_t>(kmer_size));
    mix(static_cast<uint32_t>(sketch_size));
    mix(static_cast<uint32_t>(syncmer_s));
    // encode ani_threshold as fixed-point (×1000, rounded)
    mix(static_cast<uint32_t>(static_cast<int>(ani_threshold * 1000.0 + 0.5)));
    return h;
}

} // namespace geodf
