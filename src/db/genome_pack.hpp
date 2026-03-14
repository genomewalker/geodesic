#pragma once
#include <duckdb.hpp>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace derep::db {

// GenomePack: taxonomy-indexed, zstd-compressed genome sequence store.
// Converts 5.2M scattered per-genome .fa.gz NFS files into ~130k per-taxon
// packed files on fast local storage, eliminating small-file NFS metadata overhead.
//
// File layout:
//   {pack_dir}/pack.db                      — DuckDB index
//   {pack_dir}/data/{aa}/{taxon_hash}.gpack — per-taxon packed file
//
// .gpack file format (entire file is one zstd-compressed stream):
//   [magic: 5 bytes = "GPACK"]
//   [version: uint8 = 1]
//   [n_genomes: uint32_t LE]
//   for each genome:
//     [accession_len: uint16_t LE]
//     [accession: accession_len bytes]
//     [byte_offset: uint64_t LE]  -- offset of genome FASTA in payload
//     [byte_length: uint64_t LE]  -- byte length of genome FASTA in payload
//   [FASTA payload: concatenated raw FASTA bytes for all genomes]
class GenomePack {
public:
    struct Config {
        int    zstd_level           = 6;    // 1-22; 6 = fast (~100 MB/s) with good cross-genome ratio
        size_t taxa_per_checkpoint  = 500;  // DuckDB checkpoint cadence
    };

    struct GenomeSlice {
        std::string accession;
        uint64_t    offset = 0;
        uint64_t    length = 0;
    };

    // Decompressed taxon: full decompressed buffer (header + FASTA) + per-genome slices.
    // payload_start marks the byte offset where FASTA payload begins inside payload.
    // Avoids copying the payload out of the decompressed buffer.
    struct TaxonData {
        std::vector<char>        payload;
        size_t                   payload_start = 0;
        std::vector<GenomeSlice> genomes;
        const char* data(size_t i) const { return payload.data() + payload_start + genomes[i].offset; }
        size_t      size(size_t i) const { return static_cast<size_t>(genomes[i].length); }
        bool        empty()        const { return genomes.empty(); }
    };

    GenomePack() : GenomePack(Config{}) {}
    explicit GenomePack(Config cfg);
    ~GenomePack();
    GenomePack(const GenomePack&) = delete;
    GenomePack& operator=(const GenomePack&) = delete;

    void open_read (const std::filesystem::path& pack_dir);
    void open_write(const std::filesystem::path& pack_dir);
    void close();
    bool is_open() const { return database_ != nullptr; }

    bool      has_taxon  (const std::string& taxonomy) const;
    TaxonData fetch_taxon(const std::string& taxonomy) const;

    // Write one taxon's genomes. genome_bufs: (accession, raw_fasta_bytes).
    // Thread-safe: serialized internally.
    void write_taxon(const std::string& taxonomy,
                     const std::vector<std::pair<std::string, std::vector<char>>>& genome_bufs);

    std::unordered_set<std::string> load_completed_taxa() const;
    void checkpoint();

private:
    std::filesystem::path pack_dir_;
    Config cfg_;
    std::unique_ptr<duckdb::DuckDB> database_;

    struct ThreadIdHash {
        std::size_t operator()(const std::thread::id& id) const noexcept {
            return std::hash<std::thread::id>{}(id);
        }
    };
    mutable std::mutex pool_mutex_;
    mutable std::unordered_map<std::thread::id,
                               std::unique_ptr<duckdb::Connection>,
                               ThreadIdHash> pool_;

    size_t packs_since_checkpoint_ = 0;
    std::mutex write_mutex_;   // serialise taxon writes

    duckdb::Connection& thread_connection() const;
    void create_schema();
    std::filesystem::path taxon_file_path(const std::string& taxonomy) const;
    static std::string taxon_hash(const std::string& taxonomy);
};

} // namespace derep::db
