#include "geodf_writer.hpp"
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <zstd.h>
#include <spdlog/spdlog.h>
#include <thread>
#include <chrono>

namespace geodf {

// ── FNV-1a hash for taxonomy strings ─────────────────────────────────────────

// taxonomy_hash is defined in geodf.hpp

// ── pwrite with ENOSPC/EIO retry ─────────────────────────────────────────────

static void safe_pwrite(int fd, const void* data, size_t len, uint64_t offset) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    off_t pos = static_cast<off_t>(offset);
    int retry_secs = 5;
    size_t written = 0;
    while (written < len) {
        ssize_t n = ::pwrite(fd, p + written, len - written, pos + static_cast<off_t>(written));
        if (n < 0) {
            if (errno == EINTR) continue;
            if ((errno == ENOSPC || errno == EIO) && retry_secs <= 300) {
                spdlog::warn("geodf: pwrite failed ({}), retrying in {}s", strerror(errno), retry_secs);
                std::this_thread::sleep_for(std::chrono::seconds(retry_secs));
                retry_secs = std::min(retry_secs * 2, 300);
                continue;
            }
            throw std::runtime_error(std::string("geodf: pwrite failed: ") + strerror(errno));
        }
        written += static_cast<size_t>(n);
    }
}

// ── zstd helpers ─────────────────────────────────────────────────────────────

static std::vector<uint8_t> compress(const void* data, size_t size, int level = 3) {
    size_t bound = ZSTD_compressBound(size);
    std::vector<uint8_t> out(bound);
    size_t csize = ZSTD_compress(out.data(), bound, data, size, level);
    if (ZSTD_isError(csize))
        throw std::runtime_error(std::string("geodf: zstd compress: ") + ZSTD_getErrorName(csize));
    out.resize(csize);
    return out;
}

// ── GeodfWriter ──────────────────────────────────────────────────────────────

GeodfWriter::GeodfWriter(const std::filesystem::path& path) {
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());

    fd_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd_ < 0)
        throw std::runtime_error("geodf: cannot open for writing: " + path.string());

    write_file_header();
}

GeodfWriter::~GeodfWriter() {
    if (!closed_ && fd_ >= 0) {
        try { close(); } catch (...) {}
    }
    if (fd_ >= 0) ::close(fd_);
}

void GeodfWriter::write_file_header() {
    FileHeader hdr{};
    std::memcpy(hdr.magic, FILE_MAGIC, 8);
    hdr.version_major    = FORMAT_MAJOR;
    hdr.version_minor    = FORMAT_MINOR;
    hdr.gpk_snapshot_id  = gpk_snapshot_id_;
    hdr.params_hash      = params_hash_;
    append_bytes(&hdr, sizeof(hdr));
}

void GeodfWriter::set_provenance(uint64_t gpk_snapshot_id, uint32_t params_hash) {
    gpk_snapshot_id_ = gpk_snapshot_id;
    params_hash_     = params_hash;
    // Rewrite the FileHeader in-place with the provenance fields now populated.
    FileHeader hdr{};
    std::memcpy(hdr.magic, FILE_MAGIC, 8);
    hdr.version_major   = FORMAT_MAJOR;
    hdr.version_minor   = FORMAT_MINOR;
    hdr.gpk_snapshot_id = gpk_snapshot_id_;
    hdr.params_hash     = params_hash_;
    write_bytes(&hdr, sizeof(hdr), 0);
}

uint32_t GeodfWriter::intern_string(const std::string& s) {
    auto it = string_index_.find(s);
    if (it != string_index_.end()) return it->second;
    uint32_t idx = static_cast<uint32_t>(strings_.size());
    strings_.push_back(s);
    string_index_[s] = idx;
    return idx;
}

void GeodfWriter::write_taxon(const TaxonResult& r) {
    uint32_t strtable_off = intern_string(r.taxonomy);

    // ── Build uncompressed columnar payload ───────────────────────────────────
    // Taxonomy string is embedded first in every payload for crash recovery:
    // the StringTable is only written on close(), but each payload is self-contained.
    std::vector<uint8_t> raw;
    {
        uint32_t tax_len = static_cast<uint32_t>(r.taxonomy.size());
        raw.insert(raw.end(),
            reinterpret_cast<const uint8_t*>(&tax_len),
            reinterpret_cast<const uint8_t*>(&tax_len) + sizeof(tax_len));
        raw.insert(raw.end(), r.taxonomy.begin(), r.taxonomy.end());
    }

    // genome_ids
    const uint32_t n = static_cast<uint32_t>(r.genome_ids.size());
    raw.insert(raw.end(),
        reinterpret_cast<const uint8_t*>(r.genome_ids.data()),
        reinterpret_cast<const uint8_t*>(r.genome_ids.data() + n));

    // is_rep bitpacked
    uint32_t nbytes_rep = (n + 7) / 8;
    std::vector<uint8_t> rep_bits(nbytes_rep, 0);
    for (uint32_t i = 0; i < n; ++i)
        if (r.is_rep[i]) rep_bits[i / 8] |= (1u << (i % 8));
    raw.insert(raw.end(), rep_bits.begin(), rep_bits.end());

    // contamination scores
    raw.insert(raw.end(),
        reinterpret_cast<const uint8_t*>(r.contamination.data()),
        reinterpret_cast<const uint8_t*>(r.contamination.data() + n));

    // all genome accessions (FORMAT_MINOR=1: all genomes, not just reps)
    const uint32_t n_reps = static_cast<uint32_t>(r.reps.size());
    std::vector<uint32_t> acc_offsets(n);
    std::string acc_data;
    for (uint32_t i = 0; i < n; ++i) {
        acc_offsets[i] = static_cast<uint32_t>(acc_data.size());
        if (i < r.all_accessions.size()) {
            acc_data += r.all_accessions[i];
        }
        acc_data.push_back('\0');
    }
    raw.insert(raw.end(),
        reinterpret_cast<const uint8_t*>(acc_offsets.data()),
        reinterpret_cast<const uint8_t*>(acc_offsets.data() + n));
    raw.insert(raw.end(), acc_data.begin(), acc_data.end());

    // rep_indices: indices into genome_ids for each rep (FORMAT_MINOR=1)
    // Build a genome_id → index map for fast lookup
    std::unordered_map<uint32_t, uint32_t> gid_to_idx;
    gid_to_idx.reserve(n);
    for (uint32_t i = 0; i < n; ++i)
        gid_to_idx[r.genome_ids[i]] = i;
    std::vector<uint32_t> rep_indices(n_reps);
    for (uint32_t i = 0; i < n_reps; ++i) {
        auto it = gid_to_idx.find(r.reps[i].genome_id);
        rep_indices[i] = (it != gid_to_idx.end()) ? it->second : i;
    }
    raw.insert(raw.end(),
        reinterpret_cast<const uint8_t*>(rep_indices.data()),
        reinterpret_cast<const uint8_t*>(rep_indices.data() + n_reps));

    // rep embeddings (with dim prefix)
    uint32_t embed_dim = 0;
    for (const auto& rep : r.reps)
        if (!rep.embedding.empty()) { embed_dim = static_cast<uint32_t>(rep.embedding.size()); break; }
    raw.insert(raw.end(),
        reinterpret_cast<const uint8_t*>(&embed_dim),
        reinterpret_cast<const uint8_t*>(&embed_dim) + sizeof(embed_dim));
    for (const auto& rep : r.reps) {
        if (embed_dim > 0 && rep.embedding.size() == embed_dim) {
            raw.insert(raw.end(),
                reinterpret_cast<const uint8_t*>(rep.embedding.data()),
                reinterpret_cast<const uint8_t*>(rep.embedding.data() + embed_dim));
        } else if (embed_dim > 0) {
            // pad with zeros for missing embeddings
            std::vector<float> zeros(embed_dim, 0.0f);
            raw.insert(raw.end(),
                reinterpret_cast<const uint8_t*>(zeros.data()),
                reinterpret_cast<const uint8_t*>(zeros.data() + embed_dim));
        }
    }

    // error_message for FAILED taxa (null-terminated, empty string for others)
    {
        const std::string& msg = r.error_message;
        uint32_t msg_len = static_cast<uint32_t>(msg.size());
        raw.insert(raw.end(),
            reinterpret_cast<const uint8_t*>(&msg_len),
            reinterpret_cast<const uint8_t*>(&msg_len) + sizeof(msg_len));
        raw.insert(raw.end(), msg.begin(), msg.end());
    }

    // ── Compress payload ──────────────────────────────────────────────────────
    auto compressed = compress(raw.data(), raw.size());
    uint64_t payload_offset = offset_;
    append_bytes(compressed.data(), compressed.size());

    // ── Build and write TaxonHeader (AFTER payload — completion marker) ───────
    TaxonHeader hdr{};
    std::memcpy(hdr.magic, TAXON_MAGIC, 4);
    hdr.stage                = r.stage;
    hdr.n_genomes            = n;
    hdr.n_reps               = n_reps;
    // contaminated = score > 0.0f (positive score indicates contamination)
    hdr.n_contaminated       = static_cast<uint32_t>(
        std::count_if(r.contamination.begin(), r.contamination.end(),
                      [](float v){ return v > 0.0f; }));
    hdr.diversity_threshold  = r.diversity_threshold;
    hdr.ani_threshold        = r.ani_threshold;
    hdr.strtable_string_id   = strtable_off;
    hdr.taxonomy_hash        = taxonomy_hash(r.taxonomy);
    hdr.payload_offset       = payload_offset;
    hdr.payload_size         = static_cast<uint32_t>(compressed.size());
    hdr.contamination_rate   = n > 0
        ? static_cast<float>(hdr.n_contaminated) / n
        : 0.0f;
    hdr.taxon_id             = next_id_++;

    uint64_t header_offset = offset_;
    append_bytes(&hdr, sizeof(hdr));

    // ── Record in index ───────────────────────────────────────────────────────
    TaxonIndexEntry entry{};
    entry.taxonomy_hash  = hdr.taxonomy_hash;
    entry.header_offset  = header_offset;
    entry.taxon_id       = hdr.taxon_id;
    entry.stage          = hdr.stage;
    index_.push_back(entry);
}

void GeodfWriter::close() {
    if (closed_) return;
    closed_ = true;

    // ── Write StringTable ─────────────────────────────────────────────────────
    uint64_t strtable_offset = offset_;
    {
        // Serialize: n_strings, then each string null-terminated
        std::string raw_strings;
        uint32_t n = static_cast<uint32_t>(strings_.size());
        raw_strings.append(reinterpret_cast<const char*>(&n), sizeof(n));
        for (const auto& s : strings_) {
            raw_strings += s;
            raw_strings.push_back('\0');
        }
        auto compressed = compress(raw_strings.data(), raw_strings.size());
        StringTableHeader sth{};
        sth.n_strings        = n;
        sth.compressed_size  = static_cast<uint32_t>(compressed.size());
        sth.uncompressed_size = static_cast<uint32_t>(raw_strings.size());
        append_bytes(&sth, sizeof(sth));
        append_bytes(compressed.data(), compressed.size());
    }

    // ── Write TaxonIndex (sorted by taxonomy_hash) ────────────────────────────
    uint64_t index_offset = offset_;
    {
        std::sort(index_.begin(), index_.end(),
            [](const TaxonIndexEntry& a, const TaxonIndexEntry& b) {
                return a.taxonomy_hash < b.taxonomy_hash;
            });
        uint32_t n = static_cast<uint32_t>(index_.size());
        append_bytes(&n, sizeof(n));
        append_bytes(index_.data(), index_.size() * sizeof(TaxonIndexEntry));
    }

    // ── Write FileTrailer ─────────────────────────────────────────────────────
    FileTrailer trailer{};
    trailer.index_offset        = index_offset;
    trailer.strtable_offset     = strtable_offset;
    trailer.sketch_block_offset = 0;
    std::memcpy(trailer.magic, TRAILER_MAGIC, 8);
    append_bytes(&trailer, sizeof(trailer));

    // Flush
    ::fsync(fd_);
    spdlog::info("geodf: wrote {} taxa, {} strings, {} bytes total",
                 index_.size(), strings_.size(), offset_);
}

void GeodfWriter::append_bytes(const void* data, size_t len) {
    safe_pwrite(fd_, data, len, offset_);
    offset_ += len;
}

void GeodfWriter::write_bytes(const void* data, size_t len, uint64_t offset) {
    safe_pwrite(fd_, data, len, offset);
}

} // namespace geodf
