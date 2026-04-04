#include "geodf_reader.hpp"
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <zstd.h>
#include <spdlog/spdlog.h>

namespace geodf {

// ── pread helper ─────────────────────────────────────────────────────────────

static void safe_pread(int fd, void* buf, size_t len, uint64_t offset) {
    uint8_t* p = static_cast<uint8_t*>(buf);
    off_t pos = static_cast<off_t>(offset);
    size_t total = 0;
    while (total < len) {
        ssize_t n = ::pread(fd, p + total, len - total, pos + static_cast<off_t>(total));
        if (n <= 0)
            throw std::runtime_error("geodf: pread failed at offset " + std::to_string(offset));
        total += static_cast<size_t>(n);
    }
}

static uint64_t file_size(int fd) {
    off_t sz = ::lseek(fd, 0, SEEK_END);
    if (sz < 0) throw std::runtime_error("geodf: lseek failed");
    return static_cast<uint64_t>(sz);
}

// ── zstd decompress ───────────────────────────────────────────────────────────

static std::vector<uint8_t> decompress(const uint8_t* data, size_t csize, size_t usize) {
    std::vector<uint8_t> out(usize);
    size_t r = ZSTD_decompress(out.data(), usize, data, csize);
    if (ZSTD_isError(r))
        throw std::runtime_error(std::string("geodf: zstd decompress: ") + ZSTD_getErrorName(r));
    out.resize(r);
    return out;
}

// ── GeodfReader ──────────────────────────────────────────────────────────────

GeodfReader::GeodfReader(const std::filesystem::path& path) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0)
        throw std::runtime_error("geodf: cannot open: " + path.string());

    uint64_t sz = file_size(fd_);
    if (sz < sizeof(FileHeader) + sizeof(FileTrailer))
        throw std::runtime_error("geodf: file too small");

    // Validate FileHeader
    safe_pread(fd_, &file_header_, sizeof(file_header_), 0);
    if (std::memcmp(file_header_.magic, FILE_MAGIC, 8) != 0)
        throw std::runtime_error("geodf: bad file magic");

    // Validate FileTrailer
    safe_pread(fd_, &trailer_, sizeof(trailer_), sz - sizeof(FileTrailer));
    if (std::memcmp(trailer_.magic, TRAILER_MAGIC, 8) != 0) {
        // Trailer invalid — file may be incomplete. Fall back to header scan.
        spdlog::warn("geodf: trailer invalid — scanning headers for recovery");
        trailer_ = {};
        scan_and_recover(sz);
        return;
    }

    // Load index from trailer
    load_index();
    spdlog::info("geodf: opened with {} taxa", index_.size());
}

GeodfReader::~GeodfReader() {
    if (fd_ >= 0) ::close(fd_);
}

void GeodfReader::scan_and_recover(uint64_t sz) {
    // Scan file sequentially for valid TaxonHeaders to recover after crash.
    // A TaxonHeader is valid if its magic matches and payload_offset+payload_size
    // fits within the file.
    uint64_t pos = sizeof(FileHeader);
    while (pos + sizeof(TaxonHeader) <= sz) {
        TaxonHeader hdr{};
        try { safe_pread(fd_, &hdr, sizeof(hdr), pos); }
        catch (...) { break; }

        if (std::memcmp(hdr.magic, TAXON_MAGIC, 4) == 0 &&
            hdr.payload_offset + hdr.payload_size <= sz) {
            TaxonIndexEntry e{};
            e.taxonomy_hash  = hdr.taxonomy_hash;
            e.header_offset  = pos;
            e.taxon_id       = hdr.taxon_id;
            e.stage          = hdr.stage;
            index_.push_back(e);

            // Extract taxonomy string from payload prefix (embedded for crash recovery)
            try {
                std::vector<uint8_t> cbuf(hdr.payload_size);
                safe_pread(fd_, cbuf.data(), hdr.payload_size, hdr.payload_offset);
                uint64_t usize = ZSTD_getFrameContentSize(cbuf.data(), cbuf.size());
                if (usize != ZSTD_CONTENTSIZE_ERROR && usize != ZSTD_CONTENTSIZE_UNKNOWN) {
                    auto raw = decompress(cbuf.data(), cbuf.size(), static_cast<size_t>(usize));
                    if (raw.size() >= 4) {
                        uint32_t tax_len; std::memcpy(&tax_len, raw.data(), 4);
                        if (4 + tax_len <= raw.size()) {
                            std::string tax(reinterpret_cast<const char*>(raw.data() + 4), tax_len);
                            // Key by payload_offset (matches resolve_taxonomy lookup key)
                            recovered_strings_[hdr.payload_offset] = std::move(tax);
                        }
                    }
                }
            } catch (...) {}

            pos = hdr.payload_offset + hdr.payload_size + sizeof(TaxonHeader);
        } else {
            pos++;
        }
    }
    // Sort index by taxonomy_hash for binary search in find()
    std::sort(index_.begin(), index_.end(),
              [](const TaxonIndexEntry& a, const TaxonIndexEntry& b) {
                  return a.taxonomy_hash < b.taxonomy_hash; });
    spdlog::info("geodf: recovered {} taxa from header scan", index_.size());
}

void GeodfReader::load_index() {
    if (trailer_.index_offset == 0) return;
    uint32_t n = 0;
    safe_pread(fd_, &n, sizeof(n), trailer_.index_offset);
    index_.resize(n);
    if (n > 0)
        safe_pread(fd_, index_.data(), n * sizeof(TaxonIndexEntry),
                   trailer_.index_offset + sizeof(n));
}

void GeodfReader::load_strings() const {
    if (strings_loaded_) return;
    strings_loaded_ = true;
    if (trailer_.strtable_offset == 0) return;

    StringTableHeader sth{};
    safe_pread(fd_, &sth, sizeof(sth), trailer_.strtable_offset);

    std::vector<uint8_t> cbuf(sth.compressed_size);
    safe_pread(fd_, cbuf.data(), sth.compressed_size,
               trailer_.strtable_offset + sizeof(sth));
    auto raw = decompress(cbuf.data(), sth.compressed_size, sth.uncompressed_size);

    // Parse: u32 n, then n null-terminated strings
    const uint8_t* p = raw.data();
    uint32_t n = 0;
    std::memcpy(&n, p, sizeof(n)); p += sizeof(n);
    strings_.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        strings_[i] = std::string(reinterpret_cast<const char*>(p));
        p += strings_[i].size() + 1;
    }
}

std::string GeodfReader::lookup_string(uint32_t id) const {
    load_strings();
    if (id < strings_.size()) return strings_[id];
    return {};
}

// Resolve taxonomy for a header: use string table if available, else recovered map.
std::string GeodfReader::lookup_recovered(size_t idx) const {
    (void)idx;
    return {};
}

// Resolve taxonomy: string table (normal path) or recovered map (crash recovery).
std::string GeodfReader::resolve_taxonomy(const TaxonHeader& hdr) const {
    // Try string table first (populated after a clean close())
    auto s = lookup_string(hdr.strtable_string_id);
    if (!s.empty()) return s;
    // Fall back to recovered strings (populated during scan_and_recover())
    auto it = recovered_strings_.find(hdr.payload_offset);
    if (it != recovered_strings_.end()) return it->second;
    return {};
}

std::unordered_set<std::string> GeodfReader::completed_taxa() const {
    load_strings();
    std::unordered_set<std::string> result;
    for (const auto& e : index_) {
        if (e.stage != PipelineStage::COMPLETE) continue;
        TaxonHeader hdr{};
        safe_pread(fd_, &hdr, sizeof(hdr), e.header_offset);
        if (std::memcmp(hdr.magic, TAXON_MAGIC, 4) == 0) {
            auto tax = resolve_taxonomy(hdr);
            if (!tax.empty()) result.insert(std::move(tax));
        }
    }
    return result;
}

std::vector<std::pair<TaxonHeader, std::string>> GeodfReader::scan_headers() const {
    load_strings();
    std::vector<std::pair<TaxonHeader, std::string>> result;
    result.reserve(index_.size());
    for (const auto& e : index_) {
        TaxonHeader hdr{};
        safe_pread(fd_, &hdr, sizeof(hdr), e.header_offset);
        if (std::memcmp(hdr.magic, TAXON_MAGIC, 4) == 0)
            result.emplace_back(hdr, resolve_taxonomy(hdr));
    }
    return result;
}

std::optional<TaxonData> GeodfReader::find(const std::string& taxonomy) const {
    load_strings();
    // Binary search index by taxonomy_hash (same FNV-1a as writer)
    uint64_t h = taxonomy_hash(taxonomy);
    auto it = std::lower_bound(index_.begin(), index_.end(), h,
        [](const TaxonIndexEntry& e, uint64_t hash) { return e.taxonomy_hash < hash; });
    while (it != index_.end() && it->taxonomy_hash == h) {
        TaxonHeader hdr{};
        safe_pread(fd_, &hdr, sizeof(hdr), it->header_offset);
        if (resolve_taxonomy(hdr) == taxonomy)
            return decompress_taxon(hdr);
        ++it;
    }
    return std::nullopt;
}

void GeodfReader::for_each_complete(const std::function<void(const TaxonData&)>& cb) const {
    load_strings();
    for (const auto& e : index_) {
        if (e.stage != PipelineStage::COMPLETE) continue;
        TaxonHeader hdr{};
        safe_pread(fd_, &hdr, sizeof(hdr), e.header_offset);
        if (std::memcmp(hdr.magic, TAXON_MAGIC, 4) != 0) continue;
        cb(decompress_taxon(hdr));
    }
}

TaxonData GeodfReader::decompress_taxon(const TaxonHeader& hdr) const {
    // Read compressed payload
    std::vector<uint8_t> cbuf(hdr.payload_size);
    safe_pread(fd_, cbuf.data(), hdr.payload_size, hdr.payload_offset);

    // Decompress — use streaming to get uncompressed size
    uint64_t usize = ZSTD_getFrameContentSize(cbuf.data(), cbuf.size());
    if (usize == ZSTD_CONTENTSIZE_ERROR || usize == ZSTD_CONTENTSIZE_UNKNOWN)
        throw std::runtime_error("geodf: cannot determine decompressed size");
    auto raw = decompress(cbuf.data(), cbuf.size(), static_cast<size_t>(usize));

    const uint8_t* p = raw.data();
    const uint8_t* end = raw.data() + raw.size();
    auto read_u32 = [&]() -> uint32_t {
        if (p + 4 > end) throw std::runtime_error("geodf: payload truncated");
        uint32_t v; std::memcpy(&v, p, 4); p += 4; return v;
    };
    auto read_f32 = [&]() -> float {
        if (p + 4 > end) throw std::runtime_error("geodf: payload truncated");
        float v; std::memcpy(&v, p, 4); p += 4; return v;
    };

    TaxonData d;
    d.stage               = hdr.stage;
    d.diversity_threshold = hdr.diversity_threshold;
    d.ani_threshold       = hdr.ani_threshold;
    d.taxon_id            = hdr.taxon_id;

    const uint32_t n = hdr.n_genomes;
    const uint32_t n_reps = hdr.n_reps;

    // taxonomy string (first field in payload, for crash recovery)
    {
        uint32_t tax_len; std::memcpy(&tax_len, p, 4); p += 4;
        if (p + tax_len > end) throw std::runtime_error("geodf: truncated taxonomy");
        d.taxonomy = std::string(reinterpret_cast<const char*>(p), tax_len);
        p += tax_len;
    }

    // genome_ids
    d.genome_ids.resize(n);
    if (p + n * 4 > end) throw std::runtime_error("geodf: truncated genome_ids");
    std::memcpy(d.genome_ids.data(), p, n * 4); p += n * 4;

    // is_rep bitpacked
    uint32_t nbytes = (n + 7) / 8;
    d.is_rep.resize(n, false);
    if (p + nbytes > end) throw std::runtime_error("geodf: truncated is_rep");
    for (uint32_t i = 0; i < n; ++i)
        d.is_rep[i] = (p[i / 8] >> (i % 8)) & 1;
    p += nbytes;

    // contamination
    d.contamination.resize(n);
    if (p + n * 4 > end) throw std::runtime_error("geodf: truncated contamination");
    std::memcpy(d.contamination.data(), p, n * 4); p += n * 4;

    if (file_header_.version_minor >= 1) {
        // FORMAT_MINOR=1: all_accession_offsets[n_genomes] + accession_data + rep_indices[n_reps]
        std::vector<uint32_t> acc_offsets(n);
        if (p + n * 4 > end) throw std::runtime_error("geodf: truncated all_accession_offsets");
        std::memcpy(acc_offsets.data(), p, n * 4); p += n * 4;
        // accession data: walk all null-terminated strings
        const char* acc_base = reinterpret_cast<const char*>(p);
        d.all_accessions.resize(n);
        for (uint32_t i = 0; i < n; ++i) {
            d.all_accessions[i] = acc_base + acc_offsets[i];
        }
        // Advance p past accession data
        if (n > 0) {
            const char* last_str = acc_base + acc_offsets[n - 1];
            p = reinterpret_cast<const uint8_t*>(last_str + strlen(last_str) + 1);
        }
        // rep_indices[n_reps]
        std::vector<uint32_t> rep_indices(n_reps);
        if (p + n_reps * 4 > end) throw std::runtime_error("geodf: truncated rep_indices");
        std::memcpy(rep_indices.data(), p, n_reps * 4); p += n_reps * 4;
        // Populate rep_accessions from all_accessions via rep_indices
        d.rep_accessions.resize(n_reps);
        for (uint32_t i = 0; i < n_reps; ++i) {
            uint32_t idx = rep_indices[i];
            d.rep_accessions[i] = (idx < n) ? d.all_accessions[idx] : std::string{};
        }
    } else {
        // FORMAT_MINOR=0 (legacy): rep_accession_offsets[n_reps] + accession_data
        std::vector<uint32_t> acc_offsets(n_reps);
        if (p + n_reps * 4 > end) throw std::runtime_error("geodf: truncated acc_offsets");
        std::memcpy(acc_offsets.data(), p, n_reps * 4); p += n_reps * 4;
        const char* acc_base = reinterpret_cast<const char*>(p);
        d.rep_accessions.resize(n_reps);
        for (uint32_t i = 0; i < n_reps; ++i) {
            d.rep_accessions[i] = acc_base + acc_offsets[i];
        }
        if (n_reps > 0) {
            const char* last_str = acc_base + acc_offsets[n_reps - 1];
            p = reinterpret_cast<const uint8_t*>(last_str + strlen(last_str) + 1);
        }
        // all_accessions not available in v0 — leave empty
    }

    // embed_dim + embeddings
    if (p + 4 > end) throw std::runtime_error("geodf: truncated embed_dim");
    uint32_t embed_dim; std::memcpy(&embed_dim, p, 4); p += 4;
    d.rep_embeddings.resize(n_reps);
    for (uint32_t i = 0; i < n_reps; ++i) {
        if (embed_dim > 0) {
            if (p + embed_dim * 4 > end) throw std::runtime_error("geodf: truncated embeddings");
            d.rep_embeddings[i].resize(embed_dim);
            std::memcpy(d.rep_embeddings[i].data(), p, embed_dim * 4);
            p += embed_dim * 4;
        }
    }

    // error_message (u32 len + bytes)
    if (p + 4 <= end) {
        uint32_t msg_len; std::memcpy(&msg_len, p, 4); p += 4;
        if (msg_len > 0 && p + msg_len <= end) {
            d.error_message = std::string(reinterpret_cast<const char*>(p), msg_len);
            p += msg_len;
        }
    }

    return d;
}

} // namespace geodf
