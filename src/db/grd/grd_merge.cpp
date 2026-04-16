// grd_merge.cpp — GRD shard merge implementation
#include "grd_merge.hpp"
#include "grd.hpp"
#include "grd_writer.hpp"

#include <algorithm>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <unistd.h>
#include <unordered_map>
#include <vector>
#include <zstd.h>

namespace grd {

namespace {

// Read entire file into memory.
std::vector<uint8_t> read_file(const std::filesystem::path& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0)
        throw std::runtime_error("grd_merge: cannot open " + path.string());

    auto size = std::filesystem::file_size(path);
    std::vector<uint8_t> buf(size);
    size_t total = 0;
    while (total < size) {
        ssize_t n = ::read(fd, buf.data() + total, size - total);
        if (n <= 0) {
            ::close(fd);
            throw std::runtime_error("grd_merge: read error on " + path.string());
        }
        total += static_cast<size_t>(n);
    }
    ::close(fd);
    return buf;
}

std::vector<uint8_t> zstd_decompress(const uint8_t* data, size_t csize, size_t usize) {
    std::vector<uint8_t> out(usize);
    size_t r = ZSTD_decompress(out.data(), usize, data, csize);
    if (ZSTD_isError(r))
        throw std::runtime_error(std::string("grd_merge: zstd: ") + ZSTD_getErrorName(r));
    out.resize(r);
    return out;
}

// Parse a GRD file's TOC and return all section descriptors.
struct ShardInfo {
    std::vector<SectionDesc> sections;
    uint64_t n_taxa = 0;
    uint64_t n_genomes = 0;
};

ShardInfo parse_shard_toc(const std::vector<uint8_t>& buf) {
    if (buf.size() < 64 + 128)
        throw std::runtime_error("grd_merge: file too small");

    ShardInfo info;

    // Read TailLocator at EOF-64
    size_t tail_off = buf.size() - 64;
    TailLocator tail;
    std::memcpy(&tail, buf.data() + tail_off, sizeof(tail));
    if (tail.magic != GRDT_MAGIC)
        throw std::runtime_error("grd_merge: bad tail magic");

    // Decompress TOC
    auto toc_raw = zstd_decompress(
        buf.data() + tail.toc_offset,
        tail.toc_size,
        tail.toc_size * 10);  // estimate; ZSTD handles actual size

    // Parse TOC header
    if (toc_raw.size() < sizeof(TocHeader))
        throw std::runtime_error("grd_merge: truncated TOC");

    TocHeader toc_hdr;
    std::memcpy(&toc_hdr, toc_raw.data(), sizeof(toc_hdr));
    if (toc_hdr.magic != TOCB_MAGIC)
        throw std::runtime_error("grd_merge: bad TOC magic");

    info.n_taxa = toc_hdr.n_taxa;
    info.n_genomes = toc_hdr.n_genomes_total;

    // Parse section descriptors
    for (uint64_t i = 0; i < toc_hdr.section_count; ++i) {
        size_t off = sizeof(TocHeader) + i * sizeof(SectionDesc);
        if (off + sizeof(SectionDesc) > toc_raw.size()) break;
        SectionDesc desc;
        std::memcpy(&desc, toc_raw.data() + off, sizeof(desc));
        info.sections.push_back(desc);
    }

    return info;
}

// Per-taxon section types (written per taxon, need renumbering)
bool is_per_taxon_section(uint32_t type) {
    return type == SEC_TMTA || type == SEC_GMET || type == SEC_EMBD ||
           type == SEC_PRJ3 || type == SEC_EDGE;
}

} // namespace

MergeStats merge_grd(const std::vector<std::filesystem::path>& shards,
                     const std::filesystem::path& output) {
    if (shards.empty())
        throw std::runtime_error("grd_merge: no shards to merge");

    // Phase 1: Read all shards, parse TOCs, collect per-taxon sections.
    // We extract each per-taxon section's compressed data and re-emit it
    // into the merged file with renumbered section_ids.

    struct TaxonSections {
        std::string taxonomy;  // extracted from TMTA
        std::vector<std::pair<SectionDesc, std::vector<uint8_t>>> sections;  // desc + compressed data
        uint32_t n_genomes = 0;
        uint32_t n_reps = 0;
        uint32_t n_contam = 0;
    };

    std::vector<TaxonSections> all_taxa;

    for (const auto& shard_path : shards) {
        spdlog::debug("grd_merge: reading {}", shard_path.string());
        auto buf = read_file(shard_path);
        auto info = parse_shard_toc(buf);

        // Group sections by taxon (section_id_base = section_id & ~0xF)
        std::map<uint64_t, std::vector<size_t>> taxon_groups;
        for (size_t i = 0; i < info.sections.size(); ++i) {
            const auto& sec = info.sections[i];
            if (is_per_taxon_section(sec.type)) {
                uint64_t base = (sec.section_id / 16) * 16;
                taxon_groups[base].push_back(i);
            }
        }

        for (auto& [base, indices] : taxon_groups) {
            TaxonSections ts;

            for (size_t idx : indices) {
                const auto& sec = info.sections[idx];

                // Copy compressed data directly (no decompress/recompress)
                std::vector<uint8_t> cdata(
                    buf.data() + sec.file_offset,
                    buf.data() + sec.file_offset + sec.compressed_size);

                // Extract taxonomy string from TMTA section
                if (sec.type == SEC_TMTA) {
                    auto raw = zstd_decompress(cdata.data(), cdata.size(),
                                               sec.uncompressed_size);
                    if (raw.size() >= sizeof(TaxonMetaHeader) + 4) {
                        TaxonMetaHeader hdr;
                        std::memcpy(&hdr, raw.data(), sizeof(hdr));
                        ts.n_genomes = hdr.n_genomes;
                        ts.n_reps = hdr.n_reps;
                        ts.n_contam = hdr.n_contaminated;

                        uint32_t tax_len;
                        std::memcpy(&tax_len, raw.data() + sizeof(TaxonMetaHeader), 4);
                        if (sizeof(TaxonMetaHeader) + 4 + tax_len <= raw.size())
                            ts.taxonomy.assign(
                                reinterpret_cast<const char*>(raw.data() + sizeof(TaxonMetaHeader) + 4),
                                tax_len);
                    }
                }

                ts.sections.emplace_back(sec, std::move(cdata));
            }

            if (!ts.taxonomy.empty())
                all_taxa.push_back(std::move(ts));
        }
    }

    spdlog::info("grd_merge: {} taxa from {} shards", all_taxa.size(), shards.size());

    // Phase 2: Write merged GRD file.
    // Open output, write FileHeader, then all per-taxon sections with
    // renumbered section_ids, then global sections.

    int fd = ::open(output.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0)
        throw std::runtime_error("grd_merge: cannot open output: " + output.string());

    auto safe_write = [&](const void* data, size_t len, uint64_t offset) {
        const uint8_t* p = static_cast<const uint8_t*>(data);
        size_t written = 0;
        while (written < len) {
            ssize_t n = ::pwrite(fd, p + written, len - written,
                                 static_cast<off_t>(offset + written));
            if (n < 0) {
                if (errno == EINTR) continue;
                ::close(fd);
                throw std::runtime_error("grd_merge: write error");
            }
            written += static_cast<size_t>(n);
        }
    };

    uint64_t file_offset = 0;

    // Write FileHeader
    FileHeader fhdr{};
    fhdr.magic = GRD1_MAGIC;
    fhdr.version_major = 1;
    fhdr.version_minor = 0;
    fhdr.created_at_unix = static_cast<uint64_t>(std::time(nullptr));
    safe_write(&fhdr, sizeof(fhdr), file_offset);
    file_offset += sizeof(fhdr);

    // Accumulate section descriptors for final TOC
    std::vector<SectionDesc> all_sections;
    std::vector<TaxonIndexEntry> tidx_entries;
    std::vector<AccessionIndexEntry> accx_entries;
    std::vector<std::string> strtable;
    std::unordered_map<std::string, uint32_t> str_index;
    uint64_t total_genomes = 0;

    auto intern = [&](const std::string& s) -> uint32_t {
        auto it = str_index.find(s);
        if (it != str_index.end()) return it->second;
        uint32_t idx = static_cast<uint32_t>(strtable.size());
        strtable.push_back(s);
        str_index[s] = idx;
        return idx;
    };

    auto write_compressed_section = [&](uint32_t type, uint64_t section_id,
                                        const std::vector<uint8_t>& cdata,
                                        size_t usize, uint64_t item_count,
                                        uint64_t aux0 = 0, uint64_t aux1 = 0) {
        uint64_t sec_off = file_offset;
        safe_write(cdata.data(), cdata.size(), file_offset);
        file_offset += cdata.size();

        SectionDesc desc{};
        desc.type = type;
        desc.version = 1;
        desc.section_id = section_id;
        desc.file_offset = sec_off;
        desc.compressed_size = cdata.size();
        desc.uncompressed_size = usize;
        desc.item_count = item_count;
        desc.aux0 = aux0;
        desc.aux1 = aux1;
        all_sections.push_back(desc);
    };

    // Write all per-taxon sections with renumbered ordinals
    for (uint32_t ord = 0; ord < static_cast<uint32_t>(all_taxa.size()); ++ord) {
        auto& ts = all_taxa[ord];
        uint64_t sid_base = static_cast<uint64_t>(ord) * 16;

        for (auto& [orig_desc, cdata] : ts.sections) {
            // Compute new section_id: same offset within taxon, new base
            uint64_t offset_in_taxon = orig_desc.section_id % 16;
            write_compressed_section(
                orig_desc.type,
                sid_base + offset_in_taxon,
                cdata,
                orig_desc.uncompressed_size,
                orig_desc.item_count,
                orig_desc.aux0,
                orig_desc.aux1);
        }

        // Build TIDX entry
        TaxonIndexEntry te{};
        te.taxonomy_hash = fnv1a_64(ts.taxonomy);
        te.section_id_base = sid_base;
        te.strtable_offset = intern(ts.taxonomy);
        te.n_genomes = ts.n_genomes;
        te.n_reps = ts.n_reps;
        te.n_contaminated = ts.n_contam;
        tidx_entries.push_back(te);

        // Build ACCX entries — need to decompress GMET to get accessions
        for (auto& [desc, cdata] : ts.sections) {
            if (desc.type != SEC_GMET) continue;
            auto raw = zstd_decompress(cdata.data(), cdata.size(),
                                       desc.uncompressed_size);
            if (raw.size() < 4) break;
            uint32_t n;
            std::memcpy(&n, raw.data(), 4);
            // Accession offsets at raw[4..4+n*4], then string data
            if (raw.size() < 4 + n * 4) break;
            std::vector<uint32_t> acc_off(n);
            std::memcpy(acc_off.data(), raw.data() + 4, n * 4);
            const char* str_base = reinterpret_cast<const char*>(raw.data()) + 4 + n * 4;
            for (uint32_t i = 0; i < n; ++i) {
                const char* s = str_base + acc_off[i];
                AccessionIndexEntry ae{};
                ae.accession_hash = fnv1a_64(s, std::strlen(s));
                ae.taxon_ordinal = ord;
                ae.genome_idx = i;
                accx_entries.push_back(ae);
            }
            break;
        }

        total_genomes += ts.n_genomes;
    }

    // Compress helper (for global sections)
    auto compress = [](const void* data, size_t size) -> std::vector<uint8_t> {
        size_t bound = ZSTD_compressBound(size);
        std::vector<uint8_t> out(bound);
        size_t csize = ZSTD_compress(out.data(), bound, data, size, 3);
        if (ZSTD_isError(csize))
            throw std::runtime_error(std::string("grd_merge: zstd: ") +
                                     ZSTD_getErrorName(csize));
        out.resize(csize);
        return out;
    };

    auto write_section = [&](uint32_t type, uint64_t section_id,
                             const void* data, size_t size, uint64_t item_count) {
        auto cdata = compress(data, size);
        write_compressed_section(type, section_id, cdata, size, item_count);
    };

    // Write TIDX
    std::sort(tidx_entries.begin(), tidx_entries.end(),
              [](const TaxonIndexEntry& a, const TaxonIndexEntry& b) {
                  return a.taxonomy_hash < b.taxonomy_hash;
              });
    write_section(SEC_TIDX, 0xFFFF'0000,
                  tidx_entries.data(),
                  tidx_entries.size() * sizeof(TaxonIndexEntry),
                  tidx_entries.size());

    // Write ACCX
    std::sort(accx_entries.begin(), accx_entries.end(),
              [](const AccessionIndexEntry& a, const AccessionIndexEntry& b) {
                  return a.accession_hash < b.accession_hash;
              });
    write_section(SEC_ACCX, 0xFFFF'0001,
                  accx_entries.data(),
                  accx_entries.size() * sizeof(AccessionIndexEntry),
                  accx_entries.size());

    // Write STRT
    {
        std::vector<uint8_t> raw;
        uint32_t n_strings = static_cast<uint32_t>(strtable.size());
        raw.insert(raw.end(),
                   reinterpret_cast<const uint8_t*>(&n_strings),
                   reinterpret_cast<const uint8_t*>(&n_strings) + 4);
        for (const auto& s : strtable) {
            raw.insert(raw.end(), s.begin(), s.end());
            raw.push_back('\0');
        }
        write_section(SEC_STRT, 0xFFFF'0002,
                      raw.data(), raw.size(), n_strings);
    }

    // Write TOC
    uint64_t toc_offset = file_offset;
    {
        TocHeader toc_hdr{};
        toc_hdr.magic = TOCB_MAGIC;
        toc_hdr.version = 1;
        toc_hdr.section_count = all_sections.size();
        toc_hdr.n_taxa = all_taxa.size();
        toc_hdr.n_genomes_total = total_genomes;

        std::vector<uint8_t> toc_raw(sizeof(TocHeader));
        std::memcpy(toc_raw.data(), &toc_hdr, sizeof(toc_hdr));
        for (const auto& sec : all_sections) {
            size_t off = toc_raw.size();
            toc_raw.resize(off + sizeof(SectionDesc));
            std::memcpy(toc_raw.data() + off, &sec, sizeof(sec));
        }

        auto toc_compressed = compress(toc_raw.data(), toc_raw.size());
        safe_write(toc_compressed.data(), toc_compressed.size(), file_offset);
        uint64_t toc_csize = toc_compressed.size();
        file_offset += toc_csize;

        // Write TailLocator
        TailLocator tail{};
        tail.magic = GRDT_MAGIC;
        tail.version = 1;
        tail.toc_offset = toc_offset;
        tail.toc_size = toc_csize;
        tail.n_taxa = all_taxa.size();
        safe_write(&tail, sizeof(tail), file_offset);
        file_offset += sizeof(tail);
    }

    ::close(fd);

    MergeStats stats;
    stats.n_shards = shards.size();
    stats.n_taxa = all_taxa.size();
    stats.n_genomes = total_genomes;
    stats.output_bytes = file_offset;

    spdlog::info("grd_merge: wrote {} taxa, {} genomes, {} bytes → {}",
                 stats.n_taxa, stats.n_genomes, stats.output_bytes,
                 output.string());

    return stats;
}

} // namespace grd
