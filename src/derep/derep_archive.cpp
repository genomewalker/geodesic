#include "derep_archive.hpp"
#include <genopack/archive_set_reader.hpp>
#include <genopack/archive.hpp>

#define XXH_STATIC_LINKING_ONLY
#include <xxhash.h>
// XXH_IMPLEMENTATION is provided by libgenopack_lib (derep_view.cpp); defining
// it here too produces multiple-definition link errors when both objects are
// linked into the same binary (e.g. the integration test that uses DerepView).

#include <zstd.h>
#include <zlib.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <random>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace geodesic {

// ── on-disk structures ────────────────────────────────────────────────────────

static constexpr uint32_t FILE_MAGIC  = 0x46445047u; // 'GPDF'
static constexpr uint32_t TAIL_MAGIC  = 0x54445047u; // 'GPDT'
static constexpr uint32_t HDR_MAGIC   = 0x48445047u; // 'GPDH'
static constexpr uint32_t ASTR_MAGIC  = 0x52545341u; // 'ASTR' (raw pool, no section magic in payload)
static constexpr uint32_t ASOF_MAGIC  = 0x4F534147u; // 'GASO'
static constexpr uint32_t ARMP_MAGIC  = 0x4D524147u; // 'GARM'
static constexpr uint32_t RTBL_MAGIC  = 0x42545247u; // 'GRTB'
static constexpr uint32_t G2RM_MAGIC  = 0x4D523247u; // 'G2RM'
static constexpr uint32_t EMBD_MAGIC  = 0x424D4547u; // 'GEMB'
static constexpr uint32_t TOC_MAGIC   = 0x434F5447u; // 'GTOC'

static constexpr uint32_t GPD_SEC_HDR = 0x52444847u; // 'GHDR'
static constexpr uint32_t GPD_SEC_AST = 0x54534147u; // 'GAST'
static constexpr uint32_t GPD_SEC_ASO = 0x4F534147u; // 'GASO'
static constexpr uint32_t GPD_SEC_ARM = 0x4D524147u; // 'GARM'
static constexpr uint32_t GPD_SEC_RTB = 0x42545247u; // 'GRTB'
static constexpr uint32_t GPD_SEC_G2R = 0x4D523247u; // 'G2RM'
static constexpr uint32_t GPD_SEC_EMB = 0x424D4547u; // 'GEMB'
static constexpr uint32_t GPD_SEC_TOC = 0x434F5447u; // 'GTOC'

static constexpr uint32_t SENTINEL_UNCLUSTERED = 0xFFFFFFFEu;
static constexpr uint32_t EMPTY_BUCKET         = 0xFFFFFFFFu;

#pragma pack(push, 1)
struct GpdFileHeader {
    uint32_t magic;
    uint16_t format_major;
    uint16_t format_minor;
    uint64_t toc_offset;
    uint64_t toc_size;
    uint64_t reserved[5];
};
static_assert(sizeof(GpdFileHeader) == 64);

struct GpdTailLocator {
    uint64_t toc_offset;
    uint32_t magic;
    uint32_t crc32;
};
static_assert(sizeof(GpdTailLocator) == 16);

struct GpdSectionDesc {
    uint32_t type;
    uint32_t flags;
    uint64_t file_offset;
    uint64_t compressed_size;
    uint64_t uncompressed_size;
    uint64_t section_id;
    uint64_t reserved[2];
};
static_assert(sizeof(GpdSectionDesc) == 56);

struct GpdHeader {
    uint32_t magic;
    uint16_t format_major;
    uint16_t format_minor;
    uint64_t created_at_unix;
    uint8_t  run_id[16];
    uint16_t n_parts;
    uint16_t embedding_dim;
    uint8_t  embedding_dtype;
    uint8_t  has_cstats;
    uint8_t  pad0[2];
    uint64_t n_genomes;
    uint64_t n_reps;
    uint64_t n_unclustered;
};
static_assert(sizeof(GpdHeader) == 64);

struct GpdSourcePart {
    uint8_t  archive_uuid[16];
    uint64_t generation;
    uint64_t n_genomes_total;
    uint64_t n_genomes_live;
    uint64_t accession_set_hash;
};
static_assert(sizeof(GpdSourcePart) == 48);

struct GpdDerepParams {
    uint8_t  n_kmer_sizes;
    uint8_t  kmer_sizes[7];
    uint32_t sketch_size;
    uint64_t sig1_seed;
    uint64_t sig2_seed;
    float    jaccard_thresh;
    uint16_t geodesic_ver_len;
    uint8_t  pad1[2];
};
static_assert(sizeof(GpdDerepParams) == 36);

struct GpdRepEntry {
    uint32_t rep_acc_ord;
    uint32_t cluster_size;
    uint64_t source_locator;
    uint16_t sketch_kmer;
    uint8_t  flags;
    uint8_t  pad;
    uint32_t cstat_offset;
};
static_assert(sizeof(GpdRepEntry) == 24);

struct GpdArmpEntry {
    uint64_t hash;
    uint32_t ordinal;
    uint32_t pad;
};
static_assert(sizeof(GpdArmpEntry) == 16);
#pragma pack(pop)

// ── helpers ───────────────────────────────────────────────────────────────────

static void pwrite_all(int fd, const void* data, size_t len, uint64_t offset) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    while (len > 0) {
        ssize_t n = ::pwrite(fd, p, len, static_cast<off_t>(offset));
        if (n < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(std::string("gpd: pwrite: ") + strerror(errno));
        }
        p      += n;
        offset += static_cast<uint64_t>(n);
        len    -= static_cast<size_t>(n);
    }
}

static std::vector<uint8_t> zstd_compress(const void* data, size_t size, int level) {
    size_t bound = ZSTD_compressBound(size);
    std::vector<uint8_t> out(bound);
    size_t csize = ZSTD_compress(out.data(), bound, data, size, level);
    if (ZSTD_isError(csize))
        throw std::runtime_error(std::string("gpd: zstd: ") + ZSTD_getErrorName(csize));
    out.resize(csize);
    return out;
}

static uint64_t align8(uint64_t x) { return (x + 7u) & ~uint64_t{7}; }

static void gen_uuid_v4(uint8_t out[16]) {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    uint64_t a = gen(), b = gen();
    std::memcpy(out,     &a, 8);
    std::memcpy(out + 8, &b, 8);
    out[6] = (out[6] & 0x0fu) | 0x40u;
    out[8] = (out[8] & 0x3fu) | 0x80u;
}

static uint32_t next_pow2_ge(uint32_t n) {
    uint32_t v = 1;
    while (v < n) v <<= 1;
    return v;
}

static uint64_t compute_accession_set_hash(genopack::ArchiveReader& reader) {
    std::vector<std::string> accs;
    reader.scan_genome_accessions([&](std::string_view a, genopack::GenomeId) {
        accs.emplace_back(a);
    });
    std::sort(accs.begin(), accs.end());
    XXH3_state_t* st = XXH3_createState();
    XXH3_64bits_reset(st);
    for (size_t i = 0; i < accs.size(); ++i) {
        if (i > 0) { char nl = '\n'; XXH3_64bits_update(st, &nl, 1); }
        XXH3_64bits_update(st, accs[i].data(), accs[i].size());
    }
    uint64_t h = XXH3_64bits_digest(st);
    XXH3_freeState(st);
    return h;
}

// ── Impl ──────────────────────────────────────────────────────────────────────

struct DerepArchiveBuilder::Impl {
    DerepArchiveBuilderConfig cfg;

    std::vector<DerepArchiveBuilder::PartFingerprint> parts;

    std::vector<uint8_t> kmer_sizes_v;
    uint32_t             sketch_size   = 0;
    uint64_t             sig1_seed     = 0;
    uint64_t             sig2_seed     = 0;
    float                jaccard_thresh = 0.f;

    struct GenomeRecord {
        std::string accession;
        std::string rep_accession;
        uint64_t    source_locator;
        uint32_t    cluster_size;
        uint16_t    sketch_kmer;
        DerepArchiveBuilder::Kind kind;
        std::vector<uint8_t> embedding;
    };
    std::vector<GenomeRecord> records;

    explicit Impl(DerepArchiveBuilderConfig c) : cfg(std::move(c)) {}
};

// ── DerepArchiveBuilder ───────────────────────────────────────────────────────

DerepArchiveBuilder::DerepArchiveBuilder(DerepArchiveBuilderConfig cfg)
    : impl_(std::make_unique<Impl>(std::move(cfg))) {}

DerepArchiveBuilder::~DerepArchiveBuilder() = default;

void DerepArchiveBuilder::set_source_pack(const genopack::ArchiveSetReader& pack) {
    const auto& paths = pack.part_paths();
    impl_->parts.clear();
    impl_->parts.reserve(paths.size());

    for (size_t i = 0; i < paths.size(); ++i) {
        genopack::ArchiveReader reader;
        reader.open(paths[i]);

        auto stats = reader.archive_stats();

        PartFingerprint fp{};
        // genopack FileHeader layout: magic(4)+major(2)+minor(2)+uuid_lo(8)+uuid_hi(8)+...
        // Read from archive file (single .gpk) or toc.bin (directory layout).
        {
            auto try_read_uuid = [&](const std::filesystem::path& p) -> bool {
                int rfd = ::open(p.c_str(), O_RDONLY);
                if (rfd < 0) return false;
                uint8_t buf[24] = {};
                bool ok = (::read(rfd, buf, sizeof(buf)) == static_cast<ssize_t>(sizeof(buf)));
                ::close(rfd);
                if (!ok) return false;
                std::memcpy(fp.archive_uuid,     buf + 8,  8);
                std::memcpy(fp.archive_uuid + 8, buf + 16, 8);
                return true;
            };
            if (!try_read_uuid(paths[i]))
                try_read_uuid(paths[i] / "toc.bin");
        }
        fp.generation      = stats.generation;
        fp.n_genomes_total = stats.n_genomes_total;
        fp.n_genomes_live  = stats.n_genomes_live;
        fp.accession_set_hash = compute_accession_set_hash(reader);
        impl_->parts.push_back(fp);
    }
}

void DerepArchiveBuilder::set_source_pack_manual(
    std::vector<PartFingerprint> parts) {
    impl_->parts = std::move(parts);
}

void DerepArchiveBuilder::set_params(const std::vector<uint8_t>& kmer_sizes,
                                     uint32_t sketch_size,
                                     uint64_t sig1_seed, uint64_t sig2_seed,
                                     float    jaccard_thresh) {
    impl_->kmer_sizes_v    = kmer_sizes;
    impl_->sketch_size     = sketch_size;
    impl_->sig1_seed       = sig1_seed;
    impl_->sig2_seed       = sig2_seed;
    impl_->jaccard_thresh  = jaccard_thresh;
}

void DerepArchiveBuilder::add(std::string_view accession,
                              Kind             kind,
                              std::string_view rep_accession,
                              uint64_t         source_locator,
                              uint16_t         sketch_kmer,
                              uint32_t         cluster_size,
                              const void*      embedding_or_null) {
    Impl::GenomeRecord rec;
    rec.accession      = std::string(accession);
    rec.rep_accession  = std::string(rep_accession);
    rec.source_locator = source_locator;
    rec.cluster_size   = cluster_size;
    rec.sketch_kmer    = sketch_kmer;
    rec.kind           = kind;

    if (kind == Kind::Representative && embedding_or_null) {
        size_t bytes = static_cast<size_t>(impl_->cfg.embedding_dim)
                       * (impl_->cfg.embedding_dtype == 1 ? 2u : 4u);
        rec.embedding.resize(bytes);
        std::memcpy(rec.embedding.data(), embedding_or_null, bytes);
    }
    impl_->records.push_back(std::move(rec));
}

void DerepArchiveBuilder::finalize() {
    auto& imp = *impl_;
    auto& cfg = imp.cfg;

    // ── validation ────────────────────────────────────────────────────────────
    uint64_t n_reps_found       = 0;
    uint64_t n_unclustered      = 0;
    for (auto& r : imp.records) {
        if (r.kind == Kind::Representative) {
            ++n_reps_found;
            if (r.embedding.empty())
                throw std::runtime_error("gpd: rep without embedding: " + r.accession);
        } else if (r.kind == Kind::Member) {
            if (r.rep_accession.empty())
                throw std::runtime_error("gpd: member without rep_accession: " + r.accession);
        } else {
            ++n_unclustered;
        }
    }
    const uint64_t n_genomes = imp.records.size();
    const uint64_t n_reps    = n_reps_found;

    // ── sort accessions → assign ordinals ────────────────────────────────────
    std::vector<size_t> order(n_genomes);
    for (size_t i = 0; i < n_genomes; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return imp.records[a].accession < imp.records[b].accession;
    });

    // ordinal_of[original_idx] = sorted ordinal
    std::vector<uint32_t> ordinal_of(n_genomes);
    for (uint32_t o = 0; o < static_cast<uint32_t>(n_genomes); ++o)
        ordinal_of[order[o]] = o;

    // ── ASTR ──────────────────────────────────────────────────────────────────
    std::string astr;
    astr.reserve(n_genomes * 20);
    for (size_t o = 0; o < n_genomes; ++o)
        astr += imp.records[order[o]].accession;

    // ── ASOF ──────────────────────────────────────────────────────────────────
    struct AsofPayload {
        uint32_t magic    = ASOF_MAGIC;
        uint32_t n_genomes;
        uint64_t pad      = 0;
    };
    std::vector<uint32_t> offsets(n_genomes + 1);
    {
        uint32_t pos = 0;
        for (size_t o = 0; o < n_genomes; ++o) {
            offsets[o] = pos;
            pos       += static_cast<uint32_t>(imp.records[order[o]].accession.size());
        }
        offsets[n_genomes] = pos;
    }

    std::vector<uint8_t> asof_raw;
    {
        AsofPayload hdr2{};
        hdr2.n_genomes = static_cast<uint32_t>(n_genomes);
        asof_raw.resize(sizeof(AsofPayload) + (n_genomes + 1) * sizeof(uint32_t));
        std::memcpy(asof_raw.data(), &hdr2, sizeof(AsofPayload));
        std::memcpy(asof_raw.data() + sizeof(AsofPayload),
                    offsets.data(), (n_genomes + 1) * sizeof(uint32_t));
    }

    // acc → ordinal map for rep resolution
    std::unordered_map<std::string_view, uint32_t> acc_to_ord;
    acc_to_ord.reserve(n_genomes * 2);
    for (size_t orig = 0; orig < n_genomes; ++orig)
        acc_to_ord[std::string_view(imp.records[orig].accession)] = ordinal_of[orig];

    // ── build RTBL rows (before sorting) ──────────────────────────────────────
    struct RepRow {
        uint32_t rep_acc_ord;
        uint32_t cluster_size;
        uint64_t source_locator;
        uint16_t sketch_kmer;
        size_t   orig_idx;
    };
    std::vector<RepRow> rep_rows;
    rep_rows.reserve(n_reps);
    for (size_t i = 0; i < n_genomes; ++i) {
        if (imp.records[i].kind != Kind::Representative) continue;
        RepRow rr{};
        rr.rep_acc_ord   = ordinal_of[i];
        rr.cluster_size  = imp.records[i].cluster_size;
        rr.source_locator = imp.records[i].source_locator;
        rr.sketch_kmer   = imp.records[i].sketch_kmer;
        rr.orig_idx      = i;
        rep_rows.push_back(rr);
    }
    std::sort(rep_rows.begin(), rep_rows.end(),
              [](const RepRow& a, const RepRow& b) {
                  return a.rep_acc_ord < b.rep_acc_ord;
              });

    // validate monotonic
    for (size_t i = 1; i < rep_rows.size(); ++i)
        assert(rep_rows[i].rep_acc_ord > rep_rows[i-1].rep_acc_ord);

    // rep_id: index in rep_rows
    // build rep_acc_ord → rep_id map
    std::unordered_map<uint32_t, uint32_t> ord_to_repid;
    ord_to_repid.reserve(n_reps * 2);
    for (uint32_t r = 0; r < static_cast<uint32_t>(rep_rows.size()); ++r)
        ord_to_repid[rep_rows[r].rep_acc_ord] = r;

    // rep_acc string → rep_id (for member lookup)
    std::unordered_map<std::string_view, uint32_t> rep_acc_to_repid;
    rep_acc_to_repid.reserve(n_reps * 2);
    for (size_t r = 0; r < rep_rows.size(); ++r) {
        size_t orig = rep_rows[r].orig_idx;
        rep_acc_to_repid[std::string_view(imp.records[orig].accession)] =
            static_cast<uint32_t>(r);
    }

    // ── G2RM ──────────────────────────────────────────────────────────────────
    std::vector<uint32_t> g2rm(n_genomes, EMPTY_BUCKET);
    for (size_t i = 0; i < n_genomes; ++i) {
        uint32_t ord = ordinal_of[i];
        if (imp.records[i].kind == Kind::Unclustered) {
            g2rm[ord] = SENTINEL_UNCLUSTERED;
        } else if (imp.records[i].kind == Kind::Representative) {
            auto it = ord_to_repid.find(ordinal_of[i]);
            assert(it != ord_to_repid.end());
            g2rm[ord] = it->second;
        } else {
            auto it = rep_acc_to_repid.find(
                std::string_view(imp.records[i].rep_accession));
            if (it == rep_acc_to_repid.end())
                throw std::runtime_error("gpd: member references unknown rep: "
                                         + imp.records[i].rep_accession);
            g2rm[ord] = it->second;
        }
    }

    // ── ARMP ──────────────────────────────────────────────────────────────────
    std::vector<uint8_t> armp_raw;
    if (cfg.emit_armp) {
        uint32_t n_buckets = next_pow2_ge(
            static_cast<uint32_t>(n_genomes * 143 / 100 + 1));

        struct ArmpHeader {
            uint32_t magic     = ARMP_MAGIC;
            uint32_t n_buckets;
            uint32_t hash_seed = 0;
            uint32_t pad       = 0;
        };
        ArmpHeader ah{};
        ah.n_buckets = n_buckets;

        std::vector<GpdArmpEntry> entries(n_buckets);
        for (auto& e : entries) { e.hash = 0; e.ordinal = EMPTY_BUCKET; e.pad = 0; }

        for (size_t o = 0; o < n_genomes; ++o) {
            const std::string& acc = imp.records[order[o]].accession;
            XXH128_hash_t h128 = XXH3_128bits(acc.data(), acc.size());
            uint64_t      hkey = h128.high64;
            uint32_t      bucket = static_cast<uint32_t>(hkey & (n_buckets - 1));
            while (entries[bucket].ordinal != EMPTY_BUCKET)
                bucket = (bucket + 1) & (n_buckets - 1);
            entries[bucket].hash    = hkey;
            entries[bucket].ordinal = static_cast<uint32_t>(o);
        }

        armp_raw.resize(sizeof(ArmpHeader) + n_buckets * sizeof(GpdArmpEntry));
        std::memcpy(armp_raw.data(), &ah, sizeof(ArmpHeader));
        std::memcpy(armp_raw.data() + sizeof(ArmpHeader),
                    entries.data(), n_buckets * sizeof(GpdArmpEntry));
    }

    // ── RTBL raw ──────────────────────────────────────────────────────────────
    struct RtblHeader {
        uint32_t magic  = RTBL_MAGIC;
        uint32_t n_reps;
        uint64_t pad    = 0;
    };
    RtblHeader rtbl_hdr{};
    rtbl_hdr.n_reps = static_cast<uint32_t>(n_reps);

    std::vector<GpdRepEntry> rtbl_entries(n_reps);
    for (size_t r = 0; r < rep_rows.size(); ++r) {
        GpdRepEntry& e = rtbl_entries[r];
        e.rep_acc_ord   = rep_rows[r].rep_acc_ord;
        e.cluster_size  = rep_rows[r].cluster_size;
        e.source_locator= rep_rows[r].source_locator;
        e.sketch_kmer   = rep_rows[r].sketch_kmer;
        e.flags         = 0x01; // has_embedding
        e.pad           = 0;
        e.cstat_offset  = 0xFFFFFFFFu;
    }

    // ── EMBD raw ──────────────────────────────────────────────────────────────
    struct EmbdHeader {
        uint32_t magic  = EMBD_MAGIC;
        uint16_t dim;
        uint8_t  dtype;
        uint8_t  pad0   = 0;
        uint32_t n_reps;
        uint32_t pad1   = 0;
    };
    EmbdHeader embd_hdr{};
    embd_hdr.dim    = cfg.embedding_dim;
    embd_hdr.dtype  = cfg.embedding_dtype;
    embd_hdr.n_reps = static_cast<uint32_t>(n_reps);

    size_t embd_bytes_per_rep = static_cast<size_t>(cfg.embedding_dim)
                                * (cfg.embedding_dtype == 1 ? 2u : 4u);
    std::vector<uint8_t> embd_matrix(n_reps * embd_bytes_per_rep);
    for (size_t r = 0; r < rep_rows.size(); ++r) {
        const auto& emb = imp.records[rep_rows[r].orig_idx].embedding;
        std::memcpy(embd_matrix.data() + r * embd_bytes_per_rep,
                    emb.data(), embd_bytes_per_rep);
    }

    // ── HDR raw ───────────────────────────────────────────────────────────────
    GpdHeader hdr{};
    hdr.magic          = HDR_MAGIC;
    hdr.format_major   = 1;
    hdr.format_minor   = 0;
    hdr.created_at_unix = static_cast<uint64_t>(std::time(nullptr));
    gen_uuid_v4(hdr.run_id);
    hdr.n_parts        = static_cast<uint16_t>(imp.parts.size());
    hdr.embedding_dim  = cfg.embedding_dim;
    hdr.embedding_dtype = cfg.embedding_dtype;
    hdr.has_cstats     = 0;
    hdr.n_genomes      = n_genomes;
    hdr.n_reps         = n_reps;
    hdr.n_unclustered  = n_unclustered;

    std::string version_str = cfg.geodesic_version;
    uint16_t ver_len = static_cast<uint16_t>(version_str.size());
    size_t   ver_padded = align8(ver_len);

    GpdDerepParams dparams{};
    dparams.n_kmer_sizes   = static_cast<uint8_t>(
        std::min(imp.kmer_sizes_v.size(), size_t{7}));
    for (uint8_t i = 0; i < dparams.n_kmer_sizes; ++i)
        dparams.kmer_sizes[i] = imp.kmer_sizes_v[i];
    dparams.sketch_size    = imp.sketch_size;
    dparams.sig1_seed      = imp.sig1_seed;
    dparams.sig2_seed      = imp.sig2_seed;
    dparams.jaccard_thresh = imp.jaccard_thresh;
    dparams.geodesic_ver_len = ver_len;

    std::vector<uint8_t> hdr_raw;
    hdr_raw.reserve(sizeof(GpdHeader)
                    + imp.parts.size() * sizeof(GpdSourcePart)
                    + sizeof(GpdDerepParams)
                    + ver_padded);
    auto push = [&](const void* p, size_t n) {
        const uint8_t* b = static_cast<const uint8_t*>(p);
        hdr_raw.insert(hdr_raw.end(), b, b + n);
    };
    push(&hdr, sizeof(GpdHeader));
    for (auto& fp : imp.parts) {
        GpdSourcePart sp{};
        std::memcpy(sp.archive_uuid, fp.archive_uuid, 16);
        sp.generation          = fp.generation;
        sp.n_genomes_total     = fp.n_genomes_total;
        sp.n_genomes_live      = fp.n_genomes_live;
        sp.accession_set_hash  = fp.accession_set_hash;
        push(&sp, sizeof(sp));
    }
    push(&dparams, sizeof(dparams));
    if (!version_str.empty())
        push(version_str.data(), version_str.size());
    // pad to align8
    if (ver_padded > ver_len) {
        uint8_t zeros[8] = {};
        push(zeros, ver_padded - ver_len);
    }

    // ── G2RM raw ──────────────────────────────────────────────────────────────
    struct G2rmHeader {
        uint32_t magic     = G2RM_MAGIC;
        uint32_t n_genomes;
        uint64_t pad       = 0;
    };
    G2rmHeader g2rm_hdr{};
    g2rm_hdr.n_genomes = static_cast<uint32_t>(n_genomes);

    std::vector<uint8_t> g2rm_raw;
    g2rm_raw.resize(sizeof(G2rmHeader) + n_genomes * sizeof(uint32_t));
    std::memcpy(g2rm_raw.data(), &g2rm_hdr, sizeof(G2rmHeader));
    std::memcpy(g2rm_raw.data() + sizeof(G2rmHeader),
                g2rm.data(), n_genomes * sizeof(uint32_t));

    // ── compress sections ─────────────────────────────────────────────────────
    auto astr_c  = zstd_compress(astr.data(), astr.size(), cfg.zstd_level);
    auto asof_c  = zstd_compress(asof_raw.data(), asof_raw.size(), cfg.zstd_level);
    auto g2rm_c  = zstd_compress(g2rm_raw.data(), g2rm_raw.size(), cfg.zstd_level);

    std::vector<uint8_t> armp_c;
    if (cfg.emit_armp && !armp_raw.empty())
        armp_c = zstd_compress(armp_raw.data(), armp_raw.size(), cfg.zstd_level);

    // ── open output ───────────────────────────────────────────────────────────
    {
        auto par = cfg.output_path.parent_path();
        if (!par.empty()) std::filesystem::create_directories(par);
    }
    int fd = ::open(cfg.output_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0)
        throw std::runtime_error("gpd: cannot open: " + cfg.output_path.string());

    // ── layout pass: place sections after 64-byte file header ────────────────
    struct SecInfo {
        uint32_t type;
        uint32_t flags;
        uint64_t file_offset;
        uint64_t compressed_size;
        uint64_t uncompressed_size;
    };
    std::vector<SecInfo> toc_secs;
    uint64_t pos = sizeof(GpdFileHeader);

    auto place_section = [&](uint32_t type, uint32_t flags,
                              const void* data, uint64_t csz, uint64_t usz) {
        SecInfo si{type, flags, pos, csz, usz};
        toc_secs.push_back(si);
        pwrite_all(fd, data, csz, pos);
        pos += align8(csz);
    };

    // HDR (uncompressed)
    place_section(GPD_SEC_HDR, 0,
                  hdr_raw.data(), hdr_raw.size(), hdr_raw.size());

    // ASTR (compressed)
    place_section(GPD_SEC_AST, 1,
                  astr_c.data(), astr_c.size(), astr.size());

    // ASOF (compressed)
    place_section(GPD_SEC_ASO, 1,
                  asof_c.data(), asof_c.size(), asof_raw.size());

    // ARMP (compressed, optional)
    if (cfg.emit_armp && !armp_c.empty())
        place_section(GPD_SEC_ARM, 1,
                      armp_c.data(), armp_c.size(), armp_raw.size());

    // RTBL (uncompressed — fixed 24-byte rows for O(1) random access)
    {
        std::vector<uint8_t> rtbl_raw;
        rtbl_raw.resize(sizeof(RtblHeader) + n_reps * sizeof(GpdRepEntry));
        std::memcpy(rtbl_raw.data(), &rtbl_hdr, sizeof(RtblHeader));
        std::memcpy(rtbl_raw.data() + sizeof(RtblHeader),
                    rtbl_entries.data(), n_reps * sizeof(GpdRepEntry));
        place_section(GPD_SEC_RTB, 0,
                      rtbl_raw.data(), rtbl_raw.size(), rtbl_raw.size());
    }

    // G2RM (compressed)
    place_section(GPD_SEC_G2R, 1,
                  g2rm_c.data(), g2rm_c.size(), g2rm_raw.size());

    // EMBD (uncompressed — mmap-friendly)
    {
        std::vector<uint8_t> embd_raw;
        embd_raw.resize(sizeof(EmbdHeader) + embd_matrix.size());
        std::memcpy(embd_raw.data(), &embd_hdr, sizeof(EmbdHeader));
        std::memcpy(embd_raw.data() + sizeof(EmbdHeader),
                    embd_matrix.data(), embd_matrix.size());
        place_section(GPD_SEC_EMB, 0,
                      embd_raw.data(), embd_raw.size(), embd_raw.size());
    }

    // ── TOC ───────────────────────────────────────────────────────────────────
    uint64_t toc_offset = pos;

    struct TocPayloadHeader {
        uint32_t magic              = TOC_MAGIC;
        uint32_t n_sections;
        uint32_t crc32_of_descs;
        uint32_t pad                = 0;
    };

    uint32_t n_sec = static_cast<uint32_t>(toc_secs.size());
    std::vector<GpdSectionDesc> descs(n_sec);
    for (uint32_t i = 0; i < n_sec; ++i) {
        descs[i].type              = toc_secs[i].type;
        descs[i].flags             = toc_secs[i].flags;
        descs[i].file_offset       = toc_secs[i].file_offset;
        descs[i].compressed_size   = toc_secs[i].compressed_size;
        descs[i].uncompressed_size = toc_secs[i].uncompressed_size;
        descs[i].section_id        = i + 1;
        descs[i].reserved[0]       = 0;
        descs[i].reserved[1]       = 0;
    }

    uLong crc = crc32(0L, reinterpret_cast<const Bytef*>(descs.data()),
                      static_cast<uInt>(descs.size() * sizeof(GpdSectionDesc)));

    TocPayloadHeader tph{};
    tph.n_sections       = n_sec;
    tph.crc32_of_descs   = static_cast<uint32_t>(crc);

    uint64_t toc_size = sizeof(TocPayloadHeader) + n_sec * sizeof(GpdSectionDesc);
    pwrite_all(fd, &tph,  sizeof(tph),   pos);
    pwrite_all(fd, descs.data(), n_sec * sizeof(GpdSectionDesc),
               pos + sizeof(tph));
    pos += align8(toc_size);

    // ── FileHeader ────────────────────────────────────────────────────────────
    GpdFileHeader fhdr{};
    fhdr.magic        = FILE_MAGIC;
    fhdr.format_major = 1;
    fhdr.format_minor = 0;
    fhdr.toc_offset   = toc_offset;
    fhdr.toc_size     = toc_size;
    std::memset(fhdr.reserved, 0, sizeof(fhdr.reserved));
    pwrite_all(fd, &fhdr, sizeof(fhdr), 0);

    // ── TailLocator ───────────────────────────────────────────────────────────
    GpdTailLocator tail{};
    tail.toc_offset = toc_offset;
    tail.magic      = TAIL_MAGIC;
    tail.crc32      = static_cast<uint32_t>(crc);
    pwrite_all(fd, &tail, sizeof(tail), pos);
    pos += sizeof(tail);

    ::close(fd);
}

} // namespace geodesic
