#include "derep_archive.hpp"
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <random>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#define XXH_STATIC_LINKING_ONLY
#include <xxhash.h>

#include <zstd.h>

namespace {

static constexpr uint32_t FILE_MAGIC  = 0x46445047u;
static constexpr uint32_t TAIL_MAGIC  = 0x54445047u;
static constexpr uint32_t HDR_MAGIC   = 0x48445047u;
static constexpr uint32_t ASOF_MAGIC  = 0x4F534147u;
static constexpr uint32_t RTBL_MAGIC  = 0x42545247u;
static constexpr uint32_t G2RM_MAGIC  = 0x4D523247u;
static constexpr uint32_t EMBD_MAGIC  = 0x424D4547u;
static constexpr uint32_t TOC_MAGIC   = 0x434F5447u;

static constexpr uint32_t GPD_SEC_HDR = 0x52444847u;
static constexpr uint32_t GPD_SEC_AST = 0x54534147u;
static constexpr uint32_t GPD_SEC_ASO = 0x4F534147u;
static constexpr uint32_t GPD_SEC_RTB = 0x42545247u;
static constexpr uint32_t GPD_SEC_G2R = 0x4D523247u;
static constexpr uint32_t GPD_SEC_EMB = 0x424D4547u;

#pragma pack(push,1)
struct GpdFileHeader {
    uint32_t magic; uint16_t format_major; uint16_t format_minor;
    uint64_t toc_offset; uint64_t toc_size; uint64_t reserved[5];
};
static_assert(sizeof(GpdFileHeader) == 64);

struct GpdTailLocator { uint64_t toc_offset; uint32_t magic; uint32_t crc32; };
static_assert(sizeof(GpdTailLocator) == 16);

struct GpdSectionDesc {
    uint32_t type; uint32_t flags;
    uint64_t file_offset; uint64_t compressed_size; uint64_t uncompressed_size;
    uint64_t section_id; uint64_t reserved[2];
};
static_assert(sizeof(GpdSectionDesc) == 56);

struct GpdHeader {
    uint32_t magic; uint16_t format_major; uint16_t format_minor;
    uint64_t created_at_unix; uint8_t run_id[16];
    uint16_t n_parts; uint16_t embedding_dim;
    uint8_t embedding_dtype; uint8_t has_cstats; uint8_t pad0[2];
    uint64_t n_genomes; uint64_t n_reps; uint64_t n_unclustered;
};
static_assert(sizeof(GpdHeader) == 64);

struct AsofPayload { uint32_t magic; uint32_t n_genomes; uint64_t pad; };

struct RtblHeader { uint32_t magic; uint32_t n_reps; uint64_t pad; };

struct GpdRepEntry {
    uint32_t rep_acc_ord; uint32_t cluster_size;
    uint64_t source_locator; uint16_t sketch_kmer;
    uint8_t flags; uint8_t pad; uint32_t cstat_offset;
};
static_assert(sizeof(GpdRepEntry) == 24);

struct G2rmHeader { uint32_t magic; uint32_t n_genomes; uint64_t pad; };

struct EmbdHeader {
    uint32_t magic; uint16_t dim; uint8_t dtype; uint8_t pad0;
    uint32_t n_reps; uint32_t pad1;
};
struct TocHeader { uint32_t magic; uint32_t n_sections; uint32_t crc32; uint32_t pad; };
#pragma pack(pop)

// Synthesise 1000 genomes, 100 reps (every 10th).
// Accessions: fake_acc_0000 ... fake_acc_0999 (sorted ASCIIbetically = numeric order).
static std::string make_acc(int i) {
    char buf[32];
    snprintf(buf, sizeof(buf), "fake_acc_%04d", i);
    return buf;
}

struct MmapFile {
    void*  ptr  = MAP_FAILED;
    size_t size = 0;
    int    fd   = -1;

    explicit MmapFile(const std::filesystem::path& p) {
        fd = ::open(p.c_str(), O_RDONLY);
        REQUIRE(fd >= 0);
        struct stat st;
        fstat(fd, &st);
        size = static_cast<size_t>(st.st_size);
        ptr  = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        REQUIRE(ptr != MAP_FAILED);
    }
    ~MmapFile() {
        if (ptr != MAP_FAILED) ::munmap(ptr, size);
        if (fd  >= 0) ::close(fd);
    }
    template<typename T>
    const T* at(size_t off) const {
        return reinterpret_cast<const T*>(static_cast<const uint8_t*>(ptr) + off);
    }
    const uint8_t* base() const { return static_cast<const uint8_t*>(ptr); }
};

static std::vector<uint8_t> decompress(const uint8_t* data, size_t csz, size_t usz) {
    std::vector<uint8_t> out(usz);
    size_t r = ZSTD_decompress(out.data(), usz, data, csz);
    REQUIRE(!ZSTD_isError(r));
    REQUIRE(r == usz);
    return out;
}

} // anonymous namespace

TEST_CASE("derep_archive round-trip 1000 genomes", "[derep]") {
    const int N      = 1000;
    const int N_REPS = 100;
    const int DIM    = 256;

    // build a tmp path
    auto tmp = std::filesystem::temp_directory_path()
               / "geodesic_derep_test.gpd";
    std::filesystem::remove(tmp);

    // random embeddings (f16 = 2 bytes per elem)
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint16_t> dist;
    auto rand_emb = [&]() {
        std::vector<uint8_t> v(DIM * 2);
        for (size_t i = 0; i < v.size(); i += 2) {
            uint16_t x = dist(rng);
            std::memcpy(v.data() + i, &x, 2);
        }
        return v;
    };

    // pre-compute rep embeddings
    std::vector<std::vector<uint8_t>> rep_embeddings(N_REPS);
    for (int r = 0; r < N_REPS; ++r)
        rep_embeddings[r] = rand_emb();

    geodesic::DerepArchiveBuilderConfig cfg;
    cfg.output_path     = tmp;
    cfg.embedding_dim   = DIM;
    cfg.embedding_dtype = 1; // f16
    cfg.emit_armp       = true;
    cfg.zstd_level      = 3; // fast for tests

    geodesic::DerepArchiveBuilder ab(cfg);

    // single fake part
    geodesic::DerepArchiveBuilder::PartFingerprint fp{};
    fp.generation       = 1;
    fp.n_genomes_total  = N;
    fp.n_genomes_live   = N;
    fp.accession_set_hash = 0xDEADBEEF12345678ULL;
    ab.set_source_pack_manual({fp});

    ab.set_params({21}, 10000, 42, 43, 0.95f);

    // add genomes: rep = every 10th, rest are members assigned round-robin
    for (int i = 0; i < N; ++i) {
        int rep_idx = i / 10; // which rep owns this genome
        std::string acc     = make_acc(i);
        std::string rep_acc = make_acc(rep_idx * 10);
        uint64_t loc = (uint64_t{0} << 48) | static_cast<uint64_t>(i);

        if (i % 10 == 0) {
            // representative
            ab.add(acc, geodesic::DerepArchiveBuilder::Kind::Representative,
                   acc, loc, 21, 10,
                   rep_embeddings[rep_idx].data());
        } else {
            ab.add(acc, geodesic::DerepArchiveBuilder::Kind::Member,
                   rep_acc, loc, 21, 0, nullptr);
        }
    }

    ab.finalize();

    REQUIRE(std::filesystem::exists(tmp));
    REQUIRE(std::filesystem::file_size(tmp) > 64);

    MmapFile mm(tmp);

    // ── FileHeader ────────────────────────────────────────────────────────────
    const auto* fhdr = mm.at<GpdFileHeader>(0);
    CHECK(fhdr->magic        == FILE_MAGIC);
    CHECK(fhdr->format_major == 1);
    CHECK(fhdr->format_minor == 0);
    CHECK(fhdr->toc_offset   > 0);
    CHECK(fhdr->toc_offset   < mm.size);

    // ── TailLocator ───────────────────────────────────────────────────────────
    const auto* tail = mm.at<GpdTailLocator>(mm.size - sizeof(GpdTailLocator));
    CHECK(tail->magic      == TAIL_MAGIC);
    CHECK(tail->toc_offset == fhdr->toc_offset);

    // ── TOC ───────────────────────────────────────────────────────────────────
    const auto* tph = mm.at<TocHeader>(fhdr->toc_offset);
    CHECK(tph->magic == TOC_MAGIC);
    CHECK(tph->n_sections >= 6u);

    auto n_sec = tph->n_sections;
    const auto* descs = mm.at<GpdSectionDesc>(fhdr->toc_offset + sizeof(TocHeader));

    auto find_sec = [&](uint32_t type) -> const GpdSectionDesc* {
        for (uint32_t i = 0; i < n_sec; ++i)
            if (descs[i].type == type) return &descs[i];
        return nullptr;
    };

    // ── HDR section ───────────────────────────────────────────────────────────
    {
        auto* sd = find_sec(GPD_SEC_HDR);
        REQUIRE(sd != nullptr);
        CHECK((sd->flags & 1) == 0); // uncompressed
        const auto* h = mm.at<GpdHeader>(sd->file_offset);
        CHECK(h->magic         == HDR_MAGIC);
        CHECK(h->n_genomes     == static_cast<uint64_t>(N));
        CHECK(h->n_reps        == static_cast<uint64_t>(N_REPS));
        CHECK(h->n_unclustered == 0u);
        CHECK(h->embedding_dim == DIM);
        CHECK(h->embedding_dtype == 1u);
    }

    // ── ASTR + ASOF ───────────────────────────────────────────────────────────
    std::vector<std::string> sorted_accs(N);
    for (int i = 0; i < N; ++i) sorted_accs[i] = make_acc(i);
    std::sort(sorted_accs.begin(), sorted_accs.end());

    std::string expected_astr;
    for (auto& s : sorted_accs) expected_astr += s;

    {
        auto* sd = find_sec(GPD_SEC_AST);
        REQUIRE(sd != nullptr);
        CHECK((sd->flags & 1) == 1u); // compressed
        auto raw = decompress(mm.base() + sd->file_offset,
                              sd->compressed_size, sd->uncompressed_size);
        std::string got(raw.begin(), raw.end());
        CHECK(got == expected_astr);
    }

    {
        auto* sd = find_sec(GPD_SEC_ASO);
        REQUIRE(sd != nullptr);
        CHECK((sd->flags & 1) == 1u);
        auto raw = decompress(mm.base() + sd->file_offset,
                              sd->compressed_size, sd->uncompressed_size);
        const auto* ap = reinterpret_cast<const AsofPayload*>(raw.data());
        CHECK(ap->magic     == ASOF_MAGIC);
        CHECK(ap->n_genomes == static_cast<uint32_t>(N));
        const uint32_t* off = reinterpret_cast<const uint32_t*>(raw.data() + sizeof(AsofPayload));
        // monotonic
        for (int i = 0; i < N; ++i)
            CHECK(off[i] <= off[i+1]);
        CHECK(raw.size() == sizeof(AsofPayload) + (N + 1) * sizeof(uint32_t));
        // verify a sample accession
        size_t idx42 = static_cast<size_t>(
            std::find(sorted_accs.begin(), sorted_accs.end(), make_acc(42))
            - sorted_accs.begin());
        std::string s(expected_astr.data() + off[idx42],
                      off[idx42+1] - off[idx42]);
        CHECK(s == make_acc(42));
    }

    // ── RTBL ──────────────────────────────────────────────────────────────────
    {
        auto* sd = find_sec(GPD_SEC_RTB);
        REQUIRE(sd != nullptr);
        CHECK((sd->flags & 1) == 0u); // uncompressed
        const auto* rh = mm.at<RtblHeader>(sd->file_offset);
        CHECK(rh->magic  == RTBL_MAGIC);
        CHECK(rh->n_reps == static_cast<uint32_t>(N_REPS));
        const auto* entries = reinterpret_cast<const GpdRepEntry*>(
            mm.base() + sd->file_offset + sizeof(RtblHeader));
        // sorted by rep_acc_ord ascending
        for (int i = 1; i < N_REPS; ++i)
            CHECK(entries[i].rep_acc_ord > entries[i-1].rep_acc_ord);
        // all cluster_size == 10
        for (int i = 0; i < N_REPS; ++i)
            CHECK(entries[i].cluster_size == 10u);
    }

    // ── G2RM ──────────────────────────────────────────────────────────────────
    {
        auto* sd = find_sec(GPD_SEC_G2R);
        REQUIRE(sd != nullptr);
        CHECK((sd->flags & 1) == 1u);
        auto raw = decompress(mm.base() + sd->file_offset,
                              sd->compressed_size, sd->uncompressed_size);
        const auto* gh = reinterpret_cast<const G2rmHeader*>(raw.data());
        CHECK(gh->magic     == G2RM_MAGIC);
        CHECK(gh->n_genomes == static_cast<uint32_t>(N));
        const uint32_t* rep_id = reinterpret_cast<const uint32_t*>(
            raw.data() + sizeof(G2rmHeader));

        // The ordinal of fake_acc_0000 in sorted order:
        size_t ord0 = static_cast<size_t>(
            std::find(sorted_accs.begin(), sorted_accs.end(), make_acc(0))
            - sorted_accs.begin());
        // rep 0 (fake_acc_0000) should map to rep_id that corresponds to itself
        // (first rep when sorted by rep_acc_ord = ordinal of fake_acc_0000)
        CHECK(rep_id[ord0] < static_cast<uint32_t>(N_REPS));

        // fake_acc_0001 is member of fake_acc_0000
        size_t ord1 = static_cast<size_t>(
            std::find(sorted_accs.begin(), sorted_accs.end(), make_acc(1))
            - sorted_accs.begin());
        CHECK(rep_id[ord1] == rep_id[ord0]);
    }

    // ── EMBD ──────────────────────────────────────────────────────────────────
    {
        auto* sd = find_sec(GPD_SEC_EMB);
        REQUIRE(sd != nullptr);
        CHECK((sd->flags & 1) == 0u); // uncompressed
        const auto* eh = mm.at<EmbdHeader>(sd->file_offset);
        CHECK(eh->magic  == EMBD_MAGIC);
        CHECK(eh->dim    == static_cast<uint16_t>(DIM));
        CHECK(eh->dtype  == 1u);
        CHECK(eh->n_reps == static_cast<uint32_t>(N_REPS));
        size_t expected_bytes = sizeof(EmbdHeader)
                                + static_cast<size_t>(N_REPS) * DIM * 2;
        CHECK(sd->uncompressed_size == expected_bytes);
    }

    std::filesystem::remove(tmp);
}
