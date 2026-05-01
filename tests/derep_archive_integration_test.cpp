// Phase-3 cross-repo integration test: geodesic writer ↔ genopack reader.
//
// Drives geodesic::DerepArchiveBuilder against a real multipart .gpk archive
// and validates the produced .gpd through genopack::DerepView.

#include <catch2/catch_test_macros.hpp>

#include "derep/derep_archive.hpp"

#include <genopack/archive.hpp>
#include <genopack/archive_set_reader.hpp>
#include <genopack/derep_view.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <random>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace {

// f32 → f16 via half-precision IEEE 754 (round to nearest even). Sufficient
// for non-overflow/non-denormal values in [0, 65504]; keeps the test
// hermetic w/o pulling extra deps.
static uint16_t f32_to_f16(float fv) {
    uint32_t f;
    std::memcpy(&f, &fv, 4);
    uint32_t sign = (f >> 16) & 0x8000u;
    int32_t  exp  = static_cast<int32_t>((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = f & 0x7FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant |= 0x800000u;
        uint32_t shift = static_cast<uint32_t>(14 - exp);
        uint32_t halfm = mant >> shift;
        if ((mant >> (shift - 1)) & 1) ++halfm;
        return static_cast<uint16_t>(sign | halfm);
    }
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    uint32_t out = sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
    if (mant & 0x1000u) ++out;
    return static_cast<uint16_t>(out);
}

static std::filesystem::path pick_archive() {
    if (const char* env = std::getenv("GEODESIC_TEST_GPK")) {
        if (*env) return std::filesystem::path(env);
    }
    return std::filesystem::path(
        "/maps/projects/caeg/scratch/kbd606/tmp/ecoli1k_v4fix.gpk");
}

} // namespace

TEST_CASE("derep archive integration: writer ↔ reader on real .gpk", "[derep][integration]") {
    namespace fs = std::filesystem;

    fs::path src_path = pick_archive();
    if (!fs::exists(src_path)) {
        WARN("no source archive at " << src_path << " — set GEODESIC_TEST_GPK");
        return;
    }

    genopack::ArchiveSetReader src;
    src.open(src_path);
    REQUIRE(src.is_open());

    const size_t n_parts = src.part_count();
    INFO("source archive: " << src_path << " (parts=" << n_parts << ")");

    // ── Enumerate live accessions per part (with ordinal == genome_id local idx) ──
    std::vector<std::string> accessions;
    std::vector<uint64_t>    src_locators; // (part<<48) | local_id (advisory)
    {
        for (size_t p = 0; p < n_parts; ++p) {
            genopack::ArchiveReader rd;
            rd.open(src.part_paths()[p]);
            rd.scan_genome_accessions(
                [&](std::string_view acc, genopack::GenomeId id) {
                    accessions.emplace_back(acc);
                    src_locators.push_back(
                        (uint64_t{p} << 48) | static_cast<uint64_t>(id));
                });
        }
    }
    const uint64_t n_genomes = accessions.size();
    REQUIRE(n_genomes > 0);

    // Sort by accession (stable assignment of clustering deterministic).
    std::vector<size_t> order(n_genomes);
    for (size_t i = 0; i < n_genomes; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return accessions[a] < accessions[b];
    });

    // Pick reps: every 3rd in sorted order.
    std::vector<bool>     is_rep(n_genomes, false);
    std::vector<uint32_t> rep_ord_by_genome(n_genomes); // rep ord (sorted-order index)
    std::vector<uint32_t> rep_origs;                    // original idx of each rep
    for (size_t k = 0; k < n_genomes; k += 3) {
        is_rep[order[k]] = true;
        rep_origs.push_back(static_cast<uint32_t>(order[k]));
    }
    const uint32_t n_reps = static_cast<uint32_t>(rep_origs.size());
    REQUIRE(n_reps > 0);

    // Each non-rep maps to rep_origs[its_sorted_pos % n_reps].
    std::vector<uint32_t> assigned_rep_orig(n_genomes, UINT32_MAX);
    std::vector<uint32_t> cluster_size_by_rep(n_reps, 1u);
    for (size_t pos = 0; pos < n_genomes; ++pos) {
        size_t orig = order[pos];
        if (is_rep[orig]) {
            assigned_rep_orig[orig] = static_cast<uint32_t>(orig);
        } else {
            uint32_t r        = static_cast<uint32_t>(pos % n_reps);
            uint32_t rep_orig = rep_origs[r];
            assigned_rep_orig[orig] = rep_orig;
            ++cluster_size_by_rep[r];
        }
    }

    // ── drive the builder ───────────────────────────────────────────────────────
    fs::path tmp = fs::temp_directory_path() /
                   ("geodesic_derep_integration_" +
                    std::to_string(::getpid()) + ".gpd");
    fs::remove(tmp);

    geodesic::DerepArchiveBuilderConfig cfg;
    cfg.output_path      = tmp;
    cfg.embedding_dim    = 8;
    cfg.embedding_dtype  = 1; // f16
    cfg.emit_armp        = true;
    cfg.zstd_level       = 3;
    cfg.geodesic_version = "integration-test";

    geodesic::DerepArchiveBuilder ab(cfg);
    ab.set_source_pack(src);
    ab.set_params({21}, 10000, 42, 43, 0.95f);

    // Map original idx → rep_id (in rep_origs order, NOT sorted-order; the
    // writer re-sorts by rep_acc_ord internally, so we only need to know
    // *which* rep_orig we expect at query time).
    std::unordered_map<uint32_t, uint32_t> rep_orig_to_synth_id;
    rep_orig_to_synth_id.reserve(n_reps);
    for (uint32_t r = 0; r < n_reps; ++r)
        rep_orig_to_synth_id[rep_origs[r]] = r;

    // Build f16 embeddings keyed by synthetic rep_id, row[0] = float(rep_id),
    // remaining = 0.5.
    std::vector<std::array<uint16_t, 8>> rep_embeddings(n_reps);
    for (uint32_t r = 0; r < n_reps; ++r) {
        rep_embeddings[r][0] = f32_to_f16(static_cast<float>(r));
        for (int j = 1; j < 8; ++j)
            rep_embeddings[r][j] = f32_to_f16(0.5f);
    }

    // Add records.
    for (uint64_t i = 0; i < n_genomes; ++i) {
        if (is_rep[i]) {
            uint32_t r = rep_orig_to_synth_id[static_cast<uint32_t>(i)];
            ab.add(accessions[i],
                   geodesic::DerepArchiveBuilder::Kind::Representative,
                   accessions[i],
                   src_locators[i],
                   21,
                   cluster_size_by_rep[r],
                   rep_embeddings[r].data());
        } else {
            uint32_t rep_orig = assigned_rep_orig[i];
            ab.add(accessions[i],
                   geodesic::DerepArchiveBuilder::Kind::Member,
                   accessions[rep_orig],
                   src_locators[i],
                   21,
                   0,
                   nullptr);
        }
    }

    ab.finalize();
    REQUIRE(fs::exists(tmp));
    REQUIRE(fs::file_size(tmp) > 64);

    // ── open with reader ────────────────────────────────────────────────────────
    genopack::DerepView view;
    view.open(tmp);
    REQUIRE(view.is_open());

    // 1. staleness vs the same archive == Valid
    auto stale = view.check(src);
    INFO("staleness level = " << static_cast<int>(stale));
    CHECK(stale == genopack::DerepStaleness::Valid);

    // 2. stats
    auto st = view.stats();
    CHECK(st.n_genomes_indexed == n_genomes);
    CHECK(st.n_reps            == n_reps);
    CHECK(st.n_unclustered     == 0u);
    CHECK(view.embedding_dim() == 8u);
    CHECK(view.source_n_parts() == n_parts);

    // 3. status_for_accession spot-check up to 1000 accessions
    {
        std::mt19937 rng(2026);
        std::uniform_int_distribution<uint64_t> dist(0, n_genomes - 1);
        size_t n_check = std::min<uint64_t>(1000, n_genomes);
        size_t bad = 0;
        for (size_t k = 0; k < n_check; ++k) {
            uint64_t i = dist(rng);
            auto rs = view.status_for_accession(accessions[i]);
            if (is_rep[i]) {
                if (rs.kind != genopack::RepStatus::Kind::Representative) ++bad;
                if (rs.rep_accession != accessions[i])                    ++bad;
            } else {
                if (rs.kind != genopack::RepStatus::Kind::Member) ++bad;
                if (rs.rep_accession != accessions[assigned_rep_orig[i]]) ++bad;
            }
        }
        CHECK(bad == 0u);
    }

    // 4. scan_representatives — count + cluster_size sum
    {
        uint32_t seen = 0;
        uint64_t cs_sum = 0;
        view.scan_representatives(
            [&](uint32_t /*rid*/, std::string_view /*acc*/, uint32_t cs) {
                ++seen;
                cs_sum += cs;
            });
        CHECK(seen   == n_reps);
        CHECK(cs_sum == n_genomes - st.n_unclustered);
    }

    // 5. embedding_for_rep — first elem ≈ float(synthetic_rep_id) under f16 epsilon.
    //    The reader's rep_id ordering = sorted-by-rep_acc_ord, NOT our synth id.
    //    Resolve via status_for_accession of the rep's own accession.
    {
        std::mt19937 rng(7);
        std::uniform_int_distribution<uint32_t> dist(0, n_reps - 1);
        size_t n_check = std::min<uint32_t>(100, n_reps);
        size_t bad = 0;
        for (size_t k = 0; k < n_check; ++k) {
            uint32_t synth = dist(rng);
            uint32_t rep_orig = rep_origs[synth];
            auto rs = view.status_for_accession(accessions[rep_orig]);
            REQUIRE(rs.kind == genopack::RepStatus::Kind::Representative);
            std::array<float, 8> out{};
            bool ok = view.embedding_for_rep(rs.rep_id, out);
            REQUIRE(ok);
            float diff = out[0] - static_cast<float>(synth);
            if (diff < 0) diff = -diff;
            // f16 quant for integers ≤ 2048 is exact; epsilon 1e-2 is generous.
            if (diff > 1e-2f) {
                ++bad;
                WARN("synth=" << synth << " row[0]=" << out[0]);
            }
        }
        CHECK(bad == 0u);
    }

    // ── 6. CRC mutation: flip one byte mid-file → re-open should throw ─────────
    {
        view.close();
        fs::path mutated = tmp;
        mutated += ".bad";
        fs::copy_file(tmp, mutated, fs::copy_options::overwrite_existing);
        size_t sz = fs::file_size(mutated);
        REQUIRE(sz > 128);
        int fd = ::open(mutated.c_str(), O_RDWR);
        REQUIRE(fd >= 0);
        off_t off = static_cast<off_t>(sz / 2);
        uint8_t b = 0;
        ::pread(fd, &b, 1, off);
        b ^= 0xFFu;
        ::pwrite(fd, &b, 1, off);
        ::close(fd);

        genopack::DerepView v2;
        bool threw = false;
        try {
            v2.open(mutated);
        } catch (const std::exception&) {
            threw = true;
        }
        CHECK(threw);
        fs::remove(mutated);
    }

    fs::remove(tmp);
    src.close();
}
