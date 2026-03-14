#include "db/genome_pack.hpp"
#include <duckdb.hpp>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <cstring>
#include <fstream>

#ifdef HAVE_ZSTD
#include <zstd.h>
#endif

namespace derep::db {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static void check(duckdb::unique_ptr<duckdb::MaterializedQueryResult>& res, const char* ctx) {
    if (res->HasError())
        throw std::runtime_error(std::string(ctx) + ": " + res->GetError());
}

// FNV-1a 64-bit hash
static std::string fnv1a_hex(const std::string& s) {
    uint64_t h = 14695981039346656037ULL;
    for (unsigned char c : s) {
        h ^= static_cast<uint64_t>(c);
        h *= 1099511628211ULL;
    }
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(h));
    return buf;
}

// ---------------------------------------------------------------------------
// GenomePack
// ---------------------------------------------------------------------------
GenomePack::GenomePack(Config cfg) : cfg_(cfg) {}

GenomePack::~GenomePack() {
    close();
}

std::string GenomePack::taxon_hash(const std::string& taxonomy) {
    return fnv1a_hex(taxonomy);
}

std::filesystem::path GenomePack::taxon_file_path(const std::string& taxonomy) const {
    auto hash = taxon_hash(taxonomy);
    return pack_dir_ / "data" / hash.substr(0, 2) / (hash + ".gpack");
}

duckdb::Connection& GenomePack::thread_connection() const {
    auto tid = std::this_thread::get_id();
    std::lock_guard lock(pool_mutex_);
    auto [it, inserted] = pool_.emplace(tid, nullptr);
    if (inserted)
        it->second = std::make_unique<duckdb::Connection>(*database_);
    return *it->second;
}

void GenomePack::create_schema() {
    auto conn = std::make_unique<duckdb::Connection>(*database_);

    auto r0 = conn->Query(R"(
        CREATE TABLE IF NOT EXISTS pack_meta (
            key   VARCHAR PRIMARY KEY,
            value VARCHAR NOT NULL
        )
    )");
    check(r0, "create pack_meta");

    auto r1 = conn->Query(R"(
        CREATE TABLE IF NOT EXISTS pack_taxa (
            taxonomy  VARCHAR PRIMARY KEY,
            file_path VARCHAR NOT NULL,
            n_genomes UINTEGER NOT NULL
        )
    )");
    check(r1, "create pack_taxa");
}

void GenomePack::open_write(const std::filesystem::path& pack_dir) {
    pack_dir_ = pack_dir;
    std::filesystem::create_directories(pack_dir_ / "data");
    duckdb::DBConfig dcfg;
    dcfg.options.maximum_threads = 2;
    database_ = std::make_unique<duckdb::DuckDB>((pack_dir_ / "pack.db").string(), &dcfg);
    create_schema();
    spdlog::info("GenomePack opened for writing: {}", pack_dir_.string());
}

void GenomePack::open_read(const std::filesystem::path& pack_dir) {
    pack_dir_ = pack_dir;
    auto db_path = pack_dir_ / "pack.db";
    if (!std::filesystem::exists(db_path))
        throw std::runtime_error("GenomePack: pack.db not found at " + pack_dir_.string());
    duckdb::DBConfig dcfg;
    dcfg.options.maximum_threads = 2;
    database_ = std::make_unique<duckdb::DuckDB>(db_path.string(), &dcfg);
    create_schema();  // IF NOT EXISTS is safe
    spdlog::info("GenomePack opened for reading: {}", pack_dir_.string());
}

void GenomePack::close() {
    if (!database_) return;
    {
        std::lock_guard lock(pool_mutex_);
        pool_.clear();
    }
    database_.reset();
}

void GenomePack::write_taxon(
    const std::string& taxonomy,
    const std::vector<std::pair<std::string, std::vector<char>>>& genome_bufs)
{
#ifndef HAVE_ZSTD
    throw std::runtime_error("GenomePack::write_taxon: zstd not compiled in (rebuild with HAVE_ZSTD)");
#else
    if (genome_bufs.empty()) return;

    // 1. Compute offsets
    std::vector<uint64_t> offsets(genome_bufs.size());
    uint64_t total_payload = 0;
    for (size_t i = 0; i < genome_bufs.size(); ++i) {
        offsets[i] = total_payload;
        total_payload += genome_bufs[i].second.size();
    }

    // 2. Build header only (small — fits in memory regardless of taxon size)
    // Header: magic[5], version[1], n_genomes[4],
    //         for each genome: acc_len[2], accession[acc_len], offset[8], length[8]
    size_t header_size = 5 + 1 + 4;
    for (const auto& [acc, buf] : genome_bufs)
        header_size += 2 + acc.size() + 8 + 8;

    std::vector<char> header(header_size);
    char* p = header.data();

    std::memcpy(p, "GPACK", 5); p += 5;
    *reinterpret_cast<uint8_t*>(p) = 1; p += 1;
    uint32_t n = static_cast<uint32_t>(genome_bufs.size());
    std::memcpy(p, &n, 4); p += 4;
    for (size_t i = 0; i < genome_bufs.size(); ++i) {
        const auto& [acc, buf] = genome_bufs[i];
        uint16_t acc_len = static_cast<uint16_t>(acc.size());
        std::memcpy(p, &acc_len, 2); p += 2;
        std::memcpy(p, acc.data(), acc.size()); p += acc.size();
        std::memcpy(p, &offsets[i], 8); p += 8;
        uint64_t blen = buf.size();
        std::memcpy(p, &blen, 8); p += 8;
    }

    // 3. Stream-compress header + genome FASTA data directly into output file.
    // Avoids double-buffering (no full raw + full compressed allocation).
    // Large windowLog (27 = 128 MB) allows cross-genome compression for related species.

    // 4. Write file (under write_mutex_ to serialise)
    std::lock_guard lock(write_mutex_);

    auto fpath = taxon_file_path(taxonomy);
    std::filesystem::create_directories(fpath.parent_path());
    {
        std::ofstream ofs(fpath, std::ios::binary);
        if (!ofs)
            throw std::runtime_error("GenomePack: cannot write " + fpath.string());

        ZSTD_CCtx* cctx = ZSTD_createCCtx();
        ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, cfg_.zstd_level);
        ZSTD_CCtx_setParameter(cctx, ZSTD_c_windowLog, 27);  // 128 MB window for cross-genome refs
        ZSTD_CCtx_setParameter(cctx, ZSTD_c_enableLongDistanceMatching, 1);
        // Pledge total uncompressed size so zstd writes it into the frame header.
        // Required for ZSTD_getFrameContentSize() to work in fetch_taxon().
        ZSTD_CCtx_setPledgedSrcSize(cctx, static_cast<unsigned long long>(header_size + total_payload));

        const size_t out_chunk = ZSTD_CStreamOutSize();
        std::vector<char> out_buf(out_chunk);

        auto feed = [&](const char* data, size_t len, bool last) {
            ZSTD_inBuffer in = {data, len, 0};
            ZSTD_EndDirective end_op = last ? ZSTD_e_end : ZSTD_e_continue;
            size_t ret;
            do {
                ZSTD_outBuffer out = {out_buf.data(), out_chunk, 0};
                ret = ZSTD_compressStream2(cctx, &out, &in, end_op);
                if (ZSTD_isError(ret)) {
                    ZSTD_freeCCtx(cctx);
                    throw std::runtime_error(std::string("zstd compress error: ")
                                             + ZSTD_getErrorName(ret));
                }
                ofs.write(out_buf.data(), static_cast<std::streamsize>(out.pos));
            } while ((end_op == ZSTD_e_end && ret != 0) || in.pos < in.size);
        };

        // Compress header, then each genome's FASTA in sequence.
        // The 128 MB window captures cross-genome k-mer matches across ~25 typical genomes.
        feed(header.data(), header.size(), genome_bufs.empty());
        for (size_t i = 0; i < genome_bufs.size(); ++i)
            feed(genome_bufs[i].second.data(), genome_bufs[i].second.size(),
                 i + 1 == genome_bufs.size());

        ZSTD_freeCCtx(cctx);
        if (!ofs)
            throw std::runtime_error("GenomePack: write failed for " + fpath.string());
    }

    // 5. Insert into pack_taxa
    {
        auto& conn = thread_connection();
        auto stmt = conn.Prepare(
            "INSERT OR REPLACE INTO pack_taxa (taxonomy, file_path, n_genomes) VALUES ($1, $2, $3)");
        stmt->Execute(taxonomy, fpath.string(), static_cast<uint32_t>(genome_bufs.size()));
    }

    // 6. Checkpoint cadence
    if (++packs_since_checkpoint_ >= cfg_.taxa_per_checkpoint) {
        checkpoint();
        packs_since_checkpoint_ = 0;
    }
#endif
}

bool GenomePack::has_taxon(const std::string& taxonomy) const {
    auto& conn = thread_connection();
    auto stmt = conn.Prepare("SELECT 1 FROM pack_taxa WHERE taxonomy = $1 LIMIT 1");
    auto res = stmt->Execute(taxonomy);
    if (res->HasError()) return false;
    auto chunk = res->Fetch();
    return chunk && chunk->size() > 0;
}

GenomePack::TaxonData GenomePack::fetch_taxon(const std::string& taxonomy) const {
#ifndef HAVE_ZSTD
    throw std::runtime_error("GenomePack::fetch_taxon: zstd not compiled in");
#else
    // 1. Query file_path
    auto& conn = thread_connection();
    auto stmt = conn.Prepare("SELECT file_path FROM pack_taxa WHERE taxonomy = $1 LIMIT 1");
    auto res = stmt->Execute(taxonomy);
    if (res->HasError())
        throw std::runtime_error("GenomePack::fetch_taxon query error: " + res->GetError());
    auto chunk = res->Fetch();
    if (!chunk || chunk->size() == 0)
        throw std::runtime_error("GenomePack::fetch_taxon: taxonomy not found: " + taxonomy);
    std::string fpath_str = chunk->GetValue(0, 0).GetValue<std::string>();

    // 2. Read compressed file
    std::ifstream ifs(fpath_str, std::ios::binary | std::ios::ate);
    if (!ifs)
        throw std::runtime_error("GenomePack::fetch_taxon: cannot open " + fpath_str);
    auto file_size = static_cast<size_t>(ifs.tellg());
    ifs.seekg(0);
    std::vector<char> compressed(file_size);
    ifs.read(compressed.data(), static_cast<std::streamsize>(file_size));
    if (!ifs)
        throw std::runtime_error("GenomePack::fetch_taxon: read error " + fpath_str);

    // 3. zstd decompress
    unsigned long long decompressed_size = ZSTD_getFrameContentSize(compressed.data(), compressed.size());
    if (decompressed_size == ZSTD_CONTENTSIZE_ERROR || decompressed_size == ZSTD_CONTENTSIZE_UNKNOWN)
        throw std::runtime_error("GenomePack::fetch_taxon: cannot determine decompressed size for " + fpath_str);
    std::vector<char> raw(static_cast<size_t>(decompressed_size));
    size_t actual = ZSTD_decompress(raw.data(), raw.size(), compressed.data(), compressed.size());
    if (ZSTD_isError(actual))
        throw std::runtime_error(std::string("GenomePack::fetch_taxon zstd error: ") + ZSTD_getErrorName(actual));

    // 4. Parse header
    const char* p = raw.data();
    const char* end = raw.data() + raw.size();

    auto require = [&](size_t n, const char* what) {
        if (p + n > end)
            throw std::runtime_error(std::string("GenomePack: truncated header reading ") + what);
    };

    require(5, "magic");
    if (std::memcmp(p, "GPACK", 5) != 0)
        throw std::runtime_error("GenomePack: bad magic in " + fpath_str);
    p += 5;

    require(1, "version");
    uint8_t version = *reinterpret_cast<const uint8_t*>(p); p += 1;
    if (version != 1)
        throw std::runtime_error("GenomePack: unsupported version " + std::to_string(version));

    require(4, "n_genomes");
    uint32_t n_genomes;
    std::memcpy(&n_genomes, p, 4); p += 4;

    TaxonData td;
    td.genomes.resize(n_genomes);

    for (uint32_t i = 0; i < n_genomes; ++i) {
        require(2, "acc_len");
        uint16_t acc_len;
        std::memcpy(&acc_len, p, 2); p += 2;

        require(acc_len, "accession");
        td.genomes[i].accession.assign(p, acc_len); p += acc_len;

        require(8, "offset");
        std::memcpy(&td.genomes[i].offset, p, 8); p += 8;

        require(8, "length");
        std::memcpy(&td.genomes[i].length, p, 8); p += 8;
    }

    // 5. Payload is everything after the header
    size_t header_consumed = static_cast<size_t>(p - raw.data());
    size_t payload_size = raw.size() - header_consumed;
    td.payload.resize(payload_size);
    std::memcpy(td.payload.data(), p, payload_size);

    return td;
#endif
}

std::unordered_set<std::string> GenomePack::load_completed_taxa() const {
    auto conn = std::make_unique<duckdb::Connection>(*database_);
    auto res = conn->Query("SELECT taxonomy FROM pack_taxa");
    if (res->HasError())
        throw std::runtime_error("GenomePack::load_completed_taxa: " + res->GetError());
    std::unordered_set<std::string> out;
    for (;;) {
        auto chunk = res->Fetch();
        if (!chunk || chunk->size() == 0) break;
        for (size_t row = 0; row < chunk->size(); ++row)
            out.insert(chunk->GetValue(0, row).GetValue<std::string>());
    }
    return out;
}

void GenomePack::checkpoint() {
    auto conn = std::make_unique<duckdb::Connection>(*database_);
    auto r = conn->Query("CHECKPOINT");
    if (r->HasError())
        spdlog::warn("GenomePack checkpoint failed: {}", r->GetError());
}

} // namespace derep::db
