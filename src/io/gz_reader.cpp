#include "io/gz_reader.hpp"
#include <rapidgzip/ParallelGzipReader.hpp>
#include <fcntl.h>
#include <unistd.h>
#include <fstream>
#ifdef HAVE_ZSTD
#include <zstd.h>
#endif

namespace derep {

struct GzReader::Impl {
    rapidgzip::ParallelGzipReader<> reader;
    Impl(const std::string& path, size_t parallelism)
        : reader(std::make_unique<rapidgzip::StandardFileReader>(path), parallelism) {}
};

GzReader::GzReader(const std::string& path, size_t parallelism)
    : impl_(std::make_unique<Impl>(path, parallelism)) {
    // Hint NFS client to prefetch file data into page cache.
    // Issued after rapidgzip opens the file so the kernel starts async read-ahead
    // while we prepare for the first read() call.
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd >= 0) {
        ::posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL | POSIX_FADV_WILLNEED);
        ::close(fd);
    }
}

GzReader::~GzReader() = default;

size_t GzReader::read(char* buf, size_t n) {
    return impl_->reader.read(-1, buf, n);
}

std::vector<char> GzReader::decompress_file(const std::string& path) {
    constexpr size_t CHUNK = 1 << 22;  // 4MB
    std::vector<char> result;
    result.reserve(6 * 1024 * 1024);  // 6MB — typical bacterial genome

    const bool is_gz  = path.size() >= 3 && path.compare(path.size() - 3, 3, ".gz")  == 0;
    const bool is_zst = path.size() >= 4 && path.compare(path.size() - 4, 4, ".zst") == 0;

    // Hint kernel (NFS client) to prefetch file into page cache before we open it.
    {
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd >= 0) {
            ::posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL | POSIX_FADV_WILLNEED);
            ::close(fd);
        }
    }

    if (is_gz) {
        GzReader gz(path);
        while (true) {
            const size_t off = result.size();
            result.resize(off + CHUNK);
            const size_t n = gz.read(result.data() + off, CHUNK);
            result.resize(off + n);
            if (n == 0) break;
        }
    } else if (is_zst) {
#ifdef HAVE_ZSTD
        // Read entire compressed file, then one-shot decompress.
        std::vector<char> compressed;
        {
            std::ifstream in(path, std::ios::binary | std::ios::ate);
            if (!in) return result;
            const auto sz = static_cast<std::streamoff>(in.tellg());
            if (sz <= 0) return result;
            compressed.resize(static_cast<size_t>(sz));
            in.seekg(0);
            in.read(compressed.data(), sz);
        }
        // Query decompressed size from zstd frame header.
        const unsigned long long decompressed_size =
            ZSTD_getFrameContentSize(compressed.data(), compressed.size());
        if (decompressed_size != ZSTD_CONTENTSIZE_UNKNOWN &&
            decompressed_size != ZSTD_CONTENTSIZE_ERROR) {
            result.resize(static_cast<size_t>(decompressed_size));
            const size_t r = ZSTD_decompress(result.data(), result.size(),
                                             compressed.data(), compressed.size());
            if (ZSTD_isError(r)) result.clear();
        } else {
            // Multi-frame or unknown size: streaming decompression.
            ZSTD_DStream* dstream = ZSTD_createDStream();
            ZSTD_DCtx_setParameter(dstream, ZSTD_d_windowLogMax, 27);  // 128 MB window max
            ZSTD_inBuffer  in_buf  = { compressed.data(), compressed.size(), 0 };
            std::vector<char> out_chunk(CHUNK);
            while (in_buf.pos < in_buf.size) {
                ZSTD_outBuffer out_buf = { out_chunk.data(), CHUNK, 0 };
                const size_t ret = ZSTD_decompressStream(dstream, &out_buf, &in_buf);
                if (ZSTD_isError(ret)) { result.clear(); break; }
                result.insert(result.end(), out_chunk.data(), out_chunk.data() + out_buf.pos);
            }
            ZSTD_freeDStream(dstream);
        }
#else
        // zstd not compiled in — return empty to signal unsupported format
        (void)is_zst;
#endif
    } else {
        std::ifstream in(path, std::ios::binary | std::ios::ate);
        if (!in) return result;
        const auto sz = static_cast<std::streamoff>(in.tellg());
        if (sz > 0) {
            in.seekg(0);
            result.resize(static_cast<size_t>(sz));
            in.read(result.data(), sz);
        }
    }

    return result;
}

} // namespace derep
