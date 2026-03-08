#pragma once
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace derep {

// PIMPL wrapper: rapidgzip headers compiled only once (in gz_reader.cpp).
// Drop-in replacement for gzopen/gzread/gzclose.
class GzReader {
public:
    explicit GzReader(const std::string& path, size_t parallelism = 1);
    ~GzReader();
    // Returns bytes written to buf; 0 at EOF.
    size_t read(char* buf, size_t n);

    // Decompress entire file (gz, zst, or plain) into memory.
    // Issues posix_fadvise(WILLNEED) before reading for NFS prefetch.
    // Supports: .gz (rapidgzip), .zst (libzstd if HAVE_ZSTD), plain (mmap-style ifstream).
    static std::vector<char> decompress_file(const std::string& path);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace derep
