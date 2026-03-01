#pragma once
#include <cstddef>
#include <memory>
#include <string>

namespace derep {

// PIMPL wrapper: rapidgzip headers compiled only once (in gz_reader.cpp).
// Drop-in replacement for gzopen/gzread/gzclose.
class GzReader {
public:
    explicit GzReader(const std::string& path, size_t parallelism = 1);
    ~GzReader();
    // Returns bytes written to buf; 0 at EOF.
    size_t read(char* buf, size_t n);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace derep
