#include "io/gz_reader.hpp"
#include <rapidgzip/ParallelGzipReader.hpp>

namespace derep {

struct GzReader::Impl {
    rapidgzip::ParallelGzipReader<> reader;
    Impl(const std::string& path, size_t parallelism)
        : reader(std::make_unique<rapidgzip::StandardFileReader>(path), parallelism) {}
};

GzReader::GzReader(const std::string& path, size_t parallelism)
    : impl_(std::make_unique<Impl>(path, parallelism)) {}

GzReader::~GzReader() = default;

size_t GzReader::read(char* buf, size_t n) {
    return impl_->reader.read(-1, buf, n);
}

} // namespace derep
