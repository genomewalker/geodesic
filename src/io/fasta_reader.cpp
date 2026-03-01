#include "fasta_reader.hpp"

#include "io/gz_reader.hpp"
#include <fstream>
#include <string>
#include <cstring>
#include <vector>

namespace derep {

bool is_gzip_file(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    unsigned char magic[2] = {0, 0};
    f.read(reinterpret_cast<char*>(magic), 2);
    return f.gcount() == 2 && magic[0] == 0x1f && magic[1] == 0x8b;
}

namespace {

constexpr bool is_valid_sequence_char(char c) {
    switch (c) {
        case 'A': case 'C': case 'G': case 'T': case 'N':
        case 'a': case 'c': case 'g': case 't': case 'n':
        case 'R': case 'Y': case 'S': case 'W': case 'K':
        case 'M': case 'B': case 'D': case 'H': case 'V':
        case 'r': case 'y': case 's': case 'w': case 'k':
        case 'm': case 'b': case 'd': case 'h': case 'v':
        case '-':
            return true;
        default:
            return false;
    }
}

constexpr bool is_whitespace(char c) {
    return c == '\n' || c == '\r' || c == ' ' || c == '\t';
}

std::uint64_t count_from_buffer(
    const char* buf, std::size_t len,
    bool& in_header, bool strict,
    const std::filesystem::path& path)
{
    std::uint64_t count = 0;
    for (std::size_t i = 0; i < len; ++i) {
        char c = buf[i];
        if (c == '>') {
            in_header = true;
            continue;
        }
        if (in_header) {
            if (c == '\n') in_header = false;
            continue;
        }
        if (is_whitespace(c)) continue;
        if (strict && !is_valid_sequence_char(c)) {
            throw FastaParseError(
                "Invalid sequence character '" + std::string(1, c) +
                "' in file: " + path.string());
        }
        ++count;
    }
    return count;
}

} // namespace

std::uint64_t calculate_genome_length(
    const std::filesystem::path& path,
    const FastaReadOptions& opts)
{
    if (!std::filesystem::exists(path)) {
        throw FastaParseError("File not found: " + path.string());
    }

    CompressionMode mode = opts.compression;
    if (mode == CompressionMode::kAuto) {
        mode = is_gzip_file(path) ? CompressionMode::kGzip : CompressionMode::kPlain;
    }

    std::uint64_t total = 0;
    bool in_header = false;

    if (mode == CompressionMode::kPlain) {
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            throw FastaParseError("Cannot open file: " + path.string());
        }
        std::vector<char> buf(opts.buffer_bytes);
        while (f) {
            f.read(buf.data(), static_cast<std::streamsize>(buf.size()));
            auto n = static_cast<std::size_t>(f.gcount());
            if (n == 0) break;
            total += count_from_buffer(buf.data(), n, in_header, opts.strict_ascii_sequence, path);
        }
    } else {
        GzReader gz(path.string());
        std::vector<char> buf(opts.buffer_bytes);
        for (;;) {
            size_t n = gz.read(buf.data(), buf.size());
            if (n == 0) break;
            total += count_from_buffer(buf.data(), n, in_header, opts.strict_ascii_sequence, path);
        }
    }

    return total;
}

} // namespace derep
