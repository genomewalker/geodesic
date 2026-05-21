#include "skani.hpp"
#include <xxhash.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string_view>
#include <vector>

namespace derep {

namespace {

inline uint8_t encode_base(char c) {
    switch (c) {
    case 'A': case 'a': return 0;
    case 'C': case 'c': return 1;
    case 'G': case 'g': return 2;
    case 'T': case 't': return 3;
    default:            return 255;
    }
}

inline uint8_t comp_base(uint8_t b) { return b ^ 3u; }

inline uint64_t hash_buf(const uint8_t* buf, int len) {
    return XXH3_64bits(buf, static_cast<size_t>(len));
}

} // anonymous namespace

// Exact skani compression model: select canonical k-mer if hash % c == 0.
// Sketch size scales with genome_length / c, matching skani -c parameter.
SkaniSketch build_sketch(std::string_view accession, std::string_view fasta,
                          int k, int c) {
    SkaniSketch sk;
    sk.accession = std::string(accession);

    const uint64_t c64 = static_cast<uint64_t>(c);
    std::vector<uint64_t> hashes;
    hashes.reserve(fasta.size() / static_cast<size_t>(c) + 64);

    uint8_t kbuf[64] = {};
    int     filled   = 0;
    bool    in_header = false;

    auto process_base = [&](char ch) {
        if (ch == '\n' || ch == '\r') return;
        uint8_t b = encode_base(ch);
        for (int i = 0; i < k - 1; ++i) kbuf[i] = kbuf[i + 1];
        kbuf[k - 1] = b;
        if (b == 255) { filled = 0; return; }
        if (filled < k) ++filled;
        if (filled < k) return;

        ++sk.genome_length;

        uint8_t rc[64];
        for (int i = 0; i < k; ++i) rc[i] = comp_base(kbuf[k - 1 - i]);
        const uint64_t h_fwd = hash_buf(kbuf, k);
        const uint64_t h_rc  = hash_buf(rc,   k);
        const uint64_t h     = std::min(h_fwd, h_rc);

        if (h % c64 != 0) return;
        hashes.push_back(h);
    };

    const char* data = fasta.data();
    const size_t len = fasta.size();
    for (size_t i = 0; i < len; ++i) {
        char ch = data[i];
        if (ch == '>') { in_header = true; filled = 0; continue; }
        if (in_header)  { if (ch == '\n') in_header = false; continue; }
        process_base(ch);
    }

    std::sort(hashes.begin(), hashes.end());
    hashes.erase(std::unique(hashes.begin(), hashes.end()), hashes.end());
    sk.hashes = std::move(hashes);
    return sk;
}

SkaniResult compute_ani(const SkaniSketch& a, const SkaniSketch& b, int k) {
    SkaniResult res;
    if (a.hashes.empty() || b.hashes.empty()) return res;

    // Count intersection size via sorted-merge.
    size_t isect = 0;
    {
        auto it_a = a.hashes.begin(), end_a = a.hashes.end();
        auto it_b = b.hashes.begin(), end_b = b.hashes.end();
        while (it_a != end_a && it_b != end_b) {
            if (*it_a == *it_b) { ++isect; ++it_a; ++it_b; }
            else if (*it_a < *it_b) ++it_a;
            else ++it_b;
        }
    }

    res.c_ab = double(isect) / double(a.hashes.size());
    res.c_ba = double(isect) / double(b.hashes.size());
    res.af   = std::max(res.c_ab, res.c_ba);

    double max_c = res.af;
    if (max_c <= 0.0) { res.ani = 0.0; return res; }

    res.ani = std::pow(max_c, 1.0 / k) * 100.0;
    return res;
}

} // namespace derep
