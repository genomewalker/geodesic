#include "skani.hpp"
#include <xxhash.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string_view>
#include <vector>

namespace derep {

namespace {

// Encode nucleotide to 2-bit value; returns 255 for non-ACGT.
inline uint8_t encode_base(char c) {
    switch (c) {
    case 'A': case 'a': return 0;
    case 'C': case 'c': return 1;
    case 'G': case 'g': return 2;
    case 'T': case 't': return 3;
    default:            return 255;
    }
}

// Complement of a 2-bit encoded base.
inline uint8_t comp_base(uint8_t b) { return b ^ 3u; }

// Hash a k-mer given as a buffer of 2-bit encoded bases.
inline uint64_t hash_encoded(const uint8_t* buf, int len) {
    return XXH3_64bits(buf, static_cast<size_t>(len));
}

// Open syncmer: k-mer is selected if the minimum s-mer sub-hash occurs at position 0.
bool is_open_syncmer(const uint8_t* fwd, int k, int s) {
    // There are (k - s + 1) s-mers; we want argmin == 0.
    uint64_t h0 = hash_encoded(fwd, s);
    uint64_t min_h = h0;
    for (int pos = 1; pos <= k - s; ++pos) {
        uint64_t h = hash_encoded(fwd + pos, s);
        if (h < min_h) min_h = h;
    }
    return h0 == min_h;
}

} // anonymous namespace

SkaniSketch build_sketch(std::string_view accession, std::string_view fasta,
                          int k, int s, int sketch_size) {
    SkaniSketch sk;
    sk.accession = std::string(accession);

    // We collect all syncmer hashes, then take the smallest sketch_size.
    // For very large genomes this can be large; we use a bounded max-heap instead.
    std::vector<uint64_t> candidates;
    candidates.reserve(static_cast<size_t>(sketch_size) * 4);

    uint8_t kbuf[64] = {};  // rolling encoded k-mer buffer; k <= 63 assumed

    const char* data = fasta.data();
    const size_t len = fasta.size();

    size_t contig_start = 0;
    bool in_header = false;

    // Sliding window state
    size_t pos = 0;    // current position in fasta
    int    filled = 0; // number of valid bases in kbuf

    auto flush_window = [&]() { filled = 0; };

    auto process_base = [&](char c) {
        if (c == '\n' || c == '\r') return;
        uint8_t b = encode_base(c);
        // Shift window left by 1
        for (int i = 0; i < k - 1; ++i) kbuf[i] = kbuf[i + 1];
        kbuf[k - 1] = b;
        if (b == 255) {
            // N or bad base — reset window
            filled = 0;
            return;
        }
        if (filled < k) ++filled;
        if (filled < k) return;

        ++sk.genome_length;

        // Build RC and pick canonical orientation — shared k-mers on opposite strands
        // must select identically, so syncmer criterion must use the canonical form.
        uint8_t rc_buf[64];
        for (int i = 0; i < k; ++i) rc_buf[i] = comp_base(kbuf[k - 1 - i]);
        const uint64_t h_fwd = hash_encoded(kbuf, k);
        const uint64_t h_rc  = hash_encoded(rc_buf, k);
        const uint8_t* canon = (h_fwd <= h_rc) ? kbuf : rc_buf;

        if (!is_open_syncmer(canon, k, s)) return;

        candidates.push_back(std::min(h_fwd, h_rc));
    };

    (void)contig_start;
    for (size_t i = 0; i < len; ++i) {
        char c = data[i];
        if (c == '>') {
            in_header = true;
            flush_window();
            continue;
        }
        if (in_header) {
            if (c == '\n') in_header = false;
            continue;
        }
        process_base(c);
    }

    // Keep the sketch_size smallest hashes (sorted).
    if ((int)candidates.size() > sketch_size) {
        std::nth_element(candidates.begin(),
                         candidates.begin() + sketch_size,
                         candidates.end());
        candidates.resize(static_cast<size_t>(sketch_size));
    }
    std::sort(candidates.begin(), candidates.end());
    // Deduplicate (same canonical hash from fwd/rc of adjacent k-mers).
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());

    sk.hashes = std::move(candidates);
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
