#include "skani.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string_view>
#include <vector>

namespace derep {

namespace {

// ntHash seed table (Mohamadi et al. 2016, Bioinformatics).
// Indexed by 2-bit encoding: A=0, C=1, G=2, T=3.
static constexpr uint64_t SEED_FWD[4] = {
    0x3c8bfbb395c60474ULL,  // A
    0x3193c18562a02b4cULL,  // C
    0x20323ed082572324ULL,  // G
    0x295549f54be24456ULL,  // T
};
// Complement seeds: SEED_RC[b] = SEED_FWD[complement(b)]
static constexpr uint64_t SEED_RC[4] = {
    SEED_FWD[3], SEED_FWD[2], SEED_FWD[1], SEED_FWD[0],
};

inline uint8_t encode_base(char c) {
    switch (c) {
    case 'A': case 'a': return 0;
    case 'C': case 'c': return 1;
    case 'G': case 'g': return 2;
    case 'T': case 't': return 3;
    default:             return 255;
    }
}

inline uint64_t rol64(uint64_t v, int n) { return (v << n) | (v >> (64 - n)); }
inline uint64_t ror64(uint64_t v, int n) { return (v >> n) | (v << (64 - n)); }

} // anonymous namespace

namespace {

// Lemire fast divisibility: h % c == 0  iff  h * c_inv <= limit
// Requires c to be odd. Returns {c_inv, limit} pair.
constexpr uint64_t lemire_inv(uint64_t c) {
    uint64_t x = 1;
    for (int i = 0; i < 6; ++i) x *= 2 - c * x;  // Newton, 6 steps → 64 bits
    return x;
}

// Branchless base encode: A=0 C=1 G=2 T=3, everything else=255.
alignas(64) static constexpr uint8_t BASE_ENC[256] = {
    #define B(c) ((c)=='A'||(c)=='a'?0:(c)=='C'||(c)=='c'?1:(c)=='G'||(c)=='g'?2:(c)=='T'||(c)=='t'?3:255)
    B(0),B(1),B(2),B(3),B(4),B(5),B(6),B(7),B(8),B(9),B(10),B(11),B(12),B(13),B(14),B(15),
    B(16),B(17),B(18),B(19),B(20),B(21),B(22),B(23),B(24),B(25),B(26),B(27),B(28),B(29),B(30),B(31),
    B(32),B(33),B(34),B(35),B(36),B(37),B(38),B(39),B(40),B(41),B(42),B(43),B(44),B(45),B(46),B(47),
    B(48),B(49),B(50),B(51),B(52),B(53),B(54),B(55),B(56),B(57),B(58),B(59),B(60),B(61),B(62),B(63),
    B(64),B(65),B(66),B(67),B(68),B(69),B(70),B(71),B(72),B(73),B(74),B(75),B(76),B(77),B(78),B(79),
    B(80),B(81),B(82),B(83),B(84),B(85),B(86),B(87),B(88),B(89),B(90),B(91),B(92),B(93),B(94),B(95),
    B(96),B(97),B(98),B(99),B(100),B(101),B(102),B(103),B(104),B(105),B(106),B(107),B(108),B(109),B(110),B(111),
    B(112),B(113),B(114),B(115),B(116),B(117),B(118),B(119),B(120),B(121),B(122),B(123),B(124),B(125),B(126),B(127),
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    #undef B
};

} // inner anonymous namespace

// ntHash canonical rolling hash sketch.
// Roll (drop prev, add next):
//   hf = rol1(hf) ^ tab_fk[prev] ^ SEED_FWD[next]    (tab_fk = rol(SEED_FWD, k))
//   hr = ror1(hr) ^ tab_r1[prev] ^ tab_rk1[next]      (tab_r1 = ror(SEED_RC,1), tab_rk1 = rol(SEED_RC,k-1))
// Canonical = min(hf,hr). Select if canonical % c == 0.
SkaniSketch build_sketch(std::string_view accession, std::string_view fasta,
                          int k, int c) {
    SkaniSketch sk;
    sk.accession = std::string(accession);

    // Precompute per-base rotation tables for this k — avoids runtime rol(x,k) in hot loop.
    uint64_t tab_fk[4], tab_r1[4], tab_rk1[4];
    for (int b = 0; b < 4; ++b) {
        tab_fk[b]  = rol64(SEED_FWD[b], k);
        tab_r1[b]  = ror64(SEED_RC[b],  1);
        tab_rk1[b] = rol64(SEED_RC[b],  k - 1);
    }

    // Lemire divisibility: h % c == 0 iff h * c_inv <= limit.
    // For even c: strip factors of 2 first (cheap bit-test).
    int    shift = 0;
    uint64_t c_odd = static_cast<uint64_t>(c);
    while ((c_odd & 1) == 0) { c_odd >>= 1; ++shift; }
    const uint64_t mask2  = (1ULL << shift) - 1;            // low bits that must be 0
    const uint64_t c_inv  = lemire_inv(c_odd);
    const uint64_t limit  = UINT64_MAX / c_odd;

    std::vector<uint64_t> hashes;
    hashes.reserve(fasta.size() / static_cast<size_t>(c) + 64);

    uint64_t hf = 0, hr = 0;
    int      filled    = 0;
    bool     in_header = false;

    alignas(64) uint8_t ring[64] = {};  // ring buffer of last k 2-bit bases (k ≤ 63)
    int ring_pos = 0;

    const char*  data = fasta.data();
    const size_t len  = fasta.size();

    for (size_t i = 0; i < len; ++i) {
        const uint8_t ch = static_cast<uint8_t>(data[i]);

        // Header/newline/CR: rare in hot loop — handled with predictable branches.
        if (__builtin_expect(ch == '>', 0)) { in_header = true; hf = hr = 0; filled = 0; ring_pos = 0; continue; }
        if (__builtin_expect(in_header,  0)) { if (ch == '\n') in_header = false; continue; }
        if (__builtin_expect(ch <= '\r', 0)) continue;  // '\n'=10, '\r'=13, skip both

        const uint8_t b = BASE_ENC[ch];
        if (__builtin_expect(b == 255, 0)) { hf = hr = 0; filled = 0; ring_pos = 0; continue; }

        if (__builtin_expect(filled < k, 0)) {
            // Init: accumulate first k-mer (entered only k times per contig).
            hf ^= rol64(SEED_FWD[b], k - 1 - filled);
            hr ^= rol64(SEED_RC[b],  filled);
            ring[filled] = b;
            if (++filled < k) continue;
            // ring_pos stays 0 after init.
        } else {
            // Hot path: O(1) rolling update using precomputed tables.
            const uint8_t prev = ring[ring_pos];
            ring[ring_pos] = b;
            if (++ring_pos >= k) ring_pos = 0;

            hf = rol64(hf, 1) ^ tab_fk[prev]  ^ SEED_FWD[b];
            hr = ror64(hr, 1) ^ tab_r1[prev]   ^ tab_rk1[b];
        }

        ++sk.genome_length;
        const uint64_t h = std::min(hf, hr);
        // Lemire divisibility check (no integer division).
        if ((h & mask2) == 0 && h * c_inv <= limit) hashes.push_back(h);
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
