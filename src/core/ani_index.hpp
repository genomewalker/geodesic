#pragma once
#include "skani.hpp"
#include <parallel/algorithm>
#include <cstdint>
#include <vector>

namespace derep {

// A reference's marker hash (h % 8 == 0) in the flat ANI screen index.
struct AniMarker { uint64_t hash; int ri; };

// Every reference marker, sorted by (hash, ri), plus each reference's marker
// count in nm_ref. Counts and fills each reference's slice in parallel, then
// sorts in parallel; the result does not depend on `threads`.
inline std::vector<AniMarker> build_marker_index(const std::vector<const SkaniSketch*>& rsks,
                                                 std::vector<size_t>& nm_ref, int threads) {
    const int nr = static_cast<int>(rsks.size());
    nm_ref.assign(nr, 0);
    #pragma omp parallel for num_threads(threads) schedule(dynamic, 64)
    for (int ri = 0; ri < nr; ++ri)
        for (auto h : rsks[ri]->hashes) nm_ref[ri] += (h % 8 == 0);
    std::vector<size_t> offset(nr + 1, 0);
    for (int ri = 0; ri < nr; ++ri) offset[ri + 1] = offset[ri] + nm_ref[ri];
    std::vector<AniMarker> index(offset[nr]);
    #pragma omp parallel for num_threads(threads) schedule(dynamic, 64)
    for (int ri = 0; ri < nr; ++ri) {
        size_t at = offset[ri];
        for (auto h : rsks[ri]->hashes)
            if (h % 8 == 0) index[at++] = {h, ri};
    }
    __gnu_parallel::sort(index.begin(), index.end(), [](const AniMarker& a, const AniMarker& b) {
        return a.hash < b.hash || (a.hash == b.hash && a.ri < b.ri);
    }, __gnu_parallel::multiway_mergesort_tag(threads));
    return index;
}

} // namespace derep
