#include <catch2/catch_test_macros.hpp>
#include "core/ani_index.hpp"
#include <algorithm>
#include <memory>

namespace {
// The serial index run_ani built before the parallel builder.
std::vector<derep::AniMarker> serial_index(const std::vector<const derep::SkaniSketch*>& rsks,
                                           std::vector<size_t>& nm_ref) {
    std::vector<derep::AniMarker> index;
    nm_ref.assign(rsks.size(), 0);
    for (int ri = 0; ri < int(rsks.size()); ++ri)
        for (auto h : rsks[ri]->hashes)
            if (h % 8 == 0) { index.push_back({h, ri}); ++nm_ref[ri]; }
    std::sort(index.begin(), index.end(), [](const derep::AniMarker& a, const derep::AniMarker& b) {
        return a.hash < b.hash || (a.hash == b.hash && a.ri < b.ri);
    });
    return index;
}
}

TEST_CASE("Parallel ANI marker index equals the serial index at any thread count", "[ani-index]") {
    // 400 sketches over a 20,000-hash universe: markers are shared by many references.
    uint64_t state = 7;
    auto next = [&] { state = state * 6364136223846793005ULL + 1442695040888963407ULL; return state >> 20; };
    std::vector<derep::SkaniSketch> sketches(400);
    for (size_t i = 0; i < sketches.size(); ++i) {
        auto& h = sketches[i].hashes;
        for (size_t n = i % 37 == 0 ? 0 : 500 + next() % 4000; n--;) h.push_back(next() % 20000);
        std::sort(h.begin(), h.end());
        h.erase(std::unique(h.begin(), h.end()), h.end());
    }
    std::vector<const derep::SkaniSketch*> rsks;
    for (auto& s : sketches) rsks.push_back(&s);
    std::vector<size_t> want_nm;
    const auto want = serial_index(rsks, want_nm);
    REQUIRE(want.size() > 100000);
    for (int threads : {1, 2, 3, 8, 32}) {
        std::vector<size_t> nm;
        const auto got = derep::build_marker_index(rsks, nm, threads);
        INFO("threads " << threads);
        REQUIRE(nm == want_nm);
        REQUIRE(got.size() == want.size());
        REQUIRE(std::equal(got.begin(), got.end(), want.begin(),
            [](const derep::AniMarker& a, const derep::AniMarker& b) { return a.hash == b.hash && a.ri == b.ri; }));
    }
}
