#include "core/geodesic/geodesic.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace derep {

float GeodesicDerep::find_diversity_threshold(
        const std::vector<float>& sorted_nn_dists,
        float ani_cap_angular) {
    const size_t n = sorted_nn_dists.size();
    if (n < 10) return ani_cap_angular;

    // Skip exact zeros
    size_t first_nz = 0;
    while (first_nz < n && sorted_nn_dists[first_nz] <= 1e-9f) ++first_nz;
    if (first_nz >= n) return 1e-6f;  // All identical

    float log_min = std::log(sorted_nn_dists[first_nz]);
    float log_max = std::log(sorted_nn_dists[n - 1]);

    auto fallback_mad = [&]() -> float {
        // Robust: median + 2 * left-half MAD
        float median = sorted_nn_dists[n / 2];
        std::vector<float> devs;
        devs.reserve(n / 2 + 1);
        for (size_t i = 0; i <= n / 2; ++i)
            devs.push_back(std::abs(sorted_nn_dists[i] - median));
        std::nth_element(devs.begin(), devs.begin() + devs.size() / 2, devs.end());
        float mad = devs[devs.size() / 2] * 1.4826f;
        return std::clamp(median + 2.0f * mad, 1e-6f, ani_cap_angular);
    };

    if (log_max - log_min < 0.01f) return fallback_mad();

    // Log-space histogram with 50 bins
    constexpr int NBINS = 50;
    const float bw = (log_max - log_min) / NBINS;
    std::vector<int> hist(NBINS, 0);
    for (size_t i = first_nz; i < n; ++i) {
        int b = std::min(NBINS - 1, static_cast<int>((std::log(sorted_nn_dists[i]) - log_min) / bw));
        ++hist[b];
    }

    // 3-bin moving average smoothing
    std::vector<float> sm(NBINS);
    for (int i = 0; i < NBINS; ++i) {
        int lo = std::max(0, i - 1), hi = std::min(NBINS - 1, i + 1);
        float s = 0; for (int j = lo; j <= hi; ++j) s += hist[j];
        sm[i] = s / static_cast<float>(hi - lo + 1);
    }

    // Find first local max (intra-strain peak)
    int peak1 = 0;
    for (int i = 1; i < NBINS - 1; ++i) {
        if (sm[i] >= sm[i-1] && sm[i] >= sm[i+1] && sm[i] > 0) { peak1 = i; break; }
    }

    // Find valley after peak1
    int valley = -1;
    float valley_val = std::numeric_limits<float>::max();
    for (int i = peak1 + 1; i < NBINS - 1; ++i) {
        if (sm[i] <= sm[i-1] && sm[i] <= sm[i+1]) {
            valley = i; valley_val = sm[i]; break;
        }
    }

    // Find second peak after valley
    int peak2 = -1;
    if (valley >= 0) {
        for (int i = valley + 1; i < NBINS; ++i) {
            if (sm[i] > 0 && (i == NBINS-1 || (sm[i] >= sm[i-1] && sm[i] >= sm[i+1]))) {
                peak2 = i; break;
            }
        }
    }

    // Validate bimodality: valley must be < 80% of smaller peak
    if (valley >= 0 && peak2 >= 0 &&
        valley_val < 0.8f * std::min(sm[peak1], sm[peak2])) {
        float thresh = std::exp(log_min + (static_cast<float>(valley) + 0.5f) * bw);
        return std::clamp(thresh, 1e-6f, ani_cap_angular);
    }

    return fallback_mad();
}

} // namespace derep
