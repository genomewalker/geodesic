#include "core/kmer_probe.hpp"
#include "core/pack_reader.hpp"
#include <genopack/archive.hpp>
#include <spdlog/spdlog.h>
#include <algorithm>

namespace derep {

static int select_best_k_for_diversity(float p95_nn_dist) {
    if (p95_nn_dist < 0.002f) return 31;
    if (p95_nn_dist < 0.010f) return 21;
    return 16;
}

int probe_taxon_kmer(const std::vector<std::string>& accessions,
                     IPackReader& gpk,
                     int /*default_k*/,
                     int sketch_size)
{
    static constexpr size_t kProbeN    = 300;
    static constexpr size_t kProbeBins = 500;

    const size_t n = accessions.size();
    if (n < 20) return 0;

    const auto avail_ks = gpk.available_kmer_sizes();
    if (avail_ks.size() < 2) return 0;

    const size_t probe_n  = std::min(kProbeN, n / 5);
    const size_t stride   = n / probe_n;
    // Use the smallest available k for probing — it's the preloaded/cached k (avail_ks
    // is sorted ascending). Probing with the largest k reads from NFS on every taxon
    // when only the smallest k is preloaded, adding ~5s of NFS latency per taxon.
    // Clonal vs. diverse signal is direction-identical across k values.
    const uint32_t probe_k  = avail_ks.front();
    const uint32_t probe_sz = static_cast<uint32_t>(
        std::min(kProbeBins, static_cast<size_t>(sketch_size)));

    std::vector<std::string> probe_accs;
    probe_accs.reserve(probe_n);
    for (size_t i = 0; i < probe_n; ++i)
        probe_accs.push_back(accessions[i * stride]);

    std::vector<std::vector<uint32_t>> sigs_slots(probe_n);
    std::vector<uint8_t>               sig_valid(probe_n, 0);
    gpk.visit_sketch_batches(probe_accs, probe_k, probe_sz,
        [&](size_t i, const genopack::SketchResult& sk) {
            sigs_slots[i].assign(sk.sig, sk.sig + sk.sketch_size);
            sig_valid[i] = 1;
        });
    std::vector<std::vector<uint32_t>> sigs;
    sigs.reserve(probe_n);
    for (size_t i = 0; i < probe_n; ++i)
        if (sig_valid[i]) sigs.push_back(std::move(sigs_slots[i]));
    if (sigs.size() < 10) return 0;

    const size_t m = sigs.size();
    std::vector<float> nn(m, 1.0f);
    for (size_t i = 0; i < m; ++i) {
        const size_t bins = sigs[i].size();
        for (size_t j = 0; j < m; ++j) {
            if (i == j) continue;
            size_t matches = 0;
            for (size_t b = 0; b < bins; ++b)
                matches += (sigs[i][b] == sigs[j][b]);
            float d = 1.0f - float(matches) / float(bins);
            if (d < nn[i]) nn[i] = d;
        }
    }
    std::sort(nn.begin(), nn.end());
    const float p5  = nn[static_cast<size_t>(m * 0.05)];
    const float p95 = nn[static_cast<size_t>(m * 0.95)];

    int best_k = select_best_k_for_diversity(p95);

    spdlog::info("GEODESIC: k pre-probe (n={} sample, bins={}): p5_nn={:.4f} p95_nn={:.4f} → k={}",
                 m, probe_sz, p5, p95, best_k);
    return best_k;
}

} // namespace derep
