#include "preloaded_pack_reader.hpp"
#include "logging.hpp"
#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace derep {

namespace {
inline void trim_heap() {
#if defined(__GLIBC__)
    malloc_trim(0);
#endif
}
}

void PreloadedPackReader::clear_store_() {
    k_stores_.clear();
    acc_to_idx_.clear();
    trim_heap();
}

size_t PreloadedPackReader::bytes() const noexcept {
    size_t b = 0;
    for (const auto& [k, st] : k_stores_) {
        b += st.sigs.size()  * sizeof(uint16_t);
        b += st.sig2s.size() * sizeof(uint16_t);
        b += st.masks.size() * sizeof(uint64_t);
        b += st.n_real_bins.size()    * sizeof(uint32_t);
        b += st.genome_lengths.size() * sizeof(uint64_t);
    }
    return b;
}

void PreloadedPackReader::populate_store_(
    KStore& st, const std::vector<std::string>& accessions, int /*n_threads*/)
{
    const size_t n = accessions.size();
    const uint32_t sz = st.sz;
    const uint32_t mw = st.mask_words;
    st.sigs.assign(static_cast<size_t>(n) * sz, 0);
    st.sig2s.assign(static_cast<size_t>(n) * sz, 0);
    st.masks.assign(static_cast<size_t>(n) * mw, 0);
    st.n_real_bins.assign(n, 0);
    st.genome_lengths.assign(n, 0);

    std::atomic<size_t> hits{0};
    std::mutex map_mu;

    inner_->visit_sketch_batches(accessions, st.k, sz,
        [&](size_t i, const genopack::SketchResult& sk) {
            if (sk.sketch_size != sz || sk.mask_words != mw) return;
            std::memcpy(st.sigs.data()  + static_cast<size_t>(i) * sz,
                        sk.sig,  sz * sizeof(uint16_t));
            std::memcpy(st.sig2s.data() + static_cast<size_t>(i) * sz,
                        sk.sig2, sz * sizeof(uint16_t));
            std::memcpy(st.masks.data() + static_cast<size_t>(i) * mw,
                        sk.mask, mw * sizeof(uint64_t));
            st.n_real_bins[i]    = sk.n_real_bins;
            st.genome_lengths[i] = sk.genome_length;
            {
                std::lock_guard<std::mutex> lk(map_mu);
                acc_to_idx_.emplace(accessions[i], static_cast<uint32_t>(i));
            }
            hits.fetch_add(1, std::memory_order_relaxed);
        });
    spdlog::info("PRELOAD: k={} sz={} hits={}/{}",
                 st.k, sz, hits.load(std::memory_order_relaxed), n);
}

std::pair<size_t, size_t> PreloadedPackReader::preload(
    const std::vector<std::string>& accessions,
    uint32_t k, uint32_t sz, int n_threads) {
    return preload_multi(accessions, {k}, sz, n_threads);
}

std::pair<size_t, size_t> PreloadedPackReader::preload_multi(
    const std::vector<std::string>& accessions,
    const std::vector<uint32_t>& ks,
    uint32_t sz, int n_threads)
{
    clear_store_();
    preload_set_ = accessions;
    const uint32_t mw = (sz + 63u) / 64u;
    auto t0 = std::chrono::steady_clock::now();

    for (uint32_t k : ks) {
        try {
            KStore st;
            st.k = k;
            st.sz = sz;
            st.mask_words = mw;
            populate_store_(st, accessions, n_threads);
            k_stores_.emplace(k, std::move(st));
            // Drop inner caches between ks: prevents per-k cache stacking
            // that pushes peak above the cgroup limit on large buckets.
            inner_->release_sketches();
        } catch (const std::bad_alloc& e) {
            spdlog::error("preload_multi: allocation failed at k={} (sz={}): {}",
                          k, sz, e.what());
            clear_store_();
            throw;
        }
    }
    trim_heap();

    auto t1 = std::chrono::steady_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    const size_t total_bytes = bytes();
    spdlog::info(
        "PRELOAD: multi-k done ks=[{}] sz={} acc_n={} hits={} bytes={} ({:.2f} GiB) wall={}ms",
        [&]{ std::string s; for (uint32_t k : ks) { if(!s.empty()) s+=','; s+=std::to_string(k); } return s; }(),
        sz, accessions.size(), acc_to_idx_.size(),
        total_bytes, total_bytes / double(1ull << 30), ms);
    return {acc_to_idx_.size(), total_bytes};
}

void PreloadedPackReader::reload_for_k(uint32_t new_k, int n_threads) {
    if (preload_set_.empty()) {
        spdlog::warn("PRELOAD: reload_for_k called without prior preload — ignoring");
        return;
    }
    // Preserve any existing sz from an existing store; assume all stores share sz.
    uint32_t sz = 0;
    if (!k_stores_.empty()) sz = k_stores_.begin()->second.sz;
    if (sz == 0) {
        spdlog::warn("PRELOAD: reload_for_k has no prior sz — ignoring");
        return;
    }
    preload_multi(preload_set_, {new_k}, sz, n_threads);
}

void PreloadedPackReader::visit_sketch_batches(
    const std::vector<std::string>& accessions,
    uint32_t k, uint32_t sz,
    const std::function<void(size_t idx,
                             const genopack::SketchResult& sk)>& cb) const {

    auto it = k_stores_.find(k);
    if (it != k_stores_.end() && it->second.sz == sz && !acc_to_idx_.empty()) {
        const KStore& st = it->second;
        std::vector<std::string> miss_accs;
        std::vector<size_t>      miss_orig;
        miss_accs.reserve(accessions.size() / 32);
        miss_orig.reserve(accessions.size() / 32);

        for (size_t i = 0; i < accessions.size(); ++i) {
            auto pos = acc_to_idx_.find(accessions[i]);
            if (pos == acc_to_idx_.end()) {
                miss_accs.push_back(accessions[i]);
                miss_orig.push_back(i);
                continue;
            }
            const uint32_t j = pos->second;
            genopack::SketchResult r;
            r.sig         = st.sigs.data()  + static_cast<size_t>(j) * st.sz;
            r.sig2        = st.sig2s.data() + static_cast<size_t>(j) * st.sz;
            r.mask        = st.masks.data() + static_cast<size_t>(j) * st.mask_words;
            r.n_real_bins = st.n_real_bins[j];
            r.mask_words  = st.mask_words;
            r.genome_length = st.genome_lengths[j];
            r.sketch_size = st.sz;
            r.kmer_size   = st.k;
            cb(i, r);
        }

        if (!miss_accs.empty()) {
            inner_->visit_sketch_batches(miss_accs, k, sz,
                [&](size_t mi, const genopack::SketchResult& sk) {
                    cb(miss_orig[mi], sk);
                });
        }
        return;
    }

    // Requested (k, sz) not preloaded — delegate entirely.
    inner_->visit_sketch_batches(accessions, k, sz, cb);
}

} // namespace derep
