#pragma once
#include <BS_thread_pool.hpp>
#include <latch>
#include <algorithm>

namespace derep {

// Parallel-for over [0, n) split into nt = min(n_threads, n) blocks.
// Safe to call from a pool thread as long as the pool has enough slack
// (pool size >= total_budget + cfg.workers ensures this; see pipeline.cpp).
template<typename Fn>
inline void par_for(BS::thread_pool* pool, int n, int n_threads, Fn&& fn) {
    if (!pool || n_threads <= 1 || n <= 1) {
        for (int i = 0; i < n; ++i) fn(i);
        return;
    }
    const int nt = std::min(n_threads, n);
    std::latch latch(nt);
    const int block = (n + nt - 1) / nt;
    for (int t = 0; t < nt; ++t) {
        const int s = t * block;
        const int e = std::min(s + block, n);
        pool->detach_task([s, e, &fn, &latch] {
            for (int i = s; i < e; ++i) fn(i);
            latch.count_down();
        });
    }
    latch.wait();
}

// Run n_threads workers with indices [0, n_threads): fn(tid, n_threads).
// Use when per-thread setup or thread-indexed data structures are needed.
template<typename Fn>
inline void par_workers(BS::thread_pool* pool, int n_threads, Fn&& fn) {
    if (!pool || n_threads <= 1) {
        fn(0, 1);
        return;
    }
    std::latch latch(n_threads);
    for (int t = 0; t < n_threads; ++t) {
        pool->detach_task([t, n_threads, &fn, &latch] {
            fn(t, n_threads);
            latch.count_down();
        });
    }
    latch.wait();
}

} // namespace derep
