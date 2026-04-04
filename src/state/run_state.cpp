#include "state/run_state.hpp"

namespace derep {

void RunState::push(TaxonOutput output) {
    std::lock_guard lock(mutex_);
    taxa_.push_back(std::move(output));
}

size_t RunState::total_genomes() const {
    std::lock_guard lock(mutex_);
    size_t total = 0;
    for (const auto& t : taxa_)
        total += t.result.n_genomes;
    return total;
}

size_t RunState::total_reps() const {
    std::lock_guard lock(mutex_);
    size_t total = 0;
    for (const auto& t : taxa_)
        total += t.result.n_representatives;
    return total;
}

size_t RunState::total_failed() const {
    std::lock_guard lock(mutex_);
    size_t total = 0;
    for (const auto& t : taxa_)
        if (t.result.status == TaxonStatus::FAILED)
            ++total;
    return total;
}

size_t RunState::total_singletons() const {
    std::lock_guard lock(mutex_);
    size_t total = 0;
    for (const auto& t : taxa_)
        if (t.result.status == TaxonStatus::SINGLETON)
            ++total;
    return total;
}

size_t RunState::total_contaminated() const {
    std::lock_guard lock(mutex_);
    size_t total = 0;
    for (const auto& t : taxa_)
        total += t.result.n_representatives > 0 ? t.diversity_stats.n_outliers_excluded : 0;
    return total;
}

const std::vector<TaxonOutput>& RunState::taxa() const {
    std::lock_guard lock(mutex_);
    return taxa_;
}

} // namespace derep
