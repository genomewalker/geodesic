#include "state/run_state.hpp"
#include <algorithm>

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



const std::vector<TaxonOutput>& RunState::taxa() const {
    std::lock_guard lock(mutex_);
    return taxa_;
}

void RunState::finalize_sort() {
    std::lock_guard lock(mutex_);
    std::sort(taxa_.begin(), taxa_.end(),
              [](const TaxonOutput& a, const TaxonOutput& b) {
                  return a.result.taxonomy < b.result.taxonomy;
              });
}

} // namespace derep
