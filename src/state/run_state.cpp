#include "state/run_state.hpp"
#include <algorithm>
#include <string_view>
#include <unordered_set>
#include <spdlog/spdlog.h>

namespace derep {

namespace {
// Genomes that resolve in the archive but are absent from its SKCH section have
// no retrievable sketch, so geodesic cannot embed them (visit_sketch_batches
// never fires their callback). Rather than silently dropping them to failed.tsv,
// keep each as its own single-genome cluster (self-representative). Accessions
// that are genuinely unresolvable ("accession not found ...") stay failed.
constexpr std::string_view kNoSketchPrefix = "sketch not found";

void promote_missing_sketch_to_self_reps(TaxonOutput& out) {
    std::unordered_set<std::string> reps(out.representatives.begin(),
                                         out.representatives.end());
    std::vector<FailedGenomeRecord> kept;
    kept.reserve(out.failed_genomes.size());
    size_t promoted = 0;
    for (auto& f : out.failed_genomes) {
        if (f.reason.rfind(kNoSketchPrefix, 0) == 0 && reps.insert(f.accession).second) {
            out.representatives.push_back(f.accession);
            out.rep_cluster_size[f.accession] = 1u;
            ++promoted;
        } else {
            kept.push_back(std::move(f));
        }
    }
    if (promoted == 0) return;
    out.failed_genomes = std::move(kept);
    out.result.n_representatives = out.representatives.size();
    if (out.result.status == TaxonStatus::FAILED) {
        out.result.status = TaxonStatus::SUCCESS;
        if (out.diversity_stats.taxonomy.empty())
            out.diversity_stats.taxonomy = out.result.taxonomy;
        if (out.diversity_stats.method.empty())
            out.diversity_stats.method = "geodesic-self-rep";
        out.diversity_stats.n_genomes = static_cast<int>(out.result.n_genomes);
    }
    out.diversity_stats.n_representatives = static_cast<int>(out.representatives.size());
    if (out.result.n_genomes > 0)
        out.diversity_stats.reduction_ratio =
            1.0 - static_cast<double>(out.representatives.size()) /
                  static_cast<double>(out.result.n_genomes);
    spdlog::info("[{}] {} genome(s) absent from SKCH kept as self-representatives",
                 out.result.taxonomy, promoted);
}
} // namespace

void RunState::push(TaxonOutput output) {
    promote_missing_sketch_to_self_reps(output);
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

void RunState::merge(RunState&& src) {
    std::lock_guard lock(mutex_);
    std::lock_guard src_lock(src.mutex_);
    taxa_.insert(taxa_.end(),
                 std::make_move_iterator(src.taxa_.begin()),
                 std::make_move_iterator(src.taxa_.end()));
    src.taxa_.clear();
}

} // namespace derep
