#pragma once
#include <cstdint>
#include <string>
#include <vector>
namespace derep { struct IPackReader; }
namespace derep {
// Probes a handful of sketches to pick the best k from the archive's available k-mer sizes.
// Returns the chosen k, or 0 if probing fails / no sketches available.
// Matches the behaviour previously embedded in GeodesicDerep::probe_kmer_size_.
int probe_taxon_kmer(const std::vector<std::string>& accessions,
                     IPackReader& gpk,
                     int default_k,
                     int sketch_size);
} // namespace derep
