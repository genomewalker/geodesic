#pragma once
#include <string>
#include <string_view>

namespace derep::taxonomy {

// Strip NCBI prefix (RS_/GB_) and version suffix from accession.
// RS_GCF_003697165.2 → GCF_003697165
std::string species_stem(std::string_view acc);

// True when the pack's species rank equals the accession stem — i.e. genopack's
// build-time normaliser left the species unresolved and filled it with a
// per-accession name. Such genomes are natural 1-genome taxa after grouping.
// Read-side derep diagnostic; all taxonomy normalisation lives in genopack.
bool has_accession_species(const std::string& taxonomy, const std::string& accession);

} // namespace derep::taxonomy
