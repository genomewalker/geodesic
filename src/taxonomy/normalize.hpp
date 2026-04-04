#pragma once
#include <array>
#include <string>
#include <string_view>

namespace derep { class NcbiTaxdb; }

namespace derep::taxonomy {

// Canonical 10-rank order: d, l, k, p, c, o, f, g, s, S.
// l__ (lineage) and k__ (kingdom) bridge GTDB and NCBI taxonomy.
// S__ (subspecies) is always derived from the species name.
inline constexpr std::array<std::string_view, 10> kRanks = {
    "d__", "l__", "k__", "p__", "c__", "o__", "f__", "g__", "s__", "S__"
};

// Strip NCBI prefix (RS_/GB_) and version suffix from accession.
// RS_GCF_003697165.2 → GCF_003697165
std::string species_stem(std::string_view acc);

// Normalise a taxonomy string to the canonical 10-rank format.
// Every rank is filled by propagation from its parent; no empty stubs.
//   - GTDB prokaryotes (d__Bacteria/d__Archaea): rank propagation + accession stem for s__.
//   - Non-prokaryotes: resolved via NcbiTaxdb if provided, else synthetic singleton.
//   - Non-d__ strings: fully synthetic per-accession singleton.
std::string normalize_taxonomy(const std::string& taxonomy,
                               const std::string& accession,
                               const NcbiTaxdb* ncbi = nullptr);

// Returns true when the species rank was unresolved and filled with the
// accession stem — such genomes become natural 1-genome taxa after grouping.
bool has_accession_species(const std::string& taxonomy, const std::string& accession);

} // namespace derep::taxonomy
