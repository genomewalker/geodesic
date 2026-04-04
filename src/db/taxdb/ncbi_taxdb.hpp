#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <optional>

namespace derep {

namespace fs = std::filesystem;

// NCBI taxonomy database loader and 10-rank normaliser.
//
// Downloads new_taxdump.tar.gz from NCBI FTP on first use (or when stale).
// Parses nodes.dmp + names.dmp into a fast in-memory lookup.
// Converts NCBI lineages to the canonical 10-rank format used across this database:
//   d__ (superkingdom) | l__ (lineage/clade) | k__ (kingdom) |
//   p__ (phylum) | c__ (class) | o__ (order) | f__ (family) |
//   g__ (genus) | s__ (species) | S__ (subspecies)
//
// For Bacteria and Archaea, callers should use the GTDB path (pipeline.cpp).
// This class handles Eukaryota, Viruses, and other NCBI-only domains.

class NcbiTaxdb {
public:
    static constexpr int kMaxAgeDays = 30;
    static constexpr const char* kDownloadUrl =
        "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/new_taxdump/new_taxdump.tar.gz";

    // Load from an existing taxdump directory (must contain nodes.dmp + names.dmp).
    static NcbiTaxdb load(const fs::path& dir);

    // Ensure taxdump is present and not older than max_age_days.
    // Downloads and extracts new_taxdump.tar.gz if needed.
    // Writes a .timestamp file alongside the dump files.
    static void ensure_fresh(const fs::path& dir, int max_age_days = kMaxAgeDays);

    // Return the timestamp of the current dump, or nullopt if not present.
    static std::optional<std::chrono::system_clock::time_point>
        dump_timestamp(const fs::path& dir);

    // Build a canonical 10-rank taxonomy string for a given NCBI taxid.
    // Returns empty string if taxid is not found.
    // accession is used as the species leaf when s__ cannot be resolved.
    std::string taxonomy_for_taxid(int taxid, const std::string& accession) const;

    // Look up a taxid by assembly accession (GCF_/GCA_ stem → taxid).
    // Returns -1 if not found. Requires load_accession_map() to have been called.
    int taxid_for_accession(const std::string& accession) const;

    // Load an NCBI assembly_summary.txt to populate the accession → taxid map.
    void load_accession_map(const fs::path& assembly_summary);

    bool empty() const { return name_.empty(); }
    std::size_t size() const { return name_.size(); }

private:
    struct Node {
        int parent_taxid = 0;
        std::string rank;
    };

    std::unordered_map<int, Node>        nodes_;   // taxid → {parent, rank}
    std::unordered_map<int, std::string> name_;    // taxid → scientific name
    std::unordered_map<std::string, int> acc2tid_; // GCF_XXXXX → taxid

    // Walk up the parent chain from taxid, collecting all nodes.
    std::vector<std::pair<std::string, std::string>> // [(rank, name), ...]
        lineage(int taxid) const;

    // Map a full NCBI lineage to the 10 canonical rank slots.
    // Implements the same logic as taxdb-integration/scripts/format_ncbi_taxpath.R:
    // - l__ = kingdom-level clade (Opisthokonta, SAR, Viridiplantae, etc.)
    // - Missing ranks propagated from parent with prefix swap
    std::string lineage_to_10rank(
        const std::vector<std::pair<std::string, std::string>>& lineage,
        const std::string& accession) const;
};

} // namespace derep
