#include "species.hpp"

namespace derep::taxonomy {

std::string species_stem(std::string_view acc) {
    if (acc.starts_with("RS_") || acc.starts_with("GB_"))
        acc = acc.substr(3);
    if (acc.starts_with("GCF_") || acc.starts_with("GCA_")) {
        auto dot = acc.rfind('.');
        if (dot != std::string_view::npos)
            return std::string(acc.substr(0, dot));
    }
    return std::string(acc);
}

bool has_accession_species(const std::string& taxonomy, const std::string& accession) {
    const std::string needle = ";s__" + species_stem(accession) + ";";
    return taxonomy.find(needle) != std::string::npos;
}

} // namespace derep::taxonomy
