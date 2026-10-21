#pragma once
#include "config.hpp"
#include "core/types.hpp"
#include <map>
namespace derep {
struct IPackReader;
namespace unnamed {
struct Hit { double ani = 0, af = 0; auto operator<=>(const Hit&) const = default; };
using Edges = std::map<std::pair<size_t, size_t>, Hit>;
Hit hit(const Edges&, size_t, size_t);
std::map<size_t,size_t> greedy(const std::vector<size_t>& ordered, const Edges&);
bool unresolved(const Genome&);
std::string rank(const std::string&, char);
std::string replace_rank(const std::string&, char, const std::string&);
std::string synthetic_digits(const std::string&);
}
std::vector<Taxon> cluster_unnamed_species(std::vector<Genome>&,
    const std::unordered_map<std::string,std::string>&, IPackReader&, const Config&);
}
