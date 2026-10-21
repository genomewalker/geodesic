#pragma once
#include "config.hpp"
#include "core/types.hpp"
#include <map>
#include <algorithm>
#include <istream>
#include <tuple>
namespace derep {
struct IPackReader;
namespace unnamed {
struct Hit { double ani = 0, af = 0; auto operator<=>(const Hit&) const = default; };
using OrderKey = std::tuple<bool,int,double,uint32_t,int,std::string>;
struct Ordering { int tier; double completeness; uint32_t contigs; int ncbi; };
std::map<std::string,Ordering> read_ordering(std::istream&);
OrderKey order_key(const std::string&, bool prior, const Ordering&);
bool replace_prior(const Ordering& candidate, const Ordering& current);
struct AnchorHit {
    std::string ref, species;
    double ani=0, af_ref=0, af_query=0, radius=95;
    double af() const { return std::max(af_ref,af_query); }
    bool anchored() const { return af()>=50 && ani>=radius; }
};
void consider_anchor(std::optional<AnchorHit>&, const AnchorHit&);
using Edges = std::map<std::pair<size_t, size_t>, Hit>;
Hit hit(const Edges&, size_t, size_t);
std::map<size_t,size_t> greedy(const std::vector<size_t>& ordered, const Edges&);
bool unresolved(const Genome&);
bool ncbi_accession(std::string);
std::string rank(const std::string&, char);
std::string replace_rank(const std::string&, char, const std::string&);
std::string synthetic_digits(const std::string&);
}
std::vector<Taxon> cluster_unnamed_species(std::vector<Genome>&,
    const std::unordered_map<std::string,std::string>&, IPackReader&, const Config&);
}
