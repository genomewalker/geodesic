#include "unnamed.hpp"
#include "taxonomy/species.hpp"
#include "core/pack_reader.hpp"
#include "core/subprocess.hpp"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <array>
#include <bit>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <mutex>
#include <regex>
#include <set>
#include <sstream>
#include <unistd.h>
#include <chrono>
#include <cmath>

namespace derep::unnamed {
std::map<std::string,Ordering> read_ordering(std::istream& in) {
    auto split=[](const std::string& line) {
        std::vector<std::string> cells; std::istringstream row(line); std::string cell;
        while(std::getline(row,cell,'\t')) cells.push_back(cell);
        if(!cells.empty() && !cells.back().empty() && cells.back().back()=='\r') cells.back().pop_back();
        return cells;
    };
    std::string line; if(!std::getline(in,line)) throw std::runtime_error("Empty ordering TSV");
    auto header=split(line); std::map<std::string,size_t> col;
    for(size_t i=0;i<header.size();++i) col[header[i]]=i;
    for(auto name:{"accession","quality_tier","completeness_effective","n_contigs","ncbi"})
        if(!col.contains(name)) throw std::runtime_error("Missing ordering TSV column: "+std::string(name));
    std::map<std::string,Ordering> result;
    while(std::getline(in,line)) {
        if(line.empty()) continue;
        auto cells=split(line);
        auto value=[&](const char* name)->const std::string& {return cells.at(col.at(name));};
        auto tier=value("quality_tier");
        if(tier!="HQ" && tier!="MQ" && tier!="LQ") throw std::runtime_error("Invalid ordering quality tier: "+tier);
        auto number=[&](const char* name) {size_t end; auto v=std::stod(value(name),&end); if(end!=value(name).size() || !std::isfinite(v)) throw std::runtime_error("Invalid ordering number"); return v;};
        auto contigs=number("n_contigs"), ncbi=number("ncbi");
        if(contigs<0 || contigs>UINT32_MAX || std::floor(contigs)!=contigs || (ncbi!=0 && ncbi!=1)) throw std::runtime_error("Invalid ordering contigs/ncbi");
        Ordering o{tier=="HQ"?0:tier=="MQ"?1:2,number("completeness_effective"),static_cast<uint32_t>(contigs),static_cast<int>(ncbi)};
        if(value("accession").empty() || !result.emplace(value("accession"),o).second) throw std::runtime_error("Empty or duplicate ordering accession");
    }
    return result;
}
OrderKey order_key(const std::string& a,bool prior,const Ordering& o) {return {!prior,o.tier,-o.completeness,o.contigs,-o.ncbi,a};}
bool replace_prior(const Ordering& b,const Ordering& c) {return b.tier<c.tier || (b.tier==c.tier && b.completeness-c.completeness>=0.05);}
void consider_anchor(std::optional<AnchorHit>& best,const AnchorHit& h) {
    if(h.af()>=50 && (!best || std::pair{h.ani,h.af()}>std::pair{best->ani,best->af()})) best=h;
}
std::string rank(const std::string& tax, char r) {
    std::istringstream in(tax); std::string part;
    while (std::getline(in, part, ';')) {
        auto start = part.find_first_not_of(" \t");
        if (start != std::string::npos) part.erase(0, start);
        if (part.size() >= 3 && part[0] == r && part.substr(1,2) == "__") return part.substr(3);
    }
    return {};
}
std::string replace_rank(const std::string& tax, char r, const std::string& value) {
    std::istringstream in(tax); std::string part, out; bool found = false;
    while (std::getline(in, part, ';')) {
        auto start = part.find_first_not_of(" \t");
        if (start != std::string::npos && part.size() >= start+3 && part[start] == r && part.substr(start+1,2) == "__") {
            part = std::string(1,r)+"__"+value; found = true;
        }
        if (!out.empty()) out += ';';
        out += part;
    }
    if (!found) { if (!out.empty()) out += ';'; out += std::string(1,r)+"__"+value; }
    return out;
}
bool unresolved(const Genome& g) {
    const auto species = rank(g.taxonomy, 's');
    const auto upper = rank(g.taxonomy, 'S');
    std::istringstream ranks(g.taxonomy); std::string part; bool has_lower = false;
    while (std::getline(ranks, part, ';')) {
        auto start = part.find_first_not_of(" \t");
        if (start != std::string::npos && part.compare(start, 3, "s__") == 0) has_lower = true;
    }
    const auto& value = has_lower ? species : upper;
    return value.empty() || value == taxonomy::species_stem(g.accession);
}
bool ncbi_accession(std::string a) {
    if (a.starts_with("RS_") || a.starts_with("GB_")) a.erase(0,3);
    return a.starts_with("GCA_") || a.starts_with("GCF_");
}
Hit hit(const Edges& e, size_t a, size_t b) {
    if (a == b) return {100,100};
    auto it = e.find(std::minmax(a,b)); return it == e.end() ? Hit{} : it->second;
}
std::map<size_t,size_t> greedy(const std::vector<size_t>& ordered, const Edges& e) {
    std::set<size_t> centroids;
    // All centroids are selected before membership is assigned (S2).
    std::map<size_t,std::vector<size_t>> neighbors;
    for (const auto& [pair,h] : e) if (h.ani >= 95 && h.af >= 50) {
        neighbors[pair.first].push_back(pair.second); neighbors[pair.second].push_back(pair.first);
    }
    for (auto x : ordered) {
        bool blocked = false;
        for (auto c : neighbors[x]) if (centroids.contains(c)) { blocked = true; break; }
        if (!blocked) centroids.insert(x);
    }
    std::map<size_t,size_t> assigned;
    for (auto x : ordered) {
        size_t best = x;
        if (!centroids.contains(x)) {
            Hit score{};
            for (auto c : neighbors[x]) if (centroids.contains(c)) {
                auto h = hit(e,x,c);
                if (h > score || (h == score && c > best)) { best = c; score = h; }
            }
        }
        assigned[x] = best;
    }
    return assigned;
}
// SHA-1, reduced modulo 10^8 exactly as int(sha1(accession).hexdigest(),16).
std::string synthetic_digits(const std::string& accession) {
    std::vector<uint8_t> msg(accession.begin(), accession.end());
    uint64_t bits = msg.size()*8; msg.push_back(128);
    while (msg.size()%64 != 56) msg.push_back(0);
    for (int i=7;i>=0;--i) msg.push_back(bits>>(i*8));
    std::array<uint32_t,5> h{0x67452301,0xefcdab89,0x98badcfe,0x10325476,0xc3d2e1f0};
    for (size_t base=0;base<msg.size();base+=64) {
        uint32_t w[80]{};
        for(int i=0;i<16;++i) for(int j=0;j<4;++j) w[i]=(w[i]<<8)|msg[base+4*i+j];
        for(int i=16;i<80;++i) w[i]=std::rotl(w[i-3]^w[i-8]^w[i-14]^w[i-16],1);
        auto [a,b,c,d,e]=h;
        for(int i=0;i<80;++i) {
            uint32_t f,k;
            if(i<20) {f=(b&c)|(~b&d);k=0x5a827999;}
            else if(i<40) {f=b^c^d;k=0x6ed9eba1;}
            else if(i<60) {f=(b&c)|(b&d)|(c&d);k=0x8f1bbcdc;}
            else {f=b^c^d;k=0xca62c1d6;}
            auto t=std::rotl(a,5)+f+e+k+w[i];e=d;d=c;c=std::rotl(b,30);b=a;a=t;
        }
        h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;
    }
    uint64_t n=0; for(auto x:h) for(int i=3;i>=0;--i) n=(n*256+((x>>(i*8))&255))%100000000;
    return std::to_string(900000000+n);
}
} // namespace derep::unnamed

namespace derep {
namespace {
using namespace unnamed;
namespace fs = std::filesystem;
using Block = std::map<std::string,std::vector<size_t>>;
std::string canonical(std::string a) {
    if (a.starts_with("RS_") || a.starts_with("GB_")) a.erase(0,3);
    return a;
}
// Every operation owns its mkdtemp directory; never remove caller-owned files.
struct Scratch {
    fs::path path;
    explicit Scratch(const fs::path& parent) {
        fs::create_directories(parent);
        auto s=(parent/"unnamed-XXXXXX").string();
        if (!mkdtemp(s.data())) throw std::runtime_error("Cannot create unnamed-species scratch directory");
        path=s;
    }
    ~Scratch() { std::error_code ec; fs::remove_all(path,ec); }
};
class Skani {
    IPackReader& pack; const Config& cfg; const std::vector<Genome>& genomes;
public:
    Skani(IPackReader& p,const Config& c,const std::vector<Genome>& g):pack(p),cfg(c),genomes(g){}
    Edges compare(const std::vector<size_t>& query, const std::vector<size_t>& refs, bool triangle) {
        if (query.empty() || refs.empty() || (triangle && query.size()<2)) return {};
        Scratch tmp(cfg.temp_dir);
        std::set<size_t> ids(query.begin(),query.end()); ids.insert(refs.begin(),refs.end());
        std::vector<std::string> accessions; std::unordered_map<std::string,size_t> index;
        for(auto i:ids) {accessions.push_back(genomes[i].accession); index.emplace(genomes[i].accession,i);}
        std::set<size_t> written; std::mutex written_mutex;
        pack.visit_shard_batches(accessions,[&](genopack::ArchiveReader::ShardBatch& batch) {
            for(auto& [batch_index,g]:batch) {
                auto it=index.find(g.accession); if(it==index.end()) continue;
                std::ofstream out(tmp.path/(std::to_string(it->second)+".fna"));
                out << g.fasta; if(!out) throw std::runtime_error("Cannot write extracted FASTA");
                std::lock_guard<std::mutex> lock(written_mutex); written.insert(it->second);
            }
        });
        if(written.size()!=ids.size()) throw std::runtime_error("Missing FASTA during unnamed species extraction");
        auto list=[&](const char* name,const std::vector<size_t>& v) {
            auto p=tmp.path/name; std::ofstream out(p);
            for(auto i:v) out << (tmp.path/(std::to_string(i)+".fna")).string() << '\n';
            if(!out) throw std::runtime_error("Cannot write skani list"); return p.string();
        };
        std::vector<std::string> args{cfg.skani_bin,triangle?"triangle":"dist"};
        if(triangle) {args.insert(args.end(),{"-l",list("q.list",query),"-E"});}
        else args.insert(args.end(),{"--ql",list("q.list",query),"--rl",list("r.list",refs)});
        args.insert(args.end(),{"-s","90","--min-af","0","-t",std::to_string(cfg.threads)});
        SubprocessOptions opts; opts.stdout_file=tmp.path/"hits.tsv";
        auto result=run_subprocess(args,opts);
        if(!result.ok()) throw std::runtime_error("skani failed: "+result.stderr_output);
        Edges edges; std::ifstream in(*opts.stdout_file); std::string line; std::getline(in,line);
        while(std::getline(in,line)) {
            std::istringstream row(line); std::string a,b; double ani,afr,afq;
            if(!(row>>a>>b>>ani>>afr>>afq)) throw std::runtime_error("Malformed skani row: "+line);
            auto i=std::stoull(fs::path(a).stem().string()),j=std::stoull(fs::path(b).stem().string());
            if(!ids.contains(i)||!ids.contains(j)) throw std::runtime_error("Unexpected skani genome");
            if(i!=j && ani>=90 && std::max(afr,afq)>=50) edges[std::minmax(i,j)]={ani,std::max(afr,afq)};
        }
        return edges;
    }
};
std::string block_key(const Genome& g,char r) {
    // Full lineage prevents unrelated unresolved family/phylum ranks merging.
    std::istringstream in(g.taxonomy); std::string part,key;
    while(std::getline(in,part,';')) {auto start=part.find_first_not_of(" \t"); if(start!=std::string::npos) part.erase(0,start);key+=part+';'; if(part.starts_with(std::string(1,r)+"__")) break;}
    return key;
}
}
std::vector<Taxon> cluster_unnamed_species(std::vector<Genome>& input,
    const std::unordered_map<std::string,std::string>& fixed, IPackReader& pack,const Config& cfg) {
    using namespace unnamed;
    // Stable accession order also implements S2's accession tie-break in best hits.
    std::vector<Genome> genomes=input;
    std::sort(genomes.begin(),genomes.end(),[](auto& a,auto& b){return a.accession<b.accession;});
    std::set<std::string> selected,used_names,needed_families;
    for(auto& g:genomes) {
        selected.insert(g.accession);
        if(unresolved(g)) needed_families.insert(block_key(g,'f'));
    }
    auto seen=selected;
    // Include named representatives in the pack even when outside --genomes.
    pack.scan_taxonomy([&](std::string_view a,std::string_view t){
        Genome g;g.accession=a;g.taxonomy=t;
        if(!unresolved(g)) {
            used_names.insert(rank(g.taxonomy,'s'));
            if(!cfg.anchor_skani_db && needed_families.contains(block_key(g,'f')) && seen.insert(g.accession).second) genomes.push_back(std::move(g));
        }
    });
    std::sort(genomes.begin(),genomes.end(),[](auto& a,auto& b){return a.accession<b.accession;});
    std::map<std::string,double> radii;
    std::map<std::string,std::string> ref_species;
    if(cfg.species_radii) {
        std::ifstream in(*cfg.species_radii); std::string line;
        if(!in) throw std::runtime_error("Cannot read species radii");
        while(std::getline(in,line)) {
            auto last=line.rfind('\t'), first=line.find('\t');
            if(first==std::string::npos || first==last) throw std::runtime_error("Invalid species radii row");
            auto ref=canonical(line.substr(first+1,last-first-1));
            radii[ref]=std::stod(line.substr(last+1));
            ref_species[ref]=line.substr(0,first);
        }
    }
    auto radius=[&](size_t r){auto it=radii.find(canonical(genomes[r].accession));return it==radii.end()?95.:it->second;};
    Block families,named;
    std::map<std::string,size_t> named_reps;
    for(size_t i=0;i<genomes.size();++i) if(!unresolved(genomes[i])) {
        auto key=block_key(genomes[i],'s');
        auto it=named_reps.find(key);
        if(it==named_reps.end() || (radii.contains(canonical(genomes[i].accession)) && !radii.contains(canonical(genomes[it->second].accession)))) named_reps[key]=i;
    }
    for(size_t i=0;i<genomes.size();++i) {
        if(unresolved(genomes[i])) {if(selected.contains(genomes[i].accession)) families[block_key(genomes[i],'f')].push_back(i);}
        else if(named_reps.at(block_key(genomes[i],'s'))==i) named[block_key(genomes[i],'f')].push_back(i);
    }
    std::set<std::string> prior;
    if(cfg.references_file) {std::ifstream in(*cfg.references_file);std::string s;while(std::getline(in,s)) prior.insert(s);}
    for(auto& [t,a]:fixed) prior.insert(a);
    std::map<std::string,Ordering> overrides;
    if(cfg.ordering_tsv) {std::ifstream in(*cfg.ordering_tsv); if(!in) throw std::runtime_error("Cannot read ordering TSV"); overrides=read_ordering(in);}
    std::map<size_t,OrderKey> keys;
    std::map<size_t,Ordering> ordering;
    for(auto& [f,ids]:families) for(auto i:ids) {
        auto& g=genomes[i]; auto meta=pack.genome_meta_by_accession(g.accession);
        auto tier=pack.quality_tier_for_accession(g.accession).value_or(1);
        double completeness=pack.completeness_effective_for_accession(g.accession).value_or(g.completeness.value_or(meta ? meta->completeness_x10/10. : 0)/100.);
        Ordering o{3-std::clamp(int(tier),1,3),completeness,meta?meta->n_contigs:UINT32_MAX,int(ncbi_accession(g.accession))};
        if(overrides.contains(g.accession)) o=overrides.at(g.accession);
        ordering[i]=o; keys[i]=order_key(g.accession,prior.contains(g.accession),o);
    }
    auto ordered=[&](std::vector<size_t> ids){std::sort(ids.begin(),ids.end(),[&](auto a,auto b){return keys.at(a)<keys.at(b);});return ids;};
    Skani skani(pack,cfg,genomes);
    std::map<size_t,size_t> assigned,anchors,family_assignment;
    std::map<size_t,Hit> evidence;
    std::map<size_t,double> nearest,residual_nearest,cluster_radius;
    Block centroids;
    std::map<size_t,AnchorHit> database_anchors;
    if(cfg.anchor_skani_db) {
        if(!cfg.species_radii) throw std::runtime_error("--anchor-skani-db requires --species-radii");
        Scratch tmp(cfg.temp_dir);
        std::unordered_map<std::string,size_t> index;
        std::vector<std::string> accessions;
        for(auto& [f,ids]:families) for(auto x:ids) {index[genomes[x].accession]=x;accessions.push_back(genomes[x].accession);}
        std::mutex mutex; std::set<size_t> written;
        pack.visit_shard_batches(accessions,[&](genopack::ArchiveReader::ShardBatch& batch) {
            for(auto& [bi,g]:batch) {auto it=index.find(g.accession);if(it==index.end()) continue;
                std::ofstream out(tmp.path/(std::to_string(it->second)+".fna"));out<<g.fasta;
                if(!out) throw std::runtime_error("Cannot write anchor FASTA");
                std::lock_guard<std::mutex> lock(mutex);written.insert(it->second);
            }
        });
        if(written.size()!=index.size()) throw std::runtime_error("Missing anchor FASTA");
        auto list=tmp.path/"query.list";
        {std::ofstream out(list);for(auto x:written) out<<(tmp.path/(std::to_string(x)+".fna")).string()<<'\n';if(!out) throw std::runtime_error("Cannot write anchor list");}
        SubprocessOptions opts;opts.stdout_file=tmp.path/"hits.tsv";
        auto start=std::chrono::steady_clock::now();
        if(!written.empty()) {
            auto result=run_subprocess({cfg.anchor_skani_bin,"search","-d",cfg.anchor_skani_db->string(),"--ql",list.string(),"-t",std::to_string(cfg.threads),"--min-af","0"},opts);
            if(!result.ok()) throw std::runtime_error("Anchor skani search failed: "+result.stderr_output);
        }
        double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        spdlog::info("Anchor skani search wall: {} seconds",seconds);
        {std::ofstream out(cfg.results_dir/"anchor_search_seconds.txt");out<<std::setprecision(12)<<seconds<<'\n';}
        std::map<size_t,std::optional<AnchorHit>> top;
        std::ifstream in(*opts.stdout_file);std::string line;std::getline(in,line);
        std::regex accession("GC[AF]_[0-9]+\\.[0-9]+");
        while(std::getline(in,line)) {
            std::istringstream row(line);std::string ref,q;AnchorHit h;
            if(!(row>>ref>>q>>h.ani>>h.af_ref>>h.af_query)) throw std::runtime_error("Malformed anchor skani row");
            std::smatch match;auto basename=fs::path(ref).filename().string();
            if(!std::regex_search(basename,match,accession) || !radii.contains(match.str())) continue;
            h.ref=match.str();h.species=ref_species.at(h.ref);h.radius=radii.at(h.ref);
            auto x=std::stoull(fs::path(q).stem().string());if(!written.contains(x)) throw std::runtime_error("Unexpected anchor query");
            consider_anchor(top[x],h);
        }
        std::ofstream out(cfg.results_dir/"anchor_hits.tsv");out<<"accession\tref\tspecies\tani\taf_ref\taf_query\tradius\tanchored\n";
        for(auto& [x,h]:top) if(h) {
            nearest[x]=h->ani;if(h->anchored()) database_anchors[x]=*h;
            out<<genomes[x].accession<<'\t'<<h->ref<<'\t'<<h->species<<'\t'<<h->ani<<'\t'<<h->af_ref<<'\t'<<h->af_query<<'\t'<<h->radius<<'\t'<<h->anchored()<<'\n';
        }
    }
    for(auto& [family,ids]:families) {
        spdlog::info("Unnamed species family: {} genomes, {} named references",ids.size(),named[family].size());
        auto hits=cfg.anchor_skani_db ? Edges{} : skani.compare(ids,named[family],false);
        std::map<size_t,std::pair<Hit,size_t>> top;
        for(auto& [pair,h]:hits) {
            auto [a,b]=pair; if(!keys.contains(a)) std::swap(a,b);
            auto v=std::pair{h,b}; if(!top.contains(a)||v>top[a]) top[a]=v;
            nearest[a]=std::max(nearest[a],h.ani);
        }
        std::vector<size_t> remaining;
        for(auto x:ids) {
            if(database_anchors.contains(x)) continue;
            auto it=top.find(x);
            if(it!=top.end() && it->second.first.ani>=radius(it->second.second)) {
                anchors[x]=it->second.second; evidence[x]=it->second.first;
            } else remaining.push_back(x);
        }
        auto edges=skani.compare(remaining,remaining,true);
        auto fa=greedy(ordered(remaining),edges);
        family_assignment.insert(fa.begin(),fa.end());
        for(auto [x,c]:fa) if(x==c) centroids[block_key(genomes[x],'p')].push_back(x);
    }
    for(auto& [phylum,ids]:centroids) {
        auto edges=skani.compare(ids,ids,true);
        auto ga=greedy(ordered(ids),edges);
        std::set<size_t> reps; for(auto [x,r]:ga) reps.insert(r);
        for(auto [pair,h]:edges) if(reps.contains(pair.first)&&reps.contains(pair.second)) {
            residual_nearest[pair.first]=std::max(residual_nearest[pair.first],h.ani);
            residual_nearest[pair.second]=std::max(residual_nearest[pair.second],h.ani);
        }
        for(auto r:reps) cluster_radius[r]=std::clamp(std::max(residual_nearest[r],nearest[r]),95.,97.);
        for(auto [x,c]:family_assignment) if(ga.contains(c)) assigned[x]=ga.at(c);
        for(auto r:reps) if(cfg.ordering_tsv && prior.contains(genomes[r].accession)) {
            std::vector<size_t> members;for(auto [x,c]:assigned) if(c==r) members.push_back(x);
            auto best=*std::min_element(members.begin(),members.end(),[&](auto a,auto b){return order_key(genomes[a].accession,false,ordering.at(a))<order_key(genomes[b].accession,false,ordering.at(b));});
            if(best!=r && replace_prior(ordering.at(best),ordering.at(r))) {
                cluster_radius[best]=std::clamp(std::max(residual_nearest[r],nearest[best]),95.,97.);
                for(auto x:members) assigned[x]=best;
                cluster_radius.erase(r);
            }
        }
    }
    // S2 re-check batches: members grouped by the family of their assigned rep.
    Block queries,refs;
    for(auto [x,r]:assigned) {auto f=block_key(genomes[r],'f');queries[f].push_back(x);if(x==r) refs[f].push_back(r);}
    auto initial=assigned;
    const auto initial_radii=cluster_radius;
    std::map<size_t,size_t> split_from;
    for(auto& [family,ids]:queries) {
        auto hits=skani.compare(ids,refs[family],false);
        std::map<size_t,std::vector<std::pair<size_t,Hit>>> nb;
        for(auto [pair,h]:hits) { nb[pair.first].push_back({pair.second,h});nb[pair.second].push_back({pair.first,h}); }
        for(auto x:ids) {
            auto r=initial.at(x);auto h=hit(hits,x,r);
            if(h.ani>=cluster_radius.at(r)&&h.af>=50) {evidence[x]=h;continue;}
            std::pair<Hit,size_t> best{};double nr=95;
            for(auto [other,v]:nb[x]) if(initial_radii.contains(other)) {
                nr=std::max(nr,v.ani);
                if(v.ani>=initial_radii.at(other)&&v.af>=50) best=std::max(best,std::pair{v,other});
            }
            if(best.first.ani) {assigned[x]=best.second;evidence[x]=best.first;}
            else {assigned[x]=x;split_from[x]=r;cluster_radius[x]=std::clamp(nr,95.,97.);evidence[x]={100,100};}
        }
    }
    Block final_reps;
    for(auto [x,r]:assigned) if(x==r) final_reps[block_key(genomes[r],'p')].push_back(r);
    size_t conflicts=0;
    std::ofstream conflict_out(cfg.results_dir/"final_rep_conflicts.tsv");
    conflict_out<<"rep_a\trep_b\tani\tmax_af\tradius_a\tradius_b\tstatus\n";
    // A re-check singleton below its former rep's radius keeps its edge to that rep (S2 reval_fail): logged, not a conflict.
    size_t split_off=0;
    for(auto& [p,ids]:final_reps) for(auto [pair,h]:skani.compare(ids,ids,true)) if(h.ani>=95&&h.af>=50) {
        auto [a,b]=pair;
        bool split=(split_from.contains(a)&&split_from.at(a)==b)||(split_from.contains(b)&&split_from.at(b)==a);
        split?++split_off:++conflicts;
        conflict_out<<genomes[a].accession<<'\t'<<genomes[b].accession<<'\t'<<h.ani<<'\t'<<h.af<<'\t'<<cluster_radius.at(a)<<'\t'<<cluster_radius.at(b)<<'\t'<<(split?"split_off":"conflict")<<'\n';
    }
    spdlog::info("Unnamed species final check: {} split-off singleton edges to their former rep logged",split_off);
    if(conflicts) {
        std::ofstream diagnostic(cfg.results_dir/"failed_assignments.tsv");
        diagnostic<<"accession\trep\tradius\tani_to_rep\taf_to_rep\tsource\n";
        for(auto [x,r]:assigned) diagnostic<<genomes[x].accession<<'\t'<<genomes[r].accession<<'\t'<<cluster_radius.at(r)<<'\t'<<evidence[x].ani<<'\t'<<evidence[x].af<<"\tcluster\n";
        for(auto [x,r]:anchors) diagnostic<<genomes[x].accession<<'\t'<<genomes[r].accession<<'\t'<<radius(r)<<'\t'<<evidence[x].ani<<'\t'<<evidence[x].af<<"\tanchor\n";
        for(auto& [x,h]:database_anchors) diagnostic<<genomes[x].accession<<'\t'<<h.ref<<'\t'<<h.radius<<'\t'<<h.ani<<'\t'<<h.af()<<"\tanchor\n";
    }
    spdlog::info("Unnamed species final check: {} representative edges; cross-phylum pairs not searched",conflicts);
    if(conflicts) throw std::runtime_error("Unnamed species final representatives share species-rule edges");
    std::map<size_t,std::vector<size_t>> members;
    for(auto [x,r]:assigned) members[r].push_back(x);
    std::map<size_t,std::string> species,genera;
    std::vector<size_t> rep_order;for(auto& [r,m]:members) rep_order.push_back(r);
    std::sort(rep_order.begin(),rep_order.end(),[&](auto a,auto b){return genomes[a].accession<genomes[b].accession;});
    size_t unresolved_clusters=0;
    std::regex ncbi("^(?:RS_|GB_)?GC[AF]_([0-9]{9})\\.[0-9]+$");
    for(auto r:rep_order) {
        std::map<std::string,size_t> votes;
        for(auto x:members[r]) {auto genus=rank(genomes[x].taxonomy,'g');bool inherited=false; for(char rank_id : std::string("dlkpcof")) if(genus==rank(genomes[x].taxonomy,rank_id)) inherited=true;
            if(!genus.empty() && !inherited && genus!=taxonomy::species_stem(genomes[x].accession) && genus!=genomes[x].accession) ++votes[genus];}
        if(votes.empty()) {++unresolved_clusters;continue;}
        size_t count=0;std::string genus;
        for(auto [g,n]:votes) if(n>count) {genus=g;count=n;}
        auto rg=rank(genomes[r].taxonomy,'g');if(votes.contains(rg)&&votes.at(rg)==count) genus=rg;
        std::smatch match;std::string name;
        if(std::regex_match(genomes[r].accession,match,ncbi)) name=genus+" sp"+match[1].str();
        if(name.empty()||used_names.contains(name)) {
            auto digits=std::stoull(synthetic_digits(genomes[r].accession))-900000000;
            do {name=genus+" sp"+std::to_string(900000000+digits);digits=(digits+1)%100000000;} while(used_names.contains(name));
        }
        used_names.insert(name);species[r]=name;genera[r]=genus;
    }
    // Genopack uses column three as full taxonomy when extra columns are present.
    std::ofstream out(cfg.results_dir/"species_assignments.tsv");
    out << "accession\tspecies\ttaxonomy\tcluster\trep\tgenus\tradius\tani_to_rep\taf_to_rep\tsource\n";
    std::map<std::string,Taxon> groups;
    for(size_t x=0;x<genomes.size();++x) if(selected.contains(genomes[x].accession)) {
        auto g=genomes[x]; std::string key=g.taxonomy;
        if(database_anchors.contains(x)) {
            const auto& h=database_anchors.at(x);
            auto sp=h.species.starts_with("s__")?h.species.substr(3):h.species;
            auto genus=sp.substr(0,sp.find(' '));
            g.taxonomy=replace_rank(replace_rank(replace_rank(g.taxonomy,'s',sp),'g',genus),'S',"");key=g.taxonomy;
            out<<g.accession<<"\ts__"<<sp<<'\t'<<g.taxonomy<<"\tanchor_"<<h.ref<<'\t'<<h.ref<<"\tg__"<<genus<<'\t'<<h.radius<<'\t'<<h.ani<<'\t'<<h.af()<<"\tanchor\n";
        } else if(anchors.contains(x)) {
            auto r=anchors.at(x);g.taxonomy=genomes[r].taxonomy;key=g.taxonomy;
            out<<g.accession<<"\ts__"<<rank(g.taxonomy,'s')<<'\t'<<g.taxonomy<<"\tanchor_"<<genomes[r].accession<<'\t'<<genomes[r].accession<<"\tg__"<<rank(g.taxonomy,'g')<<'\t'<<radius(r)<<'\t'<<evidence[x].ani<<'\t'<<evidence[x].af<<"\tanchor\n";
        } else if(assigned.contains(x)) {
            auto r=assigned.at(x);g.taxonomy=replace_rank(genomes[r].taxonomy,'g',genera[r]);
            g.taxonomy=replace_rank(g.taxonomy,'s',species[r]);g.taxonomy=replace_rank(g.taxonomy,'S',"");
            key="cluster_"+genomes[r].accession;
            if(species.contains(r)&&!species[r].empty()) out<<g.accession<<"\ts__"<<species[r]<<'\t'<<g.taxonomy<<"\tsn_"<<genomes[r].accession<<'\t'<<genomes[r].accession<<"\tg__"<<genera[r]<<'\t'<<cluster_radius.at(r)<<'\t'<<evidence[x].ani<<'\t'<<evidence[x].af<<"\tcluster\n";
        }
        auto& t=groups[key];t.taxonomy=g.taxonomy;t.genomes.push_back(std::move(g));
    }
    if(!out) throw std::runtime_error("Cannot write species assignments");
    std::vector<Taxon> taxa;
    for(auto& [key,t]:groups) {if(auto it=fixed.find(t.taxonomy);it!=fixed.end()) t.forced_representative=it->second;taxa.push_back(std::move(t));}
    std::sort(taxa.begin(),taxa.end(),[](auto& a,auto& b){return a.size()>b.size();});
    spdlog::info("Unnamed species: {} anchors, {} clusters, {} genus-unresolved clusters held out of naming",anchors.size()+database_anchors.size(),members.size(),unresolved_clusters);
    return taxa;
}
} // namespace derep
