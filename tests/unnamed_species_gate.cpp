// Gate harness: exercise production grouping without downstream dereplication cost.
#include "taxonomy/unnamed.hpp"
#include "core/multi_pack_reader.hpp"
#include <fstream>
#include <iostream>
#include <filesystem>
int main(int argc,char** argv) {
    try {
        if(argc<6) throw std::runtime_error("usage: unnamed_species_gate PACK LIST OUTPUT SKANI RADII");
        derep::Config cfg; cfg.pack_dir=argv[1];cfg.results_dir=argv[3];cfg.temp_dir="/tmp/kbd606_geoB";
        cfg.skani_bin=argv[4];cfg.species_radii=argv[5];cfg.threads=32;
        bool bare=false;
        for(int i=6;i<argc;++i) {
            std::string option=argv[i];
            if(option=="bare") bare=true;
            else if(option=="--ordering-tsv" && i+1<argc) cfg.ordering_tsv=argv[++i];
            else if(option=="--anchor-skani-db" && i+1<argc) cfg.anchor_skani_db=argv[++i];
            else if(option=="--anchor-skani-bin" && i+1<argc) cfg.anchor_skani_bin=argv[++i];
            else throw std::runtime_error("Unknown gate option: "+option);
        }
        std::filesystem::create_directories(cfg.results_dir);
        auto pack=derep::MultiPackReader::open_dir(argv[1]);
        std::vector<derep::Genome> genomes;std::ifstream in(argv[2]);std::string a;
        while(std::getline(in,a)) {derep::Genome g;g.accession=a;g.taxonomy=pack->taxonomy_for_accession(a);if(g.taxonomy.empty())throw std::runtime_error("Missing taxonomy: "+a);if(bare && derep::unnamed::unresolved(g)) g.taxonomy=derep::unnamed::replace_rank(g.taxonomy,'s',"");genomes.push_back(g);}
        auto taxa=derep::cluster_unnamed_species(genomes,{},*pack,cfg);
        std::ofstream out(cfg.results_dir/"membership.tsv");out<<"accession\tcluster\ttaxonomy\n";
        for(size_t i=0;i<taxa.size();++i) for(auto& g:taxa[i].genomes) out<<g.accession<<'\t'<<i<<'\t'<<taxa[i].taxonomy<<'\n';
        std::cout<<"clusters="<<taxa.size()<<" genomes="<<genomes.size()<<'\n';
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
