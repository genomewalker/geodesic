#include <catch2/catch_test_macros.hpp>
#include <sstream>
#include "taxonomy/unnamed.hpp"
using namespace derep;
using namespace derep::unnamed;
TEST_CASE("Unnamed species recognition supports bare and accession ranks", "[unnamed]") {
    Genome g;g.accession="RS_GCF_123456789.1";
    for(auto t:{"g__Foo;s__;S__", "g__Foo;s__;S__strain", "g__Foo;s__GCF_123456789;S__", "g__Foo;S__"}) {
        g.taxonomy=t;REQUIRE(unresolved(g));
    }
    g.taxonomy="g__Foo;s__Foo bar;S__";REQUIRE_FALSE(unresolved(g));
    g.taxonomy="g__Foo;  s__Foo bar;S__";REQUIRE_FALSE(unresolved(g));
    g.taxonomy="g__Foo;S__Foo bar";REQUIRE_FALSE(unresolved(g));
}
TEST_CASE("Greedy species clustering is not connected components", "[unnamed]") {
    Edges edges{{{0,1},{96,60}},{{1,2},{97,70}}};
    auto a=greedy({0,1,2},edges);
    REQUIRE(a.at(0)==0);REQUIRE(a.at(2)==2);REQUIRE(a.at(1)==2);
}
TEST_CASE("Species edges accept either AF and include threshold equality", "[unnamed]") {
    Edges edges{{{0,1},{95,50}},{{0,2},{99,49.99}},{{0,3},{94.99,100}}};
    auto a=greedy({0,1,2,3},edges);
    REQUIRE(a.at(1)==0);REQUIRE(a.at(2)==2);REQUIRE(a.at(3)==3);
}
TEST_CASE("Membership picks ANI then AF then accession", "[unnamed]") {
    Edges edges{{{0,2},{96,60}},{{1,2},{96,70}}};
    auto a=greedy({0,1,2},edges);REQUIRE(a.at(2)==1);
    edges[{0,2}]={96,70};a=greedy({0,1,2},edges);REQUIRE(a.at(2)==1);
}
TEST_CASE("Synthetic names use SHA1 modulo and rank replacement preserves bare ranks", "[unnamed]") {
    REQUIRE(synthetic_digits("abc")=="966479517");
    REQUIRE(replace_rank("d__Bacteria;g__;s__;S__",'g',"Foo")=="d__Bacteria;g__Foo;s__;S__");
}

TEST_CASE("NCBI ordering excludes GCMeta catalogue accessions", "[unnamed]") {
    REQUIRE(ncbi_accession("GCA_000000001.1"));
    REQUIRE(ncbi_accession("RS_GCF_000000001.1"));
    REQUIRE_FALSE(ncbi_accession("GCMeta_00778976"));
    REQUIRE_FALSE(ncbi_accession("custom_genome"));
}

TEST_CASE("S2 ordering TSV overrides use supplied values and column names", "[unnamed]") {
    std::istringstream in("ncbi\tn_contigs\taccession\tcompleteness_effective\tquality_tier\textra\n0\t8\tA\t1.25\tHQ\tx\n1\t8\tB\t1.25\tHQ\tx\n0\t1\tC\t2\tMQ\tx\n");
    auto rows=read_ordering(in);
    REQUIRE(rows.size()==3);REQUIRE(rows.at("A").completeness==1.25);
    REQUIRE(order_key("B",false,rows.at("B"))<order_key("A",false,rows.at("A")));
    REQUIRE(order_key("A",false,rows.at("A"))<order_key("C",false,rows.at("C")));
    REQUIRE(order_key("C",true,rows.at("C"))<order_key("A",false,rows.at("A")));
    REQUIRE_FALSE(rows.contains("unlisted"));
    std::istringstream missing("accession\tquality_tier\nA\tHQ\n");REQUIRE_THROWS(read_ordering(missing));
    std::istringstream invalid("accession\tquality_tier\tcompleteness_effective\tn_contigs\tncbi\nA\tBAD\t1\t1\t1\n");REQUIRE_THROWS(read_ordering(invalid));
}
TEST_CASE("Prior representative replacement uses tier then effective completeness margin", "[unnamed]") {
    Ordering current{1,1.2,5,0};
    REQUIRE(replace_prior({0,.1,99,0},current));
    REQUIRE_FALSE(replace_prior({2,2,1,1},current));
    REQUIRE(replace_prior({1,1.251,5,0},current));
    REQUIRE_FALSE(replace_prior({1,1.249,5,0},current));
    REQUIRE(replace_prior({1,.55,5,0},{1,.5,5,0}));
}
TEST_CASE("Database anchors rank AF-passing hits before applying the top reference radius", "[unnamed]") {
    std::optional<AnchorHit> top;
    consider_anchor(top,{"a","s__A a",99,49.99,40,95});REQUIRE_FALSE(top);
    consider_anchor(top,{"b","s__B b",96,20,50,95});REQUIRE(top->anchored());
    consider_anchor(top,{"c","s__C c",96,60,10,97});REQUIRE(top->ref=="c");REQUIRE_FALSE(top->anchored());
    consider_anchor(top,{"d","s__D d",96,10,60,95});REQUIRE(top->ref=="c");
    consider_anchor(top,{"e","s__E e",97,50,10,97});REQUIRE(top->anchored());REQUIRE(top->ref=="e");
}
