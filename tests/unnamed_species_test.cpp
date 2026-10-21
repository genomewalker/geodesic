#include <catch2/catch_test_macros.hpp>
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
