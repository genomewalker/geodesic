#include <catch2/catch_test_macros.hpp>
#include "../src/db/geodf/geodf_writer.hpp"
#include "../src/db/geodf/geodf_reader.hpp"
#include <filesystem>

using namespace geodf;

TEST_CASE("GEODF round-trip: single successful taxon", "[geodf]") {
    auto path = std::filesystem::temp_directory_path() / "geodf_test_single.geodf";
    std::filesystem::remove(path);

    TaxonResult r;
    r.taxonomy = "d__Bacteria;s__Test bacterium";
    r.genome_ids = {1, 2, 3};
    r.is_rep = {true, false, true};
    r.contamination = {0.0f, 0.5f, 0.0f};
    r.reps = {
        {1, "ACC001", {0.1f, 0.2f}},
        {3, "ACC003", {0.3f, 0.4f}}
    };
    r.stage = PipelineStage::COMPLETE;
    r.diversity_threshold = 0.25f;
    r.ani_threshold = 95.0f;

    { GeodfWriter w(path); w.write_taxon(r); w.close(); }

    GeodfReader reader(path);
    REQUIRE(reader.n_taxa() == 1);
    auto completed = reader.completed_taxa();
    REQUIRE(completed.count("d__Bacteria;s__Test bacterium") == 1);
    auto td = reader.find("d__Bacteria;s__Test bacterium");
    REQUIRE(td.has_value());
    REQUIRE(td->taxonomy == "d__Bacteria;s__Test bacterium");
    REQUIRE(td->genome_ids.size() == 3);
    REQUIRE(td->rep_accessions.size() == 2);
    std::filesystem::remove(path);
}

TEST_CASE("GEODF round-trip: failed taxon error_message", "[geodf]") {
    auto path = std::filesystem::temp_directory_path() / "geodf_test_failed.geodf";
    std::filesystem::remove(path);

    TaxonResult r;
    r.taxonomy = "d__Bacteria;s__Failing taxon";
    r.stage = PipelineStage::FAILED;
    r.error_message = "Something went wrong";

    { GeodfWriter w(path); w.write_taxon(r); w.close(); }

    GeodfReader reader(path);
    auto td = reader.find("d__Bacteria;s__Failing taxon");
    REQUIRE(td.has_value());
    REQUIRE(td->stage == PipelineStage::FAILED);
    REQUIRE(td->error_message == "Something went wrong");
    std::filesystem::remove(path);
}

TEST_CASE("GEODF crash recovery: missing trailer", "[geodf]") {
    auto path = std::filesystem::temp_directory_path() / "geodf_test_crash.geodf";
    std::filesystem::remove(path);

    TaxonResult r;
    r.taxonomy = "d__Bacteria;s__Crash taxon";
    r.stage = PipelineStage::COMPLETE;
    r.genome_ids = {42};
    r.is_rep = {true};
    r.contamination = {0.0f};
    r.reps = {{42, "CRASH001", {}}};
    r.diversity_threshold = 0.3f;

    // Write and close normally to get a valid file
    { GeodfWriter w(path); w.write_taxon(r); w.close(); }

    // Simulate crash by truncating the trailer + index + string table from the end.
    // The FileTrailer is 32 bytes at the very end — remove it to force recovery mode.
    {
        auto sz = std::filesystem::file_size(path);
        REQUIRE(sz > sizeof(geodf::FileTrailer));
        std::filesystem::resize_file(path, sz - sizeof(geodf::FileTrailer));
    }

    // Reader should detect the invalid trailer and fall back to header scan
    GeodfReader reader(path);
    REQUIRE(reader.n_taxa() == 1);
    auto completed = reader.completed_taxa();
    REQUIRE(completed.count("d__Bacteria;s__Crash taxon") == 1);
    std::filesystem::remove(path);
}
