add_executable(tests
    test_main.cpp
    parallel_shard_visit_test.cpp
    ${CMAKE_SOURCE_DIR}/src/core/multi_pack_reader.cpp
    unnamed_species_test.cpp
    ${CMAKE_SOURCE_DIR}/src/taxonomy/unnamed.cpp
    ${CMAKE_SOURCE_DIR}/src/taxonomy/species.cpp
    ${CMAKE_SOURCE_DIR}/src/core/subprocess.cpp
    geodf_test.cpp
    derep_archive_integration_test.cpp
    ${CMAKE_SOURCE_DIR}/src/derep/derep_archive_test.cpp
    ${CMAKE_SOURCE_DIR}/src/derep/derep_archive.cpp
    ${CMAKE_SOURCE_DIR}/src/db/geodf/geodf_writer.cpp
    ${CMAKE_SOURCE_DIR}/src/db/geodf/geodf_reader.cpp
)

target_include_directories(tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    "${thread_pool_SOURCE_DIR}/include"
    "${rapidgzip_SOURCE_DIR}/src"
    "${GENOPACK_INCLUDE_DIR}"
    "${Xxhash_INCLUDE_DIR}"
)

target_link_libraries(tests PRIVATE
    Catch2::Catch2WithMain
    spdlog::spdlog
    librapidgzip
    Threads::Threads
    "${GENOPACK_LIBRARY}"
    "${LIBDEFLATE_LIBRARY}"
    ZLIB::ZLIB
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(tests PRIVATE OpenMP::OpenMP_CXX)
endif()
if(Zstd_FOUND)
    target_include_directories(tests PRIVATE "${Zstd_INCLUDE_DIRS}")
    target_link_libraries(tests PRIVATE ${Zstd_LIBRARIES})
endif()

include(CTest)
include(Catch)
catch_discover_tests(tests)

add_executable(unnamed_species_gate unnamed_species_gate.cpp
    ${CMAKE_SOURCE_DIR}/src/taxonomy/unnamed.cpp
    ${CMAKE_SOURCE_DIR}/src/taxonomy/species.cpp
    ${CMAKE_SOURCE_DIR}/src/core/subprocess.cpp
    ${CMAKE_SOURCE_DIR}/src/core/multi_pack_reader.cpp)
get_target_property(gate_includes tests INCLUDE_DIRECTORIES)
target_include_directories(unnamed_species_gate PRIVATE ${gate_includes})
get_target_property(gate_libs tests LINK_LIBRARIES)
target_link_libraries(unnamed_species_gate PRIVATE ${gate_libs})
