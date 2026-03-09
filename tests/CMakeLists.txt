add_executable(tests
    test_main.cpp
    test_sketch.cpp
    ${CMAKE_SOURCE_DIR}/src/core/genome_cache.cpp
    ${CMAKE_SOURCE_DIR}/src/core/sketch/minhash.cpp
    ${CMAKE_SOURCE_DIR}/src/core/geodesic/diversity_threshold.cpp
    ${CMAKE_SOURCE_DIR}/src/io/fasta_reader.cpp
    ${CMAKE_SOURCE_DIR}/src/io/gz_reader.cpp
)

target_include_directories(tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    "${thread_pool_SOURCE_DIR}/include"
    "${rapidgzip_SOURCE_DIR}/src"
)

target_link_libraries(tests PRIVATE
    Catch2::Catch2WithMain
    spdlog::spdlog
    librapidgzip
    Threads::Threads
)

include(CTest)
include(Catch)
catch_discover_tests(tests)
