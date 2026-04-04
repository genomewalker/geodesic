add_executable(tests
    test_main.cpp
    test_sketch.cpp
    geodf_test.cpp
    ${CMAKE_SOURCE_DIR}/src/core/genome_cache.cpp
    ${CMAKE_SOURCE_DIR}/src/core/sketch/minhash.cpp
    ${CMAKE_SOURCE_DIR}/src/io/fasta_reader.cpp
    ${CMAKE_SOURCE_DIR}/src/io/gz_reader.cpp
    ${CMAKE_SOURCE_DIR}/src/db/geodf/geodf_writer.cpp
    ${CMAKE_SOURCE_DIR}/src/db/geodf/geodf_reader.cpp
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
if(Zstd_FOUND)
    target_include_directories(tests PRIVATE "${Zstd_INCLUDE_DIRS}")
    target_link_libraries(tests PRIVATE ${Zstd_LIBRARIES})
endif()

include(CTest)
include(Catch)
catch_discover_tests(tests)
