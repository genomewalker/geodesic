add_executable(tests
    test_main.cpp
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
