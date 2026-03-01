cmake_minimum_required(VERSION 3.20)
project(geodesic VERSION 1.0.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Enable AVX2 for SIMD optimizations
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)
if(COMPILER_SUPPORTS_AVX2)
    add_compile_options(-mavx2)
    message(STATUS "AVX2 enabled for SIMD optimizations")
endif()
if(COMPILER_SUPPORTS_FMA)
    add_compile_options(-mfma)
    message(STATUS "FMA enabled for fused multiply-add")
endif()

# Enable OpenMP for parallel processing
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found, enabling parallel optimizations")
endif()

# Custom find modules (DuckDB, etc.)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(FetchContent)

# CLI11 v2.4.2
FetchContent_Declare(CLI11 GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git GIT_TAG v2.4.2)
FetchContent_MakeAvailable(CLI11)

# spdlog v1.14.1
FetchContent_Declare(spdlog GIT_REPOSITORY https://github.com/gabime/spdlog.git GIT_TAG v1.14.1)
FetchContent_MakeAvailable(spdlog)

# BS::thread_pool v4.1.0 (bshoshany/thread-pool)
FetchContent_Declare(thread_pool GIT_REPOSITORY https://github.com/bshoshany/thread-pool.git GIT_TAG v4.1.0)
FetchContent_MakeAvailable(thread_pool)

# hnswlib (header-only HNSW implementation)
FetchContent_Declare(hnswlib GIT_REPOSITORY https://github.com/nmslib/hnswlib.git GIT_TAG v0.8.0)
FetchContent_MakeAvailable(hnswlib)

# Catch2 v3.7.1
FetchContent_Declare(Catch2 GIT_REPOSITORY https://github.com/catchorg/Catch2.git GIT_TAG v3.7.1)
FetchContent_MakeAvailable(Catch2)

find_package(Threads REQUIRED)

# rapidgzip v0.16.0 — parallel gzip decompression (replaces zlib gzopen/gzread)
FetchContent_Declare(rapidgzip
    GIT_REPOSITORY https://github.com/mxmlnkn/rapidgzip.git
    GIT_TAG rapidgzip-v0.16.0
    GIT_SHALLOW TRUE
    GIT_SUBMODULES_RECURSE TRUE
    SOURCE_SUBDIR src)
set(RAPIDGZIP_BUILD_STANDALONE OFF CACHE BOOL "" FORCE)
set(RAPIDGZIP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(rapidgzip)

find_package(DuckDB MODULE REQUIRED)

# RPATH: embed DuckDB library search path so the binary runs without LD_LIBRARY_PATH
get_filename_component(_duckdb_dir "${DuckDB_LIBRARIES}" DIRECTORY)
set(CMAKE_BUILD_RPATH "${_duckdb_dir}")

add_executable(geodesic
    src/main.cpp
    src/config.cpp
    src/pipeline.cpp
    src/io/tsv_reader.cpp
    src/io/fasta_reader.cpp
    src/io/gz_reader.cpp
    src/io/results_writer.cpp
    src/io/report_writer.cpp
    src/db/schema.cpp
    src/db/db_manager.cpp
    src/db/operations.cpp
    src/db/embedding_store.cpp
    src/core/subprocess.cpp
    src/core/genome_cache.cpp
    src/core/similarity/similarity_base.cpp
    src/core/similarity/skani.cpp
    src/core/sketch/minhash.cpp
    src/core/geodesic/geodesic.cpp
    src/parallel/taxon_processor.cpp
)

target_include_directories(geodesic PRIVATE
    src
    "${thread_pool_SOURCE_DIR}/include"
    "${hnswlib_SOURCE_DIR}"
    "${rapidgzip_SOURCE_DIR}/src"
)

target_link_libraries(geodesic PRIVATE
    CLI11::CLI11
    spdlog::spdlog
    ${DuckDB_LIBRARIES}
    librapidgzip
    pthread
)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(geodesic PRIVATE OpenMP::OpenMP_CXX)
endif()

# Optional TBB for parallel STL
find_package(TBB QUIET)
if(TBB_FOUND)
    target_link_libraries(geodesic PRIVATE TBB::tbb)
endif()

# Tests
enable_testing()
add_subdirectory(tests)
