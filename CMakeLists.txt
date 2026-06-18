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

# Frame pointers for perf call-graph profiling (negligible overhead)
add_compile_options(-fno-omit-frame-pointer)

# AddressSanitizer for heap corruption debugging
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
if(ENABLE_ASAN)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer -O1)
    add_link_options(-fsanitize=address)
    message(STATUS "AddressSanitizer enabled")
endif()

# Enable OpenMP for parallel processing
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found, enabling parallel optimizations")
endif()

# genopack dependency — configure with:
#   cmake -DGENOPACK_INCLUDE_DIR=/path/to/genopack/include \
#         -DGENOPACK_LIBRARY=/path/to/libgenopack_lib.a
set(GENOPACK_INCLUDE_DIR "" CACHE PATH "genopack include directory (required)")
set(GENOPACK_LIBRARY     "" CACHE FILEPATH "genopack static library (required)")
if(NOT GENOPACK_INCLUDE_DIR OR NOT GENOPACK_LIBRARY)
    # Attempt auto-detection from a sibling checkout (common dev layout)
    get_filename_component(_repo_root "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
    set(_auto_inc "${_repo_root}/genopack/include")
    set(_auto_lib "${_repo_root}/genopack/build/libgenopack_lib.a")
    if(EXISTS "${_auto_inc}" AND EXISTS "${_auto_lib}")
        set(GENOPACK_INCLUDE_DIR "${_auto_inc}" CACHE PATH "genopack include directory" FORCE)
        set(GENOPACK_LIBRARY     "${_auto_lib}" CACHE FILEPATH "genopack static library" FORCE)
        message(STATUS "genopack: auto-detected at ${_auto_inc}")
    else()
        message(FATAL_ERROR
            "genopack not found. Set -DGENOPACK_INCLUDE_DIR= and -DGENOPACK_LIBRARY= "
            "or place genopack as a sibling directory of geodesic.")
    endif()
endif()
message(STATUS "genopack include: ${GENOPACK_INCLUDE_DIR}")
message(STATUS "genopack library: ${GENOPACK_LIBRARY}")

# Custom find modules
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
FetchContent_Declare(hnswlib
    GIT_REPOSITORY https://github.com/nmslib/hnswlib.git
    GIT_TAG v0.8.0
    PATCH_COMMAND git apply --check ${CMAKE_CURRENT_SOURCE_DIR}/patches/hnswlib-prefetch-fix.patch
                  && git apply ${CMAKE_CURRENT_SOURCE_DIR}/patches/hnswlib-prefetch-fix.patch || true)
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

# Eigen3 for Nyström spectral embedding (matrix operations)
FetchContent_Declare(Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(Eigen3)

find_package(Zstd MODULE QUIET)
find_package(ZLIB REQUIRED)

# xxhash — required by derep_archive (accession_set_hash, ARMP)
find_path(Xxhash_INCLUDE_DIR xxhash.h
    HINTS /usr/local/include /usr/include
          /maps/projects/fernandezguerra/apps/opt/conda/envs/bioinfo/include)
if(NOT Xxhash_INCLUDE_DIR)
    message(FATAL_ERROR "xxhash.h not found. Install xxhash or set Xxhash_INCLUDE_DIR.")
endif()
message(STATUS "xxhash include: ${Xxhash_INCLUDE_DIR}")

# libdeflate — required by genopack's gzip decompression path
find_library(LIBDEFLATE_LIBRARY NAMES deflate libdeflate
    HINTS "${GENOPACK_INCLUDE_DIR}/../.." ENV CONDA_PREFIX
    PATH_SUFFIXES lib lib64)
if(NOT LIBDEFLATE_LIBRARY)
    message(FATAL_ERROR "libdeflate not found (required by genopack). "
        "Install libdeflate or set LIBDEFLATE_LIBRARY.")
endif()
message(STATUS "libdeflate: ${LIBDEFLATE_LIBRARY}")

add_executable(geodesic
    src/main.cpp
    src/config.cpp
    src/pipeline.cpp
    src/update.cpp
    src/io/tsv_reader.cpp
    src/io/lock_writer.cpp
    src/io/results_writer.cpp
    src/io/report_writer.cpp
    src/db/geodf/geodf_writer.cpp
    src/db/geodf/geodf_reader.cpp
    src/db/grd/grd_writer.cpp
    src/db/grd/grd_merge.cpp
    src/distributed.cpp
    src/taxonomy/partition.cpp
    src/db/taxdb/ncbi_taxdb.cpp
    src/core/subprocess.cpp
    src/core/multi_pack_reader.cpp
    src/core/kmer_probe.cpp
    src/core/preloaded_pack_reader.cpp
    src/taxonomy/normalize.cpp
    src/core/geodesic/geodesic.cpp
    src/parallel/taxon_processor.cpp
    src/validate_ani.cpp

    src/check/run_check.cpp
    src/run_ani.cpp
    src/state/run_state.cpp
    src/derep/derep_archive.cpp
)

target_include_directories(geodesic PRIVATE
    src
    "${thread_pool_SOURCE_DIR}/include"
    "${hnswlib_SOURCE_DIR}"
    "${rapidgzip_SOURCE_DIR}/src"
    "${eigen3_SOURCE_DIR}"
    "${GENOPACK_INCLUDE_DIR}"
    "${Xxhash_INCLUDE_DIR}"
)
if(Zstd_FOUND)
    target_include_directories(geodesic PRIVATE "${Zstd_INCLUDE_DIRS}")
    target_compile_definitions(geodesic PRIVATE HAVE_ZSTD)
endif()

target_link_libraries(geodesic PRIVATE
    CLI11::CLI11
    spdlog::spdlog
    librapidgzip
    pthread
    "${GENOPACK_LIBRARY}"
    "${LIBDEFLATE_LIBRARY}"
    ZLIB::ZLIB
)
if(Zstd_FOUND)
    target_link_libraries(geodesic PRIVATE ${Zstd_LIBRARIES})
    message(STATUS "zstd found: ${Zstd_LIBRARIES} — .zst genome files supported")
else()
    message(STATUS "zstd not found — .zst genome files unsupported")
endif()

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
