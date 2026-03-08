find_path(Zstd_INCLUDE_DIR zstd.h
    HINTS /usr/local/include /usr/include
          /maps/projects/fernandezguerra/apps/opt/conda/envs/bioinfo/include)
find_library(Zstd_LIBRARY zstd
    HINTS /usr/local/lib /usr/lib
          /maps/projects/fernandezguerra/apps/opt/conda/envs/bioinfo/lib)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Zstd DEFAULT_MSG Zstd_LIBRARY Zstd_INCLUDE_DIR)
if(Zstd_FOUND)
    set(Zstd_LIBRARIES ${Zstd_LIBRARY})
    set(Zstd_INCLUDE_DIRS ${Zstd_INCLUDE_DIR})
endif()
