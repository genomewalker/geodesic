find_path(DuckDB_INCLUDE_DIR duckdb.hpp
    HINTS /usr/local/include /usr/include
          $ENV{HOME}/.claude/include
          /maps/projects/fernandezguerra/apps/opt/conda/envs/bioinfo/include)
find_library(DuckDB_LIBRARY duckdb
    HINTS /usr/local/lib /usr/lib
          $ENV{HOME}/.claude/bin
          /maps/projects/fernandezguerra/apps/opt/conda/envs/bioinfo/lib)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(DuckDB DEFAULT_MSG DuckDB_LIBRARY DuckDB_INCLUDE_DIR)
if(DuckDB_FOUND)
    set(DuckDB_LIBRARIES ${DuckDB_LIBRARY})
    set(DuckDB_INCLUDE_DIRS ${DuckDB_INCLUDE_DIR})
endif()
