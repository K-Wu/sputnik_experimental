GEN_OP_CMAKELISTS_TEMPLATE = """
set(DIR_SRCS)

##
### Find all sources in this directory.
##

file(GLOB TMP *.h)
list(APPEND DIR_SRCS ${{TMP}})
file(GLOB TMP *.cc)
list(APPEND DIR_SRCS ${{TMP}})

##
### Filter files that we don't want in the main library.
##

set(FILTER_SRCS)

# Don't need test files.
file(GLOB TMP *_test.cu.cc)
list(APPEND FILTER_SRCS ${{TMP}})

# Don't need benchmark files.
file(GLOB TMP *_benchmark.cu.cc)
list(APPEND FILTER_SRCS ${{TMP}})

foreach(FILE ${{FILTER_SRCS}})
  list(REMOVE_ITEM DIR_SRCS ${{FILE}})
endforeach(FILE)

# Add the sources to the build.
set(SPUTNIK_SRCS ${{SPUTNIK_SRCS}} ${{DIR_SRCS}} PARENT_SCOPE)

set(CURRDIR_CUSTOM_OP_DIR_NAME_LISTS ${{SPUTNIK_CUSTOM_OP_DIR_NAME_LISTS}})
set(CURRDIR_CUSTOM_OP_TEST_SRCS_LISTS ${{SPUTNIK_CUSTOM_OP_TEST_SRCS_LISTS}})
set(CURRDIR_CUSTOM_OP_BENCHMARK_SRCS_LISTS ${{SPUTNIK_CUSTOM_OP_BENCHMARK_SRCS_LISTS}})

list(APPEND CURRDIR_CUSTOM_OP_DIR_NAME_LISTS "{opname}")

# Conditionally gather test sources.
if (BUILD_TEST)
  file(GLOB TMP *_test.cu.cc)
  # set(SPUTNIK_{opname_capitalized}_TEST_SRCS ${{SPUTNIK_{opname_capitalized}_TEST_SRCS}} ${{TMP}} PARENT_SCOPE)
  list(APPEND CURRDIR_CUSTOM_OP_TEST_SRCS_LISTS ${{TMP}})
endif()

# Conditionally gather the benchmark sources.
if (BUILD_BENCHMARK)
  file(GLOB TMP *_benchmark.cu.cc)
  # set(SPUTNIK_{opname_capitalized}_BENCHMARK_SRCS ${{SPUTNIK_{opname_capitalized}_BENCHMARK_SRCS}} ${{TMP}} PARENT_SCOPE)
  list(APPEND CURRDIR_CUSTOM_OP_BENCHMARK_SRCS_LISTS ${{TMP}})
endif()

set(SPUTNIK_CUSTOM_OP_DIR_NAME_LISTS ${{CURRDIR_CUSTOM_OP_DIR_NAME_LISTS}} PARENT_SCOPE)
set(SPUTNIK_CUSTOM_OP_TEST_SRCS_LISTS ${{CURRDIR_CUSTOM_OP_TEST_SRCS_LISTS}} PARENT_SCOPE)
set(SPUTNIK_CUSTOM_OP_BENCHMARK_SRCS_LISTS ${{CURRDIR_CUSTOM_OP_BENCHMARK_SRCS_LISTS}} PARENT_SCOPE)

set(INSTALL_BASE "include/sputnik")
install(FILES "cuda_{opname}.h" DESTINATION "${{INSTALL_BASE}}/gen_{opname}")
"""

if __name__ == "__main__":
    opname = "custom_spmm"
    opname_capitalized = opname.upper()
    print(
        GEN_OP_CMAKELISTS_TEMPLATE.format(
            opname=opname, opname_capitalized=opname_capitalized
        )
    )
