set(DIR_SRCS)

##
### Find all sources in this directory.
##

file(GLOB TMP *.h)
list(APPEND DIR_SRCS ${TMP})
file(GLOB TMP *.cc)
list(APPEND DIR_SRCS ${TMP})

##
### Filter files that we don't want in the main library.
##

set(FILTER_SRCS)

# Don't need test files.
file(GLOB TMP *_test.cu.cc)
list(APPEND FILTER_SRCS ${TMP})

# Don't need benchmark files.
file(GLOB TMP *_benchmark.cu.cc)
list(APPEND FILTER_SRCS ${TMP})

foreach(FILE ${FILTER_SRCS})
  list(REMOVE_ITEM DIR_SRCS ${FILE})
endforeach(FILE)

# Build SpMM cubin
# Result will be at sputnik/build/sputnik/spmm/CMakeFiles/cuda_spmm.dir/cuda_spmm.cu.cubin
add_library(cuda_spmm OBJECT cuda_spmm.cu.cc)
set_source_files_properties(cuda_spmm.cu.cc PROPERTIES LANGUAGE CUDA)
set_property(TARGET cuda_spmm PROPERTY CUDA_CUBIN_COMPILATION ON)

# Add the sources to the build.
set(SPUTNIK_SRCS ${SPUTNIK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# Conditionally gather test sources.
if (BUILD_TEST)
  file(GLOB TMP *_test.cu.cc)
  set(SPUTNIK_SPMM_TEST_SRCS ${SPUTNIK_SPMM_TEST_SRCS} ${TMP} PARENT_SCOPE)
endif()

# Conditionally gather the benchmark sources.
if (BUILD_BENCHMARK)
  file(GLOB TMP *_benchmark.cu.cc)
  set(SPUTNIK_SPMM_BENCHMARK_SRCS ${SPUTNIK_SPMM_BENCHMARK_SRCS} ${TMP} PARENT_SCOPE)
endif()
