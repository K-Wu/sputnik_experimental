# For use in subdirs.
set(SPUTNIK_SRCS)

set(SPUTNIK_SPMM_TEST_SRCS)
set(SPUTNIK_SPMM_BENCHMARK_SRCS)
add_subdirectory(spmm)

set(SPUTNIK_SDDMM_TEST_SRCS)
set(SPUTNIK_SDDMM_BENCHMARK_SRCS)
add_subdirectory(sddmm)

set(SPUTNIK_DEPTHWISE_TEST_SRCS)
set(SPUTNIK_DEPTHWISE_BENCHMARK_SRCS)
add_subdirectory(depthwise)

add_subdirectory(bias_relu)
add_subdirectory(softmax)
add_subdirectory(utils)

# NB: <codegen> add_subdirectory here for custom_spmm|sddmm
set(SPUTNIK_CUSTOM_SPMM_TEST_SRCS)
set(SPUTNIK_CUSTOM_SPMM_BENCHMARK_SRCS)
add_subdirectory(custom_spmm)
set(SPUTNIK_CUSTOM_SDDMM_TEST_SRCS)
set(SPUTNIK_CUSTOM_SDDMM_BENCHMARK_SRCS)
add_subdirectory(custom_sddmm)

set(DIR_SRCS)

##
### Find all sources in this directory.
##

# Lib srcs.
file(GLOB TMP *.h)
list(APPEND DIR_SRCS ${TMP})
file(GLOB TMP *.cc)
list(APPEND DIR_SRCS ${TMP})

##
### Filter files that we don't want in the main library.
##

set(FILTER_SRCS)

# Don't want test related code.
file(GLOB TMP test_utils*)
list(APPEND FILTER_SRCS ${TMP})

# Don't need matrix utilities.
file(GLOB TMP matrix_utils*)
list(APPEND FILTER_SRCS ${TMP})

foreach(FILE ${FILTER_SRCS})
  list(REMOVE_ITEM DIR_SRCS ${FILE})
endforeach(FILE)

# Add the sources to the build.
list(APPEND SPUTNIK_SRCS ${DIR_SRCS})

# Set .cu.cc files to be compiled as CUDA.
set(SPUTNIK_CUDA_SRCS ${SPUTNIK_SRCS})
list(FILTER SPUTNIK_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
set_source_files_properties(${SPUTNIK_CUDA_SRCS} PROPERTIES LANGUAGE CUDA)

# Create libsputnik.
add_library(sputnik SHARED ${SPUTNIK_SRCS})
target_link_libraries(sputnik ${SPUTNIK_LIBS})

# Library installation.
install(TARGETS sputnik)

set(INSTALL_BASE "include/sputnik")
install(FILES "cuda_utils.h" DESTINATION ${INSTALL_BASE})
install(FILES "sputnik.h" DESTINATION ${INSTALL_BASE})
install(FILES "bias_relu/bias_relu.h" DESTINATION "${INSTALL_BASE}/bias_relu")
install(FILES "depthwise/cuda_depthwise.h" DESTINATION "${INSTALL_BASE}/depthwise")
install(FILES "spmm/cuda_spmm.h" DESTINATION "${INSTALL_BASE}/spmm")
install(FILES "sddmm/cuda_sddmm.h" DESTINATION "${INSTALL_BASE}/sddmm")
install(FILES "softmax/softmax.h" DESTINATION "${INSTALL_BASE}/softmax")
install(FILES "softmax/sparse_softmax.h" DESTINATION "${INSTALL_BASE}/softmax")
install(FILES "utils/index_format.h" DESTINATION "${INSTALL_BASE}/utils")
# NB: <codegen> add custom_spmm|sddmm header install here.
install(FILES "spmm/cuda_custom_spmm.h" DESTINATION "${INSTALL_BASE}/custom_spmm")
install(FILES "sddmm/cuda_custom_sddmm.h" DESTINATION "${INSTALL_BASE}/custom_sddmm")

# Optionally build the test suite.
if (BUILD_TEST)
  # Test sources for all targets.
  set(SPUTNIK_TEST_SRCS)
  file(GLOB TMP matrix_utils*)
  list(APPEND SPUTNIK_TEST_SRCS ${TMP})
  file(GLOB TMP test_utils*)
  list(APPEND SPUTNIK_TEST_SRCS ${TMP})

  # NB: <codegen> add custom_spmm|sddmm test build here.
  # Custom SpMM test build.
  list(APPEND SPUTNIK_CUSTOM_SPMM_TEST_SRCS ${SPUTNIK_TEST_SRCS})
  
  set(SPUTNIK_CUSTOM_SPMM_TEST_CUDA_SRCS ${SPUTNIK_CUSTOM_SPMM_TEST_SRCS})
  list(FILTER SPUTNIK_CUSTOM_SPMM_TEST_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
  set_source_files_properties(${SPUTNIK_CUSTOM_SPMM_TEST_CUDA_SRCS} PROPERTIES LANGUAGE CUDA)  
  add_executable(custom_spmm_test ${SPUTNIK_CUSTOM_SPMM_TEST_SRCS})
  target_link_libraries(custom_spmm_test sputnik ${SPUTNIK_TEST_LIBS})

  # Custom SDDMM test build.
  list(APPEND SPUTNIK_CUSTOM_SDDMM_TEST_SRCS ${SPUTNIK_TEST_SRCS})
  
  set(SPUTNIK_CUSTOM_SDDMM_TEST_CUDA_SRCS ${SPUTNIK_CUSTOM_SDDMM_TEST_SRCS})
  list(FILTER SPUTNIK_CUSTOM_SDDMM_TEST_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
  set_source_files_properties(${SPUTNIK_CUSTOM_SDDMM_TEST_CUDA_SRCS} PROPERTIES LANGUAGE CUDA)  
  add_executable(custom_sddmm_test ${SPUTNIK_CUSTOM_SDDMM_TEST_SRCS})
  target_link_libraries(custom_sddmm_test sputnik ${SPUTNIK_TEST_LIBS})

  # SpMM test build.
  list(APPEND SPUTNIK_SPMM_TEST_SRCS ${SPUTNIK_TEST_SRCS})
  
  set(SPUTNIK_SPMM_TEST_CUDA_SRCS ${SPUTNIK_SPMM_TEST_SRCS})
  list(FILTER SPUTNIK_SPMM_TEST_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
  set_source_files_properties(${SPUTNIK_SPMM_TEST_CUDA_SRCS} PROPERTIES LANGUAGE CUDA)  
  add_executable(spmm_test ${SPUTNIK_SPMM_TEST_SRCS})
  target_link_libraries(spmm_test sputnik ${SPUTNIK_TEST_LIBS})

  # SDDMM test build.
  list(APPEND SPUTNIK_SDDMM_TEST_SRCS ${SPUTNIK_TEST_SRCS})
  
  set(SPUTNIK_SDDMM_TEST_CUDA_SRCS ${SPUTNIK_SDDMM_TEST_SRCS})
  list(FILTER SPUTNIK_SDDMM_TEST_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
  set_source_files_properties(${SPUTNIK_SDDMM_TEST_CUDA_SRCS} PROPERTIES LANGUAGE CUDA)  
  add_executable(sddmm_test ${SPUTNIK_SDDMM_TEST_SRCS})
  target_link_libraries(sddmm_test sputnik ${SPUTNIK_TEST_LIBS})

  # Depthwise test build.
  list(APPEND SPUTNIK_DEPTHWISE_TEST_SRCS ${SPUTNIK_TEST_SRCS})
  
  set(SPUTNIK_DEPTHWISE_TEST_CUDA_SRCS ${SPUTNIK_DEPTHWISE_TEST_SRCS})
  list(FILTER SPUTNIK_DEPTHWISE_TEST_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
  set_source_files_properties(
    ${SPUTNIK_DEPTHWISE_TEST_CUDA_SRCS}
    PROPERTIES LANGUAGE CUDA)  
  add_executable(depthwise_test ${SPUTNIK_DEPTHWISE_TEST_SRCS})
  target_link_libraries(depthwise_test sputnik ${SPUTNIK_TEST_LIBS})
endif()

# Optionally build the benchmark suite.
if (BUILD_BENCHMARK)
  # Benchmark sources for all targets.
  set(SPUTNIK_BENCHMARK_SRCS)
  file(GLOB TMP matrix_utils*)
  list(APPEND SPUTNIK_BENCHMARK_SRCS ${TMP})
  file(GLOB TMP test_utils*)
  list(APPEND SPUTNIK_BENCHMARK_SRCS ${TMP})

  # NB: <codegen> add custom_spmm|sddmm benchmark build here.
  # Custom SpMM benchmark build.
  list(APPEND SPUTNIK_CUSTOM_SPMM_BENCHMARK_SRCS ${SPUTNIK_BENCHMARK_SRCS})
  
  set(SPUTNIK_CUSTOM_SPMM_BENCHMARK_CUDA_SRCS ${SPUTNIK_CUSTOM_SPMM_BENCHMARK_SRCS})
  list(FILTER SPUTNIK_CUSTOM_SPMM_BENCHMARK_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
  set_source_files_properties(${SPUTNIK_CUSTOM_SPMM_BENCHMARK_CUDA_SRCS} PROPERTIES LANGUAGE CUDA)  
  add_executable(custom_spmm_benchmark ${SPUTNIK_CUSTOM_SPMM_BENCHMARK_SRCS})
  target_link_libraries(custom_spmm_benchmark sputnik ${SPUTNIK_BENCHMARK_LIBS})

  # Custom SDDMM benchmark build.
  list(APPEND SPUTNIK_CUSTOM_SDDMM_BENCHMARK_SRCS ${SPUTNIK_BENCHMARK_SRCS})
  
  set(SPUTNIK_CUSTOM_SDDMM_BENCHMARK_CUDA_SRCS ${SPUTNIK_CUSTOM_SDDMM_BENCHMARK_SRCS})
  list(FILTER SPUTNIK_CUSTOM_SDDMM_BENCHMARK_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
  set_source_files_properties(${SPUTNIK_CUSTOM_SDDMM_BENCHMARK_CUDA_SRCS} PROPERTIES LANGUAGE CUDA)  
  add_executable(custom_sddmm_benchmark ${SPUTNIK_CUSTOM_SDDMM_BENCHMARK_SRCS})
  target_link_libraries(custom_sddmm_benchmark sputnik ${SPUTNIK_BENCHMARK_LIBS})
  
  # SpMM benchmark build.
  list(APPEND SPUTNIK_SPMM_BENCHMARK_SRCS ${SPUTNIK_BENCHMARK_SRCS})
  
  set(SPUTNIK_SPMM_BENCHMARK_CUDA_SRCS ${SPUTNIK_SPMM_BENCHMARK_SRCS})
  list(FILTER SPUTNIK_SPMM_BENCHMARK_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
  set_source_files_properties(${SPUTNIK_SPMM_BENCHMARK_CUDA_SRCS} PROPERTIES LANGUAGE CUDA)  
  add_executable(spmm_benchmark ${SPUTNIK_SPMM_BENCHMARK_SRCS})
  target_link_libraries(spmm_benchmark sputnik ${SPUTNIK_BENCHMARK_LIBS})

  # SDDMM benchmark build.
  list(APPEND SPUTNIK_SDDMM_BENCHMARK_SRCS ${SPUTNIK_BENCHMARK_SRCS})
  
  set(SPUTNIK_SDDMM_BENCHMARK_CUDA_SRCS ${SPUTNIK_SDDMM_BENCHMARK_SRCS})
  list(FILTER SPUTNIK_SDDMM_BENCHMARK_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
  set_source_files_properties(${SPUTNIK_SDDMM_BENCHMARK_CUDA_SRCS} PROPERTIES LANGUAGE CUDA)  
  add_executable(sddmm_benchmark ${SPUTNIK_SDDMM_BENCHMARK_SRCS})
  target_link_libraries(sddmm_benchmark sputnik ${SPUTNIK_BENCHMARK_LIBS})

  # Depthwise benchmark build.
  list(APPEND SPUTNIK_DEPTHWISE_BENCHMARK_SRCS ${SPUTNIK_BENCHMARK_SRCS})
  
  set(SPUTNIK_DEPTHWISE_BENCHMARK_CUDA_SRCS ${SPUTNIK_DEPTHWISE_BENCHMARK_SRCS})
  list(FILTER SPUTNIK_DEPTHWISE_BENCHMARK_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
  set_source_files_properties(
    ${SPUTNIK_DEPTHWISE_BENCHMARK_CUDA_SRCS}
    PROPERTIES LANGUAGE CUDA)  
  add_executable(depthwise_benchmark ${SPUTNIK_DEPTHWISE_BENCHMARK_SRCS})
  target_link_libraries(depthwise_benchmark sputnik ${SPUTNIK_BENCHMARK_LIBS})    
endif()
