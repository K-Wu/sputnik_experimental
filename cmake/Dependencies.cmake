include(cmake/Cuda.cmake)

# TODO(tgale): Move cuSPARSE, cuBLAS deps to test & benchmark only.
if (WIN32)
  # alternative routine on windows to find out cuda libraries. Some are not available in static and replaced with dynamic libraries.
  find_package(CUDAToolkit REQUIRED)
  list(APPEND SPUTNIK_LIBS "CUDA::cudart_static;CUDA::cublas;CUDA::cusparse;CUDA::cublasLt")
else()
  cuda_find_library(CUDART_LIBRARY cudart_static)
  cuda_find_library(CUBLAS_LIBRARY cublas_static)
  cuda_find_library(CUSPARSE_LIBRARY cusparse_static)
  list(APPEND SPUTNIK_LIBS "cudart_static;cublas_static;cusparse_static;culibos;cublasLt_static")
endif()

# Google Glog.
find_package(Glog REQUIRED)
list(APPEND SPUTNIK_LIBS ${GLOG_LIBRARIES})

if (WIN32)
  include_directories(SYSTEM ${GLOG_INCLUDE_DIR})
  add_definitions(-DGLOG_NO_ABBREVIATED_SEVERITIES)
  # if (MSVC)
  #   if ("${languages}" MATCHES "CUDA")  
  #     add_compile_options(-Wa,-mbig-obj)
  #   else()
  #     add_compile_options(/bigobj)
  #   endif()
  # endif ()
endif()

if (BUILD_TEST)
  # Google Abseil.
  add_subdirectory(third_party/abseil-cpp)

  # Google Test and Google Mock.
  add_subdirectory(third_party/googletest)
  set(BUILD_GTEST ON CACHE INTERNAL "Build gtest submodule.")
  set(BUILD_GMOCK ON CACHE INTERNAL "Build gmock submodule.")
  include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/third_party/googletest/googletest/include)
  include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/third_party/googletest/googlemock/include)

  list(APPEND SPUTNIK_TEST_LIBS "gtest_main;gmock;absl::random_random")
endif()

if (BUILD_BENCHMARK)
  # Google Benchmark.
  add_subdirectory(third_party/benchmark)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE INTERNAL "Build benchmark test suite.")
  include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/third_party/benchmark/include)

  list(APPEND SPUTNIK_BENCHMARK_LIBS "gtest;absl::random_random;benchmark::benchmark_main")
endif()
