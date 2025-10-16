add_library(benchmarks_impl STATIC
  ${CMAKE_SOURCE_DIR}/benchmarks/functions_version.cpp
  ${CMAKE_SOURCE_DIR}/benchmarks/dataset.cpp
  ${CMAKE_SOURCE_DIR}/benchmarks/benchmark.cpp
  ${CMAKE_SOURCE_DIR}/tests/sanity_checks.cpp
)

target_include_directories(benchmarks_impl PUBLIC
  ${PROJECT_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}/benchmarks/include
  ${CMAKE_SOURCE_DIR}/tests/include
)

set_target_properties(benchmarks_impl PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)
