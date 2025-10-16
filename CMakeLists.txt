cmake_minimum_required(VERSION 3.22)
project(QuantOptions LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(ENABLE_OPENMP "Enable OpenMP parallelization" ON)

# Keep headers discoverable via targets (no global include_directories needed)
set(PROJECT_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)

enable_testing()

# Core libraries
add_subdirectory(src/models)
add_subdirectory(src/backends/cpu)
if (ENABLE_OPENMP)
  add_subdirectory(src/backends/openmp)
endif()

# Tests / benchmarks can remain as they are (optional)
if (EXISTS "${CMAKE_SOURCE_DIR}/tests/CMakeLists.txt")
  add_subdirectory(tests)
endif()
if (EXISTS "${CMAKE_SOURCE_DIR}/benchmarks/CMakeLists.txt")
  add_subdirectory(benchmarks)
endif()

# Apps (fetches CLI11 & links only apps to it)
add_subdirectory(apps)
