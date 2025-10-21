cmake_minimum_required(VERSION 3.22)
project(QuantOptions LANGUAGES CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_testing()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(-Wall -Wextra -Wpedantic)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options(/W4 /permissive-)
endif()


add_subdirectory(src/models)
add_subdirectory(src/backends/cpu)
add_subdirectory(src/backends/cuda)
add_subdirectory(src/backends/openmp)
add_subdirectory(tests)
add_subdirectory(benchmarks)
add_subdirectory(apps)
