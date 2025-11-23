cmake_minimum_required(VERSION 3.22)
project(QuantOptions LANGUAGES CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_testing()


# If no build type was provided (single-config generators), default to Debug.
# For multi-config generators (CMAKE_CONFIGURATION_TYPES) we don't override.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No CMAKE_BUILD_TYPE specified, defaulting to Debug")
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# Append sensible debug flags when in Debug mode.
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        # Disable optimizations, enable debug info and define a DEBUG macro
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -DDEBUG=1")
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Od /Zi /DDEBUG=1")
    endif()
endif()

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
